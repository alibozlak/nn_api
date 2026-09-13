//! The endpoints.
//!
//! Every handler follows the same three steps: validate the request against
//! the stored model, do the neuralflow work on a blocking thread, and answer
//! with JSON. The engine work is wrapped in `catch_engine_panic`, so a panic
//! neuralflow raises comes back as a 422 instead of a dropped connection.

use std::time::Instant;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use neuralflow::prelude::FitOptions;
use uuid::Uuid;

use crate::dto::*;
use crate::engine;
use crate::error::{ApiError, catch_engine_panic};
use crate::extract::ApiJson;
use crate::model::{ModelSpec, NewModel, StoredModel};
use crate::store::{AppState, now_ms};

/// `GET /health`
pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: env!("CARGO_PKG_NAME"),
        version: env!("CARGO_PKG_VERSION"),
        models: state.count(),
    })
}

/// `POST /create-compile-model` -- Keras' `Sequential(...)` and `compile(...)`
/// in one call, which is what the path is named after.
pub async fn create_model(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<CreateModelRequest>,
) -> Result<(StatusCode, Json<ModelResponse>), ApiError> {
    let NewModel { name, spec, weights: imported_weights, batch_size } = request.into_parts(&state.config().limits)?;

    // Building the model is what turns the spec into real weights: Glorot
    // uniform for every layer, overwritten by any weights the request carried.
    // A model of millions of parameters is millions of random draws, so it
    // happens on a blocking thread like every other call into the engine.
    let report = {
        let spec = spec.clone();
        run_blocking(move || catch_engine_panic(|| engine::describe(&spec, &imported_weights))).await?
    };
    let stored = state.insert(NewModel { name, spec, weights: report.weights.clone(), batch_size })?;

    Ok((StatusCode::CREATED, Json(ModelResponse::new(&stored, report))))
}

/// `GET /models`
pub async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let models = state.list().await;
    let entries: Vec<ModelListEntry> = models.iter().map(ModelListEntry::from).collect();

    Json(ModelListResponse { count: entries.len(), models: entries })
}

/// `GET /models/{id}`
pub async fn get_model(State(state): State<AppState>, Path(id): Path<String>) -> Result<Json<ModelResponse>, ApiError> {
    let handle = state.handle(parse_id(&id)?)?;
    let model = handle.read().await.clone();

    describe(model).await.map(Json)
}

/// `DELETE /models/{id}`
pub async fn delete_model(State(state): State<AppState>, Path(id): Path<String>) -> Result<StatusCode, ApiError> {
    state.remove(parse_id(&id)?)?;

    Ok(StatusCode::NO_CONTENT)
}

/// `POST /models/{id}/train` -- Keras' `fit`.
pub async fn train(
    State(state): State<AppState>,
    Path(id): Path<String>,
    ApiJson(mut request): ApiJson<TrainRequest>,
) -> Result<Json<TrainResponse>, ApiError> {
    let limits = state.config().limits;
    let id = parse_id(&id)?;
    let handle = state.handle(id)?;
    // Taken before anything is read, and held to the end: one run at a time on
    // this model. It locks nothing else -- reads of the model stay open.
    let _training = handle.training_permit().await;

    if request.epochs == 0 || request.epochs > limits.max_epochs {
        return Err(ApiError::bad_request(format!(
            "'epochs' must be between 1 and {}, got {}",
            limits.max_epochs, request.epochs
        )));
    }

    // x and y arrive either inline or as a path into the server's data
    // directory; opening and parsing a file is blocking work, like the
    // training that follows it.
    let (x_rows, y_rows) = {
        let config = state.config().clone();
        let (x, x_path) = (request.x.take(), request.x_path.take());
        let (y, y_path) = (request.y.take(), request.y_path.take());

        run_blocking(move || {
            Ok((
                matrix_from_request(&config, "x", x, x_path)?,
                matrix_from_request(&config, "y", y, y_path)?,
            ))
        })
        .await?
    };

    // Everything the run needs, copied out under a short read lock.
    let (run_spec, weights, loss, optimizer, batch_size) = {
        let model = handle.read().await;
        validate_samples(&model.spec, &x_rows, &y_rows, &limits)?;

        // The request's batch size wins for this run; the model's default is
        // what a request that sends none gets.
        let batch_size = request.batch_size.unwrap_or(model.batch_size);
        if batch_size == 0 {
            return Err(ApiError::bad_request("'batch_size' must be at least 1"));
        }

        // The request may compile the model again, the way Keras' `compile` does.
        let loss = request.loss.or(model.spec.loss).ok_or_else(|| {
            ApiError::bad_request("this model has no loss yet; send 'loss' here or set it when creating the model")
        })?;
        let optimizer = match request.optimizer {
            Some(_) => resolve_optimizer(request.optimizer, Some(loss))?,
            None => model.spec.optimizer.or(resolve_optimizer(None, Some(loss))?),
        }
        .expect("a loss is present, so an optimizer was resolved");
        // The seed applies to this run only; the model keeps the one it was made with.
        let run_spec = ModelSpec {
            loss: Some(loss),
            optimizer: Some(optimizer),
            seed: request.seed.unwrap_or(model.spec.seed),
            ..model.spec.clone()
        };

        (run_spec, model.weights.clone(), loss, optimizer, batch_size)
    };

    let x = engine::matrix_of(&x_rows, "x")?;
    let y = engine::matrix_of(&y_rows, "y")?;
    let samples = x_rows.len();
    let options = FitOptions { epochs: request.epochs, batch_size, shuffle: request.shuffle, verbose: false };

    let started = Instant::now();
    let outcome = run_blocking(move || catch_engine_panic(|| engine::train(&run_spec, &weights, &x, &y, options))).await?;
    let duration_ms = started.elapsed().as_millis() as u64;

    let loss_history = outcome.loss_history;
    let (initial_loss, final_loss) = match (loss_history.first(), loss_history.last()) {
        (Some(first), Some(last)) => (*first, *last),
        _ => return Err(ApiError::internal("training returned no epochs")),
    };

    // One short write: the new weights land in a single step, so a prediction
    // running alongside sees either all of them or none.
    let trained_epochs = {
        let mut model = handle.write().await;
        model.weights = outcome.weights;
        model.spec.loss = Some(loss);
        model.spec.optimizer = Some(optimizer);
        model.trained_epochs += request.epochs;
        model.last_loss = Some(final_loss);
        model.updated_at_ms = now_ms();

        model.trained_epochs
    };

    Ok(Json(TrainResponse {
        id,
        epochs: request.epochs,
        batch_size,
        samples,
        loss_function: loss,
        optimizer,
        initial_loss,
        final_loss,
        loss: request.return_history.then_some(loss_history),
        trained_epochs,
        duration_ms,
    }))
}

/// `POST /models/{id}/predict` -- Keras' `predict`.
pub async fn predict(
    State(state): State<AppState>,
    Path(id): Path<String>,
    ApiJson(mut request): ApiJson<PredictRequest>,
) -> Result<Json<PredictResponse>, ApiError> {
    let limits = state.config().limits;
    let handle = state.handle(parse_id(&id)?)?;

    let x_rows = {
        let config = state.config().clone();
        let (x, x_path) = (request.x.take(), request.x_path.take());

        run_blocking(move || matrix_from_request(&config, "x", x, x_path)).await?
    };

    let (model_id, spec, weights) = {
        let model = handle.read().await;
        validate_features(&model.spec, &x_rows, &limits)?;

        (model.id, model.spec.clone(), model.weights.clone())
    };

    let x = engine::matrix_of(&x_rows, "x")?;
    let predictions = run_blocking(move || catch_engine_panic(|| engine::predict(&spec, &weights, &x))).await?;

    Ok(Json(PredictResponse {
        id: model_id,
        rows: predictions.len(),
        columns: predictions.first().map_or(0, Vec::len),
        predictions,
    }))
}

/// `POST /models/{id}/evaluate` -- Keras' `evaluate`.
pub async fn evaluate(
    State(state): State<AppState>,
    Path(id): Path<String>,
    ApiJson(mut request): ApiJson<EvaluateRequest>,
) -> Result<Json<EvaluateResponse>, ApiError> {
    let limits = state.config().limits;
    let handle = state.handle(parse_id(&id)?)?;

    // x and y arrive either inline or as a path into the server's data
    // directory; opening and parsing a file is blocking work, like the
    // training that follows it.
    let (x_rows, y_rows) = {
        let config = state.config().clone();
        let (x, x_path) = (request.x.take(), request.x_path.take());
        let (y, y_path) = (request.y.take(), request.y_path.take());

        run_blocking(move || {
            Ok((
                matrix_from_request(&config, "x", x, x_path)?,
                matrix_from_request(&config, "y", y, y_path)?,
            ))
        })
        .await?
    };

    let (model_id, mut spec, weights) = {
        let model = handle.read().await;
        validate_samples(&model.spec, &x_rows, &y_rows, &limits)?;

        (model.id, model.spec.clone(), model.weights.clone())
    };

    // Measuring against another loss does not recompile the stored model.
    let loss = request.loss.or(spec.loss).ok_or_else(|| {
        ApiError::bad_request("this model has no loss yet; send 'loss' here or set it when creating the model")
    })?;
    spec.loss = Some(loss);
    spec.optimizer = resolve_optimizer(None, Some(loss))?;

    let x = engine::matrix_of(&x_rows, "x")?;
    let y = engine::matrix_of(&y_rows, "y")?;
    let samples = x_rows.len();
    let loss_value = run_blocking(move || catch_engine_panic(|| engine::evaluate(&spec, &weights, &x, &y))).await?;

    Ok(Json(EvaluateResponse { id: model_id, loss_function: loss, loss: loss_value, samples }))
}

/// `GET /models/{id}/weights` -- Keras' `get_weights`.
pub async fn get_weights(State(state): State<AppState>, Path(id): Path<String>) -> Result<Json<WeightsResponse>, ApiError> {
    let handle = state.handle(parse_id(&id)?)?;
    let model = handle.read().await;

    Ok(Json(WeightsResponse { id: model.id, layers: model.weights.clone() }))
}

/// `PUT /models/{id}/weights` -- Keras' `set_weights`. Layers left out of the
/// request keep the weights they have.
pub async fn set_weights(
    State(state): State<AppState>,
    Path(id): Path<String>,
    ApiJson(request): ApiJson<SetWeightsRequest>,
) -> Result<Json<WeightsResponse>, ApiError> {
    let handle = state.handle(parse_id(&id)?)?;
    let mut model = handle.write().await;

    validate_weights(&model.spec, &request.layers)?;
    for incoming in request.layers {
        let index = model
            .spec
            .layer_index(&incoming.name)
            .expect("validate_weights already matched every name to a layer");
        model.weights[index] = incoming;
    }
    model.updated_at_ms = now_ms();

    Ok(Json(WeightsResponse { id: model.id, layers: model.weights.clone() }))
}

/// `GET /models/{id}/onnx` -- the model as an ONNX file. The one endpoint that
/// answers with bytes rather than JSON.
pub async fn export_onnx(State(state): State<AppState>, Path(id): Path<String>) -> Result<Response, ApiError> {
    let handle = state.handle(parse_id(&id)?)?;
    let (name, spec, weights) = {
        let model = handle.read().await;

        (model.name.clone(), model.spec.clone(), model.weights.clone())
    };

    let bytes = run_blocking(move || catch_engine_panic(|| engine::onnx_bytes(&spec, &weights))).await?;
    let disposition = format!("attachment; filename=\"{}.onnx\"", file_name_of(&name));
    let disposition = HeaderValue::from_str(&disposition).map_err(|error| ApiError::internal(error.to_string()))?;

    let mut response = bytes.into_response();
    response.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/octet-stream"));
    response.headers_mut().insert(header::CONTENT_DISPOSITION, disposition);

    Ok(response)
}

/// `POST /utils/scale` -- `column_based_scaling`, which divides every column
/// by a power of ten taken from that column's first row.
pub async fn scale(State(state): State<AppState>, ApiJson(request): ApiJson<ScaleRequest>) -> Result<Json<ScaleResponse>, ApiError> {
    let limits = state.config().limits;
    if request.x.is_empty() {
        return Err(ApiError::bad_request("'x' has no samples"));
    }
    if request.x.len() > limits.max_samples {
        return Err(ApiError::bad_request(format!("'x' has {} samples, the limit is {}", request.x.len(), limits.max_samples)));
    }
    if request.y.len() != request.x.len() {
        return Err(ApiError::bad_request(format!("'x' has {} rows but 'y' has {}", request.x.len(), request.y.len())));
    }
    if let Some((row, values)) = request.y.iter().enumerate().find(|(_, values)| values.len() != 1) {
        return Err(ApiError::bad_request(format!(
            "'y' must have exactly one column, but row {row} has {}",
            values.len()
        )));
    }

    let x = engine::matrix_of(&request.x, "x")?;
    let y = engine::matrix_of(&request.y, "y")?;
    let (x, y, ten_power_ratios) = run_blocking(move || catch_engine_panic(|| engine::scale_columns(x, y))).await?;

    Ok(Json(ScaleResponse { x, y, ten_power_ratios }))
}

/// Anything else, still as JSON.
pub async fn not_found() -> ApiError {
    ApiError::not_found("no endpoint matches this path; the README lists the routes, and GET /health says whether the server is up")
}

/// The full `ModelResponse`, summary table included, which means building the
/// model once. It is the only place the exact shapes neuralflow ended up with
/// are read back, rather than worked out from the spec.
async fn describe(model: StoredModel) -> Result<ModelResponse, ApiError> {
    let (spec, weights) = (model.spec.clone(), model.weights.clone());
    let report = run_blocking(move || catch_engine_panic(|| engine::describe(&spec, &weights))).await?;

    Ok(ModelResponse::new(&model, report))
}

/// Runs CPU work off the async runtime. neuralflow is synchronous and a long
/// `fit` would otherwise block every other request on the same worker.
async fn run_blocking<T>(work: impl FnOnce() -> Result<T, ApiError> + Send + 'static) -> Result<T, ApiError>
where
    T: Send + 'static,
{
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|error| ApiError::internal(format!("the worker thread did not finish: {error}")))?
}

fn parse_id(raw: &str) -> Result<Uuid, ApiError> {
    Uuid::parse_str(raw).map_err(|_| ApiError::bad_request(format!("'{raw}' is not a model id; an id is a UUID, as returned by POST /create-compile-model")))
}

/// A model name is whatever the client called it, and it ends up in a header
/// here, so everything outside a safe set becomes an underscore.
fn file_name_of(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|character| if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') { character } else { '_' })
        .collect();

    match cleaned.trim_matches('.') {
        "" => String::from("model"),
        trimmed => trimmed.chars().take(64).collect(),
    }
}
