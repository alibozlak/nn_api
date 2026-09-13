//! The API driven end to end, the way the Java client will drive it: JSON in,
//! JSON out, over the real router.

use std::path::{Path, PathBuf};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use http_body_util::BodyExt;
use rust_nn_api::config::Config;
use rust_nn_api::store::AppState;
use serde_json::{Value, json};
use tower::ServiceExt;

fn api() -> Router {
    rust_nn_api::app(AppState::new(Config::default()))
}

/// One request, its status and its parsed JSON body.
async fn call(app: &Router, method: Method, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let (status, bytes) = call_raw(app, method, path, body).await;
    let json = serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("{path} did not answer with JSON ({error}): {}", String::from_utf8_lossy(&bytes)));

    (status, json)
}

async fn call_raw(app: &Router, method: Method, path: &str, body: Option<Value>) -> (StatusCode, Vec<u8>) {
    let builder = Request::builder().method(method).uri(path);
    let request = match body {
        Some(json) => builder
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json.to_string()))
            .unwrap(),
        None => builder.body(Body::empty()).unwrap(),
    };

    let response = app.clone().oneshot(request).await.expect("the router always answers");
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes().to_vec();

    (status, bytes)
}

/// The XOR model from neuralflow's own example.
fn xor_model() -> Value {
    json!({
        "name": "xor",
        "features": 2,
        "seed": 1234,
        "layers": [
            { "units": 8, "activation": "relu", "name": "layer1" },
            { "units": 1, "activation": "sigmoid", "name": "layer2" }
        ],
        "loss": "binary_crossentropy",
        "optimizer": { "type": "adam", "learning_rate": 0.05 }
    })
}

fn xor_samples() -> (Value, Value) {
    (json!([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]), json!([[0.], [1.], [1.], [0.]]))
}

async fn create(app: &Router, body: Value) -> Value {
    let (status, model) = call(app, Method::POST, "/create-compile-model", Some(body)).await;
    assert_eq!(status, StatusCode::CREATED, "{model}");

    model
}

async fn weights_of(app: &Router, id: &str) -> Value {
    let (status, weights) = call(app, Method::GET, &format!("/models/{id}/weights"), None).await;
    assert_eq!(status, StatusCode::OK, "{weights}");

    weights["layers"].clone()
}

fn id_of(model: &Value) -> String {
    model["id"].as_str().expect("every model has an id").to_string()
}

#[tokio::test]
async fn health_reports_the_service() {
    let (status, body) = call(&api(), Method::GET, "/health", None).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["service"], "rust_nn_api");
    assert_eq!(body["models"], 0);
}

#[tokio::test]
async fn creating_a_model_reports_its_shape() {
    let model = create(&api(), xor_model()).await;

    assert_eq!(model["features"], 2);
    assert_eq!(model["output_units"], 1);
    // (2 * 8 + 8) + (8 * 1 + 1)
    assert_eq!(model["total_params"], 33);
    assert_eq!(model["layers"][0]["name"], "layer1");
    assert_eq!(model["layers"][0]["input_count"], 2);
    assert_eq!(model["layers"][0]["params"], 24);
    assert_eq!(model["layers"][1]["activation"], "sigmoid");
    assert_eq!(model["optimizer"]["type"], "adam");
    assert_eq!(model["optimizer"]["learning_rate"], 0.05);
    assert_eq!(model["trained_epochs"], 0);
    assert!(model["summary"].as_str().unwrap().contains("Total params: 33"), "{}", model["summary"]);
}

#[tokio::test]
async fn layers_are_named_the_way_keras_names_them() {
    let model = create(
        &api(),
        json!({
            "features": 3,
            "layers": [
                { "units": 4, "activation": "relu" },
                { "units": 2, "activation": "linear" }
            ]
        }),
    )
    .await;

    assert_eq!(model["layers"][0]["name"], "dense");
    assert_eq!(model["layers"][1]["name"], "dense_1");
    // No loss was given, so the model is not compiled yet.
    assert_eq!(model["loss"], Value::Null);
    assert_eq!(model["optimizer"], Value::Null);
}

#[tokio::test]
async fn xor_trains_and_then_predicts_it() {
    let app = api();
    let model = create(&app, xor_model()).await;
    let id = id_of(&model);
    let (x, y) = xor_samples();

    let (status, training) = call(
        &app,
        Method::POST,
        &format!("/models/{id}/train"),
        Some(json!({ "x": x, "y": y, "epochs": 500 })),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{training}");
    assert_eq!(training["epochs"], 500);
    assert_eq!(training["samples"], 4);
    assert_eq!(training["loss"].as_array().unwrap().len(), 500);
    let initial = training["initial_loss"].as_f64().unwrap();
    let final_loss = training["final_loss"].as_f64().unwrap();
    assert!(final_loss < initial, "the loss did not fall: {initial} -> {final_loss}");
    assert!(final_loss < 0.1, "XOR did not converge, final loss {final_loss}");

    let (status, prediction) = call(&app, Method::POST, &format!("/models/{id}/predict"), Some(json!({ "x": x }))).await;
    assert_eq!(status, StatusCode::OK, "{prediction}");
    assert_eq!(prediction["rows"], 4);
    assert_eq!(prediction["columns"], 1);

    let predictions: Vec<f64> = prediction["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row[0].as_f64().unwrap())
        .collect();
    for (sample, expected) in predictions.iter().zip([0., 1., 1., 0.]) {
        assert!((sample - expected).abs() < 0.5, "XOR predicted {predictions:?}");
    }

    // Training is cumulative, and the model remembers where it got to.
    let (_, model) = call(&app, Method::GET, &format!("/models/{id}"), None).await;
    assert_eq!(model["trained_epochs"], 500);
    assert!((model["last_loss"].as_f64().unwrap() - final_loss).abs() < 1e-12);
}

#[tokio::test]
async fn a_model_without_a_seed_is_given_one() {
    let app = api();
    let architecture = json!([
        { "units": 8, "activation": "relu", "name": "layer1" },
        { "units": 1, "activation": "sigmoid", "name": "layer2" }
    ]);

    // Leaving the field out and sending null are the same thing, and both are
    // answered with a seed rather than a null.
    let omitted = create(&app, json!({ "features": 2, "layers": architecture })).await;
    let explicit_null = create(&app, json!({ "features": 2, "seed": null, "layers": architecture })).await;

    let drawn = omitted["seed"].as_u64().expect("a model with no seed of its own is given one");
    let other = explicit_null["seed"].as_u64().expect("null means the same as leaving it out");
    assert_ne!(drawn, other, "two models were given the same seed");

    // And the seed reported back is the one that was used: a model built with
    // it has the very same weights.
    let rebuilt = create(&app, json!({ "features": 2, "seed": drawn, "layers": architecture })).await;
    assert_eq!(
        weights_of(&app, &id_of(&rebuilt)).await,
        weights_of(&app, &id_of(&omitted)).await,
        "the seed in the response did not rebuild the model"
    );
}

#[tokio::test]
async fn a_model_remembers_its_batch_size() {
    let app = api();
    let (x, y) = xor_samples();
    let with_default = |batch: Value| {
        json!({
            "features": 2,
            "seed": 1234,
            "layers": [{ "units": 8, "activation": "relu" }, { "units": 1, "activation": "sigmoid" }],
            "loss": "bce",
            "batch_size": batch
        })
    };

    // Left out, it is Keras' 32.
    let plain = create(&app, with_default(json!(32))).await;
    assert_eq!(plain["batch_size"], 32);

    // Set here, a train request that sends none gets it.
    let model = create(&app, with_default(json!(2))).await;
    let id = id_of(&model);
    assert_eq!(model["batch_size"], 2);

    let (status, training) = call(&app, Method::POST, &format!("/models/{id}/train"), Some(json!({ "x": x, "y": y }))).await;
    assert_eq!(status, StatusCode::OK, "{training}");
    assert_eq!(training["batch_size"], 2, "the model's own batch size was not used");

    // A train request may still override it, for that run only.
    let (_, training) = call(
        &app,
        Method::POST,
        &format!("/models/{id}/train"),
        Some(json!({ "x": x, "y": y, "batch_size": 4 })),
    )
    .await;
    assert_eq!(training["batch_size"], 4);

    let (_, model) = call(&app, Method::GET, &format!("/models/{id}"), None).await;
    assert_eq!(model["batch_size"], 2, "a run must not change the model's default");
}

#[tokio::test]
async fn training_is_reproducible_from_the_seed() {
    let app = api();
    let (x, y) = xor_samples();
    let mut finals = Vec::new();

    for _ in 0..2 {
        let id = id_of(&create(&app, xor_model()).await);
        let (_, training) = call(
            &app,
            Method::POST,
            &format!("/models/{id}/train"),
            Some(json!({ "x": x, "y": y, "epochs": 50, "return_history": false })),
        )
        .await;
        finals.push(training["final_loss"].as_f64().unwrap());
    }

    assert_eq!(finals[0], finals[1], "the same seed gave two different runs");
    assert!(finals[0].is_finite());
}

#[tokio::test]
async fn evaluate_measures_without_training() {
    let app = api();
    let id = id_of(&create(&app, xor_model()).await);
    let (x, y) = xor_samples();

    let (status, evaluation) = call(&app, Method::POST, &format!("/models/{id}/evaluate"), Some(json!({ "x": x, "y": y }))).await;

    assert_eq!(status, StatusCode::OK, "{evaluation}");
    assert_eq!(evaluation["samples"], 4);
    assert_eq!(evaluation["loss_function"], "binary_crossentropy");
    assert!(evaluation["loss"].as_f64().unwrap() > 0.);

    let (_, model) = call(&app, Method::GET, &format!("/models/{id}"), None).await;
    assert_eq!(model["trained_epochs"], 0, "evaluate must not train");
}

#[tokio::test]
async fn weights_survive_a_round_trip() {
    let app = api();
    let id = id_of(&create(&app, xor_model()).await);

    let (status, weights) = call(&app, Method::GET, &format!("/models/{id}/weights"), None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(weights["layers"].as_array().unwrap().len(), 2);
    assert_eq!(weights["layers"][0]["weights"].as_array().unwrap().len(), 2, "W1 has one row per feature");
    assert_eq!(weights["layers"][0]["bias"].as_array().unwrap().len(), 8);

    // Known weights in, the same weights out.
    let layers = json!([{ "name": "layer2", "weights": [[1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]], "bias": [0.5] }]);
    let (status, updated) = call(&app, Method::PUT, &format!("/models/{id}/weights"), Some(json!({ "layers": layers }))).await;
    assert_eq!(status, StatusCode::OK, "{updated}");
    assert_eq!(updated["layers"][1]["bias"][0], 0.5);
    assert_eq!(updated["layers"][1]["weights"][0][0], 1.0);
    assert_eq!(updated["layers"][0], weights["layers"][0], "a layer left out keeps its weights");

    // And a model can be recreated from them, which is how a client keeps one
    // over a restart.
    let (_, all) = call(&app, Method::GET, &format!("/models/{id}/weights"), None).await;
    let restored = create(
        &app,
        json!({
            "name": "restored",
            "features": 2,
            "layers": [
                { "units": 8, "activation": "relu", "name": "layer1" },
                { "units": 1, "activation": "sigmoid", "name": "layer2" }
            ],
            "weights": all["layers"]
        }),
    )
    .await;

    let (_, restored_weights) = call(&app, Method::GET, &format!("/models/{}/weights", id_of(&restored)), None).await;
    assert_eq!(restored_weights["layers"], all["layers"]);
}

#[tokio::test]
async fn a_model_exports_as_onnx() {
    let app = api();
    let id = id_of(&create(&app, xor_model()).await);

    let (status, bytes) = call_raw(&app, Method::GET, &format!("/models/{id}/onnx"), None).await;

    assert_eq!(status, StatusCode::OK);
    assert!(bytes.len() > 100, "an ONNX file of 33 parameters is not {} bytes", bytes.len());
    // The ModelProto starts with field 1, ir_version, as a varint.
    assert_eq!(bytes[0], 0x08, "this does not look like a protobuf ModelProto");
}

#[tokio::test]
async fn models_are_listed_and_deleted() {
    let app = api();
    let id = id_of(&create(&app, xor_model()).await);

    let (status, list) = call(&app, Method::GET, "/models", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(list["count"], 1);
    assert_eq!(list["models"][0]["name"], "xor");
    assert_eq!(list["models"][0]["total_params"], 33);
    assert_eq!(list["models"][0]["seed"], 1234);
    assert_eq!(list["models"][0]["batch_size"], 32);

    let (status, _) = call_raw(&app, Method::DELETE, &format!("/models/{id}"), None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);

    let (status, error) = call(&app, Method::GET, &format!("/models/{id}"), None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error["error"]["code"], "not_found");

    let (status, _) = call(&app, Method::DELETE, &format!("/models/{id}"), None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn the_scaling_helper_returns_the_exponents() {
    let (status, scaled) = call(
        &api(),
        Method::POST,
        "/utils/scale",
        Some(json!({ "x": [[1500., 3.], [2500., 4.]], "y": [[250000.], [300000.]] })),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{scaled}");
    // 1500 has four integer digits, so the column is divided by 10^3.
    assert_eq!(scaled["ten_power_ratios"], json!([3, 0, 5]));
    assert_eq!(scaled["x"][0][0], 1.5);
    assert_eq!(scaled["x"][0][1], 3.0);
    assert_eq!(scaled["y"][0][0], 2.5);
}

// ------------------------------------------------------------------ errors

#[tokio::test]
async fn a_wrong_feature_count_is_a_400_that_says_so() {
    let app = api();
    let id = id_of(&create(&app, xor_model()).await);

    let (status, error) = call(&app, Method::POST, &format!("/models/{id}/predict"), Some(json!({ "x": [[1., 2., 3.]] }))).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["error"]["code"], "invalid_request");
    assert!(
        error["error"]["message"].as_str().unwrap().contains("takes 2 features"),
        "{}",
        error["error"]["message"]
    );
}

#[tokio::test]
async fn a_wrong_target_width_is_a_400_that_says_so() {
    let app = api();
    let id = id_of(&create(&app, xor_model()).await);
    let (x, _) = xor_samples();

    let (status, error) = call(
        &app,
        Method::POST,
        &format!("/models/{id}/train"),
        Some(json!({ "x": x, "y": [[0., 0.], [1., 1.], [1., 1.], [0., 0.]] })),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["error"]["message"].as_str().unwrap().contains("1 units"), "{}", error["error"]["message"]);
}

#[tokio::test]
async fn training_without_a_loss_says_what_is_missing() {
    let app = api();
    let id = id_of(&create(&app, json!({ "features": 2, "layers": [{ "units": 1, "activation": "sigmoid" }] })).await);
    let (x, y) = xor_samples();

    let (status, error) = call(&app, Method::POST, &format!("/models/{id}/train"), Some(json!({ "x": x, "y": y }))).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["error"]["message"].as_str().unwrap().contains("no loss"), "{}", error["error"]["message"]);

    // Sending the loss with the training data compiles it, Keras' way.
    let (status, training) = call(
        &app,
        Method::POST,
        &format!("/models/{id}/train"),
        Some(json!({ "x": x, "y": y, "epochs": 5, "loss": "mse", "optimizer": "sgd" })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{training}");
    assert_eq!(training["loss_function"], "mean_squared_error");
    assert_eq!(training["optimizer"]["type"], "sgd");
    assert_eq!(training["optimizer"]["learning_rate"], 0.01);

    let (_, model) = call(&app, Method::GET, &format!("/models/{id}"), None).await;
    assert_eq!(model["loss"], "mean_squared_error", "the train call compiled the model");
}

#[tokio::test]
async fn bad_requests_all_answer_with_the_same_json_shape() {
    let app = api();
    let id = id_of(&create(&app, xor_model()).await);
    let (x, y) = xor_samples();

    let cases: Vec<(Method, String, Option<Value>, StatusCode, &str)> = vec![
        (Method::GET, "/nope".into(), None, StatusCode::NOT_FOUND, "not_found"),
        (Method::GET, "/models/not-a-uuid".into(), None, StatusCode::BAD_REQUEST, "invalid_request"),
        (
            Method::POST,
            "/create-compile-model".into(),
            Some(json!({ "features": 0, "layers": [{ "units": 1, "activation": "relu" }] })),
            StatusCode::BAD_REQUEST,
            "invalid_request",
        ),
        (
            Method::POST,
            "/create-compile-model".into(),
            Some(json!({ "features": 2, "layers": [] })),
            StatusCode::BAD_REQUEST,
            "invalid_request",
        ),
        (
            Method::POST,
            "/create-compile-model".into(),
            Some(json!({ "features": 2, "layers": [{ "units": 1, "activation": "relu" }], "batch_size": 0 })),
            StatusCode::BAD_REQUEST,
            "invalid_request",
        ),
        (
            Method::POST,
            "/create-compile-model".into(),
            Some(json!({ "features": 2, "layers": [{ "units": 1, "activation": "softmax" }] })),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_json",
        ),
        (
            Method::POST,
            "/create-compile-model".into(),
            Some(json!({ "features": 2, "layers": [{ "units": 1, "activation": "relu" }], "typo": 1 })),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_json",
        ),
        (
            Method::POST,
            format!("/models/{id}/train"),
            Some(json!({ "x": x, "y": y, "epochs": 0 })),
            StatusCode::BAD_REQUEST,
            "invalid_request",
        ),
        (
            Method::POST,
            format!("/models/{id}/train"),
            Some(json!({ "x": x, "y": y, "batch_size": 0 })),
            StatusCode::BAD_REQUEST,
            "invalid_request",
        ),
        (
            Method::POST,
            format!("/models/{id}/predict"),
            Some(json!({ "x": [] })),
            StatusCode::BAD_REQUEST,
            "invalid_request",
        ),
        (
            Method::POST,
            format!("/models/{id}/predict"),
            Some(json!({ "x": [[1., 2.], [3.]] })),
            StatusCode::BAD_REQUEST,
            "invalid_request",
        ),
        (
            Method::PUT,
            format!("/models/{id}/weights"),
            Some(json!({ "layers": [{ "name": "layer9", "weights": [[1.]], "bias": [0.] }] })),
            StatusCode::BAD_REQUEST,
            "invalid_request",
        ),
        (
            Method::PUT,
            format!("/models/{id}/weights"),
            Some(json!({ "layers": [{ "name": "layer2", "weights": [[1.]], "bias": [0.] }] })),
            StatusCode::BAD_REQUEST,
            "invalid_request",
        ),
    ];

    for (method, path, body, expected_status, expected_code) in cases {
        let (status, error) = call(&app, method.clone(), &path, body).await;

        assert_eq!(status, expected_status, "{method} {path} answered {error}");
        assert_eq!(error["error"]["code"], expected_code, "{method} {path} answered {error}");
        assert!(
            !error["error"]["message"].as_str().unwrap_or_default().is_empty(),
            "{method} {path} answered without a message"
        );
    }
}

#[tokio::test]
async fn an_unknown_optimizer_names_the_ones_that_exist() {
    let app = api();

    // The name on its own, and the object form.
    for optimizer in [json!("rmsprop"), json!({ "type": "rmsprop", "learning_rate": 0.1 })] {
        let (status, error) = call(
            &app,
            Method::POST,
            "/create-compile-model",
            Some(json!({
                "features": 2,
                "layers": [{ "units": 1, "activation": "sigmoid" }],
                "loss": "mse",
                "optimizer": optimizer
            })),
        )
        .await;

        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let message = error["error"]["message"].as_str().unwrap();
        assert!(message.contains("rmsprop") && message.contains("adam"), "{message}");
    }
}

#[tokio::test]
async fn a_wrong_method_is_a_405() {
    let app = api();

    let (status, error) = call(&app, Method::DELETE, "/health", None).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(error["error"]["code"], "method_not_allowed");

    // /models lists models and nothing else; creating one is its own path.
    let (status, error) = call(&app, Method::POST, "/models", Some(xor_model())).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED, "{error}");
}

#[tokio::test]
async fn an_optimizer_is_accepted_as_a_bare_name_or_an_object() {
    let app = api();

    let named = create(
        &app,
        json!({
            "features": 2,
            "layers": [{ "units": 1, "activation": "sigmoid" }],
            "loss": "bce",
            "optimizer": "adam"
        }),
    )
    .await;
    assert_eq!(named["optimizer"]["type"], "adam");
    assert_eq!(named["optimizer"]["learning_rate"], 0.001, "a bare name means the optimizer's own default rate");

    let configured = create(
        &app,
        json!({
            "features": 2,
            "layers": [{ "units": 1, "activation": "sigmoid" }],
            "loss": "bce",
            "optimizer": { "type": "sgd", "learning_rate": 0.5 }
        }),
    )
    .await;
    assert_eq!(configured["optimizer"]["type"], "sgd");
    assert_eq!(configured["optimizer"]["learning_rate"], 0.5);
}

#[tokio::test]
async fn malformed_json_is_still_answered_with_json() {
    let request = Request::builder()
        .method(Method::POST)
        .uri("/create-compile-model")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("{ not json"))
        .unwrap();

    let response = api().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let error: Value = serde_json::from_slice(&bytes).expect("even a parse failure answers with JSON");

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["error"]["code"], "invalid_json");
}

#[tokio::test]
async fn duplicate_layer_names_are_refused_before_the_engine_panics() {
    let (status, error) = call(
        &api(),
        Method::POST,
        "/create-compile-model",
        Some(json!({
            "features": 2,
            "layers": [
                { "units": 2, "activation": "relu", "name": "same" },
                { "units": 1, "activation": "sigmoid", "name": "same" }
            ]
        })),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["error"]["message"].as_str().unwrap().contains("unique"), "{}", error["error"]["message"]);
}

#[tokio::test]
async fn a_non_finite_sample_is_refused() {
    let app = api();
    let id = id_of(&create(&app, xor_model()).await);

    // JSON has no NaN literal, but 1e400 parses as an infinity.
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("/models/{id}/predict"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(r#"{"x": [[1e400, 0.0]]}"#))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let error: Value = serde_json::from_slice(&bytes).unwrap();

    assert!(status.is_client_error(), "an infinite feature was accepted");
    assert!(!error["error"]["message"].as_str().unwrap_or_default().is_empty());
}


// ----------------------------------------------- training data from a file

/// A directory of this test's own, for the server to read data out of.
fn data_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nn_api_{}_{label}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a temp directory can be made");

    dir
}

/// A server allowed to read files, out of `dir` and nowhere else.
fn api_reading(dir: &Path) -> Router {
    let config = Config { data_dir: Some(dir.canonicalize().expect("the temp directory exists")), ..Config::default() };

    rust_nn_api::app(AppState::new(config))
}

fn write_json(dir: &Path, name: &str, value: &Value) {
    std::fs::write(dir.join(name), value.to_string()).expect("the temp file can be written");
}

#[tokio::test]
async fn training_data_can_come_from_files() {
    let dir = data_dir("from_files");
    let (x, y) = xor_samples();
    write_json(&dir, "x.json", &x);
    write_json(&dir, "y.json", &y);

    let app = api_reading(&dir);
    let id = id_of(&create(&app, xor_model()).await);

    let (status, training) = call(
        &app,
        Method::POST,
        &format!("/models/{id}/train"),
        Some(json!({ "x_path": "x.json", "y_path": "y.json", "epochs": 500, "return_history": false })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{training}");
    assert_eq!(training["samples"], 4);
    assert!(training["final_loss"].as_f64().unwrap() < 0.1, "{training}");

    // predict and evaluate read a file the same way.
    let (status, prediction) = call(&app, Method::POST, &format!("/models/{id}/predict"), Some(json!({ "x_path": "x.json" }))).await;
    assert_eq!(status, StatusCode::OK, "{prediction}");
    assert_eq!(prediction["rows"], 4);

    let (status, evaluation) = call(
        &app,
        Method::POST,
        &format!("/models/{id}/evaluate"),
        Some(json!({ "x_path": "x.json", "y_path": "y.json" })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{evaluation}");
    assert_eq!(evaluation["samples"], 4);

    // A file further down is fine; only leaving the directory is not.
    std::fs::create_dir_all(dir.join("sets")).unwrap();
    write_json(&dir.join("sets"), "x.json", &x);
    let (status, answer) = call(&app, Method::POST, &format!("/models/{id}/predict"), Some(json!({ "x_path": "sets/x.json" }))).await;
    assert_eq!(status, StatusCode::OK, "{answer}");

    // And the two ways of sending data may be mixed.
    let (status, answer) = call(
        &app,
        Method::POST,
        &format!("/models/{id}/evaluate"),
        Some(json!({ "x_path": "x.json", "y": y })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{answer}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_path_cannot_leave_the_data_directory() {
    let dir = data_dir("jail");
    let (x, _) = xor_samples();
    write_json(&dir, "x.json", &x);

    // A readable file beside the data directory: what a caller must not reach.
    let outside = dir.parent().unwrap().join(format!("nn_api_{}_outside.json", std::process::id()));
    std::fs::write(&outside, x.to_string()).unwrap();

    let app = api_reading(&dir);
    let id = id_of(&create(&app, xor_model()).await);

    let mut attempts = vec![
        json!(format!("../{}", outside.file_name().unwrap().to_string_lossy())),
        json!("../../etc/passwd"),
        json!("/etc/passwd"),
        json!(outside.to_string_lossy()),
        json!("sets/../../etc/passwd"),
    ];
    // A symbolic link is what canonicalising the path defends against.
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&outside, dir.join("link.json")).unwrap();
        attempts.push(json!("link.json"));
    }

    for attempt in attempts {
        let (status, error) = call(
            &app,
            Method::POST,
            &format!("/models/{id}/predict"),
            Some(json!({ "x_path": attempt })),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST, "{attempt} was allowed through: {error}");
        assert_eq!(error["error"]["code"], "invalid_request", "{attempt}");
    }

    let _ = std::fs::remove_file(&outside);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn reading_files_is_off_until_a_data_directory_is_set() {
    let app = api();
    let id = id_of(&create(&app, xor_model()).await);

    let (status, error) = call(&app, Method::POST, &format!("/models/{id}/predict"), Some(json!({ "x_path": "x.json" }))).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        error["error"]["message"].as_str().unwrap().contains("NN_API_DATA_DIR"),
        "{}",
        error["error"]["message"]
    );
}

#[tokio::test]
async fn x_must_arrive_exactly_one_way() {
    let dir = data_dir("one_way");
    let (x, _) = xor_samples();
    write_json(&dir, "x.json", &x);
    std::fs::write(dir.join("not_a_matrix.json"), r#"{ "hello": 1 }"#).unwrap();

    let app = api_reading(&dir);
    let id = id_of(&create(&app, xor_model()).await);
    let predict = format!("/models/{id}/predict");

    let cases: Vec<(Value, &str)> = vec![
        (json!({ "x": x, "x_path": "x.json" }), "not both"),
        (json!({}), "missing"),
        (json!({ "x_path": "nope.json" }), "there is no file"),
        (json!({ "x_path": "not_a_matrix.json" }), "array of arrays"),
    ];

    for (body, expected) in cases {
        let (status, error) = call(&app, Method::POST, &predict, Some(body.clone())).await;
        let message = error["error"]["message"].as_str().unwrap_or_default();

        assert_eq!(status, StatusCode::BAD_REQUEST, "{body} answered {error}");
        assert!(message.contains(expected), "{body} answered '{message}', expected it to mention '{expected}'");
    }

    let _ = std::fs::remove_dir_all(&dir);
}
