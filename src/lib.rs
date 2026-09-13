//! An HTTP/JSON API over [neuralflow](https://crates.io/crates/neuralflow): JSON in,
//! JSON out, so a client in any language can build, train and query a neural
//! network without linking Rust.
//!
//! ```text
//! GET    /health
//! POST   /create-compile-model    create (and compile) a model
//! GET    /models                  list them
//! GET    /models/{id}             one model, summary table included
//! DELETE /models/{id}
//! POST   /models/{id}/train       Keras' fit
//! POST   /models/{id}/predict     Keras' predict
//! POST   /models/{id}/evaluate    Keras' evaluate
//! GET    /models/{id}/weights     Keras' get_weights
//! PUT    /models/{id}/weights     Keras' set_weights
//! GET    /models/{id}/onnx        the model as an ONNX file
//! POST   /utils/scale             column_based_scaling
//! ```

pub mod config;
pub mod dto;
pub mod engine;
pub mod error;
pub mod extract;
pub mod model;
pub mod routes;
pub mod store;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::config::Config;
use crate::error::{ApiError, panic_message};
use crate::store::AppState;

/// The whole API, ready to serve. Tests build it without a socket.
pub fn app(state: AppState) -> Router {
    let permissive_cors = state.config().permissive_cors;
    let body_limit = state.config().body_limit_bytes;

    let router = Router::new()
        .route("/health", get(routes::health))
        .route("/create-compile-model", post(routes::create_model))
        .route("/models", get(routes::list_models))
        .route("/models/{id}", get(routes::get_model).delete(routes::delete_model))
        .route("/models/{id}/train", post(routes::train))
        .route("/models/{id}/predict", post(routes::predict))
        .route("/models/{id}/evaluate", post(routes::evaluate))
        .route("/models/{id}/weights", get(routes::get_weights).put(routes::set_weights))
        .route("/models/{id}/onnx", get(routes::export_onnx))
        .route("/utils/scale", post(routes::scale))
        .fallback(routes::not_found)
        .method_not_allowed_fallback(method_not_allowed)
        // Training data arrives as JSON numbers, so the body limit is the real
        // limit on how many samples one call can carry.
        .layer(DefaultBodyLimit::max(body_limit))
        .layer(TraceLayer::new_for_http())
        // The handlers already catch neuralflow's panics; this is the net under
        // everything else, so one bad request never takes the server down.
        .layer(CatchPanicLayer::custom(panic_to_json))
        .with_state(state);

    match permissive_cors {
        true => router.layer(CorsLayer::permissive()),
        false => router,
    }
}

/// The router plus the state, from the environment.
pub fn app_from_config(config: Config) -> Router {
    app(AppState::new(config))
}

async fn method_not_allowed() -> ApiError {
    ApiError::method_not_allowed("that path exists, but not with this method; the README lists what each route accepts")
}

/// Keeps a panic from closing the connection with nothing on it.
fn panic_to_json(payload: Box<dyn std::any::Any + Send + 'static>) -> axum::response::Response {
    let message = panic_message(&payload);
    tracing::error!(panic = %message, "a handler panicked");

    (
        StatusCode::INTERNAL_SERVER_ERROR,
        axum::Json(serde_json::json!({
            "error": { "code": "internal_error", "message": message }
        })),
    )
        .into_response()
}
