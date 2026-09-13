//! Every failure leaves this API as the same JSON shape, so the calling
//! client can parse an error exactly the way it parses a result:
//!
//! ```json
//! { "error": { "code": "invalid_request", "message": "..." } }
//! ```

use std::any::Any;
use std::panic::AssertUnwindSafe;

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

/// An error on its way back to the client: an HTTP status, a stable machine
/// readable code, and a message meant for the developer reading the response.
#[derive(Debug, Clone)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    /// The request was malformed or asked for something impossible: 400.
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request", message)
    }

    /// No model with that id: 404.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found", message)
    }

    /// The body parsed, but neuralflow refused the values (a shape mismatch,
    /// a zero batch size, ...): 422. This is where a caught panic lands.
    pub fn engine(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, "engine_error", message)
    }

    /// The server's own fault: 500.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", message)
    }

    /// The path exists, but not with this method: 405.
    pub fn method_not_allowed(message: impl Into<String>) -> Self {
        Self::new(StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed", message)
    }

    /// Too many models held at once: 409.
    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, "conflict", message)
    }

    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self { status, code, message: message.into() }
    }

    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn code(&self) -> &str {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    error: ErrorDetail<'a>,
}

#[derive(Serialize)]
struct ErrorDetail<'a> {
    code: &'a str,
    message: &'a str,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(ErrorEnvelope {
            error: ErrorDetail { code: self.code, message: &self.message },
        });

        (self.status, body).into_response()
    }
}

impl From<JsonRejection> for ApiError {
    /// Axum's own JSON errors are plain text; this turns them into the
    /// envelope above so *every* response of this API is JSON.
    fn from(rejection: JsonRejection) -> Self {
        let status = rejection.status();
        let code = match status {
            StatusCode::PAYLOAD_TOO_LARGE => "payload_too_large",
            StatusCode::UNSUPPORTED_MEDIA_TYPE => "unsupported_media_type",
            _ => "invalid_json",
        };

        Self::new(status, code, rejection.body_text())
    }
}

/// Runs `work` and turns a panic inside it into a 422 instead of a dropped
/// connection.
///
/// neuralflow reports every bad argument by panicking (`"y has 1 columns but
/// the last layer has 2 units !!"`), and those messages are exactly what the
/// caller needs to see. Validation upstream catches the common mistakes with a
/// 400; this is the net under everything it does not know about.
pub fn catch_engine_panic<T>(work: impl FnOnce() -> T) -> Result<T, ApiError> {
    std::panic::catch_unwind(AssertUnwindSafe(work)).map_err(|payload| ApiError::engine(panic_message(&payload)))
}

/// The text a `panic!` was given, if it was given text at all.
pub fn panic_message(payload: &Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        String::from("the neural network engine panicked without a message")
    }
}
