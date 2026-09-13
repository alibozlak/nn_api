//! The OpenAPI description of the API, and where it is served.
//!
//! The schemas come from the request and response types themselves and the
//! operations from the `#[utoipa::path]` on each handler, so the document is
//! generated from the same code that parses and answers a request.

use utoipa::{OpenApi, ToSchema};

use crate::routes;

/// The document itself, as JSON.
pub const SPEC_PATH: &str = "/api-docs/openapi.json";

/// Swagger UI, which reads [`SPEC_PATH`] and can send requests to this server.
pub const UI_PATH: &str = "/swagger-ui";

// Left to itself utoipa would describe `Vec<u8>` as an array of integers, so
// the export is spelled out as the binary file it is.
/// An ONNX model file: the graph and every weight, readable by onnxruntime,
/// tract or any other ONNX runtime.
#[derive(ToSchema)]
#[schema(value_type = String, format = Binary)]
pub struct OnnxFile(#[allow(dead_code)] Vec<u8>);

#[derive(OpenApi)]
#[openapi(
    info(
        title = "nn_api",
        description = "An HTTP/JSON API over neuralflow: build, train and query a neural network. \
                       Every non-2xx answer has the ErrorResponse shape."
    ),
    paths(
        routes::health,
        routes::create_model,
        routes::list_models,
        routes::get_model,
        routes::delete_model,
        routes::train,
        routes::train_with_column_scale,
        routes::predict,
        routes::evaluate,
        routes::get_weights,
        routes::set_weights,
        routes::export_onnx,
        routes::scale,
    ),
    tags(
        (name = "models", description = "Create, list, inspect and delete models."),
        (name = "learning", description = "Keras' fit, predict and evaluate. x and y can be sent inline or read from a file with x_path and y_path."),
        (name = "weights", description = "Keras' get_weights and set_weights, and the ONNX export."),
        (name = "utils", description = "Helpers from neuralflow that need no model."),
        (name = "server", description = "Whether the server is up."),
    )
)]
pub struct ApiDoc;
