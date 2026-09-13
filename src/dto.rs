//! The JSON the client sends and receives, and the validation that stands
//! between it and neuralflow.
//!
//! neuralflow answers a bad argument with a panic. Everything checked here is
//! answered with a 400 and a sentence saying what to fix instead.

use std::fmt;

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::config::{Config, Limits};
use crate::dataset;
use crate::engine::ModelReport;
use crate::error::ApiError;
use crate::model::{
    ActivationKind, LayerSpec, LayerWeights, LossKind, ModelSpec, NewModel, OptimizerConfig, OptimizerKind, StoredModel,
    random_seed,
};

// ---------------------------------------------------------------- requests

/// `"adam"` or `{"type": "adam", "learning_rate": 0.05}`.
#[derive(Debug, Clone, Copy)]
pub enum OptimizerRequest {
    /// Just the name, so the optimizer's own default learning rate.
    Kind(OptimizerKind),
    Config { kind: OptimizerKind, learning_rate: Option<f64> },
}

/// What `optimizer` looks like on the wire, for the OpenAPI schema only:
/// [`OptimizerRequest`] parses it by hand, so utoipa cannot read the shape off
/// that type.
#[derive(Deserialize, ToSchema)]
#[serde(untagged)]
#[schema(as = OptimizerRequest)]
#[allow(dead_code)]
pub enum OptimizerRequestSchema {
    /// Just the name, so the optimizer's own default learning rate:
    /// 0.001 for Adam, 0.01 for SGD.
    Name(OptimizerKind),
    /// The name and a learning rate.
    Config {
        #[serde(rename = "type")]
        kind: OptimizerKind,
        #[schema(example = 0.05)]
        learning_rate: Option<f64>,
    },
}

impl<'de> Deserialize<'de> for OptimizerRequest {
    /// Written by hand rather than as an untagged enum: serde answers a bad
    /// untagged value with "data did not match any variant", which tells the
    /// caller nothing. This names the field that is wrong.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(OptimizerRequestVisitor)
    }
}

struct OptimizerRequestVisitor;

impl<'de> Visitor<'de> for OptimizerRequestVisitor {
    type Value = OptimizerRequest;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(r#"an optimizer name such as "adam", or an object such as {"type": "adam", "learning_rate": 0.05}"#)
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        OptimizerKind::deserialize(de::value::StrDeserializer::new(value)).map(OptimizerRequest::Kind)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut kind = None;
        let mut learning_rate = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "type" | "kind" | "name" => match kind {
                    Some(_) => return Err(de::Error::duplicate_field("type")),
                    None => kind = Some(map.next_value()?),
                },
                "learning_rate" => match learning_rate {
                    Some(_) => return Err(de::Error::duplicate_field("learning_rate")),
                    None => learning_rate = Some(map.next_value()?),
                },
                unknown => return Err(de::Error::unknown_field(unknown, &["type", "learning_rate"])),
            }
        }

        Ok(OptimizerRequest::Config {
            kind: kind.ok_or_else(|| de::Error::missing_field("type"))?,
            learning_rate,
        })
    }
}

impl OptimizerRequest {
    fn resolve(self) -> Result<OptimizerConfig, ApiError> {
        let (kind, learning_rate) = match self {
            Self::Kind(kind) => (kind, kind.default_learning_rate()),
            Self::Config { kind, learning_rate } => (kind, learning_rate.unwrap_or_else(|| kind.default_learning_rate())),
        };
        if !(learning_rate.is_finite() && learning_rate > 0.) {
            return Err(ApiError::bad_request(format!("'learning_rate' must be a positive number, got {learning_rate}")));
        }

        Ok(OptimizerConfig { kind, learning_rate })
    }
}

/// Keras' `Dense(units, activation=..., name=...)`.
#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct LayerRequest {
    pub units: usize,
    pub activation: ActivationKind,
    /// Left out, the layer is named the way Keras names it: "dense",
    /// "dense_1", ... The name is what `PUT /weights` matches on.
    #[serde(default)]
    pub name: Option<String>,
}

/// `POST /create-compile-model`
#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
#[schema(example = json!({
    "name": "xor",
    "features": 2,
    "seed": 1234,
    "layers": [
        { "units": 8, "activation": "relu", "name": "layer1" },
        { "units": 1, "activation": "sigmoid", "name": "layer2" }
    ],
    "loss": "binary_crossentropy",
    "optimizer": { "type": "adam", "learning_rate": 0.05 },
    "batch_size": 4
}))]
pub struct CreateModelRequest {
    #[serde(default)]
    pub name: Option<String>,
    /// Keras' `Input(shape=(features,))`.
    #[serde(alias = "input_features", alias = "input")]
    pub features: usize,
    pub layers: Vec<LayerRequest>,
    /// Needed by `train` and `evaluate`, not by `predict`. It can also be
    /// given later, in the train request itself.
    #[serde(default)]
    pub loss: Option<LossKind>,
    #[serde(default)]
    #[schema(value_type = Option<OptimizerRequestSchema>)]
    pub optimizer: Option<OptimizerRequest>,
    /// Makes the initial weights and the batch order reproducible. Left out or
    /// null, the server draws one and reports it back, so the model can still
    /// be rebuilt exactly later.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Samples per gradient step, remembered as this model's default so a train
    /// request need not repeat it. Keras' own default, 32, when left out.
    #[serde(default)]
    pub batch_size: Option<usize>,
    /// Weights from an earlier model, to carry one over a restart. Layers left
    /// out keep their fresh Glorot-uniform weights.
    #[serde(default)]
    pub weights: Option<Vec<LayerWeights>>,
}

impl CreateModelRequest {
    /// The validated model this request asks for.
    pub fn into_parts(self, limits: &Limits) -> Result<NewModel, ApiError> {
        if self.features == 0 || self.features > limits.max_features {
            return Err(ApiError::bad_request(format!(
                "'features' must be between 1 and {}, got {}",
                limits.max_features, self.features
            )));
        }
        if self.layers.is_empty() {
            return Err(ApiError::bad_request("'layers' must hold at least one layer"));
        }
        if self.layers.len() > limits.max_layers {
            return Err(ApiError::bad_request(format!(
                "'layers' holds {} layers, the limit is {}",
                self.layers.len(),
                limits.max_layers
            )));
        }

        let names = resolve_layer_names(&self.layers)?;
        let mut layers = Vec::with_capacity(self.layers.len());
        for (layer, name) in self.layers.iter().zip(names) {
            if layer.units == 0 || layer.units > limits.max_units_per_layer {
                return Err(ApiError::bad_request(format!(
                    "layer '{name}' has {} units, it must be between 1 and {}",
                    layer.units, limits.max_units_per_layer
                )));
            }
            layers.push(LayerSpec { name, units: layer.units, activation: layer.activation });
        }

        let spec = ModelSpec {
            features: self.features,
            layers,
            loss: self.loss,
            optimizer: resolve_optimizer(self.optimizer, self.loss)?,
            seed: self.seed.unwrap_or_else(random_seed),
        };

        let weights = self.weights.unwrap_or_default();
        validate_weights(&spec, &weights)?;

        let batch_size = self.batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        if batch_size == 0 {
            return Err(ApiError::bad_request("'batch_size' must be at least 1"));
        }

        let name = match self.name {
            Some(name) if name.trim().is_empty() => return Err(ApiError::bad_request("'name' must not be blank")),
            Some(name) => name,
            None => String::from("sequential"),
        };

        Ok(NewModel { name, spec, weights, batch_size })
    }
}

/// `POST /models/{id}/train` -- Keras' `fit`.
#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
#[schema(example = json!({
    "x": [[0, 0], [0, 1], [1, 0], [1, 1]],
    "y": [[0], [1], [1], [0]],
    "epochs": 500,
    "return_history": false
}))]
pub struct TrainRequest {
    /// One row per sample, one column per feature. Send this or `x_path`.
    #[serde(default)]
    pub x: Option<Vec<Vec<f64>>>,
    /// A JSON file below the server's data directory holding the same array
    /// `x` would have held, for data too big to put in the request.
    #[serde(default)]
    pub x_path: Option<String>,
    /// One row per sample, one column per unit of the last layer.
    #[serde(default)]
    pub y: Option<Vec<Vec<f64>>>,
    /// `y` from a file, like `x_path`.
    #[serde(default)]
    pub y_path: Option<String>,
    /// Keras' default is 1.
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    /// Samples per gradient step; more than the data set has means one step per
    /// epoch. Left out, the model's own default is used. It applies to this run
    /// only and does not change that default.
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default = "default_true")]
    pub shuffle: bool,
    /// Compiles the model before training, like calling `compile` again.
    #[serde(default)]
    pub loss: Option<LossKind>,
    #[serde(default)]
    #[schema(value_type = Option<OptimizerRequestSchema>)]
    pub optimizer: Option<OptimizerRequest>,
    /// Reseeds the model before this run.
    #[serde(default)]
    pub seed: Option<u64>,
    /// `false` leaves the per-epoch losses out of the response, which matters
    /// when the run is tens of thousands of epochs long.
    #[serde(default = "default_true")]
    pub return_history: bool,
}

/// `POST /train-with-column-scale` -- Keras' `fit` on data `column_based_scaling`
/// has divided down first. The same fields as a train request, and the model id
/// with them, since the path carries none.
#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
#[schema(example = json!({
    "model_id": "0583b9da-ba2a-4737-88ca-410819418c85",
    "x": [[1500, 3], [2500, 4], [1800, 3], [3200, 5]],
    "y": [[250000], [410000], [300000], [520000]],
    "epochs": 500,
    "return_history": false
}))]
pub struct ScaledTrainRequest {
    /// The model to train: the id `POST /create-compile-model` returned. Its
    /// last layer must have exactly one unit, as `y` may only have one column.
    #[schema(value_type = String, format = Uuid)]
    pub model_id: String,
    /// One row per sample, one column per feature. Send this or `x_path`.
    #[serde(default)]
    pub x: Option<Vec<Vec<f64>>>,
    /// A JSON file below the server's data directory holding the same array
    /// `x` would have held, for data too big to put in the request.
    #[serde(default)]
    pub x_path: Option<String>,
    /// One row per sample, one column per unit of the last layer.
    #[serde(default)]
    pub y: Option<Vec<Vec<f64>>>,
    /// `y` from a file, like `x_path`.
    #[serde(default)]
    pub y_path: Option<String>,
    /// Keras' default is 1.
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    /// Samples per gradient step; more than the data set has means one step per
    /// epoch. Left out, the model's own default is used. It applies to this run
    /// only and does not change that default.
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default = "default_true")]
    pub shuffle: bool,
    /// Compiles the model before training, like calling `compile` again.
    #[serde(default)]
    pub loss: Option<LossKind>,
    #[serde(default)]
    #[schema(value_type = Option<OptimizerRequestSchema>)]
    pub optimizer: Option<OptimizerRequest>,
    /// Reseeds the model before this run.
    #[serde(default)]
    pub seed: Option<u64>,
    /// `false` leaves the per-epoch losses out of the response, which matters
    /// when the run is tens of thousands of epochs long.
    #[serde(default = "default_true")]
    pub return_history: bool,
}

impl ScaledTrainRequest {
    /// The model id, and everything else as the train request it otherwise is.
    pub fn into_parts(self) -> (String, TrainRequest) {
        let request = TrainRequest {
            x: self.x,
            x_path: self.x_path,
            y: self.y,
            y_path: self.y_path,
            epochs: self.epochs,
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            loss: self.loss,
            optimizer: self.optimizer,
            seed: self.seed,
            return_history: self.return_history,
        };

        (self.model_id, request)
    }
}

/// `POST /models/{id}/predict` -- Keras' `predict`.
#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
#[schema(example = json!({ "x": [[0, 0], [0, 1], [1, 0], [1, 1]] }))]
pub struct PredictRequest {
    /// One row per sample, one column per feature. Send this or `x_path`.
    #[serde(default)]
    pub x: Option<Vec<Vec<f64>>>,
    /// `x` from a JSON file below the server's data directory.
    #[serde(default)]
    pub x_path: Option<String>,
    /// For a model trained through `/train-with-column-scale`: scale `x` with
    /// the ratios the model keeps and answer in `y`'s original units. Send
    /// false to pass `x` through as it is, e.g. when it is already scaled. A
    /// model with no ratios is never scaled.
    #[serde(default = "default_true")]
    pub scale: bool,
}

/// `POST /models/{id}/evaluate` -- Keras' `evaluate`.
#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
#[schema(example = json!({ "x": [[0, 0], [0, 1], [1, 0], [1, 1]], "y": [[0], [1], [1], [0]] }))]
pub struct EvaluateRequest {
    /// One row per sample, one column per feature. Send this or `x_path`.
    #[serde(default)]
    pub x: Option<Vec<Vec<f64>>>,
    /// `x` from a JSON file below the server's data directory.
    #[serde(default)]
    pub x_path: Option<String>,
    /// One row per sample, one column per unit of the last layer. Send this or `y_path`.
    #[serde(default)]
    pub y: Option<Vec<Vec<f64>>>,
    /// `y` from a JSON file below the server's data directory.
    #[serde(default)]
    pub y_path: Option<String>,
    /// Evaluates against this loss instead of the compiled one.
    #[serde(default)]
    pub loss: Option<LossKind>,
}

/// `PUT /models/{id}/weights` -- Keras' `set_weights`.
#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct SetWeightsRequest {
    /// Layers left out keep the weights they have.
    pub layers: Vec<LayerWeights>,
}

/// `POST /utils/scale` -- `column_based_scaling`.
#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
#[schema(example = json!({ "x": [[1500, 3], [2500, 4]], "y": [[250000], [300000]] }))]
pub struct ScaleRequest {
    pub x: Vec<Vec<f64>>,
    /// Exactly one column, the way `manipulate_datas_between_0_and_10` wants it.
    pub y: Vec<Vec<f64>>,
}

/// Keras' `fit` default, used when neither the model nor the request names one.
pub const DEFAULT_BATCH_SIZE: usize = 32;

fn default_epochs() -> usize {
    1
}

fn default_true() -> bool {
    true
}

// --------------------------------------------------------------- responses

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct LayerResponse {
    pub name: String,
    pub units: usize,
    pub activation: ActivationKind,
    /// Values this layer takes per sample, i.e. W's row count.
    pub input_count: usize,
    pub params: usize,
}

/// One model in full, Keras' `summary()` included.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ModelResponse {
    pub id: Uuid,
    pub name: String,
    pub features: usize,
    pub output_units: usize,
    pub total_params: usize,
    pub layers: Vec<LayerResponse>,
    pub loss: Option<LossKind>,
    pub optimizer: Option<OptimizerConfig>,
    /// The seed in use, whether the request brought it or the server drew it.
    pub seed: u64,
    /// The samples per gradient step a train request gets when it sends none.
    pub batch_size: usize,
    /// Every epoch ever run on this model.
    pub trained_epochs: usize,
    pub last_loss: Option<f64>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    /// Set by `/train-with-column-scale`; `predict` scales with these.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ten_power_ratios: Option<Vec<usize>>,
    /// The table `model.summary()` prints.
    pub summary: String,
}

impl ModelResponse {
    pub fn new(model: &StoredModel, report: ModelReport) -> Self {
        let layers = model
            .spec
            .layers
            .iter()
            .zip(report.layers)
            .map(|(spec, report)| LayerResponse {
                name: report.name,
                units: report.units,
                activation: spec.activation,
                input_count: report.input_count,
                params: report.params,
            })
            .collect();

        Self {
            id: model.id,
            name: model.name.clone(),
            features: model.spec.features,
            output_units: model.spec.output_units(),
            total_params: report.total_params,
            layers,
            loss: model.spec.loss,
            optimizer: model.spec.optimizer,
            seed: model.spec.seed,
            batch_size: model.batch_size,
            trained_epochs: model.trained_epochs,
            last_loss: model.last_loss,
            created_at_ms: model.created_at_ms,
            updated_at_ms: model.updated_at_ms,
            ten_power_ratios: model.ten_power_ratios.clone(),
            summary: report.summary,
        }
    }
}

/// A model in the list, without the summary table.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ModelListEntry {
    pub id: Uuid,
    pub name: String,
    pub features: usize,
    pub output_units: usize,
    pub total_params: usize,
    pub loss: Option<LossKind>,
    pub optimizer: Option<OptimizerConfig>,
    pub seed: u64,
    pub batch_size: usize,
    pub trained_epochs: usize,
    pub last_loss: Option<f64>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ten_power_ratios: Option<Vec<usize>>,
}

impl From<&StoredModel> for ModelListEntry {
    fn from(model: &StoredModel) -> Self {
        Self {
            id: model.id,
            name: model.name.clone(),
            features: model.spec.features,
            output_units: model.spec.output_units(),
            total_params: model.spec.total_params(),
            loss: model.spec.loss,
            optimizer: model.spec.optimizer,
            seed: model.spec.seed,
            batch_size: model.batch_size,
            trained_epochs: model.trained_epochs,
            last_loss: model.last_loss,
            created_at_ms: model.created_at_ms,
            updated_at_ms: model.updated_at_ms,
            ten_power_ratios: model.ten_power_ratios.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ModelListResponse {
    pub count: usize,
    pub models: Vec<ModelListEntry>,
}

/// What `fit` returned, plus what it cost.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct TrainResponse {
    pub id: Uuid,
    pub epochs: usize,
    pub batch_size: usize,
    pub samples: usize,
    pub loss_function: LossKind,
    pub optimizer: OptimizerConfig,
    /// The first epoch's mean loss.
    pub initial_loss: f64,
    /// The last epoch's mean loss.
    pub final_loss: f64,
    /// One entry per epoch, unless the request set `return_history` to false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss: Option<Vec<f64>>,
    /// Every epoch ever run on this model, this run included.
    pub trained_epochs: usize,
    pub duration_ms: u64,
}

/// A train response, plus how the data was scaled before `fit` saw it.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ScaledTrainResponse {
    #[serde(flatten)]
    pub training: TrainResponse,
    /// The power of ten each column was divided by: one entry per column of
    /// `x`, then `y`'s as the last. Taken from each column's first row.
    ///
    /// The model keeps them, and `predict` scales its `x` with them and answers
    /// in the original units. `evaluate` does not: divide column `j` of its `x`
    /// by `10^ten_power_ratios[j]` and its `y` by `10^` the last entry.
    #[schema(example = json!([3, 0, 5]))]
    pub ten_power_ratios: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct PredictResponse {
    pub id: Uuid,
    pub rows: usize,
    pub columns: usize,
    /// One row per sample, one column per unit of the last layer. When `scaled`
    /// is true, already multiplied back into `y`'s original units.
    pub predictions: Vec<Vec<f64>>,
    /// Whether `x` was scaled with the model's `ten_power_ratios` first.
    pub scaled: bool,
    /// The ratios used, x's columns first and y's last; absent when `scaled` is false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ten_power_ratios: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct EvaluateResponse {
    pub id: Uuid,
    pub loss_function: LossKind,
    pub loss: f64,
    pub samples: usize,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct WeightsResponse {
    pub id: Uuid,
    pub layers: Vec<LayerWeights>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ScaleResponse {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<Vec<f64>>,
    /// One exponent per feature column, and y's as the last entry. A prediction
    /// goes back to the original scale multiplied by `10^` that last entry.
    pub ten_power_ratios: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: &'static str,
    pub service: &'static str,
    pub version: &'static str,
    pub models: usize,
}

// -------------------------------------------------------------- validation

/// The values of `x` or `y`, whichever way the request chose to send them.
///
/// Reading and parsing a file is blocking work, so callers run this on the
/// blocking pool rather than on an async worker.
pub fn matrix_from_request(
    config: &Config,
    field: &str,
    inline: Option<Vec<Vec<f64>>>,
    path: Option<String>,
) -> Result<Vec<Vec<f64>>, ApiError> {
    match (inline, path) {
        (Some(_), Some(_)) => Err(ApiError::bad_request(format!(
            "send either '{field}' or '{field}_path', not both"
        ))),
        (Some(values), None) => Ok(values),
        (None, Some(path)) => dataset::load_matrix(config, field, &path),
        (None, None) => Err(ApiError::bad_request(format!(
            "'{field}' is missing; send the values inline as '{field}', or name a file with '{field}_path'"
        ))),
    }
}

/// Keras' naming, done here rather than inside neuralflow so that every layer
/// reaches the engine with an explicit name: "dense", "dense_1", "dense_2", ...
/// and a name the request gave wins.
fn resolve_layer_names(layers: &[LayerRequest]) -> Result<Vec<String>, ApiError> {
    let mut names: Vec<String> = Vec::with_capacity(layers.len());

    for (index, layer) in layers.iter().enumerate() {
        let name = match &layer.name {
            Some(name) if name.trim().is_empty() => {
                return Err(ApiError::bad_request(format!("layer {index} has a blank 'name'")));
            }
            Some(name) => name.clone(),
            None => (0_usize..)
                .map(|suffix| if suffix == 0 { String::from("dense") } else { format!("dense_{suffix}") })
                .find(|candidate| !names.contains(candidate))
                .expect("the sequence is infinite, some name is always free"),
        };

        if names.contains(&name) {
            return Err(ApiError::bad_request(format!("layer names must be unique, '{name}' is used twice")));
        }
        names.push(name);
    }

    Ok(names)
}

/// An optimizer is stored whenever there is a loss to train against, so the
/// response can always say which one a run would use.
pub fn resolve_optimizer(request: Option<OptimizerRequest>, loss: Option<LossKind>) -> Result<Option<OptimizerConfig>, ApiError> {
    match request {
        Some(request) => request.resolve().map(Some),
        None if loss.is_some() => Ok(Some(OptimizerConfig {
            kind: OptimizerKind::Adam,
            learning_rate: OptimizerKind::Adam.default_learning_rate(),
        })),
        None => Ok(None),
    }
}

/// Every weight matrix has to be `input_count` x `units` and every bias row
/// `units` long, or `set_weights` panics.
pub fn validate_weights(spec: &ModelSpec, weights: &[LayerWeights]) -> Result<(), ApiError> {
    let mut seen: Vec<&str> = Vec::with_capacity(weights.len());

    for layer_weights in weights {
        let name = layer_weights.name.as_str();
        let index = spec
            .layer_index(name)
            .ok_or_else(|| ApiError::bad_request(format!("there is no layer named '{name}' in this model")))?;
        if seen.contains(&name) {
            return Err(ApiError::bad_request(format!("layer '{name}' is given weights twice")));
        }
        seen.push(name);

        let (input_count, units) = (spec.input_count(index), spec.layers[index].units);
        if layer_weights.weights.len() != input_count {
            return Err(ApiError::bad_request(format!(
                "layer '{name}' needs a {input_count}x{units} weight matrix, got {} rows",
                layer_weights.weights.len()
            )));
        }
        if let Some((row, values)) = layer_weights.weights.iter().enumerate().find(|(_, values)| values.len() != units) {
            return Err(ApiError::bad_request(format!(
                "layer '{name}' needs a {input_count}x{units} weight matrix, but row {row} has {} values",
                values.len()
            )));
        }
        if layer_weights.bias.len() != units {
            return Err(ApiError::bad_request(format!(
                "layer '{name}' needs {units} bias values, one per unit, got {}",
                layer_weights.bias.len()
            )));
        }

        let non_finite = layer_weights.weights.iter().flatten().chain(&layer_weights.bias).any(|value| !value.is_finite());
        if non_finite {
            return Err(ApiError::bad_request(format!("layer '{name}' has a weight that is not finite")));
        }
    }

    Ok(())
}

/// x and y as `fit` and `evaluate` need them: one row per sample, x as wide as
/// the model's input and y as wide as its last layer.
pub fn validate_samples(spec: &ModelSpec, x: &[Vec<f64>], y: &[Vec<f64>], limits: &Limits) -> Result<(), ApiError> {
    validate_features(spec, x, limits)?;

    if y.len() != x.len() {
        return Err(ApiError::bad_request(format!("'x' has {} samples but 'y' has {}", x.len(), y.len())));
    }
    let output_units = spec.output_units();
    if let Some((row, values)) = y.iter().enumerate().find(|(_, values)| values.len() != output_units) {
        return Err(ApiError::bad_request(format!(
            "the last layer has {output_units} units, so every row of 'y' needs {output_units} values, but row {row} has {}",
            values.len()
        )));
    }

    Ok(())
}

/// x alone, as `predict` needs it.
pub fn validate_features(spec: &ModelSpec, x: &[Vec<f64>], limits: &Limits) -> Result<(), ApiError> {
    if x.is_empty() {
        return Err(ApiError::bad_request("'x' has no samples"));
    }
    if x.len() > limits.max_samples {
        return Err(ApiError::bad_request(format!("'x' has {} samples, the limit is {}", x.len(), limits.max_samples)));
    }
    if let Some((row, values)) = x.iter().enumerate().find(|(_, values)| values.len() != spec.features) {
        return Err(ApiError::bad_request(format!(
            "the model takes {} features per sample, but row {row} of 'x' has {}",
            spec.features,
            values.len()
        )));
    }

    Ok(())
}
