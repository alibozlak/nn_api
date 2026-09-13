//! What the server keeps about a model.
//!
//! `Sequential` is not `Send` (it boxes `dyn Loss` and `dyn Optimizer`), so it
//! can never live in the shared state of a multi threaded server. A model is
//! therefore stored as plain data -- the architecture and the weights -- and
//! [`crate::engine`] rebuilds a `Sequential` from it inside the blocking task
//! that needs one. The data is `Send + Sync`, serialises straight to JSON, and
//! a client can save it and hand it back later.

use neuralflow::prelude::Activation;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

/// Keras' activations, as the JSON names the client sends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ActivationKind {
    #[serde(alias = "ReLU", alias = "RELU", alias = "ReLu")]
    Relu,
    #[serde(alias = "Sigmoid")]
    Sigmoid,
    #[serde(alias = "Linear", alias = "identity", alias = "none")]
    Linear,
}

impl From<ActivationKind> for Activation {
    fn from(kind: ActivationKind) -> Self {
        match kind {
            ActivationKind::Relu => Activation::ReLU,
            ActivationKind::Sigmoid => Activation::Sigmoid,
            ActivationKind::Linear => Activation::Linear,
        }
    }
}

/// The losses neuralflow implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum LossKind {
    #[serde(alias = "bce", alias = "BinaryCrossentropy", alias = "binary_cross_entropy")]
    BinaryCrossentropy,
    #[serde(alias = "mse", alias = "MeanSquaredError", alias = "mean_square_error")]
    MeanSquaredError,
}

/// The optimizers neuralflow implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    #[serde(alias = "SGD", alias = "gradient_descent")]
    Sgd,
    #[serde(alias = "Adam", alias = "ADAM")]
    Adam,
}

impl OptimizerKind {
    /// What `SGD::default()` and `Adam::default()` use -- Keras' own defaults.
    /// neuralflow has no getter for the rate, so the numbers are repeated here
    /// to be able to report the one actually in use.
    pub fn default_learning_rate(self) -> f64 {
        match self {
            Self::Sgd => 0.01,
            Self::Adam => 0.001,
        }
    }
}

/// An optimizer with its learning rate resolved.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct OptimizerConfig {
    #[serde(rename = "type")]
    pub kind: OptimizerKind,
    pub learning_rate: f64,
}

/// One `Dense` layer. The name is always filled in, so weights can be matched
/// to layers by name in both directions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerSpec {
    pub name: String,
    pub units: usize,
    pub activation: ActivationKind,
}

/// Everything needed to rebuild the model, minus the weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Keras' `Input(shape=(features,))`: columns per sample.
    pub features: usize,
    pub layers: Vec<LayerSpec>,
    /// Set by `compile`; `fit` and `evaluate` need it, `predict` does not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss: Option<LossKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer: Option<OptimizerConfig>,
    /// Makes the initial weights and the batch order reproducible. A request
    /// that brings no seed is given one from [`random_seed`], so every model
    /// has one and every model can be built again exactly.
    pub seed: u64,
}

impl ModelSpec {
    /// How many inputs the layer at `index` takes: the features for the first
    /// layer, the previous layer's units for the rest.
    pub fn input_count(&self, index: usize) -> usize {
        match index {
            0 => self.features,
            _ => self.layers[index - 1].units,
        }
    }

    /// The units of the last layer, i.e. the columns every target row needs.
    pub fn output_units(&self) -> usize {
        self.layers.last().map_or(0, |layer| layer.units)
    }

    /// Keras' `count_params()`, straight from the shapes: every layer
    /// contributes `input_count * units` weights and `units` biases.
    pub fn total_params(&self) -> usize {
        (0..self.layers.len()).map(|index| self.input_count(index) * self.layers[index].units + self.layers[index].units).sum()
    }

    pub fn layer_index(&self, name: &str) -> Option<usize> {
        self.layers.iter().position(|layer| layer.name == name)
    }
}

/// Keras' `get_weights()` for one layer: W as rows of the input, b as one
/// value per unit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({ "name": "layer2", "weights": [[1.0], [0.0]], "bias": [0.5] }))]
pub struct LayerWeights {
    pub name: String,
    /// `input_count` rows of `units` values each.
    pub weights: Vec<Vec<f64>>,
    /// One bias per unit.
    pub bias: Vec<f64>,
}

/// A model as the store holds it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredModel {
    pub id: Uuid,
    pub name: String,
    pub spec: ModelSpec,
    pub weights: Vec<LayerWeights>,
    /// Samples per gradient step, used whenever a train request leaves it out.
    /// Keras has no such thing -- `batch_size` is an argument to `fit` there --
    /// but a client would otherwise repeat it on every call. It sits beside the
    /// spec rather than in it because rebuilding the network does not need it.
    pub batch_size: usize,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    /// Epochs run over this model's whole life, summed over every train call.
    pub trained_epochs: usize,
    /// The last epoch's loss of the last train call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_loss: Option<f64>,
    /// The power of ten each column was divided by in the last column-scaled
    /// training run: one per feature, then y's. `predict` scales x with these
    /// and turns the output back into y's units. `None` until such a run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ten_power_ratios: Option<Vec<usize>>,
}

/// Everything a new model is made of, before the store gives it an id.
pub struct NewModel {
    pub name: String,
    pub spec: ModelSpec,
    pub weights: Vec<LayerWeights>,
    pub batch_size: usize,
}

/// A seed from the operating system's randomness, for a request that did not
/// bring one of its own.
///
/// `uuid` is already here and its v4 generator reads the system random source,
/// so this saves a dependency on `rand`. Six of a v4's bits are its version and
/// variant fields rather than entropy; folding the two halves together keeps
/// every random bit of the other 122.
pub fn random_seed() -> u64 {
    let bytes = Uuid::new_v4().into_bytes();
    let half = |slice: &[u8]| u64::from_le_bytes(slice.try_into().expect("each half of a uuid is eight bytes"));

    half(&bytes[..8]) ^ half(&bytes[8..])
}
