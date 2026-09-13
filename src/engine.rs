//! The bridge to neuralflow.
//!
//! Every function here builds a `Sequential` from stored data, does one thing
//! with it, and returns plain data again. Nothing in this module is `Send`
//! while it runs, so callers run it inside `spawn_blocking` -- the model is
//! created and dropped on that one thread.
//!
//! neuralflow reports bad arguments by panicking, so callers wrap these calls
//! in [`crate::error::catch_engine_panic`].

use neuralflow::prelude::*;

use crate::error::ApiError;
use crate::model::{LayerWeights, LossKind, ModelSpec, OptimizerConfig, OptimizerKind};

/// What one training run produced.
pub struct TrainOutcome {
    /// The mean loss of every epoch, first to last.
    pub loss_history: Vec<f64>,
    pub weights: Vec<LayerWeights>,
}

/// A built model, read back as data: what it weighs and what it looks like.
pub struct ModelReport {
    pub weights: Vec<LayerWeights>,
    /// The table `model.summary()` prints.
    pub summary: String,
    pub total_params: usize,
    pub layers: Vec<LayerReport>,
}

/// The numbers a client sees about a layer without asking for its weights.
pub struct LayerReport {
    pub name: String,
    pub units: usize,
    pub input_count: usize,
    pub params: usize,
}

/// Keras' `Sequential(...)` + `compile(...)` + `set_weights(...)`.
///
/// A model is always built with every layer's name spelled out, so neuralflow's
/// own "dense", "dense_1" naming never has to be guessed at from the outside.
pub fn build(spec: &ModelSpec, weights: &[LayerWeights]) -> Sequential {
    // Before the layers: the weights are drawn as soon as `Sequential::new`
    // runs, and `fit` shuffles from the same generator afterwards. Every spec
    // carries a seed, so this never leaves the generator on whatever state the
    // blocking thread happened to be in.
    set_random_seed(spec.seed);

    let layers = spec
        .layers
        .iter()
        .map(|layer| Dense::new(layer.units, layer.activation.into()).name(&layer.name))
        .collect();
    let mut model = Sequential::new(Input::new(spec.features), layers);

    if let (Some(loss), Some(optimizer)) = (spec.loss, spec.optimizer) {
        compile(&mut model, loss, optimizer);
    }

    for layer_weights in weights {
        let (row_count, col_count) = (layer_weights.weights.len(), layer_weights.bias.len());
        let flat: Vec<f64> = layer_weights.weights.iter().flatten().copied().collect();
        let weight_matrix = Matrix::from_vec(row_count, col_count, flat).expect("weights are validated before they are stored");
        let bias_matrix = Matrix::from_vec(1, col_count, layer_weights.bias.clone()).expect("a bias row is always 1 x units");

        model.get_layer_mut(&layer_weights.name).set_weights(weight_matrix, bias_matrix);
    }

    model
}

/// Keras' `compile(loss=..., optimizer=...)`. `compile` takes the loss and the
/// optimizer by value as generics, so each pair is spelled out once.
fn compile(model: &mut Sequential, loss: LossKind, optimizer: OptimizerConfig) {
    match loss {
        LossKind::BinaryCrossentropy => compile_with(model, BinaryCrossentropy, optimizer),
        LossKind::MeanSquaredError => compile_with(model, MeanSquaredError, optimizer),
    }
}

fn compile_with(model: &mut Sequential, loss: impl Loss + 'static, optimizer: OptimizerConfig) {
    match optimizer.kind {
        OptimizerKind::Sgd => model.compile(loss, SGD::new(optimizer.learning_rate)),
        OptimizerKind::Adam => model.compile(loss, Adam::new(optimizer.learning_rate)),
    }
}

/// Keras' `get_weights()` for every layer, in the order the layers run.
pub fn weights_of(model: &Sequential) -> Vec<LayerWeights> {
    model
        .get_layers()
        .iter()
        .map(|layer| {
            let (weight_matrix, bias_matrix) = layer.get_weights();

            LayerWeights {
                name: layer.get_name().to_string(),
                weights: rows_of(weight_matrix),
                bias: bias_matrix.as_slice().to_vec(),
            }
        })
        .collect()
}

/// Keras' `fit(x, y, ...)`, then the weights it left behind.
pub fn train(spec: &ModelSpec, weights: &[LayerWeights], x: &Matrix, y: &Matrix, options: FitOptions) -> TrainOutcome {
    let mut model = build(spec, weights);
    // `verbose` would print one line per epoch to the server's stdout; the
    // caller gets the same numbers in the response instead.
    let history = model.fit(x, y, FitOptions { verbose: false, ..options });

    TrainOutcome { loss_history: history.loss, weights: weights_of(&model) }
}

/// `column_based_scaling` on x and y, then `fit` on what it produced.
///
/// Returns the run and the power of ten every column was divided by, x's
/// columns first and y's last. Scaling draws nothing from the random generator,
/// so a seeded run is as repeatable as an unscaled one.
pub fn train_with_column_scale(
    spec: &ModelSpec,
    weights: &[LayerWeights],
    x: Matrix,
    y: Matrix,
    options: FitOptions,
) -> (TrainOutcome, Vec<usize>) {
    let (scaled_x, scaled_y, ten_power_ratios) =
        neuralflow::column_based_scaling::manipulate_datas_between_0_and_10(x, y);

    (train(spec, weights, &scaled_x, &scaled_y, options), ten_power_ratios)
}

/// Keras' `predict(x)`.
pub fn predict(spec: &ModelSpec, weights: &[LayerWeights], x: &Matrix) -> Vec<Vec<f64>> {
    rows_of(&build(spec, weights).predict(x))
}

/// `column_based_scaling`'s division, replayed on new samples: column `j` of
/// every row is multiplied by `10^-ratios[j]`, the exact expression neuralflow
/// uses, so a row the model was trained on comes out bit for bit the same.
pub fn scale_x_columns(rows: &mut [Vec<f64>], ratios: &[usize]) {
    for row in rows {
        for (value, ratio) in row.iter_mut().zip(ratios) {
            *value *= ten_to_the(-(*ratio as i32));
        }
    }
}

/// The other way for the output: every value multiplied by `10^y_ratio`, back
/// into the units `y` was sent in.
pub fn unscale_predictions(rows: &mut [Vec<f64>], y_ratio: usize) {
    let factor = ten_to_the(y_ratio as i32);
    rows.iter_mut().flatten().for_each(|value| *value *= factor);
}

fn ten_to_the(exponent: i32) -> f64 {
    10.0_f64.powi(exponent)
}

/// Keras' `evaluate(x, y)`.
pub fn evaluate(spec: &ModelSpec, weights: &[LayerWeights], x: &Matrix, y: &Matrix) -> f64 {
    build(spec, weights).evaluate(x, y)
}

/// Builds the model once and reads everything back off it: the weights it
/// ended up with, Keras' summary table, and one row per layer.
///
/// Creating a model needs the weights and the description both, and this is
/// what keeps that to a single build -- for a model of a million parameters,
/// building twice would mean drawing a million random weights twice.
pub fn describe(spec: &ModelSpec, weights: &[LayerWeights]) -> ModelReport {
    let model = build(spec, weights);
    let layers = model
        .get_layers()
        .iter()
        .map(|layer| LayerReport {
            name: layer.get_name().to_string(),
            units: layer.get_units(),
            input_count: layer.get_weights().0.row_count(),
            params: layer.count_params(),
        })
        .collect();

    ModelReport {
        weights: weights_of(&model),
        summary: model.to_string(),
        total_params: model.count_params(),
        layers,
    }
}

/// The model as an ONNX file, ready to run under onnxruntime or tract.
pub fn onnx_bytes(spec: &ModelSpec, weights: &[LayerWeights]) -> Vec<u8> {
    build(spec, weights).to_onnx_bytes()
}

/// `column_based_scaling`: divides every column by a power of ten. Returns the
/// scaled x, the scaled y and the exponent used for each, y's last.
pub fn scale_columns(x: Matrix, y: Matrix) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>) {
    let (scaled_x, scaled_y, ten_power_ratios) =
        neuralflow::column_based_scaling::manipulate_datas_between_0_and_10(x, y);

    (rows_of(&scaled_x), rows_of(&scaled_y), ten_power_ratios)
}

/// A matrix as the nested JSON array the client sends and receives.
pub fn rows_of(matrix: &Matrix) -> Vec<Vec<f64>> {
    matrix.as_slice().chunks(matrix.col_count().max(1)).map(<[f64]>::to_vec).collect()
}

/// The JSON array back as a matrix. Every row has to be as long as the first.
pub fn matrix_of(rows: &[Vec<f64>], field: &str) -> Result<Matrix, ApiError> {
    let col_count = rows.first().map_or(0, Vec::len);
    if let Some((index, row)) = rows.iter().enumerate().find(|(_, row)| row.len() != col_count) {
        return Err(ApiError::bad_request(format!(
            "'{field}' must be rectangular: row 0 has {col_count} values but row {index} has {}",
            row.len()
        )));
    }

    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    if let Some(position) = flat.iter().position(|value| !value.is_finite()) {
        return Err(ApiError::bad_request(format!(
            "'{field}' contains a value that is not finite at row {}, column {}",
            position / col_count.max(1),
            position % col_count.max(1)
        )));
    }

    Matrix::from_vec(rows.len(), col_count, flat)
        .map_err(|error| ApiError::bad_request(format!("'{field}' is not a valid matrix: {error:?}")))
}
