//! Server settings, all overridable from the environment.

use std::env;
use std::net::SocketAddr;

/// Caps on what one request may ask for. They exist so a single call cannot
/// pin a core or exhaust memory; raise them if your data is bigger.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_models: usize,
    pub max_layers: usize,
    pub max_units_per_layer: usize,
    pub max_features: usize,
    pub max_samples: usize,
    pub max_epochs: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_models: 256,
            max_layers: 64,
            max_units_per_layer: 4096,
            max_features: 4096,
            max_samples: 1_000_000,
            max_epochs: 100_000,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub addr: SocketAddr,
    /// The biggest request body accepted; training data is JSON, so this is
    /// the real limit on how many samples one call can carry.
    pub body_limit_bytes: usize,
    /// Browsers only. A Java client never sends a CORS preflight.
    pub permissive_cors: bool,
    pub limits: Limits,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            addr: SocketAddr::from(([127, 0, 0, 1], 8080)),
            body_limit_bytes: 32 * 1024 * 1024,
            permissive_cors: false,
            limits: Limits::default(),
        }
    }
}

impl Config {
    /// `NN_API_ADDR`, `NN_API_BODY_LIMIT_MB`, `NN_API_CORS`, `NN_API_MAX_*`.
    pub fn from_env() -> Result<Self, String> {
        let default = Self::default();

        let addr = match env::var("NN_API_ADDR") {
            Ok(value) => value
                .parse()
                .map_err(|error| format!("NN_API_ADDR is not a socket address ('{value}'): {error}"))?,
            Err(_) => default.addr,
        };
        let body_limit_mb = number_from_env("NN_API_BODY_LIMIT_MB", default.body_limit_bytes / (1024 * 1024))?;
        let limits = Limits {
            max_models: number_from_env("NN_API_MAX_MODELS", default.limits.max_models)?,
            max_layers: number_from_env("NN_API_MAX_LAYERS", default.limits.max_layers)?,
            max_units_per_layer: number_from_env("NN_API_MAX_UNITS", default.limits.max_units_per_layer)?,
            max_features: number_from_env("NN_API_MAX_FEATURES", default.limits.max_features)?,
            max_samples: number_from_env("NN_API_MAX_SAMPLES", default.limits.max_samples)?,
            max_epochs: number_from_env("NN_API_MAX_EPOCHS", default.limits.max_epochs)?,
        };

        Ok(Self {
            addr,
            body_limit_bytes: body_limit_mb * 1024 * 1024,
            permissive_cors: matches!(env::var("NN_API_CORS").as_deref(), Ok("permissive")),
            limits,
        })
    }
}

fn number_from_env(key: &str, fallback: usize) -> Result<usize, String> {
    match env::var(key) {
        Ok(value) => value.parse().map_err(|error| format!("{key} is not a number ('{value}'): {error}")),
        Err(_) => Ok(fallback),
    }
}
