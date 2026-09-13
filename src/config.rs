//! Server settings, all overridable from the environment.

use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;

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

#[derive(Debug, Clone)]
pub struct Config {
    pub addr: SocketAddr,
    /// The biggest request body accepted; training data is JSON, so this is
    /// the real limit on how many samples one inline call can carry.
    pub body_limit_bytes: usize,
    /// Browsers only. A Java client never sends a CORS preflight.
    pub permissive_cors: bool,
    /// Where `x_path` and `y_path` are resolved, already canonical.
    ///
    /// `None`, the default, turns file input off: without it a request naming
    /// a path is refused. Letting a caller choose a path on the server's disk
    /// is how arbitrary files get read, so the directory has to be named on
    /// purpose, and nothing outside it can be reached.
    pub data_dir: Option<PathBuf>,
    /// The biggest file `x_path` may point at.
    pub max_file_bytes: u64,
    pub limits: Limits,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // Not 8080: that is Spring Boot's default, and the client calling
            // this server is expected to be a Spring Boot application on the
            // same machine. Left unconfigured, the two would fight over it.
            addr: SocketAddr::from(([127, 0, 0, 1], 8079)),
            body_limit_bytes: 32 * 1024 * 1024,
            permissive_cors: false,
            data_dir: None,
            max_file_bytes: 256 * 1024 * 1024,
            limits: Limits::default(),
        }
    }
}

impl Config {
    /// `NN_API_ADDR`, `NN_API_BODY_LIMIT_MB`, `NN_API_CORS`, `NN_API_DATA_DIR`,
    /// `NN_API_MAX_FILE_MB`, `NN_API_MAX_*`.
    pub fn from_env() -> Result<Self, String> {
        let default = Self::default();

        let addr = match env::var("NN_API_ADDR") {
            Ok(value) => value
                .parse()
                .map_err(|error| format!("NN_API_ADDR is not a socket address ('{value}'): {error}"))?,
            Err(_) => default.addr,
        };
        check_reachability(addr, matches!(env::var("NN_API_ALLOW_REMOTE").as_deref(), Ok("yes")))?;

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
            data_dir: data_dir_from_env()?,
            max_file_bytes: number_from_env("NN_API_MAX_FILE_MB", default.max_file_bytes as usize / (1024 * 1024))? as u64 * 1024 * 1024,
            limits,
        })
    }
}

/// This server is meant to be reached from the machine it runs on and no
/// further: it has no authentication, and with `NN_API_DATA_DIR` set it reads
/// files on behalf of whoever asks. So an address that is not loopback has to
/// be asked for on purpose.
fn check_reachability(addr: SocketAddr, allow_remote: bool) -> Result<(), String> {
    if addr.ip().is_loopback() || allow_remote {
        return Ok(());
    }

    Err(format!(
        "NN_API_ADDR '{addr}' would accept connections from other machines, and this server has \
         no authentication of its own. Bind it to 127.0.0.1 and reach it from this machine, or, \
         if something that authenticates sits in front of it, set NN_API_ALLOW_REMOTE=yes."
    ))
}

/// The directory `x_path` and `y_path` are read from, canonical so that a
/// resolved path can be checked against it.
fn data_dir_from_env() -> Result<Option<PathBuf>, String> {
    let Ok(value) = env::var("NN_API_DATA_DIR") else {
        return Ok(None);
    };

    let canonical = PathBuf::from(&value)
        .canonicalize()
        .map_err(|error| format!("NN_API_DATA_DIR '{value}' cannot be opened: {error}"))?;
    if !canonical.is_dir() {
        return Err(format!("NN_API_DATA_DIR '{value}' is not a directory"));
    }

    Ok(Some(canonical))
}

fn number_from_env(key: &str, fallback: usize) -> Result<usize, String> {
    match env::var(key) {
        Ok(value) => value.parse().map_err(|error| format!("{key} is not a number ('{value}'): {error}")),
        Err(_) => Ok(fallback),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(text: &str) -> SocketAddr {
        text.parse().expect("the test addresses are well formed")
    }

    #[test]
    fn loopback_addresses_are_allowed() {
        assert!(check_reachability(addr("127.0.0.1:8079"), false).is_ok());
        assert!(check_reachability(addr("127.0.0.5:8079"), false).is_ok());
        assert!(check_reachability(addr("[::1]:8079"), false).is_ok());
    }

    #[test]
    fn anything_reachable_from_outside_has_to_be_asked_for() {
        for text in ["0.0.0.0:8079", "192.168.1.10:8079", "[::]:8079"] {
            let refusal = check_reachability(addr(text), false).expect_err(&format!("{text} was allowed"));
            assert!(refusal.contains("NN_API_ALLOW_REMOTE"), "{refusal}");

            assert!(check_reachability(addr(text), true).is_ok(), "{text} was refused even when asked for");
        }
    }
}
