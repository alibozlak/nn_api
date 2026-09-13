//! Training data read from a file instead of the request body.
//!
//! A request naming a path chooses a file on the **server's** disk, which is
//! how arbitrary files get read if nothing stands in the way. Three things do:
//!
//! - the feature is off until `NN_API_DATA_DIR` names a directory,
//! - a path must be relative and may only walk down from that directory,
//! - the resolved path is canonicalised, so a symbolic link pointing out of
//!   the directory is caught rather than followed.
//!
//! Error messages name only the path the request asked for, never where it
//! landed on disk and never what the file held.

use std::path::{Component, Path, PathBuf};

use crate::config::Config;
use crate::error::ApiError;

/// A matrix from `<data dir>/<requested>`: a JSON array of equal-length arrays
/// of numbers, the same shape the request body would have carried inline.
pub fn load_matrix(config: &Config, field: &str, requested: &str) -> Result<Vec<Vec<f64>>, ApiError> {
    let path = resolve(config, field, requested)?;
    let size = std::fs::metadata(&path)
        .map_err(|error| ApiError::bad_request(format!("'{field}_path': '{requested}' cannot be read: {error}")))?
        .len();
    if size > config.max_file_bytes {
        return Err(ApiError::bad_request(format!(
            "'{field}_path': '{requested}' is {size} bytes, the limit is {}",
            config.max_file_bytes
        )));
    }

    let text = std::fs::read_to_string(&path)
        .map_err(|error| ApiError::bad_request(format!("'{field}_path': '{requested}' cannot be read: {error}")))?;

    serde_json::from_str(&text).map_err(|error| {
        ApiError::bad_request(format!(
            "'{field}_path': '{requested}' does not hold a JSON array of arrays of numbers ({error})"
        ))
    })
}

/// The file a request's path names, or an error saying why it may not have it.
fn resolve(config: &Config, field: &str, requested: &str) -> Result<PathBuf, ApiError> {
    let Some(base) = config.data_dir.as_ref() else {
        return Err(ApiError::bad_request(format!(
            "'{field}_path' needs the server to have been started with NN_API_DATA_DIR set; send '{field}' inline instead"
        )));
    };

    let relative = Path::new(requested);
    // `Path::join` with an absolute path throws the base away, so this has to
    // be refused before the join rather than after it.
    if relative.is_absolute() {
        return Err(ApiError::bad_request(format!(
            "'{field}_path' must be relative to the server's data directory, not an absolute path"
        )));
    }
    if !relative.components().all(|component| matches!(component, Component::Normal(_))) {
        return Err(ApiError::bad_request(format!(
            "'{field}_path' may only name a file below the server's data directory"
        )));
    }

    let canonical = base
        .join(relative)
        .canonicalize()
        .map_err(|_| ApiError::bad_request(format!("'{field}_path': there is no file '{requested}' in the server's data directory")))?;
    // Canonical on both sides, so a symbolic link out of the directory fails here.
    if !canonical.starts_with(base) {
        return Err(ApiError::bad_request(format!(
            "'{field}_path': '{requested}' leads outside the server's data directory"
        )));
    }

    Ok(canonical)
}
