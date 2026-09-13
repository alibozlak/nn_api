//! The models the server is holding.
//!
//! Two locks, and neither one is ever held while a model trains.
//!
//! The map is guarded by a `std` lock held just long enough to look an id up.
//! Each model then has a `tokio` lock, taken twice by a training run: once to
//! copy the weights out and once to write the new ones back. A run of ten
//! minutes therefore blocks nothing -- `GET /models` still answers, and a
//! prediction alongside it reads the weights training started from.
//!
//! Two training runs on one model would still overwrite each other, so a run
//! also holds that model's training permit from beginning to end. Runs on
//! *different* models are untouched by it and go in parallel.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::{Mutex as AsyncMutex, MutexGuard, RwLock as AsyncRwLock, RwLockReadGuard, RwLockWriteGuard};
use uuid::Uuid;

use crate::config::Config;
use crate::error::ApiError;
use crate::model::{NewModel, StoredModel};

/// A model and the locks that order the work on it.
pub struct ModelEntry {
    model: AsyncRwLock<StoredModel>,
    /// Held for a whole training run, so two runs on one model queue up
    /// instead of each writing over the other's weights.
    training: AsyncMutex<()>,
}

impl ModelEntry {
    fn new(model: StoredModel) -> Self {
        Self { model: AsyncRwLock::new(model), training: AsyncMutex::new(()) }
    }

    pub async fn read(&self) -> RwLockReadGuard<'_, StoredModel> {
        self.model.read().await
    }

    pub async fn write(&self) -> RwLockWriteGuard<'_, StoredModel> {
        self.model.write().await
    }

    /// Waits for any run in progress on this model, then keeps the next one to
    /// itself until the guard is dropped.
    pub async fn training_permit(&self) -> MutexGuard<'_, ()> {
        self.training.lock().await
    }
}

pub type ModelHandle = Arc<ModelEntry>;

#[derive(Clone)]
pub struct AppState {
    inner: Arc<Inner>,
}

struct Inner {
    config: Config,
    models: RwLock<HashMap<Uuid, ModelHandle>>,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        Self { inner: Arc::new(Inner { config, models: RwLock::new(HashMap::new()) }) }
    }

    pub fn config(&self) -> &Config {
        &self.inner.config
    }

    /// Stores a freshly built model and hands back a copy of it.
    pub fn insert(&self, new: NewModel) -> Result<StoredModel, ApiError> {
        let now = now_ms();
        let model = StoredModel {
            id: Uuid::new_v4(),
            name: new.name,
            spec: new.spec,
            weights: new.weights,
            batch_size: new.batch_size,
            created_at_ms: now,
            updated_at_ms: now,
            trained_epochs: 0,
            last_loss: None,
            ten_power_ratios: None,
        };

        let mut models = self.write_models();
        let limit = self.inner.config.limits.max_models;
        if models.len() >= limit {
            return Err(ApiError::conflict(format!(
                "the server is holding {limit} models, the limit; delete one before creating another"
            )));
        }
        models.insert(model.id, Arc::new(ModelEntry::new(model.clone())));

        Ok(model)
    }

    /// The handle for `id`, or a 404 naming it.
    pub fn handle(&self, id: Uuid) -> Result<ModelHandle, ApiError> {
        self.read_models()
            .get(&id)
            .cloned()
            .ok_or_else(|| ApiError::not_found(format!("there is no model with id '{id}'")))
    }

    /// Removes `id`, reporting whether it was there.
    pub fn remove(&self, id: Uuid) -> Result<(), ApiError> {
        match self.write_models().remove(&id) {
            Some(_) => Ok(()),
            None => Err(ApiError::not_found(format!("there is no model with id '{id}'"))),
        }
    }

    /// Every model, newest first.
    pub async fn list(&self) -> Vec<StoredModel> {
        let handles: Vec<ModelHandle> = self.read_models().values().cloned().collect();

        let mut models = Vec::with_capacity(handles.len());
        for handle in handles {
            models.push(handle.read().await.clone());
        }
        models.sort_by(|left, right| right.created_at_ms.cmp(&left.created_at_ms).then(left.id.cmp(&right.id)));

        models
    }

    pub fn count(&self) -> usize {
        self.read_models().len()
    }

    /// A poisoned lock means some other request panicked while holding it. The
    /// map itself is still consistent, so the panic is not spread any further.
    fn read_models(&self) -> std::sync::RwLockReadGuard<'_, HashMap<Uuid, ModelHandle>> {
        self.inner.models.read().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn write_models(&self) -> std::sync::RwLockWriteGuard<'_, HashMap<Uuid, ModelHandle>> {
        self.inner.models.write().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Unix milliseconds, what `Instant.ofEpochMilli` on the Java side reads.
pub fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |since| since.as_millis() as u64)
}
