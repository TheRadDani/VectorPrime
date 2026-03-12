// crates/vectorprime-optimizer/src/cache.rs
//
// Persistent cache for optimization results. Stores each result as a JSON
// file in `~/.llmforge/cache/{key}.json`, where `key` is the SHA-256 hex of
// `{model_mtime_size}_{hardware_json}`. Cache misses and failures are always
// silent — callers should never see a hard error from this module.
//
// Used by `run_optimization` in `lib.rs` to avoid re-running expensive
// benchmarks when the same model on the same hardware has been seen before.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use vectorprime_core::{HardwareProfile, ModelInfo, OptimizationResult};

/// A single cache entry stored on disk.
#[derive(Serialize, Deserialize)]
pub struct CachedResult {
    /// The cached optimization result.
    pub result: OptimizationResult,
    /// String form of the model path that produced this result.
    pub model_path: String,
    /// Unix timestamp (seconds) when this entry was written.
    pub created_at: u64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Look up a previously cached optimization result for the given model and
/// hardware combination.
///
/// Returns `None` when no entry exists, the cache file is unreadable, or the
/// JSON is malformed — never returns an error.
pub fn cache_lookup(hw: &HardwareProfile, model: &ModelInfo) -> Option<OptimizationResult> {
    let key = compute_key(hw, model)?;
    let path = cache_file_path(&key)?;

    let bytes = std::fs::read(&path).ok()?;
    let entry: CachedResult = serde_json::from_slice(&bytes).ok()?;
    Some(entry.result)
}

/// Persist an optimization result so it can be retrieved by [`cache_lookup`]
/// on the next run with the same model and hardware.
///
/// Failures (e.g. the home directory is unwritable) are silently swallowed
/// and reported only as a stderr warning, so the optimizer never fails due
/// to a cache write error.
pub fn cache_store(
    hw: &HardwareProfile,
    model: &ModelInfo,
    result: &OptimizationResult,
) -> Result<()> {
    let key = compute_key(hw, model)
        .ok_or_else(|| anyhow::anyhow!("could not compute cache key"))?;
    let path = cache_file_path(&key)
        .ok_or_else(|| anyhow::anyhow!("could not resolve cache directory"))?;

    // Create the cache directory if it does not exist yet.
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }

    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let entry = CachedResult {
        result: result.clone(),
        model_path: model.path.to_string_lossy().into_owned(),
        created_at,
    };

    let json = serde_json::to_string(&entry)?;
    std::fs::write(&path, json)?;
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Compute a stable cache key from the model file identity and the hardware
/// profile.
///
/// Model identity uses `mtime + size` as a cheap content proxy so that large
/// models are not fully hashed on every run.  Returns `None` when the model
/// file metadata cannot be read or the hardware profile cannot be serialised.
fn compute_key(hw: &HardwareProfile, model: &ModelInfo) -> Option<String> {
    // --- Model proxy: mtime (seconds) + file size ---
    let meta = std::fs::metadata(&model.path).ok()?;
    let size = meta.len();
    let mtime_secs = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let model_proxy = format!("{mtime_secs}_{size}");

    // --- Hardware fingerprint ---
    let hw_json = serde_json::to_string(hw).ok()?;

    // --- Combine and hash ---
    let combined = format!("{model_proxy}_{hw_json}");
    let mut hasher = Sha256::new();
    hasher.update(combined.as_bytes());
    let digest = hasher.finalize();
    Some(hex::encode(digest))
}

/// Resolve the full path to the cache file for a given key.
///
/// Returns `None` when the home directory cannot be determined.
fn cache_file_path(key: &str) -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    Some(home.join(".llmforge").join("cache").join(format!("{key}.json")))
}
