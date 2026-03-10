use anyhow::Result;
use vectorprime_core::{BenchmarkResult, ModelInfo, RuntimeConfig};

use crate::AdapterRegistry;

/// Run a full benchmark cycle for the given `config` against `model`.
///
/// Sequence: `initialize` → `load_model` → warmup `run_inference` (discarded)
/// → measured `run_inference` → `teardown`.
///
/// If the adapter returns [`vectorprime_core::RuntimeError::NotInstalled`] from
/// `initialize`, the error is propagated as-is so callers can skip the adapter.
pub fn dispatch(
    registry: &mut AdapterRegistry,
    config: &RuntimeConfig,
    model: &ModelInfo,
    prompt: &str,
) -> Result<BenchmarkResult> {
    let adapter = registry
        .get_mut(&config.runtime)
        .ok_or_else(|| anyhow::anyhow!("no adapter registered for {:?}", config.runtime))?;

    adapter.initialize(config)?;
    adapter.load_model(model)?;

    // Warmup pass — result discarded.
    let _ = adapter.run_inference(prompt)?;

    // Measured pass.
    let result = adapter.run_inference(prompt)?;

    adapter.teardown()?;

    Ok(result)
}
