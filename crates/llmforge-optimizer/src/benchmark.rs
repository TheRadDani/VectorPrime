use std::sync::Arc;

use anyhow::Result;
use tokio::sync::Semaphore;

use llmforge_core::{BenchmarkResult, HardwareProfile, ModelInfo, RuntimeConfig};
use llmforge_runtime::{dispatch, AdapterRegistry};

/// Prompt used for every benchmark run.
const BENCH_PROMPT: &str = "Summarize the following text in one sentence:";

/// Maximum number of concurrent benchmark processes.
const MAX_CONCURRENT: usize = 3;

/// Run all `candidates` in parallel (up to [`MAX_CONCURRENT`] at once).
///
/// Returns a result for every candidate — including `Err` entries so the
/// caller can log and filter them.
pub async fn run_benchmarks(
    candidates: Vec<RuntimeConfig>,
    model: &ModelInfo,
    _hw: &HardwareProfile,
) -> Vec<(RuntimeConfig, Result<BenchmarkResult>)> {
    let sem = Arc::new(Semaphore::new(MAX_CONCURRENT));
    let model = Arc::new(model.clone());

    let handles: Vec<_> = candidates
        .into_iter()
        .map(|config| {
            let sem = Arc::clone(&sem);
            let model = Arc::clone(&model);
            tokio::spawn(async move {
                let _permit = sem.acquire().await.expect("semaphore closed");
                // AdapterRegistry is not Send; create it inside the task after
                // acquiring the permit so it lives entirely on one thread.
                let result = tokio::task::spawn_blocking({
                    let config = config.clone();
                    let model = Arc::clone(&model);
                    move || {
                        let mut registry = AdapterRegistry::new();
                        dispatch(&mut registry, &config, &model, BENCH_PROMPT)
                    }
                })
                .await
                .unwrap_or_else(|join_err| {
                    Err(anyhow::anyhow!("benchmark task panicked: {join_err}"))
                });
                (config, result)
            })
        })
        .collect();

    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok(pair) => results.push(pair),
            Err(join_err) => {
                // The outer spawn panicked — this should not happen but we
                // must not lose the slot.
                eprintln!("benchmark outer task panicked: {join_err}");
            }
        }
    }
    results
}
