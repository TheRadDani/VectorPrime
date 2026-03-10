//! Optimization engine for VectorPrime.
//!
//! Orchestrates candidate generation, parallel benchmarking, and result
//! selection to find the best runtime configuration for a given model on the
//! current hardware.

pub mod benchmark;
pub mod estimate;
pub mod search;
pub mod selector;

pub use estimate::estimate_llamacpp;
pub use search::{bytes_per_param, generate_candidates};
pub use selector::select_best;

use anyhow::Result;
use vectorprime_core::{HardwareProfile, ModelInfo, OptimizationResult};

/// Run the full optimization pipeline and return the best configuration.
///
/// Steps:
/// 1. Generate candidate configs via [`generate_candidates`].
/// 2. Benchmark all candidates in parallel (≤ 3 concurrent) via
///    [`benchmark::run_benchmarks`].
/// 3. Select the winner via [`select_best`], optionally constrained by
///    `max_latency_ms`.
///
/// Returns `Err` if no valid configuration was found, with a diagnostic
/// message that includes the unique benchmark failure reasons so the user
/// knows which runtime binaries are missing or misconfigured.
pub async fn run_optimization(
    model: ModelInfo,
    hw: HardwareProfile,
    max_latency_ms: Option<f64>,
) -> Result<OptimizationResult> {
    let candidates = generate_candidates(&hw, &model);

    if candidates.is_empty() {
        return Err(anyhow::anyhow!(
            "no candidate configurations generated for this hardware / model combination"
        ));
    }

    let results = benchmark::run_benchmarks(candidates, &model, &hw).await;

    // Collect unique failure reasons before consuming `results` so we can
    // produce an actionable error message when nothing survives selection.
    let failure_reasons: Vec<String> = {
        let mut seen = std::collections::BTreeSet::new();
        for (_, outcome) in &results {
            if let Err(e) = outcome {
                // Use root-cause message (first line) to deduplicate noisy output.
                let msg = e
                    .chain()
                    .last()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| e.to_string());
                seen.insert(msg);
            }
        }
        seen.into_iter().collect()
    };

    // Detect the "all LlamaCpp attempts failed with NotInstalled" case.
    // When llama-cli is absent every GGUF candidate will carry an error whose
    // chain contains "was not found in PATH" (from RuntimeError::NotInstalled).
    // In that situation we replace those failed entries with static estimates
    // so the user receives a ranked recommendation without needing the binary.
    let all_llamacpp_not_installed: bool = {
        use vectorprime_core::RuntimeKind;
        let llamacpp_results: Vec<_> = results
            .iter()
            .filter(|(cfg, _)| cfg.runtime == RuntimeKind::LlamaCpp)
            .collect();
        !llamacpp_results.is_empty()
            && llamacpp_results.iter().all(|(_, r)| {
                r.as_ref()
                    .err()
                    .map(|e| {
                        e.chain()
                            .any(|c| c.to_string().contains("was not found in PATH"))
                    })
                    .unwrap_or(false)
            })
    };

    // If llama-cli is absent, replace failed LlamaCpp entries with static
    // hardware-aware estimates so select_best can still pick a winner.
    let results: Vec<(
        vectorprime_core::RuntimeConfig,
        anyhow::Result<vectorprime_core::BenchmarkResult>,
    )> = if all_llamacpp_not_installed {
        eprintln!(
            "[vectorprime] llama-cli not found — using hardware-aware estimates for GGUF configs"
        );
        results
            .into_iter()
            .map(|(cfg, outcome)| {
                if cfg.runtime == vectorprime_core::RuntimeKind::LlamaCpp && outcome.is_err() {
                    let est = estimate::estimate_llamacpp(&cfg, &model, &hw);
                    (cfg, Ok(est))
                } else {
                    (cfg, outcome)
                }
            })
            .collect()
    } else {
        results
    };

    select_best(results, &hw, max_latency_ms).ok_or_else(|| {
        if failure_reasons.is_empty() {
            // All benchmarks succeeded but results were filtered (e.g. OOM or latency).
            if let Some(limit) = max_latency_ms {
                anyhow::anyhow!(
                    "no valid configuration found: no configuration meets the latency \
                     constraint of {limit:.1} ms. Try relaxing --latency or freeing RAM."
                )
            } else {
                anyhow::anyhow!(
                    "no valid configuration found: all benchmark results exceeded the available \
                     memory budget. Try freeing RAM or using a smaller model."
                )
            }
        } else {
            let reasons = failure_reasons.join("; ");
            anyhow::anyhow!(
                "no compatible runtimes found — install the required binaries and retry.\n\
                 Failure reasons: {reasons}"
            )
        }
    })
}
