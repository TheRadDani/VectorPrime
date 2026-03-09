//! Optimization engine for LLMForge.
//!
//! Orchestrates candidate generation, parallel benchmarking, and result
//! selection to find the best runtime configuration for a given model on the
//! current hardware.

pub mod benchmark;
pub mod search;
pub mod selector;

pub use search::{bytes_per_param, generate_candidates};
pub use selector::select_best;

use anyhow::Result;
use llmforge_core::{HardwareProfile, ModelInfo, OptimizationResult};

/// Run the full optimization pipeline and return the best configuration.
///
/// Steps:
/// 1. Generate candidate configs via [`generate_candidates`].
/// 2. Benchmark all candidates in parallel (≤ 3 concurrent) via
///    [`benchmark::run_benchmarks`].
/// 3. Select the winner via [`select_best`].
///
/// Returns `Err` if no valid configuration was found, with a diagnostic
/// message that includes the unique benchmark failure reasons so the user
/// knows which runtime binaries are missing or misconfigured.
pub async fn run_optimization(model: ModelInfo, hw: HardwareProfile) -> Result<OptimizationResult> {
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

    select_best(results, &hw).ok_or_else(|| {
        if failure_reasons.is_empty() {
            // All benchmarks succeeded but results were filtered (e.g. OOM).
            anyhow::anyhow!(
                "no valid configuration found: all benchmark results exceeded the available \
                 memory budget. Try freeing RAM or using a smaller model."
            )
        } else {
            let reasons = failure_reasons.join("; ");
            anyhow::anyhow!(
                "no compatible runtimes found — install the required binaries and retry.\n\
                 Failure reasons: {reasons}"
            )
        }
    })
}
