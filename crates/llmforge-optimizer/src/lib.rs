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
/// Returns `Err` if no valid configuration was found.
pub async fn run_optimization(model: ModelInfo, hw: HardwareProfile) -> Result<OptimizationResult> {
    let candidates = generate_candidates(&hw, &model);

    if candidates.is_empty() {
        return Err(anyhow::anyhow!(
            "no candidate configurations generated for this hardware / model combination"
        ));
    }

    let results = benchmark::run_benchmarks(candidates, &model, &hw).await;

    select_best(results, &hw)
        .ok_or_else(|| anyhow::anyhow!("no valid configuration found after benchmarking"))
}
