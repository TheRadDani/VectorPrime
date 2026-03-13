//! Optimization engine for VectorPrime.
//!
//! Orchestrates candidate generation, parallel benchmarking, and result
//! selection to find the best runtime configuration for a given model on the
//! current hardware.
//!
//! ## Optimization strategies
//!
//! Two strategies are provided:
//!
//! - **Staged** (`run_optimization`, the default): Searches one parameter at a
//!   time in impact order (runtime → quantization → GPU layers → threads →
//!   batch size). Each stage fixes the winner from the previous stage before
//!   exploring the next parameter. This is significantly faster than a full
//!   cartesian product while still finding near-optimal configurations.
//!
//! - **Full cartesian** (`run_optimization_cartesian`): Benchmarks every
//!   combination of all parameters at once. Retained for backward compatibility
//!   and as a fallback when staged search yields no results.

pub mod benchmark;
pub mod estimate;
pub mod hierarchical;
pub mod search;
pub mod selector;

pub use estimate::estimate_llamacpp;
pub use search::{bytes_per_param, default_base_config, generate_candidates, generate_stage_candidates};
pub use selector::select_best;

use anyhow::Result;
use vectorprime_core::{HardwareProfile, ModelInfo, OptimizationResult, RuntimeConfig};

// ──────────────────────────────────────────────────────────────────────────────
// Stage name labels used in progress logging
// ──────────────────────────────────────────────────────────────────────────────

const STAGE_NAMES: [&str; 5] = [
    "Runtime",
    "Quantization",
    "GPU Layers",
    "Threads",
    "Batch Size",
];

// ──────────────────────────────────────────────────────────────────────────────
// LlamaCpp fallback helper
// ──────────────────────────────────────────────────────────────────────────────

/// Return `true` when every LlamaCpp benchmark result failed with "not found in
/// PATH". Used to detect the "llama-cli not installed" scenario and replace
/// failures with static estimates so the user receives a useful recommendation.
fn all_llamacpp_not_installed(
    results: &[(RuntimeConfig, anyhow::Result<vectorprime_core::BenchmarkResult>)],
) -> bool {
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
}

/// Replace failed LlamaCpp entries with static hardware-aware estimates when
/// `llama-cli` is absent. Returns the (possibly modified) result vec.
fn apply_llamacpp_fallback(
    results: Vec<(RuntimeConfig, anyhow::Result<vectorprime_core::BenchmarkResult>)>,
    model: &ModelInfo,
    hw: &HardwareProfile,
) -> Vec<(RuntimeConfig, anyhow::Result<vectorprime_core::BenchmarkResult>)> {
    if !all_llamacpp_not_installed(&results) {
        return results;
    }
    eprintln!(
        "[vectorprime] llama-cli not found — using hardware-aware estimates for GGUF configs"
    );
    results
        .into_iter()
        .map(|(cfg, outcome)| {
            if cfg.runtime == vectorprime_core::RuntimeKind::LlamaCpp && outcome.is_err() {
                let est = estimate::estimate_llamacpp(&cfg, model, hw);
                (cfg, Ok(est))
            } else {
                (cfg, outcome)
            }
        })
        .collect()
}

/// Collect the unique root-cause failure messages from a result set.
///
/// Deduplicates by using the innermost error in each chain so noisy
/// multi-line errors don't produce dozens of identical reasons.
fn collect_failure_reasons(
    results: &[(RuntimeConfig, anyhow::Result<vectorprime_core::BenchmarkResult>)],
) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    for (_, outcome) in results {
        if let Err(e) = outcome {
            let msg = e
                .chain()
                .last()
                .map(|c| c.to_string())
                .unwrap_or_else(|| e.to_string());
            seen.insert(msg);
        }
    }
    seen.into_iter().collect()
}

/// Build the "no valid configuration" error from accumulated failure reasons
/// and the optional latency constraint.
fn no_config_error(failure_reasons: &[String], max_latency_ms: Option<f64>) -> anyhow::Error {
    if failure_reasons.is_empty() {
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
}

// ──────────────────────────────────────────────────────────────────────────────
// Staged optimization (primary path)
// ──────────────────────────────────────────────────────────────────────────────

/// Run the staged optimization pipeline and return the best configuration.
///
/// Parameters are tuned in impact order across 5 sequential stages:
///
/// | Stage | Parameter      | Impact      |
/// |-------|----------------|-------------|
/// | 1     | Runtime        | Highest     |
/// | 2     | Quantization   | High        |
/// | 3     | GPU Layers     | Medium-high |
/// | 4     | Threads        | Medium      |
/// | 5     | Batch Size     | Low         |
///
/// Each stage benchmarks only the variants of its parameter (with all
/// previously-fixed values held constant), then locks in the winner before
/// proceeding to the next stage. If a stage produces no successful benchmarks,
/// the current best config is kept unchanged and the loop continues.
///
/// After all 5 stages, the `llama-cli` fallback is applied when no binary is
/// found, then `select_best` picks the final winner from the last-stage results.
///
/// Returns `Err` if no valid configuration was found across all stages.
pub async fn run_optimization(
    model: ModelInfo,
    hw: HardwareProfile,
    max_latency_ms: Option<f64>,
) -> Result<OptimizationResult> {
    // Start with a sensible default base config. Stage 1 will immediately
    // vary the runtime, so these defaults only affect stages that fail to
    // find any better value.
    let mut current_best: RuntimeConfig = default_base_config(&hw);

    // Accumulate failure reasons across all stages so we can report a useful
    // error if all stages ultimately produce nothing.
    let mut all_failure_reasons: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();

    // Track the last successful OptimizationResult so we can return it at
    // the end. Holding the full result (config + metrics) avoids needing to
    // replay select_best on a stored vec of non-Clone results.
    let mut last_best: Option<OptimizationResult> = None;

    for stage in 1u8..=5 {
        let stage_name = STAGE_NAMES[(stage - 1) as usize];
        let candidates = generate_stage_candidates(stage, &current_best, &hw, &model);

        if candidates.is_empty() {
            eprintln!(
                "[vectorprime] Stage {stage}/{total} ({stage_name}): no candidates generated, \
                 keeping current best",
                total = STAGE_NAMES.len()
            );
            continue;
        }

        eprintln!(
            "[vectorprime] Stage {stage}/{total} ({stage_name}): benchmarking {} candidate(s)…",
            candidates.len(),
            total = STAGE_NAMES.len()
        );

        let mut results = benchmark::run_benchmarks(candidates, &model, &hw).await;

        // Apply the LlamaCpp fallback on Stage 1 (runtime sweep) so that GGUF
        // models on machines without llama-cli still get useful estimates.
        if stage == 1 {
            results = apply_llamacpp_fallback(results, &model, &hw);
        }

        // Collect failure reasons from this stage before consuming results.
        for reason in collect_failure_reasons(&results) {
            all_failure_reasons.insert(reason);
        }

        // Pick the best result from this stage. select_best consumes results,
        // so we do not need to clone the (non-Clone) anyhow::Result values.
        match select_best(results, &hw, max_latency_ms) {
            Some(best) => {
                eprintln!(
                    "[vectorprime] Stage {stage}/{total} ({stage_name}): winner = {:?} / {:?} \
                     ({:.1} tok/s)",
                    best.config.runtime,
                    best.config.quantization,
                    best.metrics.tokens_per_sec,
                    total = STAGE_NAMES.len()
                );
                // Lock in this stage's winner as the base for the next stage.
                current_best = best.config.clone();
                last_best = Some(best);
            }
            None => {
                eprintln!(
                    "[vectorprime] Stage {stage}/{total} ({stage_name}): no valid result, \
                     keeping current best",
                    total = STAGE_NAMES.len()
                );
            }
        }
    }

    last_best.ok_or_else(|| {
        no_config_error(
            &all_failure_reasons.into_iter().collect::<Vec<_>>(),
            max_latency_ms,
        )
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Full cartesian optimization (backward-compat / fallback)
// ──────────────────────────────────────────────────────────────────────────────

/// Run the full cartesian optimization pipeline and return the best configuration.
///
/// Generates every combination of (runtime × quantization × gpu_layers ×
/// threads) and benchmarks all of them in parallel (≤ 3 concurrent).
///
/// This was the original `run_optimization` implementation. It is retained for
/// backward compatibility and as a fallback; the primary path is now the staged
/// optimizer (`run_optimization`).
pub async fn run_optimization_cartesian(
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

    // Collect unique failure reasons before consuming `results`.
    let failure_reasons = collect_failure_reasons(&results);

    // Apply the LlamaCpp fallback when llama-cli is absent.
    let results = apply_llamacpp_fallback(results, &model, &hw);

    select_best(results, &hw, max_latency_ms)
        .ok_or_else(|| no_config_error(&failure_reasons, max_latency_ms))
}
