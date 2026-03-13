//! Optimization engine for VectorPrime.
//!
//! Orchestrates candidate generation, parallel benchmarking, and result
//! selection to find the best runtime configuration for a given model on the
//! current hardware.
//!
//! ## Optimization strategies
//!
//! Three strategies are provided:
//!
//! - **Bayesian** (`run_optimization`, the default): 4-stage pipeline where
//!   stages 1–3 prune the search space (hardware profiling, model analysis,
//!   runtime preselection) and stage 4 runs a Tree-structured Parzen Estimator
//!   (TPE) over the remaining space. Total: 12 benchmark calls max.
//!
//! - **Staged** (`run_optimization_staged`): The previous default; searches one
//!   parameter at a time in impact order (runtime → quantization → GPU layers
//!   → threads → final benchmark). Retained for reference and testing.
//!
//! - **Full cartesian** (`run_optimization_cartesian`): Benchmarks every
//!   combination of all parameters at once. Retained for backward compatibility
//!   and as a fallback when Bayesian search yields no results.

pub mod bayes;
pub mod benchmark;
pub mod estimate;
pub mod hierarchical;
pub mod search;
pub mod selector;

pub use estimate::estimate_llamacpp;
pub use search::{bytes_per_param, default_base_config, generate_candidates, generate_stage_candidates};
pub use selector::select_best;

use anyhow::Result;
use vectorprime_core::{BenchmarkResult, HardwareProfile, ModelInfo, OptimizationResult, RuntimeConfig};

use bayes::{SearchSpace, TpeModel, thread_options_from_cores};
use search::{
    HardwareContext, ModelContext, batch_size_for_model, estimate_max_gpu_layers,
    gpu_layer_candidates, preselect_runtimes, quantization_candidates, thread_candidates,
};

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
// Bayesian optimization (primary path — 4 stages, TPE acquisition)
// ──────────────────────────────────────────────────────────────────────────────

/// Number of initial quasi-random (Halton) samples before TPE takes over.
const N_INITIAL: usize = 5;
/// Number of TPE refinement iterations after the initial samples.
const N_ITER: usize = 7;
/// Number of random candidates generated per TPE iteration.
const N_CANDIDATES: usize = 24;

/// Run the 4-stage Bayesian optimization pipeline and return the best configuration.
///
/// **Stages 1–3** are pure analysis (zero benchmark calls) and produce the
/// pruned search space consumed by Stage 4.
///
/// **Stage 4** runs a Tree-structured Parzen Estimator (TPE) over the remaining
/// parameter space (runtime × quantization × gpu_layers × threads × batch_size).
///
/// | Stage | Name                    | Benchmarks             |
/// |-------|-------------------------|------------------------|
/// | 1     | Hardware Profiling      | 0                      |
/// | 2     | Model Graph Analysis    | 0                      |
/// | 3     | Runtime Preselection    | 0                      |
/// | 4     | Bayesian Optimization   | N_INITIAL + N_ITER ≤ 12 |
///
/// Falls back to `run_optimization_cartesian` when all 12 evaluations fail.
/// Returns `Err` only when the fallback also produces no valid configuration.
pub async fn run_optimization(
    model: ModelInfo,
    hw: HardwareProfile,
    max_latency_ms: Option<f64>,
) -> Result<OptimizationResult> {
    // ── Stage 1: Hardware Profiling ───────────────────────────────────────────
    let hw_ctx = HardwareContext::from_hw(&hw);
    eprintln!(
        "[Stage 1/4] Hardware: {} cores, GPU={}, VRAM={}, RAM={}MB",
        hw_ctx.cpu_cores,
        if hw_ctx.has_gpu { "yes" } else { "no" },
        hw_ctx
            .vram_mb
            .map(|v| format!("{v}MB"))
            .unwrap_or_else(|| "none".to_string()),
        hw_ctx.ram_available_mb,
    );

    // ── Stage 2: Model Graph Analysis ─────────────────────────────────────────
    let model_ctx = ModelContext::from_model_and_hw(&model, &hw_ctx);
    eprintln!(
        "[Stage 2/4] Model: workload={:?}, params={}, FP16={}",
        model_ctx.workload,
        model_ctx
            .param_count
            .map(|p| format!("{:.1}B", p as f64 / 1e9))
            .unwrap_or_else(|| "unknown".to_string()),
        model_ctx
            .memory_fp16_mb
            .map(|m| format!("{:.0}MB", m))
            .unwrap_or_else(|| "unknown".to_string()),
    );

    // ── Stage 3: Runtime Preselection ─────────────────────────────────────────
    let selected_runtimes = preselect_runtimes(&hw_ctx, &model_ctx, &model.format);
    eprintln!(
        "[Stage 3/4] Preselected runtimes: {:?}",
        selected_runtimes
    );

    if selected_runtimes.is_empty() {
        return Err(anyhow::anyhow!(
            "no compatible runtimes available for this model format and hardware"
        ));
    }

    // Build the viable quantization list from Stage 4 of the old pipeline.
    let viable_quants = quantization_candidates(&hw_ctx, &model_ctx, &model.format);
    let quants = if viable_quants.is_empty() {
        // Fall back to the first quant from the default base config.
        vec![vectorprime_core::QuantizationStrategy::Q4_K_M]
    } else {
        viable_quants
    };

    // Determine max_gpu_layers for Stage 4 search space.
    let max_gpu_layers = if hw_ctx.has_gpu {
        hw.gpu
            .as_ref()
            .map(|g| estimate_max_gpu_layers(g, &model))
            .unwrap_or(0)
    } else {
        0
    };

    // Thread options: [cores/2, cores, cores*2] clamped to [1,64].
    let thread_opts = thread_options_from_cores(hw_ctx.cpu_cores);

    // Batch options: standard set; KV-cache default included.
    let batch_base = batch_size_for_model(&hw, &model);
    let mut batch_set: std::collections::BTreeSet<u32> = [128u32, 256, 512].iter().cloned().collect();
    batch_set.insert(batch_base);
    let batch_opts: Vec<u32> = batch_set.into_iter().collect();

    let space = SearchSpace {
        runtimes: selected_runtimes,
        quants,
        max_gpu_layers,
        thread_options: thread_opts,
        batch_options: batch_opts,
    };

    eprintln!(
        "[Stage 4/4] Bayesian optimization: {N_INITIAL} initial samples + {N_ITER} refinement iterations"
    );

    // ── Stage 4a: Initial Halton samples ──────────────────────────────────────
    let initial_points = space.halton_samples(N_INITIAL);
    let initial_configs: Vec<RuntimeConfig> = initial_points
        .iter()
        .map(|pt| space.decode(&hw, pt))
        .collect();

    let mut all_failure_reasons: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();

    let mut init_results =
        benchmark::run_benchmarks(initial_configs, &model, &hw).await;
    init_results = apply_llamacpp_fallback(init_results, &model, &hw);

    // Record failures and seed the TPE model from successful observations.
    let mut tpe = TpeModel::new(0.25);
    let mut best_config_opt: Option<RuntimeConfig> = None;
    let mut best_metrics_opt: Option<vectorprime_core::BenchmarkResult> = None;
    let mut best_tps: f64 = f64::NEG_INFINITY;

    for (idx, ((cfg, outcome), pt)) in init_results.into_iter().zip(initial_points.iter()).enumerate() {
        match outcome {
            Ok(metrics) => {
                let score = metrics.tokens_per_sec;
                eprintln!(
                    "[Bayes init {}/{N_INITIAL}] {:?}/{:?}/{} threads/{} gpu_layers → {:.1} tok/s",
                    idx + 1,
                    cfg.runtime,
                    cfg.quantization,
                    cfg.threads,
                    cfg.gpu_layers,
                    score,
                );
                tpe.observe(pt.clone(), score);
                if score > best_tps {
                    best_tps = score;
                    best_config_opt = Some(cfg);
                    best_metrics_opt = Some(metrics);
                }
            }
            Err(e) => {
                eprintln!(
                    "[Bayes init {}/{N_INITIAL}] {:?}/{:?} → ERROR: {}",
                    idx + 1,
                    cfg.runtime,
                    cfg.quantization,
                    e
                );
                let msg = e
                    .chain()
                    .last()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| e.to_string());
                all_failure_reasons.insert(msg);
            }
        }
    }

    // ── Stage 4b: TPE refinement iterations ───────────────────────────────────
    for iter in 0..N_ITER {
        let point = tpe.suggest(N_CANDIDATES, iter as u64);
        let cfg = space.decode(&hw, &point);

        eprintln!(
            "[Bayes iter {}/{N_ITER}] {:?}/{:?}/{} threads/{} gpu_layers",
            iter + 1,
            cfg.runtime,
            cfg.quantization,
            cfg.threads,
            cfg.gpu_layers,
        );

        let mut iter_results =
            benchmark::run_benchmarks(vec![cfg.clone()], &model, &hw).await;
        iter_results = apply_llamacpp_fallback(iter_results, &model, &hw);

        for (result_cfg, outcome) in iter_results {
            match outcome {
                Ok(metrics) => {
                    let score = metrics.tokens_per_sec;
                    eprintln!(
                        "[Bayes iter {}/{N_ITER}] Result: {:.1} tok/s",
                        iter + 1,
                        score,
                    );
                    tpe.observe(point.clone(), score);
                    if score > best_tps {
                        best_tps = score;
                        best_config_opt = Some(result_cfg);
                        best_metrics_opt = Some(metrics);
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[Bayes iter {}/{N_ITER}] ERROR: {}",
                        iter + 1,
                        e
                    );
                    let msg = e
                        .chain()
                        .last()
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| e.to_string());
                    all_failure_reasons.insert(msg);
                }
            }
        }
    }

    // ── Stage 4c: Report best and return ─────────────────────────────────────
    if let Some(best_cfg) = best_config_opt {
        eprintln!(
            "[Bayes final] Best: {:?}/{:?}/{} threads/{} gpu_layers → {:.1} tok/s",
            best_cfg.runtime,
            best_cfg.quantization,
            best_cfg.threads,
            best_cfg.gpu_layers,
            best_tps,
        );

        // Return the best observed result using the real benchmark metrics
        // captured alongside the winning tokens_per_sec in the loop above.
        let result = OptimizationResult {
            config: best_cfg,
            metrics: best_metrics_opt.unwrap_or(BenchmarkResult {
                tokens_per_sec: best_tps,
                latency_ms: 1_000.0 / best_tps.max(1e-6),
                peak_memory_mb: 0,
            }),
        };

        // Apply latency constraint if set.
        if let Some(limit) = max_latency_ms {
            if result.metrics.latency_ms > limit {
                // Best config violates latency — fall back to cartesian.
                eprintln!(
                    "[Bayes final] Best config latency {:.1}ms exceeds limit {:.1}ms — \
                     falling back to cartesian search",
                    result.metrics.latency_ms,
                    limit,
                );
            } else {
                return Ok(result);
            }
        } else {
            return Ok(result);
        }
    }

    // ── Fallback: all 12 evaluations failed — try cartesian ───────────────────
    let reasons: Vec<String> = all_failure_reasons.into_iter().collect();
    eprintln!(
        "[Bayes final] No successful evaluations — falling back to cartesian search. \
         Failure reasons: {}",
        if reasons.is_empty() {
            "none recorded".to_string()
        } else {
            reasons.join("; ")
        }
    );

    match run_optimization_cartesian(model, hw, max_latency_ms).await {
        Ok(r) => Ok(r),
        Err(cartesian_err) => Err(no_config_error(&reasons, max_latency_ms)
            .context(format!("cartesian fallback also failed: {cartesian_err}"))),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Staged optimization (reference / testing path — 7 stages)
// ──────────────────────────────────────────────────────────────────────────────

/// Run the 7-stage Efficient Search Pipeline and return the best configuration.
///
/// Retained as a reference implementation and for testing. The primary
/// optimization path is now `run_optimization` (Bayesian / TPE).
///
/// | Stage | Name                  | Benchmarks |
/// |-------|-----------------------|------------|
/// | 1     | Hardware Profiling    | 0          |
/// | 2     | Model Graph Analysis  | 0          |
/// | 3     | Runtime Preselection  | 0          |
/// | 4     | Quantization Search   | ≤ 3        |
/// | 5     | GPU Offload Tuning    | ≤ 3        |
/// | 6     | Threading Optim.      | ≤ 3        |
/// | 7     | Final Benchmark       | 1          |
///
/// Returns `Err` if no valid configuration was found across all stages.
pub async fn run_optimization_staged(
    model: ModelInfo,
    hw: HardwareProfile,
    max_latency_ms: Option<f64>,
) -> Result<OptimizationResult> {
    // Accumulate failure reasons across all stages so we can report a useful
    // error if all stages ultimately produce nothing.
    let mut all_failure_reasons: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();

    // ── Stage 1: Hardware Profiling (pure analysis) ───────────────────────────
    let hw_ctx = HardwareContext::from_hw(&hw);
    eprintln!(
        "[Stage 1/7] Hardware: {} cores, GPU={}, VRAM={}, RAM={}MB",
        hw_ctx.cpu_cores,
        if hw_ctx.has_gpu { "yes" } else { "no" },
        hw_ctx
            .vram_mb
            .map(|v| format!("{v}MB"))
            .unwrap_or_else(|| "none".to_string()),
        hw_ctx.ram_available_mb,
    );

    // ── Stage 2: Model Graph Analysis (pure analysis) ─────────────────────────
    let model_ctx = ModelContext::from_model_and_hw(&model, &hw_ctx);
    eprintln!(
        "[Stage 2/7] Model: workload={:?}, params={}, FP16={}",
        model_ctx.workload,
        model_ctx
            .param_count
            .map(|p| format!("{:.1}B", p as f64 / 1e9))
            .unwrap_or_else(|| "unknown".to_string()),
        model_ctx
            .memory_fp16_mb
            .map(|m| format!("{:.0}MB", m))
            .unwrap_or_else(|| "unknown".to_string()),
    );

    // ── Stage 3: Runtime Preselection (prediction + pruning, no benchmarks) ───
    let selected_runtimes = preselect_runtimes(&hw_ctx, &model_ctx, &model.format);
    eprintln!(
        "[Stage 3/7] Preselected runtimes: {:?}",
        selected_runtimes
    );

    if selected_runtimes.is_empty() {
        return Err(anyhow::anyhow!(
            "no compatible runtimes available for this model format and hardware"
        ));
    }

    // Use the best (first) preselected runtime as the base for subsequent stages.
    let best_runtime = selected_runtimes[0].clone();

    // Base batch size from KV-cache heuristic (unchanged from prior approach).
    let batch_size = batch_size_for_model(&hw, &model);

    // ── Stage 4: Quantization Search (≤ 3 benchmark calls) ───────────────────
    let quant_candidates = quantization_candidates(&hw_ctx, &model_ctx, &model.format);
    eprintln!(
        "[Stage 4/7] Quantization search: testing {:?}",
        quant_candidates
    );

    let mut best_config = default_base_config(&hw);
    best_config.runtime = best_runtime.clone();
    best_config.batch_size = batch_size;

    let stage4_candidates: Vec<RuntimeConfig> = if quant_candidates.is_empty() {
        eprintln!("[Stage 4/7] No quantization candidates after pruning, using default");
        vec![best_config.clone()]
    } else {
        quant_candidates
            .iter()
            .map(|q| RuntimeConfig {
                runtime: best_runtime.clone(),
                quantization: q.clone(),
                threads: hw_ctx.cpu_cores,
                batch_size,
                gpu_layers: 0,
            })
            .collect()
    };

    {
        let mut results = benchmark::run_benchmarks(stage4_candidates, &model, &hw).await;
        // Apply LlamaCpp fallback here (moved from old Stage 1 runtime sweep).
        results = apply_llamacpp_fallback(results, &model, &hw);
        for reason in collect_failure_reasons(&results) {
            all_failure_reasons.insert(reason);
        }
        if let Some(winner) = select_best(results, &hw, max_latency_ms) {
            eprintln!(
                "[Stage 4/7] Winner: {:?} / {:?} ({:.1} tok/s)",
                winner.config.runtime, winner.config.quantization, winner.metrics.tokens_per_sec
            );
            best_config = winner.config.clone();
        } else {
            eprintln!("[Stage 4/7] No valid result — keeping default config");
        }
    }

    // ── Stage 5: GPU Offload Tuning (≤ 3–4 benchmark calls) ─────────────────
    if !hw_ctx.has_gpu {
        eprintln!("[Stage 5/7] Skipped: no GPU detected");
    } else {
        // Compute max layers at F16 precision using the GPU from the HW profile.
        let max_layers_f16 = hw
            .gpu
            .as_ref()
            .map(|g| search::estimate_max_gpu_layers(g, &model))
            .unwrap_or(0);

        let layer_candidates = gpu_layer_candidates(
            &hw_ctx,
            &model_ctx,
            &best_config.quantization,
            max_layers_f16,
        );
        eprintln!(
            "[Stage 5/7] GPU tuning: max_layers_f16={max_layers_f16}, testing {:?}",
            layer_candidates
        );

        let stage5_candidates: Vec<RuntimeConfig> = layer_candidates
            .iter()
            .map(|&gpu_layers| RuntimeConfig {
                runtime: best_config.runtime.clone(),
                quantization: best_config.quantization.clone(),
                threads: best_config.threads,
                batch_size: best_config.batch_size,
                gpu_layers,
            })
            .collect();

        let mut results = benchmark::run_benchmarks(stage5_candidates, &model, &hw).await;
        results = apply_llamacpp_fallback(results, &model, &hw);
        for reason in collect_failure_reasons(&results) {
            all_failure_reasons.insert(reason);
        }
        if let Some(winner) = select_best(results, &hw, max_latency_ms) {
            eprintln!(
                "[Stage 5/7] Winner: {} gpu_layers ({:.1} tok/s)",
                winner.config.gpu_layers, winner.metrics.tokens_per_sec
            );
            best_config = winner.config.clone();
        } else {
            eprintln!("[Stage 5/7] No valid result — keeping previous best");
        }
    }

    // ── Stage 6: Threading Optimization (≤ 3 benchmark calls) ────────────────
    {
        let thread_cands = thread_candidates(&hw_ctx, best_config.gpu_layers);
        eprintln!(
            "[Stage 6/7] Threading: predicted={}, testing {:?}",
            if best_config.gpu_layers > 16 {
                (hw_ctx.cpu_cores / 2).max(1)
            } else {
                hw_ctx.cpu_cores
            },
            thread_cands
        );

        let stage6_candidates: Vec<RuntimeConfig> = thread_cands
            .iter()
            .map(|&threads| RuntimeConfig {
                runtime: best_config.runtime.clone(),
                quantization: best_config.quantization.clone(),
                threads,
                batch_size: best_config.batch_size,
                gpu_layers: best_config.gpu_layers,
            })
            .collect();

        let mut results = benchmark::run_benchmarks(stage6_candidates, &model, &hw).await;
        results = apply_llamacpp_fallback(results, &model, &hw);
        for reason in collect_failure_reasons(&results) {
            all_failure_reasons.insert(reason);
        }
        if let Some(winner) = select_best(results, &hw, max_latency_ms) {
            eprintln!(
                "[Stage 6/7] Winner: {} threads ({:.1} tok/s)",
                winner.config.threads, winner.metrics.tokens_per_sec
            );
            best_config = winner.config.clone();
        } else {
            eprintln!("[Stage 6/7] No valid result — keeping previous best");
        }
    }

    // ── Stage 7: Final Benchmark (single confirmation run) ───────────────────
    eprintln!(
        "[Stage 7/7] Final benchmark: {:?}/{:?}/{} threads/{} gpu_layers",
        best_config.runtime,
        best_config.quantization,
        best_config.threads,
        best_config.gpu_layers
    );

    let mut final_results =
        benchmark::run_benchmarks(vec![best_config.clone()], &model, &hw).await;
    final_results = apply_llamacpp_fallback(final_results, &model, &hw);
    for reason in collect_failure_reasons(&final_results) {
        all_failure_reasons.insert(reason);
    }

    select_best(final_results, &hw, max_latency_ms).ok_or_else(|| {
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
