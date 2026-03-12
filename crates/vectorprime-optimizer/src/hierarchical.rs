// crates/vectorprime-optimizer/src/hierarchical.rs
//
// Two-phase hierarchical search strategy for the VectorPrime optimizer.
//
// Phase 1 (Runtime Elimination): benchmarks one config per available runtime
// using a short probe prompt and retains only the top-N runtimes by throughput.
//
// Phase 2 (Full Search): generates the full candidate space but restricts it to
// the surviving runtimes from Phase 1, then selects the best result.
//
// Used by: lib.rs (`run_optimization_hierarchical`), and exposed to Python via
// vectorprime-bindings.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;

use vectorprime_core::{
    HardwareProfile, ModelInfo, OptimizationResult, QuantizationStrategy, RuntimeConfig,
    RuntimeKind,
};
use vectorprime_runtime::{dispatch, AdapterRegistry};

use crate::benchmark::run_benchmarks;
use crate::search::generate_candidates;
use crate::selector::select_best;

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the two-phase hierarchical search.
///
/// Controls how many runtimes survive Phase 1 and the parameters used for the
/// short Phase 1 probe benchmark.
#[derive(Debug, Clone)]
pub struct HierarchicalSearchConfig {
    /// Number of top runtimes to retain after Phase 1 ranking.
    ///
    /// For example, `top_n_runtimes = 2` keeps the two fastest runtimes by
    /// Phase 1 throughput and discards the rest before the full search.
    pub top_n_runtimes: usize,

    /// Short prompt used in Phase 1 probe benchmarks.
    ///
    /// Should be shorter than the full benchmark prompt so each probe
    /// completes quickly. The result is used only for relative runtime ranking,
    /// not as the final performance measurement.
    pub phase1_prompt: String,

    /// Batch size to use in Phase 1 probe configs.
    ///
    /// Smaller values reduce the resource footprint of the probe run.
    /// Defaults to 1 so the probe measures the minimal overhead of each runtime.
    pub phase1_batch_size: u32,
}

impl Default for HierarchicalSearchConfig {
    fn default() -> Self {
        Self {
            top_n_runtimes: 2,
            phase1_prompt: "Hello".to_string(),
            phase1_batch_size: 1,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Phase 1: Runtime Elimination
// ──────────────────────────────────────────────────────────────────────────────

/// Probe a single runtime with a minimal configuration and return its throughput.
///
/// Returns `None` when the runtime binary is not installed or any other error
/// occurs — the caller should skip that runtime gracefully.
fn probe_runtime(
    runtime: RuntimeKind,
    model: &ModelInfo,
    cfg: &HierarchicalSearchConfig,
) -> Option<f64> {
    // Construct a minimal probe config: default quantization, minimal threads,
    // no GPU offload (Phase 1 is only for relative runtime ranking).
    let probe_config = RuntimeConfig {
        runtime: runtime.clone(),
        quantization: QuantizationStrategy::Q4_K_M,
        threads: 1,
        batch_size: cfg.phase1_batch_size,
        gpu_layers: 0,
    };

    let mut registry = AdapterRegistry::new();
    match dispatch(&mut registry, &probe_config, model, &cfg.phase1_prompt) {
        Ok(result) => Some(result.tokens_per_sec),
        Err(e) => {
            let reason = e.to_string();
            eprintln!(
                "[vectorprime-optimizer] Phase 1: {:?} → skipped ({})",
                runtime, reason
            );
            None
        }
    }
}

/// Phase 1: benchmark each candidate runtime once and return a ranked list.
///
/// Runtimes whose binary is not installed (`RuntimeError::NotInstalled`) or
/// that fail for any other reason are silently dropped from the ranking.
///
/// Returns a `Vec<(RuntimeKind, f64)>` sorted by `tokens_per_sec` descending.
/// If all runtimes fail, the returned Vec is empty — the caller must handle
/// this case and fall back to the full unfiltered search.
async fn run_phase1(
    runtimes: &[RuntimeKind],
    model: Arc<ModelInfo>,
    cfg: Arc<HierarchicalSearchConfig>,
) -> Vec<(RuntimeKind, f64)> {
    // Probe each runtime in parallel using spawn_blocking (same pattern as
    // benchmark::run_benchmarks, since AdapterRegistry is not Send).
    let handles: Vec<_> = runtimes
        .iter()
        .cloned()
        .map(|runtime| {
            let model = Arc::clone(&model);
            let cfg = Arc::clone(&cfg);
            tokio::task::spawn_blocking(move || {
                let tps = probe_runtime(runtime.clone(), &model, &cfg);
                (runtime, tps)
            })
        })
        .collect();

    let mut ranked: Vec<(RuntimeKind, f64)> = Vec::new();
    for handle in handles {
        match handle.await {
            Ok((runtime, Some(tps))) => {
                eprintln!(
                    "[vectorprime-optimizer] Phase 1: {:?} → {:.1} tok/s",
                    runtime, tps
                );
                ranked.push((runtime, tps));
            }
            Ok((runtime, None)) => {
                // Already logged inside probe_runtime; nothing to do here.
                let _ = runtime; // suppress unused warning
            }
            Err(join_err) => {
                eprintln!(
                    "[vectorprime-optimizer] Phase 1: probe task panicked: {}",
                    join_err
                );
            }
        }
    }

    // Sort descending by tokens_per_sec.
    ranked.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

// ──────────────────────────────────────────────────────────────────────────────
// Public entry point
// ──────────────────────────────────────────────────────────────────────────────

/// Run the two-phase hierarchical optimization and return the best configuration.
///
/// **Phase 1 — Runtime Elimination**
///
/// Each runtime kind present in the generated candidate space is probed once
/// with a minimal configuration. Runtimes are ranked by their Phase 1
/// throughput (tokens/sec) and only the top `cfg.top_n_runtimes` are retained.
/// Runtimes that return `RuntimeError::NotInstalled` or fail for any other
/// reason are silently eliminated.
///
/// **Phase 2 — Full Search on Top Runtimes**
///
/// The same `generate_candidates` / `run_benchmarks` / `select_best` pipeline
/// as [`run_optimization`](crate::run_optimization) is used, but the candidate
/// set is filtered to the runtimes that survived Phase 1.
///
/// If Phase 1 eliminates *all* runtimes (e.g. none of the required binaries are
/// installed), the function falls back to the full unfiltered candidate space
/// so the user still receives the best static estimate available.
///
/// # Parameters
/// - `model`          — metadata about the model file under optimization
/// - `hw`             — snapshot of host hardware (CPU, GPU, RAM)
/// - `max_latency_ms` — optional latency cap; configs that exceed this are excluded
/// - `cfg`            — hierarchical search parameters (default: [`HierarchicalSearchConfig::default`])
///
/// # Returns
/// The best [`OptimizationResult`] found, or an error when no valid
/// configuration survives the full benchmark and selection pipeline.
pub async fn run_optimization_hierarchical(
    model: ModelInfo,
    hw: HardwareProfile,
    max_latency_ms: Option<f64>,
    cfg: HierarchicalSearchConfig,
) -> Result<OptimizationResult> {
    // Generate the full candidate space up front so we know which runtimes are
    // relevant for this hardware / model combination.
    let all_candidates = generate_candidates(&hw, &model);

    if all_candidates.is_empty() {
        return Err(anyhow::anyhow!(
            "no candidate configurations generated for this hardware / model combination"
        ));
    }

    // Collect the unique runtime kinds present in the candidate space.
    let candidate_runtimes: Vec<RuntimeKind> = {
        let mut seen = HashSet::new();
        all_candidates
            .iter()
            .filter_map(|c| {
                if seen.insert(c.runtime.clone()) {
                    Some(c.runtime.clone())
                } else {
                    None
                }
            })
            .collect()
    };

    let model_arc = Arc::new(model.clone());
    let cfg_arc = Arc::new(cfg.clone());

    // ── Phase 1: probe each runtime ────────────────────────────────────────
    eprintln!(
        "[vectorprime-optimizer] Phase 1: probing {} runtimes…",
        candidate_runtimes.len()
    );

    let phase1_ranked = run_phase1(&candidate_runtimes, Arc::clone(&model_arc), Arc::clone(&cfg_arc)).await;

    // Determine which runtimes advance to Phase 2.
    let top_runtimes: HashSet<RuntimeKind> = if phase1_ranked.is_empty() {
        // All probes failed — fall back to the full candidate space.
        eprintln!(
            "[vectorprime-optimizer] Phase 1: all runtime probes failed; \
             falling back to full candidate space"
        );
        candidate_runtimes.into_iter().collect()
    } else {
        let keep_n = cfg.top_n_runtimes.max(1);
        let top: HashSet<_> = phase1_ranked
            .iter()
            .take(keep_n)
            .map(|(rt, _)| rt.clone())
            .collect();

        // Log eliminated runtimes.
        for (rt, tps) in &phase1_ranked {
            if !top.contains(rt) {
                eprintln!(
                    "[vectorprime-optimizer] Phase 1: {:?} → {:.1} tok/s, eliminated",
                    rt, tps
                );
            }
        }
        top
    };

    eprintln!(
        "[vectorprime-optimizer] Phase 2: searching across {} top runtime(s): {:?}",
        top_runtimes.len(),
        top_runtimes
    );

    // ── Phase 2: full search restricted to top runtimes ───────────────────
    let phase2_candidates: Vec<_> = all_candidates
        .into_iter()
        .filter(|c| top_runtimes.contains(&c.runtime))
        .collect();

    if phase2_candidates.is_empty() {
        return Err(anyhow::anyhow!(
            "no candidates remain after Phase 1 runtime elimination"
        ));
    }

    let results = run_benchmarks(phase2_candidates, &model, &hw).await;

    // Collect unique failure reasons for a better error message.
    let failure_reasons: Vec<String> = {
        let mut seen = std::collections::BTreeSet::new();
        for (_, outcome) in &results {
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
    };

    select_best(results, &hw, max_latency_ms).ok_or_else(|| {
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
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use vectorprime_core::{
        BenchmarkResult, CpuInfo, HardwareProfile, ModelFormat, ModelInfo,
        QuantizationStrategy, RamInfo, RuntimeConfig, RuntimeKind, SimdLevel,
    };
    use std::path::PathBuf;

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn cpu_only_hw(cores: u32) -> HardwareProfile {
        HardwareProfile {
            cpu: CpuInfo {
                core_count: cores,
                brand: "Test CPU".to_string(),
                simd_level: SimdLevel::AVX2,
            },
            gpu: None,
            ram: RamInfo {
                total_mb: 32768,
                available_mb: 16384,
            },
        }
    }

    fn gguf_model() -> ModelInfo {
        ModelInfo {
            path: PathBuf::from("/tmp/model.gguf"),
            format: ModelFormat::GGUF,
            param_count: Some(7_000_000_000),
            hidden_size: None,
            attention_head_count: None,
            attention_head_count_kv: None,
            feed_forward_length: None,
            kv_cache_size_mb: None,
            memory_footprint_mb: None,
            flops_per_token: None,
        }
    }

    fn bench_ok(tps: f64) -> anyhow::Result<BenchmarkResult> {
        Ok(BenchmarkResult {
            tokens_per_sec: tps,
            latency_ms: 1_000.0 / tps,
            peak_memory_mb: 512,
        })
    }

    fn rt_config(runtime: RuntimeKind) -> RuntimeConfig {
        RuntimeConfig {
            runtime,
            quantization: QuantizationStrategy::Q4_K_M,
            threads: 4,
            batch_size: 512,
            gpu_layers: 0,
        }
    }

    // ── Phase 1 ranking ───────────────────────────────────────────────────────

    /// Phase 1 ranking: the list returned by run_phase1 must be sorted
    /// descending by tokens_per_sec.  We verify this by checking that any
    /// runtimes that successfully returned results are in order.
    ///
    /// We simulate the ranking by replicating the sorting logic directly since
    /// the runtimes on this test host are not installed.
    #[test]
    fn test_phase1_ranking_sort_order() {
        // Simulate phase1 result: build a vec of (runtime, tps) pairs and
        // verify sort descending works correctly.
        let mut ranked: Vec<(RuntimeKind, f64)> = vec![
            (RuntimeKind::OnnxRuntime, 62.0),
            (RuntimeKind::LlamaCpp, 120.0),
            (RuntimeKind::TensorRT, 210.0),
        ];
        ranked.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        assert_eq!(ranked[0].0, RuntimeKind::TensorRT);
        assert!((ranked[0].1 - 210.0).abs() < f64::EPSILON);
        assert_eq!(ranked[1].0, RuntimeKind::LlamaCpp);
        assert!((ranked[1].1 - 120.0).abs() < f64::EPSILON);
        assert_eq!(ranked[2].0, RuntimeKind::OnnxRuntime);
        assert!((ranked[2].1 - 62.0).abs() < f64::EPSILON);
    }

    /// Phase 2 filtering: only top-N runtimes from Phase 1 should appear in
    /// the Phase 2 candidate set.
    #[test]
    fn test_phase2_filters_to_top_n_runtimes() {
        let hw = cpu_only_hw(8);
        let model = gguf_model();
        let all_candidates = generate_candidates(&hw, &model);

        // Simulate Phase 1 picking only LlamaCpp as the top-1 runtime.
        let top_runtimes: HashSet<RuntimeKind> = [RuntimeKind::LlamaCpp].into_iter().collect();

        let phase2_candidates: Vec<_> = all_candidates
            .iter()
            .filter(|c| top_runtimes.contains(&c.runtime))
            .collect();

        // Every Phase 2 candidate must be LlamaCpp.
        for candidate in &phase2_candidates {
            assert_eq!(
                candidate.runtime,
                RuntimeKind::LlamaCpp,
                "Phase 2 candidate {:?} should have been filtered out",
                candidate.runtime
            );
        }

        // Phase 2 must still have candidates (not empty).
        assert!(
            !phase2_candidates.is_empty(),
            "Phase 2 must retain at least one candidate for LlamaCpp"
        );
    }

    /// When Phase 1 selects top-2 runtimes, Phase 2 must include candidates
    /// from both and exclude any third runtime.
    #[test]
    fn test_phase2_retains_top2_runtimes() {
        let hw = cpu_only_hw(8);
        let model = gguf_model();
        let all_candidates = generate_candidates(&hw, &model);

        // GGUF model generates LlamaCpp + Ollama candidates. Simulate keeping both.
        let top_runtimes: HashSet<RuntimeKind> =
            [RuntimeKind::LlamaCpp, RuntimeKind::Ollama].into_iter().collect();

        let phase2_candidates: Vec<_> = all_candidates
            .iter()
            .filter(|c| top_runtimes.contains(&c.runtime))
            .collect();

        let has_llamacpp = phase2_candidates
            .iter()
            .any(|c| c.runtime == RuntimeKind::LlamaCpp);
        let has_ollama = phase2_candidates
            .iter()
            .any(|c| c.runtime == RuntimeKind::Ollama);

        assert!(has_llamacpp, "Phase 2 should include LlamaCpp candidates");
        assert!(has_ollama, "Phase 2 should include Ollama candidates");

        // No other runtimes should appear.
        for c in &phase2_candidates {
            assert!(
                c.runtime == RuntimeKind::LlamaCpp || c.runtime == RuntimeKind::Ollama,
                "unexpected runtime {:?} in Phase 2 candidates",
                c.runtime
            );
        }
    }

    /// NotInstalled runtimes (None from probe_runtime) must be excluded from
    /// the ranked list so they never advance to Phase 2.
    #[test]
    fn test_not_installed_runtimes_excluded_from_ranking() {
        // Simulate a phase1 result where one runtime returned None (not installed).
        // The run_phase1 function only pushes runtimes with Some(tps) into ranked.
        // We test the filtering invariant directly.
        let probed: Vec<(RuntimeKind, Option<f64>)> = vec![
            (RuntimeKind::LlamaCpp, Some(100.0)),
            (RuntimeKind::OnnxRuntime, None), // not installed
            (RuntimeKind::TensorRT, Some(200.0)),
        ];

        let mut ranked: Vec<(RuntimeKind, f64)> = probed
            .into_iter()
            .filter_map(|(rt, tps)| tps.map(|t| (rt, t)))
            .collect();
        ranked.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // OnnxRuntime must not be in ranked.
        assert!(
            ranked.iter().all(|(rt, _)| *rt != RuntimeKind::OnnxRuntime),
            "NotInstalled runtime must be excluded from Phase 1 ranking"
        );
        assert_eq!(ranked.len(), 2, "only the 2 installed runtimes should remain");
        assert_eq!(ranked[0].0, RuntimeKind::TensorRT, "TensorRT should rank first (200 tps)");
        assert_eq!(ranked[1].0, RuntimeKind::LlamaCpp, "LlamaCpp should rank second (100 tps)");
    }

    /// HierarchicalSearchConfig defaults must be sane.
    #[test]
    fn test_hierarchical_config_defaults() {
        let cfg = HierarchicalSearchConfig::default();
        assert_eq!(cfg.top_n_runtimes, 2, "default top_n_runtimes must be 2");
        assert!(!cfg.phase1_prompt.is_empty(), "default phase1_prompt must not be empty");
        assert_eq!(cfg.phase1_batch_size, 1, "default phase1_batch_size must be 1");
    }

    /// select_best used in Phase 2 must pick the highest-tps result from the
    /// restricted candidate set (same invariant as the original optimizer).
    #[test]
    fn test_phase2_select_best_picks_highest_tps() {
        let hw = cpu_only_hw(8);

        let results = vec![
            (rt_config(RuntimeKind::LlamaCpp), bench_ok(80.0)),
            (rt_config(RuntimeKind::Ollama), bench_ok(140.0)),
        ];

        let best = select_best(results, &hw, None).expect("should pick a winner");
        assert_eq!(best.config.runtime, RuntimeKind::Ollama);
        assert!((best.metrics.tokens_per_sec - 140.0).abs() < f64::EPSILON);
    }
}
