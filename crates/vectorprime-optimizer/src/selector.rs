use anyhow::Result;
use vectorprime_core::{BenchmarkResult, HardwareProfile, OptimizationResult, RuntimeConfig};

/// Pick the best runtime configuration from the benchmark results.
///
/// Steps:
/// 1. Drop `Err` entries (log them to stderr).
/// 2. Drop configs whose `peak_memory_mb` exceeds 90 % of available RAM.
/// 3. Drop configs whose `latency_ms` exceeds `max_latency_ms` (when `Some`).
/// 4. Sort survivors by `tokens_per_sec` descending.
/// 5. Return the top result as [`OptimizationResult`], or `None` if nothing
///    survives.
pub fn select_best(
    results: Vec<(RuntimeConfig, Result<BenchmarkResult>)>,
    hw: &HardwareProfile,
    max_latency_ms: Option<f64>,
) -> Option<OptimizationResult> {
    let ram_budget_mb = (hw.ram.available_mb as f64 * 0.9) as u64;

    let mut survivors: Vec<(RuntimeConfig, BenchmarkResult)> = results
        .into_iter()
        .filter_map(|(config, result)| match result {
            Ok(metrics) => Some((config, metrics)),
            Err(e) => {
                eprintln!("[vectorprime-optimizer] benchmark error: {e}");
                None
            }
        })
        .filter(|(_, metrics)| metrics.peak_memory_mb <= ram_budget_mb)
        .filter(|(_, metrics)| {
            max_latency_ms
                .map(|limit| metrics.latency_ms <= limit)
                .unwrap_or(true)
        })
        .collect();

    // Sort descending by tokens_per_sec.
    survivors.sort_by(|(_, a), (_, b)| {
        b.tokens_per_sec
            .partial_cmp(&a.tokens_per_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    survivors
        .into_iter()
        .next()
        .map(|(config, metrics)| OptimizationResult { config, metrics })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use vectorprime_core::{
        BenchmarkResult, CpuInfo, HardwareProfile, QuantizationStrategy, RamInfo, RuntimeConfig,
        RuntimeKind, SimdLevel,
    };

    fn hw(available_mb: u64) -> HardwareProfile {
        HardwareProfile {
            cpu: CpuInfo {
                core_count: 8,
                brand: "Test".to_string(),
                simd_level: SimdLevel::AVX2,
            },
            gpu: None,
            ram: RamInfo {
                total_mb: available_mb * 2,
                available_mb,
            },
        }
    }

    fn config(runtime: RuntimeKind) -> RuntimeConfig {
        RuntimeConfig {
            runtime,
            quantization: QuantizationStrategy::Q4_K_M,
            threads: 8,
            batch_size: 512,
            gpu_layers: 0,
        }
    }

    fn ok_result(tps: f64, peak_mb: u64) -> Result<BenchmarkResult> {
        Ok(BenchmarkResult {
            tokens_per_sec: tps,
            latency_ms: 1000.0 / tps,
            peak_memory_mb: peak_mb,
        })
    }

    #[test]
    fn test_select_best_picks_highest_tps() {
        let results = vec![
            (config(RuntimeKind::LlamaCpp), ok_result(50.0, 1024)),
            (config(RuntimeKind::OnnxRuntime), ok_result(120.0, 2048)),
            (config(RuntimeKind::TensorRT), ok_result(80.0, 1536)),
        ];
        let best = select_best(results, &hw(16384), None).expect("should pick a winner");
        assert!(
            (best.metrics.tokens_per_sec - 120.0).abs() < 0.001,
            "expected 120 tps, got {}",
            best.metrics.tokens_per_sec
        );
        assert_eq!(best.config.runtime, RuntimeKind::OnnxRuntime);
    }

    #[test]
    fn test_select_best_filters_oom() {
        // Available RAM = 4096 MB → budget = 3686 MB.
        // peak_memory_mb = 4000 exceeds budget; only the 1024 MB entry survives.
        let results = vec![
            (config(RuntimeKind::LlamaCpp), ok_result(200.0, 4000)), // OOM
            (config(RuntimeKind::OnnxRuntime), ok_result(80.0, 1024)),
        ];
        let best = select_best(results, &hw(4096), None).expect("should pick OnnxRuntime");
        assert_eq!(best.config.runtime, RuntimeKind::OnnxRuntime);
        assert!(
            (best.metrics.tokens_per_sec - 80.0).abs() < 0.001,
            "tps={}",
            best.metrics.tokens_per_sec
        );
    }

    #[test]
    fn test_select_best_empty_input_returns_none() {
        let result = select_best(vec![], &hw(16384), None);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_best_all_errors_returns_none() {
        let results = vec![
            (
                config(RuntimeKind::LlamaCpp),
                Err(anyhow::anyhow!("failed")),
            ),
            (
                config(RuntimeKind::TensorRT),
                Err(anyhow::anyhow!("also failed")),
            ),
        ];
        let result = select_best(results, &hw(16384), None);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_best_all_oom_returns_none() {
        // Budget = 900 MB; both results exceed it.
        let results = vec![
            (config(RuntimeKind::LlamaCpp), ok_result(50.0, 1000)),
            (config(RuntimeKind::OnnxRuntime), ok_result(80.0, 2000)),
        ];
        let result = select_best(results, &hw(1000), None);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_best_latency_filter_excludes_slow_configs() {
        // LlamaCpp: 50 tps → latency = 20 ms.  OnnxRuntime: 10 tps → 100 ms.
        // With a 50 ms budget only LlamaCpp survives even though it has lower tps.
        let results = vec![
            (config(RuntimeKind::LlamaCpp), ok_result(50.0, 1024)),
            (config(RuntimeKind::OnnxRuntime), ok_result(10.0, 1024)),
        ];
        let best = select_best(results, &hw(16384), Some(50.0)).expect("LlamaCpp should survive");
        assert_eq!(best.config.runtime, RuntimeKind::LlamaCpp);
    }

    #[test]
    fn test_select_best_latency_filter_all_too_slow_returns_none() {
        // Both latencies exceed the 5 ms limit.
        let results = vec![
            (config(RuntimeKind::LlamaCpp), ok_result(50.0, 1024)), // latency = 20 ms
            (config(RuntimeKind::OnnxRuntime), ok_result(10.0, 1024)), // latency = 100 ms
        ];
        let result = select_best(results, &hw(16384), Some(5.0));
        assert!(result.is_none());
    }

    #[test]
    fn test_select_best_no_latency_constraint_returns_highest_tps() {
        // No latency constraint → same as original behaviour.
        let results = vec![
            (config(RuntimeKind::LlamaCpp), ok_result(50.0, 1024)),
            (config(RuntimeKind::OnnxRuntime), ok_result(200.0, 1024)),
        ];
        let best = select_best(results, &hw(16384), None).expect("should pick OnnxRuntime");
        assert_eq!(best.config.runtime, RuntimeKind::OnnxRuntime);
    }
}
