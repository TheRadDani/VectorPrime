// crates/llmforge-optimizer/src/estimate.rs
//
// Pure hardware-aware throughput estimator for LlamaCpp/GGUF configs.
// Used as a fallback when llama-cli is not installed so the optimizer can
// still rank quantization and thread-count candidates and return the best
// static configuration for the current hardware.
//
// This module is consumed only by `lib.rs` (the `run_optimization` fallback
// path). It has no I/O, no `unwrap()`, and cannot panic.

use llmforge_core::{BenchmarkResult, HardwareProfile, ModelInfo, QuantizationStrategy, RuntimeConfig, SimdLevel};

use crate::search::bytes_per_param;

/// Estimate a `BenchmarkResult` for a LlamaCpp config without running inference.
///
/// The formula is intentionally simple: correctness of RANKING matters more
/// than accuracy of the absolute numbers. All candidates are estimated with
/// the same formula, so relative ordering is preserved.
///
/// # Parameters
/// - `config`  — the candidate runtime configuration to score
/// - `model`   — model metadata (used for memory estimation)
/// - `hw`      — host hardware profile (CPU, GPU, RAM)
///
/// # Returns
/// A `BenchmarkResult` with plausible `tokens_per_sec`, `latency_ms`, and
/// `peak_memory_mb` values derived from hardware characteristics alone.
pub fn estimate_llamacpp(
    config: &RuntimeConfig,
    model: &ModelInfo,
    hw: &HardwareProfile,
) -> BenchmarkResult {
    let cores = hw.cpu.core_count as f64;

    // Quantization speed multiplier.
    // Q4_K_M is the fastest quantization for llama.cpp; F16 is ~3.6× slower.
    let quant_mult = match config.quantization {
        QuantizationStrategy::Q4_K_M => 1.00,
        QuantizationStrategy::Q4_0   => 0.95,
        QuantizationStrategy::Q8_0   => 0.55,
        QuantizationStrategy::F16    => 0.28,
        QuantizationStrategy::Int8   => 0.55,
        QuantizationStrategy::Int4   => 1.00,
    };

    // SIMD instruction-set multiplier.
    // AVX512 is ~50% faster than baseline AVX for matrix operations.
    let simd_mult = match hw.cpu.simd_level {
        SimdLevel::AVX512 => 1.5,
        SimdLevel::AVX2   => 1.2,
        SimdLevel::AVX    => 1.0,
        SimdLevel::None   => 0.8,
    };

    // Thread efficiency: diminishing returns above physical core count.
    // At thread_count == core_count the effective factor is ~1.0;
    // above that, contention reduces gains.
    let threads = config.threads as f64;
    let thread_eff = (threads / cores).min(1.0) * 0.8 + 0.2;

    // Base throughput: approximately 2 tokens/sec per core for Q4_K_M on AVX.
    let cpu_tps = cores * 2.0 * quant_mult * simd_mult * thread_eff;

    // GPU offload boost: up to 5× when all 33 layers are on GPU.
    // Scales linearly from 1× (0 layers) to 5× (33 layers).
    let gpu_boost = if config.gpu_layers > 0 && hw.gpu.is_some() {
        1.0 + (config.gpu_layers as f64 / 33.0) * 4.0
    } else {
        1.0
    };

    let tokens_per_sec = cpu_tps * gpu_boost;
    let latency_ms = 1_000.0 / tokens_per_sec;

    // Memory estimate from parameter count and bytes-per-param for the chosen
    // quantization. When `param_count` is unknown we report 0 (no pruning).
    let peak_memory_mb = model
        .param_count
        .map(|p| (p as f64 * bytes_per_param(&config.quantization) / 1_000_000.0) as u64)
        .unwrap_or(0);

    BenchmarkResult {
        tokens_per_sec,
        latency_ms,
        peak_memory_mb,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use llmforge_core::{
        CpuInfo, GpuInfo, HardwareProfile, ModelFormat, ModelInfo, RamInfo, RuntimeKind,
    };
    use std::path::PathBuf;

    fn base_hw(cores: u32, simd: SimdLevel) -> HardwareProfile {
        HardwareProfile {
            cpu: CpuInfo {
                core_count: cores,
                brand: "Test CPU".to_string(),
                simd_level: simd,
            },
            gpu: None,
            ram: RamInfo {
                total_mb: 32768,
                available_mb: 16384,
            },
        }
    }

    fn hw_with_gpu(cores: u32) -> HardwareProfile {
        HardwareProfile {
            cpu: CpuInfo {
                core_count: cores,
                brand: "Test CPU".to_string(),
                simd_level: SimdLevel::AVX2,
            },
            gpu: Some(GpuInfo {
                name: "NVIDIA RTX 4090".to_string(),
                vram_mb: 24576,
                compute_capability: Some((8, 9)),
            }),
            ram: RamInfo {
                total_mb: 65536,
                available_mb: 32768,
            },
        }
    }

    fn config(quant: QuantizationStrategy, threads: u32, gpu_layers: u32) -> RuntimeConfig {
        RuntimeConfig {
            runtime: RuntimeKind::LlamaCpp,
            quantization: quant,
            threads,
            batch_size: 512,
            gpu_layers,
        }
    }

    fn model_with_params(param_count: Option<u64>) -> ModelInfo {
        ModelInfo {
            path: PathBuf::from("/tmp/model.gguf"),
            format: ModelFormat::GGUF,
            param_count,
        }
    }

    /// Q4_K_M has a higher quant_mult (1.0) than F16 (0.28), so on identical
    /// hardware and thread config the estimated tps must be higher.
    #[test]
    fn test_q4km_faster_than_f16() {
        let hw = base_hw(8, SimdLevel::AVX2);
        let model = model_with_params(Some(7_000_000_000));

        let q4km = estimate_llamacpp(&config(QuantizationStrategy::Q4_K_M, 8, 0), &model, &hw);
        let f16 = estimate_llamacpp(&config(QuantizationStrategy::F16, 8, 0), &model, &hw);

        assert!(
            q4km.tokens_per_sec > f16.tokens_per_sec,
            "Q4_K_M ({:.2} tps) should be faster than F16 ({:.2} tps)",
            q4km.tokens_per_sec,
            f16.tokens_per_sec,
        );
    }

    /// GPU offload (gpu_layers=33) must yield higher tps than CPU-only
    /// (gpu_layers=0) on the same config when a GPU is present.
    #[test]
    fn test_gpu_boost_increases_tps() {
        let hw = hw_with_gpu(8);
        let model = model_with_params(Some(7_000_000_000));

        let cpu_only = estimate_llamacpp(&config(QuantizationStrategy::Q4_K_M, 8, 0), &model, &hw);
        let full_gpu = estimate_llamacpp(&config(QuantizationStrategy::Q4_K_M, 8, 33), &model, &hw);

        assert!(
            full_gpu.tokens_per_sec > cpu_only.tokens_per_sec,
            "gpu_layers=33 ({:.2} tps) should be faster than gpu_layers=0 ({:.2} tps)",
            full_gpu.tokens_per_sec,
            cpu_only.tokens_per_sec,
        );
    }

    /// On an 8-core machine, using 8 threads should yield higher tps than 4
    /// threads because thread_eff is higher when threads <= cores.
    #[test]
    fn test_more_threads_up_to_cores_increases_tps() {
        let hw = base_hw(8, SimdLevel::AVX2);
        let model = model_with_params(Some(7_000_000_000));

        let four_threads = estimate_llamacpp(&config(QuantizationStrategy::Q4_K_M, 4, 0), &model, &hw);
        let eight_threads = estimate_llamacpp(&config(QuantizationStrategy::Q4_K_M, 8, 0), &model, &hw);

        assert!(
            eight_threads.tokens_per_sec > four_threads.tokens_per_sec,
            "8 threads ({:.2} tps) should be faster than 4 threads ({:.2} tps) on an 8-core machine",
            eight_threads.tokens_per_sec,
            four_threads.tokens_per_sec,
        );
    }

    /// When `param_count` is `None` we cannot estimate memory; peak_memory_mb
    /// must be 0 rather than panicking or fabricating a number.
    #[test]
    fn test_peak_memory_zero_when_no_param_count() {
        let hw = base_hw(8, SimdLevel::AVX2);
        let model = model_with_params(None);

        let result = estimate_llamacpp(&config(QuantizationStrategy::Q4_K_M, 8, 0), &model, &hw);

        assert_eq!(
            result.peak_memory_mb, 0,
            "peak_memory_mb should be 0 when param_count is None"
        );
    }
}
