// Location: crates/vectorprime-optimizer/src/search.rs
//
// Generates the candidate configuration space for the benchmark loop.
// Called by the optimizer engine to produce RuntimeConfig candidates that
// are plausible on the detected hardware, then benchmarked in parallel.

use vectorprime_core::{
    GpuInfo, GpuVendor, HardwareProfile, ModelFormat, ModelInfo, QuantizationStrategy,
    RuntimeConfig, RuntimeKind,
};

/// Return the storage cost in bytes per model parameter for a given
/// quantization strategy.
pub fn bytes_per_param(q: &QuantizationStrategy) -> f64 {
    match q {
        QuantizationStrategy::F16 => 2.0,
        QuantizationStrategy::Q8_0 | QuantizationStrategy::Int8 => 1.0,
        QuantizationStrategy::Q4_K_M | QuantizationStrategy::Q4_0 | QuantizationStrategy::Int4 => {
            0.5
        }
    }
}

/// Estimate VRAM usage in MB for a given model + quantization.
fn estimate_vram_mb(model: &ModelInfo, q: &QuantizationStrategy) -> Option<f64> {
    let params = model.param_count? as f64;
    Some(params * bytes_per_param(q) / 1_000_000.0)
}

/// Return `true` when the detected GPU is from NVIDIA.
///
/// TRT eligibility is based purely on vendor identity — we let `trtexec` itself
/// report whether the specific GPU supports the requested precision at runtime.
/// This removes the compute-capability threshold (>= 7.0) that was hardcoded
/// here and prevents us from unnecessarily excluding newer GPUs or
/// misclassifying vendor-unknown devices.
fn is_nvidia_gpu(hw: &HardwareProfile) -> bool {
    hw.gpu
        .as_ref()
        .map(|g| g.vendor == GpuVendor::Nvidia)
        .unwrap_or(false)
}

/// Estimate the maximum number of transformer layers that fit in GPU VRAM.
///
/// Uses F16 precision as the worst-case storage cost to produce a conservative
/// upper bound. When `model.param_count` is unknown, falls back to
/// VRAM-capacity tiers that represent practical offload limits.
///
/// The returned value is the *maximum* layers to consider; the caller is
/// responsible for generating evenly-spaced steps from 0 to this value.
pub fn estimate_max_gpu_layers(gpu: &GpuInfo, model: &ModelInfo) -> u32 {
    if let Some(params) = model.param_count {
        // F16 worst-case: 2 bytes per parameter.
        let model_size_mb = params as f64 * 2.0 / 1_048_576.0;
        let vram_mb = gpu.vram_mb as f64;

        if model_size_mb <= 0.0 {
            return 0;
        }

        // Fraction of the model that fits in VRAM (capped at 1.0 for full
        // offload). Use a reference of 32 transformer layers to convert the
        // fraction into a layer count — this is an approximation suitable for
        // pruning the search space; real layer counts vary by architecture.
        let fit_fraction = (vram_mb / model_size_mb).min(1.0);
        const REFERENCE_LAYERS: f64 = 32.0;
        (fit_fraction * REFERENCE_LAYERS).floor() as u32
    } else {
        // Param count unknown: use VRAM tiers as capacity estimates.
        // These are not architecture constants — they represent practical
        // limits on how many layers can be offloaded given the VRAM budget.
        match gpu.vram_mb {
            0..=4095 => 16,
            4096..=12287 => 32,
            _ => 48,
        }
    }
}

/// Classifies a model's inference bottleneck based on architecture metrics.
///
/// - `MemoryBound`: large KV cache or narrow FFN relative to hidden size
///   (attention dominates). Favours aggressive quantization.
/// - `ComputeBound`: wide FFN (FFN >> 8× hidden, e.g. Mixtral MoE). Favours
///   precision-preserving formats when tensor cores are available.
/// - `Balanced`: in between; apply default heuristics.
enum WorkloadType {
    MemoryBound,
    ComputeBound,
    Balanced,
}

/// Classify the workload type from model architecture metrics.
///
/// Returns `Balanced` when the required fields are absent.
fn classify_workload(model: &ModelInfo) -> WorkloadType {
    match (model.feed_forward_length, model.hidden_size) {
        (Some(ffn), Some(hidden)) if hidden > 0 => {
            let ffn_ratio = ffn / hidden as u64;
            if ffn_ratio >= 8 {
                WorkloadType::ComputeBound // FFN >> 8× hidden (MoE-style)
            } else if ffn_ratio <= 2 {
                WorkloadType::MemoryBound // attention dominates
            } else {
                WorkloadType::Balanced
            }
        }
        _ => WorkloadType::Balanced,
    }
}

/// Generate the cross-product of (runtime × quantization × threads × gpu_layers)
/// pruned to configs that are plausible on the given hardware.
pub fn generate_candidates(hw: &HardwareProfile, model: &ModelInfo) -> Vec<RuntimeConfig> {
    let cores = hw.cpu.core_count;

    // Thread counts: [cores/2, cores, cores*2] clamped to [1, 64].
    let thread_counts: Vec<u32> = [cores / 2, cores, cores * 2]
        .iter()
        .map(|&t| t.clamp(1, 64))
        .collect::<std::collections::BTreeSet<_>>() // deduplicate
        .into_iter()
        .collect();

    // GPU layer options: VRAM-proportional steps from 0% to 100% offload.
    // When no GPU is present, only cpu-only (0 layers) is considered.
    let gpu_layer_options: Vec<u32> = if let Some(gpu) = &hw.gpu {
        let max_layers = estimate_max_gpu_layers(gpu, model);
        if max_layers == 0 {
            vec![0]
        } else {
            // 5 evenly-spaced steps: 0, 25%, 50%, 75%, 100%
            let steps = 4u32;
            (0..=steps)
                .map(|i| (max_layers * i / steps).min(max_layers))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect()
        }
    } else {
        vec![0]
    };

    // Classify the workload to influence candidate ordering.
    // All quant/runtime combos are still generated; ordering only affects
    // which candidates are evaluated first (the scorer picks the final winner).
    let workload = classify_workload(model);

    // Per-format runtime + quantization pairings, ordered by workload type.
    //
    // Ollama supports GGUF models on any hardware (wraps llama.cpp internally).
    // vLLM supports ONNX/HuggingFace models and requires a GPU for meaningful
    // performance; it prefers F16/Q8_0 precision.
    let combos: Vec<(RuntimeKind, QuantizationStrategy)> = match model.format {
        ModelFormat::GGUF => {
            let mut quants = vec![
                QuantizationStrategy::Q4_K_M,
                QuantizationStrategy::Q4_0,
                QuantizationStrategy::Q8_0,
                QuantizationStrategy::F16,
            ];
            // Reorder by workload: memory-bound prefers aggressive quants (already
            // first). Compute-bound prefers precision-preserving formats.
            match workload {
                WorkloadType::ComputeBound => {
                    // Move F16 and Q8_0 to the front so the scorer sees them first.
                    quants.sort_by_key(|q| match q {
                        QuantizationStrategy::F16 => 0,
                        QuantizationStrategy::Q8_0 => 1,
                        QuantizationStrategy::Q4_K_M => 2,
                        QuantizationStrategy::Q4_0 => 3,
                        _ => 4,
                    });
                }
                WorkloadType::MemoryBound | WorkloadType::Balanced => {
                    // Default ordering: aggressive quants first (Q4_K_M already at front).
                }
            }

            // Both LlamaCpp and Ollama support GGUF on any hardware.
            // Ollama manages its own llama.cpp backend; interleave the same
            // quantization options under both runtimes so the benchmarker can
            // pick the faster of the two.
            let mut combos: Vec<(RuntimeKind, QuantizationStrategy)> = Vec::new();
            for q in &quants {
                combos.push((RuntimeKind::LlamaCpp, q.clone()));
                combos.push((RuntimeKind::Ollama, q.clone()));
            }
            combos
        }

        ModelFormat::ONNX => {
            let mut v: Vec<(RuntimeKind, QuantizationStrategy)> = vec![
                (RuntimeKind::OnnxRuntime, QuantizationStrategy::F16),
                (RuntimeKind::OnnxRuntime, QuantizationStrategy::Int8),
            ];
            // TRT is NVIDIA-only; trtexec will handle GPU-specific compatibility.
            if is_nvidia_gpu(hw) {
                v.push((RuntimeKind::TensorRT, QuantizationStrategy::F16));
                v.push((RuntimeKind::TensorRT, QuantizationStrategy::Int8));
            }
            // vLLM is GPU-accelerated; add it when any GPU is present.
            // It only meaningfully competes on F16/Q8_0 — sub-8-bit quantizations
            // fall back to float16 inside vLLM anyway (see vllm.rs).
            if hw.gpu.is_some() {
                v.push((RuntimeKind::Vllm, QuantizationStrategy::F16));
                v.push((RuntimeKind::Vllm, QuantizationStrategy::Q8_0));
            }
            // For memory-bound ONNX workloads, move Int8 ahead of F16.
            if matches!(workload, WorkloadType::MemoryBound) {
                v.sort_by_key(|(rt, q)| match (rt, q) {
                    (_, QuantizationStrategy::Int8 | QuantizationStrategy::Int4) => 0,
                    _ => 1,
                });
            }
            v
        }
    };

    // KV-cache-aware batch sizing: reduce batch when KV cache pressure is high.
    // A full-context KV cache consuming > 30% of VRAM leaves little room for
    // activations, so we halve the batch to reduce runtime memory pressure.
    let base_batch_size: u32 = 512;
    let batch_size = if let (Some(kv_mb), Some(vram_mb)) = (
        model.kv_cache_size_mb,
        hw.gpu.as_ref().map(|g| g.vram_mb as f64),
    ) {
        if kv_mb > vram_mb * 0.3 {
            (base_batch_size / 2).max(128)
        } else {
            base_batch_size
        }
    } else {
        base_batch_size
    };

    let vram_budget_mb = hw.gpu.as_ref().map(|g| g.vram_mb as f64 * 0.9);

    let mut candidates = Vec::new();
    for (runtime, quant) in combos {
        // Prune: skip if we know the model won't fit in VRAM.
        if let (Some(budget), Some(vram)) = (vram_budget_mb, estimate_vram_mb(model, &quant)) {
            if vram > budget {
                continue;
            }
        }

        for &threads in &thread_counts {
            for &gpu_layers in &gpu_layer_options {
                candidates.push(RuntimeConfig {
                    runtime: runtime.clone(),
                    quantization: quant.clone(),
                    threads,
                    batch_size,
                    gpu_layers,
                });
            }
        }
    }

    candidates
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use vectorprime_core::{
        CpuInfo, GpuInfo, GpuVendor, HardwareProfile, ModelFormat, ModelInfo, RamInfo, SimdLevel,
    };
    use std::path::PathBuf;

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

    fn hw_with_nvidia(cores: u32, vram_mb: u64, compute_major: u32) -> HardwareProfile {
        HardwareProfile {
            cpu: CpuInfo {
                core_count: cores,
                brand: "Test CPU".to_string(),
                simd_level: SimdLevel::AVX2,
            },
            gpu: Some(GpuInfo {
                name: "NVIDIA Test GPU".to_string(),
                vram_mb,
                compute_capability: Some((compute_major, 0)),
                vendor: GpuVendor::Nvidia,
            }),
            ram: RamInfo {
                total_mb: 65536,
                available_mb: 32768,
            },
        }
    }

    fn hw_with_amd(cores: u32, vram_mb: u64) -> HardwareProfile {
        HardwareProfile {
            cpu: CpuInfo {
                core_count: cores,
                brand: "Test CPU".to_string(),
                simd_level: SimdLevel::AVX2,
            },
            gpu: Some(GpuInfo {
                name: "AMD Radeon RX 7900 XTX".to_string(),
                vram_mb,
                compute_capability: None,
                vendor: GpuVendor::Amd,
            }),
            ram: RamInfo {
                total_mb: 65536,
                available_mb: 32768,
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

    fn onnx_model() -> ModelInfo {
        ModelInfo {
            path: PathBuf::from("/tmp/model.onnx"),
            format: ModelFormat::ONNX,
            param_count: Some(1_000_000_000),
            hidden_size: None,
            attention_head_count: None,
            attention_head_count_kv: None,
            feed_forward_length: None,
            kv_cache_size_mb: None,
            memory_footprint_mb: None,
            flops_per_token: None,
        }
    }

    #[test]
    fn test_gguf_candidates_only_llamacpp_and_ollama() {
        // GGUF format should produce candidates for LlamaCpp and Ollama only —
        // not OnnxRuntime, TensorRT, or Vllm.
        let hw = cpu_only_hw(8);
        let model = gguf_model();
        let candidates = generate_candidates(&hw, &model);
        assert!(!candidates.is_empty());
        for c in &candidates {
            assert!(
                c.runtime == RuntimeKind::LlamaCpp || c.runtime == RuntimeKind::Ollama,
                "unexpected runtime {:?} in GGUF candidates (expected LlamaCpp or Ollama)",
                c.runtime
            );
        }
    }

    #[test]
    fn test_onnx_no_gpu_excludes_tensorrt() {
        let hw = cpu_only_hw(8);
        let model = onnx_model();
        let candidates = generate_candidates(&hw, &model);
        assert!(!candidates.is_empty());
        for c in &candidates {
            assert_ne!(
                c.runtime,
                RuntimeKind::TensorRT,
                "TensorRT should not appear without GPU"
            );
        }
    }

    #[test]
    fn test_onnx_with_nvidia_includes_tensorrt() {
        let hw = hw_with_nvidia(8, 24576, 8); // compute cap 8.x
        let model = onnx_model();
        let candidates = generate_candidates(&hw, &model);
        assert!(
            candidates
                .iter()
                .any(|c| c.runtime == RuntimeKind::TensorRT),
            "expected TensorRT candidates with NVIDIA GPU"
        );
    }

    /// Previously this test expected TRT to be EXCLUDED for compute cap < 7.
    /// After the refactor, TRT inclusion is based on vendor (Nvidia) alone —
    /// trtexec handles GPU-specific compatibility at runtime. Any NVIDIA GPU
    /// produces TRT candidates.
    #[test]
    fn test_onnx_with_nvidia_always_includes_tensorrt_candidates() {
        let hw = hw_with_nvidia(8, 8192, 6); // compute cap 6.x (old Pascal GPU)
        let model = onnx_model();
        let candidates = generate_candidates(&hw, &model);
        assert!(
            candidates
                .iter()
                .any(|c| c.runtime == RuntimeKind::TensorRT),
            "TRT should be a candidate for any NVIDIA GPU; trtexec handles compat"
        );
    }

    /// Non-NVIDIA GPUs (AMD, Apple, Unknown) must never produce TRT candidates.
    #[test]
    fn test_onnx_non_nvidia_excludes_tensorrt() {
        let hw = hw_with_amd(8, 24576);
        let model = onnx_model();
        let candidates = generate_candidates(&hw, &model);
        for c in &candidates {
            assert_ne!(
                c.runtime,
                RuntimeKind::TensorRT,
                "TensorRT should not appear for AMD GPU"
            );
        }
    }

    #[test]
    fn test_bytes_per_param_all_variants() {
        use QuantizationStrategy::*;
        for q in [F16, Q8_0, Q4_K_M, Q4_0, Int8, Int4] {
            assert!(
                bytes_per_param(&q) > 0.0,
                "bytes_per_param({q:?}) must be > 0"
            );
        }
    }

    #[test]
    fn test_gpu_layer_options_cpu_only() {
        let hw = cpu_only_hw(4);
        let model = gguf_model();
        let candidates = generate_candidates(&hw, &model);
        for c in &candidates {
            assert_eq!(c.gpu_layers, 0, "cpu-only: gpu_layers must be 0");
        }
    }

    #[test]
    fn test_gpu_layer_options_with_gpu() {
        let hw = hw_with_nvidia(8, 24576, 8);
        let model = gguf_model();
        let candidates = generate_candidates(&hw, &model);
        let has_nonzero = candidates.iter().any(|c| c.gpu_layers > 0);
        assert!(
            has_nonzero,
            "expected non-zero gpu_layers when GPU is present"
        );
    }

    #[test]
    fn test_estimate_max_gpu_layers_with_param_count() {
        // 7B model at F16 = ~13.4 GB; 24 GB VRAM fits ~1.79x → all 32 ref layers
        let gpu = GpuInfo {
            name: "Test GPU".to_string(),
            vram_mb: 24576,
            compute_capability: None,
            vendor: GpuVendor::Nvidia,
        };
        let model = ModelInfo {
            path: PathBuf::from("/tmp/m.gguf"),
            format: ModelFormat::GGUF,
            param_count: Some(7_000_000_000),
            hidden_size: None,
            attention_head_count: None,
            attention_head_count_kv: None,
            feed_forward_length: None,
            kv_cache_size_mb: None,
            memory_footprint_mb: None,
            flops_per_token: None,
        };
        let max = estimate_max_gpu_layers(&gpu, &model);
        assert!(max > 0, "should produce non-zero layer estimate");
        assert!(max <= 32, "reference cap is 32 layers");
    }

    #[test]
    fn test_estimate_max_gpu_layers_fallback_tiers() {
        let make_gpu = |vram_mb: u64| GpuInfo {
            name: "Test GPU".to_string(),
            vram_mb,
            compute_capability: None,
            vendor: GpuVendor::Nvidia,
        };
        let model = ModelInfo {
            path: PathBuf::from("/tmp/m.onnx"),
            format: ModelFormat::ONNX,
            param_count: None, // unknown — triggers tier fallback
            hidden_size: None,
            attention_head_count: None,
            attention_head_count_kv: None,
            feed_forward_length: None,
            kv_cache_size_mb: None,
            memory_footprint_mb: None,
            flops_per_token: None,
        };

        assert_eq!(estimate_max_gpu_layers(&make_gpu(2048), &model), 16);
        assert_eq!(estimate_max_gpu_layers(&make_gpu(8192), &model), 32);
        assert_eq!(estimate_max_gpu_layers(&make_gpu(24576), &model), 48);
    }

    #[test]
    fn test_gpu_layer_options_proportional_not_hardcoded() {
        // With a 7B model on 8 GB VRAM (F16 ~13.4 GB), partial offload expected.
        let hw = hw_with_nvidia(8, 8192, 8);
        let model = gguf_model();
        let candidates = generate_candidates(&hw, &model);
        let layer_values: std::collections::BTreeSet<u32> =
            candidates.iter().map(|c| c.gpu_layers).collect();
        // Should NOT be exactly the old hardcoded set [0, 10, 20, 33]
        assert_ne!(
            layer_values,
            [0u32, 10, 20, 33].iter().cloned().collect::<std::collections::BTreeSet<_>>(),
            "gpu_layers must be VRAM-proportional, not the old hardcoded set"
        );
    }
}
