use llmforge_core::{
    HardwareProfile, ModelFormat, ModelInfo, QuantizationStrategy, RuntimeConfig, RuntimeKind,
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

/// Return true when NVIDIA GPU with compute capability major >= 7 is present.
fn has_trt_capable_gpu(hw: &HardwareProfile) -> bool {
    hw.gpu
        .as_ref()
        .and_then(|g| g.compute_capability)
        .map(|(major, _)| major >= 7)
        .unwrap_or(false)
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

    // GPU layer options depend on GPU presence.
    let gpu_layer_options: Vec<u32> = if hw.gpu.is_some() {
        vec![0, 10, 20, 33]
    } else {
        vec![0]
    };

    // Per-format runtime + quantization pairings.
    let combos: Vec<(RuntimeKind, QuantizationStrategy)> = match model.format {
        ModelFormat::GGUF => [
            QuantizationStrategy::Q4_K_M,
            QuantizationStrategy::Q4_0,
            QuantizationStrategy::Q8_0,
            QuantizationStrategy::F16,
        ]
        .into_iter()
        .map(|q| (RuntimeKind::LlamaCpp, q))
        .collect(),

        ModelFormat::ONNX => {
            let mut v: Vec<(RuntimeKind, QuantizationStrategy)> = vec![
                (RuntimeKind::OnnxRuntime, QuantizationStrategy::F16),
                (RuntimeKind::OnnxRuntime, QuantizationStrategy::Int8),
            ];
            if has_trt_capable_gpu(hw) {
                v.push((RuntimeKind::TensorRT, QuantizationStrategy::F16));
                v.push((RuntimeKind::TensorRT, QuantizationStrategy::Int8));
            }
            v
        }
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
                    batch_size: 512,
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
    use llmforge_core::{
        CpuInfo, GpuInfo, HardwareProfile, ModelFormat, ModelInfo, RamInfo, SimdLevel,
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
        }
    }

    fn onnx_model() -> ModelInfo {
        ModelInfo {
            path: PathBuf::from("/tmp/model.onnx"),
            format: ModelFormat::ONNX,
            param_count: Some(1_000_000_000),
        }
    }

    #[test]
    fn test_gguf_candidates_only_llamacpp() {
        let hw = cpu_only_hw(8);
        let model = gguf_model();
        let candidates = generate_candidates(&hw, &model);
        assert!(!candidates.is_empty());
        for c in &candidates {
            assert_eq!(
                c.runtime,
                RuntimeKind::LlamaCpp,
                "non-llama runtime in GGUF candidates"
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
        let hw = hw_with_nvidia(8, 24576, 8); // compute cap 8.x >= 7
        let model = onnx_model();
        let candidates = generate_candidates(&hw, &model);
        assert!(
            candidates
                .iter()
                .any(|c| c.runtime == RuntimeKind::TensorRT),
            "expected TensorRT candidates with capable GPU"
        );
    }

    #[test]
    fn test_onnx_with_old_gpu_excludes_tensorrt() {
        let hw = hw_with_nvidia(8, 8192, 6); // compute cap 6.x < 7
        let model = onnx_model();
        let candidates = generate_candidates(&hw, &model);
        for c in &candidates {
            assert_ne!(
                c.runtime,
                RuntimeKind::TensorRT,
                "TensorRT should be excluded for cap < 7"
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
}
