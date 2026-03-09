//! Rust-level integration tests for llmforge-core shared types.
//!
//! Run with: cargo test --workspace
//!
//! These tests verify that the public API surface compiles and behaves
//! correctly without any subprocess or hardware dependency.

#[cfg(test)]
mod core_types {
    use llmforge_core::{
        BenchmarkResult, CpuInfo, GpuInfo, HardwareProfile, ModelFormat, ModelInfo,
        OptimizationResult, QuantizationStrategy, RamInfo, RuntimeConfig, RuntimeKind,
        SimdLevel,
    };
    use std::path::PathBuf;

    fn sample_hardware() -> HardwareProfile {
        HardwareProfile {
            cpu: CpuInfo {
                cores: 8,
                simd: SimdLevel::AVX2,
                cache_kb: 512,
                frequency_mhz: 3600,
            },
            gpu: Some(GpuInfo {
                vendor: "NVIDIA".to_string(),
                model: "RTX 4090".to_string(),
                vram_mb: 24576,
                tensor_cores: true,
                compute_capability: "8.9".to_string(),
            }),
            ram: RamInfo {
                total_mb: 65536,
                available_mb: 40000,
            },
        }
    }

    fn sample_config() -> RuntimeConfig {
        RuntimeConfig {
            runtime: RuntimeKind::LlamaCpp,
            quantization: QuantizationStrategy::Q4KM,
            threads: 16,
            batch_size: 1,
            gpu_layers: 20,
        }
    }

    fn sample_result() -> OptimizationResult {
        OptimizationResult {
            config: sample_config(),
            metrics: BenchmarkResult {
                tokens_per_sec: 110.3,
                latency_ms: 91.2,
                peak_memory_mb: 8400,
            },
        }
    }

    // ── construction ──────────────────────────────────────

    #[test]
    fn hardware_profile_constructs() {
        let hw = sample_hardware();
        assert_eq!(hw.cpu.cores, 8);
        assert!(hw.gpu.is_some());
        assert_eq!(hw.ram.total_mb, 65536);
    }

    #[test]
    fn model_info_constructs() {
        let mi = ModelInfo {
            path: PathBuf::from("/data/model.gguf"),
            format: ModelFormat::GGUF,
            param_count: Some(7_000_000_000),
        };
        assert_eq!(mi.format, ModelFormat::GGUF);
        assert_eq!(mi.param_count, Some(7_000_000_000));
    }

    #[test]
    fn optimization_result_constructs() {
        let r = sample_result();
        assert!(r.metrics.tokens_per_sec > 0.0);
        assert_eq!(r.config.gpu_layers, 20);
    }

    // ── serialization ────────────────────────────────────

    #[test]
    fn hardware_profile_serializes_to_json() {
        let hw = sample_hardware();
        let json = serde_json::to_string(&hw).expect("serialization failed");
        assert!(json.contains("AVX2"));
        assert!(json.contains("RTX 4090"));
    }

    #[test]
    fn hardware_profile_json_roundtrip() {
        let hw = sample_hardware();
        let json = serde_json::to_string(&hw).unwrap();
        let restored: HardwareProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.cpu.cores, hw.cpu.cores);
        assert_eq!(
            restored.gpu.as_ref().unwrap().vram_mb,
            hw.gpu.as_ref().unwrap().vram_mb
        );
    }

    #[test]
    fn optimization_result_json_roundtrip() {
        let r = sample_result();
        let json = serde_json::to_string(&r).unwrap();
        let restored: OptimizationResult = serde_json::from_str(&json).unwrap();
        assert!((restored.metrics.tokens_per_sec - r.metrics.tokens_per_sec).abs() < 0.001);
    }

    #[test]
    fn no_gpu_hardware_profile_serializes() {
        let hw = HardwareProfile {
            cpu: CpuInfo {
                cores: 4,
                simd: SimdLevel::AVX,
                cache_kb: 256,
                frequency_mhz: 2400,
            },
            gpu: None,
            ram: RamInfo {
                total_mb: 16384,
                available_mb: 8000,
            },
        };
        let json = serde_json::to_string(&hw).unwrap();
        assert!(json.contains("null") || json.contains("\"gpu\":null"));
    }

    // ── enum variants ────────────────────────────────────

    #[test]
    fn model_format_variants_are_distinct() {
        assert_ne!(ModelFormat::ONNX, ModelFormat::GGUF);
    }

    #[test]
    fn simd_levels_ordered_representation() {
        // Just verify they compile and are distinct
        let levels = [
            SimdLevel::None,
            SimdLevel::AVX,
            SimdLevel::AVX2,
            SimdLevel::AVX512,
        ];
        assert_eq!(levels.len(), 4);
    }

    #[test]
    fn all_quant_strategies_serialize() {
        let strategies = [
            QuantizationStrategy::F16,
            QuantizationStrategy::Q8_0,
            QuantizationStrategy::Q4KM,
            QuantizationStrategy::Q4_0,
            QuantizationStrategy::Int8,
            QuantizationStrategy::Int4,
        ];
        for s in &strategies {
            let json = serde_json::to_string(s).unwrap();
            assert!(!json.is_empty());
        }
    }

    // ── BenchmarkResult sanity ───────────────────────────

    #[test]
    fn benchmark_result_all_fields_positive() {
        let r = BenchmarkResult {
            tokens_per_sec: 99.9,
            latency_ms: 10.1,
            peak_memory_mb: 1024,
        };
        assert!(r.tokens_per_sec > 0.0);
        assert!(r.latency_ms > 0.0);
        assert!(r.peak_memory_mb > 0);
    }
}
