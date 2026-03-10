use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

// ──────────────────────────────────────────────────────────────────────────────
// Enums
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelFormat {
    ONNX,
    GGUF,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SimdLevel {
    None,
    AVX,
    AVX2,
    AVX512,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(non_camel_case_types)] // Q4_K_M is a domain-standard quantization name
pub enum QuantizationStrategy {
    F16,
    Q8_0,
    Q4_K_M,
    Q4_0,
    Int8,
    Int4,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuntimeKind {
    LlamaCpp,
    OnnxRuntime,
    TensorRT,
}

/// GPU vendor, used to make runtime-eligibility decisions without hardcoding
/// microarchitecture thresholds.
///
/// `Unknown` is the safe fallback when detection succeeds but vendor
/// identification does not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Apple,
    Unknown,
}

// ──────────────────────────────────────────────────────────────────────────────
// Hardware structs
// ──────────────────────────────────────────────────────────────────────────────

/// CPU information collected from the host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// Logical core count.
    pub core_count: u32,
    /// Brand/model string (e.g. "Intel Core i9-13900K").
    pub brand: String,
    /// Highest supported SIMD extension.
    pub simd_level: SimdLevel,
}

/// GPU information collected from the host. `None` when no GPU is present.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU device name (e.g. "NVIDIA RTX 4090").
    pub name: String,
    /// VRAM capacity in megabytes.
    pub vram_mb: u64,
    /// CUDA compute capability (e.g. `(8, 9)` for Ada Lovelace).
    /// Only populated for NVIDIA GPUs; `None` for AMD and Apple.
    pub compute_capability: Option<(u32, u32)>,
    /// Which GPU vendor produced this device.
    pub vendor: GpuVendor,
}

/// RAM information collected from the host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RamInfo {
    /// Total system RAM in megabytes.
    pub total_mb: u64,
    /// Currently available RAM in megabytes.
    pub available_mb: u64,
}

/// Snapshot of host hardware, passed through the optimization pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub cpu: CpuInfo,
    pub gpu: Option<GpuInfo>,
    pub ram: RamInfo,
}

// ──────────────────────────────────────────────────────────────────────────────
// Model / runtime structs
// ──────────────────────────────────────────────────────────────────────────────

/// Metadata about the model file under optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub path: PathBuf,
    pub format: ModelFormat,
    /// Parameter count when known (e.g. 7_000_000_000 for a 7B model).
    pub param_count: Option<u64>,
}

/// A concrete runtime configuration to be benchmarked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub runtime: RuntimeKind,
    pub quantization: QuantizationStrategy,
    /// Number of CPU threads the runtime should use.
    pub threads: u32,
    /// Token batch size for a single inference call.
    pub batch_size: u32,
    /// Number of transformer layers offloaded to the GPU (0 = CPU only).
    pub gpu_layers: u32,
}

/// Performance metrics collected during a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Sustained throughput in tokens per second.
    pub tokens_per_sec: f64,
    /// End-to-end latency in milliseconds for the benchmark prompt.
    pub latency_ms: f64,
    /// Peak RSS / VRAM usage during the run, in megabytes.
    pub peak_memory_mb: u64,
}

/// Final output of the optimization pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub config: RuntimeConfig,
    pub metrics: BenchmarkResult,
}

// ──────────────────────────────────────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────────────────────────────────────

/// Structured errors produced by runtime adapters.
#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("required binary '{binary}' was not found in PATH — {install_hint}")]
    NotInstalled {
        binary: String,
        install_hint: String,
    },

    #[error("model initialization failed: {reason}")]
    InitializationFailed { reason: String },

    #[error("model load failed for '{path}': {reason}")]
    ModelLoadFailed { path: String, reason: String },

    #[error("inference failed: {reason}")]
    InferenceFailed { reason: String },

    #[error("unsupported configuration: {detail}")]
    UnsupportedConfiguration { detail: String },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// ──────────────────────────────────────────────────────────────────────────────
// Traits
// ──────────────────────────────────────────────────────────────────────────────

/// Abstraction over an external inference runtime (llama.cpp, ONNX, TensorRT).
///
/// Implementations shell out to external binaries rather than linking directly.
/// If a required binary is absent, `initialize` must return
/// [`RuntimeError::NotInstalled`] so the optimizer can skip the adapter.
pub trait RuntimeAdapter: Send + Sync {
    /// Validate the configuration and prepare internal state.
    fn initialize(&mut self, config: &RuntimeConfig) -> Result<()>;

    /// Load the model from disk into the runtime.
    fn load_model(&mut self, model: &ModelInfo) -> Result<()>;

    /// Run a single inference pass and return performance metrics.
    fn run_inference(&self, prompt: &str) -> Result<BenchmarkResult>;

    /// Release any resources held by the runtime (processes, temp files, …).
    fn teardown(&mut self) -> Result<()>;
}

/// Pluggable GPU detection interface.
///
/// Implement this trait for each GPU vendor (NVIDIA, AMD, Apple Metal, …)
/// without changing the hardware profiler.
pub trait GpuProbe: Send + Sync {
    /// Attempt to detect GPU hardware.
    ///
    /// Returns `Some(GpuInfo)` when a supported GPU is present, `None` otherwise.
    fn probe(&self) -> Option<GpuInfo>;
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_hardware_profile() -> HardwareProfile {
        HardwareProfile {
            cpu: CpuInfo {
                core_count: 16,
                brand: "Intel Core i9-13900K".to_string(),
                simd_level: SimdLevel::AVX512,
            },
            gpu: Some(GpuInfo {
                name: "NVIDIA RTX 4090".to_string(),
                vram_mb: 24576,
                compute_capability: Some((8, 9)),
                vendor: GpuVendor::Nvidia,
            }),
            ram: RamInfo {
                total_mb: 65536,
                available_mb: 32768,
            },
        }
    }

    fn sample_optimization_result() -> OptimizationResult {
        OptimizationResult {
            config: RuntimeConfig {
                runtime: RuntimeKind::LlamaCpp,
                quantization: QuantizationStrategy::Q4_K_M,
                threads: 16,
                batch_size: 512,
                gpu_layers: 20,
            },
            metrics: BenchmarkResult {
                tokens_per_sec: 110.3,
                latency_ms: 91.2,
                peak_memory_mb: 8396,
            },
        }
    }

    // ── Enum serialization ────────────────────────────────────────────────────

    #[test]
    fn model_format_serializes() {
        let json = serde_json::to_string(&ModelFormat::GGUF).unwrap();
        assert_eq!(json, r#""GGUF""#);
    }

    #[test]
    fn simd_level_serializes() {
        let json = serde_json::to_string(&SimdLevel::AVX2).unwrap();
        assert_eq!(json, r#""AVX2""#);
    }

    #[test]
    fn quantization_strategy_serializes() {
        let json = serde_json::to_string(&QuantizationStrategy::Q4_K_M).unwrap();
        assert_eq!(json, r#""Q4_K_M""#);
    }

    #[test]
    fn runtime_kind_serializes() {
        let json = serde_json::to_string(&RuntimeKind::OnnxRuntime).unwrap();
        assert_eq!(json, r#""OnnxRuntime""#);
    }

    // ── Struct construction + serialization ───────────────────────────────────

    #[test]
    fn model_info_serializes() {
        let info = ModelInfo {
            path: PathBuf::from("/models/llama3.gguf"),
            format: ModelFormat::GGUF,
            param_count: Some(7_000_000_000),
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("GGUF"));
        assert!(json.contains("7000000000"));
    }

    #[test]
    fn runtime_config_serializes() {
        let cfg = RuntimeConfig {
            runtime: RuntimeKind::TensorRT,
            quantization: QuantizationStrategy::Int8,
            threads: 8,
            batch_size: 256,
            gpu_layers: 40,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("TensorRT"));
        assert!(json.contains("Int8"));
    }

    #[test]
    fn benchmark_result_serializes() {
        let result = BenchmarkResult {
            tokens_per_sec: 110.3,
            latency_ms: 91.2,
            peak_memory_mb: 8192,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("tokens_per_sec"));
        assert!(json.contains("110.3"));
    }

    // ── JSON roundtrip ────────────────────────────────────────────────────────

    #[test]
    fn hardware_profile_roundtrip() {
        let original = sample_hardware_profile();
        let json = serde_json::to_string(&original).unwrap();
        let restored: HardwareProfile = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.cpu.core_count, original.cpu.core_count);
        assert_eq!(restored.cpu.brand, original.cpu.brand);
        assert_eq!(restored.cpu.simd_level, original.cpu.simd_level);
        assert!(restored.gpu.is_some());
        let gpu = restored.gpu.unwrap();
        assert_eq!(gpu.name, "NVIDIA RTX 4090");
        assert_eq!(gpu.vram_mb, 24576);
        assert_eq!(gpu.compute_capability, Some((8, 9)));
        assert_eq!(gpu.vendor, GpuVendor::Nvidia);
        assert_eq!(restored.ram.total_mb, 65536);
        assert_eq!(restored.ram.available_mb, 32768);
    }

    #[test]
    fn hardware_profile_no_gpu_roundtrip() {
        let original = HardwareProfile {
            cpu: CpuInfo {
                core_count: 8,
                brand: "AMD Ryzen 7 5800X".to_string(),
                simd_level: SimdLevel::AVX2,
            },
            gpu: None,
            ram: RamInfo {
                total_mb: 32768,
                available_mb: 16384,
            },
        };
        let json = serde_json::to_string(&original).unwrap();
        let restored: HardwareProfile = serde_json::from_str(&json).unwrap();
        assert!(restored.gpu.is_none());
        assert_eq!(restored.cpu.simd_level, SimdLevel::AVX2);
    }

    #[test]
    fn optimization_result_roundtrip() {
        let original = sample_optimization_result();
        let json = serde_json::to_string(&original).unwrap();
        let restored: OptimizationResult = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.config.runtime, RuntimeKind::LlamaCpp);
        assert_eq!(restored.config.quantization, QuantizationStrategy::Q4_K_M);
        assert_eq!(restored.config.threads, 16);
        assert_eq!(restored.config.batch_size, 512);
        assert_eq!(restored.config.gpu_layers, 20);
        assert!((restored.metrics.tokens_per_sec - 110.3).abs() < f64::EPSILON);
        assert!((restored.metrics.latency_ms - 91.2).abs() < f64::EPSILON);
        assert_eq!(restored.metrics.peak_memory_mb, 8396);
    }

    // ── RuntimeError ──────────────────────────────────────────────────────────

    #[test]
    fn runtime_error_not_installed_message() {
        let err = RuntimeError::NotInstalled {
            binary: "llama-cli".to_string(),
            install_hint: "see https://example.com".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("llama-cli"), "unexpected message: {msg}");
        assert!(msg.contains("not found"), "unexpected message: {msg}");
        assert!(
            msg.contains("see https://example.com"),
            "unexpected message: {msg}"
        );
    }

    #[test]
    fn runtime_error_unsupported_config_message() {
        let err = RuntimeError::UnsupportedConfiguration {
            detail: "TensorRT requires compute capability ≥ 7.0".to_string(),
        };
        assert!(err.to_string().contains("TensorRT"));
    }
}
