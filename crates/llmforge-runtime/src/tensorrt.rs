use std::path::PathBuf;
use std::process::Command;

use anyhow::Result;
use llmforge_core::{
    BenchmarkResult, ModelFormat, ModelInfo, QuantizationStrategy, RuntimeAdapter, RuntimeConfig,
    RuntimeError,
};

pub struct TensorRtAdapter {
    config: Option<RuntimeConfig>,
    model: Option<ModelInfo>,
    binary: Option<PathBuf>,
}

impl TensorRtAdapter {
    pub fn new() -> Self {
        Self {
            config: None,
            model: None,
            binary: None,
        }
    }
}

impl Default for TensorRtAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeAdapter for TensorRtAdapter {
    fn initialize(&mut self, config: &RuntimeConfig) -> Result<()> {
        let path = which::which("trtexec").map_err(|_| {
            anyhow::anyhow!(RuntimeError::NotInstalled {
                binary: "trtexec".to_string(),
            })
        })?;

        // Verify compute capability via nvidia-smi.
        let cap = query_compute_capability();
        if let Some((major, _)) = cap {
            if major < 7 {
                return Err(anyhow::anyhow!(RuntimeError::UnsupportedConfiguration {
                    detail: format!("TensorRT requires compute capability ≥ 7.0, found {major}.x"),
                }));
            }
        }
        // If nvidia-smi is absent we proceed; trtexec will fail later if truly
        // no GPU is present.

        self.binary = Some(path);
        self.config = Some(config.clone());
        Ok(())
    }

    fn load_model(&mut self, model: &ModelInfo) -> Result<()> {
        if model.format != ModelFormat::ONNX {
            return Err(anyhow::anyhow!(RuntimeError::UnsupportedConfiguration {
                detail: format!("TensorRT only supports ONNX models, got {:?}", model.format),
            }));
        }
        if !model.path.exists() {
            return Err(anyhow::anyhow!(RuntimeError::ModelLoadFailed {
                path: model.path.display().to_string(),
                reason: "file not found".to_string(),
            }));
        }
        self.model = Some(model.clone());
        Ok(())
    }

    fn run_inference(&self, _prompt: &str) -> Result<BenchmarkResult> {
        let binary = self.binary.as_ref().ok_or_else(|| {
            anyhow::anyhow!(RuntimeError::InitializationFailed {
                reason: "adapter not initialized".to_string(),
            })
        })?;
        let config = self.config.as_ref().ok_or_else(|| {
            anyhow::anyhow!(RuntimeError::InitializationFailed {
                reason: "adapter not initialized".to_string(),
            })
        })?;
        let model = self.model.as_ref().ok_or_else(|| {
            anyhow::anyhow!(RuntimeError::InitializationFailed {
                reason: "model not loaded".to_string(),
            })
        })?;

        let precision = quant_to_flag(&config.quantization);
        let output = Command::new(binary)
            .args([
                &format!("--onnx={}", model.path.display()),
                precision,
                &format!("--batch={}", config.batch_size),
                "--workspace=4096",
                "--iterations=10",
                "--warmUp=2000",
                "--duration=0",
            ])
            .output()
            .map_err(|e| {
                anyhow::anyhow!(RuntimeError::InferenceFailed {
                    reason: e.to_string(),
                })
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!(RuntimeError::InferenceFailed {
                reason: format!("trtexec exited non-zero: {}", stderr.trim()),
            }));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        let tokens_per_sec = parse_throughput(&stdout).ok_or_else(|| {
            anyhow::anyhow!(RuntimeError::InferenceFailed {
                reason: "could not parse throughput from trtexec output".to_string(),
            })
        })?;
        let latency_ms = parse_latency(&stdout).ok_or_else(|| {
            anyhow::anyhow!(RuntimeError::InferenceFailed {
                reason: "could not parse latency from trtexec output".to_string(),
            })
        })?;
        let peak_memory_mb = parse_memory(&stdout).unwrap_or(0);

        Ok(BenchmarkResult {
            tokens_per_sec,
            latency_ms,
            peak_memory_mb,
        })
    }

    fn teardown(&mut self) -> Result<()> {
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Pure helpers (public for unit testing)
// ──────────────────────────────────────────────────────────────────────────────

/// Map a quantization strategy to the trtexec precision flag.
pub fn quant_to_flag(q: &QuantizationStrategy) -> &'static str {
    match q {
        QuantizationStrategy::F16
        | QuantizationStrategy::Q8_0
        | QuantizationStrategy::Q4_K_M
        | QuantizationStrategy::Q4_0 => "--fp16",
        QuantizationStrategy::Int8 | QuantizationStrategy::Int4 => "--int8",
    }
}

/// Extract throughput (queries per second) from trtexec stdout.
///
/// Looks for: `Throughput: X qps`
pub fn parse_throughput(output: &str) -> Option<f64> {
    let line = output.lines().find(|l| l.contains("Throughput:"))?;
    // "Throughput: 42.5 qps"
    let after = line.split_once("Throughput:")?.1;
    after.split_whitespace().next()?.parse().ok()
}

/// Extract mean latency (ms) from trtexec stdout.
///
/// Looks for: `Latency: min = … ms, max = … ms, mean = Z ms, …`
pub fn parse_latency(output: &str) -> Option<f64> {
    let line = output
        .lines()
        .find(|l| l.contains("Latency:") && l.contains("mean"))?;
    // Walk past "mean = " and grab the number before " ms"
    let after_mean = line.split_once("mean =")?.1;
    after_mean
        .split_whitespace()
        .next()?
        .trim_end_matches(',')
        .parse()
        .ok()
}

/// Extract peak GPU memory (MiB → MB) from trtexec stdout.
///
/// Looks for: `GPU Memory: X MiB`
pub fn parse_memory(output: &str) -> Option<u64> {
    let line = output.lines().find(|l| l.contains("GPU Memory:"))?;
    let after = line.split_once("GPU Memory:")?.1;
    after.split_whitespace().next()?.parse().ok()
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Query NVIDIA compute capability via nvidia-smi.
/// Returns `None` when nvidia-smi is absent or produces unparseable output.
fn query_compute_capability() -> Option<(u32, u32)> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;

    if !out.status.success() {
        return None;
    }

    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().next()?.trim();
    let mut parts = line.splitn(2, '.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    Some((major, minor))
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use llmforge_core::{ModelFormat, ModelInfo, QuantizationStrategy, RuntimeConfig, RuntimeKind};
    use std::path::PathBuf;

    fn sample_config() -> RuntimeConfig {
        RuntimeConfig {
            runtime: RuntimeKind::TensorRT,
            quantization: QuantizationStrategy::F16,
            threads: 4,
            batch_size: 1,
            gpu_layers: 0,
        }
    }

    // ── parse_throughput ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_throughput() {
        let output = "[I] === Performance summary ===\n\
                      [I] Throughput: 42.57 qps\n\
                      [I] Latency: min = 23.1 ms, max = 25.4 ms, mean = 23.5 ms\n";
        let tps = parse_throughput(output).unwrap();
        assert!((tps - 42.57).abs() < 0.01, "tps={tps}");

        assert!(parse_throughput("no throughput here").is_none());
    }

    // ── parse_latency ─────────────────────────────────────────────────────────

    #[test]
    fn test_parse_latency_mean() {
        let output =
            "[I] Latency: min = 23.10 ms, max = 25.40 ms, mean = 23.50 ms, median = 23.48 ms\n";
        let lat = parse_latency(output).unwrap();
        assert!((lat - 23.50).abs() < 0.01, "latency={lat}");

        assert!(parse_latency("no latency info").is_none());
    }

    // ── parse_memory ──────────────────────────────────────────────────────────

    #[test]
    fn test_parse_memory() {
        let output = "[I] GPU Memory: 2048 MiB\n";
        let mem = parse_memory(output).unwrap();
        assert_eq!(mem, 2048);

        assert!(parse_memory("no gpu memory line").is_none());
    }

    // ── quant_to_flag ─────────────────────────────────────────────────────────

    #[test]
    fn test_quant_flag_int8() {
        assert_eq!(quant_to_flag(&QuantizationStrategy::Int8), "--int8");
        assert_eq!(quant_to_flag(&QuantizationStrategy::Int4), "--int8");
    }

    #[test]
    fn test_quant_flag_fp16() {
        assert_eq!(quant_to_flag(&QuantizationStrategy::F16), "--fp16");
        assert_eq!(quant_to_flag(&QuantizationStrategy::Q4_K_M), "--fp16");
        assert_eq!(quant_to_flag(&QuantizationStrategy::Q8_0), "--fp16");
    }

    // ── load_model format check ───────────────────────────────────────────────

    #[test]
    fn test_load_gguf_unsupported() {
        let mut adapter = TensorRtAdapter {
            binary: Some(PathBuf::from("/usr/bin/true")),
            config: Some(sample_config()),
            model: None,
        };
        let model = ModelInfo {
            path: PathBuf::from("/tmp/model.gguf"),
            format: ModelFormat::GGUF,
            param_count: None,
        };
        let err = adapter.load_model(&model).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("ONNX") || msg.contains("unsupported"),
            "unexpected error: {msg}"
        );
    }

    // ── initialize when trtexec absent ────────────────────────────────────────

    #[test]
    fn test_initialize_not_installed() {
        if which::which("trtexec").is_ok() {
            return; // binary present — skip
        }
        let mut adapter = TensorRtAdapter::new();
        let err = adapter.initialize(&sample_config()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not found") || msg.contains("trtexec"),
            "unexpected error: {msg}"
        );
    }
}
