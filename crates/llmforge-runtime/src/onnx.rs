use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::Result;
use llmforge_core::{
    BenchmarkResult, ModelFormat, ModelInfo, RuntimeAdapter, RuntimeConfig, RuntimeError,
};
use serde::Deserialize;

/// Path to the bundled Python runner, relative to the workspace root.
const RUNNER_REL_PATH: &str = "python/llmforge/onnx_runner.py";

/// Adapter for ONNX Runtime — shells out to `python3 onnx_runner.py` via
/// a stdin/stdout JSON bridge.
pub struct OnnxAdapter {
    config: Option<RuntimeConfig>,
    model: Option<ModelInfo>,
    /// Resolved path to the `onnx_runner.py` script.
    runner_path: Option<PathBuf>,
}

impl OnnxAdapter {
    pub fn new() -> Self {
        Self {
            config: None,
            model: None,
            runner_path: None,
        }
    }

    /// Resolve the runner script path.
    ///
    /// Looks next to the binary first (installed layout), then walks up from
    /// `CARGO_MANIFEST_DIR` (development layout).
    fn find_runner() -> Option<PathBuf> {
        // 1. Next to the current executable (installed / maturin layout).
        if let Ok(exe) = std::env::current_exe() {
            let candidate = exe
                .parent()?
                .join("python")
                .join("llmforge")
                .join("onnx_runner.py");
            if candidate.exists() {
                return Some(candidate);
            }
        }

        // 2. Relative to CARGO_MANIFEST_DIR (dev / test layout).
        //    Walk up from the manifest until we find the workspace root
        //    (identified by the presence of `python/llmforge/onnx_runner.py`).
        if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
            let mut dir = PathBuf::from(manifest);
            for _ in 0..5 {
                let candidate = dir.join(RUNNER_REL_PATH);
                if candidate.exists() {
                    return Some(candidate);
                }
                if !dir.pop() {
                    break;
                }
            }
        }

        None
    }
}

impl Default for OnnxAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeAdapter for OnnxAdapter {
    /// Verify `python3` is on PATH and that `onnxruntime` can be imported.
    fn initialize(&mut self, config: &RuntimeConfig) -> Result<()> {
        // 1. python3 must be on PATH.
        let python = which::which("python3").map_err(|_| {
            anyhow::anyhow!(RuntimeError::NotInstalled {
                binary: "python3".to_string(),
                install_hint: "install Python 3 and onnxruntime (pip install onnxruntime)"
                    .to_string(),
            })
        })?;

        // 2. Locate the runner script.
        let runner = Self::find_runner().ok_or_else(|| {
            anyhow::anyhow!(RuntimeError::NotInstalled {
                binary: "onnx_runner.py".to_string(),
                install_hint: "install Python 3 and onnxruntime (pip install onnxruntime)"
                    .to_string(),
            })
        })?;

        // 3. Verify onnxruntime is importable.
        let status = Command::new(&python)
            .arg(&runner)
            .arg("--check")
            .status()
            .map_err(|e| {
                anyhow::anyhow!(RuntimeError::InitializationFailed {
                    reason: e.to_string(),
                })
            })?;

        if !status.success() {
            return Err(anyhow::anyhow!(RuntimeError::NotInstalled {
                binary: "onnxruntime (python package)".to_string(),
                install_hint: "install Python 3 and onnxruntime (pip install onnxruntime)"
                    .to_string(),
            }));
        }

        self.runner_path = Some(runner);
        self.config = Some(config.clone());
        Ok(())
    }

    /// Verify the model file exists and is in ONNX format.
    fn load_model(&mut self, model: &ModelInfo) -> Result<()> {
        if model.format != ModelFormat::ONNX {
            return Err(anyhow::anyhow!(RuntimeError::UnsupportedConfiguration {
                detail: format!(
                    "OnnxAdapter only supports ONNX models, got {:?}",
                    model.format
                ),
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

    /// Pipe a JSON config to `onnx_runner.py` via stdin and parse its stdout.
    fn run_inference(&self, _prompt: &str) -> Result<BenchmarkResult> {
        let runner = self.runner_path.as_ref().ok_or_else(|| {
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

        let provider = if config.gpu_layers > 0 {
            "CUDAExecutionProvider"
        } else {
            "CPUExecutionProvider"
        };

        let request = serde_json::json!({
            "model_path": model.path.display().to_string(),
            "execution_provider": provider,
            "threads": config.threads,
            "batch_size": config.batch_size,
            "prompt_tokens": 50,
        });
        let request_bytes = serde_json::to_vec(&request)?;

        let mut child = Command::new("python3")
            .arg(runner)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                anyhow::anyhow!(RuntimeError::InferenceFailed {
                    reason: e.to_string(),
                })
            })?;

        child
            .stdin
            .take()
            .expect("stdin was piped")
            .write_all(&request_bytes)
            .map_err(|e| {
                anyhow::anyhow!(RuntimeError::InferenceFailed {
                    reason: e.to_string(),
                })
            })?;

        let output = child.wait_with_output().map_err(|e| {
            anyhow::anyhow!(RuntimeError::InferenceFailed {
                reason: e.to_string(),
            })
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!(RuntimeError::InferenceFailed {
                reason: format!("onnx_runner.py exited non-zero: {}", stderr.trim()),
            }));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_onnx_output(stdout.trim())
    }

    fn teardown(&mut self) -> Result<()> {
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Pure parsing helper (public for unit testing)
// ──────────────────────────────────────────────────────────────────────────────

/// Deserialisation target for a successful runner response.
#[derive(Deserialize)]
struct OnnxSuccess {
    tokens_per_sec: f64,
    latency_ms: f64,
    peak_memory_mb: u64,
}

/// Deserialisation target for an error runner response.
#[derive(Deserialize)]
struct OnnxError {
    error: String,
}

/// Parse the JSON line written to stdout by `onnx_runner.py`.
///
/// Returns a [`BenchmarkResult`] on success, or an [`Err`] whose message
/// contains the `"error"` value from the JSON if the runner reported failure.
pub fn parse_onnx_output(json_str: &str) -> Result<BenchmarkResult> {
    // Try success shape first; fall back to error shape.
    if let Ok(ok) = serde_json::from_str::<OnnxSuccess>(json_str) {
        return Ok(BenchmarkResult {
            tokens_per_sec: ok.tokens_per_sec,
            latency_ms: ok.latency_ms,
            peak_memory_mb: ok.peak_memory_mb,
        });
    }

    if let Ok(err) = serde_json::from_str::<OnnxError>(json_str) {
        return Err(anyhow::anyhow!(RuntimeError::InferenceFailed {
            reason: err.error,
        }));
    }

    Err(anyhow::anyhow!(RuntimeError::InferenceFailed {
        reason: format!("unparseable onnx_runner output: {json_str}"),
    }))
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
            runtime: RuntimeKind::OnnxRuntime,
            quantization: QuantizationStrategy::F16,
            threads: 4,
            batch_size: 1,
            gpu_layers: 0,
        }
    }

    // ── parse_onnx_output ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_onnx_output_valid() {
        let json = r#"{"tokens_per_sec": 45.2, "latency_ms": 210.0, "peak_memory_mb": 3400}"#;
        let result = parse_onnx_output(json).expect("should parse valid output");
        assert!((result.tokens_per_sec - 45.2).abs() < 0.001);
        assert!((result.latency_ms - 210.0).abs() < 0.001);
        assert_eq!(result.peak_memory_mb, 3400);
    }

    #[test]
    fn test_parse_onnx_output_error() {
        let json = r#"{"error": "onnxruntime not installed"}"#;
        let err = parse_onnx_output(json).unwrap_err();
        assert!(
            err.to_string().contains("onnxruntime not installed"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn test_parse_onnx_output_garbled() {
        let err = parse_onnx_output("not json at all").unwrap_err();
        assert!(err.to_string().contains("unparseable"), "unexpected: {err}");
    }

    // ── load_model format check ───────────────────────────────────────────────

    #[test]
    fn test_load_wrong_format() {
        let mut adapter = OnnxAdapter::new();
        // Bypass initialize — set a dummy runner path so the adapter is
        // "initialized enough" to reach load_model's format check.
        adapter.runner_path = Some(PathBuf::from("/nonexistent/onnx_runner.py"));
        adapter.config = Some(sample_config());

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

    // ── initialize when python3 absent ───────────────────────────────────────

    #[test]
    fn test_initialize_python3_present_or_skip() {
        // If python3 is present but onnxruntime is absent, initialize should
        // return NotInstalled (not panic). If python3 itself is absent, same.
        // Either way: no panic.
        let mut adapter = OnnxAdapter::new();
        let _ = adapter.initialize(&sample_config()); // result is don't-care
    }
}
