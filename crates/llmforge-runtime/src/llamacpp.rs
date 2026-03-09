use std::path::PathBuf;
use std::process::Command;

use anyhow::Result;
use llmforge_core::{
    BenchmarkResult, ModelFormat, ModelInfo, QuantizationStrategy, RuntimeAdapter, RuntimeConfig,
    RuntimeError,
};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

pub struct LlamaCppAdapter {
    config: Option<RuntimeConfig>,
    model: Option<ModelInfo>,
    binary: Option<PathBuf>,
    version: Option<String>,
}

impl LlamaCppAdapter {
    pub fn new() -> Self {
        Self {
            config: None,
            model: None,
            binary: None,
            version: None,
        }
    }
}

impl Default for LlamaCppAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeAdapter for LlamaCppAdapter {
    fn initialize(&mut self, config: &RuntimeConfig) -> Result<()> {
        let path = which::which("llama-cli").map_err(|_| {
            anyhow::anyhow!(RuntimeError::NotInstalled {
                binary: "llama-cli".to_string(),
            })
        })?;

        let version = capture_version(&path);
        self.binary = Some(path);
        self.version = version;
        self.config = Some(config.clone());
        Ok(())
    }

    fn load_model(&mut self, model: &ModelInfo) -> Result<()> {
        // Format check.
        if model.format != ModelFormat::GGUF {
            return Err(anyhow::anyhow!(RuntimeError::UnsupportedConfiguration {
                detail: format!(
                    "llama.cpp only supports GGUF models, got {:?}",
                    model.format
                ),
            }));
        }

        // Existence check.
        if !model.path.exists() {
            return Err(anyhow::anyhow!(RuntimeError::ModelLoadFailed {
                path: model.path.display().to_string(),
                reason: "file not found".to_string(),
            }));
        }

        // Magic-bytes check — first 4 bytes must be "GGUF".
        let magic = read_magic(&model.path).map_err(|e| {
            anyhow::anyhow!(RuntimeError::ModelLoadFailed {
                path: model.path.display().to_string(),
                reason: e.to_string(),
            })
        })?;

        if &magic != GGUF_MAGIC {
            return Err(anyhow::anyhow!(RuntimeError::UnsupportedConfiguration {
                detail: format!(
                    "file does not start with GGUF magic bytes (got {:?})",
                    magic
                ),
            }));
        }

        self.model = Some(model.clone());
        Ok(())
    }

    fn run_inference(&self, prompt: &str) -> Result<BenchmarkResult> {
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

        let output = Command::new(binary)
            .args([
                "-m",
                &model.path.display().to_string(),
                "--threads",
                &config.threads.to_string(),
                "--n-gpu-layers",
                &config.gpu_layers.to_string(),
                "--ctx-size",
                "512",
                "-p",
                prompt,
                "--n-predict",
                "50",
                "--log-disable",
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
                reason: format!("llama-cli exited non-zero: {}", stderr.trim()),
            }));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let (tokens_per_sec, latency_ms) =
            parse_llama_timings(&stdout).ok_or_else(|| {
                anyhow::anyhow!(RuntimeError::InferenceFailed {
                    reason: "could not parse timing output from llama-cli".to_string(),
                })
            })?;

        Ok(BenchmarkResult {
            tokens_per_sec,
            latency_ms,
            peak_memory_mb: estimate_memory_mb(model, config),
        })
    }

    fn teardown(&mut self) -> Result<()> {
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Pure helpers (public for unit testing)
// ──────────────────────────────────────────────────────────────────────────────

/// Parse llama.cpp timing output and return `(tokens_per_sec, latency_ms)`.
///
/// Looks for a line like:
/// ```text
/// llama_print_timings:     eval time =   413.02 ms /    50 tokens (    8.26 ms per token,   121.06 tokens per second)
/// ```
pub fn parse_llama_timings(output: &str) -> Option<(f64, f64)> {
    let line = output
        .lines()
        .find(|l| l.contains("eval time") && l.contains("tokens per second"))?;

    let tokens_per_sec = extract_f64_before(line, "tokens per second")?;
    let latency_ms = extract_f64_before(line, "ms per token")?;

    Some((tokens_per_sec, latency_ms))
}

/// Extract the last whitespace-delimited f64 token that appears before `marker`
/// in `s`.
fn extract_f64_before(s: &str, marker: &str) -> Option<f64> {
    let pos = s.find(marker)?;
    s[..pos].split_whitespace().last()?.parse().ok()
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

fn capture_version(binary: &std::path::Path) -> Option<String> {
    Command::new(binary)
        .arg("--version")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn read_magic(path: &std::path::Path) -> std::io::Result<[u8; 4]> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

fn estimate_memory_mb(model: &ModelInfo, config: &RuntimeConfig) -> u64 {
    let params = match model.param_count {
        Some(p) => p,
        None => return 0,
    };
    let bytes_per_param: f64 = match config.quantization {
        QuantizationStrategy::F16 => 2.0,
        QuantizationStrategy::Q8_0 | QuantizationStrategy::Int8 => 1.0,
        QuantizationStrategy::Q4_K_M | QuantizationStrategy::Q4_0 | QuantizationStrategy::Int4 => {
            0.5
        }
    };
    ((params as f64 * bytes_per_param) / 1_000_000.0) as u64
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use llmforge_core::{ModelFormat, ModelInfo, QuantizationStrategy, RuntimeConfig, RuntimeKind};
    use std::io::Write as _;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    fn sample_config() -> RuntimeConfig {
        RuntimeConfig {
            runtime: RuntimeKind::LlamaCpp,
            quantization: QuantizationStrategy::Q4_K_M,
            threads: 4,
            batch_size: 128,
            gpu_layers: 0,
        }
    }

    // ── parse_llama_timings ───────────────────────────────────────────────────

    #[test]
    fn test_parse_timings_valid() {
        let output = concat!(
            "llama_print_timings:     load time =   150.00 ms\n",
            "llama_print_timings:     eval time =   413.02 ms /    50 tokens",
            " (    8.26 ms per token,   121.06 tokens per second)\n",
        );
        let (tps, lat) = parse_llama_timings(output).expect("should parse");
        assert!((tps - 121.06).abs() < 0.01, "tps={tps}");
        assert!((lat - 8.26).abs() < 0.01, "latency={lat}");
    }

    #[test]
    fn test_parse_timings_missing() {
        assert!(parse_llama_timings("no timings here").is_none());
        assert!(parse_llama_timings("").is_none());
    }

    // ── load_model format / magic checks ─────────────────────────────────────

    #[test]
    fn test_unsupported_format_onnx() {
        let mut adapter = LlamaCppAdapter::new();
        // Bypass initialize by manually setting a dummy binary path.
        adapter.binary = Some(PathBuf::from("/usr/bin/true"));
        adapter.config = Some(sample_config());

        let model = ModelInfo {
            path: PathBuf::from("/tmp/model.onnx"),
            format: ModelFormat::ONNX,
            param_count: None,
        };
        let err = adapter.load_model(&model).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("GGUF") || msg.contains("unsupported"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_load_bad_magic() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"BADD").unwrap();

        let mut adapter = LlamaCppAdapter::new();
        adapter.binary = Some(PathBuf::from("/usr/bin/true"));
        adapter.config = Some(sample_config());

        let model = ModelInfo {
            path: tmp.path().to_path_buf(),
            format: ModelFormat::GGUF,
            param_count: None,
        };
        let err = adapter.load_model(&model).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("GGUF") || msg.contains("magic") || msg.contains("unsupported"),
            "unexpected error: {msg}"
        );
    }

    // ── initialize when binary absent ─────────────────────────────────────────

    #[test]
    fn test_initialize_not_installed() {
        // llama-cli is not present on the test machine; initialize must return
        // NotInstalled, not panic.
        let mut adapter = LlamaCppAdapter::new();
        if which::which("llama-cli").is_ok() {
            // Binary actually present — skip this assertion.
            return;
        }
        let err = adapter.initialize(&sample_config()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not found") || msg.contains("llama-cli"),
            "unexpected error: {msg}"
        );
    }
}
