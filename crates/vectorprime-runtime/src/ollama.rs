// crates/vectorprime-runtime/src/ollama.rs
//
// OllamaAdapter — shells out to the `ollama` CLI binary to run inference.
//
// Ollama wraps llama.cpp internally, supports GGUF models, and works on both
// CPU and GPU without additional configuration. The adapter follows the same
// shell-out pattern as LlamaCppAdapter: no direct linking, binary detection
// via `which`, and structured error types for each failure mode.
//
// Used by: AdapterRegistry (lib.rs) and the benchmark loop (optimizer/benchmark.rs).

use std::path::PathBuf;
use std::process::Command;

use anyhow::Result;
use vectorprime_core::{
    BenchmarkResult, ModelInfo, QuantizationStrategy, RuntimeAdapter, RuntimeConfig, RuntimeError,
};

/// Adapter that drives inference via the `ollama` CLI.
///
/// Ollama manages its own server lifecycle, so `teardown` is a no-op. The
/// adapter stores the model tag derived from the GGUF file stem and uses it
/// with `ollama run` for each inference call.
pub struct OllamaAdapter {
    /// Resolved path to the `ollama` binary.
    binary: Option<PathBuf>,
    /// The stored runtime configuration.
    config: Option<RuntimeConfig>,
    /// The loaded model metadata.
    model: Option<ModelInfo>,
    /// Ollama model tag (derived from the file stem, e.g. "llama3-8b-q4km").
    model_tag: Option<String>,
}

impl OllamaAdapter {
    pub fn new() -> Self {
        Self {
            binary: None,
            config: None,
            model: None,
            model_tag: None,
        }
    }
}

impl Default for OllamaAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeAdapter for OllamaAdapter {
    /// Locate the `ollama` binary and store the configuration.
    ///
    /// Returns [`RuntimeError::NotInstalled`] when `ollama` is not on PATH.
    fn initialize(&mut self, config: &RuntimeConfig) -> Result<()> {
        let path = which::which("ollama").map_err(|_| {
            anyhow::anyhow!(RuntimeError::NotInstalled {
                binary: "ollama".to_string(),
                install_hint: "install Ollama from https://ollama.ai or via your package manager \
                               (e.g. brew install ollama)".to_string(),
            })
        })?;

        self.binary = Some(path);
        self.config = Some(config.clone());
        Ok(())
    }

    /// Record the model path and derive its Ollama tag.
    ///
    /// Ollama identifies models by a short tag string rather than a file path.
    /// We derive the tag from the GGUF file stem so it is predictable and
    /// collision-resistant within the benchmark session.
    ///
    /// Returns [`RuntimeError::ModelLoadFailed`] when the file does not exist.
    fn load_model(&mut self, model: &ModelInfo) -> Result<()> {
        if !model.path.exists() {
            return Err(anyhow::anyhow!(RuntimeError::ModelLoadFailed {
                path: model.path.display().to_string(),
                reason: "file not found".to_string(),
            }));
        }

        // Derive a sanitised tag from the file stem (lowercase, hyphens only).
        let stem = model
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        let tag = stem
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '-' })
            .collect::<String>();

        self.model_tag = Some(tag);
        self.model = Some(model.clone());
        Ok(())
    }

    /// Run one inference pass and return performance metrics.
    ///
    /// Shells out to:
    /// ```text
    /// ollama run <model_tag> "<prompt>" --verbose
    /// ```
    /// and parses `eval rate: N tokens/s` from stderr.
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
        let model_tag = self.model_tag.as_ref().ok_or_else(|| {
            anyhow::anyhow!(RuntimeError::InitializationFailed {
                reason: "model tag not set".to_string(),
            })
        })?;

        // Ollama prints timing metrics to stderr when --verbose is passed.
        let output = Command::new(binary)
            .args(["run", model_tag.as_str(), prompt, "--verbose"])
            .output()
            .map_err(|e| {
                anyhow::anyhow!(RuntimeError::InferenceFailed {
                    reason: e.to_string(),
                })
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!(RuntimeError::InferenceFailed {
                reason: format!("ollama exited non-zero: {}", stderr.trim()),
            }));
        }

        // Ollama emits timing lines on stderr, e.g.:
        //   eval rate:           32.55 tokens/s
        //   eval duration:       1.537417792s
        let stderr = String::from_utf8_lossy(&output.stderr);
        let (tokens_per_sec, latency_ms) =
            parse_ollama_output(&stderr).ok_or_else(|| {
                anyhow::anyhow!(RuntimeError::InferenceFailed {
                    reason: "could not parse timing output from ollama".to_string(),
                })
            })?;

        Ok(BenchmarkResult {
            tokens_per_sec,
            latency_ms,
            peak_memory_mb: estimate_memory_mb(model, config),
        })
    }

    /// No-op — Ollama manages its own server lifecycle.
    fn teardown(&mut self) -> Result<()> {
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Pure helpers (pub for unit testing)
// ──────────────────────────────────────────────────────────────────────────────

/// Parse Ollama `--verbose` output and return `(tokens_per_sec, latency_ms)`.
///
/// Ollama 0.x with `--verbose` prints lines like:
/// ```text
/// eval rate:           32.55 tokens/s
/// eval duration:       1.537417792s
/// ```
/// We extract `eval rate` for throughput and derive latency from `1000 / tps`
/// when `eval duration` is not in a simple ms format.
pub fn parse_ollama_output(output: &str) -> Option<(f64, f64)> {
    // Look for "eval rate:" with a numeric value before "tokens/s".
    let tps = output
        .lines()
        .find(|l| l.contains("eval rate") && l.contains("tokens/s"))
        .and_then(|line| extract_f64_before(line, "tokens/s"))?;

    // Try to parse "eval duration:" line for latency, e.g. "1.537417792s".
    let latency_ms = output
        .lines()
        .find(|l| l.contains("eval duration"))
        .and_then(|line| {
            // Format: "eval duration:       1.537417792s"
            let last = line.split_whitespace().last()?;
            let secs_str = last.trim_end_matches('s');
            secs_str.parse::<f64>().ok().map(|s| s * 1_000.0)
        })
        // Fall back to computing latency from throughput.
        .unwrap_or_else(|| 1_000.0 / tps);

    Some((tps, latency_ms))
}

/// Extract the last whitespace-delimited f64 token that appears before `marker`.
fn extract_f64_before(s: &str, marker: &str) -> Option<f64> {
    let pos = s.find(marker)?;
    s[..pos].split_whitespace().last()?.parse().ok()
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

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
    use std::path::PathBuf;
    use vectorprime_core::{ModelFormat, ModelInfo, QuantizationStrategy, RuntimeConfig, RuntimeKind};

    fn sample_config() -> RuntimeConfig {
        RuntimeConfig {
            runtime: RuntimeKind::Ollama,
            quantization: QuantizationStrategy::Q4_K_M,
            threads: 4,
            batch_size: 128,
            gpu_layers: 0,
        }
    }

    fn sample_model() -> ModelInfo {
        ModelInfo {
            path: PathBuf::from("/tmp/llama3-8b-q4km.gguf"),
            format: ModelFormat::GGUF,
            param_count: Some(8_000_000_000),
            hidden_size: None,
            attention_head_count: None,
            attention_head_count_kv: None,
            feed_forward_length: None,
            kv_cache_size_mb: None,
            memory_footprint_mb: None,
            flops_per_token: None,
        }
    }

    // ── parse_ollama_output ───────────────────────────────────────────────────

    #[test]
    fn test_parse_ollama_output_with_eval_rate_and_duration() {
        let output = concat!(
            "some output text\n",
            "eval rate:           32.55 tokens/s\n",
            "eval duration:       1.537417792s\n",
        );
        let (tps, lat) = parse_ollama_output(output).expect("should parse");
        assert!((tps - 32.55).abs() < 0.01, "tps={tps}");
        // 1.537417792 * 1000 = 1537.4...
        assert!(lat > 1500.0 && lat < 1600.0, "latency={lat}");
    }

    #[test]
    fn test_parse_ollama_output_no_duration_falls_back() {
        let output = "eval rate:           50.0 tokens/s\n";
        let (tps, lat) = parse_ollama_output(output).expect("should parse");
        assert!((tps - 50.0).abs() < 0.01, "tps={tps}");
        // Fallback: 1000 / 50 = 20ms
        assert!((lat - 20.0).abs() < 0.01, "latency={lat}");
    }

    #[test]
    fn test_parse_ollama_output_missing() {
        assert!(parse_ollama_output("no timings here").is_none());
        assert!(parse_ollama_output("").is_none());
    }

    // ── initialize when binary absent ─────────────────────────────────────────

    #[test]
    fn test_dispatch_not_installed_ollama() {
        // When `ollama` is absent, initialize must return NotInstalled, not panic.
        if which::which("ollama").is_ok() {
            // Binary present on this machine — skip the not-installed assertion.
            return;
        }
        let mut adapter = OllamaAdapter::new();
        let err = adapter.initialize(&sample_config()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not found") || msg.contains("ollama"),
            "unexpected error: {msg}"
        );
    }

    // ── load_model file existence check ──────────────────────────────────────

    #[test]
    fn test_load_model_missing_file() {
        let mut adapter = OllamaAdapter::new();
        adapter.binary = Some(PathBuf::from("/usr/bin/true"));
        adapter.config = Some(sample_config());

        let model = sample_model(); // path /tmp/llama3-8b-q4km.gguf likely absent
        if model.path.exists() {
            return; // file happens to exist — cannot test not-found path
        }
        let err = adapter.load_model(&model).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not found") || msg.contains("file"),
            "unexpected error: {msg}"
        );
    }
}
