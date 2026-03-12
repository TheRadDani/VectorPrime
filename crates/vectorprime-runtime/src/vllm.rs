// crates/vectorprime-runtime/src/vllm.rs
//
// VllmAdapter — drives inference via the vLLM Python library using `python3`.
//
// vLLM is a high-throughput LLM inference engine designed for NVIDIA GPUs.
// It supports HuggingFace and ONNX-compatible model formats, and shines on
// F16 / BF16 / Q8_0 precision with tensor-parallelism.
//
// The adapter shells out to `python3 -c "..."` rather than linking libvllm
// directly, following the same shell-out convention used by OnnxAdapter.
//
// Used by: AdapterRegistry (lib.rs) and the benchmark loop (optimizer/benchmark.rs).

use std::path::PathBuf;
use std::process::Command;

use anyhow::Result;
use vectorprime_core::{
    BenchmarkResult, ModelInfo, QuantizationStrategy, RuntimeAdapter, RuntimeConfig, RuntimeError,
};

/// Adapter that drives inference via the vLLM Python library.
///
/// `initialize` checks for both `python3` on PATH and `import vllm` being
/// importable. If either check fails the adapter returns
/// [`RuntimeError::NotInstalled`] so the optimizer can skip it gracefully.
pub struct VllmAdapter {
    /// Resolved path to the `python3` binary.
    python_binary: Option<PathBuf>,
    /// The stored runtime configuration.
    config: Option<RuntimeConfig>,
    /// The loaded model metadata.
    model: Option<ModelInfo>,
}

impl VllmAdapter {
    pub fn new() -> Self {
        Self {
            python_binary: None,
            config: None,
            model: None,
        }
    }
}

impl Default for VllmAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeAdapter for VllmAdapter {
    /// Locate `python3` and verify vLLM is importable.
    ///
    /// Returns [`RuntimeError::NotInstalled`] when either `python3` is absent
    /// or `import vllm` fails (module not installed).
    fn initialize(&mut self, config: &RuntimeConfig) -> Result<()> {
        let python = which::which("python3").map_err(|_| {
            anyhow::anyhow!(RuntimeError::NotInstalled {
                binary: "python3".to_string(),
                install_hint:
                    "install Python 3 from https://python.org or via your system package manager"
                        .to_string(),
            })
        })?;

        // Verify vLLM is importable — a zero-cost check compared to a full
        // inference run, and surfaces missing dependencies early.
        let check = Command::new(&python)
            .args(["-c", "import vllm"])
            .output()
            .map_err(|e| {
                anyhow::anyhow!(RuntimeError::InitializationFailed {
                    reason: format!("could not invoke python3 for vLLM check: {e}"),
                })
            })?;

        if !check.status.success() {
            let stderr = String::from_utf8_lossy(&check.stderr);
            return Err(anyhow::anyhow!(RuntimeError::NotInstalled {
                binary: "vllm".to_string(),
                install_hint: format!(
                    "install vLLM via `pip install vllm`; python3 said: {}",
                    stderr.trim()
                ),
            }));
        }

        self.python_binary = Some(python);
        self.config = Some(config.clone());
        Ok(())
    }

    /// Record the model path for use during inference.
    ///
    /// Returns [`RuntimeError::ModelLoadFailed`] when the model file does not
    /// exist on disk.
    fn load_model(&mut self, model: &ModelInfo) -> Result<()> {
        if !model.path.exists() {
            return Err(anyhow::anyhow!(RuntimeError::ModelLoadFailed {
                path: model.path.display().to_string(),
                reason: "file not found".to_string(),
            }));
        }

        self.model = Some(model.clone());
        Ok(())
    }

    /// Run one inference pass and return performance metrics.
    ///
    /// Shells out to:
    /// ```text
    /// python3 -c "
    ///   import time, json
    ///   from vllm import LLM, SamplingParams
    ///   llm = LLM(model='<path>')
    ///   params = SamplingParams(max_tokens=50)
    ///   t0 = time.perf_counter()
    ///   out = llm.generate(['<prompt>'], params)
    ///   elapsed = time.perf_counter() - t0
    ///   tokens = sum(len(o.outputs[0].token_ids) for o in out)
    ///   print(json.dumps({'tokens': tokens, 'elapsed_s': elapsed}))
    /// "
    /// ```
    /// and parses the JSON result.
    fn run_inference(&self, prompt: &str) -> Result<BenchmarkResult> {
        let python = self.python_binary.as_ref().ok_or_else(|| {
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

        let model_path = model.path.to_string_lossy();

        // Map quantization strategy to vLLM dtype string.
        let dtype = quant_to_vllm_dtype(&config.quantization);

        // Build an inline Python script that runs vLLM and outputs JSON metrics.
        // We escape the prompt to prevent shell injection — the prompt is passed
        // as a hardcoded string literal in the script rather than via argv.
        let escaped_prompt = prompt.replace('\\', "\\\\").replace('"', "\\\"");
        let script = format!(
            r#"
import time, json, sys
try:
    from vllm import LLM, SamplingParams
    llm = LLM(model="{model_path}", dtype="{dtype}", max_model_len=512)
    params = SamplingParams(max_tokens=50)
    t0 = time.perf_counter()
    out = llm.generate(["{escaped_prompt}"], params)
    elapsed_s = time.perf_counter() - t0
    tokens = sum(len(o.outputs[0].token_ids) for o in out)
    tps = tokens / elapsed_s if elapsed_s > 0 else 0.0
    print(json.dumps({{"tokens": tokens, "elapsed_s": elapsed_s, "tps": tps}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}), file=sys.stderr)
    sys.exit(1)
"#,
            model_path = model_path,
            dtype = dtype,
            escaped_prompt = escaped_prompt,
        );

        let output = Command::new(python)
            .args(["-c", &script])
            .output()
            .map_err(|e| {
                anyhow::anyhow!(RuntimeError::InferenceFailed {
                    reason: e.to_string(),
                })
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!(RuntimeError::InferenceFailed {
                reason: format!("vLLM inference failed: {}", stderr.trim()),
            }));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let (tokens_per_sec, latency_ms) = parse_vllm_output(stdout.trim()).ok_or_else(|| {
            anyhow::anyhow!(RuntimeError::InferenceFailed {
                reason: format!(
                    "could not parse vLLM JSON output: {stdout}",
                    stdout = stdout.trim()
                ),
            })
        })?;

        Ok(BenchmarkResult {
            tokens_per_sec,
            latency_ms,
            peak_memory_mb: estimate_memory_mb(model, config),
        })
    }

    /// No-op — vLLM's Python process exits when the script completes.
    fn teardown(&mut self) -> Result<()> {
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Pure helpers (pub for unit testing)
// ──────────────────────────────────────────────────────────────────────────────

/// Parse the JSON output line produced by the inline vLLM script.
///
/// Expected format:
/// ```json
/// {"tokens": 48, "elapsed_s": 1.47, "tps": 32.65}
/// ```
pub fn parse_vllm_output(json_str: &str) -> Option<(f64, f64)> {
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let tps = v["tps"].as_f64()?;
    let elapsed_s = v["elapsed_s"].as_f64()?;
    let latency_ms = elapsed_s * 1_000.0;
    // Guard against degenerate zero values that indicate a failed run.
    if tps <= 0.0 || latency_ms <= 0.0 {
        return None;
    }
    Some((tps, latency_ms))
}

/// Map a [`QuantizationStrategy`] to the vLLM `dtype` argument.
///
/// vLLM accepts `"float16"`, `"bfloat16"`, and `"float8_e4m3fn"` (8-bit).
/// Aggressive GGUF quantizations (Q4_*) are not natively supported by vLLM,
/// so they fall back to `"float16"` — the optimizer's search.rs restricts
/// vLLM candidates to compatible formats.
pub fn quant_to_vllm_dtype(q: &QuantizationStrategy) -> &'static str {
    match q {
        QuantizationStrategy::F16 => "float16",
        QuantizationStrategy::Q8_0 | QuantizationStrategy::Int8 => "float8_e4m3fn",
        // Q4 and Int4 fall back to float16; vLLM does not support sub-8-bit.
        QuantizationStrategy::Q4_K_M | QuantizationStrategy::Q4_0 | QuantizationStrategy::Int4 => {
            "float16"
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

fn estimate_memory_mb(model: &ModelInfo, config: &RuntimeConfig) -> u64 {
    let params = match model.param_count {
        Some(p) => p,
        None => return 0,
    };
    // vLLM defaults to F16; map quantization to bytes per parameter.
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
    use vectorprime_core::{
        ModelFormat, ModelInfo, QuantizationStrategy, RuntimeConfig, RuntimeKind,
    };

    fn sample_config() -> RuntimeConfig {
        RuntimeConfig {
            runtime: RuntimeKind::Vllm,
            quantization: QuantizationStrategy::F16,
            threads: 4,
            batch_size: 512,
            gpu_layers: 0,
        }
    }

    fn sample_model() -> ModelInfo {
        ModelInfo {
            path: PathBuf::from("/tmp/model_nonexistent.onnx"),
            format: ModelFormat::ONNX,
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

    // ── parse_vllm_output ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_vllm_output_valid() {
        let json = r#"{"tokens": 48, "elapsed_s": 1.47, "tps": 32.65}"#;
        let (tps, lat) = parse_vllm_output(json).expect("should parse");
        assert!((tps - 32.65).abs() < 0.01, "tps={tps}");
        assert!((lat - 1470.0).abs() < 1.0, "latency={lat}");
    }

    #[test]
    fn test_parse_vllm_output_zero_tps_is_none() {
        let json = r#"{"tokens": 0, "elapsed_s": 1.0, "tps": 0.0}"#;
        assert!(parse_vllm_output(json).is_none());
    }

    #[test]
    fn test_parse_vllm_output_invalid() {
        assert!(parse_vllm_output("not json").is_none());
        assert!(parse_vllm_output("").is_none());
    }

    // ── quant_to_vllm_dtype mappings ──────────────────────────────────────────

    #[test]
    fn test_quant_to_vllm_dtype_f16() {
        assert_eq!(quant_to_vllm_dtype(&QuantizationStrategy::F16), "float16");
    }

    #[test]
    fn test_quant_to_vllm_dtype_int8() {
        assert_eq!(
            quant_to_vllm_dtype(&QuantizationStrategy::Int8),
            "float8_e4m3fn"
        );
        assert_eq!(
            quant_to_vllm_dtype(&QuantizationStrategy::Q8_0),
            "float8_e4m3fn"
        );
    }

    // ── initialize when binary absent ─────────────────────────────────────────

    #[test]
    fn test_dispatch_not_installed_vllm() {
        // When either python3 is absent or vllm is not importable, initialize
        // must return NotInstalled, not panic.
        //
        // If python3 is present AND vllm is installed, this test is skipped.
        let python_present = which::which("python3").is_ok();
        if python_present {
            // Check whether vllm is actually importable before deciding to skip.
            let vllm_ok = std::process::Command::new("python3")
                .args(["-c", "import vllm"])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            if vllm_ok {
                // vLLM is installed — cannot test not-installed path.
                return;
            }
        }
        let mut adapter = VllmAdapter::new();
        let err = adapter.initialize(&sample_config()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not found") || msg.contains("python3") || msg.contains("vllm"),
            "unexpected error: {msg}"
        );
    }

    // ── load_model file existence check ──────────────────────────────────────

    #[test]
    fn test_load_model_missing_file() {
        let mut adapter = VllmAdapter::new();
        adapter.python_binary = Some(PathBuf::from("/usr/bin/python3"));
        adapter.config = Some(sample_config());

        let model = sample_model(); // /tmp/model_nonexistent.onnx should not exist
        if model.path.exists() {
            return; // file happens to exist — skip
        }
        let err = adapter.load_model(&model).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not found") || msg.contains("file"),
            "unexpected error: {msg}"
        );
    }
}
