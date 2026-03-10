use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;
use llmforge_core::{OptimizationResult, QuantizationStrategy};

/// Describes the files and commands produced by [`export_ollama`].
#[derive(Debug)]
pub struct ExportManifest {
    pub output_dir: PathBuf,
    pub modelfile_path: PathBuf,
    pub model_gguf_path: PathBuf,
    /// Ready-to-run Ollama commands in suggested execution order.
    pub ollama_commands: Vec<String>,
}

/// Export an optimized model as an Ollama-compatible bundle.
///
/// Creates `output_dir` (and any missing parents), writes `Modelfile`,
/// `model.gguf`, and `metadata.json`, then returns an [`ExportManifest`]
/// describing what was produced.
pub fn export_ollama(
    result: &OptimizationResult,
    model_path: &Path,
    output_dir: &Path,
) -> Result<ExportManifest> {
    // 1. Create output directory.
    std::fs::create_dir_all(output_dir)?;

    // 2. Resolve GGUF path.
    let gguf_src = resolve_gguf(model_path, output_dir)?;

    // 3. Copy GGUF to output_dir/model.gguf (skip if already in place).
    let dest_gguf = output_dir.join("model.gguf");
    if gguf_src != dest_gguf {
        std::fs::copy(&gguf_src, &dest_gguf)?;
    }

    // 4. Write Modelfile.
    let modelfile_path = output_dir.join("Modelfile");
    let modelfile = build_modelfile(result);
    std::fs::write(&modelfile_path, &modelfile)?;

    // 5. Write metadata.json.
    let metadata_path = output_dir.join("metadata.json");
    let json = serde_json::to_string_pretty(result)?;
    std::fs::write(&metadata_path, json)?;

    // 6. Build manifest.
    let manifest = ExportManifest {
        ollama_commands: vec![
            format!("ollama create mymodel -f {}", modelfile_path.display()),
            "ollama run mymodel".to_string(),
        ],
        output_dir: output_dir.to_path_buf(),
        modelfile_path,
        model_gguf_path: dest_gguf,
    };

    Ok(manifest)
}

/// Re-quantize a GGUF model file using `llama-quantize`.
///
/// Maps `quant` to the string type argument expected by `llama-quantize`
/// (e.g. `Q4_K_M` → `"q4_k_m"`, `F16` → `"f16"`), then shells out to:
///
/// ```text
/// llama-quantize <input> <output> <type>
/// ```
///
/// Returns an error if `llama-quantize` is not in PATH or exits non-zero.
pub fn quantize_gguf(input: &Path, output: &Path, quant: &QuantizationStrategy) -> Result<()> {
    // Check that llama-quantize is available before invoking it.
    which::which("llama-quantize").map_err(|_| {
        anyhow::anyhow!("llama-quantize not found — install llama.cpp to enable model quantization")
    })?;

    let quant_type = quant_to_llama_quantize_type(quant);

    let status = Command::new("llama-quantize")
        .arg(input)
        .arg(output)
        .arg(quant_type)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to launch llama-quantize: {e}"))?;

    if !status.success() {
        return Err(anyhow::anyhow!(
            "llama-quantize exited with code {:?}",
            status.code()
        ));
    }

    Ok(())
}

/// Map a [`QuantizationStrategy`] to the type string `llama-quantize` expects.
///
/// `Int8` and `Int4` fall back to the nearest GGUF equivalents (`q8_0` / `q4_0`)
/// because `llama-quantize` does not have dedicated int8/int4 type identifiers.
fn quant_to_llama_quantize_type(quant: &QuantizationStrategy) -> &'static str {
    match quant {
        QuantizationStrategy::F16 => "f16",
        QuantizationStrategy::Q8_0 => "q8_0",
        QuantizationStrategy::Q4_K_M => "q4_k_m",
        QuantizationStrategy::Q4_0 => "q4_0",
        QuantizationStrategy::Int8 => "q8_0",
        QuantizationStrategy::Int4 => "q4_0",
    }
}

/// Print a human-readable summary of an [`ExportManifest`] to stdout.
pub fn print_export_summary(manifest: &ExportManifest) {
    println!("Export directory : {}", manifest.output_dir.display());
    println!("Modelfile        : {}", manifest.modelfile_path.display());
    println!("Model (GGUF)     : {}", manifest.model_gguf_path.display());
    println!();
    println!("Run with Ollama:");
    for cmd in &manifest.ollama_commands {
        println!("  {cmd}");
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

fn resolve_gguf(model_path: &Path, output_dir: &Path) -> Result<PathBuf> {
    match model_path.extension().and_then(|e| e.to_str()) {
        Some("gguf") => Ok(model_path.to_path_buf()),
        Some("onnx") => convert_onnx_to_gguf(model_path, output_dir),
        _ => {
            // Unknown extension: try treating as GGUF and let downstream fail.
            Ok(model_path.to_path_buf())
        }
    }
}

fn convert_onnx_to_gguf(model_path: &Path, output_dir: &Path) -> Result<PathBuf> {
    let script = which::which("convert_hf_to_gguf.py").map_err(|_| {
        anyhow::anyhow!(
            "ONNX-to-GGUF conversion requires llama.cpp's \
             convert_hf_to_gguf.py on PATH"
        )
    })?;

    let dest = output_dir.join("model.gguf");
    let status = Command::new("python3")
        .args([
            script.to_str().unwrap_or("convert_hf_to_gguf.py"),
            model_path.to_str().unwrap_or(""),
            "--outfile",
            dest.to_str().unwrap_or("model.gguf"),
        ])
        .status()?;

    if !status.success() {
        return Err(anyhow::anyhow!(
            "convert_hf_to_gguf.py failed with exit code {:?}",
            status.code()
        ));
    }

    Ok(dest)
}

fn build_modelfile(result: &OptimizationResult) -> String {
    let cfg = &result.config;
    format!(
        "FROM ./model.gguf\n\
         PARAMETER num_thread {threads}\n\
         PARAMETER num_gpu {gpu_layers}\n\
         PARAMETER num_ctx 4096\n\
         # Generated by LLMForge\n",
        threads = cfg.threads,
        gpu_layers = cfg.gpu_layers,
    )
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use llmforge_core::{
        BenchmarkResult, ModelFormat, ModelInfo, OptimizationResult, QuantizationStrategy,
        RuntimeConfig, RuntimeKind,
    };

    fn sample_result() -> OptimizationResult {
        OptimizationResult {
            config: RuntimeConfig {
                runtime: RuntimeKind::LlamaCpp,
                quantization: QuantizationStrategy::Q4_K_M,
                threads: 8,
                batch_size: 512,
                gpu_layers: 20,
            },
            metrics: BenchmarkResult {
                tokens_per_sec: 110.3,
                latency_ms: 91.2,
                peak_memory_mb: 4096,
            },
        }
    }

    fn sample_model_info(path: &Path) -> ModelInfo {
        ModelInfo {
            path: path.to_path_buf(),
            format: ModelFormat::GGUF,
            param_count: Some(7_000_000_000),
        }
    }

    /// Create a minimal fake GGUF file (just needs to exist; export copies it).
    fn write_fake_gguf(path: &Path) {
        std::fs::write(path, b"GGUF\x00\x00\x00\x00").unwrap();
    }

    #[test]
    fn test_export_creates_modelfile() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("model.gguf");
        write_fake_gguf(&gguf);

        let out = dir.path().join("export");
        let _ = sample_model_info(&gguf);
        let manifest = export_ollama(&sample_result(), &gguf, &out).unwrap();

        assert!(manifest.modelfile_path.exists(), "Modelfile not created");
        assert!(manifest.model_gguf_path.exists(), "model.gguf not created");
    }

    #[test]
    fn test_modelfile_contains_from() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("model.gguf");
        write_fake_gguf(&gguf);
        let out = dir.path().join("export");

        let manifest = export_ollama(&sample_result(), &gguf, &out).unwrap();
        let content = std::fs::read_to_string(&manifest.modelfile_path).unwrap();

        assert!(content.contains("FROM ./model.gguf"), "missing FROM line");
    }

    #[test]
    fn test_threads_in_modelfile() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("model.gguf");
        write_fake_gguf(&gguf);
        let out = dir.path().join("export");

        let manifest = export_ollama(&sample_result(), &gguf, &out).unwrap();
        let content = std::fs::read_to_string(&manifest.modelfile_path).unwrap();

        // sample_result has threads=8, gpu_layers=20
        assert!(content.contains("num_thread 8"), "missing thread count");
        assert!(content.contains("num_gpu 20"), "missing gpu_layers");
    }

    #[test]
    fn test_metadata_json_valid() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("model.gguf");
        write_fake_gguf(&gguf);
        let out = dir.path().join("export");

        export_ollama(&sample_result(), &gguf, &out).unwrap();

        let json_path = out.join("metadata.json");
        let raw = std::fs::read_to_string(&json_path).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&raw).expect("metadata.json is invalid JSON");
        assert!(parsed["config"]["threads"].is_number());
    }

    #[test]
    fn test_onnx_without_converter_returns_err() {
        let dir = tempfile::tempdir().unwrap();
        // A fake .onnx file — no convert_hf_to_gguf.py on PATH.
        let onnx = dir.path().join("model.onnx");
        std::fs::write(&onnx, b"fake onnx").unwrap();
        let out = dir.path().join("export");

        if which::which("convert_hf_to_gguf.py").is_ok() {
            return; // script present — skip
        }

        let err = export_ollama(&sample_result(), &onnx, &out).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("convert_hf_to_gguf.py") || msg.contains("ONNX"),
            "expected descriptive error, got: {msg}"
        );
    }

    #[test]
    fn test_quant_to_llama_quantize_type_mappings() {
        assert_eq!(
            quant_to_llama_quantize_type(&QuantizationStrategy::F16),
            "f16"
        );
        assert_eq!(
            quant_to_llama_quantize_type(&QuantizationStrategy::Q8_0),
            "q8_0"
        );
        assert_eq!(
            quant_to_llama_quantize_type(&QuantizationStrategy::Q4_K_M),
            "q4_k_m"
        );
        assert_eq!(
            quant_to_llama_quantize_type(&QuantizationStrategy::Q4_0),
            "q4_0"
        );
        assert_eq!(
            quant_to_llama_quantize_type(&QuantizationStrategy::Int8),
            "q8_0"
        );
        assert_eq!(
            quant_to_llama_quantize_type(&QuantizationStrategy::Int4),
            "q4_0"
        );
    }

    #[test]
    fn test_quantize_gguf_not_installed_returns_error() {
        // If llama-quantize is not on PATH this should return a descriptive error.
        if which::which("llama-quantize").is_ok() {
            return; // binary present — cannot test the not-installed path
        }
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("model.gguf");
        let output = dir.path().join("model-optimized.gguf");
        write_fake_gguf(&input);

        let err = quantize_gguf(&input, &output, &QuantizationStrategy::Q4_K_M).unwrap_err();
        assert!(
            err.to_string().contains("llama-quantize"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_manifest_has_two_commands() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("model.gguf");
        write_fake_gguf(&gguf);
        let out = dir.path().join("export");

        let manifest = export_ollama(&sample_result(), &gguf, &out).unwrap();
        assert_eq!(manifest.ollama_commands.len(), 2);
        assert!(manifest.ollama_commands[0].contains("ollama create"));
        assert!(manifest.ollama_commands[1].contains("ollama run"));
    }
}
