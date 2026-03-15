// crates/vectorprime-bindings/src/lib.rs
//
// PyO3 bindings for VectorPrime — exposes the Rust backend to Python as the
// `_vectorprime` native extension module.
//
// Usage:
//   Built via `maturin develop` and imported by `python/vectorprime/__init__.py`.
//   Python callers use the public classes and free functions defined here.
//
// Crate dependency chain resolved here:
//   vectorprime_hardware  -> profile()
//   vectorprime_model_ir  -> parse_model() / analyze_model()
//   vectorprime_optimizer -> run_optimization()
//   vectorprime_export    -> export_ollama()
//   vectorprime_core      -> shared types (HardwareProfile, OptimizationResult, …)
//   gpu_lookup         -> lookup_gpu() (resolves --gpu CLI flag to GpuInfo)

mod gpu_lookup;

use std::path::{Path, PathBuf};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use vectorprime_core::{HardwareProfile, ModelFormat, ModelInfo, OptimizationResult};
use vectorprime_model_ir::parse_model;
use vectorprime_runtime::{gguf_to_onnx, onnx_to_gguf};

// ──────────────────────────────────────────────────────────────────────────────
// Error conversion helper
// ──────────────────────────────────────────────────────────────────────────────

/// Convert any `anyhow::Error` into a Python `RuntimeError`.
fn to_py_err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{:#}", e))
}

// ──────────────────────────────────────────────────────────────────────────────
// PyHardwareProfile
// ──────────────────────────────────────────────────────────────────────────────

/// Python-facing wrapper around [`HardwareProfile`].
///
/// Obtained via `PyHardwareProfile.detect()` or `profile_hardware()`.
#[pyclass(name = "HardwareProfile")]
pub struct PyHardwareProfile {
    inner: HardwareProfile,
}

#[pymethods]
impl PyHardwareProfile {
    /// Detect the current machine's hardware and return a profile.
    #[staticmethod]
    fn detect() -> PyResult<Self> {
        Ok(PyHardwareProfile {
            inner: vectorprime_hardware::profile(),
        })
    }

    /// Serialize the profile to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("serialization failed: {e}")))
    }

    fn __repr__(&self) -> String {
        let gpu = self
            .inner
            .gpu
            .as_ref()
            .map(|g| g.name.clone())
            .unwrap_or_else(|| "None".to_string());
        format!(
            "HardwareProfile(cpu_cores={}, gpu={})",
            self.inner.cpu.core_count, gpu
        )
    }

    /// Number of logical CPU cores detected.
    #[getter]
    fn cpu_cores(&self) -> usize {
        self.inner.cpu.core_count as usize
    }

    /// GPU model name, or `None` if no GPU was detected.
    #[getter]
    fn gpu_model(&self) -> Option<String> {
        self.inner.gpu.as_ref().map(|g| g.name.clone())
    }

    /// GPU VRAM in megabytes, or `None` if no GPU was detected.
    #[getter]
    fn gpu_vram_mb(&self) -> Option<u64> {
        self.inner.gpu.as_ref().map(|g| g.vram_mb)
    }

    /// Total system RAM in megabytes.
    #[getter]
    fn ram_total_mb(&self) -> u64 {
        self.inner.ram.total_mb
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PyOptimizationResult
// ──────────────────────────────────────────────────────────────────────────────

/// Python-facing wrapper around [`OptimizationResult`].
///
/// Returned by `optimize()`. Contains both the benchmark result and the path
/// to the re-quantized model produced by `llama-quantize`.
#[pyclass(name = "OptimizationResult")]
pub struct PyOptimizationResult {
    inner: OptimizationResult,
    /// Original model path — needed for `export_ollama` downstream.
    model_path: PathBuf,
    /// Path to the re-quantized output model, or `None` if quantization was
    /// skipped (e.g. because `llama-quantize` is not installed).
    output_path: Option<PathBuf>,
}

#[pymethods]
impl PyOptimizationResult {
    /// Serialize the result to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("serialization failed: {e}")))
    }

    fn __repr__(&self) -> String {
        let cfg = &self.inner.config;
        let tps = self.inner.metrics.tokens_per_sec;
        format!(
            "OptimizationResult(runtime={:?}, quant={:?}, tps={:.1})",
            cfg.runtime, cfg.quantization, tps
        )
    }

    /// Runtime name (e.g. `"LlamaCpp"`, `"OnnxRuntime"`, `"TensorRT"`).
    #[getter]
    fn runtime(&self) -> String {
        format!("{:?}", self.inner.config.runtime)
    }

    /// Quantization strategy name (e.g. `"Q4_K_M"`, `"F16"`).
    #[getter]
    fn quantization(&self) -> String {
        format!("{:?}", self.inner.config.quantization)
    }

    /// Number of CPU threads used by the selected configuration.
    #[getter]
    fn threads(&self) -> usize {
        self.inner.config.threads as usize
    }

    /// Number of transformer layers offloaded to the GPU (0 = CPU-only).
    #[getter]
    fn gpu_layers(&self) -> u32 {
        self.inner.config.gpu_layers
    }

    /// Sustained throughput in tokens per second.
    #[getter]
    fn tokens_per_sec(&self) -> f64 {
        self.inner.metrics.tokens_per_sec
    }

    /// End-to-end inference latency in milliseconds.
    #[getter]
    fn latency_ms(&self) -> f64 {
        self.inner.metrics.latency_ms
    }

    /// Peak memory usage during the benchmark run, in megabytes.
    #[getter]
    fn peak_memory_mb(&self) -> u64 {
        self.inner.metrics.peak_memory_mb
    }

    /// Path to the re-quantized output model, or `None` if `llama-quantize`
    /// was not available and quantization was skipped.
    #[getter]
    fn output_path(&self) -> Option<String> {
        self.output_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Module-level free functions
// ──────────────────────────────────────────────────────────────────────────────

/// Detect and return the current machine's hardware profile.
///
/// Equivalent to `HardwareProfile.detect()`.
#[pyfunction]
fn profile_hardware() -> PyResult<PyHardwareProfile> {
    Ok(PyHardwareProfile {
        inner: vectorprime_hardware::profile(),
    })
}

/// Parse a format string into a [`ModelFormat`].
fn parse_model_format(format: &str) -> PyResult<ModelFormat> {
    match format.to_ascii_lowercase().as_str() {
        "onnx" => Ok(ModelFormat::ONNX),
        "gguf" => Ok(ModelFormat::GGUF),
        other => Err(PyRuntimeError::new_err(format!(
            "unknown model format '{other}'; expected 'onnx' or 'gguf'"
        ))),
    }
}

/// Run the full optimization pipeline for the given model.
///
/// After finding the best configuration, re-quantizes the model using
/// `llama-quantize` (if available) and writes `{stem}-optimized.gguf` next to
/// the input file (or to `output_path` when supplied).
///
/// Parameters
/// ----------
/// model_path : str
///     Absolute or relative path to the model file (`.gguf` or `.onnx`).
/// format : str
///     Model format: `"gguf"` or `"onnx"` (case-insensitive).
/// gpu : str, optional
///     Target GPU model string (e.g. `"4090"`, `"RTX 3090"`, `"a100"`).
///     Pass `"cpu"` or omit to force CPU-only mode.  When provided this
///     value **completely replaces** the auto-detected GPU in the hardware
///     profile so that the optimizer plans for the specified hardware.
/// max_latency_ms : float, optional
///     Maximum tolerated inference latency in milliseconds. Configurations
///     whose measured `latency_ms` exceeds this value are excluded.
///     Omit (or pass `None`) to apply no latency constraint.
/// output_path : str, optional
///     Destination path for the re-quantized `.gguf` file.  When omitted the
///     output is placed next to the input as `{stem}-optimized.gguf`.
///
/// Returns
/// -------
/// OptimizationResult
///     The best runtime configuration found for this hardware.  The
///     `output_path` attribute contains the path to the re-quantized model,
///     or `None` if `llama-quantize` was not available.
///
/// Raises
/// ------
/// RuntimeError
///     If no valid configuration could be benchmarked, the path is invalid,
///     or an unrecognised GPU model string is supplied.
#[pyfunction]
#[pyo3(signature = (model_path, format, gpu=None, max_latency_ms=None, output_path=None, use_cache=true))]
fn optimize(
    model_path: &str,
    format: &str,
    gpu: Option<String>,
    max_latency_ms: Option<f64>,
    output_path: Option<String>,
    use_cache: bool,
) -> PyResult<PyOptimizationResult> {
    let fmt = parse_model_format(format)?;
    let path = PathBuf::from(model_path);

    // Attempt IR analysis to populate param_count for smarter candidate generation.
    // Errors are recovered gracefully — a parse failure must never abort optimize().
    let model_ir = parse_model(&path).ok();
    let param_count = model_ir.as_ref().and_then(|ir| ir.param_count);

    let model = ModelInfo {
        path: path.clone(),
        format: fmt,
        param_count,
        hidden_size: model_ir.as_ref().and_then(|ir| ir.hidden_size),
        attention_head_count: model_ir.as_ref().and_then(|ir| ir.attention_head_count),
        attention_head_count_kv: model_ir.as_ref().and_then(|ir| ir.attention_head_count_kv),
        feed_forward_length: model_ir.as_ref().and_then(|ir| ir.feed_forward_length),
        kv_cache_size_mb: model_ir.as_ref().and_then(|ir| ir.kv_cache_size_mb),
        memory_footprint_mb: model_ir.as_ref().and_then(|ir| ir.memory_footprint_mb),
        flops_per_token: model_ir.as_ref().and_then(|ir| ir.flops_per_token),
    };

    // Auto-detect host hardware, then apply the optional GPU override.
    let mut hw = vectorprime_hardware::profile();

    // When the caller supplies *any* --gpu value (including "cpu" or ""),
    // resolve it and completely replace the auto-detected GPU field.
    // `None` here means --gpu was not passed at all — leave hw.gpu unchanged.
    if let Some(ref gpu_str) = gpu {
        let resolved =
            gpu_lookup::lookup_gpu(Some(gpu_str.as_str())).map_err(PyRuntimeError::new_err)?;
        // Replace detected GPU unconditionally; resolved may be None (CPU-only).
        hw.gpu = resolved;
    }

    // The optimizer is async; we must block here because PyO3 does not
    // support `async fn` free functions without additional plumbing.
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create tokio runtime: {e}")))?;

    let result = rt
        .block_on(vectorprime_optimizer::run_optimization(
            model,
            hw,
            max_latency_ms,
            use_cache,
        ))
        .map_err(to_py_err)?;

    // Derive the output path: either the caller-supplied value, or
    // `{stem}-optimized.gguf` placed next to the input file.
    let derived_output = derive_output_path(&path, output_path.as_deref());

    // Attempt re-quantization; if llama-quantize is absent or fails we degrade
    // gracefully rather than failing the whole optimization, but we surface the
    // error so the user can see the actual failure reason.
    let quantized_path = match vectorprime_export::quantize_gguf(
        &path,
        &derived_output,
        &result.config.quantization,
    ) {
        Ok(()) => Some(derived_output),
        Err(e) => {
            eprintln!("[vectorprime] Warning: quantization failed: {e}");
            None
        }
    };

    Ok(PyOptimizationResult {
        inner: result,
        model_path: path,
        output_path: quantized_path,
    })
}

/// Compute the output path for the re-quantized model.
///
/// Uses the caller-supplied path when present; otherwise derives
/// `{stem}-optimized.gguf` in the same directory as `input`.
fn derive_output_path(input: &Path, output_path: Option<&str>) -> PathBuf {
    if let Some(p) = output_path {
        return PathBuf::from(p);
    }

    // Derive {stem}-optimized.gguf next to the input, or in CWD if no parent.
    let stem = input
        .file_stem()
        .unwrap_or_else(|| std::ffi::OsStr::new("model"))
        .to_string_lossy();
    let file_name = format!("{stem}-optimized.gguf");

    match input.parent() {
        Some(parent) if parent != Path::new("") => parent.join(file_name),
        _ => PathBuf::from(file_name),
    }
}

/// Export an optimized model as an Ollama-compatible bundle.
///
/// Parameters
/// ----------
/// result : OptimizationResult
///     The result produced by `optimize()`.
/// output_dir : str
///     Directory where the Ollama bundle (`Modelfile`, `model.gguf`,
///     `metadata.json`) will be written. Created if it does not exist.
///
/// Returns
/// -------
/// str
///     JSON-serialized [`ExportManifest`] describing produced files and
///     the `ollama create` / `ollama run` commands.
///
/// Raises
/// ------
/// RuntimeError
///     If the export fails (e.g. missing conversion script for ONNX models).
#[pyfunction]
fn export_ollama(result: &PyOptimizationResult, output_dir: &str) -> PyResult<String> {
    let out = Path::new(output_dir);

    let manifest = vectorprime_export::export_ollama(&result.inner, &result.model_path, out)
        .map_err(to_py_err)?;

    // ExportManifest does not implement Serialize, so we build a JSON value
    // from its fields manually.
    let json = serde_json::json!({
        "output_dir": manifest.output_dir.to_string_lossy(),
        "modelfile_path": manifest.modelfile_path.to_string_lossy(),
        "model_gguf_path": manifest.model_gguf_path.to_string_lossy(),
        "ollama_commands": manifest.ollama_commands,
    });

    serde_json::to_string(&json)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to serialize export manifest: {e}")))
}

/// Inspect a model file and return its IR metadata as a Python dictionary.
///
/// Reads only the file header and metadata section; tensor data is never
/// loaded.  All fields except `format` may be `None` when the metadata is
/// absent or uncomputable.
///
/// Parameters
/// ----------
/// model_path : str
///     Path to the model file (`.gguf` or `.onnx`).
///
/// Returns
/// -------
/// dict
///     A dictionary with the following keys:
///
///     - ``format`` (`str`): ``"gguf"`` or ``"onnx"``
///     - ``param_count`` (`int | None`): total parameter count, if known
///     - ``architecture`` (`str | None`): architecture family (e.g. ``"llama"``)
///     - ``context_length`` (`int | None`): maximum context length
///     - ``layer_count`` (`int | None`): number of transformer blocks
///
/// Raises
/// ------
/// RuntimeError
///     If the file cannot be opened, has an unrecognised extension, or the
///     magic bytes are invalid.
#[pyfunction]
fn analyze_model(py: Python<'_>, model_path: &str) -> PyResult<PyObject> {
    let path = std::path::Path::new(model_path);

    let ir = parse_model(path).map_err(to_py_err)?;

    let dict = pyo3::types::PyDict::new(py);
    let format_str = match ir.format {
        ModelFormat::GGUF => "gguf",
        ModelFormat::ONNX => "onnx",
    };
    dict.set_item("format", format_str)?;
    dict.set_item("param_count", ir.param_count)?;
    dict.set_item("architecture", ir.architecture)?;
    dict.set_item("context_length", ir.context_length)?;
    dict.set_item("layer_count", ir.layer_count)?;
    dict.set_item("hidden_size", ir.hidden_size)?;
    dict.set_item("attention_head_count", ir.attention_head_count)?;
    dict.set_item("attention_head_count_kv", ir.attention_head_count_kv)?;
    dict.set_item("feed_forward_length", ir.feed_forward_length)?;
    dict.set_item("kv_cache_size_mb", ir.kv_cache_size_mb)?;
    dict.set_item("memory_footprint_mb", ir.memory_footprint_mb)?;
    dict.set_item("flops_per_token", ir.flops_per_token)?;

    Ok(dict.into())
}

// ──────────────────────────────────────────────────────────────────────────────
// Conversion functions
// ──────────────────────────────────────────────────────────────────────────────

/// Convert a GGUF model file to ONNX format.
///
/// Parameters
/// ----------
/// input_path : str
///     Path to the source `.gguf` file.
/// output_path : str
///     Destination path for the produced `.onnx` file.
///
/// Returns
/// -------
/// str
///     Absolute path to the written `.onnx` file.
///
/// Raises
/// ------
/// RuntimeError
///     If `python3` is not found, required packages (`gguf`, `onnx`,
///     `numpy`) are missing, or the input is not a valid GGUF file.
#[pyfunction]
fn convert_gguf_to_onnx(input_path: &str, output_path: &str) -> PyResult<String> {
    let out = gguf_to_onnx(
        std::path::Path::new(input_path),
        std::path::Path::new(output_path),
    )
    .map_err(to_py_err)?;
    Ok(out.to_string_lossy().into_owned())
}

/// Convert an ONNX model file to GGUF format.
///
/// Parameters
/// ----------
/// input_path : str
///     Path to the source `.onnx` file.
/// output_path : str
///     Destination path for the produced `.gguf` file.
///
/// Returns
/// -------
/// str
///     Absolute path to the written `.gguf` file.
///
/// Raises
/// ------
/// RuntimeError
///     If `python3` is not found, required packages (`onnx`, `gguf`,
///     `numpy`) are missing, or the input is not a valid ONNX file.
#[pyfunction]
fn convert_onnx_to_gguf(input_path: &str, output_path: &str) -> PyResult<String> {
    let out = onnx_to_gguf(
        std::path::Path::new(input_path),
        std::path::Path::new(output_path),
    )
    .map_err(to_py_err)?;
    Ok(out.to_string_lossy().into_owned())
}

// ──────────────────────────────────────────────────────────────────────────────
// Module registration
// ──────────────────────────────────────────────────────────────────────────────

/// Native Rust extension module for VectorPrime.
///
/// Exposes hardware profiling, optimization, and Ollama export to Python.
/// Imported as `from vectorprime._vectorprime import ...` (or via `vectorprime`'s
/// re-exports in `__init__.py`).
#[pymodule]
fn _vectorprime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHardwareProfile>()?;
    m.add_class::<PyOptimizationResult>()?;
    m.add_function(wrap_pyfunction!(profile_hardware, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_function(wrap_pyfunction!(export_ollama, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_model, m)?)?;
    m.add_function(wrap_pyfunction!(convert_gguf_to_onnx, m)?)?;
    m.add_function(wrap_pyfunction!(convert_onnx_to_gguf, m)?)?;
    Ok(())
}
