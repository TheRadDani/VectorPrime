// crates/llmforge-bindings/src/lib.rs
//
// PyO3 bindings for LLMForge — exposes the Rust backend to Python as the
// `_llmforge` native extension module.
//
// Usage:
//   Built via `maturin develop` and imported by `python/llmforge/__init__.py`.
//   Python callers use the public classes and free functions defined here.
//
// Crate dependency chain resolved here:
//   llmforge_hardware  -> profile()
//   llmforge_optimizer -> run_optimization()
//   llmforge_export    -> export_ollama()
//   llmforge_core      -> shared types (HardwareProfile, OptimizationResult, …)

use std::path::{Path, PathBuf};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use llmforge_core::{HardwareProfile, ModelFormat, ModelInfo, OptimizationResult};

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
            inner: llmforge_hardware::profile(),
        })
    }

    /// Serialize the profile to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            PyRuntimeError::new_err(format!("serialization failed: {e}"))
        })
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
/// Returned by `optimize()`. Pass to `export_ollama()` to create an Ollama bundle.
#[pyclass(name = "OptimizationResult")]
pub struct PyOptimizationResult {
    inner: OptimizationResult,
    /// Original model path — needed for `export_ollama` downstream.
    model_path: PathBuf,
}

#[pymethods]
impl PyOptimizationResult {
    /// Serialize the result to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            PyRuntimeError::new_err(format!("serialization failed: {e}"))
        })
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
        inner: llmforge_hardware::profile(),
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
/// Parameters
/// ----------
/// model_path : str
///     Absolute or relative path to the model file (`.gguf` or `.onnx`).
/// format : str
///     Model format: `"gguf"` or `"onnx"` (case-insensitive).
///
/// Returns
/// -------
/// OptimizationResult
///     The best runtime configuration found for this hardware.
///
/// Raises
/// ------
/// RuntimeError
///     If no valid configuration could be benchmarked or the path is invalid.
#[pyfunction]
fn optimize(model_path: &str, format: &str) -> PyResult<PyOptimizationResult> {
    let fmt = parse_model_format(format)?;
    let path = PathBuf::from(model_path);

    let model = ModelInfo {
        path: path.clone(),
        format: fmt,
        param_count: None,
    };

    let hw = llmforge_hardware::profile();

    // The optimizer is async; we must block here because PyO3 does not
    // support `async fn` free functions without additional plumbing.
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create tokio runtime: {e}")))?;

    let result = rt
        .block_on(llmforge_optimizer::run_optimization(model, hw))
        .map_err(to_py_err)?;

    Ok(PyOptimizationResult {
        inner: result,
        model_path: path,
    })
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

    let manifest =
        llmforge_export::export_ollama(&result.inner, &result.model_path, out)
            .map_err(to_py_err)?;

    // ExportManifest does not implement Serialize, so we build a JSON value
    // from its fields manually.
    let json = serde_json::json!({
        "output_dir": manifest.output_dir.to_string_lossy(),
        "modelfile_path": manifest.modelfile_path.to_string_lossy(),
        "model_gguf_path": manifest.model_gguf_path.to_string_lossy(),
        "ollama_commands": manifest.ollama_commands,
    });

    serde_json::to_string(&json).map_err(|e| {
        PyRuntimeError::new_err(format!("failed to serialize export manifest: {e}"))
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Module registration
// ──────────────────────────────────────────────────────────────────────────────

/// Native Rust extension module for LLMForge.
///
/// Exposes hardware profiling, optimization, and Ollama export to Python.
/// Imported as `from llmforge._llmforge import ...` (or via `llmforge`'s
/// re-exports in `__init__.py`).
#[pymodule]
fn _llmforge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHardwareProfile>()?;
    m.add_class::<PyOptimizationResult>()?;
    m.add_function(wrap_pyfunction!(profile_hardware, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_function(wrap_pyfunction!(export_ollama, m)?)?;
    Ok(())
}
