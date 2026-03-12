//! Model format conversion utilities.
//!
//! Provides [`gguf_to_onnx`] and [`onnx_to_gguf`], which shell out to
//! bundled Python runner scripts — following the same pattern as
//! [`crate::onnx::OnnxAdapter`].
//!
//! **Required Python packages** (installed in the active Python environment):
//!
//! | Conversion | Packages |
//! |------------|----------|
//! | GGUF → ONNX | `gguf`, `onnx`, `numpy` |
//! | ONNX → GGUF | `onnx`, `gguf`, `numpy` |
//!
//! If `python3` is not found, or the runner script is missing, both functions
//! return an error with an actionable install hint.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Result};
use serde::Deserialize;

use vectorprime_core::RuntimeError;

const GGUF_TO_ONNX_RUNNER: &str = "python/vectorprime/gguf_to_onnx_runner.py";
const ONNX_TO_GGUF_RUNNER: &str = "python/vectorprime/onnx_to_gguf_runner.py";

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Shared response envelope used by both runner scripts.
#[derive(Deserialize)]
struct ConvertResponse {
    output_path: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

/// Locate a Python runner script using all available strategies.
///
/// Search order (first match wins):
///
/// 1. Next to the current executable — catches future installed layouts.
/// 2. Walk up from the current working directory — works when running the
///    CLI from the workspace root during development (the common case).
/// 3. Walk up from `CARGO_MANIFEST_DIR` — set by cargo during `cargo test`
///    and `cargo build`; not available at runtime.
/// 4. Ask Python where the `vectorprime` package is installed — the most robust
///    path for `pip install` / `maturin develop` deployments.
fn find_runner(workspace_rel_path: &str) -> Option<PathBuf> {
    let script_name = workspace_rel_path
        .rsplit('/')
        .next()
        .unwrap_or(workspace_rel_path);

    // 1. Next to the current executable (installed / maturin layout).
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let candidate = parent.join("python").join("vectorprime").join(script_name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    // 2. Walk up from the current working directory.
    //    This covers running `vectorprime …` from inside the workspace root.
    if let Ok(cwd) = std::env::current_dir() {
        let mut dir = cwd;
        for _ in 0..6 {
            let candidate = dir.join(workspace_rel_path);
            if candidate.exists() {
                return Some(candidate);
            }
            if !dir.pop() {
                break;
            }
        }
    }

    // 3. Walk up from CARGO_MANIFEST_DIR (set only during cargo test/build).
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let mut dir = PathBuf::from(manifest);
        for _ in 0..5 {
            let candidate = dir.join(workspace_rel_path);
            if candidate.exists() {
                return Some(candidate);
            }
            if !dir.pop() {
                break;
            }
        }
    }

    // 4. Ask Python where the installed `vectorprime` package lives.
    //    Works for `pip install` and `maturin develop` deployments where the
    //    runner scripts sit alongside `__init__.py` in site-packages.
    if let Ok(python) = which::which("python3") {
        let snippet = format!(
            "import vectorprime, os; print(os.path.join(os.path.dirname(vectorprime.__file__), '{}'))",
            script_name
        );
        if let Ok(out) = std::process::Command::new(&python)
            .args(["-c", &snippet])
            .output()
        {
            if out.status.success() {
                let path_str = String::from_utf8_lossy(&out.stdout);
                let candidate = PathBuf::from(path_str.trim());
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

/// Spawn a conversion runner, send `request` as JSON on stdin, parse stdout.
fn run_converter(python: &Path, runner: &Path, request: serde_json::Value) -> Result<PathBuf> {
    let request_str = serde_json::to_string(&request)?;

    let mut child = Command::new(python)
        .arg(runner)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to spawn conversion process: {e}"))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(request_str.as_bytes())?;
    }

    let out = child.wait_with_output()?;

    let response: ConvertResponse = serde_json::from_slice(&out.stdout).map_err(|_| {
        let stderr = String::from_utf8_lossy(&out.stderr);
        anyhow::anyhow!(
            "conversion process returned invalid output (stderr: {})",
            stderr.trim()
        )
    })?;

    if let Some(err) = response.error {
        bail!("{err}");
    }

    response
        .output_path
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("conversion succeeded but no output_path was returned"))
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Convert a GGUF model file to ONNX format.
///
/// Shells out to the bundled `gguf_to_onnx_runner.py` script. The runner
/// dequantizes all weight tensors to `float32` and writes them as ONNX
/// initializers. Model metadata (architecture, tokenizer, hyperparameters) is
/// preserved in the `doc_string` field of the ONNX model.
///
/// # Parameters
///
/// * `input`  — path to the source `.gguf` file.
/// * `output` — destination path for the produced `.onnx` file.
///
/// # Errors
///
/// Returns an error if:
/// - `python3` is not found on `PATH`.
/// - The `gguf_to_onnx_runner.py` script cannot be located.
/// - Required Python packages (`gguf`, `onnx`, `numpy`) are not installed.
/// - The input file is not a valid GGUF file.
pub fn gguf_to_onnx(input: &Path, output: &Path) -> Result<PathBuf> {
    let python = which::which("python3").map_err(|_| {
        anyhow::anyhow!(RuntimeError::NotInstalled {
            binary: "python3".to_string(),
            install_hint: "install Python 3 from https://python.org".to_string(),
        })
    })?;

    let runner = find_runner(GGUF_TO_ONNX_RUNNER).ok_or_else(|| {
        anyhow::anyhow!(
            "gguf_to_onnx_runner.py not found — ensure vectorprime is properly installed"
        )
    })?;

    let request = serde_json::json!({
        "input_path":  input.to_string_lossy(),
        "output_path": output.to_string_lossy(),
    });

    run_converter(&python, &runner, request)
}

/// Convert an ONNX model file to GGUF format.
///
/// Shells out to the bundled `onnx_to_gguf_runner.py` script. The runner
/// extracts all weight initializers from the ONNX graph and writes them as
/// `F32` tensors in a new GGUF file. Architecture metadata stored in the
/// ONNX `doc_string` (as JSON) is round-tripped back into GGUF key/value
/// fields when present.
///
/// # Parameters
///
/// * `input`  — path to the source `.onnx` file.
/// * `output` — destination path for the produced `.gguf` file.
///
/// # Errors
///
/// Returns an error if:
/// - `python3` is not found on `PATH`.
/// - The `onnx_to_gguf_runner.py` script cannot be located.
/// - Required Python packages (`onnx`, `gguf`, `numpy`) are not installed.
/// - The input file is not a valid ONNX model.
pub fn onnx_to_gguf(input: &Path, output: &Path) -> Result<PathBuf> {
    let python = which::which("python3").map_err(|_| {
        anyhow::anyhow!(RuntimeError::NotInstalled {
            binary: "python3".to_string(),
            install_hint: "install Python 3 from https://python.org".to_string(),
        })
    })?;

    let runner = find_runner(ONNX_TO_GGUF_RUNNER).ok_or_else(|| {
        anyhow::anyhow!(
            "onnx_to_gguf_runner.py not found — ensure vectorprime is properly installed"
        )
    })?;

    let request = serde_json::json!({
        "input_path":  input.to_string_lossy(),
        "output_path": output.to_string_lossy(),
    });

    run_converter(&python, &runner, request)
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// `find_runner` must be able to locate the bundled scripts from the dev
    /// layout (CARGO_MANIFEST_DIR is set by `cargo test`).
    #[test]
    fn find_gguf_to_onnx_runner() {
        assert!(
            find_runner(GGUF_TO_ONNX_RUNNER).is_some(),
            "gguf_to_onnx_runner.py not found from CARGO_MANIFEST_DIR"
        );
    }

    #[test]
    fn find_onnx_to_gguf_runner() {
        assert!(
            find_runner(ONNX_TO_GGUF_RUNNER).is_some(),
            "onnx_to_gguf_runner.py not found from CARGO_MANIFEST_DIR"
        );
    }

    /// `gguf_to_onnx` on a non-existent file must return an error, not panic.
    #[test]
    fn gguf_to_onnx_missing_file_returns_error() {
        let result = gguf_to_onnx(
            Path::new("/nonexistent/model.gguf"),
            Path::new("/tmp/out.onnx"),
        );
        assert!(result.is_err(), "expected an error for a missing GGUF file");
    }

    /// `onnx_to_gguf` on a non-existent file must return an error, not panic.
    #[test]
    fn onnx_to_gguf_missing_file_returns_error() {
        let result = onnx_to_gguf(
            Path::new("/nonexistent/model.onnx"),
            Path::new("/tmp/out.gguf"),
        );
        assert!(result.is_err(), "expected an error for a missing ONNX file");
    }
}
