use anyhow::Result;
use llmforge_core::{BenchmarkResult, ModelInfo, RuntimeAdapter, RuntimeConfig, RuntimeError};

/// Stub adapter for ONNX Runtime (via bundled `onnx_runner.py` + `python3` binary).
///
/// Full implementation is added in Stage 3C.
pub struct OnnxAdapter {
    #[allow(dead_code)] // populated in Stage 3C
    config: Option<RuntimeConfig>,
}

impl OnnxAdapter {
    pub fn new() -> Self {
        Self { config: None }
    }
}

impl Default for OnnxAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeAdapter for OnnxAdapter {
    fn initialize(&mut self, _config: &RuntimeConfig) -> Result<()> {
        Err(anyhow::anyhow!(RuntimeError::NotInstalled {
            binary: "python3".to_string(),
        }))
    }

    fn load_model(&mut self, _model: &ModelInfo) -> Result<()> {
        Ok(())
    }

    fn run_inference(&self, _prompt: &str) -> Result<BenchmarkResult> {
        unimplemented!("Stage 3C")
    }

    fn teardown(&mut self) -> Result<()> {
        Ok(())
    }
}
