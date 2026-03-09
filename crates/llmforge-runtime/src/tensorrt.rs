use anyhow::Result;
use llmforge_core::{BenchmarkResult, ModelInfo, RuntimeAdapter, RuntimeConfig, RuntimeError};

/// Stub adapter for TensorRT (`trtexec` binary, NVIDIA only, compute cap ≥ 7.0).
///
/// Full implementation is added in Stage 3D.
pub struct TensorRtAdapter {
    #[allow(dead_code)] // populated in Stage 3D
    config: Option<RuntimeConfig>,
}

impl TensorRtAdapter {
    pub fn new() -> Self {
        Self { config: None }
    }
}

impl Default for TensorRtAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeAdapter for TensorRtAdapter {
    fn initialize(&mut self, _config: &RuntimeConfig) -> Result<()> {
        Err(anyhow::anyhow!(RuntimeError::NotInstalled {
            binary: "trtexec".to_string(),
        }))
    }

    fn load_model(&mut self, _model: &ModelInfo) -> Result<()> {
        Ok(())
    }

    fn run_inference(&self, _prompt: &str) -> Result<BenchmarkResult> {
        unimplemented!("Stage 3D")
    }

    fn teardown(&mut self) -> Result<()> {
        Ok(())
    }
}
