use anyhow::Result;
use llmforge_core::{BenchmarkResult, ModelInfo, RuntimeAdapter, RuntimeConfig, RuntimeError};

/// Stub adapter for llama.cpp (`llama-cli` binary).
///
/// Full implementation is added in Stage 3B. Until then, `initialize` returns
/// [`RuntimeError::NotInstalled`] so the optimizer can skip this adapter.
pub struct LlamaCppAdapter {
    #[allow(dead_code)] // populated in Stage 3B
    config: Option<RuntimeConfig>,
}

impl LlamaCppAdapter {
    pub fn new() -> Self {
        Self { config: None }
    }
}

impl Default for LlamaCppAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeAdapter for LlamaCppAdapter {
    fn initialize(&mut self, _config: &RuntimeConfig) -> Result<()> {
        Err(anyhow::anyhow!(RuntimeError::NotInstalled {
            binary: "llama-cli".to_string(),
        }))
    }

    fn load_model(&mut self, _model: &ModelInfo) -> Result<()> {
        Ok(())
    }

    fn run_inference(&self, _prompt: &str) -> Result<BenchmarkResult> {
        unimplemented!("Stage 3B")
    }

    fn teardown(&mut self) -> Result<()> {
        Ok(())
    }
}
