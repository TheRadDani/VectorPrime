//! Runtime adapter registry and dispatch for LLMForge.
//!
//! This crate owns the [`AdapterRegistry`] (which maps [`RuntimeKind`] to a
//! boxed [`RuntimeAdapter`]) and the [`dispatch`] function that drives the
//! full benchmark cycle through a chosen adapter.
//!
//! It also exposes the [`convert`] module with [`convert::gguf_to_onnx`] and
//! [`convert::onnx_to_gguf`] for cross-format model conversion.
//!
//! # Runtime Priority
//!
//! - **Ollama** and **TensorRT** are the primary inference backends.
//! - **llama.cpp** (`LlamaCppAdapter`) is retained for compatibility but is
//!   deprioritized; it should not be selected as the default for new deployments.
//! - **vLLM** is future scope and is not yet implemented.

pub mod convert;
pub mod dispatch;
pub mod llamacpp;
pub mod onnx;
pub mod tensorrt;

pub use convert::{gguf_to_onnx, onnx_to_gguf};
pub use dispatch::dispatch;
pub use llamacpp::LlamaCppAdapter;
pub use onnx::OnnxAdapter;
pub use tensorrt::TensorRtAdapter;

use std::collections::HashMap;

use llmforge_core::{RuntimeAdapter, RuntimeKind};

/// Registry of all available runtime adapters.
///
/// On construction, all three built-in adapters are registered. Stubs return
/// [`llmforge_core::RuntimeError::NotInstalled`] from `initialize` until their
/// respective implementation stages (3B / 3C / 3D) are completed.
pub struct AdapterRegistry {
    adapters: HashMap<RuntimeKind, Box<dyn RuntimeAdapter>>,
}

impl AdapterRegistry {
    /// Create a registry pre-populated with all three adapter stubs.
    pub fn new() -> Self {
        let mut adapters: HashMap<RuntimeKind, Box<dyn RuntimeAdapter>> = HashMap::new();
        adapters.insert(RuntimeKind::LlamaCpp, Box::new(LlamaCppAdapter::new()));
        adapters.insert(RuntimeKind::OnnxRuntime, Box::new(OnnxAdapter::new()));
        adapters.insert(RuntimeKind::TensorRT, Box::new(TensorRtAdapter::new()));
        Self { adapters }
    }

    /// Look up a mutable reference to the adapter for `kind`.
    pub fn get_mut<'a>(
        &'a mut self,
        kind: &RuntimeKind,
    ) -> Option<&'a mut (dyn RuntimeAdapter + 'static)> {
        self.adapters.get_mut(kind).map(|b| b.as_mut())
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use llmforge_core::{ModelFormat, ModelInfo, QuantizationStrategy, RuntimeConfig, RuntimeKind};
    use std::path::PathBuf;

    fn sample_config(runtime: RuntimeKind) -> RuntimeConfig {
        RuntimeConfig {
            runtime,
            quantization: QuantizationStrategy::Q4_K_M,
            threads: 4,
            batch_size: 128,
            gpu_layers: 0,
        }
    }

    fn sample_model() -> ModelInfo {
        ModelInfo {
            path: PathBuf::from("/tmp/test.gguf"),
            format: ModelFormat::GGUF,
            param_count: None,
        }
    }

    #[test]
    fn test_registry_has_all_kinds() {
        let mut registry = AdapterRegistry::new();
        assert!(registry.get_mut(&RuntimeKind::LlamaCpp).is_some());
        assert!(registry.get_mut(&RuntimeKind::OnnxRuntime).is_some());
        assert!(registry.get_mut(&RuntimeKind::TensorRT).is_some());
    }

    #[test]
    fn test_dispatch_not_installed_llamacpp() {
        let mut registry = AdapterRegistry::new();
        let config = sample_config(RuntimeKind::LlamaCpp);
        let model = sample_model();
        let err = dispatch(&mut registry, &config, &model, "hello").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not found") || msg.contains("llama-cli"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_dispatch_not_installed_onnx() {
        // The ONNX adapter may fail at different stages depending on the host:
        //   • python3 absent     → NotInstalled ("not found" / "python3")
        //   • onnxruntime absent → NotInstalled ("onnxruntime")
        //   • python3 present    → initialize() succeeds, load_model() rejects the
        //                          GGUF fixture with UnsupportedConfiguration
        // In all cases dispatch() must return Err (never panic).
        let mut registry = AdapterRegistry::new();
        let config = sample_config(RuntimeKind::OnnxRuntime);
        let model = sample_model(); // GGUF fixture — always wrong for ONNX adapter
        let result = dispatch(&mut registry, &config, &model, "hello");
        assert!(result.is_err(), "expected Err from OnnxAdapter dispatch");
    }

    #[test]
    fn test_dispatch_not_installed_tensorrt() {
        let mut registry = AdapterRegistry::new();
        let config = sample_config(RuntimeKind::TensorRT);
        let model = sample_model();
        let err = dispatch(&mut registry, &config, &model, "hello").unwrap_err();
        let msg = err.to_string();
        // When trtexec is absent: "not found" / "trtexec".
        // When trtexec is present but the GGUF fixture is rejected: "ONNX" / "unsupported".
        assert!(
            msg.contains("not found")
                || msg.contains("trtexec")
                || msg.contains("ONNX")
                || msg.contains("unsupported"),
            "unexpected error message: {msg}"
        );
    }
}
