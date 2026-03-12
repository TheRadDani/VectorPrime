Add vllm and ollama runtime adapters with extensible registry
- Add RuntimeKind::Ollama and RuntimeKind::Vllm variants to core
- Implement OllamaAdapter (shells out to `ollama run`; NotInstalled on missing binary)
- Implement VllmAdapter (shells out to `python3 -c "from vllm import LLM..."`)
- Register both adapters in AdapterRegistry dispatcher
- Extend search space: Ollama alongside LlamaCpp for GGUF candidates; vLLM (GPU-gated) alongside ONNX/TensorRT for ONNX candidates