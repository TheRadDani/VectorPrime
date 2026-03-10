try:
    from llmforge._llmforge import (  # noqa: F401
        HardwareProfile,
        OptimizationResult,
        analyze_model,
        convert_gguf_to_onnx,
        convert_onnx_to_gguf,
        export_ollama,
        optimize,
        profile_hardware,
    )
except ImportError:
    # Native extension not yet compiled. Run: maturin develop
    pass
