try:
    from llmforge._llmforge import (  # noqa: F401
        HardwareProfile,
        OptimizationResult,
        export_ollama,
        optimize,
        profile_hardware,
    )
except ImportError:
    # Native extension not yet compiled. Run: maturin develop
    pass
