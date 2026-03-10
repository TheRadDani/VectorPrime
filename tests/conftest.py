"""
Shared pytest fixtures and configuration for all vectorprime tests.
"""

import os
import pytest

# ──────────────────────────────────────────────────────────────
# Path constants
# ──────────────────────────────────────────────────────────────

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
GGUF_FIXTURE = os.path.join(FIXTURES_DIR, "tiny.gguf")
ONNX_FIXTURE = os.path.join(FIXTURES_DIR, "tiny_onnx", "model.onnx")


# ──────────────────────────────────────────────────────────────
# Custom markers (registered in pytest.ini)
# ──────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_fixtures: test needs downloaded model fixtures (see tests/fixtures/FIXTURES.md)",
    )
    config.addinivalue_line(
        "markers",
        "requires_gpu: test needs a physical GPU (set LLMFORGE_GPU_TESTS=1)",
    )


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def gguf_model_path():
    """Path to the tiny GGUF fixture. Skips if not downloaded."""
    if not os.path.exists(GGUF_FIXTURE):
        pytest.skip(f"GGUF fixture not found at {GGUF_FIXTURE}. See tests/fixtures/FIXTURES.md")
    return GGUF_FIXTURE


@pytest.fixture(scope="session")
def onnx_model_path():
    """Path to the tiny ONNX fixture. Skips if not downloaded."""
    if not os.path.exists(ONNX_FIXTURE):
        pytest.skip(f"ONNX fixture not found at {ONNX_FIXTURE}. See tests/fixtures/FIXTURES.md")
    return ONNX_FIXTURE


@pytest.fixture(scope="session")
def vectorprime():
    """Import and return the vectorprime module (requires maturin develop)."""
    try:
        import vectorprime as lf
        return lf
    except ImportError:
        pytest.skip("vectorprime bindings not compiled. Run: maturin develop")


@pytest.fixture(scope="session")
def hardware_profile(vectorprime):
    """Hardware profile detected once per test session."""
    return vectorprime.profile_hardware()
