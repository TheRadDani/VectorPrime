"""Integration tests for the optimize() binding."""

import os

import pytest

_vectorprime = pytest.importorskip(
    "vectorprime._vectorprime",
    reason="Native extension not compiled — run: maturin develop",
)

# ── fixture guards ────────────────────────────────────────────────────────────

_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
_GGUF = os.path.join(_FIXTURES_DIR, "tiny.gguf")
_ONNX = os.path.join(_FIXTURES_DIR, "tiny_onnx", "model.onnx")

skip_no_fixtures = pytest.mark.skipif(
    not os.path.exists(_GGUF),
    reason="Model fixtures not downloaded. See tests/fixtures/FIXTURES.md",
)

requires_gpu = pytest.mark.skipif(
    not os.environ.get("LLMFORGE_GPU_TESTS"),
    reason="Set LLMFORGE_GPU_TESTS=1 to enable GPU tests",
)

# ── always-run tests ──────────────────────────────────────────────────────────


import shutil as _shutil

@pytest.mark.skipif(
    not _shutil.which("llama-cli"),
    reason="llama-cli not installed; optimizer falls back to hardware estimates for missing files",
)
def test_optimize_missing_file():
    """optimize() must raise RuntimeError for non-existent paths when llama-cli is present."""
    with pytest.raises(RuntimeError):
        _vectorprime.optimize("/nonexistent/path/model.gguf", "gguf")


def test_optimize_bad_format():
    """Unknown format strings must raise RuntimeError."""
    with pytest.raises(RuntimeError):
        _vectorprime.optimize("/some/model.xyz", "xyz")


# ── fixture-dependent tests ───────────────────────────────────────────────────


@skip_no_fixtures
def test_optimize_gguf():
    """GGUF optimization selects LlamaCpp as the runtime."""
    result = _vectorprime.optimize(_GGUF, "gguf")
    assert result.runtime == "LlamaCpp"


@skip_no_fixtures
def test_optimize_gguf_throughput_positive():
    result = _vectorprime.optimize(_GGUF, "gguf")
    assert result.tokens_per_sec > 0


@skip_no_fixtures
def test_optimize_onnx():
    """ONNX optimization returns a result with positive throughput."""
    result = _vectorprime.optimize(_ONNX, "onnx")
    assert result.tokens_per_sec > 0


@skip_no_fixtures
@requires_gpu
def test_optimize_gguf_gpu_layers_nonzero():
    """When a GPU is present, at least one layer should be offloaded."""
    result = _vectorprime.optimize(_GGUF, "gguf")
    assert result.gpu_layers > 0
