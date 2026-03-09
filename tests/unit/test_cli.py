"""Unit tests for llmforge.cli (pure-Python helpers, no native module)."""

import pytest

from llmforge.cli import detect_format


def test_detect_format_gguf():
    assert detect_format("llama3-8b.gguf") == "gguf"
    assert detect_format("/models/mistral.gguf") == "gguf"


def test_detect_format_onnx():
    assert detect_format("model.onnx") == "onnx"
    assert detect_format("/tmp/bert.onnx") == "onnx"


def test_detect_format_unknown():
    with pytest.raises(ValueError, match="Cannot detect format"):
        detect_format("model.bin")

    with pytest.raises(ValueError, match="Cannot detect format"):
        detect_format("model")
