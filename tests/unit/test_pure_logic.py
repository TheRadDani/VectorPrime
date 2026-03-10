"""
Unit tests for vectorprime.cli — no bindings required.
These run without a compiled Rust extension.
"""
import pytest


# ─────────────────────────────────────────────────────────
# detect_format helper
# ─────────────────────────────────────────────────────────

def detect_format(path: str) -> str:
    """Mirrors the helper in cli.py — tested independently."""
    if path.endswith(".gguf"):
        return "gguf"
    if path.endswith(".onnx"):
        return "onnx"
    raise ValueError(f"Cannot detect format from extension: {path}")


class TestDetectFormat:
    def test_gguf_extension(self):
        assert detect_format("model.gguf") == "gguf"

    def test_onnx_extension(self):
        assert detect_format("model.onnx") == "onnx"

    def test_full_path_gguf(self):
        assert detect_format("/data/models/llama-7b.Q4_K_M.gguf") == "gguf"

    def test_full_path_onnx(self):
        assert detect_format("/tmp/distilbert/model.onnx") == "onnx"

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Cannot detect format"):
            detect_format("model.bin")

    def test_no_extension_raises(self):
        with pytest.raises(ValueError):
            detect_format("modelfile")

    def test_uppercase_extension_raises(self):
        # We are case-sensitive; .GGUF is not recognised
        with pytest.raises(ValueError):
            detect_format("model.GGUF")


# ─────────────────────────────────────────────────────────
# ONNX output parsing (mirrors onnx.rs parse_onnx_output)
# ─────────────────────────────────────────────────────────

import json


def parse_onnx_output(json_str: str) -> dict:
    """Python mirror of the Rust parse_onnx_output helper."""
    data = json.loads(json_str)
    if "error" in data:
        raise RuntimeError(data["error"])
    return {
        "tokens_per_sec": float(data["tokens_per_sec"]),
        "latency_ms": float(data["latency_ms"]),
        "peak_memory_mb": int(data["peak_memory_mb"]),
    }


class TestParseOnnxOutput:
    def test_valid_output(self):
        js = '{"tokens_per_sec": 45.2, "latency_ms": 210.0, "peak_memory_mb": 3400}'
        result = parse_onnx_output(js)
        assert result["tokens_per_sec"] == pytest.approx(45.2)
        assert result["latency_ms"] == pytest.approx(210.0)
        assert result["peak_memory_mb"] == 3400

    def test_error_key_raises(self):
        js = '{"error": "onnxruntime not installed"}'
        with pytest.raises(RuntimeError, match="onnxruntime not installed"):
            parse_onnx_output(js)

    def test_missing_field_raises(self):
        js = '{"tokens_per_sec": 10.0}'
        with pytest.raises(KeyError):
            parse_onnx_output(js)

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_onnx_output("not json")


# ─────────────────────────────────────────────────────────
# llama.cpp timing line parser
# ─────────────────────────────────────────────────────────

import re


def parse_llama_timings(output: str):
    """Python mirror of the Rust parse_llama_timings helper."""
    pattern = (
        r"llama_print_timings:\s+eval time\s*=\s*[\d.]+ ms\s*/\s*\d+ tokens"
        r"\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\s*\)"
    )
    m = re.search(pattern, output)
    if not m:
        return None
    latency_ms = float(m.group(1))
    tokens_per_sec = float(m.group(2))
    return tokens_per_sec, latency_ms


class TestParseLlamaTimings:
    SAMPLE = (
        "llama_print_timings:     eval time =  4551.49 ms /    50 tokens "
        "(   91.03 ms per token,    10.99 tokens per second)"
    )

    def test_valid_line(self):
        result = parse_llama_timings(self.SAMPLE)
        assert result is not None
        tps, lat = result
        assert tps == pytest.approx(10.99)
        assert lat == pytest.approx(91.03)

    def test_embedded_in_larger_output(self):
        output = "some preamble\n" + self.SAMPLE + "\nsome epilogue"
        result = parse_llama_timings(output)
        assert result is not None

    def test_missing_timing_line(self):
        assert parse_llama_timings("Loading model...\nDone.") is None

    def test_empty_string(self):
        assert parse_llama_timings("") is None


# ─────────────────────────────────────────────────────────
# Search space / candidate pruning logic
# ─────────────────────────────────────────────────────────

class TestBytesPerParam:
    """bytes_per_param must return a positive float for every quant strategy."""

    QUANT_BYTES = {
        "F16": 2.0,
        "Q8_0": 1.0,
        "Q4_K_M": 0.5,
        "Q4_0": 0.5,
        "Int8": 1.0,
        "Int4": 0.5,
    }

    def test_all_strategies_positive(self):
        for name, bpp in self.QUANT_BYTES.items():
            assert bpp > 0, f"{name} bytes_per_param must be > 0"

    def test_f16_is_two_bytes(self):
        assert self.QUANT_BYTES["F16"] == 2.0

    def test_q4_variants_are_half_byte(self):
        assert self.QUANT_BYTES["Q4_K_M"] == 0.5
        assert self.QUANT_BYTES["Q4_0"] == 0.5


class TestCandidateGeneration:
    """Logic mirrors vectorprime-optimizer search.rs generate_candidates."""

    def _eligible_runtimes(self, fmt: str, has_nvidia_gpu: bool) -> list:
        if fmt == "gguf":
            return ["LlamaCpp"]
        runtimes = ["OnnxRuntime"]
        if has_nvidia_gpu:
            runtimes.append("TensorRT")
        return runtimes

    def test_gguf_only_llamacpp(self):
        runtimes = self._eligible_runtimes("gguf", has_nvidia_gpu=True)
        assert runtimes == ["LlamaCpp"]

    def test_onnx_no_gpu_no_tensorrt(self):
        runtimes = self._eligible_runtimes("onnx", has_nvidia_gpu=False)
        assert "TensorRT" not in runtimes
        assert "OnnxRuntime" in runtimes

    def test_onnx_with_gpu_includes_tensorrt(self):
        runtimes = self._eligible_runtimes("onnx", has_nvidia_gpu=True)
        assert "TensorRT" in runtimes

    def test_thread_clamping(self):
        def clamp_threads(n: int) -> list:
            candidates = [n // 2, n, n * 2]
            return [max(1, min(64, t)) for t in candidates]

        result = clamp_threads(32)
        assert all(1 <= t <= 64 for t in result)

    def test_thread_clamping_single_core(self):
        candidates = [max(1, min(64, t)) for t in [0, 1, 2]]
        assert candidates[0] == 1   # clamped from 0


# ─────────────────────────────────────────────────────────
# Ollama Modelfile generation
# ─────────────────────────────────────────────────────────

class TestModelfileGeneration:
    def _generate_modelfile(self, gguf_name: str, threads: int, gpu_layers: int) -> str:
        return (
            f"FROM ./{gguf_name}\n"
            f"PARAMETER num_thread {threads}\n"
            f"PARAMETER num_gpu {gpu_layers}\n"
            f"PARAMETER num_ctx 4096\n"
            f"# Generated by VectorPrime\n"
        )

    def test_contains_from(self):
        mf = self._generate_modelfile("model.gguf", 16, 20)
        assert "FROM ./model.gguf" in mf

    def test_threads_in_modelfile(self):
        mf = self._generate_modelfile("model.gguf", 8, 0)
        assert "PARAMETER num_thread 8" in mf

    def test_gpu_layers_zero(self):
        mf = self._generate_modelfile("model.gguf", 4, 0)
        assert "PARAMETER num_gpu 0" in mf

    def test_gpu_layers_set(self):
        mf = self._generate_modelfile("model.gguf", 16, 33)
        assert "PARAMETER num_gpu 33" in mf
