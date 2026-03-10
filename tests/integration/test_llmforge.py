"""
Integration tests for vectorprime Python bindings + CLI.

Tests in this file require the Rust extension to be compiled
(run `maturin develop` first).

Fixture-dependent tests are skipped unless model files are present.
GPU tests are skipped unless LLMFORGE_GPU_TESTS=1 is set.
"""

import json
import os
import subprocess
import sys

import pytest

# ─── skip guards ─────────────────────────────────────────

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
GGUF_FIXTURE = os.path.join(FIXTURES_DIR, "tiny.gguf")
ONNX_FIXTURE = os.path.join(FIXTURES_DIR, "tiny_onnx", "model.onnx")

requires_fixtures = pytest.mark.skipif(
    not os.path.exists(GGUF_FIXTURE),
    reason=(
        "Model fixtures not downloaded. "
        "Run: curl -L <tinyllama-url> -o tests/fixtures/tiny.gguf"
    ),
)

requires_gpu = pytest.mark.skipif(
    not os.environ.get("LLMFORGE_GPU_TESTS"),
    reason="Set LLMFORGE_GPU_TESTS=1 to run GPU tests",
)


# ─── hardware profiling ──────────────────────────────────

class TestHardwareProfile:
    """Tests profile_hardware() via Python bindings."""

    @pytest.fixture(autouse=True)
    def import_bindings(self):
        try:
            import vectorprime  # noqa: F401
            self.vectorprime = vectorprime
        except ImportError:
            pytest.skip("vectorprime bindings not compiled — run `maturin develop`")

    def test_profile_returns_object(self):
        hw = self.vectorprime.profile_hardware()
        assert hw is not None

    def test_cpu_cores_at_least_one(self):
        hw = self.vectorprime.profile_hardware()
        assert hw.cpu_cores >= 1

    def test_ram_total_positive(self):
        hw = self.vectorprime.profile_hardware()
        assert hw.ram_total_mb > 0

    def test_to_json_valid(self):
        hw = self.vectorprime.profile_hardware()
        data = json.loads(hw.to_json())
        assert "cpu" in data
        assert data["cpu"]["core_count"] >= 1

    def test_repr_contains_cpu(self):
        hw = self.vectorprime.profile_hardware()
        assert "HardwareProfile" in repr(hw)

    def test_gpu_model_is_string_or_none(self):
        hw = self.vectorprime.profile_hardware()
        assert hw.gpu_model is None or isinstance(hw.gpu_model, str)


# ─── optimize() ─────────────────────────────────────────

class TestOptimize:
    @pytest.fixture(autouse=True)
    def import_bindings(self):
        try:
            import vectorprime
            self.vectorprime = vectorprime
        except ImportError:
            pytest.skip("vectorprime bindings not compiled — run `maturin develop`")

    @pytest.mark.skipif(
        not __import__("shutil").which("llama-cli"),
        reason="llama-cli not installed; optimizer falls back to hardware estimates",
    )
    def test_optimize_missing_file_raises(self):
        with pytest.raises(RuntimeError):
            self.vectorprime.optimize("/nonexistent/path/model.gguf", "gguf")

    def test_optimize_bad_format_raises(self):
        with pytest.raises((RuntimeError, ValueError)):
            self.vectorprime.optimize("/some/model.xyz", "xyz")

    @requires_fixtures
    def test_optimize_gguf_returns_llamacpp(self):
        result = self.vectorprime.optimize(GGUF_FIXTURE, "gguf")
        assert result.runtime == "LlamaCpp"
        assert result.tokens_per_sec > 0
        assert result.latency_ms > 0

    @requires_fixtures
    def test_optimize_gguf_result_has_all_fields(self):
        result = self.vectorprime.optimize(GGUF_FIXTURE, "gguf")
        assert result.threads >= 1
        assert result.gpu_layers >= 0
        assert result.peak_memory_mb >= 0

    @requires_fixtures
    def test_optimize_gguf_result_to_json(self):
        result = self.vectorprime.optimize(GGUF_FIXTURE, "gguf")
        data = json.loads(result.to_json())
        assert "config" in data
        assert "metrics" in data

    @requires_fixtures
    def test_optimize_onnx_returns_result(self):
        result = self.vectorprime.optimize(ONNX_FIXTURE, "onnx")
        assert result.tokens_per_sec > 0

    @requires_fixtures
    @requires_gpu
    def test_optimize_gguf_gpu_layers_nonzero(self):
        result = self.vectorprime.optimize(GGUF_FIXTURE, "gguf")
        assert result.gpu_layers > 0


# ─── export_ollama() ────────────────────────────────────

class TestExportOllama:
    @pytest.fixture(autouse=True)
    def import_bindings(self, tmp_path):
        try:
            import vectorprime
            self.vectorprime = vectorprime
        except ImportError:
            pytest.skip("vectorprime bindings not compiled — run `maturin develop`")
        self.tmp_path = tmp_path

    @requires_fixtures
    def test_export_creates_modelfile(self):
        result = self.vectorprime.optimize(GGUF_FIXTURE, "gguf")
        manifest_json = self.vectorprime.export_ollama(result, str(self.tmp_path))
        assert (self.tmp_path / "Modelfile").exists()

    @requires_fixtures
    def test_export_creates_gguf(self):
        result = self.vectorprime.optimize(GGUF_FIXTURE, "gguf")
        self.vectorprime.export_ollama(result, str(self.tmp_path))
        assert (self.tmp_path / "model.gguf").exists()

    @requires_fixtures
    def test_export_creates_metadata(self):
        result = self.vectorprime.optimize(GGUF_FIXTURE, "gguf")
        self.vectorprime.export_ollama(result, str(self.tmp_path))
        meta_path = self.tmp_path / "metadata.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert "config" in data

    @requires_fixtures
    def test_modelfile_contains_from(self):
        result = self.vectorprime.optimize(GGUF_FIXTURE, "gguf")
        self.vectorprime.export_ollama(result, str(self.tmp_path))
        mf = (self.tmp_path / "Modelfile").read_text()
        assert "FROM ./model.gguf" in mf

    @requires_fixtures
    def test_manifest_has_ollama_commands(self):
        result = self.vectorprime.optimize(GGUF_FIXTURE, "gguf")
        manifest_json = self.vectorprime.export_ollama(result, str(self.tmp_path))
        manifest = json.loads(manifest_json)
        assert len(manifest["ollama_commands"]) == 2


# ─── CLI smoke tests (subprocess) ───────────────────────

class TestCLI:
    def _run(self, *args, **kwargs):
        return subprocess.run(
            ["vectorprime", *args],
            capture_output=True,
            text=True,
            **kwargs,
        )

    def test_help_exits_zero(self):
        result = self._run("--help")
        assert result.returncode == 0

    def test_help_lists_subcommands(self):
        result = self._run("--help")
        assert "profile" in result.stdout
        assert "optimize" in result.stdout

    def test_profile_exits_zero(self):
        result = self._run("profile")
        assert result.returncode == 0

    def test_profile_prints_valid_json(self):
        result = self._run("profile")
        assert result.returncode == 0
        data = json.loads(result.stdout)   # must not raise
        assert "cpu" in data

    @pytest.mark.skipif(
        not __import__("shutil").which("llama-cli"),
        reason="llama-cli not installed; optimizer falls back to hardware estimates",
    )
    def test_optimize_missing_file_nonzero(self):
        result = self._run("optimize", "/nonexistent/model.gguf")
        assert result.returncode != 0

    @pytest.mark.skipif(
        not __import__("shutil").which("llama-cli"),
        reason="llama-cli not installed; optimizer falls back to hardware estimates",
    )
    def test_optimize_missing_file_error_message(self):
        result = self._run("optimize", "/nonexistent/model.gguf")
        assert "ERROR" in result.stderr or "error" in result.stderr.lower()

    def test_optimize_unknown_extension_nonzero(self):
        result = self._run("optimize", "model.safetensors")
        assert result.returncode != 0

    @requires_fixtures
    def test_optimize_gguf_full_pipeline(self):
        result = self._run("optimize", GGUF_FIXTURE, "--format", "gguf")
        assert result.returncode == 0
        assert "tokens/sec" in result.stdout

    @requires_fixtures
    def test_optimize_with_custom_output_path(self, tmp_path):
        out = str(tmp_path / "model-optimized.gguf")
        result = self._run(
            "optimize", GGUF_FIXTURE,
            "--format", "gguf",
            "--output", out,
        )
        assert result.returncode == 0
