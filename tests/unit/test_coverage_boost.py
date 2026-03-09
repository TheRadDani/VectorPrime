"""
Tests targeting uncovered branches in llmforge.cli and llmforge.onnx_runner.
All native-module calls are mocked so these run without a compiled extension.
"""

import argparse
import json
import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import llmforge as _llmforge_pkg  # imported once at collection time


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_llmforge():
    """Return a MagicMock that satisfies all _llmforge call sites in cli.py."""
    m = MagicMock()

    hw = MagicMock()
    hw.to_json.return_value = json.dumps({"cpu": {"core_count": 8, "brand": "Test CPU"}})
    hw.cpu_cores = 8
    hw.ram_total_mb = 16384
    hw.gpu_model = None
    m.profile_hardware.return_value = hw

    result = MagicMock()
    result.runtime = "LlamaCpp"
    result.quantization = "Q4_K_M"
    result.threads = 8
    result.gpu_layers = 0
    result.tokens_per_sec = 110.3
    result.latency_ms = 9.1
    result.peak_memory_mb = 4096
    result.to_json.return_value = json.dumps({"runtime": "LlamaCpp"})
    m.optimize.return_value = result
    m.optimize_from_json.return_value = result

    manifest = {
        "output_dir": "optimized_model",
        "modelfile_path": "optimized_model/Modelfile",
        "model_gguf_path": "optimized_model/model.gguf",
        "ollama_commands": [
            "ollama create mymodel -f optimized_model/Modelfile",
            "ollama run mymodel",
        ],
    }
    m.export_ollama.return_value = json.dumps(manifest)
    return m


@contextmanager
def _patch_native(mock=None):
    """
    Replace the compiled llmforge._llmforge extension with *mock* for the
    duration of the ``with`` block.

    ``import llmforge._llmforge as X`` inside cli.py is compiled to:
        IMPORT_NAME  'llmforge._llmforge'   → returns parent package 'llmforge'
        IMPORT_FROM  '_llmforge'            → getattr(llmforge_pkg, '_llmforge')
    So we must patch the *attribute* on the package object, not sys.modules.
    """
    if mock is None:
        mock = _mock_llmforge()
    with patch.dict(sys.modules, {"llmforge._llmforge": mock}):
        with patch.object(_llmforge_pkg, "_llmforge", mock, create=True):
            yield mock


# ─────────────────────────────────────────────────────────────────────────────
# cli._divider
# ─────────────────────────────────────────────────────────────────────────────

class TestDivider:
    def test_returns_string_of_dashes(self):
        from llmforge.cli import _divider
        d = _divider()
        assert set(d) == {"─"}
        assert len(d) == 33


# ─────────────────────────────────────────────────────────────────────────────
# cli.build_parser
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildParser:
    def setup_method(self):
        from llmforge.cli import build_parser
        self.parser = build_parser()

    def test_profile_subcommand(self):
        args = self.parser.parse_args(["profile"])
        assert args.command == "profile"

    def test_optimize_required_arg(self):
        args = self.parser.parse_args(["optimize", "model.gguf"])
        assert args.model_path == "model.gguf"
        assert args.format is None
        assert args.max_memory is None

    def test_optimize_with_format_and_max_memory(self):
        args = self.parser.parse_args(
            ["optimize", "model.gguf", "--format", "gguf", "--max-memory", "8000"]
        )
        assert args.format == "gguf"
        assert args.max_memory == 8000

    def test_optimize_onnx_format(self):
        args = self.parser.parse_args(["optimize", "model.onnx", "--format", "onnx"])
        assert args.format == "onnx"

    def test_export_ollama_defaults(self):
        args = self.parser.parse_args(["export-ollama", "model.gguf"])
        assert args.model_path == "model.gguf"
        assert args.output_dir == "optimized_model"
        assert args.result is None

    def test_export_ollama_with_result_and_output_dir(self):
        args = self.parser.parse_args(
            ["export-ollama", "model.gguf", "--result", "r.json", "--output-dir", "out/"]
        )
        assert args.result == "r.json"
        assert args.output_dir == "out/"

    def test_no_subcommand_exits(self):
        with pytest.raises(SystemExit):
            self.parser.parse_args([])

    def test_invalid_format_choice_exits(self):
        with pytest.raises(SystemExit):
            self.parser.parse_args(["optimize", "model.gguf", "--format", "safetensors"])


# ─────────────────────────────────────────────────────────────────────────────
# cli.cmd_profile
# ─────────────────────────────────────────────────────────────────────────────

class TestCmdProfile:
    def test_success_prints_json(self, capsys):
        from llmforge.cli import cmd_profile
        with _patch_native():
            cmd_profile(argparse.Namespace())
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["cpu"]["core_count"] == 8

    def test_runtime_error_exits_1(self, capsys):
        from llmforge.cli import cmd_profile
        mock = _mock_llmforge()
        mock.profile_hardware.side_effect = RuntimeError("hardware probe failed")
        with _patch_native(mock):
            with pytest.raises(SystemExit) as exc:
                cmd_profile(argparse.Namespace())
        assert exc.value.code == 1
        assert "ERROR" in capsys.readouterr().err


# ─────────────────────────────────────────────────────────────────────────────
# cli.cmd_optimize
# ─────────────────────────────────────────────────────────────────────────────

class TestCmdOptimize:
    def _args(self, model_path, fmt=None, max_memory=None):
        return argparse.Namespace(model_path=model_path, format=fmt, max_memory=max_memory)

    def test_unknown_extension_exits_1(self, capsys):
        from llmforge.cli import cmd_optimize
        with pytest.raises(SystemExit) as exc:
            cmd_optimize(self._args("model.bin"))
        assert exc.value.code == 1
        assert "ERROR" in capsys.readouterr().err

    def test_auto_detect_gguf(self, capsys, tmp_path):
        from llmforge.cli import cmd_optimize
        model = tmp_path / "model.gguf"
        model.touch()
        with _patch_native():
            cmd_optimize(self._args(str(model)))
        out = capsys.readouterr().out
        assert "LlamaCpp" in out

    def test_explicit_format_skips_detection(self, capsys, tmp_path):
        from llmforge.cli import cmd_optimize
        model = tmp_path / "model.bin"
        model.touch()
        with _patch_native():
            cmd_optimize(self._args(str(model), fmt="gguf"))
        out = capsys.readouterr().out
        assert "LlamaCpp" in out

    def test_runtime_error_exits_1(self, capsys):
        from llmforge.cli import cmd_optimize
        mock = _mock_llmforge()
        mock.optimize.side_effect = RuntimeError("binary not found")
        with _patch_native(mock):
            with pytest.raises(SystemExit) as exc:
                cmd_optimize(self._args("model.gguf", fmt="gguf"))
        assert exc.value.code == 1

    def test_success_writes_result_file(self, tmp_path):
        from llmforge.cli import cmd_optimize
        model = tmp_path / "model.gguf"
        model.touch()
        with _patch_native():
            cmd_optimize(self._args(str(model), fmt="gguf"))
        result_file = Path(str(model) + ".llmforge_result.json")
        assert result_file.exists()
        data = json.loads(result_file.read_text())
        assert data["runtime"] == "LlamaCpp"

    def test_success_prints_all_fields(self, capsys, tmp_path):
        from llmforge.cli import cmd_optimize
        model = tmp_path / "model.gguf"
        model.touch()
        with _patch_native():
            cmd_optimize(self._args(str(model), fmt="gguf"))
        out = capsys.readouterr().out
        for field in ("Runtime:", "Quantization:", "Threads:", "GPU Layers:", "Throughput:", "Latency:", "Memory:"):
            assert field in out

    def test_max_memory_warning_when_exceeded(self, capsys, tmp_path):
        from llmforge.cli import cmd_optimize
        model = tmp_path / "model.gguf"
        model.touch()
        # result.peak_memory_mb = 4096, set max_memory lower
        with _patch_native():
            cmd_optimize(self._args(str(model), fmt="gguf", max_memory=2048))
        assert "WARNING" in capsys.readouterr().err

    def test_max_memory_no_warning_when_within_limit(self, capsys, tmp_path):
        from llmforge.cli import cmd_optimize
        model = tmp_path / "model.gguf"
        model.touch()
        with _patch_native():
            cmd_optimize(self._args(str(model), fmt="gguf", max_memory=8192))
        assert "WARNING" not in capsys.readouterr().err

    def test_write_failure_prints_warning(self, capsys, tmp_path):
        from llmforge.cli import cmd_optimize
        model = tmp_path / "model.gguf"
        model.touch()
        with _patch_native():
            with patch("llmforge.cli.Path") as mock_path_cls:
                mock_path_cls.return_value.write_text.side_effect = OSError("disk full")
                cmd_optimize(self._args(str(model), fmt="gguf"))
        assert "WARNING" in capsys.readouterr().err


# ─────────────────────────────────────────────────────────────────────────────
# cli.cmd_export_ollama
# ─────────────────────────────────────────────────────────────────────────────

class TestCmdExportOllama:
    def test_no_result_file_auto_optimize(self, capsys, tmp_path):
        from llmforge.cli import cmd_export_ollama
        model = tmp_path / "model.gguf"
        model.touch()
        args = argparse.Namespace(
            model_path=str(model), output_dir="optimized_model", result=None
        )
        with _patch_native():
            cmd_export_ollama(args)
        out = capsys.readouterr().out
        assert "optimized_model" in out
        assert "ollama" in out.lower()

    def test_with_result_file(self, capsys, tmp_path):
        """--result path: read JSON then call export_ollama (no optimize_from_json)."""
        from llmforge.cli import cmd_export_ollama
        model = tmp_path / "model.gguf"
        model.touch()
        result_file = tmp_path / "result.json"
        result_file.write_text('{"runtime": "LlamaCpp"}', encoding="utf-8")
        args = argparse.Namespace(
            model_path=str(model), output_dir="optimized_model", result=str(result_file)
        )
        # The --result branch calls _llmforge.optimize_from_json which only
        # exists on the MagicMock, so we must ensure the mock is active.
        with _patch_native():
            cmd_export_ollama(args)
        out = capsys.readouterr().out
        assert "Modelfile" in out

    def test_runtime_error_exits_1(self, capsys, tmp_path):
        from llmforge.cli import cmd_export_ollama
        model = tmp_path / "model.gguf"
        model.touch()
        mock = _mock_llmforge()
        mock.optimize.side_effect = RuntimeError("no llamacpp binary")
        args = argparse.Namespace(
            model_path=str(model), output_dir="optimized_model", result=None
        )
        with _patch_native(mock):
            with pytest.raises(SystemExit) as exc:
                cmd_export_ollama(args)
        assert exc.value.code == 1

    def test_unknown_extension_raises_value_error_exits_1(self, capsys):
        from llmforge.cli import cmd_export_ollama
        args = argparse.Namespace(
            model_path="model.bin", output_dir="optimized_model", result=None
        )
        with pytest.raises(SystemExit) as exc:
            cmd_export_ollama(args)
        assert exc.value.code == 1


# ─────────────────────────────────────────────────────────────────────────────
# cli.main
# ─────────────────────────────────────────────────────────────────────────────

class TestMain:
    def test_dispatches_profile(self):
        from llmforge import cli
        with patch.object(cli, "cmd_profile") as mock_cmd:
            with patch("sys.argv", ["llmforge", "profile"]):
                cli.main()
            mock_cmd.assert_called_once()

    def test_dispatches_optimize(self):
        from llmforge import cli
        with patch.object(cli, "cmd_optimize") as mock_cmd:
            with patch("sys.argv", ["llmforge", "optimize", "model.gguf"]):
                cli.main()
            mock_cmd.assert_called_once()

    def test_dispatches_export_ollama(self):
        from llmforge import cli
        with patch.object(cli, "cmd_export_ollama") as mock_cmd:
            with patch("sys.argv", ["llmforge", "export-ollama", "model.gguf"]):
                cli.main()
            mock_cmd.assert_called_once()

    def test_no_subcommand_exits(self):
        with patch("sys.argv", ["llmforge"]):
            with pytest.raises(SystemExit):
                from llmforge import cli
                cli.main()


# ─────────────────────────────────────────────────────────────────────────────
# onnx_runner._error
# ─────────────────────────────────────────────────────────────────────────────

class TestOnnxRunnerError:
    def test_writes_json_error_and_exits(self, capsys):
        from llmforge.onnx_runner import _error
        with pytest.raises(SystemExit) as exc:
            _error("something went wrong")
        assert exc.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["error"] == "something went wrong"


# ─────────────────────────────────────────────────────────────────────────────
# onnx_runner._check_mode
# ─────────────────────────────────────────────────────────────────────────────

class TestOnnxRunnerCheckMode:
    def test_exits_0_when_onnxruntime_available(self):
        from llmforge.onnx_runner import _check_mode
        with patch.dict(sys.modules, {"onnxruntime": MagicMock()}):
            with pytest.raises(SystemExit) as exc:
                _check_mode()
        assert exc.value.code == 0

    def test_exits_nonzero_when_onnxruntime_absent(self, capsys):
        from llmforge.onnx_runner import _check_mode
        # Setting the module to None makes "import onnxruntime" raise ImportError
        with patch.dict(sys.modules, {"onnxruntime": None}):
            with pytest.raises(SystemExit) as exc:
                _check_mode()
        assert exc.value.code != 0
        payload = json.loads(capsys.readouterr().out)
        assert "error" in payload


# ─────────────────────────────────────────────────────────────────────────────
# onnx_runner._ort_dtype_to_numpy
# ─────────────────────────────────────────────────────────────────────────────

class TestOrtDtypeToNumpy:
    def test_known_types(self):
        import numpy as np
        from llmforge.onnx_runner import _ort_dtype_to_numpy

        assert _ort_dtype_to_numpy("tensor(float)") is np.float32
        assert _ort_dtype_to_numpy("tensor(double)") is np.float64
        assert _ort_dtype_to_numpy("tensor(int32)") is np.int32
        assert _ort_dtype_to_numpy("tensor(int64)") is np.int64
        assert _ort_dtype_to_numpy("tensor(int8)") is np.int8
        assert _ort_dtype_to_numpy("tensor(uint8)") is np.uint8
        assert _ort_dtype_to_numpy("tensor(bool)") is np.bool_
        assert _ort_dtype_to_numpy("tensor(float16)") is np.float16

    def test_unknown_type_defaults_to_float32(self):
        import numpy as np
        from llmforge.onnx_runner import _ort_dtype_to_numpy

        assert _ort_dtype_to_numpy("tensor(bfloat16)") is np.float32
        assert _ort_dtype_to_numpy("unknown") is np.float32


# ─────────────────────────────────────────────────────────────────────────────
# onnx_runner._run  (mocked onnxruntime)
# ─────────────────────────────────────────────────────────────────────────────

class TestOnnxRunnerRun:
    def _make_mock_ort(self, inference_error=None):
        import numpy as np

        mock_inp = MagicMock()
        mock_inp.name = "input_ids"
        mock_inp.type = "tensor(int64)"
        mock_inp.shape = [1, 10]

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_inp]
        if inference_error:
            mock_session.run.side_effect = inference_error
        else:
            mock_session.run.return_value = [np.zeros((1, 10), dtype=np.int64)]

        mock_ort = MagicMock()
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.InferenceSession.return_value = mock_session
        return mock_ort

    def test_success_outputs_json(self, capsys):
        from llmforge.onnx_runner import _run
        with patch.dict(sys.modules, {"onnxruntime": self._make_mock_ort(), "numpy": __import__("numpy")}):
            _run({"model_path": "fake.onnx", "threads": 1, "prompt_tokens": 5})
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "tokens_per_sec" in data
        assert "latency_ms" in data
        assert "peak_memory_mb" in data

    def test_uses_cpu_fallback_when_provider_unavailable(self, capsys):
        from llmforge.onnx_runner import _run
        mock_ort = self._make_mock_ort()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        with patch.dict(sys.modules, {"onnxruntime": mock_ort, "numpy": __import__("numpy")}):
            _run({
                "model_path": "fake.onnx",
                "execution_provider": "CUDAExecutionProvider",
                "threads": 2,
                "prompt_tokens": 10,
            })
        data = json.loads(capsys.readouterr().out)
        assert "tokens_per_sec" in data

    def test_load_failure_exits_nonzero(self, capsys):
        from llmforge.onnx_runner import _run
        mock_ort = self._make_mock_ort()
        mock_ort.InferenceSession.side_effect = Exception("bad model file")
        with patch.dict(sys.modules, {"onnxruntime": mock_ort, "numpy": __import__("numpy")}):
            with pytest.raises(SystemExit) as exc:
                _run({"model_path": "bad.onnx", "threads": 1, "prompt_tokens": 5})
        assert exc.value.code != 0
        payload = json.loads(capsys.readouterr().out)
        assert "error" in payload

    def test_inference_failure_exits_nonzero(self, capsys):
        from llmforge.onnx_runner import _run
        mock_ort = self._make_mock_ort(inference_error=RuntimeError("CUDA OOM"))
        with patch.dict(sys.modules, {"onnxruntime": mock_ort, "numpy": __import__("numpy")}):
            with pytest.raises(SystemExit) as exc:
                _run({"model_path": "fake.onnx", "threads": 1, "prompt_tokens": 5})
        assert exc.value.code != 0

    def test_dynamic_shapes_become_1(self, capsys):
        """Non-integer / zero dimensions in input shape are replaced with 1."""
        import numpy as np
        from llmforge.onnx_runner import _run

        mock_inp = MagicMock()
        mock_inp.name = "input_ids"
        mock_inp.type = "tensor(float)"
        mock_inp.shape = ["batch", 0, 512]  # dynamic / zero dims

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_inp]
        mock_session.run.return_value = [np.zeros((1, 1, 512), dtype=np.float32)]

        mock_ort = MagicMock()
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.InferenceSession.return_value = mock_session

        with patch.dict(sys.modules, {"onnxruntime": mock_ort, "numpy": np}):
            _run({"model_path": "fake.onnx", "threads": 1, "prompt_tokens": 5})
        data = json.loads(capsys.readouterr().out)
        assert "tokens_per_sec" in data


# ─────────────────────────────────────────────────────────────────────────────
# onnx_runner.main
# ─────────────────────────────────────────────────────────────────────────────

class TestOnnxRunnerMain:
    def test_check_flag_calls_check_mode(self):
        from llmforge import onnx_runner
        with patch.object(onnx_runner, "_check_mode") as mock_cm:
            with patch("sys.argv", ["onnx_runner.py", "--check"]):
                onnx_runner.main()
        mock_cm.assert_called_once()

    def test_valid_stdin_calls_run(self):
        from llmforge import onnx_runner
        payload = json.dumps({"model_path": "m.onnx", "threads": 1, "prompt_tokens": 5})
        with patch.object(onnx_runner, "_run") as mock_run:
            with patch("sys.argv", ["onnx_runner.py"]):
                with patch("sys.stdin", StringIO(payload)):
                    onnx_runner.main()
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0]["model_path"] == "m.onnx"

    def test_invalid_stdin_json_exits(self, capsys):
        from llmforge import onnx_runner
        with patch("sys.argv", ["onnx_runner.py"]):
            with patch("sys.stdin", StringIO("not valid json{{{")):
                with pytest.raises(SystemExit):
                    onnx_runner.main()
        assert "error" in capsys.readouterr().out
