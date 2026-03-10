"""
Tests targeting uncovered branches in llmforge.cli and llmforge.onnx_runner.
All native-module calls are mocked so these run without a compiled extension.
"""

import argparse
import json
import sys
from contextlib import contextmanager
from io import StringIO
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
    # output_path is None by default (llama-quantize not installed in test env)
    result.output_path = None
    m.optimize.return_value = result

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

    def test_optimize_with_output(self):
        args = self.parser.parse_args(["optimize", "model.gguf", "--output", "out.gguf"])
        assert args.output == "out.gguf"

    def test_optimize_output_defaults_to_none(self):
        args = self.parser.parse_args(["optimize", "model.gguf"])
        assert args.output is None

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
    def _args(self, model_path, fmt=None, max_memory=None, gpu=None, latency=None, output=None):
        return argparse.Namespace(
            model_path=model_path,
            format=fmt,
            max_memory=max_memory,
            gpu=gpu,
            latency=latency,
            output=output,
        )

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

    def test_no_json_result_file_written(self, tmp_path):
        """The old .llmforge_result.json file must NOT be created."""
        from llmforge.cli import cmd_optimize
        model = tmp_path / "model.gguf"
        model.touch()
        with _patch_native():
            cmd_optimize(self._args(str(model), fmt="gguf"))
        assert not (tmp_path / "model.gguf.llmforge_result.json").exists()

    def test_success_prints_all_fields(self, capsys, tmp_path):
        from llmforge.cli import cmd_optimize
        model = tmp_path / "model.gguf"
        model.touch()
        with _patch_native():
            cmd_optimize(self._args(str(model), fmt="gguf"))
        out = capsys.readouterr().out
        for field in ("Runtime:", "Quantization:", "Threads:", "GPU Layers:", "Throughput:", "Latency:", "Memory:"):
            assert field in out

    def test_output_path_printed_when_quantized(self, capsys, tmp_path):
        """When output_path is set on result, it should be printed."""
        from llmforge.cli import cmd_optimize
        model = tmp_path / "model.gguf"
        model.touch()
        mock = _mock_llmforge()
        mock.optimize.return_value.output_path = str(tmp_path / "model-optimized.gguf")
        with _patch_native(mock):
            cmd_optimize(self._args(str(model), fmt="gguf"))
        out = capsys.readouterr().out
        assert "Optimized model written to:" in out

    def test_note_printed_when_not_quantized(self, capsys, tmp_path):
        """When output_path is None (llama-quantize absent), a NOTE goes to stderr."""
        from llmforge.cli import cmd_optimize
        model = tmp_path / "model.gguf"
        model.touch()
        with _patch_native():  # default mock has output_path = None
            cmd_optimize(self._args(str(model), fmt="gguf"))
        err = capsys.readouterr().err
        assert "NOTE" in err

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
        # The "NOTE" about llama-quantize will appear in err, but not "WARNING"
        err = capsys.readouterr().err
        assert "WARNING" not in err


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

    def test_dispatches_convert_to_onnx(self):
        from llmforge import cli
        with patch.object(cli, "cmd_convert_to_onnx") as mock_cmd:
            with patch("sys.argv", ["llmforge", "convert-to-onnx", "model.gguf"]):
                cli.main()
            mock_cmd.assert_called_once()

    def test_dispatches_convert_to_gguf(self):
        from llmforge import cli
        with patch.object(cli, "cmd_convert_to_gguf") as mock_cmd:
            with patch("sys.argv", ["llmforge", "convert-to-gguf", "model.onnx"]):
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


# ─────────────────────────────────────────────────────────────────────────────
# cli._replace_ext
# ─────────────────────────────────────────────────────────────────────────────

class TestReplaceExt:
    def test_replaces_gguf_with_onnx(self):
        from llmforge.cli import _replace_ext
        assert _replace_ext("model.gguf", ".onnx") == "model.onnx"

    def test_replaces_onnx_with_gguf(self):
        from llmforge.cli import _replace_ext
        assert _replace_ext("model.onnx", ".gguf") == "model.gguf"

    def test_preserves_directory(self):
        from llmforge.cli import _replace_ext
        assert _replace_ext("/data/models/llama.gguf", ".onnx") == "/data/models/llama.onnx"

    def test_no_extension(self):
        from llmforge.cli import _replace_ext
        assert _replace_ext("modelfile", ".onnx") == "modelfile.onnx"


# ─────────────────────────────────────────────────────────────────────────────
# cli.build_parser — convert subcommands
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildParserConvert:
    def setup_method(self):
        from llmforge.cli import build_parser
        self.parser = build_parser()

    def test_convert_to_onnx_required_arg(self):
        args = self.parser.parse_args(["convert-to-onnx", "model.gguf"])
        assert args.input_path == "model.gguf"
        assert args.output is None

    def test_convert_to_onnx_with_output(self):
        args = self.parser.parse_args(["convert-to-onnx", "model.gguf", "--output", "out.onnx"])
        assert args.output == "out.onnx"

    def test_convert_to_gguf_required_arg(self):
        args = self.parser.parse_args(["convert-to-gguf", "model.onnx"])
        assert args.input_path == "model.onnx"
        assert args.output is None

    def test_convert_to_gguf_with_output(self):
        args = self.parser.parse_args(["convert-to-gguf", "model.onnx", "--output", "out.gguf"])
        assert args.output == "out.gguf"


# ─────────────────────────────────────────────────────────────────────────────
# cli.cmd_convert_to_onnx
# ─────────────────────────────────────────────────────────────────────────────

class TestCmdConvertToOnnx:
    def _args(self, input_path, output=None):
        return argparse.Namespace(input_path=input_path, output=output)

    def test_success_prints_result(self, capsys, tmp_path):
        from llmforge.cli import cmd_convert_to_onnx
        mock = _mock_llmforge()
        out_path = str(tmp_path / "model.onnx")
        mock.convert_gguf_to_onnx.return_value = out_path
        with _patch_native(mock):
            cmd_convert_to_onnx(self._args(str(tmp_path / "model.gguf")))
        out = capsys.readouterr().out
        assert "LLMForge Conversion Result" in out
        assert "model.onnx" in out

    def test_default_output_derived_from_input(self, capsys, tmp_path):
        """When --output is omitted the output path gets .onnx extension."""
        from llmforge.cli import cmd_convert_to_onnx
        mock = _mock_llmforge()
        input_path = str(tmp_path / "model.gguf")
        expected_out = str(tmp_path / "model.onnx")
        mock.convert_gguf_to_onnx.return_value = expected_out
        with _patch_native(mock):
            cmd_convert_to_onnx(self._args(input_path))
        # Verify the binding was called with the derived output path
        mock.convert_gguf_to_onnx.assert_called_once_with(input_path, expected_out)

    def test_explicit_output_passed_through(self, capsys, tmp_path):
        from llmforge.cli import cmd_convert_to_onnx
        mock = _mock_llmforge()
        out_path = "/custom/output.onnx"
        mock.convert_gguf_to_onnx.return_value = out_path
        with _patch_native(mock):
            cmd_convert_to_onnx(self._args("model.gguf", output=out_path))
        mock.convert_gguf_to_onnx.assert_called_once_with("model.gguf", out_path)

    def test_runtime_error_exits_1(self, capsys):
        from llmforge.cli import cmd_convert_to_onnx
        mock = _mock_llmforge()
        mock.convert_gguf_to_onnx.side_effect = RuntimeError("python3 not found")
        with _patch_native(mock):
            with pytest.raises(SystemExit) as exc:
                cmd_convert_to_onnx(self._args("model.gguf"))
        assert exc.value.code == 1
        assert "ERROR" in capsys.readouterr().err


# ─────────────────────────────────────────────────────────────────────────────
# cli.cmd_convert_to_gguf
# ─────────────────────────────────────────────────────────────────────────────

class TestCmdConvertToGguf:
    def _args(self, input_path, output=None):
        return argparse.Namespace(input_path=input_path, output=output)

    def test_success_prints_result(self, capsys, tmp_path):
        from llmforge.cli import cmd_convert_to_gguf
        mock = _mock_llmforge()
        out_path = str(tmp_path / "model.gguf")
        mock.convert_onnx_to_gguf.return_value = out_path
        with _patch_native(mock):
            cmd_convert_to_gguf(self._args(str(tmp_path / "model.onnx")))
        out = capsys.readouterr().out
        assert "LLMForge Conversion Result" in out
        assert "model.gguf" in out

    def test_default_output_derived_from_input(self, capsys, tmp_path):
        from llmforge.cli import cmd_convert_to_gguf
        mock = _mock_llmforge()
        input_path = str(tmp_path / "model.onnx")
        expected_out = str(tmp_path / "model.gguf")
        mock.convert_onnx_to_gguf.return_value = expected_out
        with _patch_native(mock):
            cmd_convert_to_gguf(self._args(input_path))
        mock.convert_onnx_to_gguf.assert_called_once_with(input_path, expected_out)

    def test_explicit_output_passed_through(self, capsys, tmp_path):
        from llmforge.cli import cmd_convert_to_gguf
        mock = _mock_llmforge()
        out_path = "/custom/output.gguf"
        mock.convert_onnx_to_gguf.return_value = out_path
        with _patch_native(mock):
            cmd_convert_to_gguf(self._args("model.onnx", output=out_path))
        mock.convert_onnx_to_gguf.assert_called_once_with("model.onnx", out_path)

    def test_runtime_error_exits_1(self, capsys):
        from llmforge.cli import cmd_convert_to_gguf
        mock = _mock_llmforge()
        mock.convert_onnx_to_gguf.side_effect = RuntimeError("onnx not installed")
        with _patch_native(mock):
            with pytest.raises(SystemExit) as exc:
                cmd_convert_to_gguf(self._args("model.onnx"))
        assert exc.value.code == 1
        assert "ERROR" in capsys.readouterr().err
