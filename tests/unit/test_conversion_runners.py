"""
Unit tests for gguf_to_onnx_runner.py and onnx_to_gguf_runner.py.

All heavy dependencies (gguf, onnx, numpy) are mocked so these tests run
without any optional packages installed.
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared between the two runner test suites
# ─────────────────────────────────────────────────────────────────────────────

def _absent(module_name: str):
    """Context manager: makes *module_name* appear uninstalled."""
    return patch.dict(sys.modules, {module_name: None})


# ─────────────────────────────────────────────────────────────────────────────
# gguf_to_onnx_runner — _error
# ─────────────────────────────────────────────────────────────────────────────

class TestGgufToOnnxError:
    def test_writes_json_error_and_exits(self, capsys):
        from llmforge.gguf_to_onnx_runner import _error
        with pytest.raises(SystemExit) as exc:
            _error("test error message")
        assert exc.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["error"] == "test error message"


# ─────────────────────────────────────────────────────────────────────────────
# gguf_to_onnx_runner — _check_imports (missing deps)
# ─────────────────────────────────────────────────────────────────────────────

class TestGgufToOnnxCheckImports:
    def test_missing_gguf_exits(self, capsys):
        with _absent("gguf"):
            from llmforge import gguf_to_onnx_runner
            with pytest.raises(SystemExit) as exc:
                gguf_to_onnx_runner._check_imports()
        assert exc.value.code != 0
        assert "gguf" in capsys.readouterr().out

    def test_missing_onnx_exits(self, capsys):
        with _absent("onnx"):
            from llmforge import gguf_to_onnx_runner
            with pytest.raises(SystemExit) as exc:
                gguf_to_onnx_runner._check_imports()
        assert exc.value.code != 0

    def test_missing_numpy_exits(self, capsys):
        with _absent("numpy"):
            from llmforge import gguf_to_onnx_runner
            with pytest.raises(SystemExit) as exc:
                gguf_to_onnx_runner._check_imports()
        assert exc.value.code != 0


# ─────────────────────────────────────────────────────────────────────────────
# gguf_to_onnx_runner — _gguf_value_to_serialisable
# ─────────────────────────────────────────────────────────────────────────────

class TestGgufValueToSerialisable:
    def test_int(self):
        from llmforge.gguf_to_onnx_runner import _gguf_value_to_serialisable
        assert _gguf_value_to_serialisable(42) == 42

    def test_float(self):
        from llmforge.gguf_to_onnx_runner import _gguf_value_to_serialisable
        assert _gguf_value_to_serialisable(3.14) == pytest.approx(3.14)

    def test_bool(self):
        from llmforge.gguf_to_onnx_runner import _gguf_value_to_serialisable
        assert _gguf_value_to_serialisable(True) is True

    def test_string(self):
        from llmforge.gguf_to_onnx_runner import _gguf_value_to_serialisable
        assert _gguf_value_to_serialisable("hello") == "hello"

    def test_bytes_decoded(self):
        from llmforge.gguf_to_onnx_runner import _gguf_value_to_serialisable
        assert _gguf_value_to_serialisable(b"world") == "world"

    def test_list_recursed(self):
        from llmforge.gguf_to_onnx_runner import _gguf_value_to_serialisable
        result = _gguf_value_to_serialisable([1, "a", b"b"])
        assert result == [1, "a", "b"]

    def test_unknown_type_stringified(self):
        from llmforge.gguf_to_onnx_runner import _gguf_value_to_serialisable
        class Weird:
            def __str__(self): return "weird"
        assert _gguf_value_to_serialisable(Weird()) == "weird"


# ─────────────────────────────────────────────────────────────────────────────
# gguf_to_onnx_runner — _run (mocked gguf + onnx)
# ─────────────────────────────────────────────────────────────────────────────

class TestGgufToOnnxRun:
    def _make_mocks(self):
        """Return (mock_gguf, mock_onnx, mock_np) ready for patch.dict."""
        import numpy as np

        # -- gguf mock --
        mock_field = MagicMock()
        mock_field.parts = [MagicMock()]
        mock_field.data = [0]
        mock_field.parts[0].tolist.return_value = "llama"

        mock_tensor = MagicMock()
        mock_tensor.name = "token_embd.weight"
        mock_tensor.shape = [32, 64]
        mock_tensor.data = np.zeros(32 * 64, dtype=np.float32)

        mock_reader = MagicMock()
        mock_reader.fields = {"general.name": mock_field}
        mock_reader.tensors = [mock_tensor]

        mock_gguf = MagicMock()
        mock_gguf.GGUFReader.return_value = mock_reader

        # -- onnx mock --
        mock_onnx = MagicMock()
        mock_onnx.helper.make_graph.return_value = MagicMock()
        mock_onnx.helper.make_model.return_value = MagicMock()
        mock_onnx.helper.make_opsetid.return_value = MagicMock()
        mock_onnx.checker.check_model.return_value = None
        mock_onnx.save.return_value = None
        # numpy_helper is accessed as onnx.numpy_helper inside _run
        mock_onnx.numpy_helper = MagicMock()
        mock_onnx.numpy_helper.from_array.return_value = MagicMock()
        mock_onnx.TensorProto = MagicMock()

        return mock_gguf, mock_onnx, np

    def test_success_outputs_json(self, capsys, tmp_path):
        from llmforge import gguf_to_onnx_runner
        mock_gguf, mock_onnx, np = self._make_mocks()
        out_path = str(tmp_path / "model.onnx")
        with patch.dict(sys.modules, {"gguf": mock_gguf, "onnx": mock_onnx, "numpy": np}):
            gguf_to_onnx_runner._run({
                "input_path": "fake.gguf",
                "output_path": out_path,
            })
        payload = json.loads(capsys.readouterr().out)
        assert payload["output_path"] == out_path

    def test_missing_input_path_exits(self, capsys):
        from llmforge import gguf_to_onnx_runner
        mock_gguf, mock_onnx, np = self._make_mocks()
        with patch.dict(sys.modules, {"gguf": mock_gguf, "onnx": mock_onnx, "numpy": np}):
            with pytest.raises(SystemExit):
                gguf_to_onnx_runner._run({"output_path": "out.onnx"})

    def test_missing_output_path_exits(self, capsys):
        from llmforge import gguf_to_onnx_runner
        mock_gguf, mock_onnx, np = self._make_mocks()
        with patch.dict(sys.modules, {"gguf": mock_gguf, "onnx": mock_onnx, "numpy": np}):
            with pytest.raises(SystemExit):
                gguf_to_onnx_runner._run({"input_path": "model.gguf"})

    def test_gguf_open_failure_exits(self, capsys):
        from llmforge import gguf_to_onnx_runner
        mock_gguf, mock_onnx, np = self._make_mocks()
        mock_gguf.GGUFReader.side_effect = Exception("corrupt file")
        with patch.dict(sys.modules, {"gguf": mock_gguf, "onnx": mock_onnx, "numpy": np}):
            with pytest.raises(SystemExit) as exc:
                gguf_to_onnx_runner._run({"input_path": "bad.gguf", "output_path": "out.onnx"})
        assert exc.value.code != 0
        payload = json.loads(capsys.readouterr().out)
        assert "error" in payload

    def test_onnx_write_failure_exits(self, capsys):
        from llmforge import gguf_to_onnx_runner
        mock_gguf, mock_onnx, np = self._make_mocks()
        mock_onnx.save.side_effect = Exception("disk full")
        with patch.dict(sys.modules, {"gguf": mock_gguf, "onnx": mock_onnx, "numpy": np}):
            with pytest.raises(SystemExit) as exc:
                gguf_to_onnx_runner._run({"input_path": "ok.gguf", "output_path": "out.onnx"})
        assert exc.value.code != 0


# ─────────────────────────────────────────────────────────────────────────────
# gguf_to_onnx_runner — main
# ─────────────────────────────────────────────────────────────────────────────

class TestGgufToOnnxMain:
    def test_valid_stdin_calls_run(self):
        from llmforge import gguf_to_onnx_runner
        payload = json.dumps({"input_path": "m.gguf", "output_path": "m.onnx"})
        with patch.object(gguf_to_onnx_runner, "_run") as mock_run:
            with patch("sys.stdin", StringIO(payload)):
                gguf_to_onnx_runner.main()
        mock_run.assert_called_once()

    def test_invalid_stdin_json_exits(self, capsys):
        from llmforge import gguf_to_onnx_runner
        with patch("sys.stdin", StringIO("not valid json{{{")):
            with pytest.raises(SystemExit):
                gguf_to_onnx_runner.main()
        assert "error" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────────────────────
# onnx_to_gguf_runner — _error
# ─────────────────────────────────────────────────────────────────────────────

class TestOnnxToGgufError:
    def test_writes_json_error_and_exits(self, capsys):
        from llmforge.onnx_to_gguf_runner import _error
        with pytest.raises(SystemExit) as exc:
            _error("test error")
        assert exc.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["error"] == "test error"


# ─────────────────────────────────────────────────────────────────────────────
# onnx_to_gguf_runner — _check_imports (missing deps)
# ─────────────────────────────────────────────────────────────────────────────

class TestOnnxToGgufCheckImports:
    def test_missing_onnx_exits(self, capsys):
        with _absent("onnx"):
            from llmforge import onnx_to_gguf_runner
            with pytest.raises(SystemExit) as exc:
                onnx_to_gguf_runner._check_imports()
        assert exc.value.code != 0

    def test_missing_gguf_exits(self, capsys):
        with _absent("gguf"):
            from llmforge import onnx_to_gguf_runner
            with pytest.raises(SystemExit) as exc:
                onnx_to_gguf_runner._check_imports()
        assert exc.value.code != 0

    def test_missing_numpy_exits(self, capsys):
        with _absent("numpy"):
            from llmforge import onnx_to_gguf_runner
            with pytest.raises(SystemExit) as exc:
                onnx_to_gguf_runner._check_imports()
        assert exc.value.code != 0


# ─────────────────────────────────────────────────────────────────────────────
# onnx_to_gguf_runner — _write_metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteMetadata:
    def _make_writer(self):
        return MagicMock()

    def _mock_gguf(self):
        return MagicMock()

    def test_writes_string_fields(self):
        from llmforge.onnx_to_gguf_runner import _write_metadata
        writer = self._make_writer()
        with patch.dict(sys.modules, {"gguf": self._mock_gguf()}):
            _write_metadata(writer, {"general.name": "llama"})
        writer.add_string.assert_called_with("general.name", "llama")

    def test_writes_int_fields(self):
        from llmforge.onnx_to_gguf_runner import _write_metadata
        writer = self._make_writer()
        with patch.dict(sys.modules, {"gguf": self._mock_gguf()}):
            _write_metadata(writer, {"llama.context_length": 4096})
        writer.add_int32.assert_called_with("llama.context_length", 4096)

    def test_writes_float_fields(self):
        from llmforge.onnx_to_gguf_runner import _write_metadata
        writer = self._make_writer()
        with patch.dict(sys.modules, {"gguf": self._mock_gguf()}):
            _write_metadata(writer, {"llama.rope_freq_base": 10000.0})
        writer.add_float32.assert_called_with("llama.rope_freq_base", 10000.0)

    def test_writes_bool_fields(self):
        from llmforge.onnx_to_gguf_runner import _write_metadata
        writer = self._make_writer()
        with patch.dict(sys.modules, {"gguf": self._mock_gguf()}):
            _write_metadata(writer, {"general.quantized": True})
        writer.add_bool.assert_called_with("general.quantized", True)

    def test_skips_internal_underscore_keys(self):
        from llmforge.onnx_to_gguf_runner import _write_metadata
        writer = self._make_writer()
        with patch.dict(sys.modules, {"gguf": self._mock_gguf()}):
            _write_metadata(writer, {"_skipped_tensors": [{"name": "x"}]})
        writer.add_string.assert_not_called()
        writer.add_int32.assert_not_called()

    def test_bad_value_does_not_raise(self):
        from llmforge.onnx_to_gguf_runner import _write_metadata
        writer = self._make_writer()
        writer.add_string.side_effect = Exception("boom")
        with patch.dict(sys.modules, {"gguf": self._mock_gguf()}):
            # Should silently swallow the error
            _write_metadata(writer, {"general.name": "test"})


# ─────────────────────────────────────────────────────────────────────────────
# onnx_to_gguf_runner — _run (mocked onnx + gguf)
# ─────────────────────────────────────────────────────────────────────────────

class TestOnnxToGgufRun:
    def _make_mocks(self, doc_string=""):
        import numpy as np

        # -- onnx mock --
        mock_init = MagicMock()
        mock_init.name = "weight"

        mock_graph = MagicMock()
        mock_graph.initializer = [mock_init]

        mock_model = MagicMock()
        mock_model.graph = mock_graph
        mock_model.doc_string = doc_string

        mock_numpy_helper = MagicMock()
        mock_numpy_helper.to_array.return_value = np.zeros((4, 4), dtype=np.float32)

        mock_onnx = MagicMock()
        mock_onnx.load.return_value = mock_model
        mock_onnx.numpy_helper = mock_numpy_helper

        # -- gguf mock --
        mock_writer = MagicMock()
        mock_gguf = MagicMock()
        mock_gguf.GGUFWriter.return_value = mock_writer
        mock_gguf.GGMLQuantizationType = MagicMock()
        mock_gguf.GGMLQuantizationType.F32 = 0

        return mock_onnx, mock_gguf, np

    def test_success_outputs_json(self, capsys, tmp_path):
        from llmforge import onnx_to_gguf_runner
        mock_onnx, mock_gguf, np = self._make_mocks()
        out_path = str(tmp_path / "model.gguf")
        with patch.dict(sys.modules, {"onnx": mock_onnx, "gguf": mock_gguf, "numpy": np}):
            onnx_to_gguf_runner._run({
                "input_path": "fake.onnx",
                "output_path": out_path,
            })
        payload = json.loads(capsys.readouterr().out)
        assert payload["output_path"] == out_path

    def test_metadata_round_tripped_from_doc_string(self, capsys, tmp_path):
        """JSON in doc_string should be parsed and written as GGUF metadata."""
        from llmforge import onnx_to_gguf_runner
        metadata = json.dumps({"general.architecture": "llama", "llama.context_length": 2048})
        mock_onnx, mock_gguf, np = self._make_mocks(doc_string=metadata)
        out_path = str(tmp_path / "model.gguf")
        with patch.dict(sys.modules, {"onnx": mock_onnx, "gguf": mock_gguf, "numpy": np}):
            onnx_to_gguf_runner._run({
                "input_path": "fake.onnx",
                "output_path": out_path,
            })
        writer = mock_gguf.GGUFWriter.return_value
        writer.add_string.assert_any_call("general.architecture", "llama")

    def test_non_json_doc_string_ignored(self, capsys, tmp_path):
        """A plain-text doc_string must not crash the converter."""
        from llmforge import onnx_to_gguf_runner
        mock_onnx, mock_gguf, np = self._make_mocks(doc_string="This is a plain text comment.")
        out_path = str(tmp_path / "model.gguf")
        with patch.dict(sys.modules, {"onnx": mock_onnx, "gguf": mock_gguf, "numpy": np}):
            onnx_to_gguf_runner._run({
                "input_path": "fake.onnx",
                "output_path": out_path,
            })
        payload = json.loads(capsys.readouterr().out)
        assert payload["output_path"] == out_path

    def test_missing_input_path_exits(self, capsys):
        from llmforge import onnx_to_gguf_runner
        mock_onnx, mock_gguf, np = self._make_mocks()
        with patch.dict(sys.modules, {"onnx": mock_onnx, "gguf": mock_gguf, "numpy": np}):
            with pytest.raises(SystemExit):
                onnx_to_gguf_runner._run({"output_path": "out.gguf"})

    def test_missing_output_path_exits(self, capsys):
        from llmforge import onnx_to_gguf_runner
        mock_onnx, mock_gguf, np = self._make_mocks()
        with patch.dict(sys.modules, {"onnx": mock_onnx, "gguf": mock_gguf, "numpy": np}):
            with pytest.raises(SystemExit):
                onnx_to_gguf_runner._run({"input_path": "model.onnx"})

    def test_onnx_load_failure_exits(self, capsys):
        from llmforge import onnx_to_gguf_runner
        mock_onnx, mock_gguf, np = self._make_mocks()
        mock_onnx.load.side_effect = Exception("corrupt onnx")
        with patch.dict(sys.modules, {"onnx": mock_onnx, "gguf": mock_gguf, "numpy": np}):
            with pytest.raises(SystemExit) as exc:
                onnx_to_gguf_runner._run({"input_path": "bad.onnx", "output_path": "out.gguf"})
        assert exc.value.code != 0
        payload = json.loads(capsys.readouterr().out)
        assert "error" in payload

    def test_no_tensors_exits(self, capsys):
        from llmforge import onnx_to_gguf_runner
        mock_onnx, mock_gguf, np = self._make_mocks()
        mock_onnx.load.return_value.graph.initializer = []
        with patch.dict(sys.modules, {"onnx": mock_onnx, "gguf": mock_gguf, "numpy": np}):
            with pytest.raises(SystemExit) as exc:
                onnx_to_gguf_runner._run({"input_path": "empty.onnx", "output_path": "out.gguf"})
        assert exc.value.code != 0

    def test_gguf_write_failure_exits(self, capsys):
        from llmforge import onnx_to_gguf_runner
        mock_onnx, mock_gguf, np = self._make_mocks()
        mock_gguf.GGUFWriter.side_effect = Exception("disk full")
        with patch.dict(sys.modules, {"onnx": mock_onnx, "gguf": mock_gguf, "numpy": np}):
            with pytest.raises(SystemExit) as exc:
                onnx_to_gguf_runner._run({"input_path": "ok.onnx", "output_path": "out.gguf"})
        assert exc.value.code != 0


# ─────────────────────────────────────────────────────────────────────────────
# onnx_to_gguf_runner — main
# ─────────────────────────────────────────────────────────────────────────────

class TestOnnxToGgufMain:
    def test_valid_stdin_calls_run(self):
        from llmforge import onnx_to_gguf_runner
        payload = json.dumps({"input_path": "m.onnx", "output_path": "m.gguf"})
        with patch.object(onnx_to_gguf_runner, "_run") as mock_run:
            with patch("sys.stdin", StringIO(payload)):
                onnx_to_gguf_runner.main()
        mock_run.assert_called_once()

    def test_invalid_stdin_json_exits(self, capsys):
        from llmforge import onnx_to_gguf_runner
        with patch("sys.stdin", StringIO("{{not json")):
            with pytest.raises(SystemExit):
                onnx_to_gguf_runner.main()
        assert "error" in capsys.readouterr().out
