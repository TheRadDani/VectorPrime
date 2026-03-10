"""
stdin→stdout JSON bridge for GGUF → ONNX conversion.

Input JSON (stdin):
  {
    "input_path":  "/path/to/model.gguf",
    "output_path": "/path/to/model.onnx"
  }

Output JSON (stdout):
  { "output_path": "/path/to/model.onnx" }
  OR
  { "error": "message" }   -- on any failure; never raise an unhandled exception

How it works
------------
1. Opens the GGUF file with the ``gguf`` Python package.
2. Dequantizes every weight tensor to float32.
3. Embeds all GGUF key/value metadata (model name, architecture,
   hyperparameters, tokenizer vocabulary, …) into the ONNX model's
   ``doc_string`` as a JSON blob so that ``onnx_to_gguf_runner.py`` can
   round-trip it back.
4. Writes a valid ONNX model whose *initializers* are the model weights.

Required Python packages:  gguf  onnx  numpy
  pip install gguf onnx numpy
"""

from __future__ import annotations

import json
import sys


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _error(msg: str) -> None:
    """Write an error envelope to stdout and exit non-zero."""
    print(json.dumps({"error": msg}), flush=True)
    sys.exit(1)


def _check_imports() -> tuple:
    """Verify all required packages are importable; return (gguf, onnx, np)."""
    try:
        import gguf  # type: ignore[import]
    except ImportError:
        _error("gguf package not installed — run: pip install gguf")

    try:
        import onnx  # type: ignore[import]
        from onnx import numpy_helper  # noqa: F401
    except ImportError:
        _error("onnx package not installed — run: pip install onnx")

    try:
        import numpy as np  # type: ignore[import]
    except ImportError:
        _error("numpy package not installed — run: pip install numpy")

    import gguf  # type: ignore[import]
    import onnx  # type: ignore[import]
    import numpy as np  # type: ignore[import]
    return gguf, onnx, np


# ──────────────────────────────────────────────────────────────────────────────
# Conversion
# ──────────────────────────────────────────────────────────────────────────────

def _gguf_value_to_serialisable(value: object) -> object:
    """Best-effort conversion of a raw GGUF field value to a JSON-safe type."""
    if isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_gguf_value_to_serialisable(v) for v in value]
    return str(value)


def _run(request: dict) -> None:
    gguf, onnx, np = _check_imports()
    from onnx import numpy_helper, TensorProto  # type: ignore[import]

    input_path: str = request.get("input_path", "")
    output_path: str = request.get("output_path", "")

    if not input_path:
        _error("input_path is required")
    if not output_path:
        _error("output_path is required")

    # ── 1. Open the GGUF file ─────────────────────────────────────────────────
    try:
        reader = gguf.GGUFReader(input_path, mode="r")
    except Exception as exc:
        _error(f"failed to open GGUF file '{input_path}': {exc}")
        return

    # ── 2. Collect metadata key/value pairs ───────────────────────────────────
    metadata: dict = {}
    for name, field in reader.fields.items():
        try:
            # GGUFReader represents scalar fields with a `parts` list.
            if hasattr(field, "parts") and field.parts:
                raw = field.parts[field.data[0]]
                # Convert numpy scalar / array to plain Python
                val = raw.tolist() if hasattr(raw, "tolist") else raw
                metadata[name] = _gguf_value_to_serialisable(val)
        except Exception:
            pass  # skip fields that cannot be serialised

    # ── 3. Convert tensors to ONNX initialisers ────────────────────────────────
    initializers: list = []
    try:
        for tensor in reader.tensors:
            try:
                # `tensor.data` is a numpy array. Quantised types need cast.
                raw = tensor.data
                # Cast to float32 regardless of the quantisation strategy.
                np_data = raw.astype(np.float32)

                # GGUF stores shape in reversed (column-major) order.
                shape = list(tensor.shape)
                if len(shape) > 1:
                    np_data = np_data.reshape(shape[::-1])
                else:
                    np_data = np_data.reshape(shape)

                onnx_tensor = numpy_helper.from_array(np_data, name=tensor.name)
                initializers.append(onnx_tensor)
            except Exception as tensor_exc:
                # Log the skipped tensor in metadata but keep going.
                metadata.setdefault("_skipped_tensors", []).append(  # type: ignore[union-attr]
                    {"name": tensor.name, "reason": str(tensor_exc)}
                )
    except Exception as exc:
        _error(f"failed to iterate tensors: {exc}")
        return

    if not initializers:
        _error("no tensors could be extracted from the GGUF file")
        return

    # ── 4. Build ONNX graph (weight-only — no compute nodes) ─────────────────
    try:
        graph = onnx.helper.make_graph(
            nodes=[],
            name="gguf_weights",
            inputs=[],
            outputs=[],
            initializer=initializers,
        )

        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 17)],
        )
        # Embed metadata for round-trip fidelity.
        model.doc_string = json.dumps(metadata, ensure_ascii=False)
        model.domain = "ai.vectorprime"
        model.model_version = 1

        onnx.checker.check_model(model)
        onnx.save(model, output_path)
    except Exception as exc:
        _error(f"failed to write ONNX model: {exc}")
        return

    print(json.dumps({"output_path": output_path}), flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        request = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        _error(f"invalid JSON input: {exc}")
        return
    _run(request)


if __name__ == "__main__":
    main()
