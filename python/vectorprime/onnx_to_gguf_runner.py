"""
stdin→stdout JSON bridge for ONNX → GGUF conversion.

Input JSON (stdin):
  {
    "input_path":  "/path/to/model.onnx",
    "output_path": "/path/to/model.gguf"
  }

Output JSON (stdout):
  { "output_path": "/path/to/model.gguf" }
  OR
  { "error": "message" }   -- on any failure; never raise an unhandled exception

How it works
------------
1. Loads the ONNX model with the ``onnx`` package.
2. Extracts every initializer (weight tensor) and converts it to float32.
3. If the model's ``doc_string`` contains a JSON blob (produced by
   ``gguf_to_onnx_runner.py``), parses it and writes the key/value pairs
   back into the GGUF file so that the metadata round-trips faithfully.
4. Writes a GGUF v3 file whose tensors are the extracted weights.

Required Python packages:  onnx  gguf  numpy
  pip install onnx gguf numpy
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
    """Verify all required packages are importable; return (onnx, gguf, np)."""
    try:
        import onnx  # type: ignore[import]
        from onnx import numpy_helper  # noqa: F401
    except ImportError:
        _error("onnx package not installed — run: pip install onnx")

    try:
        import gguf  # type: ignore[import]
    except ImportError:
        _error("gguf package not installed — run: pip install gguf")

    try:
        import numpy as np  # type: ignore[import]
    except ImportError:
        _error("numpy package not installed — run: pip install numpy")

    import onnx  # type: ignore[import]
    import gguf  # type: ignore[import]
    import numpy as np  # type: ignore[import]
    return onnx, gguf, np


_GGUF_STRING_KEYS = {
    "general.name",
    "general.architecture",
    "general.description",
    "general.author",
    "general.url",
    "general.license",
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
}


def _write_metadata(writer: object, metadata: dict) -> None:  # type: ignore[type-arg]
    """Write GGUF key/value metadata from a dict, best-effort."""
    import gguf  # type: ignore[import]

    for key, value in metadata.items():
        if key.startswith("_"):
            continue  # internal vectorprime fields (e.g. _skipped_tensors)
        try:
            if isinstance(value, bool):
                writer.add_bool(key, value)
            elif isinstance(value, int):
                writer.add_int32(key, value)
            elif isinstance(value, float):
                writer.add_float32(key, value)
            elif isinstance(value, str):
                writer.add_string(key, value)
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                writer.add_array(key, value)
            elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                writer.add_array(key, value)
            # Other types (nested dicts, etc.) are intentionally skipped.
        except Exception:
            pass  # best-effort; never abort for a bad metadata field


# ──────────────────────────────────────────────────────────────────────────────
# Conversion
# ──────────────────────────────────────────────────────────────────────────────

def _run(request: dict) -> None:
    onnx, gguf, np = _check_imports()
    from onnx import numpy_helper  # type: ignore[import]

    input_path: str = request.get("input_path", "")
    output_path: str = request.get("output_path", "")

    if not input_path:
        _error("input_path is required")
    if not output_path:
        _error("output_path is required")

    # ── 1. Load ONNX model ────────────────────────────────────────────────────
    try:
        model = onnx.load(input_path)
    except Exception as exc:
        _error(f"failed to load ONNX model '{input_path}': {exc}")
        return

    # ── 2. Parse round-trip metadata from doc_string ──────────────────────────
    metadata: dict = {}
    if model.doc_string:
        try:
            metadata = json.loads(model.doc_string)
        except (json.JSONDecodeError, ValueError):
            pass  # doc_string may be plain text from non-vectorprime models

    # ── 3. Extract initializer (weight) tensors ───────────────────────────────
    tensors: list[tuple[str, object]] = []
    try:
        for initializer in model.graph.initializer:
            try:
                np_data = numpy_helper.to_array(initializer).astype(np.float32)
                tensors.append((initializer.name, np_data))
            except Exception as tensor_exc:
                metadata.setdefault("_skipped_tensors", []).append(  # type: ignore[union-attr]
                    {"name": initializer.name, "reason": str(tensor_exc)}
                )
    except Exception as exc:
        _error(f"failed to iterate model initializers: {exc}")
        return

    if not tensors:
        _error("no weight tensors found in the ONNX model")
        return

    # ── 4. Write GGUF file ────────────────────────────────────────────────────
    try:
        writer = gguf.GGUFWriter(output_path, arch=metadata.get("general.architecture", "unknown"))

        # Write key/value metadata.
        _write_metadata(writer, metadata)

        # Write weight tensors (F32).
        for name, np_data in tensors:
            writer.add_tensor(name, np_data, raw_dtype=gguf.GGMLQuantizationType.F32)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    except Exception as exc:
        _error(f"failed to write GGUF file: {exc}")
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
