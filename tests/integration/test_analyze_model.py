"""
Integration tests for vectorprime.analyze_model() Python binding.

Tests verify that the PyO3 `analyze_model` function correctly:
  - Returns a dict with all required ModelIR keys
  - Reports the correct format string ("gguf" or "onnx")
  - Raises RuntimeError for nonexistent paths
  - Raises RuntimeError for unrecognised extensions
  - Works against a minimal synthetic GGUF (no large fixtures required)
  - Works against a minimal synthetic ONNX (no onnxruntime required)
  - Skips gracefully when the native extension is not compiled

All tests use synthetic fixtures constructed in tmp_path or with stdlib
primitives — no model downloads, no GPU, no llama-cli.
"""

import os
import struct

import pytest

# ─── module-level guard ───────────────────────────────────────────────────────
# Skip the entire file if the native extension is not compiled.
# This keeps CI green without requiring `maturin develop` to have been run.
vectorprime = pytest.importorskip(
    "vectorprime._vectorprime",
    reason="vectorprime native extension not compiled — run `maturin develop`",
)

# ─── real fixture paths (optional) ───────────────────────────────────────────
_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
_GGUF_FIXTURE = os.path.join(_FIXTURES_DIR, "tiny.gguf")
_ONNX_FIXTURE = os.path.join(_FIXTURES_DIR, "tiny_onnx", "model.onnx")


# ─── GGUF synthetic fixture builder ──────────────────────────────────────────

def _write_u32_le(buf: bytearray, v: int) -> None:
    buf += struct.pack("<I", v)


def _write_u64_le(buf: bytearray, v: int) -> None:
    buf += struct.pack("<Q", v)


def _write_gguf_string(buf: bytearray, s: str) -> None:
    encoded = s.encode("utf-8")
    _write_u64_le(buf, len(encoded))
    buf += encoded


# GGUF value-type codes (mirrors GgufValueType in the Rust crate)
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_STRING = 8


def _make_gguf_bytes(
    *,
    architecture: str | None = None,
    block_count: int | None = None,
    context_length: int | None = None,
    embedding_length: int | None = None,
    param_count: int | None = None,
) -> bytes:
    """
    Build a minimal well-formed GGUF byte blob (version 3, zero tensors).

    Only includes the KV entries that are explicitly requested via kwargs.
    This exercises the same byte-layout that parse_gguf() reads.
    """
    kvs: list[bytearray] = []

    def _kv_uint64(key: str, val: int) -> bytearray:
        kv = bytearray()
        _write_gguf_string(kv, key)
        _write_u32_le(kv, _GGUF_TYPE_UINT64)
        _write_u64_le(kv, val)
        return kv

    def _kv_uint32(key: str, val: int) -> bytearray:
        kv = bytearray()
        _write_gguf_string(kv, key)
        _write_u32_le(kv, _GGUF_TYPE_UINT32)
        _write_u32_le(kv, val)
        return kv

    def _kv_string(key: str, val: str) -> bytearray:
        kv = bytearray()
        _write_gguf_string(kv, key)
        _write_u32_le(kv, _GGUF_TYPE_STRING)
        _write_gguf_string(kv, val)
        return kv

    if architecture is not None:
        kvs.append(_kv_string("general.architecture", architecture))
    if param_count is not None:
        kvs.append(_kv_uint64("general.parameter_count", param_count))
    if block_count is not None:
        arch = architecture or "llama"
        kvs.append(_kv_uint32(f"{arch}.block_count", block_count))
    if context_length is not None:
        arch = architecture or "llama"
        kvs.append(_kv_uint32(f"{arch}.context_length", context_length))
    if embedding_length is not None:
        arch = architecture or "llama"
        kvs.append(_kv_uint32(f"{arch}.embedding_length", embedding_length))

    header = bytearray()
    header += b"GGUF"
    _write_u32_le(header, 3)       # version = 3
    _write_u64_le(header, 0)       # tensor_count = 0
    _write_u64_le(header, len(kvs))  # kv_count
    for kv in kvs:
        header += kv

    return bytes(header)


def _write_synthetic_gguf(
    path,
    *,
    architecture: str | None = "llama",
    block_count: int | None = 32,
    context_length: int | None = 4096,
    embedding_length: int | None = 4096,
    param_count: int | None = None,
) -> None:
    """Write a synthetic GGUF file with sane defaults for most tests."""
    data = _make_gguf_bytes(
        architecture=architecture,
        block_count=block_count,
        context_length=context_length,
        embedding_length=embedding_length,
        param_count=param_count,
    )
    path.write_bytes(data)


# ─── Minimal ONNX protobuf builder ───────────────────────────────────────────

def _make_minimal_onnx_bytes() -> bytes:
    """
    Build a minimal ONNX ModelProto protobuf with:
      - ir_version = 7            (field 1, varint)
      - opset_import[0].version=17 (field 8, embedded message: field 2 varint)
      - graph with 0 nodes and 0 initializers (field 7, embedded message)
    This is enough for parse_onnx() to produce a valid ModelIR with all-None
    optional fields rather than returning an error.
    """

    def _varint(n: int) -> bytes:
        """Encode n as a protobuf varint."""
        out = []
        while n > 0x7F:
            out.append((n & 0x7F) | 0x80)
            n >>= 7
        out.append(n & 0x7F)
        return bytes(out)

    def _field(field_number: int, wire_type: int, payload: bytes) -> bytes:
        tag = (field_number << 3) | wire_type
        if wire_type == 0:
            return _varint(tag) + payload
        elif wire_type == 2:
            return _varint(tag) + _varint(len(payload)) + payload
        raise ValueError(f"unsupported wire_type {wire_type}")

    # ir_version = 7 (field 1, varint wire-type 0)
    ir_version = _field(1, 0, _varint(7))

    # opset_import entry: OperatorSetIdProto.version = 17 (field 2, varint)
    opset_entry = _field(2, 0, _varint(17))
    opset_import = _field(8, 2, opset_entry)

    # graph: empty GraphProto (field 7, length-delimited)
    graph = _field(7, 2, b"")

    return ir_version + opset_import + graph


# ─────────────────────────────────────────────────────────────────────────────
# Tests: required dict keys
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {"format", "param_count", "architecture", "context_length", "layer_count"}


class TestAnalyzeModelDictStructure:
    """Verify the returned dict always contains all required ModelIR keys."""

    def test_gguf_returns_all_required_keys(self, tmp_path):
        model = tmp_path / "model.gguf"
        _write_synthetic_gguf(model)
        result = vectorprime.analyze_model(str(model))
        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(result.keys()), (
            f"analyze_model missing keys: {REQUIRED_KEYS - result.keys()}"
        )

    def test_gguf_extra_keys_not_present(self, tmp_path):
        """No unexpected extra keys should appear in the dict."""
        model = tmp_path / "model.gguf"
        _write_synthetic_gguf(model)
        result = vectorprime.analyze_model(str(model))
        assert set(result.keys()) == REQUIRED_KEYS, (
            f"Unexpected keys: {set(result.keys()) - REQUIRED_KEYS}"
        )

    def test_onnx_returns_all_required_keys(self, tmp_path):
        model = tmp_path / "model.onnx"
        model.write_bytes(_make_minimal_onnx_bytes())
        result = vectorprime.analyze_model(str(model))
        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(result.keys()), (
            f"analyze_model missing keys: {REQUIRED_KEYS - result.keys()}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: format field correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeModelFormatField:
    """The 'format' key must correctly reflect the file type."""

    def test_gguf_format_is_gguf(self, tmp_path):
        model = tmp_path / "model.gguf"
        _write_synthetic_gguf(model)
        result = vectorprime.analyze_model(str(model))
        assert result["format"] == "gguf"

    def test_onnx_format_is_onnx(self, tmp_path):
        model = tmp_path / "model.onnx"
        model.write_bytes(_make_minimal_onnx_bytes())
        result = vectorprime.analyze_model(str(model))
        assert result["format"] == "onnx"

    def test_format_is_string(self, tmp_path):
        model = tmp_path / "model.gguf"
        _write_synthetic_gguf(model)
        result = vectorprime.analyze_model(str(model))
        assert isinstance(result["format"], str)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: optional field types
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeModelFieldTypes:
    """Optional fields must be int or None — never strings or floats."""

    def test_param_count_is_int_or_none(self, tmp_path):
        model = tmp_path / "model.gguf"
        _write_synthetic_gguf(model)
        result = vectorprime.analyze_model(str(model))
        assert result["param_count"] is None or isinstance(result["param_count"], int)

    def test_architecture_is_str_or_none(self, tmp_path):
        model = tmp_path / "model.gguf"
        _write_synthetic_gguf(model)
        result = vectorprime.analyze_model(str(model))
        assert result["architecture"] is None or isinstance(result["architecture"], str)

    def test_context_length_is_int_or_none(self, tmp_path):
        model = tmp_path / "model.gguf"
        _write_synthetic_gguf(model)
        result = vectorprime.analyze_model(str(model))
        assert result["context_length"] is None or isinstance(result["context_length"], int)

    def test_layer_count_is_int_or_none(self, tmp_path):
        model = tmp_path / "model.gguf"
        _write_synthetic_gguf(model)
        result = vectorprime.analyze_model(str(model))
        assert result["layer_count"] is None or isinstance(result["layer_count"], int)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: GGUF metadata extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeModelGgufMetadata:
    """Verify that known KV entries are surfaced correctly."""

    def test_architecture_extracted_correctly(self, tmp_path):
        model = tmp_path / "model.gguf"
        _write_synthetic_gguf(model, architecture="mistral")
        result = vectorprime.analyze_model(str(model))
        assert result["architecture"] == "mistral"

    def test_layer_count_extracted_from_block_count(self, tmp_path):
        model = tmp_path / "model.gguf"
        _write_synthetic_gguf(model, architecture="phi", block_count=24, embedding_length=2048)
        result = vectorprime.analyze_model(str(model))
        assert result["layer_count"] == 24

    def test_context_length_extracted(self, tmp_path):
        model = tmp_path / "model.gguf"
        _write_synthetic_gguf(model, context_length=8192)
        result = vectorprime.analyze_model(str(model))
        assert result["context_length"] == 8192

    def test_explicit_param_count_used_when_present(self, tmp_path):
        model = tmp_path / "model.gguf"
        _write_synthetic_gguf(model, param_count=7_000_000_000)
        result = vectorprime.analyze_model(str(model))
        assert result["param_count"] == 7_000_000_000

    def test_param_count_fallback_computed_from_arch_keys(self, tmp_path):
        """Without general.parameter_count, param_count = 12 * block_count * embedding_length."""
        model = tmp_path / "model.gguf"
        _write_synthetic_gguf(
            model,
            architecture="llama",
            block_count=32,
            context_length=4096,
            embedding_length=4096,
            param_count=None,
        )
        result = vectorprime.analyze_model(str(model))
        expected = 12 * 32 * 4096
        assert result["param_count"] == expected, (
            f"Expected fallback param_count={expected}, got {result['param_count']}"
        )

    def test_param_count_none_when_no_arch_keys(self, tmp_path):
        """Empty GGUF header — no parameter metadata available."""
        model = tmp_path / "model.gguf"
        model.write_bytes(_make_gguf_bytes())  # no KV entries at all
        result = vectorprime.analyze_model(str(model))
        assert result["param_count"] is None
        assert result["architecture"] is None
        assert result["context_length"] is None
        assert result["layer_count"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Tests: error cases
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeModelErrors:
    """analyze_model must raise RuntimeError for bad inputs."""

    def test_nonexistent_path_raises_runtime_error(self):
        with pytest.raises(RuntimeError):
            vectorprime.analyze_model("/nonexistent/path/that/does/not/exist.gguf")

    def test_unrecognised_extension_raises_runtime_error(self, tmp_path):
        bad = tmp_path / "model.safetensors"
        bad.write_bytes(b"dummy")
        with pytest.raises(RuntimeError):
            vectorprime.analyze_model(str(bad))

    def test_bad_gguf_magic_raises_runtime_error(self, tmp_path):
        bad = tmp_path / "corrupt.gguf"
        bad.write_bytes(b"NOTG\x00\x00\x00\x00" * 4)
        with pytest.raises(RuntimeError):
            vectorprime.analyze_model(str(bad))

    def test_empty_gguf_raises_runtime_error(self, tmp_path):
        """An empty file with .gguf extension has no magic — must fail."""
        empty = tmp_path / "empty.gguf"
        empty.write_bytes(b"")
        with pytest.raises(RuntimeError):
            vectorprime.analyze_model(str(empty))

    def test_error_message_is_string(self):
        """The RuntimeError message should be a non-empty string."""
        try:
            vectorprime.analyze_model("/no/such/model.gguf")
        except RuntimeError as exc:
            assert str(exc), "RuntimeError message should not be empty"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: ONNX metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeModelOnnx:
    """Verify ONNX-specific behavior."""

    def test_onnx_architecture_is_none(self, tmp_path):
        """ONNX graphs carry no named architecture field — must be None."""
        model = tmp_path / "model.onnx"
        model.write_bytes(_make_minimal_onnx_bytes())
        result = vectorprime.analyze_model(str(model))
        assert result["architecture"] is None

    def test_onnx_context_length_is_none(self, tmp_path):
        """ONNX format does not encode context_length — must be None."""
        model = tmp_path / "model.onnx"
        model.write_bytes(_make_minimal_onnx_bytes())
        result = vectorprime.analyze_model(str(model))
        assert result["context_length"] is None

    def test_onnx_param_count_is_none_for_empty_graph(self, tmp_path):
        """An ONNX with no initializers yields param_count=None."""
        model = tmp_path / "model.onnx"
        model.write_bytes(_make_minimal_onnx_bytes())
        result = vectorprime.analyze_model(str(model))
        assert result["param_count"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Tests: real fixture (conditional — skipped in CI)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(_GGUF_FIXTURE),
    reason=(
        "GGUF fixture not present — download with: "
        "curl -L <tinyllama-url> -o tests/fixtures/tiny.gguf"
    ),
)
class TestAnalyzeModelRealGgufFixture:
    """Tests that run only when tests/fixtures/tiny.gguf has been downloaded."""

    def test_real_gguf_returns_all_keys(self):
        result = vectorprime.analyze_model(_GGUF_FIXTURE)
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_real_gguf_format_is_gguf(self):
        result = vectorprime.analyze_model(_GGUF_FIXTURE)
        assert result["format"] == "gguf"

    def test_real_gguf_param_count_is_int_or_none(self):
        result = vectorprime.analyze_model(_GGUF_FIXTURE)
        assert result["param_count"] is None or isinstance(result["param_count"], int)

    def test_real_gguf_architecture_is_str_or_none(self):
        result = vectorprime.analyze_model(_GGUF_FIXTURE)
        assert result["architecture"] is None or isinstance(result["architecture"], str)


@pytest.mark.skipif(
    not os.path.exists(_ONNX_FIXTURE),
    reason=(
        "ONNX fixture not present — export with: "
        "optimum-cli export onnx --model distilbert-base-uncased tests/fixtures/tiny_onnx/"
    ),
)
class TestAnalyzeModelRealOnnxFixture:
    """Tests that run only when tests/fixtures/tiny_onnx/model.onnx exists."""

    def test_real_onnx_returns_all_keys(self):
        result = vectorprime.analyze_model(_ONNX_FIXTURE)
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_real_onnx_format_is_onnx(self):
        result = vectorprime.analyze_model(_ONNX_FIXTURE)
        assert result["format"] == "onnx"

    def test_real_onnx_param_count_positive_when_present(self):
        result = vectorprime.analyze_model(_ONNX_FIXTURE)
        if result["param_count"] is not None:
            assert result["param_count"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: integration via vectorprime package import
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeModelViaPackage:
    """Verify analyze_model is accessible via the top-level vectorprime package."""

    def test_callable_from_vectorprime_package(self):
        import vectorprime as pkg
        assert hasattr(pkg, "analyze_model"), (
            "analyze_model not re-exported from vectorprime/__init__.py"
        )
        assert callable(pkg.analyze_model)

    def test_package_and_module_same_function(self, tmp_path):
        """vectorprime.analyze_model and vectorprime._vectorprime.analyze_model must agree."""
        import vectorprime as pkg

        model = tmp_path / "model.gguf"
        _write_synthetic_gguf(model)

        result_pkg = pkg.analyze_model(str(model))
        result_mod = vectorprime.analyze_model(str(model))

        assert result_pkg == result_mod
