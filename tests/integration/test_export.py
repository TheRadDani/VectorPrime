"""Integration tests for export_ollama() binding."""

import json
import os

import pytest

_llmforge = pytest.importorskip(
    "llmforge._llmforge",
    reason="Native extension not compiled — run: maturin develop",
)

# ── fixture guards ────────────────────────────────────────────────────────────

_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
_GGUF = os.path.join(_FIXTURES_DIR, "tiny.gguf")

skip_no_fixtures = pytest.mark.skipif(
    not os.path.exists(_GGUF),
    reason="Model fixtures not downloaded. See tests/fixtures/FIXTURES.md",
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _optimized_result():
    return _llmforge.optimize(_GGUF, "gguf")


# ── tests ─────────────────────────────────────────────────────────────────────


@skip_no_fixtures
def test_export_creates_files(tmp_path):
    """export_ollama creates Modelfile, model.gguf, and metadata.json."""
    result = _optimized_result()
    _llmforge.export_ollama(result, str(tmp_path))
    assert (tmp_path / "Modelfile").exists(), "Modelfile missing"
    assert (tmp_path / "model.gguf").exists(), "model.gguf missing"
    assert (tmp_path / "metadata.json").exists(), "metadata.json missing"


@skip_no_fixtures
def test_modelfile_has_from(tmp_path):
    """Modelfile must start with a FROM directive pointing to ./model.gguf."""
    result = _optimized_result()
    _llmforge.export_ollama(result, str(tmp_path))
    content = (tmp_path / "Modelfile").read_text(encoding="utf-8")
    assert "FROM ./model.gguf" in content


@skip_no_fixtures
def test_manifest_has_commands(tmp_path):
    """The returned manifest JSON must contain exactly 2 ollama_commands."""
    result = _optimized_result()
    manifest_json = _llmforge.export_ollama(result, str(tmp_path))
    manifest = json.loads(manifest_json)
    commands = manifest.get("ollama_commands", [])
    assert len(commands) == 2, f"expected 2 commands, got {commands}"
    assert any("ollama create" in c for c in commands)
    assert any("ollama run" in c for c in commands)
