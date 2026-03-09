"""CLI integration tests executed via subprocess."""

import json
import os
import subprocess

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
_GGUF = os.path.join(_FIXTURES_DIR, "tiny.gguf")

skip_no_fixtures = pytest.mark.skipif(
    not os.path.exists(_GGUF),
    reason="Model fixtures not downloaded. See tests/fixtures/FIXTURES.md",
)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["llmforge", *args],
        capture_output=True,
        text=True,
    )


# ── always-run tests ──────────────────────────────────────────────────────────


def test_help_exits_zero():
    result = _run("--help")
    assert result.returncode == 0


def test_help_lists_subcommands():
    result = _run("--help")
    assert "profile" in result.stdout
    assert "optimize" in result.stdout
    assert "export-ollama" in result.stdout


def test_profile_exits_zero():
    result = _run("profile")
    assert result.returncode == 0


def test_profile_prints_valid_json():
    result = _run("profile")
    assert result.returncode == 0
    data = json.loads(result.stdout)  # raises on invalid JSON
    assert "cpu" in data


def test_optimize_missing_file_nonzero():
    result = _run("optimize", "/nonexistent/model.gguf")
    assert result.returncode != 0


def test_optimize_missing_file_stderr_error():
    result = _run("optimize", "/nonexistent/model.gguf")
    combined = result.stderr.lower()
    assert "error" in combined, f"Expected 'error' in stderr, got: {result.stderr!r}"


def test_optimize_unknown_extension_nonzero():
    """Auto-detection must fail and produce a non-zero exit for unknown extensions."""
    result = _run("optimize", "model.safetensors")
    assert result.returncode != 0


# ── fixture-dependent tests ───────────────────────────────────────────────────


@skip_no_fixtures
def test_optimize_gguf_full_pipeline():
    result = _run("optimize", _GGUF, "--format", "gguf")
    assert result.returncode == 0
    assert "tokens/sec" in result.stdout
