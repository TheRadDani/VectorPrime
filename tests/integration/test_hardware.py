"""Integration tests for hardware profiling via Python bindings."""

import json

import pytest

_llmforge = pytest.importorskip(
    "llmforge._llmforge",
    reason="Native extension not compiled — run: maturin develop",
)


def test_profile_has_cores():
    hw = _llmforge.profile_hardware()
    assert hw.cpu_cores >= 1


def test_profile_json_valid():
    hw = _llmforge.profile_hardware()
    data = json.loads(hw.to_json())
    assert "cpu" in data
    assert "core_count" in data["cpu"]


def test_ram_positive():
    hw = _llmforge.profile_hardware()
    assert hw.ram_total_mb > 0
