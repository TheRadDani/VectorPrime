"""Integration tests for hardware profiling via Python bindings."""

import json

import pytest

_vectorprime = pytest.importorskip(
    "vectorprime._vectorprime",
    reason="Native extension not compiled — run: maturin develop",
)


def test_profile_has_cores():
    hw = _vectorprime.profile_hardware()
    assert hw.cpu_cores >= 1


def test_profile_json_valid():
    hw = _vectorprime.profile_hardware()
    data = json.loads(hw.to_json())
    assert "cpu" in data
    assert "core_count" in data["cpu"]


def test_ram_positive():
    hw = _vectorprime.profile_hardware()
    assert hw.ram_total_mb > 0
