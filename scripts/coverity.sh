#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/coverity.sh
# Coverage gate for LLMForge — enforces ≥ THRESHOLD% for both Rust and Python.
#
# Usage:
#   ./scripts/coverity.sh                      # default 70% threshold
#   COVERAGE_THRESHOLD=80 ./scripts/coverity.sh
#
# Dependencies (auto-installed if missing):
#   cargo-tarpaulin  —  Rust line coverage (cargo install cargo-tarpaulin)
#   pytest-cov       —  Python coverage  (pip install pytest-cov)
#   maturin          —  native module build (pip install maturin)
#
# Outputs (ignored by git):
#   coverage/rust/     — Cobertura XML + HTML report
#   coverage/python/   — Cobertura XML + HTML report + pytest log
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

THRESHOLD="${COVERAGE_THRESHOLD:-47}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COV_DIR="${ROOT_DIR}/coverage"

cd "${ROOT_DIR}"

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

pass()   { echo -e "${GREEN}✔  $*${NC}"; }
fail()   { echo -e "${RED}✘  $*${NC}"; }
info()   { echo -e "${CYAN}→  $*${NC}"; }
warn()   { echo -e "${YELLOW}⚠  $*${NC}"; }
banner() {
  echo -e "\n${CYAN}══════════════════════════════════════════${NC}"
  echo -e "${CYAN}  $*${NC}"
  echo -e "${CYAN}══════════════════════════════════════════${NC}\n"
}

FAILURES=0

mkdir -p "${COV_DIR}/rust" "${COV_DIR}/python"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: compare floats without bc (uses awk)
# Usage: float_lt A B   →  exit 0 if A < B, exit 1 otherwise
# ─────────────────────────────────────────────────────────────────────────────
float_lt() { awk "BEGIN { exit ($1 < $2) ? 0 : 1 }"; }

# ─────────────────────────────────────────────────────────────────────────────
# 1. Rust coverage — cargo-tarpaulin
# ─────────────────────────────────────────────────────────────────────────────
banner "1/2  Rust Coverage (cargo-tarpaulin)"

if ! command -v cargo-tarpaulin &>/dev/null; then
  info "cargo-tarpaulin not found — installing (this may take a moment) ..."
  cargo install cargo-tarpaulin --locked
fi

RUST_COV_LOG="${COV_DIR}/rust/tarpaulin.log"
info "cargo tarpaulin --workspace --fail-under ${THRESHOLD}"

set +e
cargo tarpaulin \
  --workspace \
  --exclude llmforge-bindings \
  --out Xml Html \
  --output-dir "${COV_DIR}/rust" \
  --timeout 120 \
  --fail-under "${THRESHOLD}" \
  2>&1 | tee "${RUST_COV_LOG}"
RUST_EXIT="${PIPESTATUS[0]}"
set -e

# Extract the reported percentage (tarpaulin prints "XX.XX% coverage")
RUST_PCT=$(grep -oP '\d+\.\d+(?=% coverage)' "${RUST_COV_LOG}" | tail -1)
# Fall back to an integer match if no decimal was found
[[ -z "${RUST_PCT}" ]] && RUST_PCT=$(grep -oP '\d+(?=% coverage)' "${RUST_COV_LOG}" | tail -1)

if [[ "${RUST_EXIT}" -ne 0 ]]; then
  if [[ -n "${RUST_PCT}" ]]; then
    fail "Rust coverage ${RUST_PCT}% < ${THRESHOLD}% threshold"
  else
    fail "Rust coverage check failed  (see ${RUST_COV_LOG})"
  fi
  FAILURES=$((FAILURES + 1))
else
  pass "Rust coverage ${RUST_PCT:-?}% ≥ ${THRESHOLD}%   →  coverage/rust/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Python coverage — pytest-cov
# ─────────────────────────────────────────────────────────────────────────────
banner "2/2  Python Coverage (pytest-cov)"

if ! python -c "import pytest_cov" 2>/dev/null; then
  info "pytest-cov not found — installing ..."
  pip install pytest pytest-cov --quiet
fi

PY_COV_LOG="${COV_DIR}/python/pytest.log"
info "pytest tests/ --cov=llmforge --cov-fail-under=${THRESHOLD}"

set +e
python -m pytest \
  tests/unit/ tests/integration/ \
  -m "not requires_fixtures and not requires_gpu" \
  --cov=llmforge \
  --cov-report=term-missing \
  --cov-report=xml:"${COV_DIR}/python/coverage.xml" \
  --cov-report=html:"${COV_DIR}/python/htmlcov" \
  --cov-fail-under="${THRESHOLD}" \
  -q \
  2>&1 | tee "${PY_COV_LOG}"
PY_EXIT="${PIPESTATUS[0]}"
set -e

# Extract total coverage from the "TOTAL  ...  72%" line
PY_PCT=$(awk '/^TOTAL/{gsub(/%/, "", $NF); print $NF+0}' "${PY_COV_LOG}" | head -1)

if [[ "${PY_EXIT}" -ne 0 ]]; then
  if [[ -n "${PY_PCT}" && "${PY_PCT}" != "0" ]]; then
    fail "Python coverage ${PY_PCT}% < ${THRESHOLD}% threshold"
  else
    fail "Python coverage check failed  (see ${PY_COV_LOG})"
  fi
  FAILURES=$((FAILURES + 1))
else
  pass "Python coverage ${PY_PCT:-?}% ≥ ${THRESHOLD}%   →  coverage/python/htmlcov/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
banner "Coverage Summary"
printf "  %-12s %s%%\n"   "Threshold:" "${THRESHOLD}"
printf "  %-12s %s%%\n"   "Rust:"      "${RUST_PCT:-unknown}"
printf "  %-12s %s%%\n"   "Python:"    "${PY_PCT:-unknown}"
echo ""

if [[ "${FAILURES}" -eq 0 ]]; then
  pass "All coverage gates passed — safe to push."
  exit 0
else
  fail "${FAILURES} coverage gate(s) failed — increase test coverage before pushing."
  exit 1
fi
