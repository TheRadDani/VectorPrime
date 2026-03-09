#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/build_release.sh
# Release builder for LLMForge — builds the wheel, tags the commit, and
# creates a GitHub Release with the wheel attached.
#
# Usage:
#   ./scripts/build_release.sh               # standard release from main
#   ./scripts/build_release.sh --dry-run     # print steps, don't execute
#   ./scripts/build_release.sh --force       # skip branch guard + allow retag
#   ./scripts/build_release.sh --force --dry-run
#
# Prerequisites:
#   maturin   — pip install maturin
#   gh        — GitHub CLI (https://cli.github.com) — CI check and release
#               If gh is absent the CI check is skipped (build still runs).
#
# What it does:
#   1. Read version from pyproject.toml
#   2. Guard: warn/abort if not on the main branch (bypass with --force)
#   3. Check: all CI runs for HEAD concluded "success" via gh (skipped if gh absent)
#   4. Build: maturin build --release  →  target/wheels/*.whl
#   5. Tag:   git tag v{version}       (or -f with --force)
#   6. Push:  git push origin v{version}
#   7. Release: gh release create v{version} with wheel(s) attached
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

# ── Flag parsing ──────────────────────────────────────────────────────────────
DRY_RUN=0
FORCE=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --force)   FORCE=1   ;;
    *)
      fail "Unknown flag: ${arg}"
      echo "Usage: $0 [--dry-run] [--force]"
      exit 1
      ;;
  esac
done

# exec_or_dry: run a command for real, or echo it when --dry-run is set.
# Usage: exec_or_dry <cmd> [args...]
exec_or_dry() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    info "[dry-run] $*"
  else
    "$@"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Read version from pyproject.toml
# ─────────────────────────────────────────────────────────────────────────────
banner "1/7  Read version"

VERSION="$(grep '^version' "${ROOT_DIR}/pyproject.toml" | head -1 | sed 's/.*= *"\(.*\)"/\1/')"

if [[ -z "${VERSION}" ]]; then
  fail "Could not read version from pyproject.toml"
  exit 1
fi

TAG="v${VERSION}"
pass "Version: ${VERSION}  →  tag: ${TAG}"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Branch guard
# ─────────────────────────────────────────────────────────────────────────────
banner "2/7  Branch check"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [[ "${CURRENT_BRANCH}" != "main" ]]; then
  if [[ "${FORCE}" -eq 1 ]]; then
    warn "Not on main (current: ${CURRENT_BRANCH}) — proceeding because --force was set."
  else
    fail "You are on branch '${CURRENT_BRANCH}', not 'main'."
    info "Switch to main or re-run with --force to bypass this guard."
    exit 1
  fi
else
  pass "On branch: main"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. CI check via gh
# ─────────────────────────────────────────────────────────────────────────────
banner "3/7  CI status check"

if ! command -v gh &>/dev/null; then
  warn "gh (GitHub CLI) not found — skipping CI status check."
  warn "Install gh from https://cli.github.com to enable this guard."
else
  HEAD_SHA="$(git rev-parse HEAD)"
  info "Checking CI runs for commit ${HEAD_SHA} ..."

  set +e
  # Fetch all runs for this commit; collect their conclusions.
  CI_OUTPUT="$(gh run list \
    --commit "${HEAD_SHA}" \
    --json "status,conclusion" \
    --jq '[.[] | {status, conclusion}]' 2>&1)"
  GH_EXIT="${PIPESTATUS[0]}"
  set -e

  if [[ "${GH_EXIT}" -ne 0 ]]; then
    warn "gh run list failed (exit ${GH_EXIT}) — skipping CI check."
    warn "Output: ${CI_OUTPUT}"
  else
    # Count runs that are not yet completed or did not succeed.
    PENDING="$(echo "${CI_OUTPUT}" | grep -c '"status": "in_progress"' || true)"
    FAILED="$(echo "${CI_OUTPUT}" | grep -v '"conclusion": "success"' \
                | grep -c '"conclusion":' || true)"
    TOTAL="$(echo "${CI_OUTPUT}" | grep -c '"status":' || true)"

    if [[ "${TOTAL}" -eq 0 ]]; then
      warn "No CI runs found for HEAD — cannot confirm CI is green."
      warn "Proceeding; verify CI manually if needed."
    elif [[ "${PENDING}" -gt 0 ]]; then
      fail "CI runs for HEAD are still in progress (${PENDING} pending of ${TOTAL} total)."
      info "Wait for CI to finish and re-run this script."
      FAILURES=$((FAILURES + 1))
    elif [[ "${FAILED}" -gt 0 ]]; then
      fail "One or more CI runs for HEAD did not conclude 'success' (${FAILED} of ${TOTAL})."
      info "Fix failing CI before releasing."
      FAILURES=$((FAILURES + 1))
    else
      pass "All ${TOTAL} CI run(s) for HEAD concluded 'success'."
    fi
  fi
fi

if [[ "${FAILURES}" -gt 0 ]]; then
  fail "Pre-flight checks failed — aborting release."
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build release wheel
# ─────────────────────────────────────────────────────────────────────────────
banner "4/7  Build release wheel"

if ! command -v maturin &>/dev/null; then
  fail "maturin not found. Install with: pip install maturin"
  exit 1
fi

info "maturin build --release"
exec_or_dry maturin build --release

if [[ "${DRY_RUN}" -eq 0 ]]; then
  WHEELS=("${ROOT_DIR}"/target/wheels/*.whl)
  if [[ ${#WHEELS[@]} -eq 0 ]] || [[ ! -f "${WHEELS[0]}" ]]; then
    fail "No wheel found in target/wheels/ after build."
    exit 1
  fi
  pass "Wheel(s) built: ${WHEELS[*]}"
else
  info "[dry-run] Would attach wheels from: ${ROOT_DIR}/target/wheels/*.whl"
  WHEELS=("${ROOT_DIR}/target/wheels/*.whl")  # placeholder for dry-run messaging
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. Create (or force-update) the git tag
# ─────────────────────────────────────────────────────────────────────────────
banner "5/7  Git tag"

# Check whether the tag already exists.
if git rev-parse "${TAG}" &>/dev/null; then
  if [[ "${FORCE}" -eq 1 ]]; then
    warn "Tag ${TAG} already exists — overwriting because --force was set."
    exec_or_dry git tag -f "${TAG}"
  else
    fail "Tag ${TAG} already exists."
    info "Delete it first or re-run with --force to overwrite."
    exit 1
  fi
else
  exec_or_dry git tag "${TAG}"
  pass "Created tag: ${TAG}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Push the tag to origin
# ─────────────────────────────────────────────────────────────────────────────
banner "6/7  Push tag"

if [[ "${FORCE}" -eq 1 ]]; then
  info "git push --force origin ${TAG}"
  exec_or_dry git push --force origin "${TAG}"
else
  info "git push origin ${TAG}"
  exec_or_dry git push origin "${TAG}"
fi

pass "Tag ${TAG} pushed to origin."

# ─────────────────────────────────────────────────────────────────────────────
# 7. Create GitHub Release
# ─────────────────────────────────────────────────────────────────────────────
banner "7/7  GitHub Release"

if ! command -v gh &>/dev/null; then
  fail "gh (GitHub CLI) not found — cannot create GitHub Release automatically."
  info "Install gh from https://cli.github.com and re-run, or create the release"
  info "manually at: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo '<owner/repo>')/releases/new"
  FAILURES=$((FAILURES + 1))
else
  info "gh release create ${TAG} --notes-from-tag ${WHEELS[*]}"
  exec_or_dry gh release create "${TAG}" \
    --notes-from-tag \
    "${WHEELS[@]}"

  if [[ "${DRY_RUN}" -eq 0 ]]; then
    pass "GitHub Release created: ${TAG}"
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
banner "Release Summary"
printf "  %-14s %s\n" "Version:"  "${VERSION}"
printf "  %-14s %s\n" "Tag:"      "${TAG}"
printf "  %-14s %s\n" "Branch:"   "${CURRENT_BRANCH}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf "  %-14s %s\n" "Mode:" "DRY-RUN (no git/gh commands executed)"
fi
echo ""

if [[ "${FAILURES}" -eq 0 ]]; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    pass "Dry-run complete — no changes made."
  else
    pass "Release ${TAG} published successfully."
  fi
  exit 0
else
  fail "${FAILURES} step(s) failed — release may be incomplete."
  exit 1
fi
