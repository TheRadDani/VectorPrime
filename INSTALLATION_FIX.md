# Installation Fix: Pre-built Wheels for End-Users

## Problem
Previously, `pip install vectorprime` would fail on hosts without Rust installed because the build system always tried to compile the Rust bindings from source.

## Solution
Set up **cibuildwheel** in CI/CD to automatically build pre-compiled wheels for multiple platforms and Python versions. End-users now get pre-built binaries and don't need Rust.

## Changes Made

### 1. GitHub Actions Workflow (`.github/workflows/ci.yml`)
- **Added Job 5: `build-wheels`** — Runs on all platforms (Linux, macOS, Windows)
  - Uses `pypa/cibuildwheel@v2.17.0` to build wheels
  - Builds for Python 3.9, 3.10, 3.11, 3.12
  - Builds for Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x86_64)
  - Uploads wheels as GitHub artifacts

- **Updated Job 6: `publish`** — Now publishes pre-built wheels to PyPI
  - Downloads all platform-specific wheels from `build-wheels` job
  - Uses `twine` to upload to PyPI
  - Only runs on version tags (v*)

### 2. Python Configuration (`pyproject.toml`)
- **Added `[tool.cibuildwheel]` section**:
  - Configures cibuildwheel to build wheels with maturin
  - Specifies Python 3.9+ and skips PyPy
  - Includes basic test command to verify wheels work

### 3. Documentation (`README.md`)
- **Split installation into two paths**:
  - **For Users**: `pip install vectorprime` — **No Rust required**
  - **For Developers**: "Build from Source" section clarifies Rust IS needed for contributing
- Added platform support matrix: Linux, macOS, Windows; Python 3.9–3.12; both x86 and Arm

## What Happens Next

### On Version Tag Push
When you create a tag like `v0.1.1` and push it:

```bash
git tag v0.1.1
git push origin v0.1.1
```

GitHub Actions will automatically:
1. ✅ Run all tests (Rust, Python, build-check, coverage)
2. 🏗️ Build wheels on Ubuntu (x86_64, aarch64), macOS (Intel, Arm), Windows
3. 📦 Publish all wheels to PyPI via `twine`

### Requirements: Set PYPI_TOKEN Secret

Before the first publish, you must configure PyPI authentication in GitHub:

1. Go to https://pypi.org/account/
2. Create an API token with permissions for "VectorPrime"
3. In your GitHub repo Settings:
   - Navigate to **Secrets and variables** → **Actions**
   - Click **New repository secret**
   - Name: `PYPI_TOKEN`
   - Value: Your PyPI API token (starts with `pypi-...`)
4. Save

**Note**: The existing `PYPI_TOKEN` secret (if any) should work as-is with the new workflow.

## User Experience Changes

| Before | After |
|--------|-------|
| `pip install vectorprime` | ✅ Works **without Rust** (uses pre-built wheel) |
| `pip install -e .` | ⚠️ Still requires Rust (development mode) |
| Installation on no-Rust systems | ✅ **Now works** |
| Supported platforms | Limited (only where you built) | **All platforms** (automatically built in CI) |

## Testing Locally (Optional)

To test the cibuildwheel configuration locally before pushing a tag:

```bash
pip install cibuildwheel
cibuildwheel --platform linux --output-dir ./dist
```

This builds a wheel locally. Wheels appear in `./dist/`.

## Backwards Compatibility

✅ **No breaking changes**. Existing users:
- Can continue using `pip install vectorprime` (now works better)
- Can continue developing with `maturin develop` (unchanged)
- Wheels are fully compatible with previous versions

## Next Steps

1. **Optional**: Test locally with `cibuildwheel`
2. **When ready**: Push a version tag: `git tag v0.1.1 && git push origin v0.1.1`
3. **Monitor**: Watch the workflow on GitHub Actions
4. **Verify**: Check PyPI for the new wheels: https://pypi.org/project/vectorprime/
