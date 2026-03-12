# Publishing VectorPrime to PyPI

This guide walks you through building the manylinux-compatible wheels and
publishing them to PyPI (or TestPyPI) using the project's Dockerfile.

---

## Prerequisites

| Requirement | Minimum version | Check |
|---|---|---|
| Docker (with BuildKit) | 23.0+ | `docker --version` |
| PyPI account | — | [pypi.org/account/register](https://pypi.org/account/register/) |
| PyPI API token | — | See *Create a token* below |

BuildKit has been the default Docker build engine since Docker 23.0.  
If you are on an older version, prefix every `docker build` call with
`DOCKER_BUILDKIT=1`.

---

## Step 1 — Create a PyPI API token

1. Log in at <https://pypi.org>
2. Go to **Account settings → API tokens → Add API token**
3. Scope: **Entire account** (first publish) or **Project: vectorprime**
   (subsequent publishes)
4. Copy the token — it starts with `pypi-` and is shown only once

> **TestPyPI**: If you want a staging run first, repeat the same steps at
> <https://test.pypi.org> to get a separate token.

---

## Step 2 — Verify `Cargo.lock` is committed

The Dockerfile passes `--locked` to both `cargo fetch` and `maturin build`,
which requires a committed `Cargo.lock`.

```bash
# In the repo root:
git status Cargo.lock
# Should show "nothing to commit" for Cargo.lock
```

If `Cargo.lock` is missing or stale, regenerate it first:

```bash
cargo fetch   # regenerates Cargo.lock
git add Cargo.lock && git commit -m "chore: update Cargo.lock"
```

---

## Step 3 — (Optional) Smoke-test with TestPyPI

Do a dry-run against TestPyPI before touching the live index.

```bash
export PYPI_TOKEN=pypi-xxxx-testpypi-token

docker build \
  --secret id=pypi_token,env=PYPI_TOKEN \
  --build-arg TWINE_REPOSITORY_URL=https://test.pypi.org/legacy/ \
  -t vectorprime-publisher \
  .
```

If the build succeeds, all wheels have been validated (`twine check`) and
uploaded to TestPyPI. Verify them at
<https://test.pypi.org/project/vectorprime/>.

---

## Step 4 — Publish to PyPI (production)

```bash
export PYPI_TOKEN=pypi-xxxx-your-real-token

docker build \
  --secret id=pypi_token,env=PYPI_TOKEN \
  -t vectorprime-publisher \
  .
```

A successful build means the wheels were validated and uploaded.  
Verify at <https://pypi.org/project/vectorprime/>.

---

## Step 5 — Install and verify

```bash
# Test installation from PyPI in a clean virtual environment:
python -m venv /tmp/vp-check
source /tmp/vp-check/bin/activate
pip install vectorprime
vectorprime --help
```

---

## Extracting wheels without uploading

If you want the wheel files locally (e.g. to inspect or upload manually):

```bash
# Build only the builder stage:
docker build --target builder -t vectorprime-builder .

# Copy wheels out of the container without running it:
CONTAINER=$(docker create vectorprime-builder)
docker cp "$CONTAINER":/dist ./dist
docker rm "$CONTAINER"

ls dist/
# vectorprime-0.1.0-cp39-cp39-manylinux_2_28_x86_64.whl
# vectorprime-0.1.0-cp310-cp310-manylinux_2_28_x86_64.whl
# vectorprime-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl
# vectorprime-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl
# vectorprime-0.1.0-cp313-cp313-manylinux_2_28_x86_64.whl
```

---

## Uploading wheels manually (without Docker)

If you extracted wheels locally and want to upload them directly:

```bash
pip install "twine>=6.0,<7.0"

# Validate first:
twine check dist/*.whl

# Upload to PyPI:
twine upload --username __token__ --password pypi-xxxx dist/*.whl

# Upload to TestPyPI:
twine upload \
  --repository-url https://test.pypi.org/legacy/ \
  --username __token__ \
  --password pypi-xxxx \
  dist/*.whl
```

---

## Bumping the version

Before a new release, update the version in **both** places:

| File | Field |
|---|---|
| `pyproject.toml` | `version = "x.y.z"` |
| `crates/vectorprime-bindings/Cargo.toml` | `version = "x.y.z"` |

Then commit, tag, and rebuild:

```bash
git add pyproject.toml crates/vectorprime-bindings/Cargo.toml
git commit -m "chore: bump version to x.y.z"
git tag vx.y.z
git push --follow-tags
```

---

## Multi-architecture builds (ARM64)

The current Dockerfile targets `linux/amd64`. To additionally publish an
`aarch64` (ARM64) wheel:

```bash
# One-time: create a buildx builder that supports multi-arch emulation
docker buildx create --use --name multiarch

PYPI_TOKEN=pypi-xxxx docker buildx build \
  --platform linux/arm64 \
  --secret id=pypi_token,env=PYPI_TOKEN \
  --load \
  -t vectorprime-publisher-arm64 \
  .
```

> For production multi-arch publishing, consider using a native ARM64 runner
> (GitHub Actions `ubuntu-24.04-arm`, AWS Graviton, etc.) rather than QEMU
> emulation, which is significantly slower for Rust compilation.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `error: package 'X' cannot be found` | `Cargo.lock` out of sync | `cargo fetch && git add Cargo.lock` |
| `HTTPError: 400 File already exists` | Version already on PyPI | Bump `version` in `pyproject.toml` |
| `Invalid distribution file` | Wheel corrupted or wrong platform tag | Rebuild from scratch; check ARCH |
| `secret not found: pypi_token` | Docker BuildKit not enabled | Use Docker >= 23.0 or set `DOCKER_BUILDKIT=1` |
| `protoc: command not found` at build | `protobuf-compiler` yum package missing | The Dockerfile installs it; check `yum` logs in build output |
| `twine check` fails | Missing metadata in `pyproject.toml` | Ensure `name`, `version`, `license`, `requires-python` are set |
