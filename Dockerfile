# =============================================================================
# VectorPrime — manylinux wheel builder + PyPI publisher
# =============================================================================
#
# USAGE
# -----
# Requires Docker >= 23.0 (BuildKit enabled by default).
#
# Build the wheel only (no upload):
#   docker build --target builder -t vectorprime-builder .
#   docker run --rm -v "$(pwd)/dist:/dist" vectorprime-builder
#
# Build and publish to PyPI (production):
#   PYPI_TOKEN=pypi-xxxx docker build \
#     --secret id=pypi_token,env=PYPI_TOKEN \
#     -t vectorprime-publisher .
#   docker run --rm vectorprime-publisher
#
# Build and publish to TestPyPI (staging/smoke test):
#   PYPI_TOKEN=pypi-xxxx docker build \
#     --secret id=pypi_token,env=PYPI_TOKEN \
#     --build-arg TWINE_REPOSITORY_URL=https://test.pypi.org/legacy/ \
#     -t vectorprime-publisher .
#   docker run --rm vectorprime-publisher
#
# Extract wheel artifacts without running the container:
#   CONTAINER=$(docker create vectorprime-builder)
#   docker cp "$CONTAINER":/dist ./dist
#   docker rm "$CONTAINER"
#
# TOKEN SECURITY NOTE
# -------------------
# The token is mounted as a BuildKit secret (--mount=type=secret) so it is
# never stored in any image layer, even transiently. It is NOT visible via
# `docker history` or `docker inspect`.
#
# ARCHITECTURE NOTE
# -----------------
# This Dockerfile targets linux/amd64.
# To build for linux/arm64, use a separate multi-arch build:
#   docker buildx build --platform linux/arm64 ...
# =============================================================================

# =============================================================================
# Stage 1 — builder
# Build the manylinux-compatible wheel.
#
# ghcr.io/pyo3/maturin is the official maturin image. It is based on
# manylinux_2_28 and ships with:
#   - A recent stable Rust toolchain
#   - maturin >=1.8 pre-installed
#   - All CPython versions (3.9-3.13) under /opt/python
#
# Using this image avoids having to install Rust or pin its version here;
# the image maintainers keep it up to date with the latest stable toolchain.
# =============================================================================
FROM ghcr.io/pyo3/maturin:latest AS builder

# Switch to root to install any OS-level deps (the base image may run as a
# non-root user — we only do system work here, not build work).
USER root

# Install protobuf compiler. vectorprime-model-ir uses the onnx-protobuf crate
# which requires protoc to be present at build time.
RUN yum install -y protobuf-compiler \
    && yum clean all \
    && rm -rf /var/cache/yum

# Return to the maturin-provided build user (uid 1000 in this image)
USER maturin

WORKDIR /workspace

# ---------------------------------------------------------------------------
# Copy dependency manifests first — these layers are cheap to rebuild when
# only source code changes. Cargo will restore from cache if neither
# Cargo.toml nor Cargo.lock changed.
# ---------------------------------------------------------------------------
COPY --chown=maturin:maturin Cargo.toml Cargo.lock ./
COPY --chown=maturin:maturin pyproject.toml LICENSE ./

# Copy each crate's Cargo.toml so Cargo can resolve the workspace graph
# without all source files present. This primes the registry download layer.
COPY --chown=maturin:maturin crates/vectorprime-core/Cargo.toml      crates/vectorprime-core/Cargo.toml
COPY --chown=maturin:maturin crates/vectorprime-hardware/Cargo.toml  crates/vectorprime-hardware/Cargo.toml
COPY --chown=maturin:maturin crates/vectorprime-runtime/Cargo.toml   crates/vectorprime-runtime/Cargo.toml
COPY --chown=maturin:maturin crates/vectorprime-optimizer/Cargo.toml crates/vectorprime-optimizer/Cargo.toml
COPY --chown=maturin:maturin crates/vectorprime-export/Cargo.toml    crates/vectorprime-export/Cargo.toml
COPY --chown=maturin:maturin crates/vectorprime-model-ir/Cargo.toml  crates/vectorprime-model-ir/Cargo.toml
COPY --chown=maturin:maturin crates/vectorprime-bindings/Cargo.toml  crates/vectorprime-bindings/Cargo.toml

# Pre-fetch Cargo registry index + download all dependencies.
# We create stub lib.rs files so `cargo fetch` can resolve the full dep graph
# without any real source. This layer is invalidated only when Cargo.* changes.
RUN bash -euxo pipefail -c ' \
    for dir in \
        crates/vectorprime-core \
        crates/vectorprime-hardware \
        crates/vectorprime-runtime \
        crates/vectorprime-optimizer \
        crates/vectorprime-export \
        crates/vectorprime-model-ir \
        crates/vectorprime-bindings; \
    do \
        mkdir -p "$dir/src" && echo "// placeholder" > "$dir/src/lib.rs"; \
    done \
    && cargo fetch --locked'

# ---------------------------------------------------------------------------
# Now copy the full source tree (invalidates from here on source changes).
# ---------------------------------------------------------------------------
COPY --chown=maturin:maturin crates/ ./crates/
COPY --chown=maturin:maturin python/ ./python/

# ---------------------------------------------------------------------------
# Build manylinux-compatible wheels for Python 3.9, 3.10, 3.11, and 3.12.
#
# Flags explained:
#   --release          Rust optimisation level 3 (required for production)
#   --strip            Strip debug symbols; reduces wheel size significantly
#   --locked           Honour Cargo.lock exactly; fails if lock is stale
#   --out /dist        Write the finished .whl files to /dist
#   --interpreter ...  Build for exactly these CPython versions
#                      (the maturin image ships all of them under /opt/python)
# ---------------------------------------------------------------------------
RUN maturin build \
      --release \
      --strip \
      --locked \
      --out /dist \
      --interpreter python3.9 python3.10 python3.11 python3.12 python3.13

# =============================================================================
# Stage 2 — publisher
# Upload the wheels to PyPI using twine.
#
# This stage is intentionally separate so the build artifacts are never
# mixed with upload credentials in a single layer.
#
# The PyPI token is mounted as a BuildKit secret (--mount=type=secret) in the
# RUN step below. It is never written to any image layer and cannot be
# recovered from `docker history` or `docker inspect`.
# =============================================================================
FROM python:3.12-slim AS publisher

# Install twine — the standard PyPI upload tool.
# Pinned to a recent stable range to ensure reproducible uploads.
RUN pip install --no-cache-dir "twine>=5.0,<6.0"

# Copy the built wheels from the builder stage.
COPY --from=builder /dist /dist

# Optional: override to target TestPyPI or a private index.
# Default is the official PyPI upload endpoint.
ARG TWINE_REPOSITORY_URL=https://upload.pypi.org/legacy/

# Validate wheels before upload — catches malformed metadata early.
# Upload using a BuildKit secret so the token never appears in any image
# layer. Pass the secret at build time:
#   PYPI_TOKEN=pypi-xxxx docker build --secret id=pypi_token,env=PYPI_TOKEN ...
#
# TWINE_USERNAME must be the literal "__token__" when using API tokens.
RUN --mount=type=secret,id=pypi_token \
    set -euxo pipefail \
    && twine check /dist/*.whl \
    && TWINE_USERNAME=__token__ \
       TWINE_PASSWORD="$(cat /run/secrets/pypi_token)" \
       TWINE_REPOSITORY_URL="${TWINE_REPOSITORY_URL}" \
       twine upload --non-interactive /dist/*.whl
