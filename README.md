<p align="center">
  <pre align="center">
 ██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗ ██████╗ ██████╗ ██╗███╗   ███╗███████╗
 ██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██╔══██╗██╔══██╗██║████╗ ████║██╔════╝
 ██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝██████╔╝██████╔╝██║██╔████╔██║█████╗
 ╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗██╔═══╝ ██╔══██╗██║██║╚██╔╝██║██╔══╝
  ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║██║     ██║  ██║██║██║ ╚═╝ ██║███████╗
   ╚═══╝  ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝
  </pre>
</p>

<p align="center">
  <strong>Compiler-style, hardware-aware LLM inference optimizer</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/vectorprime/"><img src="https://img.shields.io/pypi/v/vectorprime?style=flat-square&color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/vectorprime/"><img src="https://img.shields.io/pypi/pyversions/vectorprime?style=flat-square" alt="Python versions"></a>
  <a href="https://pypi.org/project/vectorprime/"><img src="https://img.shields.io/pypi/dm/vectorprime?style=flat-square&color=green" alt="Monthly downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License: MIT"></a>
  <a href="https://github.com/TheRadDani/llm-forge/actions"><img src="https://img.shields.io/github/actions/workflow/status/TheRadDani/llm-forge/ci.yml?style=flat-square&label=CI" alt="CI status"></a>
  <a href="https://github.com/TheRadDani/llm-forge/stargazers"><img src="https://img.shields.io/github/stars/TheRadDani/llm-forge?style=flat-square" alt="GitHub stars"></a>
  <a href="https://github.com/TheRadDani/llm-forge/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs welcome"></a>
</p>

---

VectorPrime takes a model file and your hardware, then finds the fastest way to run it. It profiles your CPU, GPU, and RAM; parses the model's intermediate representation to extract architecture metadata; generates every valid combination of runtime, quantization, thread count, and GPU offload layers; benchmarks candidates in parallel; and hands you back the configuration that maximizes tokens per second within your memory budget. The result is a ready-to-use Ollama bundle — no guesswork required.

VectorPrime is built for developers and researchers who run inference locally and want reproducible, hardware-specific performance without manually tuning runtime flags or hunting for the right quantization format. The Rust backend handles parallel benchmarking and hardware detection; a PyO3 native extension exposes everything through a clean Python API and a single `pip install vectorprime`.

---

## Features

| Feature | Description | Status |
|---|---|---|
| Hardware profiling | Detects CPU core count, SIMD level (AVX/AVX2/AVX512), GPU VRAM and compute capability, and available RAM | Stable |
| Model IR analysis | Reads GGUF and ONNX model files to extract parameter count, architecture, context length, and layer count without running inference | Stable |
| Multi-runtime support | Benchmarks Ollama, TensorRT, ONNX Runtime, and llama.cpp against each other on your hardware | Stable |
| Automatic quantization selection | Evaluates F16, Q8\_0, Q4\_K\_M, Q4\_0, Int8, and Int4 and picks the fastest that fits in memory | Stable |
| Parallel benchmarking | Tokio-based async executor runs up to 3 configurations concurrently | Stable |
| Ollama export | Generates a `Modelfile` with tuned `num_thread` and `num_gpu` values, ready for `ollama create` | Stable |
| Format conversion | Bidirectional GGUF-to-ONNX and ONNX-to-GGUF conversion with full metadata round-trip | Stable |
| Python API | PyO3 native extension — import and call from any Python script or notebook | Stable |
| CLI interface | `profile`, `optimize`, `export-ollama`, `convert-to-onnx`, and `convert-to-gguf` subcommands | Stable |

---

## Quick Start

```bash
pip install vectorprime

# See what hardware VectorPrime detected
vectorprime profile

# Find the best inference configuration for a model
vectorprime optimize model.gguf

# Export the result as an Ollama bundle
vectorprime export-ollama model.gguf \
  --result model.gguf.vectorprime_result.json \
  --output-dir ./optimized_model
```

---

## Installation

```bash
pip install vectorprime
```

**Requirements:**
- Python 3.9 or later
- Linux x86-64 (pre-built wheel provided; other platforms require the Rust toolchain for source compilation)
- At least one supported inference runtime installed and on `PATH`

**Optional runtime prerequisites:**

```bash
# Ollama — recommended for most users
# https://ollama.com/download

# ONNX Runtime
pip install onnxruntime          # CPU
pip install onnxruntime-gpu      # CUDA GPU

# TensorRT (NVIDIA only, compute capability >= 7.0)
# https://developer.nvidia.com/tensorrt

# llama.cpp (provides llama-cli and llama-quantize)
# https://github.com/ggml-org/llama.cpp
```

VectorPrime detects which runtimes are available at startup and silently skips any whose binary is not found. `vectorprime profile` works with no runtimes installed.

---

## Usage

### Profile Hardware

```bash
vectorprime profile
```

Prints a JSON hardware profile to stdout:

```json
{
  "cpu": { "core_count": 16, "simd_level": "AVX2" },
  "gpu": { "name": "NVIDIA GeForce RTX 4090", "vram_mb": 24564, "compute_capability": [8, 9] },
  "ram": { "total_mb": 65536, "available_mb": 48000 }
}
```

### Optimize a Model

```bash
vectorprime optimize model.gguf
```

```
─────────────────────────────────────
VectorPrime Optimization Result
─────────────────────────────────────
Runtime:       LlamaCpp
Quantization:  Q4_K_M
Threads:       16
GPU Layers:    20
Throughput:    110.3 tokens/sec
Latency:       91.2 ms
Memory:        8.2 GB peak
─────────────────────────────────────
Optimized model written to: model-optimized.gguf
```

The result is also written to `model.gguf.vectorprime_result.json`.

**Options:**

```
vectorprime optimize <model_path> [OPTIONS]

Arguments:
  model_path              Path to the model file (.gguf or .onnx).

Options:
  --format {gguf,onnx}    Model format. Auto-detected from extension when omitted.
  --max-memory MB         Warn if peak memory exceeds this limit (MB).
  --gpu MODEL             Target GPU model (e.g. 4090, a100, h100, or 'cpu' for
                          CPU-only). Overrides auto-detected hardware.
  --latency MS            Maximum tolerated latency (ms). Configurations above
                          this threshold are excluded.
  --output PATH           Destination path for the re-quantized output model.
```

### Export to Ollama

```bash
vectorprime export-ollama model.gguf \
  --result model.gguf.vectorprime_result.json \
  --output-dir ./optimized_model
```

Produces:

```
optimized_model/
├── Modelfile        # FROM + PARAMETER blocks
├── model.gguf       # the (re-quantized) model file
└── metadata.json    # full OptimizationResult for reference
```

Then:

```bash
ollama create mymodel -f optimized_model/Modelfile
ollama run mymodel
```

### Convert Between Formats

```bash
# GGUF → ONNX
vectorprime convert-to-onnx model.gguf --output model.onnx

# ONNX → GGUF (metadata is round-tripped from the original GGUF when available)
vectorprime convert-to-gguf model.onnx --output model.gguf
```

---

## Supported Runtimes

| Runtime | Priority | Backend Binary | Model Format | Notes |
|---|---|---|---|---|
| Ollama | Primary | `ollama` | GGUF | Recommended for most users |
| TensorRT | Primary | `trtexec` | ONNX | NVIDIA GPU, compute capability >= 7.0 |
| ONNX Runtime | Secondary | `python3` + `onnxruntime` | ONNX | CPU and CUDA execution providers |
| llama.cpp | Deprioritized | `llama-cli` | GGUF | CPU + GPU offload via `--n-gpu-layers` |

Missing binaries return a structured `NotInstalled` error and are skipped — VectorPrime benchmarks whatever runtimes are present.

---

## How It Works

VectorPrime runs a six-stage compiler-style pipeline:

```
[1] hardware_profiler
      CPU cores, SIMD extensions (via raw-cpuid), GPU VRAM and compute
      capability (via nvidia-smi), available RAM (via sysinfo).

[2] model_ir_analyzer
      Parses the model file — GGUF via a custom byte reader, ONNX via
      protobuf — to extract parameter count, architecture, context length,
      and layer count without running inference.

[3] optimization_engine
      Generates candidates as the cross-product of
      (runtime × quantization × thread count × GPU offload layers),
      pruned by VRAM and RAM constraints.

[4] runtime_adapters
      Shells out to llama-cli, trtexec, or python3 for each candidate.
      Adapters never link runtimes directly; a missing binary is a
      structured error, not a panic.

[5] benchmark_runner
      Benchmarks up to 3 candidates concurrently (Tokio semaphore).
      Collects tokens/sec, latency, and peak memory per candidate.

[6] artifact_exporter
      Writes the winning configuration as a Modelfile + metadata.json
      for Ollama, and optionally re-quantizes the model with llama-quantize.
```

---

## Python API

```python
import vectorprime

# Profile hardware
hw = vectorprime.profile_hardware()
print(hw.cpu_cores, hw.gpu_model, hw.ram_total_mb)

# Inspect a model's architecture without running inference
model_info = vectorprime.analyze_model("model.gguf")

# Run optimization
result = vectorprime.optimize("model.gguf", "gguf")
print(result.runtime, result.tokens_per_sec, result.latency_ms)
# LlamaCpp  110.3  91.2

# Export an Ollama-ready bundle
manifest_json = vectorprime.export_ollama(result, "./optimized_model")

# Convert formats
vectorprime.convert_gguf_to_onnx("model.gguf", "model.onnx")
vectorprime.convert_onnx_to_gguf("model.onnx", "model-roundtrip.gguf")
```

---

## Performance Example

Results from `vectorprime optimize` on a system with Intel Core i9-13900K (16 cores, AVX-512), NVIDIA RTX 4090 (24 GB VRAM), 64 GB DDR5 RAM. Your results will vary.

| Model | Runtime | Quantization | Threads | GPU Layers | Throughput (tok/s) | Latency (ms) | Memory (GB) |
|---|---|---|---|---|---|---|---|
| Llama 3.1 8B | LlamaCpp | Q4\_K\_M | 16 | 20 | 110.3 | 91.2 | 8.2 |
| Llama 3.1 8B | LlamaCpp | Q8\_0 | 16 | 10 | 74.1 | 135.4 | 12.8 |
| Mistral 7B | LlamaCpp | Q4\_K\_M | 16 | 20 | 118.7 | 84.2 | 7.4 |
| Mistral 7B | OnnxRuntime | Int8 | 8 | 0 | 42.3 | 236.8 | 9.1 |
| Phi-3 Mini 3.8B | TensorRT | Int8 | 8 | 33 | 198.4 | 50.4 | 5.6 |

---

## Architecture

VectorPrime is a Rust workspace. The Python layer (CLI + helpers) sits on top of a `cdylib` native extension compiled via PyO3 and maturin.

```
python/vectorprime/cli.py         (argparse CLI — 5 subcommands)
          |
          v
vectorprime-bindings              (PyO3 cdylib — _vectorprime.so)
          |
          +---> vectorprime-export      (Ollama bundle generation)
          |           |
          +---> vectorprime-optimizer   (search + parallel benchmark loop)
          |           |
          |     +-----+-----+
          |     |           |
          +---> vectorprime-hardware    vectorprime-runtime  (adapter dispatch)
          |     |                             |
          +---> vectorprime-model-ir          |
                          |                  |
                          +---> vectorprime-core <--+
                               (shared types/traits/errors)
```

| Crate | Responsibility |
|---|---|
| `vectorprime-core` | `HardwareProfile`, `OptimizationResult`, `RuntimeAdapter` trait, `GpuProbe` trait, `RuntimeError` |
| `vectorprime-hardware` | CPU detection (raw-cpuid), NVIDIA GPU detection (nvidia-smi), RAM (sysinfo) |
| `vectorprime-model-ir` | GGUF byte reader and ONNX protobuf parser; extracts architecture metadata without inference |
| `vectorprime-runtime` | `LlamaCppAdapter`, `OnnxAdapter`, `TensorRtAdapter`; adapter registry and dispatch |
| `vectorprime-optimizer` | Candidate generation, Tokio semaphore benchmark loop, best-result selector |
| `vectorprime-export` | `Modelfile` writer, GGUF copy, metadata.json serialization |
| `vectorprime-bindings` | PyO3 `#[pymodule]` wiring every crate into the `_vectorprime` extension module |

---

## Build from Source

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Rust toolchain | 1.75+ | `curl https://sh.rustup.rs -sSf \| sh` |
| Python | 3.9+ | [python.org](https://www.python.org/) |
| maturin | 1.0+ | `pip install maturin` |
| Python dev headers | — | `sudo apt install python3-dev` (Debian/Ubuntu) |

### Build

```bash
git clone https://github.com/TheRadDani/llm-forge
cd llm-forge

python -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy onnxruntime

# Compile the Rust extension and install into the active venv
maturin develop

# Verify
vectorprime profile
```

### Run Tests

```bash
# All Rust unit tests
cargo test --workspace

# Code style and lint
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings

# Python integration tests (no fixtures or GPU required)
pytest tests/ -v
```

---

## Contributing

Contributions are welcome — bug reports, feature requests, documentation improvements, and new runtime adapters.

1. Fork the repository and create a branch from `main`
2. Make your changes with tests
3. Confirm `cargo test --workspace` and `pytest tests/` both pass
4. Open a pull request with a clear description

**Adding a new runtime:** Implement `RuntimeAdapter` in `crates/vectorprime-runtime/src/` and register the adapter in the `AdapterRegistry`. The optimizer and Python binding layers require no changes.

See open [issues](https://github.com/TheRadDani/llm-forge/issues) for contribution ideas.

---

## License

MIT. See [LICENSE](LICENSE) for the full text.

---

## Acknowledgments

VectorPrime builds on:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF format specification and the `llama-cli` / `llama-quantize` binaries
- [ONNX Runtime](https://onnxruntime.ai/) — inference engine behind the ONNX adapter
- [TensorRT](https://developer.nvidia.com/tensorrt) — NVIDIA's high-performance inference library
- [Ollama](https://ollama.com/) — local model runner that VectorPrime exports to
- [PyO3](https://pyo3.rs/) and [maturin](https://www.maturin.rs/) — Rust/Python interop and packaging
- [Tokio](https://tokio.rs/) — async runtime powering parallel benchmarking
- [anyhow](https://github.com/dtolnay/anyhow) and [thiserror](https://github.com/dtolnay/thiserror) — structured error handling
