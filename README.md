<p align="center">
  <pre align="center">
  ██╗     ██╗     ███╗   ███╗███████╗ ██████╗ ██████╗  ██████╗ ███████╗
  ██║     ██║     ████╗ ████║██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
  ██║     ██║     ██╔████╔██║█████╗  ██║   ██║██████╔╝██║  ███╗█████╗
  ██║     ██║     ██║╚██╔╝██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
  ███████╗███████╗██║ ╚═╝ ██║██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
  ╚══════╝╚══════╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
  </pre>
</p>

<p align="center">
  <strong>Hardware-aware LLM inference optimizer</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/llmforge/"><img src="https://img.shields.io/pypi/v/llmforge?style=flat-square&color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/llmforge/"><img src="https://img.shields.io/pypi/pyversions/llmforge?style=flat-square" alt="Python versions"></a>
  <a href="https://pypi.org/project/llmforge/"><img src="https://img.shields.io/pypi/dm/llmforge?style=flat-square&color=green" alt="Monthly downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License: MIT"></a>
  <a href="https://github.com/TheRadDani/llm-forge/actions"><img src="https://img.shields.io/github/actions/workflow/status/TheRadDani/llm-forge/ci.yml?style=flat-square&label=CI" alt="CI status"></a>
  <a href="https://github.com/TheRadDani/llm-forge/stargazers"><img src="https://img.shields.io/github/stars/TheRadDani/llm-forge?style=flat-square" alt="GitHub stars"></a>
  <a href="https://github.com/TheRadDani/llm-forge/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs welcome"></a>
</p>

---

LLMForge takes a model file and your hardware, then finds the fastest way to run it. It profiles your CPU, GPU, and RAM; generates every valid combination of runtime, quantization, thread count, and GPU offload layers; benchmarks them in parallel; and hands you back the configuration that maximizes tokens per second within your memory budget. The result is a ready-to-use Ollama bundle — no guesswork required.

LLMForge is built for developers and researchers who run inference locally and want reproducible, hardware-specific performance without manually tuning llama.cpp flags or hunting for the right quantization format. The Rust backend handles parallel benchmarking and hardware detection; a PyO3 native extension exposes everything through a clean Python API and a single `pip install`.

---

## Features

| Feature | Description | Status |
|---|---|---|
| 🧠 Hardware profiling | Detects CPU brand, core count, SIMD level (AVX/AVX2/AVX512), GPU VRAM, compute capability, and available RAM | Stable |
| ⚡ Multi-runtime support | Benchmarks llama.cpp, ONNX Runtime, and TensorRT against each other on your hardware | Stable |
| 🎯 Automatic quantization selection | Evaluates F16, Q8_0, Q4_K_M, Q4_0, Int8, and Int4 and picks the fastest that fits in memory | Stable |
| 📊 Parallel benchmarking | Tokio-based async executor runs up to 3 configurations concurrently | Stable |
| 🚀 Ollama export | Generates a `Modelfile` with tuned `num_thread` and `num_gpu` values, ready for `ollama create` | Stable |
| 🐍 Python API | PyO3 native extension — import and call from any Python script or notebook | Stable |
| 🔧 CLI interface | `profile`, `optimize`, and `export-ollama` subcommands | Stable |

---

## 📊 Performance Results

The table below shows illustrative results from running `llmforge optimize` on typical consumer hardware. Your results will vary based on your specific CPU/GPU and available memory.

| Model | Runtime | Quantization | Threads | GPU Layers | Throughput (tok/s) | Latency (ms) | Memory (GB) |
|---|---|---|---|---|---|---|---|
| Llama 3.1 8B | llama.cpp | Q4_K_M | 16 | 20 | 110.3 | 91.2 | 8.2 |
| Llama 3.1 8B | llama.cpp | Q8_0 | 16 | 10 | 74.1 | 135.4 | 12.8 |
| Mistral 7B | llama.cpp | Q4_K_M | 16 | 20 | 118.7 | 84.2 | 7.4 |
| Mistral 7B | ONNX Runtime | Int8 | 8 | 0 | 42.3 | 236.8 | 9.1 |
| Phi-3 Mini 3.8B | TensorRT | Int8 | 8 | 33 | 198.4 | 50.4 | 5.6 |

*Results recorded on a system with Intel Core i9-13900K (16 cores, AVX-512), NVIDIA RTX 4090 (24 GB VRAM), 64 GB DDR5 RAM. Illustrative only.*

---

## Supported Runtimes

| Runtime | Backend Binary | Model Format | Hardware |
|---|---|---|---|
| llama.cpp | `llama-cli` | GGUF | CPU + NVIDIA/AMD GPU offload |
| ONNX Runtime | `python3` + `onnxruntime` | ONNX | CPU + GPU |
| TensorRT | `trtexec` | ONNX | NVIDIA GPU (compute capability >= 7.0) |

Missing binaries are detected at startup and the corresponding runtime is skipped gracefully — LLMForge will benchmark whatever runtimes are installed.

---

## Installation

```bash
pip install llmforge
```

**Requirements:**
- Python 3.9 or later
- At least one supported inference runtime (see below)

**Optional runtime prerequisites:**

```bash
# llama.cpp (recommended for most users)
# Build from source: https://github.com/ggml-org/llama.cpp
# or install via a package manager on your platform

# ONNX Runtime
pip install onnxruntime          # CPU
pip install onnxruntime-gpu      # GPU

# TensorRT (NVIDIA only, compute capability >= 7.0)
# Install via: https://developer.nvidia.com/tensorrt
```

---

## Quick Start

```bash
# 1. Profile your hardware
llmforge profile

# 2. Find the best configuration for a model
llmforge optimize model.gguf

# 3. Export the optimized model as an Ollama bundle
llmforge export-ollama model.gguf --result model.gguf.llmforge_result.json
```

**Example output of `llmforge optimize`:**

```
─────────────────────────────────────
LLMForge Optimization Result
─────────────────────────────────────
Runtime:       llama.cpp
Quantization:  Q4_K_M
Threads:       16
GPU Layers:    20

Performance:
  Throughput:  110.3 tokens/sec
  Latency:     91.2 ms
  Memory:      8.2 GB peak
─────────────────────────────────────
Result written to: model.gguf.llmforge_result.json
```

---

## Python API

```python
import llmforge

# Detect host hardware
hw = llmforge.profile_hardware()
print(hw)
# HardwareProfile { cpu: Intel Core i9-13900K (16 cores, AVX512),
#                   gpu: Some(NVIDIA RTX 4090, 24576 MB),
#                   ram: 65536 MB total }

# Find the optimal runtime configuration for a model
result = llmforge.optimize("model.gguf", "gguf")
print(result)
# OptimizationResult { runtime: LlamaCpp, quantization: Q4_K_M,
#                      threads: 16, gpu_layers: 20,
#                      tokens_per_sec: 110.3, latency_ms: 91.2 }

# Export an Ollama-ready bundle
manifest_json = llmforge.export_ollama(result, "./optimized_model")

# Then from the terminal:
# ollama create mymodel -f optimized_model/Modelfile
# ollama run mymodel
```

---

## CLI Reference

```
llmforge profile
```

Detects CPU, GPU, and RAM and prints a JSON hardware profile.

```
llmforge optimize <model_path> [--format gguf|onnx] [--max-memory MB]
```

Benchmarks all applicable runtime/quantization/thread configurations and prints the winner. Auto-detects format from `.gguf` or `.onnx` extension when `--format` is omitted. Writes the result to `<model_path>.llmforge_result.json`.

```
llmforge export-ollama <model_path> [--result FILE] [--output-dir DIR]
```

Exports an Ollama-compatible bundle. If `--result` is supplied, the previously computed optimization result is reused and no re-benchmarking occurs. The bundle is written to `--output-dir` (default: `optimized_model/`).

---

## How It Works

1. **Profile hardware** — LLMForge reads CPU brand, core count, and SIMD extensions via `raw-cpuid`; queries NVIDIA VRAM and compute capability via `nvidia-smi`; and reads total and available RAM via `sysinfo`.

2. **Generate candidates** — The search space is the cross-product of (runtime x quantization x thread counts x GPU offload layers), pruned by hardware constraints. Thread count options are `[cores/2, cores, cores*2]` capped to 64. TensorRT is excluded on non-NVIDIA hardware or compute capability < 7.0.

3. **Benchmark in parallel** — Each candidate is benchmarked by shelling out to the appropriate binary (`llama-cli`, `python3 onnx_runner.py`, or `trtexec`). A Tokio semaphore limits concurrency to three simultaneous runs to avoid resource contention.

4. **Select the best** — Results are filtered to exclude configurations that exceed 90% of available RAM, then sorted by tokens per second. The top result is returned.

5. **Export to Ollama** — LLMForge writes a `Modelfile` with the tuned `num_thread` and `num_gpu` parameters, copies the model file, and writes a `metadata.json` with the full optimization result.

---

## Architecture

```
python/llmforge/cli.py       (argparse CLI)
        |
        v
llmforge-bindings            (PyO3 native extension — _llmforge.so)
        |
        +---> llmforge-export      (Ollama bundle generation)
        |           |
        +---> llmforge-optimizer   (search space + parallel benchmark loop)
        |           |
        |     +-----+-----+
        |     |           |
        +---> llmforge-hardware    llmforge-runtime   (adapter dispatch)
              |                         |
              +-------> llmforge-core <-+
                        (shared types, traits, errors)
```

**Key crates:**

| Crate | Responsibility |
|---|---|
| `llmforge-core` | Shared types (`HardwareProfile`, `OptimizationResult`, etc.), the `RuntimeAdapter` and `GpuProbe` traits, and structured errors |
| `llmforge-hardware` | CPU/GPU/RAM detection; `GpuProbe` implementations for NVIDIA |
| `llmforge-runtime` | `RuntimeAdapter` implementations for llama.cpp, ONNX Runtime, and TensorRT |
| `llmforge-optimizer` | Candidate generation, Tokio-based parallel benchmark loop, best-result selector |
| `llmforge-export` | Ollama `Modelfile` generation, GGUF copy, metadata serialization |
| `llmforge-bindings` | PyO3 `#[pymodule]` wiring all crates into the `_llmforge` native extension |

**Design principles:**

- **Shell-out adapters** — runtimes are invoked as subprocesses rather than linked libraries, so LLMForge works with any installed version without recompilation.
- **`RuntimeAdapter` trait** — adding a new runtime means implementing four methods; the optimizer does not change.
- **`GpuProbe` trait** — GPU vendor detection is pluggable; AMD and Apple Metal probes can be added without touching the hardware profiler.

---

## Build From Source

**Prerequisites:**

- Rust 1.75 or later (`rustup install stable`)
- Python 3.9 or later
- `maturin` (`pip install maturin` or `cargo install maturin`)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/llm-forge
cd llm-forge

# Build the native extension and install into the current Python environment
maturin develop

# Alternatively, install in editable mode with pip
pip install -e .

# Run Rust tests
cargo test --workspace

# Run Python tests
pytest tests/
```

**Building a release wheel:**

```bash
maturin build --release
pip install target/wheels/llmforge-*.whl
```

---

## Contributing

Contributions are welcome. Bug reports, feature requests, documentation improvements, and new runtime adapters are all useful.

**Workflow:**

1. Fork the repository and create a branch from `main`
2. Make your changes with tests
3. Verify that `cargo test --workspace` and `pytest tests/` both pass
4. Open a pull request with a clear description of the change

**Adding a new runtime adapter:**

Implement the `RuntimeAdapter` trait in `crates/llmforge-runtime/src/` and register the adapter in the dispatcher. The optimizer, export, and Python binding layers require no changes.

See [open issues](https://github.com/YOUR_USERNAME/llm-forge/issues) for ideas, or open a new issue to discuss a change before starting work.

---

## License

MIT. See [LICENSE](LICENSE) for the full text.

---

## Acknowledgments

LLMForge builds on the shoulders of the following projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the `llama-cli` binary used by the llama.cpp adapter
- [ONNX Runtime](https://onnxruntime.ai/) — the inference engine behind the ONNX adapter
- [TensorRT](https://developer.nvidia.com/tensorrt) — NVIDIA's high-performance inference library
- [Ollama](https://ollama.com/) — the local model runner that LLMForge exports to
- [PyO3](https://pyo3.rs/) and [maturin](https://www.maturin.rs/) — Rust/Python interop and packaging
- [Tokio](https://tokio.rs/) — the async runtime powering parallel benchmarking
