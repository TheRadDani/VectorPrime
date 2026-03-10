# LLMForge — Design Plan & Agent Orchestration

## What Is LLMForge?

A **compiler-style hardware-aware LLM inference optimizer**. It behaves like
a small ML compiler for LLM inference: it profiles the host machine, inspects
the model's intermediate representation (IR), searches a hardware-aware
configuration space, benchmarks candidates, and outputs the best runtime
configuration — exposed as a CLI installable via `pip install llmforge`.

llama.cpp is **one runtime backend among several** — not the quantization
engine. The optimizer searches across Ollama, TensorRT, ONNX Runtime, and
llama.cpp; Ollama and TensorRT are the primary targets.

### Six-Module Compiler Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLMForge Pipeline                           │
│                                                                 │
│  [1] hardware_profiler    profile_hardware()                    │
│       → HardwareProfile (CPU, GPU, RAM)                         │
│                                                                 │
│  [2] model_ir_analyzer    parse_model(path)                     │
│       → ModelIR (param_count, architecture, layers, ctx_len)    │
│       Supports GGUF (custom byte reader) + ONNX (protobuf)      │
│       GGUF↔ONNX conversion available for richer inspection      │
│                                                                 │
│  [3] optimization_engine  generate_candidates(hw, model_ir)     │
│       → [RuntimeConfig] pruned by hardware constraints          │
│                                                                 │
│  [4] runtime_adapters     dispatch(config, model)               │
│       PRIMARY: Ollama, TensorRT                                 │
│       SECONDARY: ONNX Runtime                                   │
│       DEPRIORITIZED: llama.cpp                                  │
│       FUTURE: vLLM                                              │
│                                                                 │
│  [5] benchmark_runner     benchmark_all([Config])               │
│       → [BenchmarkResult] (tokens/sec, latency, memory)         │
│                                                                 │
│  [6] artifact_exporter    export_artifact(result)               │
│       → Modelfile / optimized .gguf / metadata.json             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```
llm-forge/
├── Cargo.toml                      # Rust workspace root
├── pyproject.toml                  # maturin build backend / pip packaging
├── CLAUDE.md                       # this file
├── README_agents.md                # copy-paste agent prompts
├── python/
│   └── llmforge/
│       ├── __init__.py
│       ├── cli.py                  # argparse CLI → calls _llmforge bindings
│       ├── onnx_runner.py          # Python helper invoked by ONNX adapter
│       └── _llmforge.pyi           # type stubs for native module
├── crates/
│   ├── llmforge-core/              # shared types, traits, errors (no I/O)
│   ├── llmforge-hardware/          # [hardware_profiler] CPU/GPU/RAM profiler
│   ├── llmforge-model-ir/          # [model_ir_analyzer] GGUF+ONNX IR parsing
│   ├── llmforge-runtime/           # [runtime_adapters] Ollama/TRT/ONNX/llama.cpp
│   ├── llmforge-optimizer/         # [optimization_engine + benchmark_runner]
│   ├── llmforge-export/            # [artifact_exporter] Ollama export + quantize
│   └── llmforge-bindings/          # PyO3 bindings — composes all 6 modules
└── tests/
    ├── integration/
    └── fixtures/                   # tiny GGUF + ONNX test models
```

---

## Crate Dependency Graph

```
llmforge-core          (no internal deps — pure types/traits, no I/O)
  ↑
llmforge-hardware      (→ core)      [hardware_profiler]
llmforge-model-ir      (→ core)      [model_ir_analyzer — NEW]
llmforge-runtime       (→ core)      [runtime_adapters: Ollama PRIMARY, TRT PRIMARY,
  ↑                                   ONNX SECONDARY, llama.cpp DEPRIORITIZED]
llmforge-optimizer     (→ hardware, runtime, core)  [optimization_engine + benchmark_runner]
  ↑
llmforge-export        (→ optimizer, core)           [artifact_exporter]
  ↑
llmforge-bindings      (→ all crates above + model-ir)
  ↑
python/llmforge/cli.py (imports _llmforge native module)
```

---

## Shared Types (llmforge-core)

```rust
pub enum ModelFormat        { ONNX, GGUF }
pub enum SimdLevel          { None, AVX, AVX2, AVX512 }
pub enum QuantizationStrategy { F16, Q8_0, Q4_K_M, Q4_0, Int8, Int4 }
pub enum RuntimeKind        { LlamaCpp, OnnxRuntime, TensorRT }

pub struct HardwareProfile  { cpu: CpuInfo, gpu: Option<GpuInfo>, ram: RamInfo }
pub struct ModelInfo        { path: PathBuf, format: ModelFormat, param_count: Option<u64> }
pub struct RuntimeConfig    { runtime, quantization, threads, batch_size, gpu_layers }
pub struct BenchmarkResult  { tokens_per_sec: f64, latency_ms: f64, peak_memory_mb: u64 }
pub struct OptimizationResult { config: RuntimeConfig, metrics: BenchmarkResult }

pub trait RuntimeAdapter: Send + Sync {
  fn initialize(&mut self, config: &RuntimeConfig) -> Result<()>;
  fn load_model(&mut self, model: &ModelInfo) -> Result<()>;
  fn run_inference(&self, prompt: &str) -> Result<BenchmarkResult>;
  fn teardown(&mut self) -> Result<()>;
}
```

---

## Agent Stages

| Stage | Crate / Area                    | Depends On        | Parallelizable With |
|-------|---------------------------------|-------------------|---------------------|
| 0     | Scaffolding                     | —                 | —                   |
| 1     | `llmforge-core` types + traits  | 0                 | —                   |
| 2     | `llmforge-hardware` profiler    | 1                 | 3A                  |
| 3A    | `llmforge-runtime` dispatch     | 1                 | 2                   |
| 3B    | llama.cpp adapter               | 3A                | 3C, 3D              |
| 3C    | ONNX Runtime adapter            | 3A                | 3B, 3D              |
| 3D    | TensorRT adapter                | 3A                | 3B, 3C              |
| 4     | `llmforge-optimizer` engine     | 2, 3A, 3B, 3C, 3D | —                  |
| 5     | `llmforge-export` Ollama export | 4                 | —                   |
| 6     | `llmforge-bindings` PyO3        | 2, 3, 4, 5        | —                   |
| 7     | CLI + Python packaging          | 6                 | —                   |
| 8     | Integration tests               | 7                 | —                   |

---

## Execution Order (Critical Path)

```
Stage 0 → Stage 1
              ├── Stage 2 ──────────────────────────┐
              └── Stage 3A                           │
                    ├── Stage 3B ──┐                 │
                    ├── Stage 3C ──┤ (parallel)      │
                    └── Stage 3D ──┘                 │
                                   └─ Stage 4 ←──────┘
                                         └── Stage 5
                                               └── Stage 6
                                                     └── Stage 7
                                                           └── Stage 8
```

**Wave 1 (sequential):** 0 → 1  
**Wave 2 (parallel):** 2 + 3A  
**Wave 3 (parallel):** 3B + 3C + 3D  
**Wave 4 (sequential):** 4 → 5 → 6 → 7 → 8

---

## CLI Interface

```
llmforge profile
llmforge optimize <model_path> [--format onnx|gguf] [--max-memory MB]
llmforge export-ollama <model_path> [--result result.json] [--output-dir DIR]
```

### Sample output of `llmforge optimize`

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
```

---

## Runtime Adapter Strategy

Each runtime is isolated behind the `RuntimeAdapter` trait.  
Adapters shell out to external binaries rather than linking directly:

| Runtime      | Binary      | Model Format | Notes                          |
|--------------|-------------|--------------|--------------------------------|
| llama.cpp    | `llama-cli` | GGUF         | CPU + GPU offload              |
| ONNX Runtime | `python3`   | ONNX         | Via bundled `onnx_runner.py`   |
| TensorRT     | `trtexec`   | ONNX         | NVIDIA only, compute cap ≥ 7.0 |

If a binary is not found, `initialize()` returns `RuntimeError::NotInstalled`
and the optimizer skips that adapter gracefully.

---

## Optimization Engine Strategy

1. **Generate candidates** — cross-product of (runtime × quantization × threads × gpu_layers), pruned by hardware constraints
2. **Benchmark in parallel** — tokio semaphore, max 3 concurrent
3. **Select best** — sort by `tokens_per_sec`, filter by memory budget

---

## Ollama Export

```
llmforge optimize model.gguf          # produces model.gguf.llmforge_result.json
llmforge export-ollama model.gguf --result model.gguf.llmforge_result.json

→ optimized_model/
    ├── Modelfile        # FROM + PARAMETER blocks
    ├── model.gguf
    └── metadata.json    # full OptimizationResult

ollama create mymodel -f optimized_model/Modelfile
ollama run mymodel
```

---

## Key Design Decisions

- **Rust backend** — parallel benchmarking, zero-cost FFI, system-level hardware detection
- **PyO3 + maturin** — Python packaging with native Rust extension
- **Shell-out adapters** — integrates with existing runtimes without reimplementing them
- **`GpuProbe` trait** — GPU vendors (AMD, Apple Metal) are pluggable without changing callers
- **`RuntimeAdapter` trait** — new runtimes can be added without touching the optimizer

---

## Verification Checklist (per stage)

Each agent's work is done when:
- [ ] `cargo test -p <crate>` passes (Rust stages)
- [ ] `pytest tests/` passes (Python stages)
- [ ] No `unwrap()` in production paths (use `?` + `anyhow`)
- [ ] All public items have doc comments
- [ ] Parsing logic is a pure function with unit tests
- [ ] Errors on missing binaries are structured, not panics

<!-- SESSION_START -->
## Current Session
<!-- Auto-managed by session_init hook. Overwritten each session. -->
- Resume: `claude --resume 27096e1d-9417-4f82-8322-060005c509d3`
- Team: `pact-27096e1d`
- Started: 2026-03-10 03:04:17 UTC
<!-- SESSION_END -->

## Working Memory
<!-- Auto-managed by pact-memory skill. Last 3 memories shown. Full history searchable via pact-memory skill. -->

### 2026-03-10 12:32
**Context**: Implemented the llmforge-model-ir crate on branch feat/compiler-style-model-ir. This is a new crate in the LLMForge workspace that provides a compiler-style model intermediate representation layer. The crate parses GGUF and ONNX model files to extract architecture metadata without running inference. It was integrated into llmforge-bindings to enrich the ModelInfo passed to the optimizer with param_count, and a new analyze_model() pyfunction was exposed to Python. The task also added module-level doc comments to llmforge-runtime clarifying runtime priority (Ollama + TensorRT primary; llama.cpp deprioritized; vLLM future scope).
**Goal**: Add compiler-style model IR parsing capability to LLMForge so the optimizer can receive richer model metadata (param_count, architecture, context_length, layer_count) without requiring the user to specify it manually.
**Decisions**: Use onnx-protobuf + pinned protobuf=3.4.0 for ONNX parsing, param_count fallback: 12 * block_count * embedding_length, parse_model() errors in optimize() are silently ignored via .ok()
**Lessons**: onnx-protobuf=0.2.3 requires protobuf pinned to =3.4.0 — newer 3.7.x has VERSION_3_4_0 constant mismatch that causes a compile error. Use  in Cargo.toml., GGUF v3 value type codes per llama.cpp gguf.h: UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5, FLOAT32=6, BOOL=7, STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12. Do NOT use 5 for UINT64 — that is INT32., GGUF ARRAY type skipping is HIGH RISK: must read element_type (u32) then element_count (u64) then skip element_count elements correctly by type. Failure to skip all array bytes misaligns the reader for all subsequent KV entries., The param_count fallback formula 12 * block_count * embedding_length gives a rough approximation for transformer models when general.parameter_count is absent. Use saturating_mul to prevent overflow on large models., Real-world GGUF models (tested with ahma-3b-q4_k_m.gguf) may omit general.parameter_count from their KV store — the fallback formula provides a usable estimate even if not exact., In GGUF format, tensor_count comes before kv_count in the file header (magic, version, tensor_count, kv_count). Test helpers must write bytes in exactly this order., When adding a new function to _llmforge native module, it must also be explicitly added to python/llmforge/__init__.py for  to work.
**Memory ID**: 8519e613659e494b3a802272c6b8a969
