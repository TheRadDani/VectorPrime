# VectorPrime — Agent Prompts

Each section below is a **complete, self-contained prompt** to give a Claude
agent. Copy the **Shared Context Block** first, then append the stage-specific
section. Every prompt ends with clear acceptance criteria.

---

## Shared Context Block

> Paste this at the top of **every** agent prompt.

```
# Project: VectorPrime

VectorPrime is a hardware-aware LLM inference optimizer. It profiles the host
machine, benchmarks candidate runtime + quantization configurations, and
outputs the best inference setup for a given model as a CLI installable via
`pip install vectorprime`.

## Repository root: /home/daniel/llm-forge/

## Architecture

Rust workspace crates (under crates/):
  vectorprime-core       — shared types, enums, traits, errors (no I/O)
  vectorprime-hardware   — CPU / GPU / RAM profiler
  vectorprime-model-ir   — GGUF + ONNX IR parser (param_count, architecture, ctx_len, layers)
  vectorprime-runtime    — RuntimeAdapter trait + Ollama / TensorRT / ONNX / llama.cpp adapters
  vectorprime-optimizer  — search space generator + async benchmark loop + selector
  vectorprime-export     — Ollama model export
  vectorprime-bindings   — PyO3 Python bindings

Python (under python/vectorprime/):
  cli.py                   — argparse CLI → calls _vectorprime bindings
  onnx_runner.py           — Python helper called by ONNX adapter
  gguf_to_onnx_runner.py   — GGUF → ONNX conversion bridge
  onnx_to_gguf_runner.py   — ONNX → GGUF conversion bridge
  __init__.py              — re-exports from native module

## Core shared types (vectorprime-core)

pub enum ModelFormat        { ONNX, GGUF }
pub enum SimdLevel          { None, AVX, AVX2, AVX512 }
pub enum QuantizationStrategy { F16, Q8_0, Q4_K_M, Q4_0, Int8, Int4 }
pub enum RuntimeKind        { LlamaCpp, OnnxRuntime, TensorRT }

pub struct CpuInfo { cores: usize, simd: SimdLevel, cache_kb: u64, frequency_mhz: u64 }
pub struct GpuInfo { vendor: String, model: String, vram_mb: u64, tensor_cores: bool, compute_capability: String }
pub struct RamInfo { total_mb: u64, available_mb: u64 }
pub struct HardwareProfile { cpu: CpuInfo, gpu: Option<GpuInfo>, ram: RamInfo }
pub struct ModelInfo { path: PathBuf, format: ModelFormat, param_count: Option<u64> }
pub struct RuntimeConfig { runtime: RuntimeKind, quantization: QuantizationStrategy, threads: usize, batch_size: usize, gpu_layers: u32 }
pub struct BenchmarkResult { tokens_per_sec: f64, latency_ms: f64, peak_memory_mb: u64 }
pub struct OptimizationResult { config: RuntimeConfig, metrics: BenchmarkResult }

pub trait RuntimeAdapter: Send + Sync {
  fn initialize(&mut self, config: &RuntimeConfig) -> Result<()>;
  fn load_model(&mut self, model: &ModelInfo) -> Result<()>;
  fn run_inference(&self, prompt: &str) -> Result<BenchmarkResult>;
  fn teardown(&mut self) -> Result<()>;
}

pub enum RuntimeError {
  NotInstalled(String),
  UnsupportedFormat,
  InferenceFailed(String),
}

pub trait GpuProbe: Send + Sync {
  fn detect(&self) -> Option<GpuInfo>;
  fn vendor_name(&self) -> &'static str;
}

## Crate dependency order
  vectorprime-core (no deps)
    ↑ vectorprime-hardware, vectorprime-model-ir, vectorprime-runtime
        ↑ vectorprime-optimizer
            ↑ vectorprime-export
                ↑ vectorprime-bindings
                    ↑ python/vectorprime/cli.py

## Rules that apply to all stages
- Use anyhow::Result for all fallible functions
- No unwrap() in production paths — use ? operator
- All public items must have /// doc comments
- Parsing helpers must be pure functions (no subprocess) with unit tests
- Missing binaries → RuntimeError::NotInstalled, never panic
```

---

## Stage 0 — Project Scaffolding

> **Prerequisite:** None  
> **Unblocks:** Stage 1

```
[PASTE SHARED CONTEXT BLOCK]

## Stage 0 — Your Task: Project Scaffolding

Create the complete compilable skeleton. No business logic yet.
Working directory: /home/daniel/llm-forge/ (already exists)

### 1. Root Cargo.toml (workspace)

[workspace]
resolver = "2"
members = [
  "crates/vectorprime-core",
  "crates/vectorprime-hardware",
  "crates/vectorprime-model-ir",
  "crates/vectorprime-runtime",
  "crates/vectorprime-optimizer",
  "crates/vectorprime-export",
  "crates/vectorprime-bindings",
]

### 2. Create each crate: crates/<name>/Cargo.toml + crates/<name>/src/lib.rs

vectorprime-core     deps: serde(derive), serde_json, anyhow, thiserror
vectorprime-hardware deps: vectorprime-core(path), num_cpus="1", sysinfo="0.30", raw-cpuid="11"
vectorprime-model-ir deps: vectorprime-core(path), anyhow, onnx-protobuf="0.2.3", protobuf="=3.4.0"
vectorprime-runtime  deps: vectorprime-core(path), anyhow, tokio(full)
vectorprime-optimizer deps: vectorprime-core(path), vectorprime-hardware(path), vectorprime-runtime(path), tokio(full), anyhow
vectorprime-export   deps: vectorprime-core(path), anyhow, serde_json
vectorprime-bindings deps: all 6 crates above (path), pyo3={version="0.24",features=["extension-module"]}
                     [lib] crate-type = ["cdylib"]

Each src/lib.rs: // placeholder (empty)

### 3. Python package

python/vectorprime/__init__.py  → empty
python/vectorprime/cli.py       → def main(): print("vectorprime"); if __name__=="__main__": main()

### 4. pyproject.toml at root

[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "vectorprime"
version = "0.1.0"
requires-python = ">=3.9"

[project.scripts]
vectorprime = "vectorprime.cli:main"

[tool.maturin]
python-source = "python"
module-name = "vectorprime._vectorprime"

### 5. Verify

Run: cargo build
Run: pip install maturin && maturin develop --skip-install

Both must exit 0.

## Acceptance Criteria
- cargo build exits 0
- All 7 crates exist with Cargo.toml + src/lib.rs
- python/vectorprime/{__init__.py, cli.py} exist
- pyproject.toml present at repo root
```

---

## Stage 1 — Core Types & Traits

> **Prerequisite:** Stage 0  
> **Unblocks:** All other stages

```
[PASTE SHARED CONTEXT BLOCK]

## Stage 1 — Your Task: vectorprime-core

Implement ALL shared types, enums, and traits in:
/home/daniel/llm-forge/crates/vectorprime-core/src/lib.rs

No I/O. No platform code. Pure data + traits only.

### Implement exactly:
- All enums from shared context (derive Debug, Clone, PartialEq, Serialize, Deserialize)
- All structs from shared context (derive Debug, Clone, Serialize, Deserialize)
- RuntimeAdapter trait (use anyhow::Result)
- RuntimeError enum with thiserror (#[error] annotations)
- GpuProbe trait (as shown in shared context)

All items must be pub and re-exported from lib.rs.

### Unit tests required
- Construct each major struct and serialize with serde_json::to_string()
- JSON roundtrip test for HardwareProfile
- JSON roundtrip test for OptimizationResult

## Acceptance Criteria
- cargo test -p vectorprime-core passes
- All types are pub and accessible from other crates
- Zero platform-specific code (no std::process, no file I/O)
```

---

## Stage 2 — Hardware Profiler

> **Prerequisite:** Stage 1  
> **Parallelizable with:** Stage 3A

```
[PASTE SHARED CONTEXT BLOCK]

## Stage 2 — Your Task: vectorprime-hardware

Implement hardware detection in:
/home/daniel/llm-forge/crates/vectorprime-hardware/src/

### File structure
src/lib.rs          pub fn profile() -> HardwareProfile
src/cpu.rs          CPU detection
src/gpu/mod.rs      pub fn probe_all() -> Option<GpuInfo>
src/gpu/nvidia.rs   NvidiaProbe implementing GpuProbe
src/mem.rs          RAM detection

### CPU (cpu.rs)
- Core count: num_cpus crate
- SIMD: raw-cpuid crate → AVX/AVX2/AVX512F flags
- L2 cache size in KB from CPUID
- CPU frequency from CPUID; fallback: parse /proc/cpuinfo "cpu MHz"
Returns: CpuInfo { cores, simd, cache_kb, frequency_mhz }

### GPU (gpu/nvidia.rs — NvidiaProbe)
Shell out: nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits
Parse CSV: "RTX 4090, 24564, 8.9"
  vram_mb = integer field
  tensor_cores = true if compute_capability major >= 7
  vendor = "NVIDIA"
If nvidia-smi absent or fails → return None (no panic)

probe_all() tries NvidiaProbe; stubs for AMD/Metal return None.

### RAM (mem.rs)
Parse /proc/meminfo for MemTotal and MemAvailable (in kB → convert to MB).
Fallback: sysinfo::System cross-platform.

### lib.rs
pub fn profile() -> HardwareProfile {
  HardwareProfile { cpu: cpu::detect(), gpu: gpu::probe_all(), ram: mem::detect() }
}

### Unit tests required
- test_cpu_has_cores: profile().cpu.cores >= 1
- test_ram_total_positive: profile().ram.total_mb > 0
- test_nvidia_parse: pure function parses CSV string → correct GpuInfo (no subprocess)
- test_profile_serializes: profile() → valid JSON

## Acceptance Criteria
- cargo test -p vectorprime-hardware passes on Linux
- profile() never panics when no GPU present
- NvidiaProbe CSV parsing is tested as a pure function
```

---

## Stage 3A — Runtime Dispatch Layer

> **Prerequisite:** Stage 1  
> **Unblocks:** Stages 3B, 3C, 3D (run those in parallel after this)

```
[PASTE SHARED CONTEXT BLOCK]

## Stage 3A — Your Task: Runtime Dispatch Layer

Build the adapter registry and dispatch logic with stub adapters in:
/home/daniel/llm-forge/crates/vectorprime-runtime/src/

### File structure
src/lib.rs        re-exports + AdapterRegistry
src/dispatch.rs   dispatch() function
src/llamacpp.rs   LlamaCppAdapter stub
src/onnx.rs       OnnxAdapter stub
src/tensorrt.rs   TensorRtAdapter stub

### AdapterRegistry (lib.rs)
pub struct AdapterRegistry { adapters: HashMap<RuntimeKind, Box<dyn RuntimeAdapter>> }
impl AdapterRegistry {
  pub fn new() -> Self   → registers all three adapters
  pub fn get_mut(&mut self, kind: &RuntimeKind) -> Option<&mut dyn RuntimeAdapter>
}

### dispatch.rs
pub fn dispatch(registry: &mut AdapterRegistry, config: &RuntimeConfig, model: &ModelInfo, prompt: &str) -> Result<BenchmarkResult>
Calls: initialize → load_model → run_inference (warmup, discard) → run_inference (record) → teardown
On RuntimeError::NotInstalled → propagate, do not panic.

### Stub adapters (same pattern for all three)
pub struct LlamaCppAdapter { config: Option<RuntimeConfig> }
impl RuntimeAdapter for LlamaCppAdapter {
  fn initialize(&mut self, _: &RuntimeConfig) -> Result<()> {
    Err(anyhow::anyhow!(RuntimeError::NotInstalled("llama-cli".into())))
  }
  fn load_model(&mut self, _: &ModelInfo) -> Result<()> { Ok(()) }
  fn run_inference(&self, _: &str) -> Result<BenchmarkResult> { unimplemented!("Stage 3B") }
  fn teardown(&mut self) -> Result<()> { Ok(()) }
}

### Unit tests
- test_registry_has_all_kinds: get_mut for each RuntimeKind returns Some
- test_dispatch_not_installed: dispatch returns Err containing "not installed"

## Acceptance Criteria
- cargo test -p vectorprime-runtime passes
- All three stub adapters compile and implement RuntimeAdapter
- dispatch() propagates NotInstalled as structured error, not panic
```

---

## Stage 3B — llama.cpp Adapter

> **Prerequisite:** Stage 3A  
> **Parallelizable with:** Stages 3C, 3D

```
[PASTE SHARED CONTEXT BLOCK]

## Stage 3B — Your Task: llama.cpp Adapter

Fill in LlamaCppAdapter in:
/home/daniel/llm-forge/crates/vectorprime-runtime/src/llamacpp.rs
(stub struct already exists from Stage 3A)

Add to Cargo.toml: which = "4"

### initialize()
1. which::which("llama-cli") → if Err, return RuntimeError::NotInstalled("llama-cli")
2. Run llama-cli --version, capture + store version string
3. Store config in self

### load_model()
1. Check file exists
2. Read first 4 bytes → verify == b"GGUF", else return RuntimeError::UnsupportedFormat
3. Check model.format == ModelFormat::GGUF, else return UnsupportedFormat

### run_inference()
Build Command:
  llama-cli -m <path> --threads <N> --n-gpu-layers <N> --ctx-size 512
            -p "<prompt>" --n-predict 50 --log-disable
Parse stdout with this pure helper:

pub fn parse_llama_timings(output: &str) -> Option<(f64, f64)>
  Look for: "llama_print_timings:     eval time = X ms / Y tokens (Z ms per token, W tokens per second)"
  Return: (W as tokens_per_sec, Z as latency_ms)

peak_memory_mb: estimate = (param_count * bytes_per_quant) / 1_000_000 or 0 if unknown.

### teardown() → Ok(())

### Unit tests required
- test_parse_timings_valid: given timing string → correct (tps, latency) values
- test_parse_timings_missing: unrelated string → None
- test_unsupported_format_onnx: ONNX model → Err(UnsupportedFormat)
- test_load_bad_magic: file with wrong bytes → Err(UnsupportedFormat)

## Acceptance Criteria
- cargo test -p vectorprime-runtime passes (including new tests)
- parse_llama_timings is pure (no subprocess)
- initialize() returns NotInstalled cleanly when llama-cli absent
```

---

## Stage 3C — ONNX Runtime Adapter

> **Prerequisite:** Stage 3A  
> **Parallelizable with:** Stages 3B, 3D

```
[PASTE SHARED CONTEXT BLOCK]

## Stage 3C — Your Task: ONNX Runtime Adapter

Fill in OnnxAdapter and create the Python helper in:
  /home/daniel/llm-forge/crates/vectorprime-runtime/src/onnx.rs
  /home/daniel/llm-forge/python/vectorprime/onnx_runner.py

### onnx_runner.py (stdin→stdout JSON bridge)

Input JSON (stdin):
  { "model_path": "...", "execution_provider": "CUDAExecutionProvider"|"CPUExecutionProvider",
    "threads": 8, "batch_size": 1, "prompt_tokens": 50 }

Output JSON (stdout):
  { "tokens_per_sec": 45.2, "latency_ms": 210.0, "peak_memory_mb": 3400 }
  OR { "error": "message" } on failure

The script must:
1. Try to import onnxruntime; on ImportError → output {"error": "onnxruntime not installed"}
2. Create InferenceSession with requested provider; fall back to CPU if CUDA unavailable
3. Run 5 iterations with dummy zero inputs (shape from model metadata)
4. Return mean latency, tokens_per_sec = prompt_tokens / mean_latency_sec, peak_memory_mb via tracemalloc

Support --check flag: just verify onnxruntime imports and exit 0.

### OnnxAdapter (onnx.rs)

initialize(): check python3 on PATH; run `python3 onnx_runner.py --check`; on nonzero → error
load_model(): verify file exists + model.format == ModelFormat::ONNX; else UnsupportedFormat
run_inference(): pipe JSON config to python3 onnx_runner.py via stdin; read stdout JSON

Pure parsing helper:
pub fn parse_onnx_output(json_str: &str) -> Result<BenchmarkResult>
  If JSON has "error" key → return Err with that message

### Unit tests
- test_parse_onnx_output_valid: valid JSON → BenchmarkResult
- test_parse_onnx_output_error: {"error":"X"} → Err
- test_load_wrong_format: GGUF format → UnsupportedFormat

## Acceptance Criteria
- cargo test -p vectorprime-runtime passes
- onnx_runner.py handles missing onnxruntime with JSON error, not exception
- parse_onnx_output is pure with unit tests
```

---

## Stage 3D — TensorRT Adapter

> **Prerequisite:** Stage 3A  
> **Parallelizable with:** Stages 3B, 3C

```
[PASTE SHARED CONTEXT BLOCK]

## Stage 3D — Your Task: TensorRT Adapter

Fill in TensorRtAdapter in:
/home/daniel/llm-forge/crates/vectorprime-runtime/src/tensorrt.rs

### initialize()
1. which::which("trtexec") → if missing: RuntimeError::NotInstalled("trtexec")
2. Run: nvidia-smi --query-gpu=compute_cap --format=csv,noheader
   Parse compute_capability; if major < 7 → return Err("TensorRT requires compute ≥ 7.0")
3. Store config

### load_model()
Verify file exists + model.format == ModelFormat::ONNX; else UnsupportedFormat

### run_inference()
Command:
  trtexec --onnx=<path> --<precision_flag> --batch=<N> --workspace=4096
          --iterations=10 --warmUp=2000 --duration=0

Precision mapping (pure fn):
  fn quant_to_flag(q: &QuantizationStrategy) -> &'static str
    F16, Q8_0, Q4_K_M, Q4_0 → "--fp16"
    Int8, Int4              → "--int8"

Pure parsing helpers:
  pub fn parse_throughput(output: &str) -> Option<f64>   // "Throughput: X qps"
  pub fn parse_latency(output: &str) -> Option<f64>      // "Latency: ... mean = Z ms"
  pub fn parse_memory(output: &str) -> Option<u64>       // "GPU Memory: X MiB"

### Unit tests
- test_parse_throughput: mock trtexec output → correct f64
- test_parse_latency_mean: extracts mean field
- test_parse_memory: extracts MiB value
- test_quant_flag_int8: Int8 → "--int8"
- test_load_gguf_unsupported: GGUF → UnsupportedFormat

## Acceptance Criteria
- cargo test -p vectorprime-runtime passes
- All parsers are pure functions with unit tests
- initialize() returns NotInstalled when trtexec absent, no panic
```

---

## Stage 4 — Optimization Engine

> **Prerequisite:** Stages 1, 2, 3A, 3B, 3C, 3D  
> **Unblocks:** Stage 5

```
[PASTE SHARED CONTEXT BLOCK]

## Stage 4 — Your Task: Optimization Engine

Build the optimization loop in:
/home/daniel/llm-forge/crates/vectorprime-optimizer/src/

### File structure
src/lib.rs       pub async fn run_optimization(...)
src/search.rs    candidate generation + pruning
src/benchmark.rs async parallel benchmark loop
src/selector.rs  best result picker

### search.rs

pub fn generate_candidates(hw: &HardwareProfile, model: &ModelInfo) -> Vec<RuntimeConfig>

Rules:
- GGUF → [LlamaCpp] with quants [Q4_K_M, Q4_0, Q8_0, F16]
- ONNX → [OnnxRuntime, TensorRT] with quants [F16, Int8]
  (TensorRT only if NVIDIA GPU with compute_capability major >= 7)
- Thread counts: [cores/2, cores, cores*2] clamped to [1, 64]
- GPU layers: [0, 10, 20, 33] if GPU present; [0] only if not
- Prune: skip configs where estimated VRAM > hw.gpu.vram_mb * 0.9
  Estimate VRAM: param_count * bytes_per_param(quant) / 1_000_000
  bytes_per_param: F16=2.0, Q8_0=1.0, Q4_K_M=0.5, Q4_0=0.5, Int8=1.0, Int4=0.5

pub fn bytes_per_param(q: &QuantizationStrategy) -> f64   ← pure fn

### benchmark.rs

pub async fn run_benchmarks(
  candidates: Vec<RuntimeConfig>,
  model: &ModelInfo,
  hw: &HardwareProfile,
) -> Vec<(RuntimeConfig, Result<BenchmarkResult>)>

Use tokio::sync::Semaphore with limit=3.
Each task: fresh AdapterRegistry → dispatch() → return (config, result).
Include Err results — do not drop them.

const BENCH_PROMPT: &str = "Summarize the following text in one sentence:";

### selector.rs

pub fn select_best(
  results: Vec<(RuntimeConfig, Result<BenchmarkResult>)>,
  hw: &HardwareProfile,
) -> Option<OptimizationResult>

1. Filter out Err (log to stderr)
2. Filter out peak_memory_mb > hw.ram.available_mb * 0.9
3. Sort by tokens_per_sec descending
4. Return top as OptimizationResult or None

### lib.rs public API
pub async fn run_optimization(model: ModelInfo, hw: HardwareProfile) -> Result<OptimizationResult>
  → generate candidates → benchmark → select_best → Ok(result) or Err("No valid config")

### Unit tests
- test_gguf_candidates_only_llamacpp
- test_onnx_no_gpu_excludes_tensorrt
- test_onnx_with_nvidia_includes_tensorrt
- test_bytes_per_param_all_variants (all return > 0)
- test_select_best_picks_highest_tps
- test_select_best_filters_oom
- test_select_best_empty_input_returns_none

## Acceptance Criteria
- cargo test -p vectorprime-optimizer passes
- select_best never panics on empty input
- Impossible hardware combos produce 0 candidates (not panic)
```

---

## Stage 5 — Ollama Export

> **Prerequisite:** Stage 4  
> **Unblocks:** Stage 6

```
[PASTE SHARED CONTEXT BLOCK]

## Stage 5 — Your Task: Ollama Export

Implement in:
/home/daniel/llm-forge/crates/vectorprime-export/src/lib.rs

Add to dev-dependencies: tempfile = "3"

### Public API

pub struct ExportManifest {
  pub output_dir: PathBuf,
  pub modelfile_path: PathBuf,
  pub model_gguf_path: PathBuf,
  pub ollama_commands: Vec<String>,
}

pub fn export_ollama(
  result: &OptimizationResult,
  model_path: &Path,
  output_dir: &Path,
) -> Result<ExportManifest>

### Steps
1. Create output_dir (and parents) if it doesn't exist
2. Resolve GGUF path:
   - .gguf extension → use directly
   - .onnx extension → look for convert_hf_to_gguf.py on PATH
     If found: run conversion → output_dir/model.gguf
     If not:   return Err("ONNX-to-GGUF conversion requires llama.cpp's
               convert_hf_to_gguf.py on PATH")
3. Copy GGUF to output_dir/model.gguf (if not already there)
4. Write output_dir/Modelfile:
   FROM ./model.gguf
   PARAMETER num_thread <threads>
   PARAMETER num_gpu <gpu_layers>
   PARAMETER num_ctx 4096
   # Generated by VectorPrime
5. Write output_dir/metadata.json (pretty-printed OptimizationResult JSON)
6. Return ExportManifest with:
   ollama_commands = [
     "ollama create mymodel -f <modelfile_path>",
     "ollama run mymodel"
   ]

pub fn print_export_summary(manifest: &ExportManifest)
  Prints: output dir, file paths, ollama commands to stdout.

### Unit tests (use tempfile::tempdir())
- test_export_creates_modelfile
- test_modelfile_contains_from
- test_threads_in_modelfile
- test_metadata_json_valid
- test_onnx_without_converter_returns_err (descriptive error, no panic)
- test_manifest_has_two_commands

## Acceptance Criteria
- cargo test -p vectorprime-export passes
- export_ollama creates Modelfile, model.gguf, metadata.json
- ONNX without converter returns descriptive error string
```

---

## Stage 6 — Python Bindings (PyO3)

> **Prerequisite:** Stages 2, 3, 4, 5  
> **Unblocks:** Stage 7

```
[PASTE SHARED CONTEXT BLOCK]

## Stage 6 — Your Task: PyO3 Python Bindings

Implement in:
/home/daniel/llm-forge/crates/vectorprime-bindings/src/lib.rs

### Python classes (#[pyclass])

PyHardwareProfile:
  #[staticmethod] fn detect() -> PyResult<Self>
  fn to_json(&self) -> PyResult<String>
  fn __repr__(&self) -> String  →  "HardwareProfile(cpu_cores=N, gpu=Model or None)"
  #[getter] cpu_cores() -> usize
  #[getter] gpu_model() -> Option<String>
  #[getter] gpu_vram_mb() -> Option<u64>
  #[getter] ram_total_mb() -> u64

PyOptimizationResult:
  fn to_json(&self) -> PyResult<String>
  fn __repr__(&self) -> String  →  "OptimizationResult(runtime=X, quant=Y, tps=Z)"
  #[getter] runtime() -> String   (format!("{:?}", config.runtime))
  #[getter] quantization() -> String
  #[getter] threads() -> usize
  #[getter] gpu_layers() -> u32
  #[getter] tokens_per_sec() -> f64
  #[getter] latency_ms() -> f64
  #[getter] peak_memory_mb() -> u64

### Module-level functions (#[pyfunction])

profile_hardware() -> PyResult<PyHardwareProfile>

optimize(model_path: &str, format: &str) -> PyResult<PyOptimizationResult>
  Parse format ("onnx"|"gguf") → ModelInfo
  Profile hardware → call run_optimization via tokio::runtime::Runtime::block_on()
  Map errors with: fn to_py_err(e: anyhow::Error) -> PyErr { PyRuntimeError::new_err(format!("{:#}", e)) }

export_ollama(result: &PyOptimizationResult, output_dir: &str) -> PyResult<String>
  Call vectorprime_export::export_ollama() → return manifest as JSON string

### Module registration
#[pymodule] fn _vectorprime(m: &Bound<'_, PyModule>) -> PyResult<()>
  Register: PyHardwareProfile, PyOptimizationResult, profile_hardware, optimize, export_ollama,
            analyze_model, convert_gguf_to_onnx, convert_onnx_to_gguf

### Verify after implementation
maturin develop
python3 -c "import vectorprime; print(vectorprime.profile_hardware())"

## Acceptance Criteria
- maturin develop exits 0
- python3 -c "import vectorprime; print(vectorprime.profile_hardware())" works
- All three functions importable from Python
- Rust panics → Python RuntimeError (never crash the interpreter)
```

---

## Stage 7 — CLI & Python Package

> **Prerequisite:** Stage 6  
> **Unblocks:** Stage 8

```
[PASTE SHARED CONTEXT BLOCK]

## Stage 7 — Your Task: CLI and Python Packaging

Implement in:
/home/daniel/llm-forge/python/vectorprime/cli.py

### Commands
  vectorprime profile
  vectorprime optimize <model_path> [--format onnx|gguf] [--max-memory MB]
                                    [--gpu MODEL] [--latency MS] [--output PATH]
  vectorprime export-ollama <model_path> [--result FILE] [--output-dir DIR]
  vectorprime convert-to-onnx <input_path> [--output PATH]
  vectorprime convert-to-gguf <input_path> [--output PATH]

Use argparse with subparsers.

### profile → call _vectorprime.profile_hardware(); print(hw.to_json())

### optimize
1. Auto-detect format from extension if --format not given:
   def detect_format(path: str) -> str:
     if path.endswith(".gguf"): return "gguf"
     if path.endswith(".onnx"): return "onnx"
     raise ValueError(f"Cannot detect format: {path}")
2. Call result = _vectorprime.optimize(model_path, format, gpu, latency, output)
3. Print formatted summary:
   ─────────────────────────────────
   VectorPrime Optimization Result
   ─────────────────────────────────
   Runtime:       {result.runtime}
   Quantization:  {result.quantization}
   Threads:       {result.threads}
   GPU Layers:    {result.gpu_layers}
   Throughput:    {result.tokens_per_sec:.1f} tokens/sec
   Latency:       {result.latency_ms:.1f} ms
   Memory:        {result.peak_memory_mb / 1024:.1f} GB peak
   ─────────────────────────────────
4. Write result.to_json() to <model_path>.vectorprime_result.json

### export-ollama
1. Load result from --result JSON (run optimize inline if absent)
2. Call _vectorprime.export_ollama(result, output_dir)
3. Parse returned JSON, print file paths and ollama commands

### Error handling
Wrap all _vectorprime calls in try/except RuntimeError:
  print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)

### Unit tests (tests/test_cli.py)
- test_detect_format_gguf
- test_detect_format_onnx
- test_detect_format_unknown → ValueError

### Verify
maturin develop
vectorprime --help       (exit 0, lists 5 subcommands)
vectorprime profile      (exit 0, prints JSON)
vectorprime optimize --help

## Acceptance Criteria
- vectorprime --help lists profile, optimize, export-ollama, convert-to-onnx, convert-to-gguf
- vectorprime profile prints valid JSON to stdout
- Unknown file extensions print "ERROR:" to stderr and exit 1
```

---

## Stage 8 — Integration Tests

> **Prerequisite:** Stage 7

```
[PASTE SHARED CONTEXT BLOCK]

## Stage 8 — Your Task: Integration Tests

Write the full integration test suite in:
/home/daniel/llm-forge/tests/

Install: pip install pytest

### Fixture guard (add to every fixture-dependent test)
import os, pytest
FIXTURES = os.path.exists("tests/fixtures/tiny.gguf")
skip_no_fixtures = pytest.mark.skipif(not FIXTURES, reason="fixtures not downloaded")

### Download fixtures (document commands, don't auto-run)
GGUF: curl -L <tinyllama-q4_k_m.gguf URL> -o tests/fixtures/tiny.gguf
ONNX: pip install optimum[exporters]
      optimum-cli export onnx --model distilbert-base-uncased tests/fixtures/tiny_onnx/

### tests/integration/test_hardware.py
- test_profile_has_cores: profile().cpu_cores >= 1
- test_profile_json_valid: profile().to_json() is valid JSON with "cpu" key
- test_ram_positive: profile().ram_total_mb > 0

### tests/integration/test_optimize.py
- @skip_no_fixtures test_optimize_gguf: optimize("tiny.gguf","gguf").runtime == "LlamaCpp"
- @skip_no_fixtures test_optimize_onnx: optimize("tiny_onnx/model.onnx","onnx").tokens_per_sec > 0
- test_optimize_missing_file: optimize("/nonexistent.gguf","gguf") raises RuntimeError

### tests/integration/test_export.py (use tmp_path fixture)
- @skip_no_fixtures test_export_creates_files: Modelfile + model.gguf + metadata.json all exist
- @skip_no_fixtures test_modelfile_has_from: "FROM ./model.gguf" in Modelfile content
- @skip_no_fixtures test_manifest_has_commands: manifest JSON has 2 ollama_commands

### tests/integration/test_cli.py (use subprocess.run)
- test_profile_exits_zero: vectorprime profile → returncode 0 + valid JSON stdout
- test_help_exits_zero: vectorprime --help → returncode 0
- test_optimize_missing_file_nonzero: vectorprime optimize /nonexistent.gguf → returncode != 0

### GPU tests
Gate with: @pytest.mark.skipif(not os.environ.get("VECTORPRIME_GPU_TESTS"), reason="set VECTORPRIME_GPU_TESTS=1")

### Run all tests
pytest tests/ -v --tb=short

## Acceptance Criteria
- All non-fixture tests pass on any Linux machine (with or without GPU)
- vectorprime profile CLI test passes as full-stack smoke test
- No test uses unwrap-style panic; all errors are asserted, not swallowed
```

---

## Execution Wave Summary

| Wave | Stages | Run |
|------|--------|-----|
| 1 | 0 → 1 | Sequential |
| 2 | 2 + 3A | Parallel |
| 3 | 3B + 3C + 3D | Parallel |
| 4 | 4 → 5 → 6 → 7 → 8 | Sequential |
