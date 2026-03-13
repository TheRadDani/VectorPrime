// Location: crates/vectorprime-optimizer/src/search.rs
//
// Generates the candidate configuration space for the benchmark loop and
// provides the internal context structs (HardwareContext, ModelContext) used
// by the 7-stage Efficient Search Pipeline in lib.rs.
//
// Public API (used by lib.rs and tests):
//   - bytes_per_param            : bytes per model param for a quantization
//   - estimate_max_gpu_layers    : conservative VRAM-based layer count estimate
//   - default_base_config        : sensible starting RuntimeConfig
//   - generate_candidates        : cartesian product (backward-compat / fallback)
//   - generate_stage_candidates  : single-parameter staged candidates (old 5-stage)
//   - HardwareContext / ModelContext : internal pipeline state for 7-stage search
//   - preselect_runtimes         : Stage 3 pruning + scoring
//   - quantization_candidates    : Stage 4 narrow quant list
//   - gpu_layer_candidates       : Stage 5 narrow GPU-layer list
//   - thread_candidates          : Stage 6 narrow thread list

use vectorprime_core::{
    GpuInfo, GpuVendor, HardwareProfile, ModelFormat, ModelInfo, QuantizationStrategy,
    RuntimeConfig, RuntimeKind, SimdLevel,
};

/// Return the storage cost in bytes per model parameter for a given
/// quantization strategy.
pub fn bytes_per_param(q: &QuantizationStrategy) -> f64 {
    match q {
        QuantizationStrategy::F16 => 2.0,
        QuantizationStrategy::Q8_0 | QuantizationStrategy::Int8 => 1.0,
        QuantizationStrategy::Q4_K_M | QuantizationStrategy::Q4_0 | QuantizationStrategy::Int4 => {
            0.5
        }
    }
}

/// Estimate VRAM usage in MB for a given model + quantization.
fn estimate_vram_mb(model: &ModelInfo, q: &QuantizationStrategy) -> Option<f64> {
    let params = model.param_count? as f64;
    Some(params * bytes_per_param(q) / 1_000_000.0)
}

/// Return `true` when the detected GPU is from NVIDIA.
///
/// TRT eligibility is based purely on vendor identity — we let `trtexec` itself
/// report whether the specific GPU supports the requested precision at runtime.
/// This removes the compute-capability threshold (>= 7.0) that was hardcoded
/// here and prevents us from unnecessarily excluding newer GPUs or
/// misclassifying vendor-unknown devices.
fn is_nvidia_gpu(hw: &HardwareProfile) -> bool {
    hw.gpu
        .as_ref()
        .map(|g| g.vendor == GpuVendor::Nvidia)
        .unwrap_or(false)
}

/// Estimate the maximum number of transformer layers that fit in GPU VRAM.
///
/// Uses F16 precision as the worst-case storage cost to produce a conservative
/// upper bound. When `model.param_count` is unknown, falls back to
/// VRAM-capacity tiers that represent practical offload limits.
///
/// The returned value is the *maximum* layers to consider; the caller is
/// responsible for generating evenly-spaced steps from 0 to this value.
pub fn estimate_max_gpu_layers(gpu: &GpuInfo, model: &ModelInfo) -> u32 {
    if let Some(params) = model.param_count {
        // F16 worst-case: 2 bytes per parameter.
        let model_size_mb = params as f64 * 2.0 / 1_048_576.0;
        let vram_mb = gpu.vram_mb as f64;

        if model_size_mb <= 0.0 {
            return 0;
        }

        // Fraction of the model that fits in VRAM (capped at 1.0 for full
        // offload). Use a reference of 32 transformer layers to convert the
        // fraction into a layer count — this is an approximation suitable for
        // pruning the search space; real layer counts vary by architecture.
        let fit_fraction = (vram_mb / model_size_mb).min(1.0);
        const REFERENCE_LAYERS: f64 = 32.0;
        (fit_fraction * REFERENCE_LAYERS).floor() as u32
    } else {
        // Param count unknown: use VRAM tiers as capacity estimates.
        // These are not architecture constants — they represent practical
        // limits on how many layers can be offloaded given the VRAM budget.
        match gpu.vram_mb {
            0..=4095 => 16,
            4096..=12287 => 32,
            _ => 48,
        }
    }
}

/// Classifies a model's inference bottleneck based on architecture metrics.
///
/// - `MemoryBound`: large KV cache or narrow FFN relative to hidden size
///   (attention dominates). Favours aggressive quantization.
/// - `ComputeBound`: wide FFN (FFN >> 8× hidden, e.g. Mixtral MoE). Favours
///   precision-preserving formats when tensor cores are available.
/// - `Balanced`: in between; apply default heuristics.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum WorkloadType {
    MemoryBound,
    ComputeBound,
    Balanced,
}

// ──────────────────────────────────────────────────────────────────────────────
// 7-Stage Pipeline Internal Context Structs
// ──────────────────────────────────────────────────────────────────────────────

/// Internal hardware context produced by Stage 1 of the 7-stage pipeline.
///
/// Derived purely from `HardwareProfile` — no benchmarks, no I/O.
/// Centralises hardware facts so later stages reference a single source.
pub(crate) struct HardwareContext {
    pub has_gpu: bool,
    pub is_nvidia: bool,
    pub vram_mb: Option<u64>,
    pub ram_available_mb: u64,
    pub cpu_cores: u32,
    /// SIMD level — retained for future stage use (e.g. scoring CPU-only runtimes).
    #[allow(dead_code)]
    pub simd_level: SimdLevel,
    /// GPU compute capability — retained for future TensorRT tier scoring.
    #[allow(dead_code)]
    pub gpu_compute_cap: Option<(u32, u32)>,
}

impl HardwareContext {
    /// Build from a full hardware profile snapshot.
    pub fn from_hw(hw: &HardwareProfile) -> Self {
        let has_gpu = hw.gpu.is_some();
        let is_nvidia = hw
            .gpu
            .as_ref()
            .map(|g| g.vendor == GpuVendor::Nvidia)
            .unwrap_or(false);
        let vram_mb = hw.gpu.as_ref().map(|g| g.vram_mb);
        let gpu_compute_cap = hw.gpu.as_ref().and_then(|g| g.compute_capability);

        HardwareContext {
            has_gpu,
            is_nvidia,
            vram_mb,
            ram_available_mb: hw.ram.available_mb,
            cpu_cores: hw.cpu.core_count,
            simd_level: hw.cpu.simd_level.clone(),
            gpu_compute_cap,
        }
    }
}

/// Internal model context produced by Stage 2 of the 7-stage pipeline.
///
/// Derived from `ModelInfo` + `HardwareContext` — no benchmarks, no I/O.
pub(crate) struct ModelContext {
    pub workload: WorkloadType,
    pub param_count: Option<u64>,
    /// Estimated model memory in MB at FP16 (2 bytes/param).
    pub memory_fp16_mb: Option<f64>,
    /// True when KV cache pressure is high relative to available memory.
    pub kv_pressure: bool,
}

impl ModelContext {
    /// Build from model info and the hardware context computed in Stage 1.
    pub fn from_model_and_hw(model: &ModelInfo, hw_ctx: &HardwareContext) -> Self {
        let workload = classify_workload(model);

        // FP16 memory: prefer the pre-computed field; fall back to param count.
        let memory_fp16_mb = model.memory_footprint_mb.or_else(|| {
            model
                .param_count
                .map(|p| p as f64 * 2.0 / 1_000_000.0)
        });

        // KV pressure: true if KV cache exceeds 30% of VRAM, or 20% of RAM
        // when no GPU is available.
        let kv_pressure = if let Some(kv_mb) = model.kv_cache_size_mb {
            if let Some(vram_mb) = hw_ctx.vram_mb {
                kv_mb > vram_mb as f64 * 0.3
            } else {
                kv_mb > hw_ctx.ram_available_mb as f64 * 0.2
            }
        } else {
            false
        };

        ModelContext {
            workload,
            param_count: model.param_count,
            memory_fp16_mb,
            kv_pressure,
        }
    }

    /// Return `true` when `quant` is predicted to fit within 90% of VRAM.
    ///
    /// If no GPU is present or param_count is unknown, returns `false` (cannot
    /// confirm VRAM fit — caller should allow CPU fallback paths).
    pub fn fits_in_vram(&self, quant: &QuantizationStrategy, vram_mb: Option<u64>) -> bool {
        match (self.param_count, vram_mb) {
            (Some(params), Some(vram)) => {
                let model_mb = params as f64 * bytes_per_param(quant) / 1_000_000.0;
                model_mb <= vram as f64 * 0.9
            }
            _ => false,
        }
    }

    /// Return `true` when `quant` is predicted to fit within 80% of available RAM.
    ///
    /// Used to prune quantizations that cannot fit even in system RAM.
    pub fn fits_in_ram(&self, quant: &QuantizationStrategy, ram_available_mb: u64) -> bool {
        match self.param_count {
            Some(params) => {
                let model_mb = params as f64 * bytes_per_param(quant) / 1_000_000.0;
                model_mb <= ram_available_mb as f64 * 0.8
            }
            // Unknown param count — cannot prune; allow through.
            None => true,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Stage 3: Runtime Preselection
// ──────────────────────────────────────────────────────────────────────────────

/// Preselect up to 2 runtimes based on format, hardware context, and scoring.
///
/// Eliminates runtimes that are incompatible with the model format, require
/// hardware not present, or cannot handle KV pressure. Returns runtimes in
/// descending score order (best first). Also logs eliminated runtimes with
/// their reason.
pub(crate) fn preselect_runtimes(
    hw_ctx: &HardwareContext,
    model_ctx: &ModelContext,
    format: &ModelFormat,
) -> Vec<RuntimeKind> {
    // Scoring table: (RuntimeKind, score, eligible, elimination_reason)
    // Higher score = better fit for this hardware+format combination.
    let candidates: Vec<(RuntimeKind, u32, Option<&'static str>)> = match format {
        ModelFormat::GGUF => vec![
            // LlamaCpp: always eligible for GGUF; best native runtime.
            (RuntimeKind::LlamaCpp, 10, None),
            // Ollama: eligible for GGUF; wraps llama.cpp.
            (RuntimeKind::Ollama, 8, None),
        ],
        ModelFormat::ONNX => {
            let trt_reason = if !hw_ctx.is_nvidia {
                Some("TensorRT requires NVIDIA GPU")
            } else {
                None
            };
            let vllm_reason = if !hw_ctx.has_gpu {
                Some("vLLM requires a GPU")
            } else if model_ctx.kv_pressure && hw_ctx.vram_mb.unwrap_or(0) == 0 {
                Some("vLLM cannot handle KV pressure without VRAM")
            } else {
                None
            };
            vec![
                (RuntimeKind::OnnxRuntime, 8, None),
                (RuntimeKind::TensorRT, 10, trt_reason),
                (RuntimeKind::Vllm, 7, vllm_reason),
            ]
        }
    };

    // Partition into eligible and eliminated, logging each eliminated runtime.
    let mut eligible: Vec<(RuntimeKind, u32)> = Vec::new();
    for (runtime, score, reason) in candidates {
        if let Some(why) = reason {
            eprintln!(
                "[Stage 3/7] Eliminated {:?}: {}",
                runtime, why
            );
        } else {
            eligible.push((runtime, score));
        }
    }

    // Sort descending by score, take top 2.
    eligible.sort_by(|(_, a), (_, b)| b.cmp(a));
    eligible.into_iter().map(|(r, _)| r).take(2).collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Stage 4: Quantization Candidates
// ──────────────────────────────────────────────────────────────────────────────

/// Return ≤ 3 quantization candidates for the narrow benchmark in Stage 4.
///
/// Selection is driven by workload type and VRAM/RAM fit. Quantizations that
/// cannot fit in available memory are pruned before returning.
pub(crate) fn quantization_candidates(
    hw_ctx: &HardwareContext,
    model_ctx: &ModelContext,
    format: &ModelFormat,
) -> Vec<QuantizationStrategy> {
    // Build the workload-ordered preference list.
    let ordered: Vec<QuantizationStrategy> = match format {
        ModelFormat::GGUF => match model_ctx.workload {
            WorkloadType::MemoryBound => vec![
                QuantizationStrategy::Q4_K_M,
                QuantizationStrategy::Q4_0,
                // Q8_0 and F16 only when they fit in VRAM.
            ],
            WorkloadType::ComputeBound => vec![
                QuantizationStrategy::Q8_0,
                QuantizationStrategy::Q4_K_M,
                QuantizationStrategy::F16,
            ],
            WorkloadType::Balanced => vec![
                QuantizationStrategy::Q4_K_M,
                QuantizationStrategy::Q8_0,
            ],
        },
        ModelFormat::ONNX => match model_ctx.workload {
            WorkloadType::MemoryBound => vec![
                QuantizationStrategy::Int8,
                QuantizationStrategy::F16,
            ],
            WorkloadType::ComputeBound | WorkloadType::Balanced => vec![
                QuantizationStrategy::F16,
                QuantizationStrategy::Int8,
            ],
        },
    };

    // Prune quantizations that cannot fit in VRAM (when GPU present) or RAM.
    // A quant is allowed if:
    //   - It fits in VRAM, OR no GPU (CPU path is always allowed if RAM fits)
    //   - It fits in RAM (80% budget)
    let vram_mb = hw_ctx.vram_mb;
    ordered
        .into_iter()
        .filter(|q| {
            // Must fit in RAM (absolute minimum requirement).
            if !model_ctx.fits_in_ram(q, hw_ctx.ram_available_mb) {
                return false;
            }
            // If a GPU is present and the quant doesn't fit in VRAM, only keep
            // it when a CPU RAM fallback is feasible (already confirmed above).
            // We do NOT eliminate it — we just allow it through the RAM check.
            let _ = (vram_mb, model_ctx.fits_in_vram(q, vram_mb));
            true
        })
        .take(3)
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Stage 5: GPU Layer Candidates
// ──────────────────────────────────────────────────────────────────────────────

/// Return ≤ 4 GPU layer counts to benchmark in Stage 5.
///
/// Predicts the optimal layer count based on VRAM capacity and the winning
/// quantization's memory density, then probes a narrow range around that
/// prediction. Always includes 0 (CPU-only baseline).
///
/// Returns `[0]` immediately when no GPU is present.
pub(crate) fn gpu_layer_candidates(
    hw_ctx: &HardwareContext,
    model_ctx: &ModelContext,
    best_quant: &QuantizationStrategy,
    max_layers_f16: u32,
) -> Vec<u32> {
    if !hw_ctx.has_gpu || max_layers_f16 == 0 {
        return vec![0];
    }

    // Scale the F16 max-layer estimate by the quantization density ratio.
    // A Q4_K_M model (0.5 bytes/param) is 4× denser than F16 (2.0 bytes/param),
    // so we can potentially fit 4× more layers in the same VRAM budget.
    // Cap at 2× the F16 estimate to avoid wildly optimistic values when VRAM
    // is very large relative to model size.
    let quant_ratio = 2.0_f64 / bytes_per_param(best_quant);
    let predicted_f = (max_layers_f16 as f64 * quant_ratio)
        .min(max_layers_f16 as f64 * 2.0);
    let predicted = (predicted_f as u32).min(max_layers_f16.saturating_mul(2));

    // Suppress the ratio boost when the model clearly fits in VRAM at F16 —
    // there is nothing additional to gain by scaling beyond max_layers_f16.
    // Use a VRAM-aware cap: if param count is known, cap at the true max
    // based on VRAM; otherwise keep the 2× cap.
    let vram_cap = hw_ctx.vram_mb.and_then(|vram| {
        model_ctx.param_count.map(|params| {
            let bpp = bytes_per_param(best_quant);
            let model_mb = params as f64 * bpp / 1_000_000.0;
            let vram_mb = vram as f64 * 0.9;
            // Layers that fit: ratio of available VRAM to full model size,
            // scaled to 32 reference layers (same approximation as estimate_max_gpu_layers).
            let fit_fraction = (vram_mb / model_mb).min(1.0);
            (fit_fraction * 32.0).floor() as u32
        })
    });
    let predicted = if let Some(cap) = vram_cap {
        predicted.min(cap)
    } else {
        predicted
    };

    let step = (predicted / 4).max(1);
    let mut candidates: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    candidates.insert(0);
    candidates.insert(predicted.saturating_sub(step));
    candidates.insert(predicted);
    // Only add predicted + step when it doesn't overflow the plausible ceiling.
    let upper = predicted.saturating_add(step);
    candidates.insert(upper.min(max_layers_f16.saturating_mul(2)));

    candidates.into_iter().collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Stage 6: Thread Candidates
// ──────────────────────────────────────────────────────────────────────────────

/// Return ≤ 3 thread counts to benchmark in Stage 6.
///
/// Predicts the optimal thread count from hardware and GPU offload ratio,
/// then probes a narrow range around that prediction.
/// Values are clamped to [1, 64] and deduplicated.
pub(crate) fn thread_candidates(hw_ctx: &HardwareContext, best_gpu_layers: u32) -> Vec<u32> {
    let cores = hw_ctx.cpu_cores;

    // When most of the model is on GPU, fewer CPU threads are better
    // (the bottleneck is GPU bandwidth, not CPU compute).
    // Threshold: > 50% of max layers offloaded → GPU-bound heuristic.
    // We use a simple absolute threshold of > 0 layers with a large layer
    // count relative to a 32-layer reference, or just cores when uncertain.
    let predicted = if best_gpu_layers > 16 {
        // GPU-bound: fewer threads avoids contention on the CPU↔GPU transfer.
        (cores / 2).max(1)
    } else {
        // CPU-bound or low GPU offload: use all cores.
        cores
    };

    let half = (predicted / 2).max(1);
    let double = (predicted * 2).min(64);

    let mut candidates: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    candidates.insert(half.clamp(1, 64));
    candidates.insert(predicted.clamp(1, 64));
    candidates.insert(double.clamp(1, 64));

    candidates.into_iter().collect()
}

/// Classify the workload type from model architecture metrics.
///
/// Returns `Balanced` when the required fields are absent.
fn classify_workload(model: &ModelInfo) -> WorkloadType {
    match (model.feed_forward_length, model.hidden_size) {
        (Some(ffn), Some(hidden)) if hidden > 0 => {
            let ffn_ratio = ffn / hidden as u64;
            if ffn_ratio >= 8 {
                WorkloadType::ComputeBound // FFN >> 8× hidden (MoE-style)
            } else if ffn_ratio <= 2 {
                WorkloadType::MemoryBound // attention dominates
            } else {
                WorkloadType::Balanced
            }
        }
        _ => WorkloadType::Balanced,
    }
}

/// Generate the cross-product of (runtime × quantization × threads × gpu_layers)
/// pruned to configs that are plausible on the given hardware.
///
/// Kept intact for backward compatibility and as a fallback. The staged
/// optimizer uses `generate_stage_candidates` instead.
pub fn generate_candidates(hw: &HardwareProfile, model: &ModelInfo) -> Vec<RuntimeConfig> {
    // Classify the workload to influence candidate ordering.
    // All quant/runtime combos are still generated; ordering only affects
    // which candidates are evaluated first (the scorer picks the final winner).
    let workload = classify_workload(model);

    // Per-format runtime + quantization pairings, ordered by workload type.
    let combos = runtime_quant_combos(&model.format, hw, &workload);

    let thread_counts = thread_options(hw);
    let gpu_layers_list = gpu_layer_options(hw, model);

    // KV-cache-aware batch sizing: reduce batch when KV cache pressure is high.
    let batch_size = batch_size_for(hw, model);

    let vram_budget_mb = hw.gpu.as_ref().map(|g| g.vram_mb as f64 * 0.9);

    let mut candidates = Vec::new();
    for (runtime, quant) in combos {
        // Prune: skip if we know the model won't fit in VRAM.
        if let (Some(budget), Some(vram)) = (vram_budget_mb, estimate_vram_mb(model, &quant)) {
            if vram > budget {
                continue;
            }
        }

        for &threads in &thread_counts {
            for &gpu_layers in &gpu_layers_list {
                candidates.push(RuntimeConfig {
                    runtime: runtime.clone(),
                    quantization: quant.clone(),
                    threads,
                    batch_size,
                    gpu_layers,
                });
            }
        }
    }

    candidates
}

// ──────────────────────────────────────────────────────────────────────────────
// Staged candidate generator
// ──────────────────────────────────────────────────────────────────────────────

/// Return the ordered list of quantization strategies for a given model format
/// and workload type.
///
/// Reused by both `generate_candidates` and `generate_stage_candidates` so the
/// workload-aware ordering is applied consistently in both paths.
fn quant_order_for_workload(format: &ModelFormat, workload: &WorkloadType) -> Vec<QuantizationStrategy> {
    match format {
        ModelFormat::GGUF => {
            let mut quants = vec![
                QuantizationStrategy::Q4_K_M,
                QuantizationStrategy::Q4_0,
                QuantizationStrategy::Q8_0,
                QuantizationStrategy::F16,
            ];
            if matches!(workload, WorkloadType::ComputeBound) {
                quants.sort_by_key(|q| match q {
                    QuantizationStrategy::F16 => 0,
                    QuantizationStrategy::Q8_0 => 1,
                    QuantizationStrategy::Q4_K_M => 2,
                    QuantizationStrategy::Q4_0 => 3,
                    _ => 4,
                });
            }
            quants
        }
        ModelFormat::ONNX => {
            // For ONNX the primary quant strategies tried are F16 and Int8.
            // Memory-bound workloads prefer Int8 first.
            if matches!(workload, WorkloadType::MemoryBound) {
                vec![QuantizationStrategy::Int8, QuantizationStrategy::F16]
            } else {
                vec![QuantizationStrategy::F16, QuantizationStrategy::Int8]
            }
        }
    }
}

/// Return the ordered list of (RuntimeKind, QuantizationStrategy) pairs for a
/// given model format and hardware profile.
///
/// Reused across both the cartesian-product generator and the staged generator.
fn runtime_quant_combos(
    format: &ModelFormat,
    hw: &HardwareProfile,
    workload: &WorkloadType,
) -> Vec<(RuntimeKind, QuantizationStrategy)> {
    match format {
        ModelFormat::GGUF => {
            let quants = quant_order_for_workload(format, workload);
            let mut combos: Vec<(RuntimeKind, QuantizationStrategy)> = Vec::new();
            for q in &quants {
                combos.push((RuntimeKind::LlamaCpp, q.clone()));
                combos.push((RuntimeKind::Ollama, q.clone()));
            }
            combos
        }
        ModelFormat::ONNX => {
            let mut v: Vec<(RuntimeKind, QuantizationStrategy)> = vec![
                (RuntimeKind::OnnxRuntime, QuantizationStrategy::F16),
                (RuntimeKind::OnnxRuntime, QuantizationStrategy::Int8),
            ];
            if is_nvidia_gpu(hw) {
                v.push((RuntimeKind::TensorRT, QuantizationStrategy::F16));
                v.push((RuntimeKind::TensorRT, QuantizationStrategy::Int8));
            }
            if hw.gpu.is_some() {
                v.push((RuntimeKind::Vllm, QuantizationStrategy::F16));
                v.push((RuntimeKind::Vllm, QuantizationStrategy::Q8_0));
            }
            if matches!(workload, WorkloadType::MemoryBound) {
                v.sort_by_key(|(_, q)| match q {
                    QuantizationStrategy::Int8 | QuantizationStrategy::Int4 => 0,
                    _ => 1,
                });
            }
            v
        }
    }
}

/// Return the thread count candidates for the given hardware.
///
/// Produces `[cores/2, cores, cores*2]` clamped to `[1, 64]`, deduplicated.
fn thread_options(hw: &HardwareProfile) -> Vec<u32> {
    let cores = hw.cpu.core_count;
    [cores / 2, cores, cores * 2]
        .iter()
        .map(|&t| t.clamp(1, 64))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Return the GPU layer count candidates for the given hardware and model.
///
/// Produces 5 evenly-spaced steps from 0 to `estimate_max_gpu_layers`.
/// Returns `[0]` when no GPU is present.
fn gpu_layer_options(hw: &HardwareProfile, model: &ModelInfo) -> Vec<u32> {
    if let Some(gpu) = &hw.gpu {
        let max_layers = estimate_max_gpu_layers(gpu, model);
        if max_layers == 0 {
            vec![0]
        } else {
            let steps = 4u32;
            (0..=steps)
                .map(|i| (max_layers * i / steps).min(max_layers))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect()
        }
    } else {
        vec![0]
    }
}

/// Compute the KV-cache-aware batch size for the given model and hardware.
///
/// Uses 512 as the base, halved to 256 when KV cache pressure is high (KV
/// cache > 30% of VRAM). Exposed as `batch_size_for_model` for use by lib.rs.
pub(crate) fn batch_size_for_model(hw: &HardwareProfile, model: &ModelInfo) -> u32 {
    batch_size_for(hw, model)
}

fn batch_size_for(hw: &HardwareProfile, model: &ModelInfo) -> u32 {
    let base: u32 = 512;
    if let (Some(kv_mb), Some(vram_mb)) = (
        model.kv_cache_size_mb,
        hw.gpu.as_ref().map(|g| g.vram_mb as f64),
    ) {
        if kv_mb > vram_mb * 0.3 {
            return (base / 2).max(128);
        }
    }
    base
}

/// Return the default (sensible) starting `RuntimeConfig` for staged search.
///
/// Stage 1 will immediately vary the runtime field; the other fields provide
/// reasonable defaults used as the base for all subsequent stages.
pub fn default_base_config(hw: &HardwareProfile) -> vectorprime_core::RuntimeConfig {
    vectorprime_core::RuntimeConfig {
        runtime: RuntimeKind::LlamaCpp,
        quantization: QuantizationStrategy::Q4_K_M,
        threads: hw.cpu.core_count,
        batch_size: 512,
        gpu_layers: 0,
    }
}

/// Generate candidates for a single stage of the staged optimization loop.
///
/// Each stage varies **one** parameter while keeping all other fields fixed to
/// the best values discovered in prior stages (passed in via `base`).
///
/// | Stage | Parameter varied       |
/// |-------|------------------------|
/// | 1     | `RuntimeKind`          |
/// | 2     | `QuantizationStrategy` |
/// | 3     | `gpu_layers`           |
/// | 4     | `threads`              |
/// | 5     | `batch_size`           |
///
/// Any stage value outside 1–5 returns an empty `Vec`.
pub fn generate_stage_candidates(
    stage: u8,
    base: &vectorprime_core::RuntimeConfig,
    hw: &HardwareProfile,
    model: &ModelInfo,
) -> Vec<vectorprime_core::RuntimeConfig> {
    let workload = classify_workload(model);
    let vram_budget_mb = hw.gpu.as_ref().map(|g| g.vram_mb as f64 * 0.9);

    match stage {
        // Stage 1 — vary runtime (and paired quantization) keeping base
        // threads/gpu_layers/batch_size fixed.
        //
        // We use the full (runtime, quant) combo list from the cartesian
        // generator so the workload-aware ordering is preserved. The quant
        // field travels with the runtime here; Stage 2 then sweeps all quant
        // values under the winning runtime.
        1 => {
            let combos = runtime_quant_combos(&model.format, hw, &workload);
            combos
                .into_iter()
                .filter_map(|(runtime, quant)| {
                    // Prune by VRAM budget.
                    if let (Some(budget), Some(vram)) =
                        (vram_budget_mb, estimate_vram_mb(model, &quant))
                    {
                        if vram > budget {
                            return None;
                        }
                    }
                    Some(vectorprime_core::RuntimeConfig {
                        runtime,
                        quantization: quant,
                        threads: base.threads,
                        batch_size: base.batch_size,
                        gpu_layers: base.gpu_layers,
                    })
                })
                .collect()
        }

        // Stage 2 — vary quantization; keep best runtime from Stage 1 fixed.
        //
        // Uses the same workload-aware ordering as `generate_candidates` so
        // memory-bound models still prefer aggressive quants.
        2 => {
            let quants = quant_order_for_workload(&model.format, &workload);
            quants
                .into_iter()
                .filter_map(|quant| {
                    if let (Some(budget), Some(vram)) =
                        (vram_budget_mb, estimate_vram_mb(model, &quant))
                    {
                        if vram > budget {
                            return None;
                        }
                    }
                    Some(vectorprime_core::RuntimeConfig {
                        runtime: base.runtime.clone(),
                        quantization: quant,
                        threads: base.threads,
                        batch_size: base.batch_size,
                        gpu_layers: base.gpu_layers,
                    })
                })
                .collect()
        }

        // Stage 3 — vary gpu_layers; keep best runtime+quant fixed.
        3 => gpu_layer_options(hw, model)
            .into_iter()
            .map(|gpu_layers| vectorprime_core::RuntimeConfig {
                runtime: base.runtime.clone(),
                quantization: base.quantization.clone(),
                threads: base.threads,
                batch_size: base.batch_size,
                gpu_layers,
            })
            .collect(),

        // Stage 4 — vary threads; keep best runtime+quant+gpu_layers fixed.
        4 => thread_options(hw)
            .into_iter()
            .map(|threads| vectorprime_core::RuntimeConfig {
                runtime: base.runtime.clone(),
                quantization: base.quantization.clone(),
                threads,
                batch_size: base.batch_size,
                gpu_layers: base.gpu_layers,
            })
            .collect(),

        // Stage 5 — vary batch_size; keep all prior best values fixed.
        // Try low / mid / high: 128, 256, 512.
        5 => {
            // Always include the KV-cache-aware default so Stage 5 can confirm
            // or override it.
            let kv_default = batch_size_for(hw, model);
            let mut sizes: std::collections::BTreeSet<u32> =
                [128u32, 256, 512].iter().cloned().collect();
            sizes.insert(kv_default);

            sizes
                .into_iter()
                .map(|batch_size| vectorprime_core::RuntimeConfig {
                    runtime: base.runtime.clone(),
                    quantization: base.quantization.clone(),
                    threads: base.threads,
                    batch_size,
                    gpu_layers: base.gpu_layers,
                })
                .collect()
        }

        _ => vec![],
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use vectorprime_core::{
        CpuInfo, GpuInfo, GpuVendor, HardwareProfile, ModelFormat, ModelInfo, RamInfo, SimdLevel,
    };

    fn cpu_only_hw(cores: u32) -> HardwareProfile {
        HardwareProfile {
            cpu: CpuInfo {
                core_count: cores,
                brand: "Test CPU".to_string(),
                simd_level: SimdLevel::AVX2,
            },
            gpu: None,
            ram: RamInfo {
                total_mb: 32768,
                available_mb: 16384,
            },
        }
    }

    fn hw_with_nvidia(cores: u32, vram_mb: u64, compute_major: u32) -> HardwareProfile {
        HardwareProfile {
            cpu: CpuInfo {
                core_count: cores,
                brand: "Test CPU".to_string(),
                simd_level: SimdLevel::AVX2,
            },
            gpu: Some(GpuInfo {
                name: "NVIDIA Test GPU".to_string(),
                vram_mb,
                compute_capability: Some((compute_major, 0)),
                vendor: GpuVendor::Nvidia,
            }),
            ram: RamInfo {
                total_mb: 65536,
                available_mb: 32768,
            },
        }
    }

    fn hw_with_amd(cores: u32, vram_mb: u64) -> HardwareProfile {
        HardwareProfile {
            cpu: CpuInfo {
                core_count: cores,
                brand: "Test CPU".to_string(),
                simd_level: SimdLevel::AVX2,
            },
            gpu: Some(GpuInfo {
                name: "AMD Radeon RX 7900 XTX".to_string(),
                vram_mb,
                compute_capability: None,
                vendor: GpuVendor::Amd,
            }),
            ram: RamInfo {
                total_mb: 65536,
                available_mb: 32768,
            },
        }
    }

    fn gguf_model() -> ModelInfo {
        ModelInfo {
            path: PathBuf::from("/tmp/model.gguf"),
            format: ModelFormat::GGUF,
            param_count: Some(7_000_000_000),
            hidden_size: None,
            attention_head_count: None,
            attention_head_count_kv: None,
            feed_forward_length: None,
            kv_cache_size_mb: None,
            memory_footprint_mb: None,
            flops_per_token: None,
        }
    }

    fn onnx_model() -> ModelInfo {
        ModelInfo {
            path: PathBuf::from("/tmp/model.onnx"),
            format: ModelFormat::ONNX,
            param_count: Some(1_000_000_000),
            hidden_size: None,
            attention_head_count: None,
            attention_head_count_kv: None,
            feed_forward_length: None,
            kv_cache_size_mb: None,
            memory_footprint_mb: None,
            flops_per_token: None,
        }
    }

    #[test]
    fn test_gguf_candidates_only_llamacpp_and_ollama() {
        // GGUF format should produce candidates for LlamaCpp and Ollama only —
        // not OnnxRuntime, TensorRT, or Vllm.
        let hw = cpu_only_hw(8);
        let model = gguf_model();
        let candidates = generate_candidates(&hw, &model);
        assert!(!candidates.is_empty());
        for c in &candidates {
            assert!(
                c.runtime == RuntimeKind::LlamaCpp || c.runtime == RuntimeKind::Ollama,
                "unexpected runtime {:?} in GGUF candidates (expected LlamaCpp or Ollama)",
                c.runtime
            );
        }
    }

    #[test]
    fn test_onnx_no_gpu_excludes_tensorrt() {
        let hw = cpu_only_hw(8);
        let model = onnx_model();
        let candidates = generate_candidates(&hw, &model);
        assert!(!candidates.is_empty());
        for c in &candidates {
            assert_ne!(
                c.runtime,
                RuntimeKind::TensorRT,
                "TensorRT should not appear without GPU"
            );
        }
    }

    #[test]
    fn test_onnx_with_nvidia_includes_tensorrt() {
        let hw = hw_with_nvidia(8, 24576, 8); // compute cap 8.x
        let model = onnx_model();
        let candidates = generate_candidates(&hw, &model);
        assert!(
            candidates
                .iter()
                .any(|c| c.runtime == RuntimeKind::TensorRT),
            "expected TensorRT candidates with NVIDIA GPU"
        );
    }

    /// Previously this test expected TRT to be EXCLUDED for compute cap < 7.
    /// After the refactor, TRT inclusion is based on vendor (Nvidia) alone —
    /// trtexec handles GPU-specific compatibility at runtime. Any NVIDIA GPU
    /// produces TRT candidates.
    #[test]
    fn test_onnx_with_nvidia_always_includes_tensorrt_candidates() {
        let hw = hw_with_nvidia(8, 8192, 6); // compute cap 6.x (old Pascal GPU)
        let model = onnx_model();
        let candidates = generate_candidates(&hw, &model);
        assert!(
            candidates
                .iter()
                .any(|c| c.runtime == RuntimeKind::TensorRT),
            "TRT should be a candidate for any NVIDIA GPU; trtexec handles compat"
        );
    }

    /// Non-NVIDIA GPUs (AMD, Apple, Unknown) must never produce TRT candidates.
    #[test]
    fn test_onnx_non_nvidia_excludes_tensorrt() {
        let hw = hw_with_amd(8, 24576);
        let model = onnx_model();
        let candidates = generate_candidates(&hw, &model);
        for c in &candidates {
            assert_ne!(
                c.runtime,
                RuntimeKind::TensorRT,
                "TensorRT should not appear for AMD GPU"
            );
        }
    }

    #[test]
    fn test_bytes_per_param_all_variants() {
        use QuantizationStrategy::*;
        for q in [F16, Q8_0, Q4_K_M, Q4_0, Int8, Int4] {
            assert!(
                bytes_per_param(&q) > 0.0,
                "bytes_per_param({q:?}) must be > 0"
            );
        }
    }

    #[test]
    fn test_gpu_layer_options_cpu_only() {
        let hw = cpu_only_hw(4);
        let model = gguf_model();
        let candidates = generate_candidates(&hw, &model);
        for c in &candidates {
            assert_eq!(c.gpu_layers, 0, "cpu-only: gpu_layers must be 0");
        }
    }

    #[test]
    fn test_gpu_layer_options_with_gpu() {
        let hw = hw_with_nvidia(8, 24576, 8);
        let model = gguf_model();
        let candidates = generate_candidates(&hw, &model);
        let has_nonzero = candidates.iter().any(|c| c.gpu_layers > 0);
        assert!(
            has_nonzero,
            "expected non-zero gpu_layers when GPU is present"
        );
    }

    #[test]
    fn test_estimate_max_gpu_layers_with_param_count() {
        // 7B model at F16 = ~13.4 GB; 24 GB VRAM fits ~1.79x → all 32 ref layers
        let gpu = GpuInfo {
            name: "Test GPU".to_string(),
            vram_mb: 24576,
            compute_capability: None,
            vendor: GpuVendor::Nvidia,
        };
        let model = ModelInfo {
            path: PathBuf::from("/tmp/m.gguf"),
            format: ModelFormat::GGUF,
            param_count: Some(7_000_000_000),
            hidden_size: None,
            attention_head_count: None,
            attention_head_count_kv: None,
            feed_forward_length: None,
            kv_cache_size_mb: None,
            memory_footprint_mb: None,
            flops_per_token: None,
        };
        let max = estimate_max_gpu_layers(&gpu, &model);
        assert!(max > 0, "should produce non-zero layer estimate");
        assert!(max <= 32, "reference cap is 32 layers");
    }

    #[test]
    fn test_estimate_max_gpu_layers_fallback_tiers() {
        let make_gpu = |vram_mb: u64| GpuInfo {
            name: "Test GPU".to_string(),
            vram_mb,
            compute_capability: None,
            vendor: GpuVendor::Nvidia,
        };
        let model = ModelInfo {
            path: PathBuf::from("/tmp/m.onnx"),
            format: ModelFormat::ONNX,
            param_count: None, // unknown — triggers tier fallback
            hidden_size: None,
            attention_head_count: None,
            attention_head_count_kv: None,
            feed_forward_length: None,
            kv_cache_size_mb: None,
            memory_footprint_mb: None,
            flops_per_token: None,
        };

        assert_eq!(estimate_max_gpu_layers(&make_gpu(2048), &model), 16);
        assert_eq!(estimate_max_gpu_layers(&make_gpu(8192), &model), 32);
        assert_eq!(estimate_max_gpu_layers(&make_gpu(24576), &model), 48);
    }

    #[test]
    fn test_gpu_layer_options_proportional_not_hardcoded() {
        // With a 7B model on 8 GB VRAM (F16 ~13.4 GB), partial offload expected.
        let hw = hw_with_nvidia(8, 8192, 8);
        let model = gguf_model();
        let candidates = generate_candidates(&hw, &model);
        let layer_values: std::collections::BTreeSet<u32> =
            candidates.iter().map(|c| c.gpu_layers).collect();
        // Should NOT be exactly the old hardcoded set [0, 10, 20, 33]
        assert_ne!(
            layer_values,
            [0u32, 10, 20, 33]
                .iter()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>(),
            "gpu_layers must be VRAM-proportional, not the old hardcoded set"
        );
    }
}
