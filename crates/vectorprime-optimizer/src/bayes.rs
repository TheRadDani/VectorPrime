// crates/vectorprime-optimizer/src/bayes.rs
//
// Bayesian optimization layer for the VectorPrime 4-stage pipeline.
//
// This module provides:
//   - `SearchSpace`  : encodes the 5-dimensional parameter space
//                      (runtime_idx, quant_idx, gpu_layers, threads, batch)
//                      and translates between normalized [0,1]^5 coordinates
//                      and concrete `RuntimeConfig` values.
//   - `ConfigPoint`  : a point in that normalized space.
//   - `TpeModel`     : Tree-structured Parzen Estimator (same algorithm as
//                      Optuna/Hyperopt). Default acquisition strategy.
//   - `GpModel`      : Gaussian Process with RBF kernel + Expected-Improvement
//                      acquisition. Alternative strategy; more accurate for
//                      smooth landscapes but capped at ~20 observations.
//
// Used by: `lib.rs` (`run_optimization_bayesian`).
//
// No external ML crates — TPE and GP are implemented from scratch.
// The only optional dependency is `rand` for seeded random candidate generation.

use vectorprime_core::{HardwareProfile, QuantizationStrategy, RuntimeConfig, RuntimeKind};

// ──────────────────────────────────────────────────────────────────────────────
// Search space encoding / decoding
// ──────────────────────────────────────────────────────────────────────────────

/// The 5-dimensional discrete/mixed search space for Bayesian optimization.
///
/// Dimensions (all normalized to [0, 1] internally):
/// - 0: runtime index  — which preselected runtime to use
/// - 1: quant index    — which quantization strategy to use
/// - 2: gpu_layers     — continuous in [0, max_gpu_layers]
/// - 3: thread index   — index into `thread_options`
/// - 4: batch index    — index into `batch_options`
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Runtimes preselected by Stage 3.
    pub runtimes: Vec<RuntimeKind>,
    /// Quantizations viable for this model/hardware.
    pub quants: Vec<QuantizationStrategy>,
    /// Maximum gpu_layers value (0 when no GPU).
    pub max_gpu_layers: u32,
    /// Thread count options (e.g. [4, 8, 16]).
    pub thread_options: Vec<u32>,
    /// Batch size options (e.g. [128, 256, 512]).
    pub batch_options: Vec<u32>,
}

/// A point in the 5-dimensional normalized space [0, 1]^5.
#[derive(Clone, Debug)]
pub struct ConfigPoint(pub [f64; 5]);

impl SearchSpace {
    /// Decode a normalized point into a concrete `RuntimeConfig`.
    ///
    /// Clamps every index so out-of-range floats never panic.
    pub fn decode(&self, _hw: &HardwareProfile, point: &ConfigPoint) -> RuntimeConfig {
        let rt_idx = clamp_idx(point.0[0], self.runtimes.len());
        let q_idx = clamp_idx(point.0[1], self.quants.len());
        let gpu = if self.max_gpu_layers == 0 {
            0
        } else {
            (point.0[2] * self.max_gpu_layers as f64).round() as u32
        };
        let t_idx = clamp_idx(point.0[3], self.thread_options.len());
        let b_idx = clamp_idx(point.0[4], self.batch_options.len());

        RuntimeConfig {
            runtime: self.runtimes[rt_idx].clone(),
            quantization: self.quants[q_idx].clone(),
            gpu_layers: gpu.min(self.max_gpu_layers),
            threads: self.thread_options[t_idx],
            batch_size: self.batch_options[b_idx],
        }
    }

    /// Encode a `RuntimeConfig` back into normalized space.
    ///
    /// Used to seed the model from staged results. Returns the midpoint when a
    /// value cannot be found in the option lists.
    pub fn encode(&self, cfg: &RuntimeConfig) -> ConfigPoint {
        let rt_val = find_normalized_idx(&self.runtimes, &cfg.runtime);
        let q_val = find_normalized_idx_by(
            self.quants.len(),
            |i| self.quants[i] == cfg.quantization,
        );
        let gpu_val = if self.max_gpu_layers == 0 {
            0.0
        } else {
            (cfg.gpu_layers as f64 / self.max_gpu_layers as f64).clamp(0.0, 1.0)
        };
        let t_val = find_normalized_idx_by(
            self.thread_options.len(),
            |i| self.thread_options[i] == cfg.threads,
        );
        let b_val = find_normalized_idx_by(
            self.batch_options.len(),
            |i| self.batch_options[i] == cfg.batch_size,
        );

        ConfigPoint([rt_val, q_val, gpu_val, t_val, b_val])
    }

    /// Generate `n` quasi-random initial points using the Halton sequence.
    ///
    /// Uses primes [2, 3, 5, 7, 11] for dimensions 0–4.
    /// The Halton sequence has much lower discrepancy than uniform random
    /// for small sample counts, which leads to better initial coverage.
    pub fn halton_samples(&self, n: usize) -> Vec<ConfigPoint> {
        // Start at index 1 (index 0 maps to all-zeros, a degenerate point).
        (1..=n).map(|i| halton_point(i)).collect()
    }
}

/// Clamp a normalized float to a valid index in a slice of `len` elements.
fn clamp_idx(val: f64, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let idx = (val * len as f64).floor() as usize;
    idx.min(len - 1)
}

/// Find the normalized position of `target` in `slice`, returning 0.5 when absent.
fn find_normalized_idx<T: PartialEq>(slice: &[T], target: &T) -> f64 {
    find_normalized_idx_by(slice.len(), |i| &slice[i] == target)
}

/// Find the normalized position using a predicate; returns 0.5 when absent.
fn find_normalized_idx_by(len: usize, pred: impl Fn(usize) -> bool) -> f64 {
    if len == 0 {
        return 0.0;
    }
    for i in 0..len {
        if pred(i) {
            // Map index → midpoint of its bucket in [0,1].
            return (i as f64 + 0.5) / len as f64;
        }
    }
    0.5 // not found — return midpoint
}

// ──────────────────────────────────────────────────────────────────────────────
// Halton low-discrepancy sequence
// ──────────────────────────────────────────────────────────────────────────────

/// One value from the Halton sequence in base `base` at position `index`.
fn halton(index: usize, base: usize) -> f64 {
    let mut f = 1.0f64;
    let mut r = 0.0f64;
    let mut i = index;
    while i > 0 {
        f /= base as f64;
        r += f * (i % base) as f64;
        i /= base;
    }
    r
}

/// A 5-dimensional Halton point using primes [2, 3, 5, 7, 11].
fn halton_point(index: usize) -> ConfigPoint {
    ConfigPoint([
        halton(index, 2),
        halton(index, 3),
        halton(index, 5),
        halton(index, 7),
        halton(index, 11),
    ])
}

// ──────────────────────────────────────────────────────────────────────────────
// Pseudo-random point generation (seeded, no external dependency)
// ──────────────────────────────────────────────────────────────────────────────

/// Generate a deterministic pseudo-random point from a u64 seed using
/// a simple LCG (linear congruential generator). No `rand` crate needed.
///
/// Each dimension uses a different multiplier to break correlation.
fn random_point(seed: u64) -> ConfigPoint {
    // LCG constants from Knuth's MMIX.
    const A: u64 = 6364136223846793005;
    const C: u64 = 1442695040888963407;

    let mut s = seed;
    let mut next = || -> f64 {
        s = s.wrapping_mul(A).wrapping_add(C);
        // Extract 52 bits and normalize to [0,1).
        let bits = (s >> 12) as f64;
        bits / (1u64 << 52) as f64
    };

    ConfigPoint([next(), next(), next(), next(), next()])
}

// ──────────────────────────────────────────────────────────────────────────────
// KDE helpers (shared by TpeModel)
// ──────────────────────────────────────────────────────────────────────────────

/// Standard deviation of dimension `dim` across a slice of observation references.
///
/// Returns 0.2 when fewer than 2 observations are present (prevents
/// division-by-zero in bandwidth calculation).
fn std_dev_dim(obs: &[&ConfigPoint], dim: usize) -> f64 {
    let n = obs.len() as f64;
    if n < 2.0 {
        return 0.2;
    }
    let mean = obs.iter().map(|p| p.0[dim]).sum::<f64>() / n;
    let var = obs.iter().map(|p| (p.0[dim] - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

/// Gaussian KDE score for `point` against a set of observation references.
///
/// Uses a product-of-Gaussians kernel with per-dimension bandwidth computed
/// by Silverman's rule: h = 1.06 × σ × n^(−1/5).
///
/// Returns a small positive floor (1e-10) when no observations are provided so
/// ratios computed by the TPE acquisition function remain finite.
fn kde_score(point: &ConfigPoint, observations: &[&ConfigPoint]) -> f64 {
    if observations.is_empty() {
        return 1e-10;
    }
    let n = observations.len() as f64;
    let dims = 5usize;
    let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();

    let mut total = 0.0f64;
    for obs in observations {
        // Product kernel: multiply Gaussian densities across all dimensions.
        let mut k = 1.0f64;
        for d in 0..dims {
            let std_d = std_dev_dim(observations, d).max(0.01);
            // Silverman's rule bandwidth.
            let bw = (1.06 * std_d * n.powf(-0.2)).max(1e-8);
            let z = (point.0[d] - obs.0[d]) / bw;
            k *= (-0.5 * z * z).exp() / (bw * sqrt_2pi);
        }
        total += k;
    }
    total / n
}

// ──────────────────────────────────────────────────────────────────────────────
// Tree-structured Parzen Estimator (TPE)
// ──────────────────────────────────────────────────────────────────────────────

/// Tree-structured Parzen Estimator (TPE) — the default acquisition strategy.
///
/// At each iteration the observations are split into a "good" set (top γ
/// fraction by score) and a "bad" set (the rest). New candidates are generated
/// by maximizing l(x)/g(x) where l and g are KDE densities of the two sets.
///
/// This is the same core algorithm used by Optuna and Hyperopt, adapted here
/// for a 5-dimensional mixed discrete/continuous space.
pub struct TpeModel {
    /// All observations collected so far: (normalized point, score).
    observations: Vec<(ConfigPoint, f64)>,
    /// Fraction of top observations treated as "good". Default: 0.25.
    gamma: f64,
}

impl TpeModel {
    /// Create a new TPE model with the given split fraction γ.
    ///
    /// `gamma` must be in (0, 1); values outside this range are clamped.
    pub fn new(gamma: f64) -> Self {
        TpeModel {
            observations: Vec::new(),
            gamma: gamma.clamp(0.05, 0.95),
        }
    }

    /// Record an observed (point, score) pair.
    pub fn observe(&mut self, point: ConfigPoint, score: f64) {
        self.observations.push((point, score));
    }

    /// Suggest the next point to evaluate.
    ///
    /// When fewer than 3 observations are available the model has insufficient
    /// data for reliable density estimation; it falls back to a seeded random
    /// point derived from the current observation count.
    ///
    /// Otherwise generates `n_candidates` random candidates, scores each by the
    /// l(x)/g(x) ratio, and returns the one with the highest ratio.
    pub fn suggest(&self, n_candidates: usize, rng_seed: u64) -> ConfigPoint {
        if self.observations.len() < 3 {
            return random_point(rng_seed.wrapping_add(self.observations.len() as u64));
        }

        // Sort descending by score (higher = better).
        let mut sorted = self.observations.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let n_good = ((sorted.len() as f64 * self.gamma).ceil() as usize).max(1);
        let good: Vec<&ConfigPoint> = sorted[..n_good].iter().map(|(p, _)| p).collect();
        let bad: Vec<&ConfigPoint> = sorted[n_good..].iter().map(|(p, _)| p).collect();

        let mut best_ratio = f64::NEG_INFINITY;
        let mut best_point = random_point(rng_seed);

        for i in 0..n_candidates {
            let candidate = random_point(rng_seed.wrapping_add(i as u64).wrapping_add(0xDEAD_BEEF));
            let l = kde_score(&candidate, &good);
            let g = kde_score(&candidate, &bad).max(1e-10);
            let ratio = l / g;
            if ratio > best_ratio {
                best_ratio = ratio;
                best_point = candidate;
            }
        }

        best_point
    }

    /// Return the best (config_point, score) seen so far, or `None` if empty.
    pub fn best_observation(&self) -> Option<&(ConfigPoint, f64)> {
        self.observations
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Gaussian Process with RBF kernel (alternative acquisition strategy)
// ──────────────────────────────────────────────────────────────────────────────

/// Gaussian Process regression with RBF (squared-exponential) kernel.
///
/// Uses Expected-Improvement (EI) as the acquisition function. Prediction is
/// via direct Gaussian elimination (no Cholesky, no external libraries),
/// which is accurate up to N ≈ 20–25 observations — well within our budget
/// of 12 total evaluations.
///
/// This is an alternative to `TpeModel`; both implement `observe`/`suggest`
/// with the same signature so they can be swapped without changing `lib.rs`.
pub struct GpModel {
    x_train: Vec<ConfigPoint>,
    y_train: Vec<f64>,
    /// RBF kernel length-scale (default: 0.5).
    lengthscale: f64,
    /// Observation noise variance added to the diagonal (default: 0.01).
    noise: f64,
}

impl GpModel {
    /// Create a new GP model with the given kernel hyper-parameters.
    pub fn new(lengthscale: f64, noise: f64) -> Self {
        GpModel {
            x_train: Vec::new(),
            y_train: Vec::new(),
            lengthscale: lengthscale.max(1e-6),
            noise: noise.max(1e-8),
        }
    }

    /// RBF (squared-exponential) kernel:  k(a, b) = exp(−‖a−b‖² / (2l²))
    fn kernel(&self, a: &ConfigPoint, b: &ConfigPoint) -> f64 {
        let sq_dist: f64 = a
            .0
            .iter()
            .zip(b.0.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum();
        (-sq_dist / (2.0 * self.lengthscale.powi(2))).exp()
    }

    /// Predict posterior mean and variance at a new point `x`.
    ///
    /// Uses closed-form GP equations solved by Gaussian elimination.
    /// Returns (0.0, 1.0) when no training data is available.
    fn predict(&self, x: &ConfigPoint) -> (f64, f64) {
        let n = self.x_train.len();
        if n == 0 {
            return (0.0, 1.0);
        }

        // Build K + noise·I matrix.
        let mut k_mat: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let kij = self.kernel(&self.x_train[i], &self.x_train[j]);
                        if i == j { kij + self.noise } else { kij }
                    })
                    .collect()
            })
            .collect();

        // k_star: kernel between x and each training point.
        let k_star: Vec<f64> = (0..n)
            .map(|i| self.kernel(x, &self.x_train[i]))
            .collect();

        // Solve (K + noise·I) α = y.
        let alpha = solve_linear(&mut k_mat.clone(), &self.y_train);

        // mean = k_star^T · α
        let mean: f64 = k_star.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum();

        // Variance: k(x,x) - k_star^T · (K+noiseI)^-1 · k_star
        let v = solve_linear(&mut k_mat, &k_star);
        let reduction: f64 = k_star.iter().zip(v.iter()).map(|(k, vi)| k * vi).sum();
        let var = (self.kernel(x, x) - reduction).max(1e-6);

        (mean, var)
    }

    /// Expected Improvement: EI(x) = σ·[z·Φ(z) + φ(z)]  where z = (μ − f*)/ σ
    ///
    /// Returns 0 when σ < 1e-8 (degenerate case, already observed this point).
    pub fn expected_improvement(&self, x: &ConfigPoint, best: f64) -> f64 {
        let (mu, var) = self.predict(x);
        let sigma = var.sqrt();
        if sigma < 1e-8 {
            return 0.0;
        }
        let z = (mu - best) / sigma;
        sigma * (z * normal_cdf(z) + normal_pdf(z))
    }

    /// Record an observed (point, score) pair.
    pub fn observe(&mut self, point: ConfigPoint, score: f64) {
        self.x_train.push(point);
        self.y_train.push(score);
    }

    /// Suggest the next point to evaluate using Expected Improvement.
    ///
    /// Falls back to a random point when fewer than 2 observations exist.
    pub fn suggest(&self, n_candidates: usize, rng_seed: u64) -> ConfigPoint {
        if self.x_train.len() < 2 {
            return random_point(rng_seed.wrapping_add(self.x_train.len() as u64));
        }

        let best_y = self
            .y_train
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut best_ei = f64::NEG_INFINITY;
        let mut best_point = random_point(rng_seed);

        for i in 0..n_candidates {
            let candidate = random_point(rng_seed.wrapping_add(i as u64).wrapping_add(0xCAFE_BABE));
            let ei = self.expected_improvement(&candidate, best_y);
            if ei > best_ei {
                best_ei = ei;
                best_point = candidate;
            }
        }

        best_point
    }

    /// Return the best (config_point, score) seen so far, or `None` if empty.
    pub fn best_observation(&self) -> Option<(ConfigPoint, f64)> {
        self.x_train
            .iter()
            .zip(self.y_train.iter())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(p, &s)| (p.clone(), s))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Linear solver (Gaussian elimination with partial pivoting)
// ──────────────────────────────────────────────────────────────────────────────

/// Solve Ax = b via Gaussian elimination with partial pivoting.
///
/// Operates only on small matrices (N ≤ 25). Returns the zero vector when the
/// system is singular or has zero rows.
///
/// `a` is modified in-place as part of elimination.
fn solve_linear(a: &mut Vec<Vec<f64>>, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 || a.is_empty() {
        return vec![];
    }

    // Augment [A | b].
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    // Forward elimination with partial pivoting.
    for col in 0..n {
        // Find the pivot row.
        let pivot = (col..n)
            .max_by(|&r1, &r2| {
                aug[r1][col]
                    .abs()
                    .partial_cmp(&aug[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);

        if aug[pivot][col].abs() < 1e-12 {
            // Singular or near-singular column — skip.
            continue;
        }

        aug.swap(col, pivot);

        let scale = aug[col][col];
        for j in col..=n {
            aug[col][j] /= scale;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in col..=n {
                let v = aug[col][j] * factor;
                aug[row][j] -= v;
            }
        }
    }

    // Extract solution from the augmented column.
    (0..n).map(|i| aug[i][n]).collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Standard normal helpers (Abramowitz & Stegun approximation)
// ──────────────────────────────────────────────────────────────────────────────

/// Standard normal CDF: Φ(z) = 0.5·(1 + erf(z/√2))
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Standard normal PDF: φ(z) = exp(−z²/2) / √(2π)
fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Error function approximation (Abramowitz & Stegun 7.1.26).
///
/// Maximum error: 1.5 × 10⁻⁷ over all real inputs.
fn erf(x: f64) -> f64 {
    // Handle negative inputs via odd symmetry.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // Rational approximation constants.
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));

    sign * (1.0 - poly * (-x * x).exp())
}

// ──────────────────────────────────────────────────────────────────────────────
// ModelInfo reference — for thread_options helper used in lib.rs
// ──────────────────────────────────────────────────────────────────────────────

/// Build the standard thread option set from a core count:
/// [cores/2, cores, cores*2] clamped to [1, 64], deduplicated.
pub fn thread_options_from_cores(cores: u32) -> Vec<u32> {
    let half = (cores / 2).max(1).clamp(1, 64);
    let full = cores.clamp(1, 64);
    let double = (cores * 2).clamp(1, 64);

    let mut v = vec![half, full, double];
    v.dedup();
    v.sort_unstable();
    v.dedup();
    v
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use vectorprime_core::{
        CpuInfo, HardwareProfile, RamInfo, SimdLevel,
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

    fn small_space() -> SearchSpace {
        SearchSpace {
            runtimes: vec![RuntimeKind::LlamaCpp, RuntimeKind::Ollama],
            quants: vec![
                QuantizationStrategy::Q4_K_M,
                QuantizationStrategy::Q8_0,
                QuantizationStrategy::F16,
            ],
            max_gpu_layers: 32,
            thread_options: vec![4, 8, 16],
            batch_options: vec![128, 256, 512],
        }
    }

    fn no_gpu_space() -> SearchSpace {
        SearchSpace {
            runtimes: vec![RuntimeKind::LlamaCpp],
            quants: vec![QuantizationStrategy::Q4_K_M],
            max_gpu_layers: 0,
            thread_options: vec![4, 8],
            batch_options: vec![256, 512],
        }
    }

    // ─── Test 1: Halton samples are in [0,1]^5 ────────────────────────────────

    /// All Halton samples must lie strictly within [0.0, 1.0] for every dimension.
    #[test]
    fn test_halton_in_range() {
        let space = small_space();
        let samples = space.halton_samples(20);
        assert_eq!(samples.len(), 20);

        for (i, pt) in samples.iter().enumerate() {
            for (d, &val) in pt.0.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&val),
                    "Halton sample {i} dim {d} = {val} is out of [0,1]"
                );
            }
        }
    }

    // ─── Test 2: TPE suggest returns a valid point after 5 observations ───────

    /// After 5 synthetic observations, `TpeModel::suggest` must return a point
    /// where all 5 coordinates lie in [0.0, 1.0].
    #[test]
    fn test_tpe_suggest_returns_valid_point() {
        let mut model = TpeModel::new(0.25);

        // Inject 5 observations spanning the space.
        let obs: &[([f64; 5], f64)] = &[
            ([0.1, 0.2, 0.3, 0.4, 0.5], 10.0),
            ([0.9, 0.8, 0.7, 0.6, 0.5], 50.0),
            ([0.5, 0.5, 0.5, 0.5, 0.5], 30.0),
            ([0.2, 0.7, 0.1, 0.9, 0.3], 20.0),
            ([0.8, 0.3, 0.9, 0.1, 0.7], 45.0),
        ];
        for (coords, score) in obs {
            model.observe(ConfigPoint(*coords), *score);
        }

        let suggested = model.suggest(24, 42);

        for (d, &val) in suggested.0.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&val),
                "TPE suggested dim {d} = {val} is out of [0,1]"
            );
        }
    }

    // ─── Test 3: GP variance at observed point is lower than at unobserved ────

    /// After a few observations, the GP posterior variance at an observed point
    /// should be lower than at a point far from all observations.
    #[test]
    fn test_gp_predict_uncertainty_decreases() {
        let mut gp = GpModel::new(0.5, 0.01);

        // Observe a cluster of points near [0.5, 0.5, 0.5, 0.5, 0.5].
        for i in 0..5u64 {
            let noise = i as f64 * 0.02;
            gp.observe(
                ConfigPoint([0.5 + noise, 0.5, 0.5, 0.5, 0.5]),
                40.0 + i as f64,
            );
        }

        // Point near the observed cluster → should have low variance.
        let (_, var_near) = gp.predict(&ConfigPoint([0.5, 0.5, 0.5, 0.5, 0.5]));

        // Point far from any observation → should have high variance.
        let (_, var_far) = gp.predict(&ConfigPoint([0.0, 0.0, 0.0, 0.0, 0.0]));

        assert!(
            var_near < var_far,
            "GP variance near observed region ({var_near:.6}) must be less than \
             variance far from observations ({var_far:.6})"
        );
    }

    // ─── Additional robustness tests ──────────────────────────────────────────

    /// Decode must clamp out-of-range floats without panicking.
    #[test]
    fn test_decode_clamping() {
        let space = small_space();
        let hw = cpu_only_hw(8);

        // All zeros
        let cfg = space.decode(&hw, &ConfigPoint([0.0; 5]));
        assert_eq!(cfg.runtime, RuntimeKind::LlamaCpp);
        assert_eq!(cfg.gpu_layers, 0);

        // All ones
        let cfg = space.decode(&hw, &ConfigPoint([1.0; 5]));
        assert_eq!(cfg.runtime, RuntimeKind::Ollama);
        assert!(cfg.gpu_layers <= space.max_gpu_layers);
    }

    /// When max_gpu_layers == 0, decode must always produce gpu_layers = 0.
    #[test]
    fn test_decode_no_gpu_always_zero_layers() {
        let space = no_gpu_space();
        let hw = cpu_only_hw(8);

        for &v in &[0.0, 0.5, 1.0] {
            let cfg = space.decode(&hw, &ConfigPoint([0.5, 0.5, v, 0.5, 0.5]));
            assert_eq!(
                cfg.gpu_layers, 0,
                "gpu_layers must be 0 when max_gpu_layers=0, got {}",
                cfg.gpu_layers
            );
        }
    }

    /// TPE with fewer than 3 observations must return a point in [0,1]^5.
    #[test]
    fn test_tpe_fallback_before_enough_observations() {
        let mut model = TpeModel::new(0.25);
        model.observe(ConfigPoint([0.3, 0.4, 0.5, 0.6, 0.7]), 25.0);
        model.observe(ConfigPoint([0.7, 0.6, 0.5, 0.4, 0.3]), 35.0);

        let pt = model.suggest(24, 99);
        for &v in pt.0.iter() {
            assert!((0.0..=1.0).contains(&v), "fallback point dim out of range: {v}");
        }
    }

    /// `solve_linear` must return the correct solution for a known 2×2 system.
    #[test]
    fn test_solve_linear_2x2() {
        // 2x + 3y = 8
        // x  - y  = 1  → x = 11/5 = 2.2, y = 7/5 = 1.4  (approx)
        // Actually: x=11/5=2.2, y=6/5=1.2 ? Let me verify:
        // 2(2.2) + 3y = 8 → 4.4 + 3y = 8 → y = 1.2  ✓
        // 2.2 - 1.2 = 1  ✓
        let mut a = vec![vec![2.0, 3.0], vec![1.0, -1.0]];
        let b = vec![8.0, 1.0];
        let x = solve_linear(&mut a, &b);
        assert_eq!(x.len(), 2);
        assert!((x[0] - 2.2).abs() < 1e-9, "x[0] = {}", x[0]);
        assert!((x[1] - 1.2).abs() < 1e-9, "x[1] = {}", x[1]);
    }

    /// The erf approximation must be accurate to within 1e-6 at key values.
    #[test]
    fn test_erf_approx_key_values() {
        // erf(0) = 0
        assert!(erf(0.0).abs() < 1e-9, "erf(0) = {}", erf(0.0));
        // erf(1) ≈ 0.842701
        assert!((erf(1.0) - 0.842701).abs() < 1e-5, "erf(1) = {}", erf(1.0));
        // erf(-1) = -erf(1) (odd symmetry)
        assert!(
            (erf(-1.0) + erf(1.0)).abs() < 1e-9,
            "erf is not odd-symmetric"
        );
        // erf(∞) → 1 (test with large x)
        assert!((erf(6.0) - 1.0).abs() < 1e-6, "erf(6) = {}", erf(6.0));
    }

    /// `thread_options_from_cores` must always produce sorted, deduplicated values in [1,64].
    #[test]
    fn test_thread_options_from_cores() {
        for cores in [1u32, 2, 4, 8, 16, 32, 64, 128] {
            let opts = thread_options_from_cores(cores);
            assert!(!opts.is_empty(), "no options for {cores} cores");
            for &v in &opts {
                assert!(v >= 1, "thread option {v} < 1");
                assert!(v <= 64, "thread option {v} > 64");
            }
            // Sorted ascending
            assert!(opts.windows(2).all(|w| w[0] <= w[1]), "not sorted: {opts:?}");
            // No duplicates
            for i in 1..opts.len() {
                assert_ne!(opts[i - 1], opts[i], "duplicate in {opts:?}");
            }
        }
    }

    /// `encode` followed by `decode` must reproduce the same config (identity roundtrip).
    #[test]
    fn test_encode_decode_roundtrip() {
        let space = small_space();
        let hw = cpu_only_hw(8);

        let original = RuntimeConfig {
            runtime: RuntimeKind::Ollama,
            quantization: QuantizationStrategy::Q8_0,
            gpu_layers: 16,
            threads: 8,
            batch_size: 256,
        };

        let encoded = space.encode(&original);
        let decoded = space.decode(&hw, &encoded);

        assert_eq!(decoded.runtime, original.runtime);
        assert_eq!(decoded.quantization, original.quantization);
        assert_eq!(decoded.gpu_layers, original.gpu_layers);
        assert_eq!(decoded.threads, original.threads);
        assert_eq!(decoded.batch_size, original.batch_size);
    }

    /// GP `suggest` must return a point in [0,1]^5.
    #[test]
    fn test_gp_suggest_returns_valid_point() {
        let mut gp = GpModel::new(0.5, 0.01);
        for i in 0..5u64 {
            gp.observe(ConfigPoint(random_point(i).0), 20.0 + i as f64 * 5.0);
        }

        let pt = gp.suggest(16, 7);
        for &v in pt.0.iter() {
            assert!((0.0..=1.0).contains(&v), "GP suggest dim out of range: {v}");
        }
    }
}
