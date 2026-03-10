// crates/vectorprime-hardware/src/cpu.rs
//
// CPU detection: brand string and SIMD capability.
// On x86/x86_64 hosts the raw-cpuid crate is used for direct CPUID instruction
// access.  On all other architectures we fall back to /proc/cpuinfo (Linux) or
// return safe defaults so the crate compiles cross-platform.

use vectorprime_core::{CpuInfo, SimdLevel};

// The raw-cpuid import is only available on x86/x86_64.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use raw_cpuid::CpuId;

pub fn detect() -> CpuInfo {
    CpuInfo {
        core_count: num_cpus::get() as u32,
        brand: detect_brand(),
        simd_level: detect_simd(),
    }
}

// ── brand string ─────────────────────────────────────────────────────────────

/// On x86/x86_64: query via CPUID leaf 0x80000002-4, fall back to /proc/cpuinfo.
/// On other architectures: read /proc/cpuinfo directly (Linux), or return "Unknown".
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_brand() -> String {
    let cpuid = CpuId::new();
    if let Some(brand) = cpuid.get_processor_brand_string() {
        let s = brand.as_str().trim().to_string();
        if !s.is_empty() {
            return s;
        }
    }
    parse_proc_cpuinfo_brand().unwrap_or_else(|| "Unknown".to_string())
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_brand() -> String {
    parse_proc_cpuinfo_brand().unwrap_or_else(|| "Unknown".to_string())
}

// ── SIMD level ────────────────────────────────────────────────────────────────

/// On x86/x86_64: query AVX/AVX2/AVX512 via CPUID feature flags.
/// On other architectures: AVX is an x86 concept; return None.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_simd() -> SimdLevel {
    let cpuid = CpuId::new();

    if let Some(ext) = cpuid.get_extended_feature_info() {
        if ext.has_avx512f() {
            return SimdLevel::AVX512;
        }
        if ext.has_avx2() {
            return SimdLevel::AVX2;
        }
    }
    if let Some(feat) = cpuid.get_feature_info() {
        if feat.has_avx() {
            return SimdLevel::AVX;
        }
    }
    SimdLevel::None
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_simd() -> SimdLevel {
    SimdLevel::None
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Read the "model name" field from /proc/cpuinfo (Linux only).
/// Returns None on non-Linux systems or when the field is absent.
fn parse_proc_cpuinfo_brand() -> Option<String> {
    let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in content.lines() {
        if line.starts_with("model name") {
            return Some(line.split_once(':')?.1.trim().to_string());
        }
    }
    None
}
