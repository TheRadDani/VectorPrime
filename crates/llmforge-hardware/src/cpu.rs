use llmforge_core::{CpuInfo, SimdLevel};
use raw_cpuid::CpuId;

pub fn detect() -> CpuInfo {
    CpuInfo {
        core_count: num_cpus::get() as u32,
        brand: detect_brand(),
        simd_level: detect_simd(),
    }
}

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

fn parse_proc_cpuinfo_brand() -> Option<String> {
    let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in content.lines() {
        if line.starts_with("model name") {
            return Some(line.splitn(2, ':').nth(1)?.trim().to_string());
        }
    }
    None
}
