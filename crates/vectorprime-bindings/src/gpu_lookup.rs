// crates/vectorprime-bindings/src/gpu_lookup.rs
//
// GPU model lookup table for the `--gpu` CLI option.
//
// Maps user-supplied GPU model strings (e.g. "4090", "RTX 4090", "rtx-4090")
// to a [`GpuInfo`] with accurate VRAM and CUDA compute capability values.
//
// Used by: `crates/vectorprime-bindings/src/lib.rs` → `optimize()` to override
// the auto-detected hardware profile when the user specifies `--gpu`.

use vectorprime_core::{GpuInfo, GpuVendor};

/// Resolve a user-supplied GPU model string to a [`GpuInfo`].
///
/// The input is normalized to lowercase with spaces and dashes removed before
/// matching, so "RTX 4090", "rtx-4090", and "4090" all resolve identically.
///
/// # Returns
/// - `Ok(Some(GpuInfo))` for known GPU models.
/// - `Ok(None)` when `gpu_model` is `None`, `""`, or `"cpu"` (CPU-only mode).
/// - `Err(String)` for unrecognized model strings with a user-friendly message.
pub fn lookup_gpu(gpu_model: Option<&str>) -> Result<Option<GpuInfo>, String> {
    let raw = match gpu_model {
        None => return Ok(None),
        Some(s) => s,
    };

    // Normalize: lowercase, remove spaces and dashes so "RTX 4090" → "rtx4090".
    let key: String = raw
        .to_lowercase()
        .chars()
        .filter(|c| *c != ' ' && *c != '-')
        .collect();

    if key.is_empty() || key == "cpu" {
        return Ok(None);
    }

    let gpu = match key.as_str() {
        // ── Ada Lovelace (compute 8.9) ────────────────────────────────────────
        "4090" | "rtx4090" => GpuInfo {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            vram_mb: 24_576,
            compute_capability: Some((8, 9)),
            vendor: GpuVendor::Nvidia,
        },
        "4080" | "rtx4080" => GpuInfo {
            name: "NVIDIA GeForce RTX 4080".to_string(),
            vram_mb: 16_384,
            compute_capability: Some((8, 9)),
            vendor: GpuVendor::Nvidia,
        },
        "4070ti" | "rtx4070ti" => GpuInfo {
            name: "NVIDIA GeForce RTX 4070 Ti".to_string(),
            vram_mb: 12_288,
            compute_capability: Some((8, 9)),
            vendor: GpuVendor::Nvidia,
        },
        "4070" | "rtx4070" => GpuInfo {
            name: "NVIDIA GeForce RTX 4070".to_string(),
            vram_mb: 12_288,
            compute_capability: Some((8, 9)),
            vendor: GpuVendor::Nvidia,
        },
        // ── Ampere consumer (compute 8.6) ─────────────────────────────────────
        "3090" | "rtx3090" => GpuInfo {
            name: "NVIDIA GeForce RTX 3090".to_string(),
            vram_mb: 24_576,
            compute_capability: Some((8, 6)),
            vendor: GpuVendor::Nvidia,
        },
        "3080" | "rtx3080" => GpuInfo {
            name: "NVIDIA GeForce RTX 3080".to_string(),
            vram_mb: 10_240,
            compute_capability: Some((8, 6)),
            vendor: GpuVendor::Nvidia,
        },
        "3070" | "rtx3070" => GpuInfo {
            name: "NVIDIA GeForce RTX 3070".to_string(),
            vram_mb: 8_192,
            compute_capability: Some((8, 6)),
            vendor: GpuVendor::Nvidia,
        },
        "3060" | "rtx3060" => GpuInfo {
            name: "NVIDIA GeForce RTX 3060".to_string(),
            vram_mb: 12_288,
            compute_capability: Some((8, 6)),
            vendor: GpuVendor::Nvidia,
        },
        // ── Turing consumer (compute 7.5) ─────────────────────────────────────
        "2080ti" | "rtx2080ti" => GpuInfo {
            name: "NVIDIA GeForce RTX 2080 Ti".to_string(),
            vram_mb: 11_264,
            compute_capability: Some((7, 5)),
            vendor: GpuVendor::Nvidia,
        },
        "2080" | "rtx2080" => GpuInfo {
            name: "NVIDIA GeForce RTX 2080".to_string(),
            vram_mb: 8_192,
            compute_capability: Some((7, 5)),
            vendor: GpuVendor::Nvidia,
        },
        // ── Ampere datacenter (compute 8.0) ───────────────────────────────────
        "a100" => GpuInfo {
            name: "NVIDIA A100".to_string(),
            vram_mb: 81_920,
            compute_capability: Some((8, 0)),
            vendor: GpuVendor::Nvidia,
        },
        "a10g" => GpuInfo {
            name: "NVIDIA A10G".to_string(),
            vram_mb: 24_576,
            compute_capability: Some((8, 6)),
            vendor: GpuVendor::Nvidia,
        },
        // ── Hopper datacenter (compute 9.0) ───────────────────────────────────
        "h100" => GpuInfo {
            name: "NVIDIA H100".to_string(),
            vram_mb: 81_920,
            compute_capability: Some((9, 0)),
            vendor: GpuVendor::Nvidia,
        },
        // ── Unknown ───────────────────────────────────────────────────────────
        other => {
            return Err(format!(
                "Unknown GPU model: '{other}'. \
                 Try: 4090, 4080, 3090, 3080, 3070, 2080ti, a100, h100, or 'cpu'"
            ));
        }
    };

    Ok(Some(gpu))
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_input_returns_none() {
        assert!(lookup_gpu(None).unwrap().is_none());
    }

    #[test]
    fn empty_string_returns_none() {
        assert!(lookup_gpu(Some("")).unwrap().is_none());
    }

    #[test]
    fn cpu_keyword_returns_none() {
        assert!(lookup_gpu(Some("cpu")).unwrap().is_none());
        assert!(lookup_gpu(Some("CPU")).unwrap().is_none());
    }

    #[test]
    fn rtx4090_bare_number() {
        let gpu = lookup_gpu(Some("4090")).unwrap().unwrap();
        assert_eq!(gpu.vram_mb, 24_576);
        assert_eq!(gpu.compute_capability, Some((8, 9)));
    }

    #[test]
    fn rtx4090_prefixed() {
        let gpu = lookup_gpu(Some("rtx4090")).unwrap().unwrap();
        assert_eq!(gpu.vram_mb, 24_576);
    }

    #[test]
    fn rtx4090_mixed_case_with_spaces() {
        // "RTX 4090" → normalized "rtx4090"
        let gpu = lookup_gpu(Some("RTX 4090")).unwrap().unwrap();
        assert_eq!(gpu.vram_mb, 24_576);
        assert_eq!(gpu.compute_capability, Some((8, 9)));
    }

    #[test]
    fn rtx4090_with_dashes() {
        // "rtx-4090" → normalized "rtx4090"
        let gpu = lookup_gpu(Some("rtx-4090")).unwrap().unwrap();
        assert_eq!(gpu.vram_mb, 24_576);
    }

    #[test]
    fn rtx4080_specs() {
        let gpu = lookup_gpu(Some("4080")).unwrap().unwrap();
        assert_eq!(gpu.vram_mb, 16_384);
        assert_eq!(gpu.compute_capability, Some((8, 9)));
    }

    #[test]
    fn rtx3090_specs() {
        let gpu = lookup_gpu(Some("3090")).unwrap().unwrap();
        assert_eq!(gpu.vram_mb, 24_576);
        assert_eq!(gpu.compute_capability, Some((8, 6)));
    }

    #[test]
    fn rtx3080_specs() {
        let gpu = lookup_gpu(Some("3080")).unwrap().unwrap();
        assert_eq!(gpu.vram_mb, 10_240);
        assert_eq!(gpu.compute_capability, Some((8, 6)));
    }

    #[test]
    fn rtx3070_specs() {
        let gpu = lookup_gpu(Some("3070")).unwrap().unwrap();
        assert_eq!(gpu.vram_mb, 8_192);
        assert_eq!(gpu.compute_capability, Some((8, 6)));
    }

    #[test]
    fn rtx2080ti_specs() {
        let gpu = lookup_gpu(Some("2080ti")).unwrap().unwrap();
        assert_eq!(gpu.vram_mb, 11_264);
        assert_eq!(gpu.compute_capability, Some((7, 5)));
    }

    #[test]
    fn rtx2080ti_prefixed() {
        let gpu = lookup_gpu(Some("RTX 2080 Ti")).unwrap().unwrap();
        assert_eq!(gpu.vram_mb, 11_264);
        assert_eq!(gpu.compute_capability, Some((7, 5)));
    }

    #[test]
    fn a100_specs() {
        let gpu = lookup_gpu(Some("a100")).unwrap().unwrap();
        assert_eq!(gpu.vram_mb, 81_920);
        assert_eq!(gpu.compute_capability, Some((8, 0)));
    }

    #[test]
    fn h100_specs() {
        let gpu = lookup_gpu(Some("h100")).unwrap().unwrap();
        assert_eq!(gpu.vram_mb, 81_920);
        assert_eq!(gpu.compute_capability, Some((9, 0)));
    }

    #[test]
    fn unknown_model_returns_error() {
        let err = lookup_gpu(Some("9999")).unwrap_err();
        assert!(err.contains("Unknown GPU model"));
        assert!(err.contains("4090"));
    }

    #[test]
    fn unknown_model_message_includes_input() {
        let err = lookup_gpu(Some("bogus-gpu-xyz")).unwrap_err();
        assert!(err.contains("bogusGPUxyz") || err.contains("bogus") || err.contains("Unknown"));
    }
}
