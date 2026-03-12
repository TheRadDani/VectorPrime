// Location: crates/vectorprime-hardware/src/gpu/amd.rs
//
// Implements GpuProbe for AMD GPUs. Tries rocm-smi first (ROCm 5.x stack),
// then falls back to amd-smi (ROCm 6.x / newer AMD driver). Returns None
// gracefully when neither tool is available or output cannot be parsed.
// Called by gpu::probe_all() in mod.rs.

use std::process::Command;
use vectorprime_core::{GpuInfo, GpuProbe, GpuVendor};

/// Probe implementation for AMD GPUs via `rocm-smi` or `amd-smi`.
pub struct AmdProbe;

impl GpuProbe for AmdProbe {
    fn probe(&self) -> Option<GpuInfo> {
        probe_via_rocm_smi().or_else(probe_via_amd_smi)
    }
}

/// Attempt GPU detection using `rocm-smi`.
///
/// Runs: `rocm-smi --showproductname --showmeminfo vram --csv`
/// Returns `None` if the binary is missing, exits non-zero, or output is
/// unparseable.
fn probe_via_rocm_smi() -> Option<GpuInfo> {
    let output = Command::new("rocm-smi")
        .args(["--showproductname", "--showmeminfo", "vram", "--csv"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_rocm_smi_output(&stdout)
}

/// Attempt GPU detection using `amd-smi` (newer AMD tool, ROCm 6.x+).
///
/// Runs: `amd-smi monitor --gpu --csv`
/// Returns `None` if the binary is missing, exits non-zero, or output is
/// unparseable.
fn probe_via_amd_smi() -> Option<GpuInfo> {
    let output = Command::new("amd-smi")
        .args(["monitor", "--gpu", "--csv"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_amd_smi_output(&stdout)
}

/// Parse `rocm-smi --showproductname --showmeminfo vram --csv` output.
///
/// The CSV format includes a header row; we look for lines that contain
/// product name and VRAM information. Returns `None` if the expected data
/// cannot be found.
///
/// Example output (format may vary by ROCm version):
/// ```text
/// GPU[0]          : Card series:         Radeon RX 6800 XT
/// GPU[0]          : Card model:          0x73bf
/// GPU[0]          : VRAM Total Memory (B): 17163091968
/// ```
/// In CSV mode, rocm-smi emits lines like:
/// ```text
/// device,Card series,...
/// card0,Radeon RX 6800 XT,...
/// ```
pub fn parse_rocm_smi_output(output: &str) -> Option<GpuInfo> {
    // rocm-smi CSV output has a header followed by data rows.
    // We look for a non-header, non-empty line and extract name and VRAM.
    let mut lines = output.lines().filter(|l| !l.trim().is_empty());

    // First non-empty line is the header — find the column indices.
    let header = lines.next()?;
    let headers: Vec<&str> = header.split(',').map(str::trim).collect();

    // Locate relevant column indices (case-insensitive to handle version drift).
    let name_idx = headers
        .iter()
        .position(|h| h.to_lowercase().contains("card") || h.to_lowercase().contains("name"))?;
    let vram_idx = headers.iter().position(|h| {
        let lower = h.to_lowercase();
        lower.contains("vram") || lower.contains("memory")
    })?;

    // Take the first data row.
    let data_line = lines.next()?;
    let cols: Vec<&str> = data_line.split(',').map(str::trim).collect();

    let name = cols.get(name_idx)?.trim().to_string();
    if name.is_empty() {
        return None;
    }

    // VRAM may be reported in bytes (rocm-smi) — convert to MB.
    let vram_raw: u64 = cols.get(vram_idx)?.trim().parse().ok()?;
    // Values > 1 GB heuristic: treat as bytes and convert, else already MB.
    let vram_mb = if vram_raw > 1_000_000 {
        vram_raw / 1_048_576
    } else {
        vram_raw
    };

    Some(GpuInfo {
        name,
        vram_mb,
        compute_capability: None, // AMD uses GFX/RDNA versioning, not CUDA caps
        vendor: GpuVendor::Amd,
    })
}

/// Parse `amd-smi monitor --gpu --csv` output.
///
/// `amd-smi` (ROCm 6.x+) emits a CSV with columns like:
/// ```text
/// GPU,NAME,MEM_USED,MEM_TOTAL,...
/// 0,Radeon RX 7900 XTX,512,24576,...
/// ```
/// Returns `None` if expected columns are absent or data is malformed.
pub fn parse_amd_smi_output(output: &str) -> Option<GpuInfo> {
    let mut lines = output.lines().filter(|l| !l.trim().is_empty());

    let header = lines.next()?;
    let headers: Vec<&str> = header.split(',').map(str::trim).collect();

    let name_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("NAME") || h.to_lowercase().contains("name"))?;
    let vram_idx = headers.iter().position(|h| {
        let lower = h.to_lowercase();
        lower.contains("mem_total") || lower.contains("memory_total") || lower.contains("vram")
    })?;

    let data_line = lines.next()?;
    let cols: Vec<&str> = data_line.split(',').map(str::trim).collect();

    let name = cols.get(name_idx)?.trim().to_string();
    if name.is_empty() {
        return None;
    }

    let vram_raw: u64 = cols.get(vram_idx)?.trim().parse().ok()?;
    let vram_mb = if vram_raw > 1_000_000 {
        vram_raw / 1_048_576
    } else {
        vram_raw
    };

    Some(GpuInfo {
        name,
        vram_mb,
        compute_capability: None,
        vendor: GpuVendor::Amd,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rocm_smi_csv_basic() {
        // Simulate a minimal rocm-smi CSV output.
        let output =
            "device,Card series,VRAM Total Memory (B)\ncard0,Radeon RX 6800 XT,17163091968\n";
        let info = parse_rocm_smi_output(output).expect("should parse");
        assert_eq!(info.name, "Radeon RX 6800 XT");
        assert_eq!(info.vram_mb, 16368); // 17163091968 / 1048576 = 16368
        assert_eq!(info.vendor, GpuVendor::Amd);
        assert!(info.compute_capability.is_none());
    }

    #[test]
    fn test_parse_amd_smi_csv_basic() {
        let output = "GPU,NAME,MEM_TOTAL\n0,Radeon RX 7900 XTX,24576\n";
        let info = parse_amd_smi_output(output).expect("should parse");
        assert_eq!(info.name, "Radeon RX 7900 XTX");
        assert_eq!(info.vram_mb, 24576);
        assert_eq!(info.vendor, GpuVendor::Amd);
        assert!(info.compute_capability.is_none());
    }

    #[test]
    fn test_parse_rocm_smi_malformed_returns_none() {
        assert!(parse_rocm_smi_output("").is_none());
        assert!(parse_rocm_smi_output("no,columns\n").is_none());
    }

    #[test]
    fn test_parse_amd_smi_malformed_returns_none() {
        assert!(parse_amd_smi_output("").is_none());
        assert!(parse_amd_smi_output("only_header\n").is_none());
    }
}
