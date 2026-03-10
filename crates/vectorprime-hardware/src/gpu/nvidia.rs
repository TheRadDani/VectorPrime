// Location: crates/vectorprime-hardware/src/gpu/nvidia.rs
//
// Implements the GpuProbe for NVIDIA GPUs by shelling out to `nvidia-smi`.
// Called by gpu::probe_all() in mod.rs; the GpuInfo it returns flows into
// HardwareProfile and from there into the optimizer candidate generator.

use vectorprime_core::{GpuInfo, GpuProbe, GpuVendor};
use std::process::Command;

/// Probe implementation for NVIDIA GPUs via `nvidia-smi`.
pub struct NvidiaProbe;

impl GpuProbe for NvidiaProbe {
    fn probe(&self) -> Option<GpuInfo> {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        // Take only the first GPU line.
        let line = stdout.lines().next()?.trim();
        parse_nvidia_csv(line)
    }
}

/// Parse one line of `nvidia-smi --format=csv,noheader,nounits` output.
///
/// Expected format: `<name>, <vram_mb>, <major>.<minor>`
/// Example:         `RTX 4090, 24564, 8.9`
///
/// Returns `None` on malformed input rather than panicking.
pub fn parse_nvidia_csv(line: &str) -> Option<GpuInfo> {
    // Use splitn(3) so GPU names with commas are not accidentally split.
    let parts: Vec<&str> = line.splitn(3, ',').collect();
    if parts.len() < 3 {
        return None;
    }

    let name = parts[0].trim().to_string();
    let vram_mb: u64 = parts[1].trim().parse().ok()?;
    let cap_str = parts[2].trim();

    let compute_capability = parse_compute_cap(cap_str);

    Some(GpuInfo {
        name,
        vram_mb,
        compute_capability,
        vendor: GpuVendor::Nvidia,
    })
}

fn parse_compute_cap(s: &str) -> Option<(u32, u32)> {
    let mut parts = s.splitn(2, '.');
    let major: u32 = parts.next()?.trim().parse().ok()?;
    let minor: u32 = parts.next()?.trim().parse().ok()?;
    Some((major, minor))
}
