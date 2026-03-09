use llmforge_core::{GpuInfo, GpuProbe};
use std::process::Command;

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
pub fn parse_nvidia_csv(line: &str) -> Option<GpuInfo> {
    // Use splitn(3) so GPU names with commas aren't accidentally split.
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
    })
}

/// Return `true` when the compute capability indicates Tensor Core support
/// (SM 7.0 / Volta and later).
#[allow(dead_code)]
pub fn has_tensor_cores(cap: Option<(u32, u32)>) -> bool {
    cap.map_or(false, |(major, _)| major >= 7)
}

fn parse_compute_cap(s: &str) -> Option<(u32, u32)> {
    let mut parts = s.splitn(2, '.');
    let major: u32 = parts.next()?.trim().parse().ok()?;
    let minor: u32 = parts.next()?.trim().parse().ok()?;
    Some((major, minor))
}
