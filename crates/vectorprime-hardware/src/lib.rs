mod cpu;
mod gpu;
mod mem;

use vectorprime_core::HardwareProfile;

/// Collect a full hardware snapshot of the current machine.
///
/// Never panics: missing GPU simply yields `None` for the `gpu` field.
pub fn profile() -> HardwareProfile {
    HardwareProfile {
        cpu: cpu::detect(),
        gpu: gpu::probe_all(),
        ram: mem::detect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_has_cores() {
        assert!(profile().cpu.core_count >= 1);
    }

    #[test]
    fn test_ram_total_positive() {
        assert!(profile().ram.total_mb > 0);
    }

    #[test]
    fn test_nvidia_parse() {
        use crate::gpu::nvidia::parse_nvidia_csv;
        use vectorprime_core::GpuVendor;

        let info = parse_nvidia_csv("RTX 4090, 24564, 8.9").unwrap();
        assert_eq!(info.name, "RTX 4090");
        assert_eq!(info.vram_mb, 24564);
        assert_eq!(info.compute_capability, Some((8, 9)));
        assert_eq!(info.vendor, GpuVendor::Nvidia);

        // Older GPU — vendor is still reported as Nvidia regardless of
        // compute capability; the optimizer no longer gates on capability.
        let old = parse_nvidia_csv("GTX 1080, 8192, 6.1").unwrap();
        assert_eq!(old.vendor, GpuVendor::Nvidia);
        assert_eq!(old.compute_capability, Some((6, 1)));

        // Malformed input returns None gracefully
        assert!(parse_nvidia_csv("bad input").is_none());
    }

    #[test]
    fn test_profile_serializes() {
        let p = profile();
        let json = serde_json::to_string(&p).expect("HardwareProfile must serialize");
        assert!(!json.is_empty());
        // Spot-check key fields exist in the JSON output.
        assert!(json.contains("core_count"));
        assert!(json.contains("total_mb"));
    }
}
