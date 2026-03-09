pub mod nvidia;

use llmforge_core::{GpuInfo, GpuProbe};

use nvidia::NvidiaProbe;

/// Try every known GPU probe in priority order and return the first hit.
/// Returns `None` when no supported GPU is detected.
pub fn probe_all() -> Option<GpuInfo> {
    let probes: &[&dyn GpuProbe] = &[
        &NvidiaProbe,
        // AMD and Apple Metal stubs — return None until implemented.
    ];

    probes.iter().find_map(|p| p.probe())
}
