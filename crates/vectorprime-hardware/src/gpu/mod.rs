// Location: crates/vectorprime-hardware/src/gpu/mod.rs
//
// GPU detection aggregator. Tries each vendor probe in priority order and
// returns the first hit. Called by vectorprime_hardware::profile() to
// populate HardwareProfile.gpu.

pub mod amd;
pub mod apple;
pub mod nvidia;

use vectorprime_core::{GpuInfo, GpuProbe};

use amd::AmdProbe;
use apple::AppleProbe;
use nvidia::NvidiaProbe;

/// Try every known GPU probe in priority order and return the first hit.
///
/// Priority: NVIDIA → AMD → Apple. Returns `None` when no supported GPU
/// is detected on this machine.
pub fn probe_all() -> Option<GpuInfo> {
    let probes: &[&dyn GpuProbe] = &[
        &NvidiaProbe,
        &AmdProbe,
        &AppleProbe,
    ];

    probes.iter().find_map(|p| p.probe())
}
