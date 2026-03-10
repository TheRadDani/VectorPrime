use vectorprime_core::RamInfo;

pub fn detect() -> RamInfo {
    parse_proc_meminfo().unwrap_or_else(detect_sysinfo)
}

fn parse_proc_meminfo() -> Option<RamInfo> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb: Option<u64> = None;
    let mut available_kb: Option<u64> = None;

    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            total_kb = parse_meminfo_kb(line);
        } else if line.starts_with("MemAvailable:") {
            available_kb = parse_meminfo_kb(line);
        }
        if total_kb.is_some() && available_kb.is_some() {
            break;
        }
    }

    Some(RamInfo {
        total_mb: total_kb? / 1024,
        available_mb: available_kb.unwrap_or(0) / 1024,
    })
}

fn parse_meminfo_kb(line: &str) -> Option<u64> {
    // Format: "MemTotal:       32768000 kB"
    line.split_whitespace().nth(1)?.parse().ok()
}

fn detect_sysinfo() -> RamInfo {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    // sysinfo 0.30 returns values in KiB.
    RamInfo {
        total_mb: sys.total_memory() / 1024,
        available_mb: sys.available_memory() / 1024,
    }
}
