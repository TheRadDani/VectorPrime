// Location: crates/vectorprime-hardware/src/gpu/apple.rs
//
// Implements GpuProbe for Apple Silicon / integrated Metal GPUs via
// `system_profiler`. This probe is a no-op on non-macOS platforms; the
// cfg guard ensures the binary-spawning code is never compiled for Linux
// or Windows, keeping cross-compilation clean.
// Called by gpu::probe_all() in mod.rs.

use vectorprime_core::{GpuInfo, GpuProbe, GpuVendor};

/// Probe implementation for Apple GPUs via `system_profiler`.
///
/// Returns `None` on non-macOS platforms unconditionally.
pub struct AppleProbe;

impl GpuProbe for AppleProbe {
    fn probe(&self) -> Option<GpuInfo> {
        // Only attempt on macOS; all other platforms return None immediately.
        #[cfg(target_os = "macos")]
        {
            probe_macos()
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }
}

/// macOS-only implementation: query `system_profiler SPDisplaysDataType -json`.
///
/// Parses GPU name and VRAM from the JSON output. Returns `None` on any
/// failure (binary absent, JSON malformed, required fields missing).
#[cfg(target_os = "macos")]
fn probe_macos() -> Option<GpuInfo> {
    use std::process::Command;

    let output = Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_system_profiler_json(&stdout)
}

/// Parse `system_profiler SPDisplaysDataType -json` JSON output.
///
/// The relevant structure is:
/// ```json
/// {
///   "SPDisplaysDataType": [
///     {
///       "sppci_model": "Apple M2 Pro",
///       "spdisplays_vram": "16 GB",
///       "spdisplays_vram_shared": "16384 MB"
///     }
///   ]
/// }
/// ```
/// Returns `None` when the expected keys are absent or the JSON is malformed.
/// This function is pub for unit-testing on non-macOS platforms.
// Allow dead_code: this function is used inside #[cfg(target_os = "macos")] and
// also referenced by tests on non-macOS platforms via #[cfg(test)].
#[allow(dead_code)]
pub fn parse_system_profiler_json(json: &str) -> Option<GpuInfo> {
    // Use a minimal hand-rolled parse to avoid adding a serde_json dependency
    // to the hardware crate. We look for the key strings directly.

    // Extract GPU model name from "sppci_model" : "..." or "sppci_vendor" : "..."
    let name = extract_json_string_value(json, "sppci_model")
        .or_else(|| extract_json_string_value(json, "_name"))
        .unwrap_or_else(|| "Apple GPU".to_string());

    // Extract VRAM from "spdisplays_vram_shared" (MB value preferred) or
    // "spdisplays_vram" (may have unit suffix like "16 GB").
    let vram_mb = extract_vram_mb(json)?;

    Some(GpuInfo {
        name,
        vram_mb,
        compute_capability: None, // Apple uses Metal, not CUDA compute caps
        vendor: GpuVendor::Apple,
    })
}

/// Extract the string value of a JSON key using simple substring search.
///
/// Handles the pattern `"key" : "value"` or `"key": "value"`.
/// Returns `None` if the key is not found.
#[allow(dead_code)]
fn extract_json_string_value(json: &str, key: &str) -> Option<String> {
    let search = format!("\"{}\"", key);
    let key_pos = json.find(&search)?;
    let after_key = &json[key_pos + search.len()..];
    // Skip whitespace and the colon.
    let colon_pos = after_key.find(':')?;
    let after_colon = &after_key[colon_pos + 1..].trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let inner = &after_colon[1..];
    let end = inner.find('"')?;
    let value = inner[..end].trim().to_string();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

/// Extract VRAM in megabytes from system_profiler JSON.
///
/// Tries `spdisplays_vram_shared` first (typically "16384 MB"), then
/// `spdisplays_vram` (may be "16 GB" or "16384 MB").
#[allow(dead_code)]
fn extract_vram_mb(json: &str) -> Option<u64> {
    // Prefer the shared-memory field which Apple Silicon reports.
    let raw = extract_json_string_value(json, "spdisplays_vram_shared")
        .or_else(|| extract_json_string_value(json, "spdisplays_vram"))?;

    parse_vram_string(&raw)
}

/// Parse a VRAM string like "16384 MB" or "16 GB" into megabytes.
///
/// Returns `None` if the numeric portion cannot be parsed.
#[allow(dead_code)]
fn parse_vram_string(s: &str) -> Option<u64> {
    let s = s.trim();
    // Split on whitespace to separate value and unit.
    let mut parts = s.splitn(2, char::is_whitespace);
    let value_str = parts.next()?;
    let unit = parts.next().unwrap_or("MB").trim().to_uppercase();

    let value: u64 = value_str.parse().ok()?;
    match unit.as_str() {
        "GB" => Some(value * 1024),
        "TB" => Some(value * 1_048_576),
        _ => Some(value), // assume MB
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_JSON: &str = r#"{
  "SPDisplaysDataType" : [
    {
      "sppci_model" : "Apple M2 Pro",
      "spdisplays_vram_shared" : "16384 MB",
      "spdisplays_vram" : "16 GB"
    }
  ]
}"#;

    #[test]
    fn test_parse_system_profiler_name_and_vram() {
        let info = parse_system_profiler_json(SAMPLE_JSON).expect("should parse");
        assert_eq!(info.name, "Apple M2 Pro");
        assert_eq!(info.vram_mb, 16384);
        assert_eq!(info.vendor, GpuVendor::Apple);
        assert!(info.compute_capability.is_none());
    }

    #[test]
    fn test_parse_vram_gb() {
        assert_eq!(parse_vram_string("16 GB"), Some(16384));
    }

    #[test]
    fn test_parse_vram_mb() {
        assert_eq!(parse_vram_string("8192 MB"), Some(8192));
    }

    #[test]
    fn test_parse_vram_no_unit_defaults_mb() {
        assert_eq!(parse_vram_string("4096"), Some(4096));
    }

    #[test]
    fn test_parse_system_profiler_malformed_returns_none() {
        // Missing VRAM fields means we return None because vram_mb is required.
        assert!(parse_system_profiler_json("{}").is_none());
        assert!(parse_system_profiler_json("").is_none());
    }

    #[test]
    fn test_apple_probe_non_macos_returns_none() {
        // On non-macOS platforms the probe must return None without panicking.
        #[cfg(not(target_os = "macos"))]
        {
            let probe = AppleProbe;
            assert!(probe.probe().is_none());
        }
        #[cfg(target_os = "macos")]
        {
            // On macOS this may or may not succeed depending on the machine.
            // We just verify it doesn't panic.
            let _result = AppleProbe.probe();
        }
    }
}
