"""VectorPrime command-line interface.

Location: python/vectorprime/cli.py

Summary: Entry point for the `vectorprime` CLI. Parses arguments and dispatches
to command handlers (profile, optimize, convert-to-onnx, convert-to-gguf, doctor).

Used by: pyproject.toml console_scripts entry point; also importable as a module.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys


def _print_logo() -> None:
    """Print the Optimus Prime ASCII art logo from assets."""
    try:
        # Get the path to the logo file relative to this module
        # cli.py is at: /home/daniel/VectorPrime/python/vectorprime/cli.py
        # We need to go up 2 levels to get: /home/daniel/VectorPrime/
        current_dir = os.path.dirname(os.path.abspath(__file__))    # vectorprime/
        package_dir = os.path.dirname(current_dir)                  # python/
        project_root = os.path.dirname(package_dir)                 # VectorPrime/
        logo_path = os.path.join(project_root, "assets", "optimus_prime_logo.txt")
        
        if os.path.exists(logo_path):
            with open(logo_path, "r", encoding="utf-8") as f:
                logo_content = f.read()
                print(logo_content)
    except Exception as e:
        # Silently fail if logo cannot be loaded
        pass


def _print_fancy_header() -> None:
    """Print fancy VectorPrime header with styling."""    
    # Print fancy VectorPrime text with colors
    print("""                                                      
██  ██ ██████ ▄█████ ██████ ▄████▄ █████▄  █████▄ █████▄  ██ ██▄  ▄██ ██████ 
██▄▄██ ██▄▄   ██       ██   ██  ██ ██▄▄██▄ ██▄▄█▀ ██▄▄██▄ ██ ██ ▀▀ ██ ██▄▄   
 ▀██▀  ██▄▄▄▄ ▀█████   ██   ▀████▀ ██   ██ ██     ██   ██ ██ ██    ██ ██▄▄▄▄                                                                                                                                                                                                                                     
    """)
    # Print logo
    _print_logo()
    print()
    sys.stdout.flush()  # Force output to display immediately


def detect_format(path: str) -> str:
    """Infer model format from file extension."""
    if path.endswith(".gguf"):
        return "gguf"
    if path.endswith(".onnx"):
        return "onnx"
    raise ValueError(f"Cannot detect format from extension: {path}")


def _divider() -> str:
    return "─" * 41


def _fmt_params(p: int) -> str:
    """Format a parameter count as a human-readable string (e.g. '7.0 B')."""
    if p >= 1_000_000_000:
        return f"{p / 1e9:.1f} B"
    if p >= 1_000_000:
        return f"{p / 1e6:.1f} M"
    return str(p)


def _fmt_flops(f: float) -> str:
    """Format a FLOPs count as a human-readable string (e.g. '14.0 T')."""
    if f >= 1e12:
        return f"{f / 1e12:.1f} T"
    if f >= 1e9:
        return f"{f / 1e9:.1f} G"
    return f"{f:.0f}"


def _fmt_mb_as_gb(mb: float) -> str:
    """Convert MB to GB string (e.g. '13.4 GB')."""
    return f"{mb / 1024:.1f} GB"


def _print_model_summary(info: dict) -> None:
    """Print a model inspection summary block to stdout.

    All fields are optional — lines are omitted when values are None.
    This function must never raise; callers wrap it in try/except.
    """
    div = _divider()
    print(div)
    print(" Model Inspection")
    print(div)

    if info.get("param_count") is not None:
        print(f" Parameters      : {_fmt_params(info['param_count'])}")
    if info.get("layer_count") is not None:
        print(f" Layers          : {info['layer_count']}")

    # Attention heads — combine query and KV counts on one line when both present.
    if info.get("attention_head_count") is not None:
        heads_str = str(info["attention_head_count"])
        if info.get("attention_head_count_kv") is not None:
            heads_str += f"  (KV heads: {info['attention_head_count_kv']})"
        print(f" Attention heads : {heads_str}")

    if info.get("hidden_size") is not None:
        print(f" Hidden size     : {info['hidden_size']}")
    if info.get("feed_forward_length") is not None:
        print(f" FFN size        : {info['feed_forward_length']}")
    if info.get("kv_cache_size_mb") is not None:
        print(f" KV cache        : {_fmt_mb_as_gb(info['kv_cache_size_mb'])}  (at full context)")
    if info.get("memory_footprint_mb") is not None:
        print(f" Memory footprint: {_fmt_mb_as_gb(info['memory_footprint_mb'])} (FP16)")
    if info.get("flops_per_token") is not None:
        print(f" FLOPs/token     : {_fmt_flops(info['flops_per_token'])}")

    # Workload classification (derived from FFN ratio).
    ffn = info.get("feed_forward_length")
    hidden = info.get("hidden_size")
    if ffn is not None and hidden is not None and hidden > 0:
        ffn_ratio = ffn / hidden
        if ffn_ratio >= 8:
            workload = "Compute-bound"
        elif ffn_ratio <= 2:
            workload = "Memory-bound"
        else:
            workload = "Balanced"
        print(div)
        print(f" Workload type   : {workload} (FFN ratio {ffn_ratio:.1f}\u00d7)")

    print(div)


def _mb_to_gb_str(mb: float) -> str:
    """Convert megabytes to a human-readable gigabyte string, e.g. '31.9 GB'."""
    return f"{mb / 1024:.1f} GB"


def _compute_cap_str(cap: list) -> str:
    """Format a compute capability list like [8, 9] as the string '8.9'."""
    if cap and len(cap) >= 2:
        return f"{cap[0]}.{cap[1]}"
    return "N/A"


def _round_to_model_size(param_billions: int) -> int:
    """Round a raw parameter count in billions to the nearest common model size."""
    common_sizes = [7, 13, 30, 70, 130]
    if param_billions <= 0:
        return param_billions
    closest = min(common_sizes, key=lambda s: abs(s - param_billions))
    return closest


def _derive_capabilities(hw: dict) -> dict:
    """Derive capability flags from a raw hardware profile dict.

    Args:
        hw: Parsed hardware profile containing 'cpu', 'gpu', and 'ram' keys.

    Returns:
        A dict with boolean capability flags.
    """
    gpu = hw.get("gpu")
    vendor = (gpu.get("vendor", "") if gpu else "").lower()
    cap = gpu.get("compute_capability", [0, 0]) if gpu else [0, 0]
    cap_tuple = tuple(cap[:2]) if len(cap) >= 2 else (0, 0)
    is_nvidia = vendor == "nvidia"
    tensorrt_ok = is_nvidia and cap_tuple >= (7, 0)
    tensor_cores = is_nvidia and cap_tuple >= (7, 0)
    return {
        "gpu_inference": gpu is not None,
        "fp16": gpu is not None,
        "int8": gpu is not None,
        "tensorrt_supported": tensorrt_ok,
        "tensor_cores": tensor_cores,
    }


def _derive_recommendation(hw: dict, caps: dict, installed_runtimes: dict) -> dict:
    """Derive runtime and precision recommendations from hardware and capabilities.

    Runtime recommendations are ordered by suitability for the detected hardware
    class and filtered to only runtimes confirmed as installed.  If nothing in
    the priority list is installed the full priority list is returned as
    aspirational recommendations (so the caller still gets a useful answer).

    Precision recommendations are based solely on hardware capabilities and do
    not depend on installed software.

    Args:
        hw: Parsed hardware profile dict.
        caps: Capability flags from _derive_capabilities().
        installed_runtimes: Boolean dict from _check_runtime_support().

    Returns:
        A dict with 'preferred_runtime' and 'preferred_precision' lists.
        Values are human-readable display strings (e.g. "llama.cpp", "FP16").
    """
    gpu = hw.get("gpu")
    vendor = (gpu.get("vendor", "") if gpu else "").lower()
    cpu = hw.get("cpu", {})
    simd_level = cpu.get("simd_level", "")

    cap = (gpu.get("compute_capability", [0, 0]) if gpu else [0, 0])
    cap_tuple = tuple(cap[:2]) if len(cap) >= 2 else (0, 0)

    is_nvidia = vendor == "nvidia"
    is_amd = vendor == "amd"
    is_apple = vendor in ("apple", "metal")

    # --- Runtime priority list (keys from _check_runtime_support) ---
    if is_nvidia and cap_tuple >= (7, 0):
        # Volta and above: TensorRT is the optimal choice.
        priority = ["tensorrt", "vllm", "onnx_runtime", "llama_cpp", "ollama"]
    elif is_nvidia:
        # Pascal and older NVIDIA: no TensorRT Tensor Core benefit.
        priority = ["vllm", "onnx_runtime", "llama_cpp", "ollama"]
    elif is_amd:
        priority = ["llama_cpp", "onnx_runtime", "ollama"]
    elif is_apple:
        priority = ["llama_cpp", "onnx_runtime", "ollama"]
    elif gpu:
        # Unknown GPU vendor — safe defaults.
        priority = ["llama_cpp", "onnx_runtime", "ollama"]
    else:
        # CPU-only.
        priority = ["llama_cpp", "ollama", "onnx_runtime"]

    # Filter priority list to installed runtimes; fall back to full list if
    # nothing is installed (aspirational recommendations).
    installed_priority = [k for k in priority if installed_runtimes.get(k)]
    effective_priority = installed_priority if installed_priority else priority
    preferred_runtime = [_RUNTIME_LABELS[k] for k in effective_priority]

    # --- Precision based on hardware capabilities only ---
    if is_nvidia and cap_tuple >= (8, 0):
        # Ampere / Ada / Hopper: full Tensor Core support including INT4.
        preferred_precision = ["FP16", "INT8", "INT4"]
    elif is_nvidia and cap_tuple >= (7, 0):
        # Volta / Turing: Tensor Cores for FP16 and INT8.
        preferred_precision = ["FP16", "INT8"]
    elif is_nvidia:
        # Pascal and older: no Tensor Cores.
        preferred_precision = ["FP32", "FP16"]
    elif is_amd or is_apple:
        # ROCm and Metal MPS both support FP16 and INT8.
        preferred_precision = ["FP16", "INT8"]
    elif gpu:
        # Unknown GPU.
        preferred_precision = ["FP16", "INT8"]
    else:
        # CPU-only: choose quantized formats based on available SIMD.
        if simd_level == "AVX512":
            preferred_precision = ["INT8", "Q4_K_M", "Q8_0"]
        elif simd_level == "AVX2":
            preferred_precision = ["Q4_K_M", "Q8_0", "INT8"]
        else:
            preferred_precision = ["Q4_K_M", "Q8_0"]

    return {
        "preferred_runtime": preferred_runtime,
        "preferred_precision": preferred_precision,
    }


def _model_capacity_estimate(vram_mb: float) -> tuple:
    """Estimate model capacity from VRAM.

    Args:
        vram_mb: Available VRAM in megabytes.

    Returns:
        Tuple of (quantized_B, full_gpu_B) as rounded parameter counts in billions.
    """
    vram_gb = vram_mb / 1024
    quantized_raw = int(vram_gb * 4)
    full_gpu_raw = int(vram_gb / 2)
    quantized_b = _round_to_model_size(quantized_raw) if quantized_raw > 0 else quantized_raw
    full_gpu_b = _round_to_model_size(full_gpu_raw) if full_gpu_raw > 0 else full_gpu_raw
    return quantized_b, full_gpu_b


def _get_nvidia_smi_info() -> dict:
    """Run nvidia-smi and parse CUDA and driver version from its output.

    Returns:
        Dict with 'cuda_version' and 'driver_version' strings, or 'N/A' on failure.
    """
    result = {"cuda_version": "N/A", "driver_version": "N/A"}
    if not shutil.which("nvidia-smi"):
        return result
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in proc.stdout.splitlines():
            if "CUDA Version:" in line:
                parts = line.split("CUDA Version:")
                if len(parts) > 1:
                    result["cuda_version"] = parts[1].strip().rstrip("|").strip()
            if "Driver Version:" in line:
                parts = line.split("Driver Version:")
                if len(parts) > 1:
                    # Driver Version: 550.xx  CUDA Version: 12.4
                    token = parts[1].strip().split()[0]
                    result["driver_version"] = token
    except Exception:
        pass
    return result


def _get_verbose_cpu_info(hw: dict) -> dict:
    """Gather verbose CPU information from /proc/cpuinfo and the hardware profile.

    Args:
        hw: Parsed hardware profile dict.

    Returns:
        Dict with cpu diagnostic fields; missing values fall back to 'N/A'.
    """
    cpu = hw.get("cpu", {})
    logical_threads = cpu.get("core_count", "N/A")
    arch = platform.machine()

    # Physical cores: try psutil, otherwise estimate as logical // 2.
    try:
        import psutil  # type: ignore[import]
        physical_cores = psutil.cpu_count(logical=False) or "N/A"
    except ImportError:
        physical_cores = (logical_threads // 2) if isinstance(logical_threads, int) else "N/A"

    # Base clock from /proc/cpuinfo.
    base_clock = "N/A"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("cpu mhz"):
                    mhz_str = line.split(":", 1)[1].strip()
                    mhz = float(mhz_str)
                    base_clock = f"{mhz / 1000:.1f} GHz"
                    break
    except Exception:
        pass

    # Expand SIMD level into a full feature list.
    simd_map = {
        "AVX512": "SSE4, AVX, AVX2, AVX512",
        "AVX2":   "SSE4, AVX, AVX2",
        "AVX":    "SSE4, AVX",
    }
    simd_raw = cpu.get("simd_level", "")
    simd_features = simd_map.get(simd_raw, "SSE2")

    # L3 cache from /proc/cpuinfo.
    l3_cache = "N/A"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if "cache size" in line.lower():
                    l3_cache = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    return {
        "arch": arch,
        "physical_cores": physical_cores,
        "logical_threads": logical_threads,
        "base_clock": base_clock,
        "simd_features": simd_features,
        "l3_cache": l3_cache,
    }


def _get_verbose_mem_info() -> dict:
    """Parse /proc/meminfo for swap total.

    Returns:
        Dict with 'swap' as a GB string, or 'N/A'.
    """
    swap = "N/A"
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("SwapTotal:"):
                    kb = int(line.split()[1])
                    swap = f"{kb / 1024 / 1024:.1f} GB"
                    break
    except Exception:
        pass
    return {"swap": swap}


def _get_gpu_bandwidth(gpu_name: str) -> str:
    """Look up approximate memory bandwidth for a known GPU model.

    Args:
        gpu_name: GPU model name string from the hardware profile.

    Returns:
        Memory bandwidth string, e.g. '~1008 GB/s', or 'N/A'.
    """
    name_lower = gpu_name.lower()
    bandwidth_table = {
        "rtx 4090 laptop": "~576 GB/s",
        "rtx 4090":        "~1008 GB/s",
        "rtx 3090":        "~936 GB/s",
        "rtx 3080":        "~760 GB/s",
        "rtx 3070":        "~448 GB/s",
        "a100":            "~2000 GB/s",
        "h100":            "~3350 GB/s",
    }
    for key, val in bandwidth_table.items():
        if key in name_lower:
            return val
    return "N/A"


def _check_runtime_support() -> dict:
    """Check which inference runtimes are available on this system.

    Uses PATH probing for binary runtimes and importlib.util.find_spec for
    Python-package-based runtimes (onnxruntime, vllm) to avoid importing them.

    Returns:
        Dict mapping internal runtime keys to booleans.
        Keys: "llama_cpp", "ollama", "tensorrt", "onnx_runtime", "vllm".
    """
    import importlib.util

    return {
        "llama_cpp":    bool(shutil.which("llama-cli") or shutil.which("llama-server")),
        "ollama":       shutil.which("ollama") is not None,
        "tensorrt":     shutil.which("trtexec") is not None,
        "onnx_runtime": importlib.util.find_spec("onnxruntime") is not None,
        "vllm":         importlib.util.find_spec("vllm") is not None,
    }


# Human-readable display labels for each internal runtime key.
_RUNTIME_LABELS: dict = {
    "tensorrt":     "TensorRT",
    "vllm":         "vLLM",
    "onnx_runtime": "ONNX Runtime",
    "llama_cpp":    "llama.cpp",
    "ollama":       "Ollama",
}


def _print_pretty_profile(hw: dict, caps: dict, rec: dict) -> None:
    """Print the default human-readable hardware profile summary.

    Args:
        hw:   Parsed hardware profile dict.
        caps: Capability flags from _derive_capabilities().
        rec:  Recommendations from _derive_recommendation().
    """
    div = "─" * 40
    cpu = hw.get("cpu", {})
    gpu = hw.get("gpu")
    ram = hw.get("ram", {})

    print("VectorPrime Hardware Profile")
    print(div)
    print()

    # CPU section.
    print("CPU")
    brand = cpu.get("brand", "N/A")
    # Strip "(R)", "(TM)" for a cleaner display name.
    display_brand = brand.replace("(R)", "").replace("(TM)", "").strip()
    print(f"  {'Model:':<22} {display_brand}")
    print(f"  {'Cores:':<22} {cpu.get('core_count', 'N/A')} threads")
    print(f"  {'SIMD Support:':<22} {cpu.get('simd_level', 'N/A')}")
    print()

    # GPU section.
    print("GPU")
    if gpu:
        gpu_name = gpu.get("name", "N/A")
        vendor = gpu.get("vendor", "N/A")
        vram_mb = gpu.get("vram_mb", 0)
        cap = gpu.get("compute_capability", [])
        print(f"  {'Model:':<22} {gpu_name}")
        print(f"  {'Vendor:':<22} {vendor.upper() if vendor != 'N/A' else vendor}")
        print(f"  {'VRAM:':<22} {_mb_to_gb_str(vram_mb)}")
        print(f"  {'Compute Capability:':<22} {_compute_cap_str(cap)}")
        tc_str = "Yes" if caps.get("tensor_cores") else "No"
        print(f"  {'Tensor Cores:':<22} {tc_str}")
    else:
        print("  No GPU detected.")
    print()

    # Memory section.
    print("Memory")
    print(f"  {'Total RAM:':<22} {_mb_to_gb_str(ram.get('total_mb', 0))}")
    print(f"  {'Available RAM:':<22} {_mb_to_gb_str(ram.get('available_mb', 0))}")
    print()

    # Acceleration support.
    print("Acceleration Support")
    _checkmark = lambda flag: "✓" if flag else "✗"
    print(f"  {_checkmark(caps['gpu_inference'])} GPU inference available")
    print(f"  {_checkmark(caps['fp16'])} FP16 supported")
    print(f"  {_checkmark(caps['int8'])} INT8 supported")
    print(f"  {_checkmark(caps['tensorrt_supported'])} TensorRT compatible")
    print()

    # Recommended inference setup.
    print("Recommended Inference Setup")
    runtime_str = " / ".join(rec.get("preferred_runtime", []))
    precision_str = " or ".join(rec.get("preferred_precision", []))
    print(f"  {'Runtime:':<22} {runtime_str}")
    print(f"  {'Precision:':<22} {precision_str}")

    if gpu:
        vram_mb = gpu.get("vram_mb", 0)
        q_b, full_b = _model_capacity_estimate(vram_mb)
        print(f"  Estimated Model Capacity:")
        print(f"      • ~{q_b}B quantized")
        print(f"      • ~{full_b}B full GPU")

    print()
    print("Tip: run `vectorprime profile --verbose` for full hardware diagnostics.")


def _print_verbose_profile(hw: dict, caps: dict, rec: dict, installed_runtimes: dict) -> None:
    """Print the full verbose hardware diagnostic report.

    Args:
        hw:                Parsed hardware profile dict.
        caps:              Capability flags from _derive_capabilities().
        rec:               Recommendations from _derive_recommendation().
        installed_runtimes: Boolean dict from _check_runtime_support().
    """
    heavy_div = "═" * 39
    div = "─" * 39
    cpu = hw.get("cpu", {})
    gpu = hw.get("gpu")
    ram = hw.get("ram", {})

    cpu_verbose = _get_verbose_cpu_info(hw)
    mem_verbose = _get_verbose_mem_info()
    nvidia_info = _get_nvidia_smi_info() if gpu else {"cuda_version": "N/A", "driver_version": "N/A"}

    print("VectorPrime Hardware Diagnostic Report")
    print(heavy_div)
    print()

    # CPU.
    print("CPU")
    brand = cpu.get("brand", "N/A")
    display_brand = brand.replace("(R)", "").replace("(TM)", "").strip()
    print(f"  {'Model:':<22} {display_brand}")
    print(f"  {'Architecture:':<22} {cpu_verbose['arch']}")
    print(f"  {'Physical Cores:':<22} {cpu_verbose['physical_cores']}")
    print(f"  {'Logical Threads:':<22} {cpu_verbose['logical_threads']}")
    print(f"  {'Base Clock:':<22} {cpu_verbose['base_clock']}")
    print(f"  {'SIMD Features:':<22} {cpu_verbose['simd_features']}")
    print(f"  {'L3 Cache:':<22} {cpu_verbose['l3_cache']}")
    print()

    # GPU.
    print("GPU")
    if gpu:
        gpu_name = gpu.get("name", "N/A")
        vendor = gpu.get("vendor", "N/A")
        vram_mb = gpu.get("vram_mb", 0)
        cap = gpu.get("compute_capability", [])
        bw = _get_gpu_bandwidth(gpu_name)
        tc_str = "Yes" if caps.get("tensor_cores") else "No"
        print(f"  {'Model:':<22} {gpu_name}")
        print(f"  {'Vendor:':<22} {vendor.upper() if vendor != 'N/A' else vendor}")
        print(f"  {'Compute Capability:':<22} {_compute_cap_str(cap)}")
        print(f"  {'VRAM:':<22} {_mb_to_gb_str(vram_mb)}")
        print(f"  {'Tensor Cores:':<22} {tc_str}")
        print(f"  {'CUDA Version:':<22} {nvidia_info['cuda_version']}")
        print(f"  {'Driver Version:':<22} {nvidia_info['driver_version']}")
        print(f"  {'Memory Bandwidth:':<22} {bw}")
    else:
        print("  No GPU detected.")
    print()

    # System Memory.
    print("System Memory")
    print(f"  {'Total RAM:':<22} {_mb_to_gb_str(ram.get('total_mb', 0))}")
    print(f"  {'Available RAM:':<22} {_mb_to_gb_str(ram.get('available_mb', 0))}")
    print(f"  {'Swap:':<22} {mem_verbose['swap']}")
    print()

    # Acceleration support.
    print("Acceleration Support")
    cuda_avail = "Available" if shutil.which("nvidia-smi") else "Not found"
    trt_avail = "Compatible" if caps.get("tensorrt_supported") else "Not available"
    fp16_avail = "Supported" if caps.get("fp16") else "Not supported"
    int8_avail = "Supported" if caps.get("int8") else "Not supported"
    print(f"  {'CUDA:':<22} {cuda_avail}")
    print(f"  {'TensorRT:':<22} {trt_avail}")
    print(f"  {'FP16 Inference:':<22} {fp16_avail}")
    print(f"  {'INT8 Inference:':<22} {int8_avail}")
    print()

    # Runtime compatibility — display clean labels with Supported / Not found status.
    print("Runtime Compatibility")
    for key, label in _RUNTIME_LABELS.items():
        status = "Supported" if installed_runtimes.get(key) else "Not found"
        print(f"  {(label + ':'):<22} {status}")
    print()

    # VectorPrime optimization hints.
    print("VectorPrime Optimization Hints")
    runtime_str = " / ".join(rec.get("preferred_runtime", []))
    precision_str = " / ".join(rec.get("preferred_precision", []))
    logical = cpu.get("core_count", 0)
    thread_lo = max(1, logical // 2) if isinstance(logical, int) else "N/A"
    thread_hi = logical if isinstance(logical, int) else "N/A"
    thread_range = f"{thread_lo}–{thread_hi}" if isinstance(thread_lo, int) else "N/A"
    gpu_offload = "High" if (gpu and gpu.get("vram_mb", 0) >= 8192) else ("Medium" if gpu else "None")
    print(f"  {'Recommended Runtime:':<26} {runtime_str}")
    print(f"  {'Recommended Precision:':<26} {precision_str}")
    print(f"  {'Suggested Threads:':<26} {thread_range}")
    print(f"  {'GPU Offload Capacity:':<26} {gpu_offload}")
    print()

    # System readiness.
    print("System Readiness")
    cuda_ok = shutil.which("nvidia-smi") is not None
    gpu_cap_ok = gpu is not None
    vram_ok = gpu is not None and gpu.get("vram_mb", 0) >= 4096
    _c = lambda flag: "✓" if flag else "✗"
    print(f"  {_c(cuda_ok)} CUDA driver detected")
    print(f"  {_c(gpu_cap_ok)} GPU compute capability supported")
    print(f"  {_c(vram_ok)} Sufficient VRAM for large LLMs")
    print()
    if cuda_ok and gpu_cap_ok and vram_ok:
        print("System ready for optimized LLM inference.")
    else:
        print("Some system components may limit inference performance.")


def cmd_profile(args: argparse.Namespace) -> None:
    """Handle the `vectorprime profile` command.

    Supports four output modes controlled by flags:
    - default:   Pretty human-readable summary.
    - --verbose: Full hardware diagnostic report.
    - --json:    Structured JSON output to stdout.
    - --save:    Save JSON to a file (combinable with --json).
    """
    try:
        import vectorprime._vectorprime as _vectorprime  # type: ignore[import]
        hw_obj = _vectorprime.profile_hardware()
        hw: dict = json.loads(hw_obj.to_json())
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to parse hardware profile: {e}", file=sys.stderr)
        sys.exit(1)

    caps = _derive_capabilities(hw)
    installed_runtimes = _check_runtime_support()
    rec = _derive_recommendation(hw, caps, installed_runtimes)

    # Determine output mode. --verbose takes precedence over --json.
    verbose: bool = getattr(args, "verbose", False)
    as_json: bool = getattr(args, "json", False)
    save_path: str | None = getattr(args, "save", None)

    if verbose:
        _print_verbose_profile(hw, caps, rec, installed_runtimes)
        return

    if as_json or save_path:
        # Build enriched JSON payload.
        gpu = hw.get("gpu")
        cap = (gpu.get("compute_capability", []) if gpu else [])
        enriched_gpu = None
        if gpu:
            enriched_gpu = {
                "name": gpu.get("name"),
                "vendor": gpu.get("vendor"),
                "vram_mb": gpu.get("vram_mb"),
                "compute_capability": cap,
                "tensor_cores": caps.get("tensor_cores", False),
            }
        payload = {
            "cpu": hw.get("cpu"),
            "gpu": enriched_gpu,
            "ram": hw.get("ram"),
            "capabilities": caps,
            "recommendation": rec,
        }
        json_str = json.dumps(payload, indent=2)

        if as_json:
            print(json_str)

        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(json_str)
                    f.write("\n")
                print(f"Hardware profile saved to: {save_path}")
            except OSError as e:
                print(f"ERROR: Could not write to {save_path}: {e}", file=sys.stderr)
                sys.exit(1)
        return

    # Default: pretty human-readable output.
    _print_pretty_profile(hw, caps, rec)


def cmd_optimize(args: argparse.Namespace) -> None:
    model_path: str = args.model_path

    # Auto-detect format when not supplied.
    fmt: str = args.format
    if not fmt:
        try:
            fmt = detect_format(model_path)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    output: str | None = getattr(args, "output", None)

    print("Running 4-stage Bayesian optimization pipeline (hardware profiling → model analysis → runtime preselection → TPE Bayesian search)…")
    try:
        import vectorprime._vectorprime as _vectorprime  # type: ignore[import]
        result = _vectorprime.optimize(model_path, fmt, args.gpu, args.latency, output, not args.no_cache)
    except RuntimeError as e:
        msg = str(e)
        if "llama-quantize" in msg:
            print(
                "llama-quantize not found — install llama.cpp to enable model quantization",
                file=sys.stderr,
            )
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    # Model inspection summary (best-effort — never fatal).
    try:
        import vectorprime._vectorprime as _vectorprime  # type: ignore[import]
        model_info = _vectorprime.analyze_model(model_path)
        _print_model_summary(model_info)
    except Exception:
        pass

    # Formatted summary.
    print(_divider())
    print("VectorPrime Optimization Result  (4-stage Bayesian / TPE)")
    print(_divider())
    print(f"Runtime:       {result.runtime}")
    print(f"Quantization:  {result.quantization}")
    print(f"Threads:       {result.threads}")
    print(f"GPU Layers:    {result.gpu_layers}")
    print(f"Throughput:    {result.tokens_per_sec:.1f} tokens/sec")
    print(f"Latency:       {result.latency_ms:.1f} ms")
    print(f"Memory:        {result.peak_memory_mb / 1024:.1f} GB peak")
    print(_divider())

    # Report the re-quantized output path when quantization succeeded.
    if result.output_path is not None:
        print(f"Optimized model written to: {result.output_path}")
    else:
        print(
            "NOTE: llama-quantize not found — model was not re-quantized. "
            "Install llama.cpp to enable quantization.",
            file=sys.stderr,
        )

    # Honour --max-memory (informational only at this stage).
    if args.max_memory is not None:
        peak_mb = result.peak_memory_mb
        if peak_mb > args.max_memory:
            print(
                f"WARNING: peak memory {peak_mb} MB exceeds --max-memory {args.max_memory} MB",
                file=sys.stderr,
            )


def cmd_doctor(_args: argparse.Namespace) -> None:
    """Handle the `vectorprime doctor` command.

    Checks system readiness for optimized LLM inference by probing for
    required binaries. Prints a checklist with pass/fail markers and a
    summary line at the end.
    """
    div = "─" * 24

    # Each entry: (display_name, detection_function)
    def _has_cuda() -> bool:
        return shutil.which("nvidia-smi") is not None

    def _has_gpu_driver() -> bool:
        # Same heuristic: nvidia-smi presence implies driver is loaded.
        return shutil.which("nvidia-smi") is not None

    def _has_tensorrt() -> bool:
        return shutil.which("trtexec") is not None

    def _has_llama_cpp_gpu() -> bool:
        return bool(shutil.which("llama-cli") or shutil.which("llama-server"))

    checks = [
        ("CUDA installed",          _has_cuda),
        ("GPU driver detected",     _has_gpu_driver),
        ("TensorRT available",      _has_tensorrt),
        ("llama.cpp GPU support",   _has_llama_cpp_gpu),
    ]

    print("VectorPrime System Check")
    print(div)
    print()

    all_ok = True
    for label, probe in checks:
        ok = probe()
        marker = "✓" if ok else "✗"
        if not ok:
            all_ok = False
        print(f"{marker} {label}")

    print()
    if all_ok:
        print("System ready for optimized inference.")
    else:
        print("Some components missing — see above.")


def cmd_convert_to_onnx(args: argparse.Namespace) -> None:
    input_path: str = args.input_path
    output_path: str = args.output or _replace_ext(input_path, ".onnx")

    try:
        import vectorprime._vectorprime as _vectorprime  # type: ignore[import]
        result = _vectorprime.convert_gguf_to_onnx(input_path, output_path)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(_divider())
    print("VectorPrime Conversion Result")
    print(_divider())
    print(f"Input  (GGUF): {input_path}")
    print(f"Output (ONNX): {result}")
    print(_divider())


def cmd_convert_to_gguf(args: argparse.Namespace) -> None:
    input_path: str = args.input_path
    output_path: str = args.output or _replace_ext(input_path, ".gguf")

    try:
        import vectorprime._vectorprime as _vectorprime  # type: ignore[import]
        result = _vectorprime.convert_onnx_to_gguf(input_path, output_path)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(_divider())
    print("VectorPrime Conversion Result")
    print(_divider())
    print(f"Input  (ONNX): {input_path}")
    print(f"Output (GGUF): {result}")
    print(_divider())


def _replace_ext(path: str, new_ext: str) -> str:
    """Return *path* with its extension replaced by *new_ext* (e.g. '.onnx')."""
    import os
    root, _ = os.path.splitext(path)
    return root + new_ext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vectorprime",
        description="Hardware-aware LLM inference optimizer.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # profile
    prof = sub.add_parser("profile", help="Detect and display hardware profile.")
    prof.add_argument(
        "--json",
        action="store_true",
        help="Output profile as JSON.",
    )
    prof.add_argument(
        "--verbose",
        action="store_true",
        help="Show full hardware diagnostics.",
    )
    prof.add_argument(
        "--save",
        metavar="PATH",
        help="Save JSON profile to file.",
    )

    # doctor
    sub.add_parser("doctor", help="Check system readiness for optimized LLM inference.")

    # optimize
    opt = sub.add_parser(
        "optimize",
        help="Find the best inference configuration for a model.",
    )
    opt.add_argument("model_path", help="Path to the model file (.gguf or .onnx).")
    opt.add_argument(
        "--format",
        choices=["gguf", "onnx"],
        default=None,
        help="Model format (auto-detected from extension when omitted).",
    )
    opt.add_argument(
        "--max-memory",
        type=int,
        default=None,
        metavar="MB",
        help="Warn if peak memory exceeds this limit (MB).",
    )
    opt.add_argument(
        "--gpu",
        default=None,
        metavar="MODEL",
        help=(
            "Target GPU model (e.g. 4090, 3090, a100, h100) or 'cpu' for CPU-only. "
            "Overrides auto-detected GPU hardware. "
            "Accepts case-insensitive names with optional spaces or dashes "
            "(e.g. 'RTX 4090', 'rtx-4090', '4090' all work)."
        ),
    )
    opt.add_argument(
        "--latency",
        type=float,
        default=None,
        metavar="MS",
        help="Maximum tolerated latency in milliseconds. Configurations that exceed this threshold are excluded.",
    )
    opt.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Destination path for the re-quantized output model "
            "(default: {stem}-optimized.gguf next to the input file)."
        ),
    )
    opt.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Bypass the optimization cache and force a full benchmark run.",
    )

    # convert-to-onnx
    c2onnx = sub.add_parser(
        "convert-to-onnx",
        help="Convert a GGUF model to ONNX format.",
    )
    c2onnx.add_argument("input_path", help="Path to the source .gguf file.")
    c2onnx.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Destination path for the .onnx file (default: same stem, .onnx extension).",
    )

    # convert-to-gguf
    c2gguf = sub.add_parser(
        "convert-to-gguf",
        help="Convert an ONNX model to GGUF format.",
    )
    c2gguf.add_argument("input_path", help="Path to the source .onnx file.")
    c2gguf.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Destination path for the .gguf file (default: same stem, .gguf extension).",
    )

    return parser


def main() -> None:
    # Display fancy header with logo
    _print_fancy_header()
    
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "profile": cmd_profile,
        "doctor": cmd_doctor,
        "optimize": cmd_optimize,
        "convert-to-onnx": cmd_convert_to_onnx,
        "convert-to-gguf": cmd_convert_to_gguf,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
