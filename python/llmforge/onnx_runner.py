"""
stdin→stdout JSON bridge between OnnxAdapter (Rust) and onnxruntime (Python).

Input JSON (stdin):
  {
    "model_path": "...",
    "execution_provider": "CUDAExecutionProvider" | "CPUExecutionProvider",
    "threads": 8,
    "batch_size": 1,
    "prompt_tokens": 50
  }

Output JSON (stdout):
  { "tokens_per_sec": 45.2, "latency_ms": 210.0, "peak_memory_mb": 3400 }
  OR
  { "error": "message" }  -- on any failure; never raise an unhandled exception

Flags:
  --check   Verify onnxruntime imports and exit 0/1; no stdin expected.
"""

import json
import sys
import tracemalloc


def _error(msg: str) -> None:
    """Write an error JSON object to stdout and exit non-zero."""
    print(json.dumps({"error": msg}), flush=True)
    sys.exit(1)


def _check_mode() -> None:
    """Verify onnxruntime is importable; exit 0 on success, 1 on failure."""
    try:
        import onnxruntime  # noqa: F401
        sys.exit(0)
    except ImportError:
        _error("onnxruntime not installed")


def _run(request: dict) -> None:
    """Run the benchmark and write a result JSON object to stdout."""
    try:
        import onnxruntime as ort
    except ImportError:
        _error("onnxruntime not installed")
        return  # unreachable; satisfies type checkers

    model_path: str = request["model_path"]
    requested_provider: str = request.get(
        "execution_provider", "CPUExecutionProvider"
    )
    threads: int = int(request.get("threads", 1))
    prompt_tokens: int = int(request.get("prompt_tokens", 50))

    # Build session options.
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = threads
    sess_opts.inter_op_num_threads = 1

    # Determine available provider; fall back to CPU if requested one absent.
    available = ort.get_available_providers()
    if requested_provider in available:
        providers = [requested_provider]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )
    except Exception as exc:
        _error(f"failed to load model: {exc}")
        return

    # Build dummy zero inputs from model metadata.
    inputs: dict = {}
    try:
        for inp in session.get_inputs():
            shape = []
            for dim in inp.shape:
                if isinstance(dim, int) and dim > 0:
                    shape.append(dim)
                else:
                    shape.append(1)  # dynamic dim → 1
            import numpy as np
            inputs[inp.name] = np.zeros(shape, dtype=_ort_dtype_to_numpy(inp.type))
    except Exception as exc:
        _error(f"failed to build inputs: {exc}")
        return

    # Warm-up + benchmark: 5 iterations.
    import time

    tracemalloc.start()
    latencies: list[float] = []
    try:
        for _ in range(5):
            t0 = time.perf_counter()
            session.run(None, inputs)
            latencies.append(time.perf_counter() - t0)
    except Exception as exc:
        _error(f"inference failed: {exc}")
        return
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    mean_latency_s = sum(latencies) / len(latencies)
    mean_latency_ms = mean_latency_s * 1000.0
    tokens_per_sec = prompt_tokens / mean_latency_s if mean_latency_s > 0 else 0.0
    peak_memory_mb = peak // (1024 * 1024)

    print(
        json.dumps(
            {
                "tokens_per_sec": tokens_per_sec,
                "latency_ms": mean_latency_ms,
                "peak_memory_mb": peak_memory_mb,
            }
        ),
        flush=True,
    )


def _ort_dtype_to_numpy(ort_type: str) -> "type":
    """Map an ONNX Runtime type string to a numpy dtype."""
    import numpy as np

    mapping = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
        "tensor(float16)": np.float16,
    }
    return mapping.get(ort_type, np.float32)


def main() -> None:
    if "--check" in sys.argv:
        _check_mode()
        return

    try:
        raw = sys.stdin.read()
        request = json.loads(raw)
    except Exception as exc:
        _error(f"failed to parse stdin JSON: {exc}")
        return

    _run(request)


if __name__ == "__main__":
    main()
