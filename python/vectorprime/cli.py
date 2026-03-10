"""VectorPrime command-line interface."""

import argparse
import sys


def detect_format(path: str) -> str:
    """Infer model format from file extension."""
    if path.endswith(".gguf"):
        return "gguf"
    if path.endswith(".onnx"):
        return "onnx"
    raise ValueError(f"Cannot detect format from extension: {path}")


def _divider() -> str:
    return "─" * 33


def cmd_profile(_args: argparse.Namespace) -> None:
    try:
        import vectorprime._vectorprime as _vectorprime  # type: ignore[import]
        hw = _vectorprime.profile_hardware()
        print(hw.to_json())
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


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

    try:
        import vectorprime._vectorprime as _vectorprime  # type: ignore[import]
        result = _vectorprime.optimize(model_path, fmt, args.gpu, args.latency, output)
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

    # Formatted summary.
    print(_divider())
    print("VectorPrime Optimization Result")
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
    sub.add_parser("profile", help="Detect and print hardware profile as JSON.")

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
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "profile": cmd_profile,
        "optimize": cmd_optimize,
        "convert-to-onnx": cmd_convert_to_onnx,
        "convert-to-gguf": cmd_convert_to_gguf,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
