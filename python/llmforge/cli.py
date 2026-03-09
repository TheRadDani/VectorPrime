"""LLMForge command-line interface."""

import argparse
import json
import sys
from pathlib import Path


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
        import llmforge._llmforge as _llmforge  # type: ignore[import]
        hw = _llmforge.profile_hardware()
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

    try:
        import llmforge._llmforge as _llmforge  # type: ignore[import]
        result = _llmforge.optimize(model_path, fmt)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Formatted summary.
    print(_divider())
    print("LLMForge Optimization Result")
    print(_divider())
    print(f"Runtime:       {result.runtime}")
    print(f"Quantization:  {result.quantization}")
    print(f"Threads:       {result.threads}")
    print(f"GPU Layers:    {result.gpu_layers}")
    print(f"Throughput:    {result.tokens_per_sec:.1f} tokens/sec")
    print(f"Latency:       {result.latency_ms:.1f} ms")
    print(f"Memory:        {result.peak_memory_mb / 1024:.1f} GB peak")
    print(_divider())

    # Persist result alongside the model file.
    out_path = model_path + ".llmforge_result.json"
    try:
        Path(out_path).write_text(result.to_json(), encoding="utf-8")
        print(f"Result written to: {out_path}")
    except OSError as e:
        print(f"WARNING: could not write result file: {e}", file=sys.stderr)

    # Honour --max-memory (informational only at this stage).
    if args.max_memory is not None:
        peak_mb = result.peak_memory_mb
        if peak_mb > args.max_memory:
            print(
                f"WARNING: peak memory {peak_mb} MB exceeds --max-memory {args.max_memory} MB",
                file=sys.stderr,
            )


def cmd_export_ollama(args: argparse.Namespace) -> None:
    model_path: str = args.model_path
    output_dir: str = args.output_dir
    result_file: str | None = args.result

    try:
        import llmforge._llmforge as _llmforge  # type: ignore[import]

        if result_file:
            # Load pre-computed result from JSON — reconstruct via a round-trip
            # through optimize() is not possible here, so we rely on the native
            # module's export path accepting an OptimizationResult directly.
            # For now we run a fresh optimize() if no native deserializer exists.
            # The simplest correct approach: pass model_path + result JSON to
            # export_ollama via a temp-loaded result object.
            result_json = Path(result_file).read_text(encoding="utf-8")
            result = _llmforge.optimize_from_json(result_json)
        else:
            fmt = detect_format(model_path)
            result = _llmforge.optimize(model_path, fmt)

        manifest_json: str = _llmforge.export_ollama(result, output_dir)
    except (RuntimeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_json)
    print(f"Output directory : {manifest['output_dir']}")
    print(f"Modelfile        : {manifest['modelfile_path']}")
    print(f"Model (GGUF)     : {manifest['model_gguf_path']}")
    print()
    print("Run with Ollama:")
    for cmd in manifest.get("ollama_commands", []):
        print(f"  {cmd}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmforge",
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

    # export-ollama
    exp = sub.add_parser(
        "export-ollama",
        help="Export an optimized model as an Ollama bundle.",
    )
    exp.add_argument("model_path", help="Path to the model file (.gguf or .onnx).")
    exp.add_argument(
        "--result",
        default=None,
        metavar="FILE",
        help="Path to a .llmforge_result.json file produced by 'optimize'.",
    )
    exp.add_argument(
        "--output-dir",
        default="optimized_model",
        metavar="DIR",
        help="Directory for the Ollama bundle (default: optimized_model/).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "profile": cmd_profile,
        "optimize": cmd_optimize,
        "export-ollama": cmd_export_ollama,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
