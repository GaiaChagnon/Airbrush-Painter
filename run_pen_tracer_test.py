#!/usr/bin/env python3
"""Batch pen tracer -- process images with optional multithreading."""

import sys
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import pen_tracer
from src.utils import logging_config


def _process_one_image(
    image_path: Path,
    out_dir: Path,
    env_cfg_path: str,
    pen_tool_cfg_path: str,
    pen_tracer_cfg_path: str,
    cmy_canvas_path: str | None,
) -> dict:
    """Process a single image and return result dict or raise on failure."""
    result = pen_tracer.make_pen_layer(
        target_rgb_path=str(image_path),
        env_cfg_path=env_cfg_path,
        pen_tool_cfg_path=pen_tool_cfg_path,
        pen_tracer_cfg_path=pen_tracer_cfg_path,
        out_dir=str(out_dir),
        cmy_canvas_path=cmy_canvas_path,
    )

    # Validate path ordering
    from src.utils import validators
    pen_vectors = validators.load_pen_vectors(result["pen_vectors_yaml"])
    paths = pen_vectors.paths

    total_paths = result["metrics"]["num_edge_paths"] + result["metrics"]["num_hatch_paths"]
    assert len(paths) == total_paths, f"Path count mismatch: {len(paths)} != {total_paths}"

    edge_indices = [i for i, p in enumerate(paths) if p.role == "outline"]
    hatch_indices = [i for i, p in enumerate(paths) if p.role == "hatch"]
    if edge_indices and hatch_indices:
        assert max(edge_indices) < min(hatch_indices), "Edges should be drawn before hatching"

    assert "travel_distance_mm" in result["metrics"], "Travel distance not in metrics"
    assert result["metrics"]["travel_distance_mm"] >= 0, "Invalid travel distance"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch pen tracer for images")
    parser.add_argument("--input", "-i", type=str,
                        default="data/target_images/cmy_only/hard",
                        help="Input image path or directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: outputs/pen_test or outputs/pen_test_bw)")
    parser.add_argument("--env-cfg", type=str,
                        default="configs/env_airbrush_v1.yaml",
                        help="Environment config path")
    parser.add_argument("--pen-tool-cfg", type=str,
                        default="configs/tools/pen_finetip_v1.yaml",
                        help="Pen tool config path")
    parser.add_argument("--pen-tracer-cfg", type=str, default=None,
                        help="Pen tracer config path (overrides --mode)")
    parser.add_argument("--mode", type=str, choices=["color", "bw"], default="color",
                        help="Tracing mode: 'color' (CMY complement) or 'bw' (standalone B&W)")
    parser.add_argument("--cmy-canvas", type=str, default=None,
                        help="Optional CMY canvas image path")
    parser.add_argument("--threads", "-t", type=int, default=10,
                        help="Max parallel threads for batch processing (default: 10)")

    args = parser.parse_args()

    # Setup logging
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    logging_config.setup_logging(log_level="INFO", log_file="outputs/logs/pen_test.log")

    # Determine input paths
    input_path = Path(args.input)
    if input_path.is_dir():
        image_paths = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.name.startswith(".")
        )
    else:
        image_paths = [input_path]

    # Resolve config
    MODE_CONFIGS = {
        "color": "configs/sim/pen_tracer_v2.yaml",
        "bw":    "configs/sim/pen_tracer_v2_bw.yaml",
    }

    if args.output is None:
        args.output = "outputs/pen_test_bw" if args.mode == "bw" else "outputs/pen_test"

    env_cfg_path = args.env_cfg
    pen_tool_cfg_path = args.pen_tool_cfg
    pen_tracer_cfg_path = args.pen_tracer_cfg or MODE_CONFIGS[args.mode]
    cmy_canvas_path = args.cmy_canvas

    mode_label = "B&W standalone" if args.mode == "bw" else "CMY complement (colour)"
    n_threads = min(args.threads, len(image_paths))

    print("=" * 80)
    print(f"PEN TRACER TEST - A4 Print Quality [{mode_label}]")
    print("=" * 80)
    print(f"\nProcessing {len(image_paths)} image(s) with {n_threads} thread(s)")
    print(f"Output directory: {args.output}")
    print(f"Config: {pen_tracer_cfg_path}")
    print(f"\nMode: {mode_label}")
    print("  Paper size: A4 (210mm x 297mm)")
    if args.mode == "bw":
        print("  Full edge detection (all visible boundaries)")
        print("  Non-gamut-aware hatching (all dark regions)")
        print("  Cross-hatch for deep shadows, 5 darkness levels")
    else:
        print("  Gamut-aware: Only out-of-CMY colours")
        print("  Single-direction hatching: 45 diagonal")
    print("=" * 80)

    results: list[dict] = []
    errors: list[dict] = []
    t_start = time.monotonic()

    if n_threads <= 1:
        # Sequential processing
        for idx, image_path in enumerate(image_paths, 1):
            print(f"\n[{idx}/{len(image_paths)}] Processing: {image_path.name}")
            print("-" * 80)
            out_dir = Path(args.output) / image_path.stem
            try:
                result = _process_one_image(
                    image_path, out_dir, env_cfg_path,
                    pen_tool_cfg_path, pen_tracer_cfg_path, cmy_canvas_path,
                )
                results.append({"image": image_path.name, "result": result})
                print(f"  SUCCESS: {image_path.name}")
                _print_result_summary(result)
            except Exception as e:
                errors.append({"image": image_path.name, "error": str(e)})
                print(f"  ERROR: {image_path.name} -- {type(e).__name__}: {e}")
    else:
        # Parallel processing
        futures = {}
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for image_path in image_paths:
                out_dir = Path(args.output) / image_path.stem
                fut = pool.submit(
                    _process_one_image,
                    image_path, out_dir, env_cfg_path,
                    pen_tool_cfg_path, pen_tracer_cfg_path, cmy_canvas_path,
                )
                futures[fut] = image_path

            for fut in as_completed(futures):
                image_path = futures[fut]
                try:
                    result = fut.result()
                    results.append({"image": image_path.name, "result": result})
                    print(f"\n  SUCCESS: {image_path.name}")
                    _print_result_summary(result)
                except Exception as e:
                    errors.append({"image": image_path.name, "error": str(e)})
                    print(f"\n  ERROR: {image_path.name} -- {type(e).__name__}: {e}")

    elapsed = time.monotonic() - t_start

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nProcessed: {len(results)}/{len(image_paths)} images "
          f"in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors:
            print(f"  {err['image']}: {err['error']}")

    if results:
        print("\nSuccessful outputs:")
        for r in sorted(results, key=lambda x: x["image"]):
            metrics = r["result"]["metrics"]
            print(f"\n{r['image']}:")
            print(f"  Resolution: {metrics['resolution'][0]}x{metrics['resolution'][1]} px")
            print(f"  Edges: {metrics['num_edge_paths']}, Hatches: {metrics['num_hatch_paths']}")
            print(f"  Coverage: {metrics['coverage_black'] * 100:.1f}%")
            print(f"  Travel distance: {metrics['travel_distance_mm']:.1f}mm")
            print(f"  Preview: {r['result']['pen_preview_png']}")

    print("\n" + "=" * 80)
    print("All results saved to:", args.output)
    print("=" * 80)

    sys.exit(0 if not errors else 1)


def _print_result_summary(result: dict) -> None:
    """Print one-line summary for a completed image."""
    m = result["metrics"]
    print(f"    Edges: {m['num_edge_paths']}, Hatches: {m['num_hatch_paths']}, "
          f"Coverage: {m['coverage_black'] * 100:.1f}%, "
          f"Travel: {m['travel_distance_mm']:.1f}mm")


if __name__ == "__main__":
    main()
