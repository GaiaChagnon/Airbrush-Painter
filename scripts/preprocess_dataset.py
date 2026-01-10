#!/usr/bin/env python3
"""Preprocess raw images for training (crop-to-fit strategy).

Converts images to training-ready format:
    1. Auto-detects orientation (landscape/portrait) from source
    2. Scales to cover target dimensions
    3. Center-crops to exact size (RGB only, no alpha)
    4. Automatically skips duplicates via manifest

Supports two modes:
    1. Batch mode (default): Process raw_images/{easy,medium,hard}/ -> cmy_only/{easy,medium,hard}/
    2. Single mode: Process one input directory to one output directory

Usage:
    # Batch mode: process all difficulty levels
    python scripts/preprocess_dataset.py

    # Single directory mode
    python scripts/preprocess_dataset.py --single data/raw_images/medium/ data/target_images/cmy_only/medium/
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.preprocess import (
    DEFAULT_TARGET_SIZE,
    DIFFICULTY_LEVELS,
    TARGET_LONG_EDGE,
    TARGET_SHORT_EDGE,
    preprocess_all_datasets,
    preprocess_dataset,
)
from src.utils.logging_config import setup_logging

# Default paths
DEFAULT_RAW_ROOT = Path("data/raw_images")
DEFAULT_TARGET_ROOT = Path("data/target_images/cmy_only")


def main() -> int:
    """CLI entrypoint for dataset preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw images for AI airbrush training (crop-to-fit)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Batch Mode (default):
  Processes all difficulty levels automatically:
    data/raw_images/easy/    -> data/target_images/cmy_only/easy/
    data/raw_images/medium/  -> data/target_images/cmy_only/medium/
    data/raw_images/hard/    -> data/target_images/cmy_only/hard/

Orientation Detection:
  - Landscape source (width > height) -> {TARGET_LONG_EDGE}x{TARGET_SHORT_EDGE} target
  - Portrait source (height > width)  -> {TARGET_SHORT_EDGE}x{TARGET_LONG_EDGE} target

Duplicate Detection:
  - SHA-256 hash of source and output
  - Manifest stored per output directory
  - Re-runs safely skip existing images

Examples:
  # Process all difficulty levels (recommended)
  python scripts/preprocess_dataset.py

  # Process single directory
  python scripts/preprocess_dataset.py --single data/raw_images/medium/ data/target_images/cmy_only/medium/

  # Force fixed orientation (all to portrait)
  python scripts/preprocess_dataset.py --single data/raw_images/medium/ data/target_images/cmy_only/medium/ --no-auto-orient
""",
    )

    # Mode selection
    parser.add_argument(
        "--single",
        action="store_true",
        help="Single directory mode (requires input_dir and output_dir)",
    )

    # Paths for single mode
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        help="Input directory (required for --single mode)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="Output directory (required for --single mode)",
    )

    # Custom roots for batch mode
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Raw images root for batch mode (default: {DEFAULT_RAW_ROOT})",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=DEFAULT_TARGET_ROOT,
        help=f"Target images root for batch mode (default: {DEFAULT_TARGET_ROOT})",
    )

    # Orientation control
    parser.add_argument(
        "--no-auto-orient",
        action="store_true",
        help="Disable auto-orientation. Force all images to --width x --height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_TARGET_SIZE[0],
        help=f"Target width when --no-auto-orient (default: {DEFAULT_TARGET_SIZE[0]})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_TARGET_SIZE[1],
        help=f"Target height when --no-auto-orient (default: {DEFAULT_TARGET_SIZE[1]})",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)

    auto_orient = not args.no_auto_orient

    if args.single:
        # Single directory mode
        if args.input_dir is None or args.output_dir is None:
            print("Error: --single mode requires input_dir and output_dir", file=sys.stderr)
            return 1

        if not args.input_dir.exists():
            print(f"Error: Input directory does not exist: {args.input_dir}", file=sys.stderr)
            return 1

        if not args.input_dir.is_dir():
            print(f"Error: Input path is not a directory: {args.input_dir}", file=sys.stderr)
            return 1

        target_size = (args.width, args.height) if not auto_orient else None
        stats = preprocess_dataset(
            args.input_dir,
            args.output_dir,
            target_size=target_size,
            auto_orient=auto_orient,
        )
        return 1 if stats.failed > 0 else 0

    else:
        # Batch mode - process all difficulty levels
        raw_root = args.raw_root
        target_root = args.target_root

        if not raw_root.exists():
            print(f"Error: Raw images root does not exist: {raw_root}", file=sys.stderr)
            print(f"Expected structure: {raw_root}/{{easy,medium,hard}}/", file=sys.stderr)
            return 1

        # Check if any difficulty folders exist
        existing_levels = [
            level for level in DIFFICULTY_LEVELS
            if (raw_root / level).exists()
        ]

        if not existing_levels:
            print(f"Error: No difficulty level folders found in {raw_root}", file=sys.stderr)
            print(f"Expected: {raw_root}/easy/, {raw_root}/medium/, {raw_root}/hard/", file=sys.stderr)
            return 1

        all_stats = preprocess_all_datasets(
            raw_root,
            target_root,
            auto_orient=auto_orient,
        )

        # Return non-zero if any failures
        total_failed = sum(s.failed for s in all_stats.values())
        return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
