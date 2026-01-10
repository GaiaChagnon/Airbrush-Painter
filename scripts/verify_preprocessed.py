#!/usr/bin/env python3
"""Verify preprocessed images are valid for training.

Checks:
    - Correct dimensions (1280x908 landscape or 908x1280 portrait)
    - RGB format (3 channels, no alpha)
    - File integrity

Usage:
    python scripts/verify_preprocessed.py data/target_images/cmy_only/medium/
"""
import sys
from pathlib import Path

import cv2

# Expected dimensions
TARGET_LONG_EDGE = 1280
TARGET_SHORT_EDGE = 908


def verify_preprocessed_directory(directory: Path) -> dict:
    """Verify all images in a preprocessed directory are valid.

    Parameters
    ----------
    directory : Path
        Directory containing preprocessed images

    Returns
    -------
    dict
        Verification results with counts and any errors
    """
    results = {
        "total": 0,
        "valid": 0,
        "landscape": 0,
        "portrait": 0,
        "errors": [],
    }

    expected_landscape = (TARGET_LONG_EDGE, TARGET_SHORT_EDGE)  # 1280x908
    expected_portrait = (TARGET_SHORT_EDGE, TARGET_LONG_EDGE)  # 908x1280

    for img_path in sorted(directory.glob("*.png")):
        if img_path.name.startswith("."):
            continue  # Skip hidden files

        results["total"] += 1

        # Load with UNCHANGED to check channels
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            results["errors"].append(f"{img_path.name}: Failed to load")
            continue

        h, w = img.shape[:2]
        channels = img.shape[2] if img.ndim == 3 else 1

        # Check RGB (3 channels)
        if channels != 3:
            results["errors"].append(
                f"{img_path.name}: Expected 3 channels (RGB), got {channels}"
            )
            continue

        # Check dimensions
        if (w, h) == expected_landscape:
            results["landscape"] += 1
            results["valid"] += 1
        elif (w, h) == expected_portrait:
            results["portrait"] += 1
            results["valid"] += 1
        else:
            results["errors"].append(
                f"{img_path.name}: Unexpected size {w}x{h}, "
                f"expected {expected_landscape[0]}x{expected_landscape[1]} (landscape) "
                f"or {expected_portrait[0]}x{expected_portrait[1]} (portrait)"
            )

    return results


def main() -> int:
    """CLI entrypoint."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_preprocessed.py <preprocessed_directory>")
        print("Example: python scripts/verify_preprocessed.py data/target_images/cmy_only/medium/")
        return 1

    directory = Path(sys.argv[1])
    if not directory.exists():
        print(f"Error: Directory does not exist: {directory}")
        return 1

    print(f"Verifying preprocessed images in: {directory}")
    results = verify_preprocessed_directory(directory)

    print(f"\nResults:")
    print(f"  Total images: {results['total']}")
    print(f"  Valid: {results['valid']}")
    print(f"  Landscape (1280x908): {results['landscape']}")
    print(f"  Portrait (908x1280): {results['portrait']}")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for err in results["errors"]:
            print(f"  - {err}")
        return 1
    else:
        print("\nAll images valid!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

