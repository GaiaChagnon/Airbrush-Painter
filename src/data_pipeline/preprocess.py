"""Target image preprocessing: crop-to-fit resizing with duplicate detection.

Converts user-provided images into training-ready targets:
    1. Load raw image (PNG/JPEG)
    2. Auto-detect orientation (landscape/portrait) from source
    3. Scale to cover target dimensions (crop-to-fit strategy)
    4. Center crop to exact target size
    5. Strip alpha channel (always output RGB)
    6. Detect duplicates via SHA-256 manifest
    7. Save to output directory with atomic writes

Public API:
    preprocess_image(raw_image_path, output_dir, base_size, auto_orient)
        -> Optional[Path] (None if duplicate)
    preprocess_dataset(input_dir, output_dir, base_size, auto_orient)
        -> PreprocessingStats

Orientation detection:
    - If auto_orient=True (default): detects source orientation and matches target
      - Landscape source (w > h) -> landscape target (1280x908)
      - Portrait source (h > w) -> portrait target (908x1280)
    - If auto_orient=False: forces all images to specified target_size

Duplicate detection:
    Uses a manifest file (.preprocessed_manifest.yaml) in each output directory.
    Tracks both source file hash and output image hash to detect:
    - Same source file processed again (skips)
    - Different source producing identical output (skips)

Outputs:
    - Target images: RGB PNG (no alpha), exact target resolution
    - Manifest: YAML file with source/output hash mappings

Used by:
    - Training: Preprocess entire dataset before training
    - Inference: Prepare custom targets
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Set, Tuple, Union

import cv2
import numpy as np

from src.utils import fs, hashing
from src.utils.logging_config import setup_logging

# Manifest filename (hidden file in output directory)
MANIFEST_FILENAME = ".preprocessed_manifest.yaml"

# Supported image extensions (case-insensitive)
SUPPORTED_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

# Base dimensions for A4-equivalent canvas
# Portrait: 908x1280 (width x height), Landscape: 1280x908
TARGET_SHORT_EDGE = 908
TARGET_LONG_EDGE = 1280

# Default target size (A4 portrait at 300 DPI equivalent)
DEFAULT_TARGET_SIZE: Tuple[int, int] = (908, 1280)  # (width, height)

# Orientation type
Orientation = Literal["portrait", "landscape"]

logger = logging.getLogger(__name__)


def detect_orientation(width: int, height: int) -> Orientation:
    """Detect image orientation from dimensions.

    Parameters
    ----------
    width : int
        Image width in pixels
    height : int
        Image height in pixels

    Returns
    -------
    Orientation
        "landscape" if width >= height, "portrait" otherwise
    """
    return "landscape" if width >= height else "portrait"


def get_target_size_for_orientation(orientation: Orientation) -> Tuple[int, int]:
    """Get target dimensions for given orientation.

    Parameters
    ----------
    orientation : Orientation
        "landscape" or "portrait"

    Returns
    -------
    Tuple[int, int]
        (width, height) in pixels
    """
    if orientation == "landscape":
        return (TARGET_LONG_EDGE, TARGET_SHORT_EDGE)  # 1280x908
    else:
        return (TARGET_SHORT_EDGE, TARGET_LONG_EDGE)  # 908x1280


@dataclass
class PreprocessingStats:
    """Statistics from a preprocessing run.

    Attributes
    ----------
    processed : int
        Number of images successfully preprocessed and saved
    skipped_duplicate : int
        Number of images skipped (already in manifest)
    failed : int
        Number of images that failed to process
    total : int
        Total input images found
    """

    processed: int = 0
    skipped_duplicate: int = 0
    failed: int = 0
    total: int = 0


def load_manifest(output_dir: Path) -> Dict[str, dict]:
    """Load preprocessing manifest from output directory.

    Parameters
    ----------
    output_dir : Path
        Directory containing manifest file

    Returns
    -------
    Dict[str, dict]
        Manifest mapping output_hash -> {source_path, source_hash, output_name, timestamp}
        Empty dict if manifest doesn't exist
    """
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}

    try:
        data = fs.load_yaml(manifest_path)
        return data if data else {}
    except Exception as e:
        logger.warning("Failed to load manifest %s: %s. Starting fresh.", manifest_path, e)
        return {}


def save_manifest(output_dir: Path, manifest: Dict[str, dict]) -> None:
    """Save preprocessing manifest atomically.

    Parameters
    ----------
    output_dir : Path
        Directory to save manifest
    manifest : Dict[str, dict]
        Manifest data to save
    """
    manifest_path = output_dir / MANIFEST_FILENAME
    fs.atomic_yaml_dump(manifest, manifest_path)


def _get_source_hashes(manifest: Dict[str, dict]) -> Set[str]:
    """Extract all source hashes from manifest for fast lookup."""
    return {entry.get("source_hash", "") for entry in manifest.values()}


def _get_output_hashes(manifest: Dict[str, dict]) -> Set[str]:
    """Get all output hashes (manifest keys) as a set."""
    return set(manifest.keys())


def is_duplicate(
    output_hash: str,
    source_hash: str,
    manifest: Dict[str, dict],
) -> Tuple[bool, Optional[str]]:
    """Check if image is a duplicate based on manifest.

    Parameters
    ----------
    output_hash : str
        SHA-256 hash of preprocessed image
    source_hash : str
        SHA-256 hash of source file
    manifest : Dict[str, dict]
        Current manifest

    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_duplicate, reason) where reason explains why it's a duplicate
    """
    # Check output hash (identical preprocessed result)
    if output_hash in manifest:
        existing = manifest[output_hash]
        return True, f"identical output to {existing.get('output_name', 'unknown')}"

    # Check source hash (same source file already processed)
    source_hashes = _get_source_hashes(manifest)
    if source_hash in source_hashes:
        # Find the entry with matching source hash
        for entry in manifest.values():
            if entry.get("source_hash") == source_hash:
                return True, f"source already processed as {entry.get('output_name', 'unknown')}"

    return False, None


def preprocess_image(
    input_path: Path,
    output_dir: Path,
    target_size: Optional[Tuple[int, int]] = None,
    manifest: Optional[Dict[str, dict]] = None,
    auto_orient: bool = True,
    output_name: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[str], Optional[str], Optional[Orientation]]:
    """Preprocess a single image using crop-to-fit strategy.

    Scales image to cover target dimensions, then center-crops to exact size.
    Auto-detects orientation from source image to preserve landscape/portrait.

    Parameters
    ----------
    input_path : Path
        Path to raw image (PNG/JPEG/WEBP/BMP/TIFF)
    output_dir : Path
        Directory to save processed image
    target_size : Optional[Tuple[int, int]]
        (width, height) in pixels. Ignored if auto_orient=True.
    manifest : Optional[Dict[str, dict]]
        Existing manifest for duplicate detection (if None, no dup check)
    auto_orient : bool
        If True (default), detect source orientation and use matching target size.
        If False, use target_size (or DEFAULT_TARGET_SIZE if None).
    output_name : Optional[str]
        Custom output filename (without extension). If None, uses input stem.

    Returns
    -------
    Tuple[Optional[Path], Optional[str], Optional[str], Optional[Orientation]]
        (output_path, output_hash, source_hash, orientation)
        Returns (None, None, source_hash, None) if duplicate/failed

    Raises
    ------
    ValueError
        If image fails to load
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # Compute source hash first for duplicate detection
    source_hash = hashing.sha256_file(input_path)

    # Load image - use IMREAD_COLOR to force 3-channel BGR (strips alpha)
    img_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {input_path}")

    # Convert to RGB for processing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]

    # Determine target size based on orientation
    if auto_orient:
        orientation = detect_orientation(w_orig, h_orig)
        target_w, target_h = get_target_size_for_orientation(orientation)
    else:
        orientation = None
        if target_size is None:
            target_size = DEFAULT_TARGET_SIZE
        target_w, target_h = target_size

    logger.debug(
        "Source: %dx%d (%s) -> Target: %dx%d",
        w_orig, h_orig,
        orientation or "forced",
        target_w, target_h
    )

    # Calculate scaling to COVER target dimensions (may crop edges)
    # Use ceiling to ensure we always have enough pixels to crop from
    scale = max(target_w / w_orig, target_h / h_orig)
    new_w = max(target_w, int(round(w_orig * scale)))
    new_h = max(target_h, int(round(h_orig * scale)))

    # Resize with appropriate interpolation
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=interp)

    # Center crop to exact target dimensions
    crop_x = max(0, (new_w - target_w) // 2)
    crop_y = max(0, (new_h - target_h) // 2)
    target_img = resized[crop_y : crop_y + target_h, crop_x : crop_x + target_w]

    # Verify output dimensions and ensure 3 channels (RGB, no alpha)
    if target_img.shape[:2] != (target_h, target_w):
        raise ValueError(
            f"Output shape mismatch: got {target_img.shape[:2]}, expected {(target_h, target_w)}. "
            f"Original: {w_orig}x{h_orig}, scaled: {new_w}x{new_h}, crop: ({crop_x}, {crop_y})"
        )
    if target_img.ndim != 3 or target_img.shape[2] != 3:
        raise ValueError(
            f"Output must be RGB (3 channels), got shape {target_img.shape}"
        )

    # Compute output hash for duplicate detection
    output_hash = hashing.sha256_numpy_image(target_img)

    # Check for duplicates if manifest provided
    if manifest is not None:
        is_dup, reason = is_duplicate(output_hash, source_hash, manifest)
        if is_dup:
            logger.debug("Skipping %s: %s", input_path.name, reason)
            return None, None, source_hash, None

    # Ensure output directory exists
    fs.ensure_dir(output_dir)

    # Determine output filename
    stem = output_name if output_name else input_path.stem
    output_path = output_dir / f"{stem}.png"
    
    # Save with atomic write (convert back to BGR for OpenCV)
    target_img_bgr = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)

    # Use atomic write via temp file
    tmp_path = output_path.with_suffix(".tmp.png")
    try:
        cv2.imwrite(str(tmp_path), target_img_bgr)
        tmp_path.replace(output_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to save {output_path}: {e}") from e

    return output_path, output_hash, source_hash, orientation


def preprocess_dataset(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    auto_orient: bool = True,
    name_prefix: Optional[str] = None,
    start_index: int = 1,
) -> PreprocessingStats:
    """Preprocess all images in a directory using crop-to-fit.

    Auto-detects orientation (landscape/portrait) from each source image
    and uses appropriate target dimensions. Skips duplicates via manifest.

    Parameters
    ----------
    input_dir : Union[str, Path]
        Directory containing raw images
    output_dir : Union[str, Path]
        Directory to save processed images
    target_size : Optional[Tuple[int, int]]
        (width, height) in pixels. Ignored if auto_orient=True.
    auto_orient : bool
        If True (default), detect source orientation and use matching target:
        - Landscape (w > h) -> 1280x908
        - Portrait (h > w) -> 908x1280
    name_prefix : Optional[str]
        Prefix for standardized naming (e.g., "hard" -> "hard-0001.png").
        If None, uses original filename.
    start_index : int
        Starting index for sequential numbering, default 1.

    Returns
    -------
    PreprocessingStats
        Statistics about the preprocessing run
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all supported image files
    input_files = [
        f
        for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    stats = PreprocessingStats(total=len(input_files))

    if not input_files:
        logger.warning("No images found in %s", input_dir)
        return stats

    if auto_orient:
        logger.info(
            "Found %d images to preprocess. Auto-orientation enabled. "
            "Landscape -> %dx%d, Portrait -> %dx%d. Strategy: crop-to-fit",
            len(input_files),
            TARGET_LONG_EDGE, TARGET_SHORT_EDGE,
            TARGET_SHORT_EDGE, TARGET_LONG_EDGE,
        )
    else:
        size = target_size or DEFAULT_TARGET_SIZE
        logger.info(
            "Found %d images to preprocess. Target: %dx%d (WxH). Strategy: crop-to-fit",
            len(input_files),
            size[0], size[1],
        )

    # Load existing manifest
    manifest = load_manifest(output_dir)
    logger.debug("Loaded manifest with %d existing entries", len(manifest))

    # Track orientation counts
    landscape_count = 0
    portrait_count = 0

    # Sequential index for standardized naming
    next_index = start_index

    # If using standardized naming, find the highest existing index
    if name_prefix:
        for entry in manifest.values():
            out_name = entry.get("output_name", "")
            if out_name.startswith(name_prefix + "-"):
                try:
                    idx = int(out_name.split("-")[1].split(".")[0])
                    next_index = max(next_index, idx + 1)
                except (ValueError, IndexError):
                    pass

    for i, input_path in enumerate(sorted(input_files), 1):
        try:
            # Generate standardized name if prefix provided
            output_name = None
            if name_prefix:
                output_name = f"{name_prefix}-{next_index:04d}"

            output_path, output_hash, source_hash, orientation = preprocess_image(
                input_path, output_dir, target_size, manifest, auto_orient, output_name
            )

            if output_path is None:
                # Duplicate detected
                stats.skipped_duplicate += 1
                logger.info("[%d/%d] %s -> SKIPPED (duplicate)", i, stats.total, input_path.name)
            else:
                # Successfully processed
                stats.processed += 1
                next_index += 1  # Increment for next image

                # Track orientation
                if orientation == "landscape":
                    landscape_count += 1
                elif orientation == "portrait":
                    portrait_count += 1

                # Update manifest with orientation info
                manifest[output_hash] = {
                    "source_path": str(input_path),
                    "source_hash": source_hash,
                    "output_name": output_path.name,
                    "orientation": orientation,
                    "timestamp": datetime.now().isoformat(),
                }

                orient_str = f" [{orientation}]" if orientation else ""
                logger.info(
                    "[%d/%d] %s -> %s%s",
                    i, stats.total, input_path.name, output_path.name, orient_str
                )

        except Exception as e:
            stats.failed += 1
            logger.error("[%d/%d] %s -> FAILED: %s", i, stats.total, input_path.name, e)

    # Save updated manifest
    if stats.processed > 0:
        save_manifest(output_dir, manifest)
        logger.debug("Saved manifest with %d entries", len(manifest))

    # Log summary
    logger.info(
        "Preprocessing complete: %d processed (%d landscape, %d portrait), "
        "%d skipped (duplicate), %d failed",
        stats.processed,
        landscape_count,
        portrait_count,
        stats.skipped_duplicate,
        stats.failed,
    )

    return stats


# Difficulty levels for dataset organization
DIFFICULTY_LEVELS = ("easy", "medium", "hard")


def preprocess_all_datasets(
    raw_root: Union[str, Path],
    target_root: Union[str, Path],
    auto_orient: bool = True,
) -> Dict[str, PreprocessingStats]:
    """Preprocess all difficulty-level datasets.

    Scans raw_root/{easy,medium,hard}/ and outputs to target_root/{easy,medium,hard}/.
    Each difficulty level gets its own manifest for duplicate tracking.

    Parameters
    ----------
    raw_root : Union[str, Path]
        Root directory containing raw images organized by difficulty.
        Expected structure: raw_root/{easy,medium,hard}/*.{png,jpg,jpeg}
    target_root : Union[str, Path]
        Root directory for processed outputs.
        Creates: target_root/{easy,medium,hard}/
    auto_orient : bool
        If True (default), auto-detect orientation per image.

    Returns
    -------
    Dict[str, PreprocessingStats]
        Stats per difficulty level: {"easy": stats, "medium": stats, "hard": stats}

    Example
    -------
    >>> stats = preprocess_all_datasets(
    ...     "data/raw_images",
    ...     "data/target_images/cmy_only"
    ... )
    >>> print(f"Easy: {stats['easy'].processed}, Medium: {stats['medium'].processed}")
    """
    raw_root = Path(raw_root)
    target_root = Path(target_root)

    all_stats: Dict[str, PreprocessingStats] = {}
    total_processed = 0
    total_skipped = 0
    total_failed = 0

    logger.info("=" * 60)
    logger.info("Processing all difficulty levels from: %s", raw_root)
    logger.info("Output root: %s", target_root)
    logger.info("=" * 60)

    for level in DIFFICULTY_LEVELS:
        input_dir = raw_root / level
        output_dir = target_root / level

        if not input_dir.exists():
            logger.warning("Skipping %s: directory does not exist", input_dir)
            all_stats[level] = PreprocessingStats()
            continue

        # Check if directory has any images
        image_count = sum(
            1 for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if image_count == 0:
            logger.info("Skipping %s: no images found", level)
            all_stats[level] = PreprocessingStats()
            continue

        logger.info("-" * 40)
        logger.info("Processing [%s] dataset (%d images)", level.upper(), image_count)
        logger.info("-" * 40)

        # Use difficulty level as name prefix for standardized naming
        stats = preprocess_dataset(
            input_dir, output_dir,
            auto_orient=auto_orient,
            name_prefix=level,
        )
        all_stats[level] = stats

        total_processed += stats.processed
        total_skipped += stats.skipped_duplicate
        total_failed += stats.failed

    # Final summary
    logger.info("=" * 60)
    logger.info("ALL DATASETS COMPLETE")
    logger.info("  Total processed: %d", total_processed)
    logger.info("  Total skipped (duplicate): %d", total_skipped)
    logger.info("  Total failed: %d", total_failed)
    for level in DIFFICULTY_LEVELS:
        s = all_stats.get(level, PreprocessingStats())
        logger.info("  [%s] %d processed, %d skipped, %d failed",
                    level, s.processed, s.skipped_duplicate, s.failed)
    logger.info("=" * 60)

    return all_stats
