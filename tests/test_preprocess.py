"""Tests for image preprocessing pipeline.

Verifies:
    - Correct output dimensions (landscape vs portrait)
    - RGB format (no alpha channel)
    - File integrity and format
    - Duplicate detection via manifest
    - Orientation auto-detection

Run: pytest tests/test_preprocess.py -v
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data_pipeline.preprocess import (
    MANIFEST_FILENAME,
    TARGET_LONG_EDGE,
    TARGET_SHORT_EDGE,
    detect_orientation,
    get_target_size_for_orientation,
    load_manifest,
    preprocess_dataset,
    preprocess_image,
    save_manifest,
)
from src.utils import fs


class TestOrientationDetection:
    """Tests for orientation detection logic."""

    def test_landscape_wider_than_tall(self):
        """Landscape: width > height."""
        assert detect_orientation(1920, 1080) == "landscape"
        assert detect_orientation(1280, 720) == "landscape"

    def test_portrait_taller_than_wide(self):
        """Portrait: height > width."""
        assert detect_orientation(1080, 1920) == "portrait"
        assert detect_orientation(720, 1280) == "portrait"

    def test_square_is_landscape(self):
        """Square images default to landscape (width >= height)."""
        assert detect_orientation(1000, 1000) == "landscape"

    def test_target_size_landscape(self):
        """Landscape target is 1280x908."""
        w, h = get_target_size_for_orientation("landscape")
        assert w == TARGET_LONG_EDGE  # 1280
        assert h == TARGET_SHORT_EDGE  # 908

    def test_target_size_portrait(self):
        """Portrait target is 908x1280."""
        w, h = get_target_size_for_orientation("portrait")
        assert w == TARGET_SHORT_EDGE  # 908
        assert h == TARGET_LONG_EDGE  # 1280


class TestPreprocessImage:
    """Tests for single image preprocessing."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def landscape_image(self, temp_dir):
        """Create a synthetic landscape image (1920x1080)."""
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        path = temp_dir / "landscape_test.png"
        cv2.imwrite(str(path), img)
        return path

    @pytest.fixture
    def portrait_image(self, temp_dir):
        """Create a synthetic portrait image (1080x1920)."""
        img = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
        path = temp_dir / "portrait_test.png"
        cv2.imwrite(str(path), img)
        return path

    @pytest.fixture
    def rgba_image(self, temp_dir):
        """Create a synthetic RGBA image (with alpha channel)."""
        img = np.random.randint(0, 255, (1080, 1920, 4), dtype=np.uint8)
        path = temp_dir / "rgba_test.png"
        cv2.imwrite(str(path), img)
        return path

    def test_landscape_auto_orient(self, landscape_image, temp_dir):
        """Landscape source -> landscape target (1280x908)."""
        output_dir = temp_dir / "output"
        output_path, _, _, orientation = preprocess_image(
            landscape_image, output_dir, auto_orient=True
        )

        assert output_path is not None
        assert orientation == "landscape"

        # Verify output dimensions
        img = cv2.imread(str(output_path))
        h, w = img.shape[:2]
        assert w == TARGET_LONG_EDGE  # 1280
        assert h == TARGET_SHORT_EDGE  # 908

    def test_portrait_auto_orient(self, portrait_image, temp_dir):
        """Portrait source -> portrait target (908x1280)."""
        output_dir = temp_dir / "output"
        output_path, _, _, orientation = preprocess_image(
            portrait_image, output_dir, auto_orient=True
        )

        assert output_path is not None
        assert orientation == "portrait"

        # Verify output dimensions
        img = cv2.imread(str(output_path))
        h, w = img.shape[:2]
        assert w == TARGET_SHORT_EDGE  # 908
        assert h == TARGET_LONG_EDGE  # 1280

    def test_output_is_rgb_no_alpha(self, landscape_image, temp_dir):
        """Output must be RGB (3 channels), no alpha."""
        output_dir = temp_dir / "output"
        output_path, _, _, _ = preprocess_image(
            landscape_image, output_dir, auto_orient=True
        )

        # Load with UNCHANGED to check channel count
        img = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        assert img.ndim == 3
        assert img.shape[2] == 3, f"Expected 3 channels (RGB), got {img.shape[2]}"

    def test_rgba_source_stripped_to_rgb(self, rgba_image, temp_dir):
        """RGBA source images have alpha stripped in output."""
        output_dir = temp_dir / "output"
        output_path, _, _, _ = preprocess_image(
            rgba_image, output_dir, auto_orient=True
        )

        # Load with UNCHANGED to check channel count
        img = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        assert img.shape[2] == 3, f"Alpha not stripped: got {img.shape[2]} channels"

    def test_forced_orientation(self, landscape_image, temp_dir):
        """Force landscape to portrait target."""
        output_dir = temp_dir / "output"
        output_path, _, _, orientation = preprocess_image(
            landscape_image,
            output_dir,
            target_size=(908, 1280),  # Force portrait
            auto_orient=False,
        )

        assert output_path is not None
        assert orientation is None  # No auto-detection

        # Verify forced dimensions
        img = cv2.imread(str(output_path))
        h, w = img.shape[:2]
        assert w == 908
        assert h == 1280


class TestDuplicateDetection:
    """Tests for manifest-based duplicate detection."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def test_image(self, temp_dir):
        """Create a test image."""
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        path = temp_dir / "test.png"
        cv2.imwrite(str(path), img)
        return path

    def test_first_process_creates_manifest(self, test_image, temp_dir):
        """First preprocessing creates manifest file."""
        output_dir = temp_dir / "output"
        preprocess_image(test_image, output_dir, auto_orient=True)

        # Manifest should NOT exist after single image (saved in batch)
        # But we can manually save and verify
        manifest_path = output_dir / MANIFEST_FILENAME
        assert not manifest_path.exists()  # Single image doesn't save manifest

    def test_duplicate_source_skipped(self, test_image, temp_dir):
        """Same source file is skipped on second run."""
        output_dir = temp_dir / "output"

        # First run
        out1, hash1, src_hash1, _ = preprocess_image(
            test_image, output_dir, auto_orient=True
        )
        assert out1 is not None

        # Create manifest manually (simulating batch save)
        manifest = {
            hash1: {
                "source_path": str(test_image),
                "source_hash": src_hash1,
                "output_name": out1.name,
            }
        }
        save_manifest(output_dir, manifest)

        # Second run with same source
        out2, _, _, _ = preprocess_image(
            test_image, output_dir, manifest=manifest, auto_orient=True
        )
        assert out2 is None  # Skipped as duplicate

    def test_manifest_load_save(self, temp_dir):
        """Manifest load/save round-trip."""
        output_dir = temp_dir / "output"
        fs.ensure_dir(output_dir)

        manifest = {
            "abc123": {
                "source_path": "/path/to/image.jpg",
                "source_hash": "def456",
                "output_name": "image.png",
            }
        }

        save_manifest(output_dir, manifest)
        loaded = load_manifest(output_dir)

        assert loaded == manifest


class TestPreprocessDataset:
    """Tests for batch dataset preprocessing."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test inputs/outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mixed_dataset(self, temp_dir):
        """Create dataset with landscape and portrait images."""
        input_dir = temp_dir / "raw"
        input_dir.mkdir()

        # Landscape images
        for i in range(2):
            img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"landscape_{i}.png"), img)

        # Portrait images
        for i in range(3):
            img = np.random.randint(0, 255, (1280, 720, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"portrait_{i}.png"), img)

        return input_dir

    def test_batch_processes_all(self, mixed_dataset, temp_dir):
        """Batch preprocessing processes all images."""
        output_dir = temp_dir / "output"
        stats = preprocess_dataset(mixed_dataset, output_dir, auto_orient=True)

        assert stats.total == 5
        assert stats.processed == 5
        assert stats.skipped_duplicate == 0
        assert stats.failed == 0

    def test_batch_creates_manifest(self, mixed_dataset, temp_dir):
        """Batch preprocessing creates manifest."""
        output_dir = temp_dir / "output"
        preprocess_dataset(mixed_dataset, output_dir, auto_orient=True)

        manifest_path = output_dir / MANIFEST_FILENAME
        assert manifest_path.exists()

        manifest = load_manifest(output_dir)
        assert len(manifest) == 5

    def test_rerun_skips_all(self, mixed_dataset, temp_dir):
        """Re-running skips all as duplicates."""
        output_dir = temp_dir / "output"

        # First run
        stats1 = preprocess_dataset(mixed_dataset, output_dir, auto_orient=True)
        assert stats1.processed == 5

        # Second run
        stats2 = preprocess_dataset(mixed_dataset, output_dir, auto_orient=True)
        assert stats2.processed == 0
        assert stats2.skipped_duplicate == 5


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

    Example
    -------
    >>> results = verify_preprocessed_directory(Path("data/target_images/cmy_only/medium/"))
    >>> assert results["errors"] == []
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
                f"{img_path.name}: Expected 3 channels, got {channels}"
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
                f"expected {expected_landscape} or {expected_portrait}"
            )

    return results


# Standalone verification for existing datasets
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path for standalone execution
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data_pipeline.preprocess import TARGET_LONG_EDGE, TARGET_SHORT_EDGE

    if len(sys.argv) < 2:
        print("Usage: python test_preprocess.py <preprocessed_directory>")
        print("Example: python test_preprocess.py data/target_images/cmy_only/medium/")
        sys.exit(1)

    directory = Path(sys.argv[1])
    if not directory.exists():
        print(f"Error: Directory does not exist: {directory}")
        sys.exit(1)

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
        sys.exit(1)
    else:
        print("\nAll images valid!")
        sys.exit(0)

