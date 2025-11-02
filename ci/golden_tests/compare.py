"""Golden test comparison script for CI.

Runs paint.py on golden images and validates output quality:
    - Loads expected tolerances from YAML (lpips_max, psnr_min, strokes_max)
    - Runs paint_main() on each golden image
    - Computes LPIPS, PSNR, SSIM at reward_px
    - Checks stroke count
    - Saves diff images and report to outputs/ci/

CLI:
    python ci/golden_tests/compare.py --golden ci/golden_tests/images/g1.png
    python ci/golden_tests/compare.py --all  # All golden images

Smoke test (CI per-push):
    - Run 1 golden image (fast)
    - Fail if out of tolerance

Full test (CI nightly):
    - Run all 5 golden images
    - Generate HTML report with diffs
    - Archive to outputs/ci/

Expected format (ci/golden_tests/expected/g1.yaml):
    image: "ci/golden_tests/images/g1.png"
    resolutions:
      render_px: {w: 908, h: 1280}
      reward_px: {w: 908, h: 1280}
    tolerances:
      lpips_max: 0.085
      psnr_min: 26.0
      strokes_max: 1200
    renderer:
      technician_steps: 12
      precision: "bf16"
      lpips_tile: {size: 0, overlap: 0}

Report includes:
    - Target, final canvas, diff (side-by-side)
    - Metrics table (LPIPS, PSNR, SSIM, stroke count)
    - Pass/fail status per test
    - Overall summary

Exit codes:
    0: All tests passed
    1: One or more tests failed
"""

