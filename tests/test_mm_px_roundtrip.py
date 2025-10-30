"""Test mm ↔ px coordinate conversion roundtrip.

Tests for src.utils.compute:
    - mm → px → mm is identity (within floating-point tolerance)
    - Edge cases: (0,0), (work_area_mm.x, work_area_mm.y)
    - Large coordinate arrays (vectorized)
    - Different work areas and resolutions

Test cases:
    - test_mm_to_px_to_mm_identity()
    - test_px_to_mm_to_px_identity()
    - test_corner_coordinates()
    - test_vectorized_conversion()
    - test_different_resolutions()

Coordinate frames:
    - image: top-left origin, +Y down
    - machine: bottom-left origin, +Y up

Test both image_mm_to_machine_mm() and inverse.

Tolerance:
    - Absolute: ≤ 1e-4 mm (sub-micron)
    - Relative: ≤ 1e-6 (for large coords)

Run:
    pytest tests/test_mm_px_roundtrip.py -v
"""

