"""Test color space conversions and perceptual metrics.

Tests for src.utils.color:
    - sRGB ↔ linear RGB roundtrip
    - RGB → Lab conversion correctness
    - ΔE2000 known pairs
    - Luminance calculation

Test cases:
    - test_srgb_linear_roundtrip()
    - test_linear_srgb_roundtrip()
    - test_rgb_to_lab_known_values()
    - test_delta_e2000_identical_zero()
    - test_delta_e2000_known_pairs()
    - test_luminance_calculation()

Known values (D65, 2° observer):
    - RGB(1,1,1) → Lab(100, 0, 0)
    - RGB(0,0,0) → Lab(0, 0, 0)
    - RGB(1,0,0) → Lab(~53, ~80, ~67)

ΔE2000 test pairs (within JND threshold):
    - Same color: ΔE = 0
    - Small shift: ΔE < 2.3 (just noticeable difference)

Run:
    pytest tests/test_color.py -v
"""

