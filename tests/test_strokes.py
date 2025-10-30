"""Test stroke ID, serialization, and visualization.

Tests for src.utils.strokes:
    - Stable stroke IDs (epoch, idx → unique string)
    - Bidirectional conversion: vec ↔ YAML roundtrip
    - Bounding box calculation
    - Heatmap generation

Test cases:
    - test_make_stroke_id()
    - test_stroke_vec_to_yaml_dict()
    - test_stroke_yaml_dict_to_vec()
    - test_stroke_vec_yaml_roundtrip()
    - test_stroke_bbox_mm()
    - test_strokes_heatmap_mm()

Roundtrip test:
    - Generate random stroke vector (15-D, valid ranges)
    - Convert to YAML dict
    - Convert back to vector
    - Assert: vectors are identical (within float tolerance)

Bbox test:
    - Known Bézier → expected bbox

Run:
    pytest tests/test_strokes.py -v
"""

