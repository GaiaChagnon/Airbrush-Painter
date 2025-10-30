"""Stroke ID generation, serialization, and visualization.

Provides:
    - Stable stroke IDs: make_stroke_id(epoch, idx) → "00012-00345-ab12cd34"
    - Bidirectional conversion: stroke_vec_to_yaml_dict() ↔ stroke_yaml_dict_to_vec()
    - Bounding box: stroke_bbox_mm() for tile culling
    - Visualization: strokes_heatmap_mm() for overpaint analysis

Critical for:
    - Training monitor: Stroke playback in GUI (loaded via src.utils.fs)
    - G-code generation: Embedding stroke IDs as comments
    - Debugging: Overpaint heatmaps identify redundant strokes

Stroke vector format (15-D, mm-space):
    [x1, y1, x2, y2, x3, y3, x4, y4, z0, z1, v0, v1, c, m, y]

YAML format (stroke.v1.yaml schema):
    id, bezier{p1,p2,p3,p4}, z_profile{z0,z1}, speed_profile{v0,v1}, color_cmy{c,m,y}

Single source of truth: stroke.v1.yaml schema defines bounds and units.
"""

