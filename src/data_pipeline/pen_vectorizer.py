"""Pen mask vectorization: binary mask → centerline paths.

Converts binary pen mask to simplified vector paths:
    1. Potrace (pypotrace or CLI): Binary → outline curves
    2. Ramer-Douglas-Peucker simplification: Reduce points within tolerance
    3. Optional path ordering: Nearest-neighbor for efficient plotting
    4. Save to YAML (pen_vectors.v1 schema): List of [[x,y], ...] paths

Public API:
    vectorize_mask_to_yaml(pen_mask, output_yaml_path,
                           simplification_tolerance=1.5)
        → None

Output format (pen_vectors.v1.yaml):
    paths:
      - [[x1,y1], [x2,y2], ...]
      - [[x3,y3], [x4,y4], ...]

Coordinates in pixels (relative to render_px).
Conversion to mm happens in gcode_generator via px_to_mm().

Used by:
    - preprocess.py: Automatic pen layer extraction
    - paint.py: Load pen paths for G-code generation
    - GUI: Visualize pen layer overlays
"""

