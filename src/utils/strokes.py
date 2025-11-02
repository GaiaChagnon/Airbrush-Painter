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

import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch


def make_stroke_id(epoch: int, idx: int) -> str:
    """Generate stable, unique stroke ID.

    Parameters
    ----------
    epoch : int
        Training epoch number
    idx : int
        Stroke index within epoch

    Returns
    -------
    str
        Stroke ID format: "EEEEE-IIIII-HHHHHHHH"
        E: epoch (5 digits), I: index (5 digits), H: uuid4 suffix (8 hex chars)

    Examples
    --------
    >>> stroke_id = make_stroke_id(epoch=12, idx=345)
    >>> print(stroke_id)
    00012-00345-a3f5b2c1

    Notes
    -----
    Used in G-code comments: ; STROKE_ID: 00012-00345-...
    Enables tracing from G-code back to training artifacts.
    """
    # Generate short uuid suffix for uniqueness
    uuid_suffix = uuid.uuid4().hex[:8]
    return f"{epoch:05d}-{idx:05d}-{uuid_suffix}"


def stroke_vec_to_yaml_dict(
    vec_mm: torch.Tensor,
    bounds: Dict[str, Tuple[float, float]],
    stroke_id: str = None
) -> Dict:
    """Convert 15-D stroke vector (mm-space) to YAML dictionary with clamping.

    Parameters
    ----------
    vec_mm : torch.Tensor
        Stroke vector, shape (15,), mm-space
        [x1, y1, x2, y2, x3, y3, x4, y4, z0, z1, v0, v1, c, m, y]
    bounds : Dict[str, Tuple[float, float]]
        Schema bounds for clamping: {'x': (min, max), 'y': ..., 'z': ..., 'speed': ..., 'cmy': ...}
        Required to ensure values stay within machine limits
    stroke_id : str, optional
        Stroke ID; auto-generated if None

    Returns
    -------
    Dict
        YAML-compatible dictionary matching stroke.v1.yaml schema

    Notes
    -----
    All coordinates in mm. Clamps values to bounds to prevent drift.
    """
    vec_mm = vec_mm.cpu().detach()
    
    if stroke_id is None:
        stroke_id = make_stroke_id(epoch=0, idx=0)
    
    # Clamp coordinates to bounds
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_min, z_max = bounds['z']
    v_min, v_max = bounds['speed']
    c_min, c_max = bounds['cmy']
    
    def clamp(val: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, val))
    
    return {
        'id': stroke_id,
        'bezier': {
            'p1': [clamp(float(vec_mm[0]), x_min, x_max), clamp(float(vec_mm[1]), y_min, y_max)],
            'p2': [clamp(float(vec_mm[2]), x_min, x_max), clamp(float(vec_mm[3]), y_min, y_max)],
            'p3': [clamp(float(vec_mm[4]), x_min, x_max), clamp(float(vec_mm[5]), y_min, y_max)],
            'p4': [clamp(float(vec_mm[6]), x_min, x_max), clamp(float(vec_mm[7]), y_min, y_max)]
        },
        'z_profile': {
            'z0': clamp(float(vec_mm[8]), z_min, z_max),
            'z1': clamp(float(vec_mm[9]), z_min, z_max)
        },
        'speed_profile': {
            'v0': clamp(float(vec_mm[10]), v_min, v_max),
            'v1': clamp(float(vec_mm[11]), v_min, v_max)
        },
        'color_cmy': {
            'c': clamp(float(vec_mm[12]), c_min, c_max),
            'm': clamp(float(vec_mm[13]), c_min, c_max),
            'y': clamp(float(vec_mm[14]), c_min, c_max)
        }
    }


def stroke_yaml_dict_to_vec(y: Dict) -> torch.Tensor:
    """Convert YAML dictionary to 15-D stroke vector (mm-space).

    Parameters
    ----------
    y : Dict
        Stroke dictionary matching stroke.v1.yaml schema

    Returns
    -------
    torch.Tensor
        Stroke vector, shape (15,), dtype float32

    Raises
    ------
    KeyError
        If required keys missing from dictionary
    """
    try:
        vec = torch.tensor([
            y['bezier']['p1'][0], y['bezier']['p1'][1],
            y['bezier']['p2'][0], y['bezier']['p2'][1],
            y['bezier']['p3'][0], y['bezier']['p3'][1],
            y['bezier']['p4'][0], y['bezier']['p4'][1],
            y['z_profile']['z0'],
            y['z_profile']['z1'],
            y['speed_profile']['v0'],
            y['speed_profile']['v1'],
            y['color_cmy']['c'],
            y['color_cmy']['m'],
            y['color_cmy']['y']
        ], dtype=torch.float32)
        return vec
    except KeyError as e:
        raise KeyError(f"Missing required stroke field: {e}. Check stroke.v1.yaml schema.") from e


def stroke_bbox_mm(stroke_yaml: Dict) -> Tuple[float, float, float, float]:
    """Compute conservative bounding box for stroke in mm.

    Parameters
    ----------
    stroke_yaml : Dict
        Stroke dictionary from stroke.v1.yaml schema

    Returns
    -------
    Tuple[float, float, float, float]
        (xmin, ymin, xmax, ymax) in mm

    Notes
    -----
    Conservative bbox from control points (not true Bézier extrema).
    Sufficient for tile culling (we want to be conservative).
    For true bbox, would need to find curve extrema via derivatives.
    """
    bezier = stroke_yaml['bezier']
    
    x_coords = [
        bezier['p1'][0], bezier['p2'][0],
        bezier['p3'][0], bezier['p4'][0]
    ]
    y_coords = [
        bezier['p1'][1], bezier['p2'][1],
        bezier['p3'][1], bezier['p4'][1]
    ]
    
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)
    
    return (xmin, ymin, xmax, ymax)


def strokes_heatmap_mm(
    strokes_yaml_path: Union[str, Path],
    work_area_mm: Tuple[float, float],
    H: int,
    W: int
) -> torch.Tensor:
    """Generate overpaint heatmap from stroke list.

    Parameters
    ----------
    strokes_yaml_path : Union[str, Path]
        Path to YAML file containing list of strokes
    work_area_mm : Tuple[float, float]
        Work area size (width_mm, height_mm), e.g., (210.0, 297.0)
    H : int
        Heatmap height (pixels)
    W : int
        Heatmap width (pixels)

    Returns
    -------
    torch.Tensor
        Heatmap, shape (H, W), values = stroke count per pixel

    Notes
    -----
    Visualizes overpaint: higher values = more strokes overlapping.
    Used for debugging redundant strokes and policy efficiency.
    
    Simple rasterization: each stroke adds 1 to bbox region.
    For precise rendering, use differentiable_renderer.py.

    Examples
    --------
    >>> heatmap = strokes_heatmap_mm("strokes.yaml", (210, 297), 256, 256)
    >>> plt.imshow(heatmap, cmap='hot')
    """
    from src.utils import fs
    
    strokes_list = fs.load_yaml(strokes_yaml_path)
    
    heatmap = torch.zeros((H, W), dtype=torch.float32)
    
    # Scaling factors mm → px
    sx = W / work_area_mm[0]
    sy = H / work_area_mm[1]
    
    for stroke in strokes_list:
        xmin, ymin, xmax, ymax = stroke_bbox_mm(stroke)
        
        # Convert mm → px
        xmin_px = int(xmin * sx)
        xmax_px = int(xmax * sx)
        ymin_px = int(ymin * sy)
        ymax_px = int(ymax * sy)
        
        # Clamp to image bounds
        xmin_px = max(0, min(W - 1, xmin_px))
        xmax_px = max(0, min(W - 1, xmax_px))
        ymin_px = max(0, min(H - 1, ymin_px))
        ymax_px = max(0, min(H - 1, ymax_px))
        
        # Increment heatmap in bbox region
        heatmap[ymin_px:ymax_px + 1, xmin_px:xmax_px + 1] += 1.0
    
    return heatmap


def filter_strokes_by_bbox(
    strokes_yaml: List[Dict],
    bbox_mm: Tuple[float, float, float, float]
) -> List[Dict]:
    """Filter strokes list to those overlapping given bbox.

    Parameters
    ----------
    strokes_yaml : List[Dict]
        List of stroke dictionaries
    bbox_mm : Tuple[float, float, float, float]
        Filter region (xmin, ymin, xmax, ymax) in mm

    Returns
    -------
    List[Dict]
        Filtered strokes list

    Notes
    -----
    Used for tile-based rendering: only process strokes in visible region.
    """
    xmin_filter, ymin_filter, xmax_filter, ymax_filter = bbox_mm
    
    filtered = []
    for stroke in strokes_yaml:
        xmin, ymin, xmax, ymax = stroke_bbox_mm(stroke)
        
        # Check overlap
        if not (xmax < xmin_filter or xmin > xmax_filter or
                ymax < ymin_filter or ymin > ymax_filter):
            filtered.append(stroke)
    
    return filtered


def count_strokes(strokes_yaml_path: Union[str, Path]) -> int:
    """Count number of strokes in YAML file.

    Parameters
    ----------
    strokes_yaml_path : Union[str, Path]
        Path to strokes YAML file

    Returns
    -------
    int
        Number of strokes

    Notes
    -----
    Used for validation and metrics (stroke count vs. stroke cap).
    """
    from src.utils import fs
    
    strokes_list = fs.load_yaml(strokes_yaml_path)
    return len(strokes_list) if isinstance(strokes_list, list) else 0


def extract_stroke_colors(strokes_yaml: List[Dict]) -> torch.Tensor:
    """Extract CMY colors from strokes list.

    Parameters
    ----------
    strokes_yaml : List[Dict]
        List of stroke dictionaries

    Returns
    -------
    torch.Tensor
        CMY colors, shape (N, 3), range [0, 1]

    Notes
    -----
    Used for color distribution analysis and debugging.
    """
    colors = []
    for stroke in strokes_yaml:
        cmy = stroke['color_cmy']
        colors.append([cmy['c'], cmy['m'], cmy['y']])
    
    return torch.tensor(colors, dtype=torch.float32)


def validate_stroke_cap(
    strokes_yaml_path: Union[str, Path],
    stroke_cap: int
) -> bool:
    """Check if stroke count exceeds cap.

    Parameters
    ----------
    strokes_yaml_path : Union[str, Path]
        Path to strokes YAML file
    stroke_cap : int
        Maximum allowed strokes

    Returns
    -------
    bool
        True if within cap, False otherwise

    Notes
    -----
    Used in tests and CI to validate stroke cap enforcement.
    """
    count = count_strokes(strokes_yaml_path)
    return count <= stroke_cap
