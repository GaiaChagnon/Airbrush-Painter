"""Gamut-aware pen layer extraction for Airbrush Painter.

This module implements intelligent pen tracing that complements CMY airbrush:
    1. EDGE MODE: Extract outlines from sharp color/luminance changes
    2. GAMUT-AWARE SHADOW MODE: Only hatch regions CMY cannot reproduce

Key features:
    - A4 print quality (300 DPI, 2184×3508 pixels)
    - CMY gamut awareness (only hatch out-of-gamut colors)
    - Calibration data integration (ready for real gamut measurements)
    - Configurable coverage limits (default 20% max)
    - Single-direction hatching (leaves space for CMY colors)
    - Exclusive darkness ranges (no double tracing)

Pipeline:
    1. Load image and optionally upscale to target resolution
    2. EDGE EXTRACTION: Canny edge detection + vectorization
    3. GAMUT ANALYSIS: Identify regions outside CMY gamut
    4. SHADOW EXTRACTION: Multi-level hatching for out-of-gamut darks only
    5. Render preview and compute metrics

All geometry in millimeters (mm) in image frame (top-left, +Y down).
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

import numpy as np
import cv2
import torch
from scipy import ndimage
from skimage import morphology, measure
from shapely.geometry import LineString
import pyclipper

from ..utils import validators, color, compute, geometry, metrics, fs

logger = logging.getLogger(__name__)


def compute_gamut_mask(
    img_lab: torch.Tensor,
    min_luminance: float,
    max_chroma: float,
    hue_ranges: list,
    margin: float = 0.05
) -> np.ndarray:
    """Compute mask of regions outside CMY gamut.
    
    Parameters
    ----------
    img_lab : torch.Tensor
        LAB image, shape (3, H, W)
    min_luminance : float
        Minimum L* achievable by CMY (e.g., 15.0 for deep black)
    max_chroma : float
        Maximum chroma achievable by CMY (e.g., 80.0)
    hue_ranges : list
        List of [h_min, h_max] hue ranges in degrees that CMY can reproduce
    margin : float
        Safety margin for gamut expansion (0.0-0.3)
    
    Returns
    -------
    out_of_gamut_mask : np.ndarray
        Binary mask, shape (H, W), 255 where pen is needed
    """
    # GPU-safe: move to CPU before numpy conversion
    L = img_lab[0].detach().cpu().numpy()
    a = img_lab[1].detach().cpu().numpy()
    b = img_lab[2].detach().cpu().numpy()
    
    # Compute chroma
    C = np.sqrt(a**2 + b**2)
    
    # Compute hue in degrees
    hue = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
    
    # Expand gamut by margin
    min_l_expanded = min_luminance * (1 + margin)
    max_c_expanded = max_chroma * (1 + margin)
    
    # Out of gamut if too dark
    too_dark = L < min_l_expanded
    
    # Out of gamut if too saturated AND within problematic hue ranges
    hue_ok = np.zeros_like(hue, dtype=bool)
    for h0, h1 in hue_ranges:
        if h0 <= h1:
            hue_ok |= (hue >= h0) & (hue <= h1)
        else:
            # Wrap-around (e.g., [330, 30] for reds across 0°)
            hue_ok |= (hue >= h0) | (hue <= h1)
    
    too_saturated = (C > max_c_expanded) & hue_ok
    
    out_of_gamut = (too_dark | too_saturated).astype(np.uint8) * 255
    
    return out_of_gamut


def extract_edges(
    img_rgb: torch.Tensor,
    cfg: validators.PenTracerV2EdgeDetection
) -> np.ndarray:
    """Extract edges using LAB color space and Canny edge detection.
    
    Parameters
    ----------
    img_rgb : torch.Tensor
        Linear RGB image, shape (3, H, W), range [0, 1]
    cfg : PenTracerV2EdgeDetection
        Edge detection configuration
    
    Returns
    -------
    edge_mask : np.ndarray
        Binary edge mask, shape (H, W), dtype uint8
    """
    if not cfg.enabled:
        H, W = img_rgb.shape[1], img_rgb.shape[2]
        return np.zeros((H, W), dtype=np.uint8)
    
    # Convert to LAB for perceptually uniform color-aware edge detection
    img_lab = color.rgb_to_lab(img_rgb)  # (3, H, W)
    L_channel = img_lab[0].detach().cpu().numpy()  # (H, W) in [0, 100]
    
    # Normalize L* to uint8
    L_u8 = np.clip(L_channel * 255.0 / 100.0, 0, 255).astype(np.uint8)
    
    # Bilateral filter: smooths gradients while preserving sharp edges
    # This eliminates the dark-light-dark artifacts that cause double lines
    L_u8 = cv2.bilateralFilter(
        L_u8,
        d=cfg.bilateral_d,
        sigmaColor=cfg.bilateral_sigma_color,
        sigmaSpace=cfg.bilateral_sigma_space
    )
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(L_u8, (0, 0), cfg.sigma_px)
    
    # Canny edge detection with configurable thresholds
    edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)
    
    # Apply morphological closing to connect nearby edge fragments
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.closing_kernel_size, cfg.closing_kernel_size)
    )
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Skeletonize to eliminate double lines (both sides of edges)
    from skimage.morphology import skeletonize
    edges = skeletonize(edges > 0).astype(np.uint8) * 255
    
    # Dilate skeleton slightly to merge nearby parallel lines (from gradient artifacts)
    # Then skeletonize again to get single centerline
    merge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.merge_kernel_size, cfg.merge_kernel_size)
    )
    edges = cv2.dilate(edges, merge_kernel, iterations=1)
    edges = skeletonize(edges > 0).astype(np.uint8) * 255
    
    return edges


def _trace_skeleton_to_polylines(skel_u8: np.ndarray) -> List[np.ndarray]:
    """Convert 1-px skeleton to ordered polylines (pixel coords).
    
    Walks skeleton graph from endpoints through junctions, producing
    long connected polylines instead of fragmented contours.
    
    Parameters
    ----------
    skel_u8 : np.ndarray
        Binary skeleton mask, shape (H, W), dtype uint8
    
    Returns
    -------
    polylines : List[np.ndarray]
        List of polylines in pixel coordinates, each shape (N, 2) as (x, y)
    """
    skel = (skel_u8 > 0).astype(np.uint8)
    H, W = skel.shape
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    
    # Degree map (8-connected)
    deg = cv2.filter2D(skel, -1, np.ones((3,3), np.uint8)) - skel
    endpoints = np.argwhere((skel==1) & (deg==1))
    visited = np.zeros_like(skel, dtype=bool)
    
    def walk(p):
        """Walk from point p until hitting junction or dead end."""
        path = [tuple(p)]
        visited[p[0], p[1]] = True
        prev = None
        cur = tuple(p)
        
        while True:
            nexts = []
            for dy, dx in nbrs:
                ny, nx = cur[0]+dy, cur[1]+dx
                if 0 <= ny < H and 0 <= nx < W and skel[ny,nx] and not visited[ny,nx]:
                    if prev is None or (ny, nx) != prev:
                        nexts.append((ny, nx))
            
            if len(nexts) == 0:
                break
            
            nxt = nexts[0]  # Prefer straight continuation if multiple
            prev, cur = cur, nxt
            visited[cur[0], cur[1]] = True
            path.append(cur)
            
            # Stop at junction
            if deg[cur[0], cur[1]] >= 3:
                break
        
        return np.array(path, dtype=np.float32)[:, ::-1]  # (y,x) -> (x,y)
    
    polylines = []
    
    # Trace from endpoints
    for p in endpoints:
        if not visited[p[0], p[1]]:
            path = walk(p)
            if len(path) >= 2:
                polylines.append(path)
    
    # Trace remaining cycles (closed loops without endpoints)
    rem = np.argwhere((skel==1) & (~visited))
    for p in rem:
        if not visited[p[0], p[1]]:
            path = walk(p)
            if len(path) >= 2:
                polylines.append(path)
    
    return polylines


def _link_and_simplify(
    polys_px: List[np.ndarray],
    sx_mm_per_px: float,
    sy_mm_per_px: float,
    link_gap_px: int,
    link_angle_deg: float,
    min_len_mm: float,
    rdp_tol_mm: float
) -> List[np.ndarray]:
    """Link nearby endpoints and simplify with mm-aware RDP.
    
    Parameters
    ----------
    polys_px : List[np.ndarray]
        List of polylines in pixel coordinates
    sx_mm_per_px : float
        Millimeters per pixel in X direction
    sy_mm_per_px : float
        Millimeters per pixel in Y direction
    link_gap_px : int
        Maximum gap distance to bridge
    link_angle_deg : float
        Maximum angle difference for linking
    min_len_mm : float
        Minimum segment length to keep
    rdp_tol_mm : float
        RDP simplification tolerance in mm
    
    Returns
    -------
    simplified : List[np.ndarray]
        Linked and simplified polylines in pixel coordinates
    """
    
    def len_mm(p):
        """Calculate polyline length in mm with anisotropic scaling."""
        d = np.diff(p, axis=0)
        L = np.sqrt((d[:,0]*sx_mm_per_px)**2 + (d[:,1]*sy_mm_per_px)**2).sum()
        return L
    
    def angle_of(p, head=True):
        """Get direction angle at head or tail."""
        v = (p[1]-p[0]) if head else (p[-1]-p[-2])
        return np.degrees(np.arctan2(v[1], v[0]))
    
    # 1. Drop short fragments
    keep = [p for p in polys_px if len_mm(p) >= min_len_mm]
    
    # 2. Endpoint linking (spatial index for O(N log N) instead of O(N²))
    # Build KD-tree for fast nearest-neighbor queries
    from scipy.spatial import cKDTree
    
    # Limit iterations to prevent infinite loops
    max_iterations = 5
    for iteration in range(max_iterations):
        if len(keep) == 0:
            break
        
        # Collect all endpoints with metadata
        endpoints = []
        for i, p in enumerate(keep):
            if len(p) >= 2:
                endpoints.append((i, 'tail', p[-1], angle_of(p, head=False)))
                endpoints.append((i, 'head', p[0], angle_of(p, head=True)))
        
        if len(endpoints) == 0:
            break
        
        # Build spatial index
        coords = np.array([ep[2] for ep in endpoints])
        tree = cKDTree(coords)
        
        # Find pairs within link_gap_px
        pairs = tree.query_pairs(link_gap_px, output_type='ndarray')
        
        if len(pairs) == 0:
            break
        
        # Process pairs (greedy linking)
        used = set()
        merged = []
        
        for idx1, idx2 in pairs:
            ep1 = endpoints[idx1]
            ep2 = endpoints[idx2]
            
            i1, end1, pt1, ang1 = ep1
            i2, end2, pt2, ang2 = ep2
            
            # Skip if same polyline or already used
            if i1 == i2 or i1 in used or i2 in used:
                continue
            
            # Check angle compatibility
            angle_diff = abs(((ang1 - ang2 + 180) % 360) - 180)
            if angle_diff > link_angle_deg:
                continue
            
            # Link tail->head or tail->tail
            if end1 == 'tail' and end2 == 'head':
                merged.append(np.vstack([keep[i1], keep[i2]]))
                used.add(i1)
                used.add(i2)
            elif end1 == 'tail' and end2 == 'tail':
                merged.append(np.vstack([keep[i1], keep[i2][::-1]]))
                used.add(i1)
                used.add(i2)
        
        # Keep unmerged + merged
        keep = [p for i, p in enumerate(keep) if i not in used] + merged
        
        if len(merged) == 0:
            break
    
    # 3. Simplify with mm-aware RDP
    eps_px = rdp_tol_mm / max(sx_mm_per_px, sy_mm_per_px)
    simplified = []
    for p in keep:
        approx = cv2.approxPolyDP(
            p.reshape(-1,1,2).astype(np.float32),
            epsilon=eps_px,
            closed=False
        ).reshape(-1,2)
        if approx.shape[0] >= 2 and len_mm(approx) >= min_len_mm:
            simplified.append(approx)
    
    return simplified


def _dedupe_parallel(
    polys_px: List[np.ndarray],
    dedupe_min_sep_px: float,
    dedupe_max_angle_deg: float
) -> List[np.ndarray]:
    """Collapse near-parallel double lines into single centerlines.
    
    Parameters
    ----------
    polys_px : List[np.ndarray]
        List of polylines in pixel coordinates
    dedupe_min_sep_px : float
        Collapse lines closer than this distance
    dedupe_max_angle_deg : float
        Maximum angle difference for parallel detection
    
    Returns
    -------
    out : List[np.ndarray]
        Deduplicated polylines
    """
    
    if len(polys_px) == 0:
        return []
    
    # Pre-compute metadata for spatial filtering
    metadata = []
    for i, p in enumerate(polys_px):
        v = p[-1] - p[0]
        angle = np.degrees(np.arctan2(v[1], v[0]))
        length = np.linalg.norm(v)
        bbox_min = p.min(axis=0)
        bbox_max = p.max(axis=0)
        center = (bbox_min + bbox_max) / 2
        metadata.append({
            'idx': i,
            'angle': angle,
            'length': length,
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'center': center,
            'poly': p
        })
    
    # Sort by length (descending) to keep longer lines
    metadata.sort(key=lambda m: m['length'], reverse=True)
    
    used = set()
    out = []
    
    for i, mi in enumerate(metadata):
        if mi['idx'] in used:
            continue
        
        # Mark as used
        used.add(mi['idx'])
        keep = True
        
        # Check against already kept lines (not all lines)
        for mj in metadata[:i]:
            if mj['idx'] in used or mj['idx'] == mi['idx']:
                continue
            
            # Quick bbox check (with padding)
            if (mi['bbox_max'][0] < mj['bbox_min'][0] - dedupe_min_sep_px or
                mi['bbox_min'][0] > mj['bbox_max'][0] + dedupe_min_sep_px or
                mi['bbox_max'][1] < mj['bbox_min'][1] - dedupe_min_sep_px or
                mi['bbox_min'][1] > mj['bbox_max'][1] + dedupe_min_sep_px):
                continue
            
            # Quick angle check
            angle_diff = abs(((mi['angle'] - mj['angle'] + 180) % 360) - 180)
            if angle_diff > dedupe_max_angle_deg:
                continue
            
            # Quick center distance check
            center_dist = np.linalg.norm(mi['center'] - mj['center'])
            if center_dist > dedupe_min_sep_px * 5:
                continue
            
            # Detailed distance check (sample 10 points)
            pi = mi['poly']
            pj = mj['poly']
            sample_idx = np.linspace(0, len(pi)-1, min(10, len(pi)), dtype=int)
            dists = []
            for idx in sample_idx:
                dists.append(np.min(np.linalg.norm(pj - pi[idx], axis=1)))
            
            mean_sep = float(np.mean(dists))
            if mean_sep <= dedupe_min_sep_px:
                # This is a duplicate of a longer line - skip it
                keep = False
                break
        
        if keep:
            out.append(mi['poly'])
    
    return out


def vectorize_edges(
    edge_mask: np.ndarray,
    simplify_tol_px: float,
    min_length_px: int,
    sx_mm_per_px: float,
    sy_mm_per_px: float
) -> List[np.ndarray]:
    """Vectorize edge mask to polylines.
    
    Parameters
    ----------
    edge_mask : np.ndarray
        Binary edge mask, shape (H, W)
    simplify_tol_px : float
        Douglas-Peucker simplification tolerance
    sx_mm_per_px : float
        Millimeters per pixel in X direction
    sy_mm_per_px : float
        Millimeters per pixel in Y direction
    
    Returns
    -------
    polylines_mm : List[np.ndarray]
        List of polylines, each shape (N, 2), in mm
    """
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    polylines_mm = []
    for cnt in contours:
        pts_px = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts_px) < 2:
            continue
        
        # Filter by minimum length (in pixels)
        if min_length_px > 0:
            perimeter_px = cv2.arcLength(cnt, closed=False)
            if perimeter_px < min_length_px:
                continue
        
        # Simplify
        if simplify_tol_px > 0:
            line = LineString(pts_px)
            line_simplified = line.simplify(simplify_tol_px, preserve_topology=True)
            pts_px = np.array(line_simplified.coords)
        
        # Clip to image bounds (H, W)
        H, W = edge_mask.shape
        pts_px[:,0] = np.clip(pts_px[:,0], 0, W-1)
        pts_px[:,1] = np.clip(pts_px[:,1], 0, H-1)
        
        # Convert to mm with anisotropic scaling
        pts_mm = np.stack([pts_px[:,0] * sx_mm_per_px,
                          pts_px[:,1] * sy_mm_per_px], axis=1)
        polylines_mm.append(pts_mm)
    
    return polylines_mm


def extract_gamut_aware_shadows(
    img_lab: torch.Tensor,
    gamut_mask: np.ndarray,
    darkness_levels: List[validators.PenTracerV2DarknessLevel],
    min_area_px: int
) -> Dict[Tuple[int, int], np.ndarray]:
    """Extract shadow regions outside CMY gamut at different darkness levels.
    
    Each level is EXCLUSIVE (only includes pixels in that specific range).
    This prevents overlapping/double-tracing.
    
    Parameters
    ----------
    img_lab : torch.Tensor
        LAB image, shape (3, H, W)
    gamut_mask : np.ndarray
        Out-of-gamut mask, shape (H, W), 255 where pen is needed
    darkness_levels : List[PenTracerV2DarknessLevel]
        Darkness thresholds and pass counts
    min_area_px : int
        Minimum region area
    
    Returns
    -------
    shadow_masks : Dict[Tuple[int, int], np.ndarray]
        Dictionary mapping (level_idx, passes) -> binary mask
    """
    L_channel = img_lab[0].numpy()
    shadow_masks = {}
    
    for level_idx, level in enumerate(darkness_levels):
        # EXCLUSIVE range: l_min <= L* < l_max
        # This ensures no overlap between levels
        in_range = (L_channel >= level.l_min) & (L_channel < level.l_max)
        dark_mask = in_range.astype(np.uint8) * 255
        
        # Intersect with out-of-gamut mask
        # Only hatch regions that are BOTH in-range AND out-of-gamut
        combined_mask = cv2.bitwise_and(dark_mask, gamut_mask)
        
        # Remove small regions
        if min_area_px > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                combined_mask, connectivity=8
            )
            filtered = np.zeros_like(combined_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                    filtered[labels == i] = 255
            combined_mask = filtered
        
        if combined_mask.sum() > 0:
            shadow_masks[(level_idx, level.passes)] = combined_mask
    
    return shadow_masks


def generate_hatch_pattern(
    mask: np.ndarray,
    hatch_spacing_px: float,
    hatch_angle_deg: float,
    sx_mm_per_px: float,
    sy_mm_per_px: float
) -> List[np.ndarray]:
    """Generate hatching lines for a region.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask, shape (H, W)
    hatch_spacing_px : float
        Spacing between hatch lines
    hatch_angle_deg : float
        Hatch angle in degrees
    sx_mm_per_px : float
        Millimeters per pixel in X direction
    sy_mm_per_px : float
        Millimeters per pixel in Y direction
    
    Returns
    -------
    hatch_lines_mm : List[np.ndarray]
        List of hatch line segments, each shape (N, 2), in mm
    """
    H, W = mask.shape
    
    # Rotation matrix
    angle_rad = np.deg2rad(hatch_angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    R_inv = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    
    # Get mask points
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return []
    
    pts = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
    
    # Rotate points
    pts_rot = pts @ R.T
    
    # Bounding box in rotated space
    x_min, y_min = pts_rot.min(axis=0)
    x_max, y_max = pts_rot.max(axis=0)
    
    # Generate hatch lines
    hatch_lines_mm = []
    y_current = y_min
    
    while y_current <= y_max:
        # Horizontal line in rotated space
        line_rot = np.array([[x_min, y_current], [x_max, y_current]])
        
        # Rotate back to image space
        line_img = line_rot @ R_inv.T
        
        # Sample along line
        num_samples = int(np.linalg.norm(line_img[1] - line_img[0]) * 2)
        if num_samples < 2:
            y_current += hatch_spacing_px
            continue
        
        t_vals = np.linspace(0, 1, num_samples)
        samples = line_img[0] + t_vals[:, None] * (line_img[1] - line_img[0])
        
        # Check which samples are inside mask
        x_int = np.clip(samples[:, 0].astype(int), 0, W-1)
        y_int = np.clip(samples[:, 1].astype(int), 0, H-1)
        inside = mask[y_int, x_int] > 0
        
        # Find continuous segments
        segments = []
        start_idx = None
        for i, is_inside in enumerate(inside):
            if is_inside and start_idx is None:
                start_idx = i
            elif not is_inside and start_idx is not None:
                segments.append((start_idx, i-1))
                start_idx = None
        if start_idx is not None:
            segments.append((start_idx, len(inside)-1))
        
        # Convert to mm with anisotropic scaling and clip to bounds
        for start, end in segments:
            if end - start >= 2:
                seg_px = samples[start:end+1]
                # Clip to image bounds
                seg_px[:,0] = np.clip(seg_px[:,0], 0, W-1)
                seg_px[:,1] = np.clip(seg_px[:,1], 0, H-1)
                seg_mm = np.stack([seg_px[:,0] * sx_mm_per_px,
                                  seg_px[:,1] * sy_mm_per_px], axis=1)
                hatch_lines_mm.append(seg_mm)
        
        y_current += hatch_spacing_px
    
    return hatch_lines_mm


def make_pen_layer(
    target_rgb_path: str,
    env_cfg_path: str,
    pen_tool_cfg_path: str,
    pen_tracer_cfg_path: str,
    out_dir: str,
    cmy_canvas_path: Optional[str] = None
) -> dict:
    """Generate gamut-aware pen layer with configurable HD resolution.
    
    Parameters
    ----------
    target_rgb_path : str
        Path to target RGB image
    env_cfg_path : str
        Path to environment YAML config
    pen_tool_cfg_path : str
        Path to pen tool YAML config
    pen_tracer_cfg_path : str
        Path to pen tracer V2 YAML config
    out_dir : str
        Output directory
    cmy_canvas_path : Optional[str]
        Optional CMY canvas for composite
    
    Returns
    -------
    result : dict
        Output paths and metrics
    """
    logger.info(f"Starting gamut-aware pen layer generation for {target_rgb_path}")
    
    # Load configs
    env_data = fs.load_yaml(env_cfg_path)
    work_area_mm_dict = env_data['work_area_mm']
    work_area_mm = [work_area_mm_dict['x'], work_area_mm_dict['y']]
    
    pen_tool_cfg = validators.load_pen_tool_config(pen_tool_cfg_path)
    pen_tracer_cfg = validators.load_pen_tracer_v2_config(pen_tracer_cfg_path)
    
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    debug_path = out_path / "debug"
    debug_path.mkdir(exist_ok=True)
    
    # Load target image
    from PIL import Image
    img_pil = Image.open(target_rgb_path).convert('RGB')
    
    # Determine target resolution
    orig_w, orig_h = img_pil.size
    if pen_tracer_cfg.output.target_height_px is not None:
        target_h = pen_tracer_cfg.output.target_height_px
        # Clamp to min/max
        target_h = max(pen_tracer_cfg.output.min_px, min(target_h, pen_tracer_cfg.output.max_px))
        
        # Maintain aspect ratio
        aspect = orig_w / orig_h
        target_w = int(target_h * aspect)
        
        # Clamp width too
        target_w = max(pen_tracer_cfg.output.min_px, min(target_w, pen_tracer_cfg.output.max_px))
        
        logger.info(f"Upscaling from {orig_w}x{orig_h} to {target_w}x{target_h} (aspect ratio: {aspect:.2f})")
        img_pil = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
    else:
        target_w, target_h = orig_w, orig_h
        logger.info(f"Using original resolution: {target_w}x{target_h}")
    
    # Convert to tensors
    img_np = np.array(img_pil)
    img_srgb = torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1)
    img_rgb = color.srgb_to_linear(img_srgb)
    img_lab = color.rgb_to_lab(img_rgb)
    
    H, W = target_h, target_w
    render_px = [W, H]
    
    # Anisotropic scaling: separate X and Y scales for correct aspect ratio
    sx_mm_per_px = work_area_mm[0] / W
    sy_mm_per_px = work_area_mm[1] / H
    
    logger.info(f"  Resolution: {W}x{H} px, sx={sx_mm_per_px:.4f} mm/px, sy={sy_mm_per_px:.4f} mm/px")
    
    # Initialize paths
    all_paths = []
    path_id_counter = 0
    
    # ========================================================================
    # GAMUT ANALYSIS
    # ========================================================================
    if pen_tracer_cfg.shadow_hatching.enabled and pen_tracer_cfg.shadow_hatching.gamut_aware:
        logger.info("Computing CMY gamut mask...")
        
        # Load calibration if available
        if pen_tracer_cfg.calibration.calibration_file:
            # TODO: Load calibration data
            logger.warning("Calibration file specified but loading not yet implemented")
            gamut_cfg = pen_tracer_cfg.shadow_hatching.cmy_gamut
        else:
            gamut_cfg = pen_tracer_cfg.shadow_hatching.cmy_gamut
        
        gamut_mask = compute_gamut_mask(
            img_lab,
            gamut_cfg.min_luminance,
            gamut_cfg.max_chroma,
            gamut_cfg.hue_ranges,
            pen_tracer_cfg.calibration.margin
        )
        
        gamut_coverage = gamut_mask.sum() / (H * W * 255)
        logger.info(f"  Out-of-gamut coverage: {gamut_coverage:.4f} ({gamut_coverage*100:.2f}%)")
        
        if pen_tracer_cfg.debug.save_gamut_mask:
            cv2.imwrite(str(debug_path / "gamut_mask.png"), gamut_mask)
    else:
        # No gamut awareness - hatch everything
        gamut_mask = np.ones((H, W), dtype=np.uint8) * 255
    
    # ========================================================================
    # EDGE EXTRACTION
    # ========================================================================
    if pen_tracer_cfg.edge_detection.enabled:
        logger.info("Extracting edges...")
        
        edge_mask = extract_edges(img_rgb, pen_tracer_cfg.edge_detection)
        
        if pen_tracer_cfg.debug.save_edge_mask:
            cv2.imwrite(str(debug_path / "edge_mask.png"), edge_mask)
        
        # Vectorize edges using contour-based approach
        edge_polylines = vectorize_edges(
            edge_mask,
            pen_tracer_cfg.edge_detection.simplify_tol_px,
            pen_tracer_cfg.edge_detection.min_length_px,
            sx_mm_per_px,
            sy_mm_per_px
        )
        
        logger.info(f"  Found {len(edge_polylines)} edge contours")
        
        # Add to paths
        for polyline_mm in edge_polylines:
            path_dict = {
                'id': f"pen-{path_id_counter:06d}",
                'kind': 'polyline',
                'role': 'outline',
                'tip_diameter_mm': float(pen_tool_cfg.tip_diameter_mm),
                'z_mm': float(pen_tool_cfg.draw_z_mm),
                'feed_mm_s': float(pen_tool_cfg.feed_mm_s),
                'points_mm': polyline_mm.tolist()  # Keep as is, numpy's tolist() is optimized
            }
            all_paths.append(path_dict)
            path_id_counter += 1
    
    # ========================================================================
    # GAMUT-AWARE SHADOW EXTRACTION
    # ========================================================================
    if pen_tracer_cfg.shadow_hatching.enabled:
        logger.info("Extracting gamut-aware shadows...")
        
        shadow_masks = extract_gamut_aware_shadows(
            img_lab,
            gamut_mask,
            pen_tracer_cfg.shadow_hatching.darkness_levels,
            pen_tracer_cfg.shadow_hatching.min_area_px
        )
        
        # Apply morphological closing to connect nearby regions (reduce pen lifts)
        if pen_tracer_cfg.shadow_hatching.close_gaps_px > 0:
            close_kernel_size = pen_tracer_cfg.shadow_hatching.close_gaps_px
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2*close_kernel_size+1, 2*close_kernel_size+1)
            )
            for key in shadow_masks:
                shadow_masks[key] = cv2.morphologyEx(
                    shadow_masks[key],
                    cv2.MORPH_CLOSE,
                    close_kernel
                )
            logger.info(f"  Applied gap closing ({close_kernel_size}px kernel)")
        
        # Dilate edge mask to create exclusion zone (prevent edge-hatch overlap)
        edge_exclusion_mask = None
        if pen_tracer_cfg.edge_detection.enabled and 'edge_mask' in locals():
            dilation_px = max(1, int(0.5 * pen_tool_cfg.tip_diameter_mm / sy_mm_per_px))  # Truncate, don't round
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation_px+1, 2*dilation_px+1))
            edge_exclusion_mask = cv2.dilate(edge_mask, kernel)
            logger.info(f"  Created edge exclusion zone ({dilation_px}px dilation)")
        
        # Generate hatching with minimum spacing enforcement
        base_spacing_px = pen_tool_cfg.tip_diameter_mm / sy_mm_per_px
        hatch_spacing_px = base_spacing_px * (1.0 - pen_tool_cfg.overlap_frac) * pen_tracer_cfg.shadow_hatching.spacing_scale
        min_spacing_px = pen_tracer_cfg.shadow_hatching.min_line_spacing_mm / sy_mm_per_px
        hatch_spacing_px = max(hatch_spacing_px, min_spacing_px)
        
        total_hatch_lines = 0
        max_hatch_pixels = int(H * W * pen_tracer_cfg.shadow_hatching.max_hatch_coverage)
        current_hatch_pixels = 0
        
        for level_idx, level in enumerate(pen_tracer_cfg.shadow_hatching.darkness_levels):
            key = (level_idx, level.passes)
            if key not in shadow_masks:
                continue
            
            mask = shadow_masks[key]
            
            # Apply edge exclusion
            if edge_exclusion_mask is not None:
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(edge_exclusion_mask))
            
            if mask.sum() == 0:
                continue
            
            # Check coverage limit
            if current_hatch_pixels >= max_hatch_pixels:
                logger.warning(f"  Skipping level {level_idx} - coverage limit reached ({current_hatch_pixels}/{max_hatch_pixels} pixels)")
                break
            
            logger.info(f"  Generating {level.passes}-pass hatching (L* {level.l_min:.0f}-{level.l_max:.0f})...")
            
            if pen_tracer_cfg.debug.save_shadow_masks:
                cv2.imwrite(str(debug_path / f"shadow_mask_level{level_idx}.png"), mask)
            
            # Generate passes
            for pass_idx, angle in enumerate(level.hatch_angles):
                hatch_lines = generate_hatch_pattern(mask, hatch_spacing_px, angle, sx_mm_per_px, sy_mm_per_px)
                
                # Filter out segments that are too short (dots/tiny strokes)
                if pen_tracer_cfg.shadow_hatching.min_segment_length_mm > 0:
                    min_len_mm = pen_tracer_cfg.shadow_hatching.min_segment_length_mm
                    filtered_lines = []
                    for line_mm in hatch_lines:
                        # Calculate segment length
                        segment_length_mm = np.linalg.norm(np.diff(line_mm, axis=0), axis=1).sum()
                        
                        # Keep if long enough OR if it's a closed contour (first == last point)
                        is_closed = np.allclose(line_mm[0], line_mm[-1], atol=0.1)
                        if segment_length_mm >= min_len_mm or (is_closed and segment_length_mm >= min_len_mm * 0.5):
                            filtered_lines.append(line_mm)
                    
                    num_filtered = len(hatch_lines) - len(filtered_lines)
                    hatch_lines = filtered_lines
                    if num_filtered > 0:
                        logger.info(f"    Filtered {num_filtered} short segments (< {min_len_mm:.1f}mm)")
                
                logger.info(f"    Pass {pass_idx+1}/{level.passes} ({angle}°): {len(hatch_lines)} lines")
                
                # Add to paths and track coverage
                for line_mm in hatch_lines:
                    # Estimate pixels covered by this line (accounting for pen tip width)
                    line_length_mm = np.linalg.norm(np.diff(line_mm, axis=0), axis=1).sum()
                    thick_px = max(1, int(pen_tool_cfg.tip_diameter_mm / sy_mm_per_px))
                    line_length_px = line_length_mm / sy_mm_per_px
                    line_area_px = int(np.ceil(line_length_px * thick_px))
                    
                    # Check if adding this line would exceed coverage limit
                    if current_hatch_pixels + line_area_px > max_hatch_pixels:
                        logger.warning(f"    Stopping - coverage limit reached")
                        break
                    
                    path_dict = {
                        'id': f"pen-{path_id_counter:06d}",
                        'kind': 'polyline',
                        'role': 'hatch',
                        'tip_diameter_mm': float(pen_tool_cfg.tip_diameter_mm),
                        'z_mm': float(pen_tool_cfg.draw_z_mm),
                        'feed_mm_s': float(pen_tool_cfg.feed_mm_s),
                        'points_mm': line_mm.tolist()
                    }
                    all_paths.append(path_dict)
                    path_id_counter += 1
                    current_hatch_pixels += line_area_px
                    total_hatch_lines += 1
                
                if current_hatch_pixels >= max_hatch_pixels:
                    break
            
            if current_hatch_pixels >= max_hatch_pixels:
                break
        
        actual_coverage = current_hatch_pixels / (H * W)
        logger.info(f"  Final hatch coverage: {actual_coverage:.4f} ({actual_coverage*100:.2f}%)")
    
    logger.info(f"Generated {len(all_paths)} total paths")
    
    # ========================================================================
    # SAVE PEN VECTORS
    # ========================================================================
    # Collect all hatch angles used
    all_angles = sorted({a for lvl in pen_tracer_cfg.shadow_hatching.darkness_levels
                           for a in lvl.hatch_angles}) if pen_tracer_cfg.shadow_hatching.enabled else []
    
    pen_vectors_data = {
        'schema': 'pen_vectors.v1',
        'render_px': render_px,
        'work_area_mm': work_area_mm,
        'paths': all_paths,
        'metadata': {
            'tool_name': pen_tool_cfg.name,
            'offset_mm': pen_tool_cfg.offset_mm,
            'hatch_angles_deg': all_angles,
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'tracer_version': 'pen_tracer.v2'
        }
    }
    
    pen_vectors_path = out_path / "pen_vectors.yaml"
    logger.info(f"Saving {len(all_paths)} paths to YAML...")
    
    # Optimize: Use faster C-based YAML dumper if available
    import yaml
    try:
        from yaml import CDumper as Dumper
    except ImportError:
        from yaml import Dumper
    
    # Write atomically with optimized dumper
    tmp_path = pen_vectors_path.with_suffix('.tmp')
    with open(tmp_path, 'w') as f:
        yaml.dump(
            pen_vectors_data,
            f,
            Dumper=Dumper,
            default_flow_style=False,
            sort_keys=False,
            width=120
        )
    tmp_path.replace(pen_vectors_path)
    
    # ========================================================================
    # RENDER PREVIEW (with proper alpha blending)
    # ========================================================================
    logger.info("Rendering preview...")
    
    # Use actual pen tip width for preview (truncate, don't round up)
    thick_px = max(1, int(pen_tool_cfg.tip_diameter_mm / sy_mm_per_px))
    
    # Render alpha mask for strokes (anti-aliased)
    alpha = np.zeros((H, W), dtype=np.uint8)
    for path in all_paths:
        pts_mm = np.array(path['points_mm'])
        # Convert mm to px with anisotropic scaling
        pts_px = np.stack([pts_mm[:,0] / sx_mm_per_px,
                          pts_mm[:,1] / sy_mm_per_px], axis=1).astype(np.int32)
        if pts_px.shape[0] >= 2:
            cv2.polylines(alpha, [pts_px], False, 255, thick_px, cv2.LINE_AA)
    
    # Alpha blend black strokes onto white canvas (for preview)
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    a = (alpha.astype(np.float32) / 255.0)[..., None]  # HxWx1
    canvas = ((1.0 - a) * canvas.astype(np.float32)).astype(np.uint8)
    
    pen_preview_path = out_path / "pen_preview.png"
    cv2.imwrite(str(pen_preview_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    
    # Composite (alpha blend black strokes onto base image)
    # If CMY canvas provided, overlay pen strokes on it
    # Otherwise, overlay on original target image for preview
    if cmy_canvas_path and Path(cmy_canvas_path).exists():
        base_img = cv2.imread(cmy_canvas_path)
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        base_img = cv2.resize(base_img, (W, H))
    else:
        # Use original target image as base
        from PIL import Image
        target_img = Image.open(target_rgb_path).convert('RGB')
        target_img = target_img.resize((W, H), Image.Resampling.LANCZOS)
        base_img = np.array(target_img)
    
    # Alpha blend black strokes onto base (NO WHITE HALO!)
    a = (alpha.astype(np.float32) / 255.0)[..., None]  # HxWx1
    composite = ((1.0 - a) * base_img.astype(np.float32)).astype(np.uint8)
    
    composite_path = out_path / "composite.png"
    cv2.imwrite(str(composite_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    
    # ========================================================================
    # METRICS
    # ========================================================================
    pen_mask = (canvas < 250).any(axis=2)
    coverage = pen_mask.sum() / (H * W)
    mean_lum = canvas.mean() / 255.0
    
    metrics_data = {
        'coverage_black': float(coverage),
        'mean_luminance': float(mean_lum),
        'num_paths': len(all_paths),
        'num_edge_paths': sum(1 for p in all_paths if p['role'] == 'outline'),
        'num_hatch_paths': sum(1 for p in all_paths if p['role'] == 'hatch'),
        'resolution': render_px,
        'gamut_aware': pen_tracer_cfg.shadow_hatching.gamut_aware
    }
    
    logger.info(f"Metrics: {metrics_data}")
    logger.info(f"Pen layer artifacts saved to {out_path}")
    
    return {
        'pen_vectors_yaml': str(pen_vectors_path),
        'pen_preview_png': str(pen_preview_path),
        'composite_png': str(composite_path),
        'metrics': metrics_data
    }

