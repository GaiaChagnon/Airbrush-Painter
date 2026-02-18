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
    
    # Out of gamut if too dark (CMY can't reach deep blacks)
    too_dark = L < min_l_expanded
    
    # Check if hue is OUTSIDE CMY's reproducible ranges
    hue_in_cmy_range = np.zeros_like(hue, dtype=bool)
    for h0, h1 in hue_ranges:
        if h0 <= h1:
            hue_in_cmy_range |= (hue >= h0) & (hue <= h1)
        else:
            # Wrap-around (e.g., [330, 30] for reds across 0°)
            hue_in_cmy_range |= (hue >= h0) | (hue <= h1)
    
    # Out of gamut if too saturated (regardless of hue - CMY has chroma limits)
    # This catches colors that are too vivid for CMY to reproduce
    too_saturated = C > max_c_expanded
    
    out_of_gamut = (too_dark | too_saturated).astype(np.uint8) * 255
    
    return out_of_gamut


def extract_edges(
    img_rgb: torch.Tensor,
    cfg: validators.PenTracerV2EdgeDetection
) -> np.ndarray:
    """Extract edges using LAB luminance and multi-scale Canny detection.

    When ``multiscale_sigmas`` is set in the config, Canny is run at each
    sigma and the results are OR-merged, capturing both fine and coarse
    edges.  When ``hysteresis_q`` is set, Canny thresholds are computed
    adaptively from the gradient magnitude distribution instead of using
    fixed ``canny_low`` / ``canny_high``.

    Parameters
    ----------
    img_rgb : torch.Tensor
        Linear RGB image, shape (3, H, W), range [0, 1].
    cfg : PenTracerV2EdgeDetection
        Edge detection configuration.

    Returns
    -------
    edge_mask : np.ndarray
        Binary edge mask, shape (H, W), dtype uint8.
    """
    if not cfg.enabled:
        H, W = img_rgb.shape[1], img_rgb.shape[2]
        return np.zeros((H, W), dtype=np.uint8)

    # Convert to LAB for perceptually uniform colour-aware edge detection
    img_lab = color.rgb_to_lab(img_rgb)  # (3, H, W)
    L_channel = img_lab[0].detach().cpu().numpy()  # (H, W) in [0, 100]
    L_u8 = np.clip(L_channel * 255.0 / 100.0, 0, 255).astype(np.uint8)

    # Bilateral filter: smooths gradients while preserving sharp edges
    L_u8 = cv2.bilateralFilter(
        L_u8,
        d=cfg.bilateral_d,
        sigmaColor=cfg.bilateral_sigma_color,
        sigmaSpace=cfg.bilateral_sigma_space,
    )

    # Determine sigma list: multi-scale if configured, else single pass
    sigmas = (
        getattr(cfg, 'multiscale_sigmas', None) or [cfg.sigma_px]
    )

    # Accumulator (union of all scales)
    edges = np.zeros(L_u8.shape, dtype=np.uint8)

    for sigma in sigmas:
        blurred = cv2.GaussianBlur(L_u8, (0, 0), sigma)

        # Adaptive hysteresis thresholds via gradient quantiles
        hq = getattr(cfg, 'hysteresis_q', None)
        if hq is not None and len(hq) == 2:
            grad_mag = cv2.Sobel(blurred, cv2.CV_64F, 1, 0) ** 2
            grad_mag += cv2.Sobel(blurred, cv2.CV_64F, 0, 1) ** 2
            grad_mag = np.sqrt(grad_mag)
            lo = float(np.quantile(grad_mag[grad_mag > 0], hq[0]))
            hi = float(np.quantile(grad_mag[grad_mag > 0], hq[1]))
            # Floor to configured fixed thresholds if quantiles are too low
            lo = max(lo, cfg.canny_low)
            hi = max(hi, cfg.canny_high)
        else:
            lo, hi = cfg.canny_low, cfg.canny_high

        edges_scale = cv2.Canny(blurred, lo, hi)
        edges = cv2.bitwise_or(edges, edges_scale)

    # Morphological closing to connect nearby edge fragments
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.closing_kernel_size, cfg.closing_kernel_size),
    )
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Skeletonize to eliminate double lines (both sides of edges)
    from skimage.morphology import skeletonize
    edges = skeletonize(edges > 0).astype(np.uint8) * 255

    # Dilate skeleton slightly to merge nearby parallel lines
    merge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.merge_kernel_size, cfg.merge_kernel_size),
    )
    edges = cv2.dilate(edges, merge_kernel, iterations=1)

    # Skeletonize again to get single centerline
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

    Iterates longest-first; a shorter line is dropped when sampled points
    lie within ``dedupe_min_sep_px`` of an already-kept longer line.

    Parameters
    ----------
    polys_px : List[np.ndarray]
        List of polylines in pixel coordinates, each shape (N, 2).
    dedupe_min_sep_px : float
        Collapse lines closer than this distance.
    dedupe_max_angle_deg : float
        Maximum angle difference for parallel detection.

    Returns
    -------
    out : List[np.ndarray]
        Deduplicated polylines.
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
            'poly': p,
        })

    # Sort by length (descending) -- longer lines are kept first
    metadata.sort(key=lambda m: m['length'], reverse=True)

    # kept_metas tracks only the lines we decided to KEEP
    kept_metas: List[dict] = []
    out: List[np.ndarray] = []

    for mi in metadata:
        dominated = False

        for mj in kept_metas:
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

            # Detailed distance check (sample up to 10 points)
            pi = mi['poly']
            pj = mj['poly']
            sample_idx = np.linspace(0, len(pi) - 1, min(10, len(pi)), dtype=int)
            dists = [float(np.min(np.linalg.norm(pj - pi[idx], axis=1)))
                     for idx in sample_idx]

            if float(np.mean(dists)) <= dedupe_min_sep_px:
                dominated = True
                break

        if not dominated:
            kept_metas.append(mi)
            out.append(mi['poly'])

    return out


def _smooth_polyline(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin corner-cutting: smooths a polyline while preserving endpoints.

    Each iteration replaces every interior segment (A, B) with two points
    at 25 % and 75 % along the segment, doubling point count and rounding
    corners.  First and last points are kept exactly.

    Parameters
    ----------
    pts : np.ndarray
        Polyline, shape (N, 2).
    iterations : int
        Number of subdivision passes.  2 gives a visually smooth result
        without excessive point count.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed polyline, shape (M, 2) with M > N.
    """
    if len(pts) < 3:
        return pts

    for _ in range(iterations):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            q = 0.75 * pts[i] + 0.25 * pts[i + 1]
            r = 0.25 * pts[i] + 0.75 * pts[i + 1]
            new_pts.extend([q, r])
        new_pts.append(pts[-1])
        pts = np.array(new_pts, dtype=pts.dtype)

    return pts


def _remove_spikes(
    pts: np.ndarray,
    angle_threshold_deg: float = 30.0,
    length_ratio: float = 0.15,
) -> np.ndarray:
    """Remove sharp spikes from a polyline.

    A spike is a vertex where the path makes a sharp turn (> 180 - threshold)
    and the detour is short relative to the direct distance.  This eliminates
    the triangular artifacts that appear at T-junctions in schematics.

    Parameters
    ----------
    pts : np.ndarray
        Polyline, shape (N, 2).
    angle_threshold_deg : float
        A junction is a spike candidate if the angle between the incoming
        and outgoing segments is less than this value (i.e., nearly a
        reversal).  Default 30 deg means angles < 30 deg (nearly 180 deg
        turn) are checked.
    length_ratio : float
        The spike arm must be shorter than this fraction of the total
        path length to be removed.

    Returns
    -------
    cleaned : np.ndarray
        Polyline with spikes removed, shape (M, 2) with M <= N.
    """
    if len(pts) < 3:
        return pts

    total_len = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    if total_len < 1e-6:
        return pts
    max_spike_len = total_len * length_ratio
    cos_thresh = np.cos(np.radians(180.0 - angle_threshold_deg))

    keep = np.ones(len(pts), dtype=bool)
    for i in range(1, len(pts) - 1):
        v_in = pts[i] - pts[i - 1]
        v_out = pts[i + 1] - pts[i]
        l_in = np.linalg.norm(v_in)
        l_out = np.linalg.norm(v_out)
        if l_in < 1e-9 or l_out < 1e-9:
            continue
        cos_a = np.dot(v_in, v_out) / (l_in * l_out)
        # cos_a = +1 → same direction (straight); cos_a = -1 → full reversal
        # A spike is a near-reversal: cos_a < cos(150°) ≈ -0.866
        if cos_a < cos_thresh and min(l_in, l_out) < max_spike_len:
            keep[i] = False

    cleaned = pts[keep]
    return cleaned if len(cleaned) >= 2 else pts


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
    min_area_px: int,
    mm_per_px: float = 0.0,
    min_dimension_mm: float = 0.5,
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
        Minimum region area in pixels
    mm_per_px : float
        Uniform mm-per-pixel scale.  When > 0, regions whose bounding
        box is smaller than ``min_dimension_mm`` in both width and height
        are discarded.
    min_dimension_mm : float
        Minimum bounding-box dimension in mm (both W and H must exceed
        this or the region is dropped).  Default 0.5 mm.

    Returns
    -------
    shadow_masks : Dict[Tuple[int, int], np.ndarray]
        Dictionary mapping (level_idx, passes) -> binary mask
    """
    L_channel = img_lab[0].numpy()
    shadow_masks = {}

    for level_idx, level in enumerate(darkness_levels):
        in_range = (L_channel >= level.l_min) & (L_channel < level.l_max)
        dark_mask = in_range.astype(np.uint8) * 255

        combined_mask = cv2.bitwise_and(dark_mask, gamut_mask)

        # Remove small regions (pixel area + mm bounding-box filter)
        if min_area_px > 0 or mm_per_px > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                combined_mask, connectivity=8
            )
            filtered = np.zeros_like(combined_mask)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_area_px:
                    continue
                if mm_per_px > 0:
                    w_mm = stats[i, cv2.CC_STAT_WIDTH] * mm_per_px
                    h_mm = stats[i, cv2.CC_STAT_HEIGHT] * mm_per_px
                    if w_mm < min_dimension_mm and h_mm < min_dimension_mm:
                        continue
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
    sy_mm_per_px: float,
) -> List[np.ndarray]:
    """Generate serpentine hatching paths for a region.

    Sweep lines are generated at ``hatch_angle_deg``.  Adjacent segments
    that overlap in the sweep direction are stitched into continuous
    zigzag (serpentine) paths so the pen stays down across multiple lines,
    dramatically reducing pen-lift count.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, shape (H, W).
    hatch_spacing_px : float
        Spacing between hatch lines (pixels).
    hatch_angle_deg : float
        Hatch angle in degrees.
    sx_mm_per_px, sy_mm_per_px : float
        Anisotropic pixel-to-mm scale factors.

    Returns
    -------
    hatch_paths_mm : List[np.ndarray]
        List of (possibly long) serpentine polylines, each shape (N, 2),
        in mm.
    """
    H, W = mask.shape

    # Rotation matrices (image <-> rotated scan space)
    angle_rad = np.deg2rad(hatch_angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    R_inv = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

    # Mask points -> rotated bbox
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return []
    pts = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
    pts_rot = pts @ R.T
    x_min, y_min = pts_rot.min(axis=0)
    x_max, y_max = pts_rot.max(axis=0)

    # -- Phase 1: sweep scan lines and collect segments per line ----------
    sweep_lines: List[List[np.ndarray]] = []  # [line_idx] -> list of seg_mm
    y_current = y_min
    line_idx = 0

    while y_current <= y_max:
        line_rot = np.array([[x_min, y_current], [x_max, y_current]])
        line_img = line_rot @ R_inv.T
        num_samples = int(np.linalg.norm(line_img[1] - line_img[0]) * 2)
        if num_samples < 2:
            y_current += hatch_spacing_px
            continue

        t_vals = np.linspace(0, 1, num_samples)
        samples = line_img[0] + t_vals[:, None] * (line_img[1] - line_img[0])
        x_int = np.clip(samples[:, 0].astype(int), 0, W - 1)
        y_int = np.clip(samples[:, 1].astype(int), 0, H - 1)
        inside = mask[y_int, x_int] > 0

        # Extract continuous inside-mask segments
        segs_mm: List[np.ndarray] = []
        start_idx = None
        for i, v in enumerate(inside):
            if v and start_idx is None:
                start_idx = i
            elif not v and start_idx is not None:
                if i - 1 - start_idx >= 2:
                    seg_px = samples[start_idx:i].copy()
                    seg_px[:, 0] = np.clip(seg_px[:, 0], 0, W - 1)
                    seg_px[:, 1] = np.clip(seg_px[:, 1], 0, H - 1)
                    segs_mm.append(np.stack(
                        [seg_px[:, 0] * sx_mm_per_px,
                         seg_px[:, 1] * sy_mm_per_px], axis=1))
                start_idx = None
        if start_idx is not None and len(inside) - 1 - start_idx >= 2:
            seg_px = samples[start_idx:].copy()
            seg_px[:, 0] = np.clip(seg_px[:, 0], 0, W - 1)
            seg_px[:, 1] = np.clip(seg_px[:, 1], 0, H - 1)
            segs_mm.append(np.stack(
                [seg_px[:, 0] * sx_mm_per_px,
                 seg_px[:, 1] * sy_mm_per_px], axis=1))

        # Alternate direction every other line for serpentine
        if line_idx % 2 == 1:
            segs_mm = [s[::-1] for s in reversed(segs_mm)]

        if segs_mm:
            sweep_lines.append(segs_mm)
            line_idx += 1

        y_current += hatch_spacing_px

    if not sweep_lines:
        return []

    # -- Phase 2: stitch consecutive sweep lines into serpentine chains ---
    # For each sweep line pick the segment whose start is closest to the
    # current chain's tail.  Bridge to it if the gap is small enough.
    # All OTHER segments on the same sweep line are saved separately
    # WITHOUT breaking the main chain.  This keeps the main serpentine
    # long (one per contiguous region) and avoids the old bug where every
    # multi-segment sweep line would fragment the chain into 2-line stubs
    # that all clustered near the edge.
    max_bridge_mm = hatch_spacing_px * max(sx_mm_per_px, sy_mm_per_px) * 1.5

    hatch_paths_mm: List[np.ndarray] = []
    overflow_segs: List[np.ndarray] = []
    current_chain: List[np.ndarray] = []

    for segs in sweep_lines:
        if not segs:
            continue

        if current_chain:
            tail = current_chain[-1][-1]
            best_k = 0
            best_d = float(np.linalg.norm(tail - segs[0][0]))
            for k in range(1, len(segs)):
                d = float(np.linalg.norm(tail - segs[k][0]))
                if d < best_d:
                    best_d = d
                    best_k = k

            if best_d <= max_bridge_mm:
                current_chain.append(segs[best_k])
            else:
                hatch_paths_mm.append(np.vstack(current_chain))
                current_chain = [segs[best_k]]

            for k in range(len(segs)):
                if k != best_k:
                    overflow_segs.append(segs[k])
        else:
            current_chain = [segs[0]]
            for seg in segs[1:]:
                overflow_segs.append(seg)

    if current_chain:
        hatch_paths_mm.append(np.vstack(current_chain))

    # Build additional serpentines from the overflow segments.
    # Sort by first-point position along the sweep axis so adjacent
    # segments in the same sub-region are consecutive, then stitch.
    if overflow_segs:
        overflow_segs.sort(key=lambda s: (s[0][0], s[0][1]))
        chain: List[np.ndarray] = [overflow_segs[0]]
        for seg in overflow_segs[1:]:
            tail = chain[-1][-1]
            head = seg[0]
            if float(np.linalg.norm(tail - head)) <= max_bridge_mm:
                chain.append(seg)
            else:
                hatch_paths_mm.append(np.vstack(chain))
                chain = [seg]
        if chain:
            hatch_paths_mm.append(np.vstack(chain))

    return hatch_paths_mm


def _path_endpoint_dist(p1: dict, p2: dict = None) -> float:
    """Distance from p1's end to p2's start (mm)."""
    if p2 is None:
        return 0.0
    pts1 = np.array(p1['points_mm'])
    pts2 = np.array(p2['points_mm'])
    return np.linalg.norm(pts1[-1] - pts2[0])


def _compute_total_distance(paths: List[dict]) -> float:
    """Compute total travel distance for ordered paths (mm)."""
    total = 0.0
    for i in range(len(paths) - 1):
        total += _path_endpoint_dist(paths[i], paths[i+1])
    return total


def _merge_adjacent_paths(
    paths: list[dict],
    merge_tol_mm: float = 0.4,
    max_angle_deg: float = 45.0,
) -> list[dict]:
    """Merge paths whose endpoints are within *merge_tol_mm*.

    Greedily chains paths end-to-start (or end-to-end / start-to-start
    with reversal) to reduce pen lifts.  Runs before the GNN ordering
    step so that the optimizer works on longer, fewer polylines.

    A direction-continuity check prevents merging paths at T-junctions
    where the bridge line would create a visible diagonal artifact.
    The bridge direction must be within ±max_angle_deg of both the
    exit direction (of the first path) and the entry direction (of the
    second path).

    Parameters
    ----------
    paths : list[dict]
        Path dicts with ``points_mm`` (list of [x, y]).
    merge_tol_mm : float
        Maximum endpoint gap (mm) that will be bridged.
    max_angle_deg : float
        Maximum angle (degrees) between the bridge direction and
        each path's endpoint tangent.  Rejects merges that would
        create a sharp visible kink.

    Returns
    -------
    list[dict]
        Merged paths (fewer, longer polylines).
    """
    if len(paths) <= 1:
        return paths

    from scipy.spatial import cKDTree
    import heapq

    n = len(paths)
    cos_min = np.cos(np.radians(max_angle_deg))

    starts = np.array([p['points_mm'][0] for p in paths])
    ends = np.array([p['points_mm'][-1] for p in paths])

    # Direction at each endpoint (tangent vector).  Use a longer baseline
    # (up to 3 points back) to get a stable direction on noisy curves.
    end_dirs = np.zeros((n, 2))
    start_dirs = np.zeros((n, 2))
    for i, p in enumerate(paths):
        pts = p['points_mm']
        k = min(3, len(pts) - 1)
        end_dirs[i] = np.array(pts[-1]) - np.array(pts[-1 - k])
        start_dirs[i] = np.array(pts[k]) - np.array(pts[0])

    def _dirs_compatible(exit_dir: np.ndarray,
                         bridge: np.ndarray,
                         entry_dir: np.ndarray) -> bool:
        """Check bridge direction aligns with exit and entry tangents."""
        bn = np.linalg.norm(bridge)
        if bn < 1e-6:
            return True  # zero-length bridge always OK
        b_hat = bridge / bn

        en = np.linalg.norm(exit_dir)
        if en > 1e-6:
            if np.dot(exit_dir / en, b_hat) < cos_min:
                return False

        inn = np.linalg.norm(entry_dir)
        if inn > 1e-6:
            if np.dot(b_hat, entry_dir / inn) < cos_min:
                return False
        return True

    start_tree = cKDTree(starts)
    end_tree = cKDTree(ends)

    pairs_end_start = end_tree.query_ball_tree(start_tree, merge_tol_mm)
    pairs_end_end = end_tree.query_ball_tree(end_tree, merge_tol_mm)
    pairs_start_start = start_tree.query_ball_tree(start_tree, merge_tol_mm)

    merge_candidates: list[tuple[float, int, int, str]] = []
    rejected_angle = 0

    for i in range(n):
        for j in pairs_end_start[i]:
            if i == j:
                continue
            bridge = starts[j] - ends[i]
            if not _dirs_compatible(end_dirs[i], bridge, start_dirs[j]):
                rejected_angle += 1
                continue
            d = float(np.linalg.norm(bridge))
            heapq.heappush(merge_candidates, (d, i, j, 'end_start'))
        for j in pairs_end_end[i]:
            if i == j:
                continue
            bridge = ends[j] - ends[i]
            # j will be reversed → its entry direction = -end_dirs[j]
            if not _dirs_compatible(end_dirs[i], bridge, -end_dirs[j]):
                rejected_angle += 1
                continue
            d = float(np.linalg.norm(bridge))
            heapq.heappush(merge_candidates, (d, i, j, 'end_end'))
        for j in pairs_start_start[i]:
            if i == j:
                continue
            bridge = starts[j] - starts[i]
            # i will be reversed → its exit direction = -start_dirs[i]
            if not _dirs_compatible(-start_dirs[i], bridge, start_dirs[j]):
                rejected_angle += 1
                continue
            d = float(np.linalg.norm(bridge))
            heapq.heappush(merge_candidates, (d, i, j, 'start_start'))

    chain_id = list(range(n))
    chains: dict[int, list[tuple[int, bool]]] = {
        i: [(i, False)] for i in range(n)
    }
    head_of: dict[int, int] = {i: i for i in range(n)}
    tail_of: dict[int, int] = {i: i for i in range(n)}

    merges_done = 0

    while merge_candidates:
        dist, i, j, kind = heapq.heappop(merge_candidates)

        ci = chain_id[i]
        cj = chain_id[j]
        if ci == cj:
            continue

        if kind == 'end_start':
            if tail_of[ci] != i or head_of[cj] != j:
                continue
        elif kind == 'end_end':
            if tail_of[ci] != i or tail_of[cj] != j:
                continue
        elif kind == 'start_start':
            if head_of[ci] != i or head_of[cj] != j:
                continue

        if kind == 'end_start':
            chains[ci].extend(chains[cj])
        elif kind == 'end_end':
            chains[ci].extend([(idx, not rev) for idx, rev in reversed(chains[cj])])
        elif kind == 'start_start':
            reversed_ci = [(idx, not rev) for idx, rev in reversed(chains[ci])]
            chains[ci] = reversed_ci
            chains[ci].extend(chains[cj])

        for idx, _ in chains[cj]:
            chain_id[idx] = ci
        del chains[cj]

        first_idx = chains[ci][0][0]
        last_idx = chains[ci][-1][0]
        first_rev = chains[ci][0][1]
        last_rev = chains[ci][-1][1]

        head_of[ci] = first_idx
        tail_of[ci] = last_idx

        merges_done += 1

    merged: list[dict] = []
    for cid, members in chains.items():
        all_pts: list[list[float]] = []
        first_path = paths[members[0][0]]

        for path_idx, is_reversed in members:
            pts = list(paths[path_idx]['points_mm'])
            if is_reversed:
                pts = pts[::-1]
            if all_pts:
                pts = pts[1:] if len(pts) > 1 else pts
            all_pts.extend(pts)

        if len(all_pts) < 2:
            continue

        merged_path = {
            'id': first_path['id'],
            'kind': first_path['kind'],
            'role': first_path['role'],
            'tip_diameter_mm': first_path['tip_diameter_mm'],
            'z_mm': first_path['z_mm'],
            'feed_mm_s': first_path['feed_mm_s'],
            'points_mm': all_pts,
        }
        merged.append(merged_path)

    logger.info(
        f"  Path merge: {n} -> {len(merged)} paths "
        f"({merges_done} merges, {rejected_angle} rejected by angle, "
        f"tol={merge_tol_mm:.1f}mm, max_angle={max_angle_deg:.0f}°)"
    )
    return merged


def _greedy_nearest_neighbor(paths: List[dict], start_pos: np.ndarray = None) -> Tuple[List[dict], float]:
    """Order paths using greedy nearest-neighbor algorithm with spatial indexing.
    
    Handles bidirectional paths (can draw from either end).
    Uses KD-tree for O(N log N) complexity instead of O(N²).
    
    Parameters
    ----------
    paths : List[dict]
        Unordered paths with 'points_mm' key
    start_pos : np.ndarray, optional
        Starting position [x, y] in mm, default [0, 0]
    
    Returns
    -------
    ordered_paths : List[dict]
        Reordered paths (may have 'reversed' flag)
    total_distance : float
        Total travel distance in mm
    """
    if len(paths) == 0:
        return [], 0.0
    
    if start_pos is None:
        start_pos = np.array([0.0, 0.0])
    
    # Build spatial index for fast nearest-neighbor queries
    # Store both start and end points for each path
    from scipy.spatial import cKDTree
    
    endpoints = []
    endpoint_meta = []  # (path_idx, is_end)
    
    for i, path in enumerate(paths):
        pts = np.array(path['points_mm'])
        endpoints.append(pts[0])
        endpoint_meta.append((i, False))  # start point
        endpoints.append(pts[-1])
        endpoint_meta.append((i, True))   # end point
    
    endpoints = np.array(endpoints)
    tree = cKDTree(endpoints)
    
    ordered = []
    used = set()
    current_pos = start_pos
    total_dist = 0.0
    
    while len(used) < len(paths):
        # Find k nearest endpoints - use larger k to handle clustered paths
        # Query more than enough to ensure we find unused paths
        k = min(len(endpoints), max(100, len(paths) - len(used) + 10))
        dists, indices = tree.query(current_pos, k=k)
        
        # Ensure indices is iterable (single result returns scalar)
        if not hasattr(indices, '__iter__'):
            indices = [indices]
            dists = [dists]
        
        # Find nearest unused path
        best_idx = None
        best_dist = float('inf')
        best_reversed = False
        
        for dist, ep_idx in zip(dists, indices):
            path_idx, is_end = endpoint_meta[ep_idx]
            
            if path_idx in used:
                continue
            
            if dist < best_dist:
                best_idx = path_idx
                best_dist = dist
                best_reversed = is_end  # If we're close to end, start from end
        
        if best_idx is None:
            # Shouldn't happen with large k, but fallback to any unused path
            for i in range(len(paths)):
                if i not in used:
                    best_idx = i
                    pts = np.array(paths[i]['points_mm'])
                    dist_start = np.linalg.norm(pts[0] - current_pos)
                    dist_end = np.linalg.norm(pts[-1] - current_pos)
                    best_dist = min(dist_start, dist_end)
                    best_reversed = dist_end < dist_start
                    break
        
        # Add best path
        path = paths[best_idx].copy()
        if best_reversed:
            path['points_mm'] = list(reversed(path['points_mm']))
            path['reversed'] = True
        
        ordered.append(path)
        used.add(best_idx)
        
        # Update position
        pts = np.array(path['points_mm'])
        current_pos = pts[-1]
        total_dist += best_dist
    
    return ordered, total_dist


def _two_opt_refinement(paths: List[dict], max_iterations: int = 5) -> Tuple[List[dict], float]:
    """Refine path order using 2-opt local search.
    
    Parameters
    ----------
    paths : List[dict]
        Initial ordered paths
    max_iterations : int
        Maximum optimization iterations, default 5
    
    Returns
    -------
    refined_paths : List[dict]
        Improved path order
    total_distance : float
        Total travel distance in mm
    
    Notes
    -----
    2-opt is O(N²) per iteration. For large path counts, we adaptively reduce
    iterations or skip entirely to keep runtime under ~5 seconds.
    """
    if len(paths) < 3:
        return paths, _compute_total_distance(paths)
    
    # Adaptive iteration limit based on path count
    # O(N²) means 100 paths = ~10K ops, 500 paths = ~250K ops, 1000 paths = ~1M ops
    n = len(paths)
    if n > 500:
        # Skip 2-opt for very large datasets (GNN is good enough)
        return paths, _compute_total_distance(paths)
    elif n > 200:
        # Only 1 iteration for medium datasets
        max_iterations = 1
    elif n > 100:
        # 2 iterations for smaller datasets
        max_iterations = min(2, max_iterations)
    
    current = list(paths)
    best_dist = _compute_total_distance(current)
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try swapping all pairs of segments
        for i in range(len(current) - 2):
            for j in range(i + 2, min(i + 50, len(current))):  # Limit window for speed
                # Standard 2-opt: reverse segment [i+1:j+1]
                # This reconnects: ...->i->i+1->...->j->j+1->... 
                # to: ...->i->j->...(reversed)->i+1->j+1->...
                
                # Calculate improvement
                # Remove edges: (i, i+1) and (j, j+1)
                # Add edges: (i, j) and (i+1, j+1)
                old_dist = (
                    _path_endpoint_dist(current[i], current[i+1]) +
                    (_path_endpoint_dist(current[j], current[j+1]) if j+1 < len(current) else 0)
                )
                
                # After reversal, we need distances from i to j (now i+1 in sequence)
                # and from old i+1 (now j in sequence) to j+1
                # But we also need to reverse each path's direction in the segment
                new_dist = (
                    _path_endpoint_dist(current[i], current[j]) +
                    (_path_endpoint_dist(current[i+1], current[j+1]) if j+1 < len(current) else 0)
                )
                
                if new_dist < old_dist:
                    # Reverse the segment and flip each path's direction
                    segment = current[i+1:j+1]
                    segment_reversed = []
                    for path in reversed(segment):
                        path_copy = path.copy()
                        path_copy['points_mm'] = list(reversed(path_copy['points_mm']))
                        segment_reversed.append(path_copy)
                    current[i+1:j+1] = segment_reversed
                    improved = True
        
        if not improved:
            break
        
        # Recompute best distance
        best_dist = _compute_total_distance(current)
    
    return current, best_dist


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
    
    # Determine target resolution (preserving aspect ratio strictly)
    orig_w, orig_h = img_pil.size
    if pen_tracer_cfg.output.target_height_px is not None:
        lo = pen_tracer_cfg.output.min_px
        hi = pen_tracer_cfg.output.max_px
        aspect = orig_w / orig_h

        # Start from the configured target height and compute the
        # matching width.  If either dimension exceeds the pixel
        # bounds, scale both down together so the aspect ratio is
        # never broken.
        target_h = max(lo, min(pen_tracer_cfg.output.target_height_px, hi))
        target_w = int(target_h * aspect)

        if target_w > hi:
            target_w = hi
            target_h = int(target_w / aspect)
        if target_w < lo:
            target_w = lo
            target_h = int(target_w / aspect)
        target_h = max(lo, min(target_h, hi))

        logger.info(f"Upscaling from {orig_w}x{orig_h} to {target_w}x{target_h} "
                     f"(aspect ratio: {aspect:.2f}, preserved: "
                     f"{abs(target_w / target_h - aspect) < 0.01})")
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
    
    # Uniform (isotropic) mm/px scaling -- preserves aspect ratio, never
    # stretches.  The work_area_mm from the env config is used as a
    # *reference size*: we pick the scale that fits the image into that
    # bounding box without distortion, then set the output work_area_mm
    # to the image's actual dimensions in mm.  Cropping / fitting to a
    # specific paper size is the responsibility of robot_control.
    s_mm_per_px = min(work_area_mm[0] / W, work_area_mm[1] / H)
    sx_mm_per_px = s_mm_per_px
    sy_mm_per_px = s_mm_per_px

    # Override work_area_mm to the image's actual scaled dimensions
    work_area_mm = [W * s_mm_per_px, H * s_mm_per_px]

    logger.info(f"  Resolution: {W}x{H} px, scale={s_mm_per_px:.4f} mm/px (uniform)")
    logger.info(f"  Output work_area_mm: {work_area_mm[0]:.1f} x {work_area_mm[1]:.1f} mm")
    
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
        
        # Debug: analyze LAB distribution
        L_vals = img_lab[0].cpu().numpy()
        logger.info(f"  Image L* range: [{L_vals.min():.1f}, {L_vals.max():.1f}], mean={L_vals.mean():.1f}")
        logger.info(f"  Pixels below L*={gamut_cfg.min_luminance*(1+pen_tracer_cfg.calibration.margin):.1f}: {(L_vals < gamut_cfg.min_luminance*(1+pen_tracer_cfg.calibration.margin)).sum()} ({(L_vals < gamut_cfg.min_luminance*(1+pen_tracer_cfg.calibration.margin)).sum()/(H*W)*100:.1f}%)")
        
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
        
        # Vectorize edges using skeleton tracing (proper single-line extraction)
        logger.info("  Tracing skeleton to polylines...")
        edge_polylines_px = _trace_skeleton_to_polylines(edge_mask)
        logger.info(f"  Initial skeleton polylines: {len(edge_polylines_px)}")
        
        # Link nearby endpoints and simplify
        logger.info("  Linking and simplifying...")
        link_gap_px = pen_tracer_cfg.edge_detection.link_gap_px if hasattr(pen_tracer_cfg.edge_detection, 'link_gap_px') else 10
        link_angle_deg = pen_tracer_cfg.edge_detection.link_angle_deg if hasattr(pen_tracer_cfg.edge_detection, 'link_angle_deg') else 30
        min_len_mm = pen_tracer_cfg.edge_detection.min_length_px * sy_mm_per_px
        rdp_tol_mm = pen_tracer_cfg.edge_detection.simplify_tol_px * sy_mm_per_px
        
        edge_polylines_px = _link_and_simplify(
            edge_polylines_px,
            sx_mm_per_px,
            sy_mm_per_px,
            link_gap_px,
            link_angle_deg,
            min_len_mm,
            rdp_tol_mm
        )
        logger.info(f"  After linking/simplifying: {len(edge_polylines_px)}")
        
        # Deduplicate parallel lines (critical for eliminating double traces!)
        logger.info("  Deduplicating parallel lines...")
        dedupe_min_sep_px = pen_tool_cfg.tip_diameter_mm / sy_mm_per_px * 1.5  # 1.5x tip diameter
        dedupe_max_angle_deg = 15.0
        edge_polylines_px = _dedupe_parallel(
            edge_polylines_px,
            dedupe_min_sep_px,
            dedupe_max_angle_deg
        )
        logger.info(f"  After deduplication: {len(edge_polylines_px)}")
        
        # Convert to mm, remove spikes, then smooth
        edge_polylines_mm = []
        for pts_px in edge_polylines_px:
            pts_mm = np.stack([pts_px[:,0] * sx_mm_per_px,
                              pts_px[:,1] * sy_mm_per_px], axis=1)
            pts_mm = _remove_spikes(pts_mm, angle_threshold_deg=30.0)
            pts_mm = _smooth_polyline(pts_mm, iterations=2)
            edge_polylines_mm.append(pts_mm)
        
        logger.info(f"  Final edge polylines: {len(edge_polylines_mm)}")
        
        # Add to paths
        for polyline_mm in edge_polylines_mm:
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
            pen_tracer_cfg.shadow_hatching.min_area_px,
            mm_per_px=s_mm_per_px,
            min_dimension_mm=0.5,
        )
        
        logger.info(f"  Extracted {len(shadow_masks)} shadow levels:")
        for (level_idx, passes), mask in shadow_masks.items():
            coverage = mask.sum() / (H * W * 255)
            level_cfg = pen_tracer_cfg.shadow_hatching.darkness_levels[level_idx]
            logger.info(f"    Level {level_idx} (L* {level_cfg.l_min:.0f}-{level_cfg.l_max:.0f}): {coverage*100:.2f}% of canvas")
        
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
        
        # Natural coverage per pass based on pen width and spacing
        pen_width_px = pen_tool_cfg.tip_diameter_mm / sy_mm_per_px
        natural_coverage_per_pass = pen_width_px / hatch_spacing_px
        
        logger.info(f"  Hatch spacing: {hatch_spacing_px:.1f}px ({hatch_spacing_px*sy_mm_per_px:.2f}mm)")
        logger.info(f"  Natural coverage per pass: {natural_coverage_per_pass*100:.1f}% (pen_width/spacing)")
        
        # Calculate total hatch-eligible pixels (sum of all shadow masks)
        total_hatch_eligible_pixels = sum(mask.sum() / 255 for mask in shadow_masks.values())
        if total_hatch_eligible_pixels == 0:
            total_hatch_eligible_pixels = 1  # Avoid division by zero
        
        logger.info(f"  Hatch-eligible pixels: {total_hatch_eligible_pixels:.0f} ({total_hatch_eligible_pixels/(H*W)*100:.1f}% of canvas)")
        
        # max_hatch_coverage limits how much of the eligible zone gets strokes
        # If natural_coverage_per_pass * num_passes > max_hatch_coverage, stop adding passes
        # This is pass-based, not pixel-based (spacing already controls pixel coverage)
        max_pass_coverage_fraction = pen_tracer_cfg.shadow_hatching.max_hatch_coverage
        
        total_hatch_lines = 0
        total_passes_added = 0
        
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
            
            # Calculate what this level would add to total coverage
            # Each pass covers natural_coverage_per_pass of this mask's area
            mask_area = mask.sum() / 255
            mask_fraction_of_eligible = mask_area / total_hatch_eligible_pixels
            level_coverage_contribution = natural_coverage_per_pass * level.passes * mask_fraction_of_eligible
            
            # Check if adding this level would exceed coverage limit
            projected_coverage = (total_passes_added * natural_coverage_per_pass) + level_coverage_contribution
            if projected_coverage > max_pass_coverage_fraction:
                logger.warning(f"  Skipping level {level_idx} - would exceed coverage limit ({projected_coverage*100:.1f}% > {max_pass_coverage_fraction*100:.0f}%)")
                break
            
            logger.info(f"  Generating {level.passes}-pass hatching (L* {level.l_min:.0f}-{level.l_max:.0f})...")
            logger.info(f"    This level will add {level_coverage_contribution*100:.2f}% coverage")
            
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
                
                # Add all lines from this pass
                for line_mm in hatch_lines:
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
                    total_hatch_lines += 1
                
                total_passes_added += 1
        
        # Report final coverage
        actual_coverage_fraction = total_passes_added * natural_coverage_per_pass
        logger.info(f"  Final hatch coverage: {actual_coverage_fraction*100:.1f}% of eligible zone ({total_passes_added} passes)")
    
    logger.info(f"Generated {len(all_paths)} total paths (unordered)")
    
    # ========================================================================
    # PATH MERGING -- reduce pen lifts by chaining near-touching polylines
    # ========================================================================
    logger.info("Merging adjacent paths to reduce pen lifts...")

    edge_paths_raw = [p for p in all_paths if p['role'] == 'outline']
    hatch_paths_raw = [p for p in all_paths if p['role'] == 'hatch']

    # Merge edges (fragmented outline segments benefit from chaining).
    edge_paths_merged = _merge_adjacent_paths(
        edge_paths_raw, merge_tol_mm=0.4, max_angle_deg=45.0,
    )

    # Do NOT merge hatch paths.  The serpentine generator already produces
    # continuous chains where possible, and aggressive merge connects
    # parallel hatch lines end-to-end creating back-and-forth artifacts
    # where the pen repeatedly traces the same bridge line.
    hatch_paths_merged = hatch_paths_raw

    all_paths = edge_paths_merged + hatch_paths_merged
    logger.info(f"After merge: {len(all_paths)} total paths")

    # ========================================================================
    # PATH ORDERING
    # ========================================================================
    logger.info("Optimizing path order (GNN + 2-opt)...")
    
    # Split by role
    edge_paths = [p for p in all_paths if p['role'] == 'outline']
    hatch_paths = [p for p in all_paths if p['role'] == 'hatch']
    
    logger.info(f"  Ordering {len(edge_paths)} edge paths...")
    edge_paths_ordered, edge_dist = _greedy_nearest_neighbor(edge_paths)
    edge_paths_ordered, edge_dist_refined = _two_opt_refinement(edge_paths_ordered)
    
    improvement_pct = ((edge_dist - edge_dist_refined) / edge_dist * 100) if edge_dist > 0 else 0
    logger.info(f"  Edge travel distance: {edge_dist:.1f}mm (GNN) -> {edge_dist_refined:.1f}mm (2-opt, {improvement_pct:+.1f}%)")
    
    # Start hatching from end of last edge
    if edge_paths_ordered:
        last_edge_end = np.array(edge_paths_ordered[-1]['points_mm'][-1])
    else:
        last_edge_end = np.array([0.0, 0.0])
    
    logger.info(f"  Ordering {len(hatch_paths)} hatch paths...")
    hatch_paths_ordered, hatch_dist = _greedy_nearest_neighbor(hatch_paths, start_pos=last_edge_end)
    
    # 2-opt (may be skipped for large path counts)
    if len(hatch_paths) > 500:
        logger.info(f"  Skipping 2-opt for {len(hatch_paths)} paths (too many for O(N²) optimization)")
        logger.info(f"  Hatch travel distance: {hatch_dist:.1f}mm (GNN only)")
        hatch_dist_refined = hatch_dist
    else:
        hatch_paths_ordered, hatch_dist_refined = _two_opt_refinement(hatch_paths_ordered)
        improvement_pct = ((hatch_dist - hatch_dist_refined) / hatch_dist * 100) if hatch_dist > 0 else 0
        logger.info(f"  Hatch travel distance: {hatch_dist:.1f}mm (GNN) -> {hatch_dist_refined:.1f}mm (2-opt, {improvement_pct:+.1f}%)")
    
    # Combine: edges first, then hatching
    all_paths = edge_paths_ordered + hatch_paths_ordered
    
    total_travel = edge_dist_refined + hatch_dist_refined
    logger.info(f"Total optimized travel distance: {total_travel:.1f}mm")
    
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
        'gamut_aware': pen_tracer_cfg.shadow_hatching.gamut_aware,
        'travel_distance_mm': float(total_travel)
    }
    
    logger.info(f"Metrics: {metrics_data}")
    logger.info(f"Pen layer artifacts saved to {out_path}")
    
    return {
        'pen_vectors_yaml': str(pen_vectors_path),
        'pen_preview_png': str(pen_preview_path),
        'composite_png': str(composite_path),
        'metrics': metrics_data
    }

