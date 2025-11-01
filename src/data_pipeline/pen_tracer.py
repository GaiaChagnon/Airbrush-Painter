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

from ..utils import validators, fs, color, compute, geometry, metrics

logger = logging.getLogger(__name__)


def compute_gamut_mask(
    img_lab: torch.Tensor,
    min_luminance: float,
    max_chroma: float,
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
    margin : float
        Safety margin for gamut expansion (0.0-0.3)
    
    Returns
    -------
    out_of_gamut_mask : np.ndarray
        Binary mask, shape (H, W), 255 where pen is needed
    """
    L = img_lab[0].numpy()
    a = img_lab[1].numpy()
    b = img_lab[2].numpy()
    
    # Compute chroma
    C = np.sqrt(a**2 + b**2)
    
    # Expand gamut by margin
    min_l_expanded = min_luminance * (1 + margin)
    max_c_expanded = max_chroma * (1 + margin)
    
    # Out of gamut if:
    # 1. Too dark (L* < min_luminance)
    # 2. Too saturated (C > max_chroma)
    too_dark = L < min_l_expanded
    too_saturated = C > max_c_expanded
    
    out_of_gamut = (too_dark | too_saturated).astype(np.uint8) * 255
    
    return out_of_gamut


def extract_edges(
    img_rgb: torch.Tensor,
    cfg: validators.PenTracerV2EdgeDetection
) -> np.ndarray:
    """Extract edges using Canny edge detection.
    
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
    
    # Convert to grayscale
    img_np = img_rgb.permute(1, 2, 0).numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Canny edge detection
    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    
    # Link nearby edges
    if cfg.link_distance_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.link_distance_px*2+1, cfg.link_distance_px*2+1)
        )
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Remove short edges
    if cfg.min_length_px > 0:
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(edges)
        for cnt in contours:
            if cv2.arcLength(cnt, False) >= cfg.min_length_px:
                cv2.drawContours(filtered, [cnt], -1, 255, 1)
        edges = filtered
    
    return edges


def vectorize_edges(
    edge_mask: np.ndarray,
    simplify_tol_px: float,
    mm_per_px: float
) -> List[np.ndarray]:
    """Vectorize edge mask to polylines.
    
    Parameters
    ----------
    edge_mask : np.ndarray
        Binary edge mask, shape (H, W)
    simplify_tol_px : float
        Douglas-Peucker simplification tolerance
    mm_per_px : float
        Millimeters per pixel
    
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
        
        # Simplify
        if simplify_tol_px > 0:
            line = LineString(pts_px)
            line_simplified = line.simplify(simplify_tol_px, preserve_topology=True)
            pts_px = np.array(line_simplified.coords)
        
        # Convert to mm
        pts_mm = pts_px * mm_per_px
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
    mm_per_px: float
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
    mm_per_px : float
        Millimeters per pixel
    
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
        
        # Convert to mm
        for start, end in segments:
            if end - start >= 2:
                seg_mm = samples[start:end+1] * mm_per_px
                hatch_lines_mm.append(seg_mm)
        
        y_current += hatch_spacing_px
    
    return hatch_lines_mm


def make_pen_layer_v3(
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
    mm_per_px = work_area_mm[1] / H
    
    logger.info(f"  Resolution: {W}x{H} px, {mm_per_px:.4f} mm/px")
    
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
        
        edge_polylines = vectorize_edges(
            edge_mask,
            pen_tracer_cfg.edge_detection.simplify_tol_px,
            mm_per_px
        )
        
        logger.info(f"  Found {len(edge_polylines)} edge contours")
        
        # Add to paths
        for polyline_mm in edge_polylines:
            path_dict = {
                'id': f"pen-{path_id_counter:06d}",
                'kind': 'polyline',
                'role': 'outline',
                'tip_diameter_mm': pen_tool_cfg.tip_diameter_mm,
                'z_mm': pen_tool_cfg.draw_z_mm,
                'feed_mm_s': pen_tool_cfg.feed_mm_s,
                'points_mm': polyline_mm.tolist()
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
        
        # Generate hatching
        base_spacing_px = pen_tool_cfg.tip_diameter_mm / mm_per_px
        hatch_spacing_px = base_spacing_px * (1.0 - pen_tool_cfg.overlap_frac) * pen_tracer_cfg.shadow_hatching.spacing_scale
        
        total_hatch_lines = 0
        max_hatch_pixels = int(H * W * pen_tracer_cfg.shadow_hatching.max_hatch_coverage)
        current_hatch_pixels = 0
        
        for level_idx, level in enumerate(pen_tracer_cfg.shadow_hatching.darkness_levels):
            key = (level_idx, level.passes)
            if key not in shadow_masks:
                continue
            
            mask = shadow_masks[key]
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
                hatch_lines = generate_hatch_pattern(mask, hatch_spacing_px, angle, mm_per_px)
                
                logger.info(f"    Pass {pass_idx+1}/{level.passes} ({angle}°): {len(hatch_lines)} lines")
                
                # Add to paths and track coverage
                for line_mm in hatch_lines:
                    # Estimate pixels covered by this line
                    line_length_mm = np.linalg.norm(np.diff(line_mm, axis=0), axis=1).sum()
                    line_pixels = int(line_length_mm / mm_per_px)
                    
                    # Check if adding this line would exceed coverage limit
                    if current_hatch_pixels + line_pixels > max_hatch_pixels:
                        logger.warning(f"    Stopping - coverage limit reached")
                        break
                    
                    path_dict = {
                        'id': f"pen-{path_id_counter:06d}",
                        'kind': 'polyline',
                        'role': 'hatch',
                        'tip_diameter_mm': pen_tool_cfg.tip_diameter_mm,
                        'z_mm': pen_tool_cfg.draw_z_mm,
                        'feed_mm_s': pen_tool_cfg.feed_mm_s,
                        'points_mm': line_mm.tolist()
                    }
                    all_paths.append(path_dict)
                    path_id_counter += 1
                    current_hatch_pixels += line_pixels
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
    pen_vectors_data = {
        'schema': 'pen_vectors.v1',
        'render_px': render_px,
        'work_area_mm': work_area_mm,
        'paths': all_paths,
        'metadata': {
            'tool_name': pen_tool_cfg.name,
            'offset_mm': pen_tool_cfg.offset_mm,
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'tracer_version': 'pen_tracer.v3',
            'gamut_aware': pen_tracer_cfg.shadow_hatching.gamut_aware
        }
    }
    
    pen_vectors_path = out_path / "pen_vectors.yaml"
    logger.info(f"Saving {len(all_paths)} paths to YAML (this may take a while)...")
    fs.atomic_yaml_dump(pen_vectors_data, pen_vectors_path)
    
    # ========================================================================
    # RENDER PREVIEW
    # ========================================================================
    logger.info("Rendering preview...")
    
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    
    for path in all_paths:
        pts_mm = np.array(path['points_mm'])
        pts_px = (pts_mm / mm_per_px).astype(np.int32)
        cv2.polylines(canvas, [pts_px], False, (0, 0, 0), 1, cv2.LINE_AA)
    
    pen_preview_path = out_path / "pen_preview.png"
    cv2.imwrite(str(pen_preview_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    
    # Composite
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
    
    # Overlay pen strokes (only where pen is black)
    composite = base_img.copy()
    pen_mask = (canvas < 250).any(axis=2)  # Black pixels in pen layer
    composite[pen_mask] = canvas[pen_mask]
    
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

