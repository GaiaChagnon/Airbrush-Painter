"""Image I/O and color space utilities for pen tracer optimization.

This module provides:
- Image loading/saving with consistent color space handling
- sRGB ↔ linear RGB gamma correction
- RGB ↔ LAB color space conversion (wraps src.utils.color)
- Linear-light alpha compositing (fixes white halo bug)
- Pen path rasterization to alpha mask
"""

from pathlib import Path
from typing import Tuple, List, Dict, Optional
import sys

import numpy as np
import cv2
from PIL import Image

# Import color utilities from main codebase
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import color as color_utils


def load_image_rgb01(path: Path) -> np.ndarray:
    """Load image as RGB float32 in [0,1] range.
    
    Parameters
    ----------
    path : Path
        Path to image file
    
    Returns
    -------
    img_rgb01 : np.ndarray
        RGB image, shape (H, W, 3), dtype float32, range [0, 1], sRGB color space
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    # Use PIL for robust format support
    img = Image.open(path).convert('RGB')
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    return img_np


def save_image_rgb01(path: Path, img_rgb01: np.ndarray) -> None:
    """Save RGB float32 image in [0,1] to disk.
    
    Parameters
    ----------
    path : Path
        Output path
    img_rgb01 : np.ndarray
        RGB image, shape (H, W, 3), dtype float32, range [0, 1]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    img_uint8 = (np.clip(img_rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)


def srgb_to_linear(img_srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear RGB (gamma expansion).
    
    Parameters
    ----------
    img_srgb : np.ndarray
        Image in sRGB, range [0, 1]
    
    Returns
    -------
    img_linear : np.ndarray
        Image in linear RGB, range [0, 1]
    """
    a = 0.055
    return np.where(
        img_srgb <= 0.04045,
        img_srgb / 12.92,
        np.power((img_srgb + a) / (1.0 + a), 2.4)
    )


def linear_to_srgb(img_linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB (gamma compression).
    
    Parameters
    ----------
    img_linear : np.ndarray
        Image in linear RGB, range [0, 1]
    
    Returns
    -------
    img_srgb : np.ndarray
        Image in sRGB, range [0, 1]
    """
    a = 0.055
    return np.where(
        img_linear <= 0.0031308,
        12.92 * img_linear,
        (1.0 + a) * np.power(img_linear, 1.0 / 2.4) - a
    )


def rgb_to_lab(img_rgb_linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB to CIE LAB.
    
    Wraps src.utils.color conversion.
    
    Parameters
    ----------
    img_rgb_linear : np.ndarray
        Linear RGB image, shape (H, W, 3), range [0, 1]
    
    Returns
    -------
    img_lab : np.ndarray
        LAB image, shape (H, W, 3)
        L: [0, 100], a: [-128, 127], b: [-128, 127]
    """
    # color_utils expects (3, H, W) tensor format
    if img_rgb_linear.ndim == 3 and img_rgb_linear.shape[2] == 3:
        # Convert (H, W, 3) -> (3, H, W)
        img_chw = np.transpose(img_rgb_linear, (2, 0, 1))
    else:
        img_chw = img_rgb_linear
    
    # Use the main codebase color utilities
    import torch
    img_tensor = torch.from_numpy(img_chw).float()
    lab_tensor = color_utils.rgb_to_lab(img_tensor)
    lab_np = lab_tensor.numpy()
    
    # Convert back to (H, W, 3)
    if img_rgb_linear.ndim == 3 and img_rgb_linear.shape[2] == 3:
        lab_np = np.transpose(lab_np, (1, 2, 0))
    
    return lab_np


def lab_to_rgb(img_lab: np.ndarray) -> np.ndarray:
    """Convert CIE LAB to linear RGB.
    
    Wraps src.utils.color conversion.
    
    Parameters
    ----------
    img_lab : np.ndarray
        LAB image, shape (H, W, 3)
        L: [0, 100], a: [-128, 127], b: [-128, 127]
    
    Returns
    -------
    img_rgb_linear : np.ndarray
        Linear RGB image, shape (H, W, 3), range [0, 1]
    """
    # color_utils expects (3, H, W) tensor format
    if img_lab.ndim == 3 and img_lab.shape[2] == 3:
        # Convert (H, W, 3) -> (3, H, W)
        img_chw = np.transpose(img_lab, (2, 0, 1))
    else:
        img_chw = img_lab
    
    # Use the main codebase color utilities
    import torch
    lab_tensor = torch.from_numpy(img_chw).float()
    rgb_tensor = color_utils.lab_to_rgb(lab_tensor)
    rgb_np = rgb_tensor.numpy()
    
    # Convert back to (H, W, 3)
    if img_lab.ndim == 3 and img_lab.shape[2] == 3:
        rgb_np = np.transpose(rgb_np, (1, 2, 0))
    
    return np.clip(rgb_np, 0.0, 1.0)


def composite_black_over(base_rgb01: np.ndarray, pen_alpha01: np.ndarray) -> np.ndarray:
    """Composite black ink over base image using linear-light blending.
    
    This function fixes the white halo bug by:
    1. Converting base image to linear RGB
    2. Compositing black (0.0) over base using alpha
    3. Converting result back to sRGB
    
    Formula (in linear space): C_out = C_in * (1 - alpha) + black * alpha
    Since black = 0, this simplifies to: C_out = C_in * (1 - alpha)
    
    Parameters
    ----------
    base_rgb01 : np.ndarray
        Base image in sRGB, shape (H, W, 3), range [0, 1]
    pen_alpha01 : np.ndarray
        Pen coverage mask, shape (H, W), range [0, 1]
        0 = no ink, 1 = full ink coverage
    
    Returns
    -------
    composite_rgb01 : np.ndarray
        Composited image in sRGB, shape (H, W, 3), range [0, 1]
    """
    # Ensure alpha is 2D
    if pen_alpha01.ndim == 3:
        pen_alpha01 = pen_alpha01[:, :, 0]
    
    # Convert base to linear RGB
    base_linear = srgb_to_linear(base_rgb01)
    
    # Black-over compositing in linear space
    # out = base * (1 - alpha) + 0 * alpha = base * (1 - alpha)
    out_linear = base_linear * (1.0 - pen_alpha01[:, :, np.newaxis])
    
    # Convert back to sRGB
    composite_srgb = linear_to_srgb(out_linear)
    
    return np.clip(composite_srgb, 0.0, 1.0)


def render_pen_alpha(
    paths_mm: List[Dict],
    height_px: int,
    width_px: int,
    mm_per_px_y: float,
    mm_per_px_x: float,
    tip_diameter_mm: float,
    min_thickness_px: int = 1
) -> np.ndarray:
    """Rasterize pen paths to alpha (coverage) mask.
    
    Renders black strokes on transparent background with anti-aliasing.
    
    Parameters
    ----------
    paths_mm : List[Dict]
        List of path dicts, each with 'points_mm' key containing (N, 2) array
        in millimeters, image frame (top-left origin, +Y down)
    height_px : int
        Canvas height in pixels
    width_px : int
        Canvas width in pixels
    mm_per_px_y : float
        Millimeters per pixel in Y direction
    mm_per_px_x : float
        Millimeters per pixel in X direction
    tip_diameter_mm : float
        Pen tip diameter in millimeters
    min_thickness_px : int
        Minimum stroke thickness in pixels (default 1)
    
    Returns
    -------
    alpha_mask : np.ndarray
        Coverage mask, shape (H, W), dtype float32, range [0, 1]
        0 = no ink, 1 = full coverage
    """
    # Create alpha mask (0 = transparent, 1 = opaque)
    alpha = np.zeros((height_px, width_px), dtype=np.float32)
    
    # Convert tip diameter to pixels (use average of x and y)
    avg_mm_per_px = (mm_per_px_x + mm_per_px_y) / 2.0
    thick_px = max(min_thickness_px, int(tip_diameter_mm / avg_mm_per_px))
    
    # Render each path
    for path in paths_mm:
        pts_mm = np.asarray(path['points_mm'], dtype=np.float32)
        if len(pts_mm) < 2:
            continue
        
        # Convert mm -> px
        pts_px = np.stack([
            pts_mm[:, 0] / mm_per_px_x,
            pts_mm[:, 1] / mm_per_px_y
        ], axis=1).astype(np.int32)
        
        # Draw with anti-aliasing (value 1.0 = full coverage)
        cv2.polylines(alpha, [pts_px], False, 1.0, thick_px, cv2.LINE_AA)
    
    # Clamp to [0, 1]
    alpha = np.clip(alpha, 0.0, 1.0)
    
    return alpha


def render_pen_preview(pen_alpha01: np.ndarray) -> np.ndarray:
    """Render pen-only preview (black lines on white background).
    
    For debugging and visualization.
    
    Parameters
    ----------
    pen_alpha01 : np.ndarray
        Pen coverage mask, shape (H, W), range [0, 1]
    
    Returns
    -------
    preview_rgb01 : np.ndarray
        Preview image, shape (H, W, 3), range [0, 1]
        White background with black ink
    """
    # 1.0 - alpha gives white background with black ink
    preview = 1.0 - pen_alpha01
    
    # Expand to 3 channels
    if preview.ndim == 2:
        preview = np.repeat(preview[:, :, np.newaxis], 3, axis=2)
    
    return np.clip(preview, 0.0, 1.0)

