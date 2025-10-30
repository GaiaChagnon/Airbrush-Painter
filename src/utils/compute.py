"""Numerics, resolution conversions, tiling, and bounds projection.

Core utilities:
    - LPIPS normalization: normalize_img_for_lpips() converts linear RGB [0,1] → [-1,1] FP32
    - Resolution conversions: mm_to_px(), px_to_mm() with frame-aware transforms
    - Coordinate frame transforms: image_mm_to_machine_mm()
    - Tiled inference: TiledLPIPS with Hann windowing for large images
    - Bounds projection: project_to_bounds() for action clamping
    - Memory guards: choose_tile_size_from_mem(), retry_cuda_oom()

Invariants:
    - Internal images are linear RGB [0,1] (F32)
    - LPIPS inputs must be [-1,1] and computed in FP32 (no autocast)
    - Geometry parameters are in mm; conversions happen at boundaries only
    - Work area: (210.0, 297.0) mm (A4 portrait) unless overridden

Multi-resolution architecture:
    - render_px: Physics simulation grid (e.g., 908×1280)
    - obs_px: Policy input (downsampled, e.g., 454×640)
    - reward_px: LPIPS scoring grid (typically == render_px)
"""

from contextlib import contextmanager
from typing import Callable, Optional, Tuple

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_0_1(x: torch.Tensor, src_range: str = "uint8") -> torch.Tensor:
    """Convert image tensor to [0, 1] range.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor, shape (3, H, W) or (B, 3, H, W)
    src_range : str
        Source range: "uint8" [0,255], "minus1_1" [-1,1], or "0_1" (no-op)

    Returns
    -------
    torch.Tensor
        Tensor in [0, 1] range, same dtype as input
    """
    if src_range == "uint8":
        return x.float() / 255.0
    elif src_range == "minus1_1":
        return (x + 1.0) / 2.0
    elif src_range == "0_1":
        return x
    else:
        raise ValueError(f"Unknown src_range: {src_range}. Use 'uint8', 'minus1_1', or '0_1'.")


def to_minus1_1(x: torch.Tensor, src_range: str = "0_1") -> torch.Tensor:
    """Convert image tensor to [-1, 1] range.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor, shape (3, H, W) or (B, 3, H, W)
    src_range : str
        Source range: "0_1" [0,1], "uint8" [0,255], or "minus1_1" (no-op)

    Returns
    -------
    torch.Tensor
        Tensor in [-1, 1] range, same dtype as input
    """
    if src_range == "0_1":
        return 2.0 * x - 1.0
    elif src_range == "uint8":
        return 2.0 * (x.float() / 255.0) - 1.0
    elif src_range == "minus1_1":
        return x
    else:
        raise ValueError(f"Unknown src_range: {src_range}. Use '0_1', 'uint8', or 'minus1_1'.")


def normalize_img_for_lpips(x: torch.Tensor, colorspace: str = "linear_rgb") -> torch.Tensor:
    """Normalize image tensor for LPIPS input (expected range: [-1, 1], FP32).

    Parameters
    ----------
    x : torch.Tensor
        Input image, shape (3, H, W) or (B, 3, H, W), linear RGB [0,1]
    colorspace : str
        "linear_rgb" (default) or "srgb" (will convert to linear first)

    Returns
    -------
    torch.Tensor
        Normalized tensor in [-1, 1], dtype FP32

    Notes
    -----
    LPIPS pretrained models expect [-1, 1] normalized RGB.
    This function ensures FP32 precision for numerical stability.
    Always compute LPIPS with autocast(enabled=False).
    """
    if colorspace == "srgb":
        from src.utils.color import srgb_to_linear
        x = srgb_to_linear(x)
    elif colorspace != "linear_rgb":
        raise ValueError(f"Unknown colorspace: {colorspace}. Use 'linear_rgb' or 'srgb'.")
    
    # Ensure [0, 1] range (clamp for safety)
    x = torch.clamp(x, 0.0, 1.0)
    # Convert to [-1, 1] and ensure FP32
    return (2.0 * x - 1.0).float()


def mm_to_px(
    x_mm: torch.Tensor,
    work_area_mm: Tuple[float, float],
    render_px: Tuple[int, int],
    image_origin: str = "top_left"
) -> torch.Tensor:
    """Convert mm coordinates to pixel coordinates.

    Parameters
    ----------
    x_mm : torch.Tensor
        Coordinates in mm, shape (..., 2) with (x, y)
    work_area_mm : Tuple[float, float]
        Work area size (width_mm, height_mm), e.g., (210.0, 297.0) for A4
    render_px : Tuple[int, int]
        Render resolution (width_px, height_px), e.g., (908, 1280)
    image_origin : str
        "top_left" (image frame, +Y down) or "bottom_left" (machine frame, +Y up)

    Returns
    -------
    torch.Tensor
        Coordinates in pixels, shape (..., 2) with (x, y)

    Notes
    -----
    Image frame: origin at top-left, +X right, +Y down (standard raster)
    Machine frame: origin at bottom-left, +X right, +Y up (standard G-code)
    """
    sx = render_px[0] / work_area_mm[0]
    sy = render_px[1] / work_area_mm[1]
    
    x_px = x_mm.clone()
    x_px[..., 0] = x_mm[..., 0] * sx
    
    if image_origin == "top_left":
        # Image: +Y down; flip Y axis
        x_px[..., 1] = (work_area_mm[1] - x_mm[..., 1]) * sy
    elif image_origin == "bottom_left":
        # Machine: +Y up; direct scaling
        x_px[..., 1] = x_mm[..., 1] * sy
    else:
        raise ValueError(f"Unknown image_origin: {image_origin}. Use 'top_left' or 'bottom_left'.")
    
    return x_px


def px_to_mm(
    x_px: torch.Tensor,
    work_area_mm: Tuple[float, float],
    render_px: Tuple[int, int],
    image_origin: str = "top_left"
) -> torch.Tensor:
    """Convert pixel coordinates to mm coordinates.

    Parameters
    ----------
    x_px : torch.Tensor
        Coordinates in pixels, shape (..., 2) with (x, y)
    work_area_mm : Tuple[float, float]
        Work area size (width_mm, height_mm)
    render_px : Tuple[int, int]
        Render resolution (width_px, height_px)
    image_origin : str
        "top_left" (image frame) or "bottom_left" (machine frame)

    Returns
    -------
    torch.Tensor
        Coordinates in mm, shape (..., 2) with (x, y)
    """
    sx = work_area_mm[0] / render_px[0]
    sy = work_area_mm[1] / render_px[1]
    
    x_mm = x_px.clone()
    x_mm[..., 0] = x_px[..., 0] * sx
    
    if image_origin == "top_left":
        # Image: +Y down; flip Y axis
        x_mm[..., 1] = work_area_mm[1] - (x_px[..., 1] * sy)
    elif image_origin == "bottom_left":
        # Machine: +Y up; direct scaling
        x_mm[..., 1] = x_px[..., 1] * sy
    else:
        raise ValueError(f"Unknown image_origin: {image_origin}. Use 'top_left' or 'bottom_left'.")
    
    return x_mm


def image_mm_to_machine_mm(
    x_img_mm: torch.Tensor,
    work_area_mm: Tuple[float, float],
    flip_y: bool = True
) -> torch.Tensor:
    """Transform image-frame mm to machine-frame mm.

    Parameters
    ----------
    x_img_mm : torch.Tensor
        Coordinates in image frame, shape (..., 2) with (x, y)
        Image frame: origin top-left, +Y down
    work_area_mm : Tuple[float, float]
        Work area size (width_mm, height_mm)
    flip_y : bool
        True if machine uses bottom-left origin (+Y up), default True

    Returns
    -------
    torch.Tensor
        Coordinates in machine frame, shape (..., 2) with (x, y)
        Machine frame: origin bottom-left, +Y up (standard G-code)

    Notes
    -----
    This transform is applied once at G-code generation boundary.
    Internal pipeline stays in image frame for consistency with rendering.
    """
    x_mach_mm = x_img_mm.clone()
    if flip_y:
        x_mach_mm[..., 1] = work_area_mm[1] - x_img_mm[..., 1]
    return x_mach_mm


def hann2d(h: int, w: int, device=None, dtype=None) -> torch.Tensor:
    """Create 2D Hann (cosine) window for tile blending.

    Parameters
    ----------
    h : int
        Window height
    w : int
        Window width
    device : torch.device, optional
        Device for tensor
    dtype : torch.dtype, optional
        Data type, defaults to torch.float32

    Returns
    -------
    torch.Tensor
        2D Hann window, shape (h, w), values in [0, 1]

    Notes
    -----
    Used for seamless tile stitching in tiled LPIPS and renderer.
    Window is multiplicative: blends overlapping regions smoothly.
    """
    if dtype is None:
        dtype = torch.float32
    
    # 1D Hann windows
    hann_h = torch.hann_window(h, periodic=False, dtype=dtype, device=device)
    hann_w = torch.hann_window(w, periodic=False, dtype=dtype, device=device)
    
    # Outer product for 2D
    return hann_h.view(-1, 1) * hann_w.view(1, -1)


def tile_slices(H: int, W: int, tile: int, overlap: int) -> list:
    """Generate tile slice indices for tiled processing.

    Parameters
    ----------
    H : int
        Image height
    W : int
        Image width
    tile : int
        Tile size (square)
    overlap : int
        Overlap size between adjacent tiles

    Returns
    -------
    list[tuple[slice, slice]]
        List of (slice_h, slice_w) tuples covering the image

    Notes
    -----
    Tiles overlap by `overlap` pixels on each side for smooth blending.
    Last tile may extend beyond image; caller should pad if needed.
    """
    stride = tile - overlap
    slices = []
    
    for y_start in range(0, H, stride):
        for x_start in range(0, W, stride):
            y_end = min(y_start + tile, H)
            x_end = min(x_start + tile, W)
            slices.append((slice(y_start, y_end), slice(x_start, x_end)))
    
    return slices


def apply_tiled(
    fn: Callable,
    img: torch.Tensor,
    tile: int,
    overlap: int,
    pad_to_multiple: int = 1,
    **kwargs
) -> torch.Tensor:
    """Apply function to image in tiles with Hann blending.

    Parameters
    ----------
    fn : Callable
        Function to apply to each tile, signature: fn(tile_tensor, **kwargs) -> tile_tensor
    img : torch.Tensor
        Input image, shape (B, C, H, W)
    tile : int
        Tile size (square)
    overlap : int
        Overlap size for blending
    pad_to_multiple : int
        Pad image to multiple of this value (for alignment), default 1 (no padding)
    **kwargs
        Additional arguments passed to fn

    Returns
    -------
    torch.Tensor
        Processed image, shape (B, C, H, W)

    Notes
    -----
    If tile == 0, processes full image without tiling.
    Uses Hann windows for smooth blending in overlap regions.
    """
    if tile == 0:
        # Full-frame processing
        return fn(img, **kwargs)
    
    B, C, H, W = img.shape
    device = img.device
    dtype = img.dtype
    
    # Pad to multiple if requested
    if pad_to_multiple > 1:
        pad_h = (pad_to_multiple - H % pad_to_multiple) % pad_to_multiple
        pad_w = (pad_to_multiple - W % pad_to_multiple) % pad_to_multiple
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='replicate')
            H_pad, W_pad = img.shape[2:]
        else:
            H_pad, W_pad = H, W
    else:
        H_pad, W_pad = H, W
    
    # Initialize output and weight accumulator
    output = torch.zeros((B, C, H_pad, W_pad), device=device, dtype=dtype)
    weights = torch.zeros((H_pad, W_pad), device=device, dtype=dtype)
    
    # Process tiles
    slices = tile_slices(H_pad, W_pad, tile, overlap)
    for slice_h, slice_w in slices:
        tile_img = img[:, :, slice_h, slice_w]
        tile_out = fn(tile_img, **kwargs)
        
        # Create blending window
        th, tw = tile_out.shape[2:]
        window = hann2d(th, tw, device=device, dtype=dtype)
        
        # Accumulate with weighting
        output[:, :, slice_h, slice_w] += tile_out * window.view(1, 1, th, tw)
        weights[slice_h, slice_w] += window
    
    # Normalize by accumulated weights
    weights = torch.clamp(weights, min=1e-8)  # Avoid division by zero
    output = output / weights.view(1, 1, H_pad, W_pad)
    
    # Crop back to original size
    return output[:, :, :H, :W]


def project_to_bounds(
    x: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor
) -> torch.Tensor:
    """Clamp tensor to [lo, hi] bounds element-wise.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor, any shape
    lo : torch.Tensor
        Lower bounds, broadcastable to x
    hi : torch.Tensor
        Upper bounds, broadcastable to x

    Returns
    -------
    torch.Tensor
        Clamped tensor, same shape as x

    Notes
    -----
    Used for action validation in environment and renderer.
    This is the ONLY place where silent clamping is acceptable.
    """
    return torch.clamp(x, min=lo, max=hi)


def clamp_finite(
    x: torch.Tensor,
    min_val: float = -1e30,
    max_val: float = 1e30
) -> torch.Tensor:
    """Replace NaN/Inf with finite values and clamp to range.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    min_val : float
        Minimum value after NaN/Inf replacement
    max_val : float
        Maximum value after NaN/Inf replacement

    Returns
    -------
    torch.Tensor
        Finite tensor, NaN/Inf replaced and clamped

    Notes
    -----
    Use sparingly; prefer explicit error handling.
    Logs warning if non-finite values detected.
    """
    if not torch.isfinite(x).all():
        import logging
        logging.getLogger(__name__).warning(
            f"Non-finite values detected in tensor: {torch.isnan(x).sum()} NaNs, "
            f"{torch.isinf(x).sum()} Infs"
        )
        x = torch.nan_to_num(x, nan=0.0, posinf=max_val, neginf=min_val)
    return torch.clamp(x, min=min_val, max=max_val)


def assert_finite(x: torch.Tensor, name: str = "tensor") -> None:
    """Assert tensor contains no NaN or Inf values.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to check
    name : str
        Tensor name for error message

    Raises
    ------
    ValueError
        If tensor contains NaN or Inf
    """
    if not torch.isfinite(x).all():
        nan_count = torch.isnan(x).sum().item()
        inf_count = torch.isinf(x).sum().item()
        raise ValueError(
            f"{name} contains non-finite values: {nan_count} NaNs, {inf_count} Infs. "
            f"Shape: {x.shape}, dtype: {x.dtype}, device: {x.device}"
        )


def choose_tile_size_from_mem(
    H: int,
    W: int,
    bytes_per_px: int = 16,
    mem_margin: float = 0.8
) -> int:
    """Heuristic to choose tile size based on available GPU memory.

    Parameters
    ----------
    H : int
        Image height
    W : int
        Image width
    bytes_per_px : int
        Estimated bytes per pixel (default 16: FP32 RGBA with overhead)
    mem_margin : float
        Safety margin (use 80% of free memory), default 0.8

    Returns
    -------
    int
        Suggested tile size (power of 2), or 0 for full-frame

    Notes
    -----
    On DGX Spark with ample memory, often returns 0 (full-frame).
    Use when processing validation images or high-res final renders.
    """
    if not torch.cuda.is_available():
        return 0  # CPU: no tiling needed
    
    free_mem = torch.cuda.mem_get_info()[0]
    usable_mem = free_mem * mem_margin
    
    # Estimate full-frame memory
    full_frame_mem = H * W * bytes_per_px
    
    if full_frame_mem <= usable_mem:
        return 0  # Full-frame fits
    
    # Find largest power-of-2 tile that fits
    for tile in [4096, 2048, 1024, 512, 256]:
        tile_mem = tile * tile * bytes_per_px
        if tile_mem <= usable_mem:
            return tile
    
    return 256  # Minimum safe tile size


def retry_cuda_oom(fn: Callable, *args, **kwargs):
    """Retry function on CUDA OOM, reducing tile size if applicable.

    Parameters
    ----------
    fn : Callable
        Function to retry
    *args, **kwargs
        Arguments to fn

    Returns
    -------
    Any
        Return value of fn

    Raises
    ------
    RuntimeError
        If all retries fail

    Notes
    -----
    Experimental; prefer explicit tile size configuration.
    Only retries if 'tile' kwarg is present and > 256.
    """
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        
        # Try reducing tile size if available
        if "tile" in kwargs and kwargs["tile"] > 256:
            import logging
            tile_old = kwargs["tile"]
            kwargs["tile"] = max(256, kwargs["tile"] // 2)
            logging.getLogger(__name__).warning(
                f"CUDA OOM, retrying with tile={kwargs['tile']} (was {tile_old})"
            )
            torch.cuda.empty_cache()
            return fn(*args, **kwargs)
        else:
            raise RuntimeError(f"CUDA OOM and no tile size reduction available: {e}") from e


@contextmanager
def autocast_context(precision: str = "bf16"):
    """Context manager for mixed precision training.

    Parameters
    ----------
    precision : str
        "bf16", "fp16", or "fp32" (no autocast)

    Yields
    ------
    torch.cuda.amp.autocast context or nullcontext

    Examples
    --------
    >>> with autocast_context("bf16"):
    ...     output = model(input)
    """
    if precision == "fp32":
        yield
    elif precision == "bf16":
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            yield
    elif precision == "fp16":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            yield
    else:
        raise ValueError(f"Unknown precision: {precision}. Use 'bf16', 'fp16', or 'fp32'.")


class TiledLPIPS(nn.Module):
    """LPIPS with optional tiling; always runs in FP32 with normalized inputs.

    Parameters
    ----------
    net : str
        Backbone network: 'vgg', 'alex', or 'squeeze'
    tile_size : int
        Tile size for tiled processing, 0 for full-frame (default 0)
    overlap : int
        Overlap size for tile blending (default 0)

    Notes
    -----
    LPIPS computation is always in FP32 for numerical accuracy.
    Inputs must be normalized to [-1, 1] via normalize_img_for_lpips().
    On DGX Spark, tile_size=0 (full-frame) is default and recommended.
    """

    def __init__(self, net: str = "vgg", tile_size: int = 0, overlap: int = 0):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net).eval()
        self.tile_size = tile_size
        self.overlap = overlap
        
        # Freeze LPIPS weights
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS distance between two images.

        Parameters
        ----------
        img1 : torch.Tensor
            First image, shape (B, 3, H, W), normalized to [-1, 1]
        img2 : torch.Tensor
            Second image, shape (B, 3, H, W), normalized to [-1, 1]

        Returns
        -------
        torch.Tensor
            LPIPS distance, scalar (mean over batch)

        Raises
        ------
        ValueError
            If inputs are not in [-1, 1] range or wrong shape
        """
        # Validate inputs
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")
        if img1.ndim != 4 or img1.shape[1] != 3:
            raise ValueError(f"Expected shape (B, 3, H, W), got {img1.shape}")
        
        # Check range (allow small numerical error)
        if not (-1.1 <= img1.min() <= 1.1 and -1.1 <= img2.min() <= 1.1):
            raise ValueError(
                f"LPIPS inputs must be in [-1, 1]. Got img1 range [{img1.min():.3f}, {img1.max():.3f}], "
                f"img2 range [{img2.min():.3f}, {img2.max():.3f}]. "
                f"Use normalize_img_for_lpips() to convert."
            )
        
        # Force FP32 for numerical accuracy
        img1 = img1.float()
        img2 = img2.float()
        
        # Disable autocast for LPIPS computation
        with torch.cuda.amp.autocast(enabled=False):
            if self.tile_size == 0:
                # Full-frame processing
                return self.lpips(img1, img2).mean()
            else:
                # Tiled processing - need to tile both images together
                B, C, H, W = img1.shape
                device = img1.device
                dtype = img1.dtype
                
                # Pad to multiple if needed
                pad_h = (self.tile_size - H % self.tile_size) % self.tile_size
                pad_w = (self.tile_size - W % self.tile_size) % self.tile_size
                if pad_h > 0 or pad_w > 0:
                    img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode='replicate')
                    img2 = F.pad(img2, (0, pad_w, 0, pad_h), mode='replicate')
                    H_pad, W_pad = img1.shape[2:]
                else:
                    H_pad, W_pad = H, W
                
                # Get tile slices
                slices = tile_slices(H_pad, W_pad, self.tile_size, self.overlap)
                
                # Accumulate LPIPS over tiles with weighting
                total_loss = 0.0
                total_weight = 0.0
                
                for slice_h, slice_w in slices:
                    tile1 = img1[:, :, slice_h, slice_w]
                    tile2 = img2[:, :, slice_h, slice_w]
                    
                    # Compute LPIPS for this tile
                    tile_loss = self.lpips(tile1, tile2)
                    
                    # Weight by tile area (for overlap handling)
                    th, tw = tile1.shape[2:]
                    weight = th * tw
                    
                    total_loss += tile_loss.sum() * weight
                    total_weight += weight * B
                
                return total_loss / total_weight
