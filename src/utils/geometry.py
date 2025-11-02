"""Geometric operations for strokes and paths.

Provides:
    - Cubic Bézier evaluation and adaptive flattening
    - Polyline operations: length, bbox, arc-length parametrization
    - Clipping polylines to rectangular regions
    - Stroke bounding box calculation for tile culling

Used by:
    - Renderer: Bézier → polyline ribbon for rasterization
    - G-code generator: Linearization with interpolated Z/speed
    - Environment: Tile culling (only render overlapped tiles)
    - Tests: Synthetic stroke generation and validation

All coordinates in millimeters (mm) unless explicitly noted as pixels.
Conversions to pixels happen at renderer boundaries via compute.mm_to_px().

Adaptive flattening uses recursive subdivision with configurable max_err_mm
tolerance (default: 0.25 mm for fine subdivision).
"""

from typing import Tuple

import torch


def bezier_cubic_eval(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    p4: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """Evaluate cubic Bézier curve at parameter t.

    Parameters
    ----------
    p1, p2, p3, p4 : torch.Tensor
        Control points, shape (..., 2) for (x, y) coordinates
    t : torch.Tensor
        Parameter values in [0, 1], shape (N,) or (..., N)

    Returns
    -------
    torch.Tensor
        Points on curve, shape (..., N, 2)

    Notes
    -----
    Standard cubic Bézier formula:
    B(t) = (1-t)³·p1 + 3(1-t)²t·p2 + 3(1-t)t²·p3 + t³·p4
    """
    # Ensure t is at least 1D
    if t.ndim == 0:
        t = t.unsqueeze(0)
    
    # Broadcast control points to match t
    # p: (..., 2), t: (N,) → output: (..., N, 2)
    t = t.view(*([1] * (p1.ndim)), -1)  # (..., 1, ..., N)
    
    one_minus_t = 1.0 - t
    
    # Bernstein polynomials
    b0 = one_minus_t ** 3
    b1 = 3.0 * (one_minus_t ** 2) * t
    b2 = 3.0 * one_minus_t * (t ** 2)
    b3 = t ** 3
    
    # Expand control points: (..., 2) → (..., 1, 2)
    p1 = p1.unsqueeze(-2)
    p2 = p2.unsqueeze(-2)
    p3 = p3.unsqueeze(-2)
    p4 = p4.unsqueeze(-2)
    
    # Weighted sum
    result = b0 * p1 + b1 * p2 + b2 * p3 + b3 * p4
    
    return result.squeeze(-2) if t.shape[-1] == 1 else result


def bezier_cubic_polyline(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    p4: torch.Tensor,
    max_err_mm: float = 0.25,
    max_depth: int = 12
) -> torch.Tensor:
    """Flatten cubic Bézier to polyline via adaptive subdivision.

    Parameters
    ----------
    p1, p2, p3, p4 : torch.Tensor
        Control points, shape (2,) for (x, y) in mm
    max_err_mm : float
        Maximum allowed deviation in mm, default 0.25 (fine subdivision)
    max_depth : int
        Maximum recursion depth, default 12

    Returns
    -------
    torch.Tensor
        Polyline vertices, shape (N, 2), N ≥ 2

    Notes
    -----
    Uses recursive subdivision with midpoint displacement test.
    Stops when curve is sufficiently flat (within max_err_mm) or max_depth reached.
    
    Flatness criterion: distance from control points to chord < max_err_mm.
    All units in mm (no px conversion needed).
    """
    # Convert to numpy-like operations (keep on same device)
    device = p1.device
    dtype = p1.dtype
    
    def subdivide(q1, q2, q3, q4, depth):
        """Recursive subdivision."""
        if depth >= max_depth:
            return torch.stack([q1, q4], dim=0)
        
        # Check flatness: distance from q2, q3 to line q1-q4
        # Distance from point to line in 2D: |det([[x, y], [chord_x, chord_y]])| / |chord|
        chord = q4 - q1
        chord_len = torch.norm(chord) + 1e-8
        
        # Perpendicular distances (2D cross product = determinant)
        # For 2D vectors: cross(a, b) = a.x * b.y - a.y * b.x
        v2 = q2 - q1
        v3 = q3 - q1
        d2 = torch.abs(v2[0] * chord[1] - v2[1] * chord[0]) / chord_len
        d3 = torch.abs(v3[0] * chord[1] - v3[1] * chord[0]) / chord_len
        
        max_dist_mm = max(d2.item(), d3.item())
        
        if max_dist_mm <= max_err_mm:
            # Sufficiently flat
            return torch.stack([q1, q4], dim=0)
        
        # De Casteljau subdivision at t=0.5
        q12 = (q1 + q2) / 2.0
        q23 = (q2 + q3) / 2.0
        q34 = (q3 + q4) / 2.0
        q123 = (q12 + q23) / 2.0
        q234 = (q23 + q34) / 2.0
        q1234 = (q123 + q234) / 2.0
        
        # Recurse on both halves
        left = subdivide(q1, q12, q123, q1234, depth + 1)
        right = subdivide(q1234, q234, q34, q4, depth + 1)
        
        # Concatenate, removing duplicate midpoint
        return torch.cat([left[:-1], right], dim=0)
    
    polyline = subdivide(p1, p2, p3, p4, depth=0)
    return polyline


def polyline_length(points: torch.Tensor) -> torch.Tensor:
    """Compute total length of polyline.

    Parameters
    ----------
    points : torch.Tensor
        Polyline vertices, shape (N, 2), N ≥ 2

    Returns
    -------
    torch.Tensor
        Total length, scalar

    Notes
    -----
    Sum of Euclidean distances between consecutive points.
    """
    if points.shape[0] < 2:
        return torch.tensor(0.0, device=points.device, dtype=points.dtype)
    
    diffs = points[1:] - points[:-1]
    segment_lengths = torch.norm(diffs, dim=1)
    return segment_lengths.sum()


def polyline_bbox(points: torch.Tensor) -> Tuple[float, float, float, float]:
    """Compute axis-aligned bounding box of polyline.

    Parameters
    ----------
    points : torch.Tensor
        Polyline vertices, shape (N, 2)

    Returns
    -------
    Tuple[float, float, float, float]
        (xmin, ymin, xmax, ymax)

    Notes
    -----
    Returns empty bbox (0, 0, 0, 0) if no points.
    """
    if points.shape[0] == 0:
        return (0.0, 0.0, 0.0, 0.0)
    
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    xmin = x_coords.min().item()
    xmax = x_coords.max().item()
    ymin = y_coords.min().item()
    ymax = y_coords.max().item()
    
    return (xmin, ymin, xmax, ymax)


def parametrize_by_arclength(points: torch.Tensor) -> torch.Tensor:
    """Compute arc-length parametrization for polyline.

    Parameters
    ----------
    points : torch.Tensor
        Polyline vertices, shape (N, 2)

    Returns
    -------
    torch.Tensor
        Arc-length parameters, shape (N,), range [0, 1]
        s[0] = 0.0, s[-1] = 1.0

    Notes
    -----
    Used for interpolating Z/speed profiles along stroke path.
    Linear interpolation between vertices.
    """
    if points.shape[0] < 2:
        return torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)
    
    diffs = points[1:] - points[:-1]
    segment_lengths = torch.norm(diffs, dim=1)
    cumulative_lengths = torch.cat([
        torch.zeros(1, device=points.device, dtype=points.dtype),
        torch.cumsum(segment_lengths, dim=0)
    ])
    
    total_length = cumulative_lengths[-1]
    if total_length < 1e-8:
        # Degenerate case: all points coincident
        return torch.linspace(0.0, 1.0, points.shape[0], device=points.device, dtype=points.dtype)
    
    return cumulative_lengths / total_length


def clip_polyline_to_rect(
    points: torch.Tensor,
    rect_xyxy: Tuple[float, float, float, float]
) -> torch.Tensor:
    """Clip polyline to rectangular region (simple vertex filtering).

    Parameters
    ----------
    points : torch.Tensor
        Polyline vertices, shape (N, 2)
    rect_xyxy : Tuple[float, float, float, float]
        Rectangle bounds (xmin, ymin, xmax, ymax)

    Returns
    -------
    torch.Tensor
        Clipped polyline vertices, shape (M, 2), M ≤ N
        Empty if no vertices inside rectangle

    Notes
    -----
    Simple implementation: keeps vertices inside rectangle.
    Does NOT compute true line-rectangle intersections.
    For stroke tile culling, bbox test is sufficient.
    """
    xmin, ymin, xmax, ymax = rect_xyxy
    
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    mask = (
        (x_coords >= xmin) & (x_coords <= xmax) &
        (y_coords >= ymin) & (y_coords <= ymax)
    )
    
    return points[mask]


def stroke_bbox_mm(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    p4: torch.Tensor,
    margin_mm: float = 5.0
) -> Tuple[float, float, float, float]:
    """Compute conservative bounding box for Bézier stroke in mm.

    Parameters
    ----------
    p1, p2, p3, p4 : torch.Tensor
        Control points, shape (2,), coordinates in mm
    margin_mm : float
        Safety margin for PSF spread, default 5.0 mm

    Returns
    -------
    Tuple[float, float, float, float]
        (xmin, ymin, xmax, ymax) in mm

    Notes
    -----
    Conservative bbox: min/max of control points + margin.
    True bbox requires finding curve extrema, but this is sufficient
    for tile culling (we want to be conservative anyway).
    """
    # Stack control points
    control_points = torch.stack([p1, p2, p3, p4], dim=0)  # (4, 2)
    
    xmin = control_points[:, 0].min().item() - margin_mm
    xmax = control_points[:, 0].max().item() + margin_mm
    ymin = control_points[:, 1].min().item() - margin_mm
    ymax = control_points[:, 1].max().item() + margin_mm
    
    return (xmin, ymin, xmax, ymax)


def interpolate_along_polyline(
    points: torch.Tensor,
    values_start: torch.Tensor,
    values_end: torch.Tensor
) -> torch.Tensor:
    """Linearly interpolate values along polyline arc-length.

    Parameters
    ----------
    points : torch.Tensor
        Polyline vertices, shape (N, 2)
    values_start : torch.Tensor
        Values at start (t=0), shape (D,) for D-dimensional values
    values_end : torch.Tensor
        Values at end (t=1), shape (D,)

    Returns
    -------
    torch.Tensor
        Interpolated values at each vertex, shape (N, D)

    Notes
    -----
    Used for Z and speed profiles along stroke path.
    Values are linearly interpolated based on arc-length parameter.
    
    Example:
        Z interpolation: z0=5.0, z1=10.0 → [5.0, 6.0, ..., 10.0]
        Speed: v0=50.0, v1=100.0 → [50.0, 60.0, ..., 100.0]
    """
    # Get arc-length parameters
    s = parametrize_by_arclength(points)  # (N,)
    
    # Linear interpolation
    # values(s) = (1-s) * values_start + s * values_end
    s = s.unsqueeze(-1)  # (N, 1)
    values_start = values_start.unsqueeze(0)  # (1, D)
    values_end = values_end.unsqueeze(0)  # (1, D)
    
    interpolated = (1.0 - s) * values_start + s * values_end
    
    return interpolated
