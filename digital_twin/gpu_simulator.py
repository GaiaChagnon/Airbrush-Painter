"""GPU stamp-splatting airbrush simulator (digital twin).

A config-driven, differentiable, batch-capable renderer that composites
circular spray stamps onto an RGB canvas using a multiplicative
light-transmission model.

Every physics parameter is read from ``calibration.yaml`` (validated by
:class:`src.utils.validators.CalibrationV1`).  There are no hardcoded
constants in this module.

Architecture (five logical parts)
---------------------------------
A. Calibration loader  -- YAML -> GPU tensors
B. Canonical stamp     -- 1-D radial profile phi(u), normalized integral
C. Stroke sampler      -- Bezier -> arc-length samples
D. GPU compositor      -- batched stamp splatting with transmission model
E. Preview / comparison tools -- calibration QA outputs

Canvas convention
-----------------
- Shape ``(B, 3, H, W)`` -- batch, RGB channels, height, width.
- Linear RGB in [0, 1].  Paper white = ``paper_white_rgb`` from config.
- FP32 throughout; LUTs remain FP32 even when network layers use BF16.

Coordinate frame
----------------
- All geometry in **millimeters**.  Pixel conversion uses DPI derived
  from ``render.canvas_hw`` and ``render.work_area_mm``.
- Image frame: origin top-left, +X right, +Y down.

Usage
-----
>>> from digital_twin.gpu_simulator import GPUStampSimulator
>>> sim = GPUStampSimulator("digital_twin/calibration.yaml")
>>> canvas = sim.reset()
>>> canvas = sim.render_stroke(canvas, stroke_dict)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.utils import (
    color as color_utils,
    compute,
    fs,
    geometry,
    metrics,
    validators,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class StrokeSamples:
    """Arc-length samples along one flattened stroke.

    Attributes
    ----------
    positions_mm : Tensor
        (N, 2) xy sample centres in mm.
    z_mm : Tensor
        (N,) nozzle height at each sample.
    speed_mm_s : Tensor
        (N,) traverse speed at each sample.
    ds_mm : Tensor
        (N,) arc-length spacing per sample.
    color_cmy : Tensor
        (3,) CMY command (uniform along stroke).
    n_samples : int
        Number of valid samples.
    """

    positions_mm: torch.Tensor
    z_mm: torch.Tensor
    speed_mm_s: torch.Tensor
    ds_mm: torch.Tensor
    color_cmy: torch.Tensor
    n_samples: int


@dataclass
class BatchedStrokeSamples:
    """Padded batch of :class:`StrokeSamples`.

    Attributes
    ----------
    positions_mm : Tensor
        (B, N_max, 2)
    z_mm : Tensor
        (B, N_max)
    speed_mm_s : Tensor
        (B, N_max)
    ds_mm : Tensor
        (B, N_max)
    color_cmy : Tensor
        (B, 3)
    valid_mask : Tensor
        (B, N_max) bool -- True for real samples, False for padding.
    n_max : int
        Padding length.
    """

    positions_mm: torch.Tensor
    z_mm: torch.Tensor
    speed_mm_s: torch.Tensor
    ds_mm: torch.Tensor
    color_cmy: torch.Tensor
    valid_mask: torch.Tensor
    n_max: int


# ============================================================================
# Differentiable interpolation helpers (pure functions, no side effects)
# ============================================================================

def _gpu_interp_1d(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
) -> torch.Tensor:
    """1-D piecewise-linear interpolation with clamped extrapolation.

    Parameters
    ----------
    x : Tensor
        Query points, arbitrary shape.
    xp : Tensor
        Sorted grid knots, shape (K,).
    fp : Tensor
        Values at knots, shape (K,).

    Returns
    -------
    Tensor
        Interpolated values, same shape as *x*.
    """
    x_flat = x.reshape(-1)
    idx = torch.searchsorted(xp, x_flat).clamp(1, len(xp) - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    f0 = fp[idx - 1]
    f1 = fp[idx]
    t = ((x_flat - x0) / (x1 - x0)).clamp(0.0, 1.0)
    result = f0 + t * (f1 - f0)
    return result.reshape(x.shape)


def _gpu_interp_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    xp: torch.Tensor,
    yp: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    """Bilinear interpolation on a 2-D regular grid.

    Parameters
    ----------
    x, y : Tensor
        Query coordinates, same shape.
    xp : Tensor
        Row knots, shape (Nx,), sorted.
    yp : Tensor
        Column knots, shape (Ny,), sorted.
    grid : Tensor
        Values, shape (Nx, Ny).

    Returns
    -------
    Tensor
        Interpolated values, same shape as *x*.
    """
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)

    ix = torch.searchsorted(xp, x_flat).clamp(1, len(xp) - 1)
    iy = torch.searchsorted(yp, y_flat).clamp(1, len(yp) - 1)

    tx = ((x_flat - xp[ix - 1]) / (xp[ix] - xp[ix - 1])).clamp(0.0, 1.0)
    ty = ((y_flat - yp[iy - 1]) / (yp[iy] - yp[iy - 1])).clamp(0.0, 1.0)

    f00 = grid[ix - 1, iy - 1]
    f01 = grid[ix - 1, iy]
    f10 = grid[ix, iy - 1]
    f11 = grid[ix, iy]

    result = (
        f00 * (1 - tx) * (1 - ty)
        + f01 * (1 - tx) * ty
        + f10 * tx * (1 - ty)
        + f11 * tx * ty
    )
    return result.reshape(x.shape)


def _gpu_trilinear_color(
    color_lut: torch.Tensor,
    cmy: torch.Tensor,
    domain_lo: float,
    domain_hi: float,
) -> torch.Tensor:
    """Trilinear lookup in a 3-D color LUT via :func:`F.grid_sample`.

    Parameters
    ----------
    color_lut : Tensor
        Shape ``(1, 3, Nc, Nm, Ny)`` -- volumetric LUT pre-shaped for
        ``grid_sample`` (batch=1, channels=3, depth=Nc, height=Nm, width=Ny).
    cmy : Tensor
        Shape ``(3,)`` or ``(B, 3)`` CMY command in *domain*.
    domain_lo, domain_hi : float
        CMY input range (usually 0..1).

    Returns
    -------
    Tensor
        Linear RGB, shape ``(3,)`` or ``(B, 3)``.
    """
    squeeze = cmy.dim() == 1
    if squeeze:
        cmy = cmy.unsqueeze(0)

    B = cmy.shape[0]
    # grid_sample expects grid in [-1, 1]
    normed = 2.0 * (cmy - domain_lo) / (domain_hi - domain_lo) - 1.0

    # grid_sample 3-D: input (N, C, D, H, W), grid (N, D_out, H_out, W_out, 3)
    # We want one output voxel per batch entry.
    grid = normed[:, [2, 1, 0]].reshape(B, 1, 1, 1, 3)
    lut_expanded = color_lut.expand(B, -1, -1, -1, -1)
    out = F.grid_sample(
        lut_expanded, grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    )
    rgb = out[:, :, 0, 0, 0]  # (B, 3)
    rgb = rgb.clamp(0.0, 1.0)

    return rgb.squeeze(0) if squeeze else rgb


# ============================================================================
# Part A+B: Simulator class (loader + canonical profile)
# ============================================================================

class GPUStampSimulator:
    """Config-driven GPU stamp-splatting airbrush renderer.

    Parameters
    ----------
    calibration_path : str or Path
        Path to ``calibration.yaml``.
    device : torch.device
        Target device (default: CUDA if available, else CPU).
    batch_size : int
        Number of parallel canvases.
    """

    def __init__(
        self,
        calibration_path: Union[str, Path],
        device: Optional[torch.device] = None,
        batch_size: int = 1,
    ) -> None:
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.device = device
        self.batch_size = batch_size

        # ---- load & validate config ------------------------------------
        self._cal = validators.load_calibration_config(calibration_path)
        self._cal_path = Path(calibration_path)

        # ---- grids -----------------------------------------------------
        self._z_grid = torch.tensor(
            self._cal.z_grid_mm, dtype=torch.float32, device=device,
        )
        self._speed_grid = torch.tensor(
            self._cal.speed_grid_mm_s, dtype=torch.float32, device=device,
        )

        # ---- LUTs ------------------------------------------------------
        self._radius_lut = torch.tensor(
            self._cal.radius_lut_mm, dtype=torch.float32, device=device,
        )
        self._mass_lut = torch.tensor(
            self._cal.mass_lut, dtype=torch.float32, device=device,
        )
        self._k_mass = float(self._cal.k_mass)

        # color LUT: (1, 3, Nc, Nm, Ny) for grid_sample
        self._color_lut = self._load_color_lut().to(device)
        self._color_domain = self._cal.color_axes.domain

        # optional gain LUTs
        self._color_gain_lut: Optional[torch.Tensor] = None
        if self._cal.color_gain_lut is not None:
            self._color_gain_lut = torch.tensor(
                self._cal.color_gain_lut,
                dtype=torch.float32, device=device,
            )

        self._layer_gain_lut: Optional[torch.Tensor] = None
        if self._cal.layer_gain_lut is not None:
            self._layer_gain_lut = torch.tensor(
                self._cal.layer_gain_lut,
                dtype=torch.float32, device=device,
            )

        # ---- paper white -----------------------------------------------
        self._paper_white = torch.tensor(
            self._cal.paper_white_rgb, dtype=torch.float32, device=device,
        )

        # ---- canvas geometry -------------------------------------------
        self._H, self._W = self._cal.render.canvas_hw
        wa_w, wa_h = self._cal.render.work_area_mm
        self._dpi_x = self._W / wa_w  # px per mm (horizontal)
        self._dpi_y = self._H / wa_h  # px per mm (vertical)

        # ---- sampling params -------------------------------------------
        self._max_step_mm = float(self._cal.sampling.max_step_mm)
        self._min_samples = int(self._cal.sampling.min_samples)
        self._normalize_xs = bool(self._cal.sampling.normalize_cross_section)

        # ---- build canonical profile -----------------------------------
        self._build_canonical_profile()

        logger.info(
            "GPUStampSimulator ready  device=%s  batch=%d  "
            "canvas=%dx%d  dpi=%.1fx%.1f",
            device, batch_size, self._W, self._H,
            self._dpi_x, self._dpi_y,
        )

    # -----------------------------------------------------------------
    # Color-LUT loader
    # -----------------------------------------------------------------
    def _load_color_lut(self) -> torch.Tensor:
        """Load or generate the (1, 3, Nc, Nm, Ny) color LUT."""
        ca = self._cal.color_axes
        Nc, Nm, Ny = ca.c_steps, ca.m_steps, ca.y_steps

        if self._cal.color_lut_path is not None:
            lut_path = Path(self._cal.color_lut_path)
            if not lut_path.is_absolute():
                lut_path = self._cal_path.parent / lut_path
            raw = torch.load(lut_path, map_location="cpu", weights_only=True)
            if raw.shape != (Nc, Nm, Ny, 3):
                raise ValueError(
                    f"color_lut shape {raw.shape} != expected "
                    f"({Nc}, {Nm}, {Ny}, 3)"
                )
        else:
            lo, hi = self._cal.color_axes.domain
            c_vals = torch.linspace(lo, hi, Nc)
            m_vals = torch.linspace(lo, hi, Nm)
            y_vals = torch.linspace(lo, hi, Ny)
            C, M, Y = torch.meshgrid(c_vals, m_vals, y_vals, indexing="ij")
            raw = torch.stack([1.0 - C, 1.0 - M, 1.0 - Y], dim=-1)

        # (Nc, Nm, Ny, 3) -> (1, 3, Nc, Nm, Ny) for F.grid_sample
        return raw.permute(3, 0, 1, 2).unsqueeze(0).float()

    # -----------------------------------------------------------------
    # Part B -- canonical 1-D radial profile
    # -----------------------------------------------------------------
    def _build_canonical_profile(self) -> None:
        """Pre-compute the 1-D radial intensity curve and its integral."""
        p = self._cal.profile
        N = p.profile_resolution

        u = torch.linspace(0.0, p.margin_factor, N, device=self.device)
        phi = torch.ones(N, device=self.device)

        core = p.core_frac
        sigma = max(p.skirt_sigma_frac, 1e-8)
        power = p.skirt_power
        margin = p.margin_factor

        skirt_mask = u > core
        skirt_u = u[skirt_mask] - core
        phi[skirt_mask] = torch.exp(
            -torch.pow(skirt_u / sigma, power)
        )
        phi[u > margin] = 0.0

        # integral = 2*pi * integral_0^margin phi(u)*u du  (trapezoid)
        integrand = phi * u
        du = u[1] - u[0]
        integral = 2.0 * math.pi * torch.trapezoid(integrand, dx=du.item())

        self._profile_1d = phi
        self._profile_u = u
        self._profile_integral = integral.item()
        self._profile_margin = margin

    def _sample_profile(self, u_query: torch.Tensor) -> torch.Tensor:
        """Look up phi(u) via linear interpolation on the 1-D table.

        Parameters
        ----------
        u_query : Tensor
            Normalised radial coordinate, arbitrary shape.

        Returns
        -------
        Tensor
            Profile values, same shape, clamped [0, 1].
        """
        N = len(self._profile_1d)
        idx_f = u_query * (N - 1) / self._profile_margin
        idx_f = idx_f.clamp(0.0, N - 1.0)
        idx_lo = idx_f.long().clamp(0, N - 2)
        idx_hi = idx_lo + 1
        frac = idx_f - idx_lo.float()
        vals = (
            self._profile_1d[idx_lo] * (1.0 - frac)
            + self._profile_1d[idx_hi] * frac
        )
        return vals.clamp(0.0, 1.0)

    # -----------------------------------------------------------------
    # Part C -- stroke sampler
    # -----------------------------------------------------------------
    def sample_stroke(self, stroke_dict: Dict) -> StrokeSamples:
        """Flatten a Bezier stroke into arc-length samples.

        Parameters
        ----------
        stroke_dict : dict
            Stroke descriptor with keys ``bezier``, ``z_profile``,
            ``speed_profile``, ``color_cmy``.

        Returns
        -------
        StrokeSamples
        """
        bz = stroke_dict["bezier"]
        p1 = torch.tensor(bz["p1"], dtype=torch.float32, device=self.device)
        p2 = torch.tensor(bz["p2"], dtype=torch.float32, device=self.device)
        p3 = torch.tensor(bz["p3"], dtype=torch.float32, device=self.device)
        p4 = torch.tensor(bz["p4"], dtype=torch.float32, device=self.device)

        polyline = geometry.bezier_cubic_polyline(
            p1, p2, p3, p4, max_err_mm=0.25, max_depth=12,
        )
        if polyline.shape[0] < 2:
            return StrokeSamples(
                positions_mm=torch.empty(0, 2, device=self.device),
                z_mm=torch.empty(0, device=self.device),
                speed_mm_s=torch.empty(0, device=self.device),
                ds_mm=torch.empty(0, device=self.device),
                color_cmy=torch.zeros(3, device=self.device),
                n_samples=0,
            )

        # arc lengths
        segs = polyline[1:] - polyline[:-1]
        seg_lens = torch.norm(segs, dim=1)
        cum = torch.cat([torch.zeros(1, device=self.device), seg_lens.cumsum(0)])
        total_len = cum[-1].item()
        if total_len < 1e-6:
            return StrokeSamples(
                positions_mm=torch.empty(0, 2, device=self.device),
                z_mm=torch.empty(0, device=self.device),
                speed_mm_s=torch.empty(0, device=self.device),
                ds_mm=torch.empty(0, device=self.device),
                color_cmy=torch.zeros(3, device=self.device),
                n_samples=0,
            )

        z0 = float(stroke_dict["z_profile"]["z0"])
        z1 = float(stroke_dict["z_profile"]["z1"])
        v0 = float(stroke_dict["speed_profile"]["v0"])
        v1 = float(stroke_dict["speed_profile"]["v1"])

        # adaptive spacing based on smallest expected radius
        z_range = torch.tensor([z0, z1], device=self.device)
        r_min = _gpu_interp_1d(z_range, self._z_grid, self._radius_lut).min()
        target_ds = min(0.25 * r_min.item(), self._max_step_mm)
        target_ds = max(target_ds, 1e-4)
        n_segs = max(self._min_samples, int(math.ceil(total_len / target_ds)))

        s_edges = torch.linspace(
            0.0, total_len, n_segs + 1, device=self.device,
        )
        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        ds = s_edges[1:] - s_edges[:-1]

        # interp positions
        positions = torch.zeros(n_segs, 2, device=self.device)
        for dim in range(2):
            positions[:, dim] = _gpu_interp_1d(
                s_centers, cum, polyline[:, dim],
            )

        frac = s_centers / total_len
        z_vals = z0 + (z1 - z0) * frac
        v_vals = v0 + (v1 - v0) * frac

        cc = stroke_dict["color_cmy"]
        cmy = torch.tensor(
            [cc["c"], cc["m"], cc["y"]],
            dtype=torch.float32, device=self.device,
        )

        return StrokeSamples(
            positions_mm=positions,
            z_mm=z_vals,
            speed_mm_s=v_vals,
            ds_mm=ds,
            color_cmy=cmy,
            n_samples=n_segs,
        )

    def sample_strokes_batch(
        self, stroke_dicts: List[Dict],
    ) -> BatchedStrokeSamples:
        """Sample a batch of strokes, pad to a common N_max.

        Parameters
        ----------
        stroke_dicts : list of dict
            One stroke descriptor per batch element.

        Returns
        -------
        BatchedStrokeSamples
        """
        samples = [self.sample_stroke(sd) for sd in stroke_dicts]
        B = len(samples)
        n_max = max(s.n_samples for s in samples)
        n_max = max(n_max, 1)

        pos = torch.zeros(B, n_max, 2, device=self.device)
        z = torch.zeros(B, n_max, device=self.device)
        v = torch.zeros(B, n_max, device=self.device)
        ds = torch.zeros(B, n_max, device=self.device)
        cmy = torch.zeros(B, 3, device=self.device)
        mask = torch.zeros(B, n_max, dtype=torch.bool, device=self.device)

        for i, s in enumerate(samples):
            n = s.n_samples
            if n > 0:
                pos[i, :n] = s.positions_mm
                z[i, :n] = s.z_mm
                v[i, :n] = s.speed_mm_s
                ds[i, :n] = s.ds_mm
                mask[i, :n] = True
            cmy[i] = s.color_cmy

        return BatchedStrokeSamples(
            positions_mm=pos, z_mm=z, speed_mm_s=v,
            ds_mm=ds, color_cmy=cmy, valid_mask=mask, n_max=n_max,
        )

    # -----------------------------------------------------------------
    # Part D -- GPU compositor
    # -----------------------------------------------------------------
    def reset(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Create a fresh white canvas.

        Parameters
        ----------
        batch_size : int, optional
            Override the default batch size.

        Returns
        -------
        Tensor
            ``(B, 3, H, W)`` canvas initialised to ``paper_white_rgb``.
        """
        B = batch_size if batch_size is not None else self.batch_size
        canvas = torch.ones(B, 3, self._H, self._W,
                            dtype=torch.float32, device=self.device)
        pw = self._paper_white.view(1, 3, 1, 1)
        canvas = canvas * pw
        return canvas

    def render_stroke(
        self,
        canvas: torch.Tensor,
        stroke_dict: Dict,
    ) -> torch.Tensor:
        """Render one stroke onto every canvas in the batch.

        Parameters
        ----------
        canvas : Tensor
            ``(B, 3, H, W)`` current canvas state.
        stroke_dict : dict
            Stroke descriptor.

        Returns
        -------
        Tensor
            Updated canvas (same shape).
        """
        samples = self.sample_stroke(stroke_dict)
        if samples.n_samples == 0:
            return canvas
        return self._composite_samples(canvas, samples)

    def render_strokes(
        self,
        canvas: torch.Tensor,
        strokes: List[Dict],
    ) -> torch.Tensor:
        """Render a sequence of strokes."""
        for sd in strokes:
            canvas = self.render_stroke(canvas, sd)
        return canvas

    def render_stroke_batch(
        self,
        canvas: torch.Tensor,
        stroke_dicts: List[Dict],
    ) -> torch.Tensor:
        """Render one distinct stroke per batch element.

        Parameters
        ----------
        canvas : Tensor
            ``(B, 3, H, W)``.
        stroke_dicts : list of dict
            Exactly ``B`` stroke descriptors.

        Returns
        -------
        Tensor
            Updated canvas.
        """
        B = canvas.shape[0]
        if len(stroke_dicts) != B:
            raise ValueError(
                f"Expected {B} stroke dicts for batch, got {len(stroke_dicts)}"
            )

        batched = self.sample_strokes_batch(stroke_dicts)
        return self._composite_samples_batch(canvas, batched)

    # ---- internal compositing ----------------------------------------

    def _composite_samples(
        self,
        canvas: torch.Tensor,
        samples: StrokeSamples,
    ) -> torch.Tensor:
        """Splat all samples of one stroke onto *every* batch element."""
        paint_rgb = _gpu_trilinear_color(
            self._color_lut, samples.color_cmy,
            self._color_domain[0], self._color_domain[1],
        )

        for i in range(samples.n_samples):
            canvas = self._splat_one(
                canvas,
                center_mm=samples.positions_mm[i],
                z=samples.z_mm[i],
                v=samples.speed_mm_s[i],
                ds=samples.ds_mm[i],
                paint_rgb=paint_rgb,
            )
        return canvas

    def _composite_samples_batch(
        self,
        canvas: torch.Tensor,
        batched: BatchedStrokeSamples,
    ) -> torch.Tensor:
        """Splat a distinct stroke per batch element (sequential samples)."""
        B = canvas.shape[0]

        paint_rgb = _gpu_trilinear_color(
            self._color_lut, batched.color_cmy,
            self._color_domain[0], self._color_domain[1],
        )  # (B, 3)

        for j in range(batched.n_max):
            mask = batched.valid_mask[:, j]
            if not mask.any():
                continue
            for b_idx in range(B):
                if not mask[b_idx]:
                    continue
                canvas[b_idx:b_idx+1] = self._splat_one(
                    canvas[b_idx:b_idx+1],
                    center_mm=batched.positions_mm[b_idx, j],
                    z=batched.z_mm[b_idx, j],
                    v=batched.speed_mm_s[b_idx, j],
                    ds=batched.ds_mm[b_idx, j],
                    paint_rgb=paint_rgb[b_idx],
                )
        return canvas

    def _splat_one(
        self,
        canvas: torch.Tensor,
        center_mm: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        ds: torch.Tensor,
        paint_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """Splat a single stamp onto the canvas.

        Parameters
        ----------
        canvas : Tensor
            ``(B, 3, H, W)``.
        center_mm : Tensor
            ``(2,)`` stamp centre in mm.
        z : Tensor
            Scalar nozzle height.
        v : Tensor
            Scalar speed.
        ds : Tensor
            Scalar arc-length spacing.
        paint_rgb : Tensor
            ``(3,)`` paint colour in linear RGB.

        Returns
        -------
        Tensor
            Updated canvas.
        """
        R = _gpu_interp_1d(
            z.unsqueeze(0), self._z_grid, self._radius_lut,
        ).squeeze(0)
        mass = _gpu_interp_2d(
            z.unsqueeze(0), v.unsqueeze(0),
            self._z_grid, self._speed_grid, self._mass_lut,
        ).squeeze(0)

        margin = self._profile_margin
        extent_mm = R * margin

        cx_px = center_mm[0] * self._dpi_x
        cy_px = center_mm[1] * self._dpi_y
        mx_px = extent_mm * self._dpi_x
        my_px = extent_mm * self._dpi_y

        x_min = int(torch.floor(cx_px - mx_px).item())
        x_max = int(torch.ceil(cx_px + mx_px).item()) + 1
        y_min = int(torch.floor(cy_px - my_px).item())
        y_max = int(torch.ceil(cy_px + my_px).item()) + 1

        x_min = max(x_min, 0)
        x_max = min(x_max, self._W)
        y_min = max(y_min, 0)
        y_max = min(y_max, self._H)

        if x_max <= x_min or y_max <= y_min:
            return canvas

        # pixel coordinate grids -> mm distances from stamp centre
        yy = torch.arange(
            y_min, y_max, dtype=torch.float32, device=self.device,
        )
        xx = torch.arange(
            x_min, x_max, dtype=torch.float32, device=self.device,
        )
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")

        dx_mm = (gx - cx_px) / self._dpi_x
        dy_mm = (gy - cy_px) / self._dpi_y
        dist_mm = torch.sqrt(dx_mm * dx_mm + dy_mm * dy_mm)

        u = dist_mm / R.clamp(min=1e-6)
        phi = self._sample_profile(u)

        if self._normalize_xs and self._profile_integral > 0:
            mass_per_sample = mass * ds
            alpha = self._k_mass * mass_per_sample * phi / self._profile_integral
        else:
            alpha = self._k_mass * mass * ds * phi

        alpha = alpha.clamp(0.0, 1.0)

        # transmission compositing (out-of-place for autograd safety)
        alpha_3d = alpha.unsqueeze(0).unsqueeze(0)  # (1, 1, Hk, Wk)
        paint_4d = paint_rgb.view(1, 3, 1, 1)

        T = (1.0 - alpha_3d) + alpha_3d * paint_4d  # (1, 3, Hk, Wk)

        roi = canvas[:, :, y_min:y_max, x_min:x_max]
        new_roi = (roi * T).clamp(0.0, 1.0)

        canvas = canvas.clone()
        canvas[:, :, y_min:y_max, x_min:x_max] = new_roi

        return canvas

    # -----------------------------------------------------------------
    # Part E -- preview & comparison tools
    # -----------------------------------------------------------------

    def render_dot_sheet(
        self,
        z_values: List[float],
        color_cmy: Tuple[float, float, float] = (0.8, 0.2, 0.1),
        spacing_mm: float = 20.0,
        speed: float = 60.0,
    ) -> torch.Tensor:
        """Render a grid of isolated dots at various z heights.

        Parameters
        ----------
        z_values : list of float
            Z heights (mm) to test.
        color_cmy : tuple
            CMY recipe for all dots.
        spacing_mm : float
            Grid spacing between dot centres (mm).
        speed : float
            Traverse speed for all dots (mm/s).

        Returns
        -------
        Tensor
            ``(1, 3, H, W)`` rendered canvas.
        """
        canvas = self.reset(batch_size=1)
        n_cols = max(1, int(self._cal.render.work_area_mm[0] / spacing_mm) - 1)
        c, m, y = color_cmy

        for row_idx, z in enumerate(z_values):
            cy_mm = spacing_mm * (row_idx + 1)
            if cy_mm > self._cal.render.work_area_mm[1] - spacing_mm:
                break
            for col_idx in range(min(n_cols, 3)):
                cx_mm = spacing_mm * (col_idx + 1)
                tiny_len = 0.01
                stroke = {
                    "bezier": {
                        "p1": (cx_mm, cy_mm),
                        "p2": (cx_mm + tiny_len, cy_mm),
                        "p3": (cx_mm + tiny_len, cy_mm),
                        "p4": (cx_mm + tiny_len, cy_mm),
                    },
                    "z_profile": {"z0": z, "z1": z},
                    "speed_profile": {"v0": speed, "v1": speed},
                    "color_cmy": {"c": c, "m": m, "y": y},
                }
                canvas = self.render_stroke(canvas, stroke)
        return canvas

    def render_line_sheet(
        self,
        z_values: List[float],
        speed_values: List[float],
        color_cmy: Tuple[float, float, float] = (0.8, 0.2, 0.1),
        length_mm: float = 40.0,
        spacing_mm: float = 20.0,
    ) -> torch.Tensor:
        """Render a grid of straight lines: rows = z, columns = speed.

        Returns
        -------
        Tensor
            ``(1, 3, H, W)``.
        """
        canvas = self.reset(batch_size=1)
        c, m, y = color_cmy
        margin = spacing_mm

        for row_idx, z in enumerate(z_values):
            cy_mm = margin + row_idx * spacing_mm
            if cy_mm > self._cal.render.work_area_mm[1] - margin:
                break
            for col_idx, spd in enumerate(speed_values):
                cx_start = margin + col_idx * (length_mm + margin)
                cx_end = cx_start + length_mm
                if cx_end > self._cal.render.work_area_mm[0] - 5.0:
                    break
                stroke = {
                    "bezier": {
                        "p1": (cx_start, cy_mm),
                        "p2": (cx_start + length_mm / 3, cy_mm),
                        "p3": (cx_start + 2 * length_mm / 3, cy_mm),
                        "p4": (cx_end, cy_mm),
                    },
                    "z_profile": {"z0": z, "z1": z},
                    "speed_profile": {"v0": spd, "v1": spd},
                    "color_cmy": {"c": c, "m": m, "y": y},
                }
                canvas = self.render_stroke(canvas, stroke)
        return canvas

    def render_color_swatches(
        self,
        cmy_recipes: List[Tuple[float, float, float]],
        swatch_size_mm: float = 15.0,
        fill_z: float = 6.0,
        fill_speed: float = 30.0,
    ) -> torch.Tensor:
        """Dense-fill swatch for each CMY recipe.

        Returns
        -------
        Tensor
            ``(1, 3, H, W)``.
        """
        canvas = self.reset(batch_size=1)
        margin = 10.0
        cols = max(
            1,
            int((self._cal.render.work_area_mm[0] - margin)
                / (swatch_size_mm + margin)),
        )
        for idx, (c, m, y) in enumerate(cmy_recipes):
            row = idx // cols
            col = idx % cols
            x0 = margin + col * (swatch_size_mm + margin)
            y0 = margin + row * (swatch_size_mm + margin)
            if (y0 + swatch_size_mm
                    > self._cal.render.work_area_mm[1] - margin):
                break

            # fill with horizontal lines
            line_spacing = 0.5
            yy = y0
            while yy < y0 + swatch_size_mm:
                stroke = {
                    "bezier": {
                        "p1": (x0, yy),
                        "p2": (x0 + swatch_size_mm / 3, yy),
                        "p3": (x0 + 2 * swatch_size_mm / 3, yy),
                        "p4": (x0 + swatch_size_mm, yy),
                    },
                    "z_profile": {"z0": fill_z, "z1": fill_z},
                    "speed_profile": {"v0": fill_speed, "v1": fill_speed},
                    "color_cmy": {"c": c, "m": m, "y": y},
                }
                canvas = self.render_stroke(canvas, stroke)
                yy += line_spacing
        return canvas

    def render_layering_staircase(
        self,
        colors: List[Tuple[float, float, float]],
        rect_size_mm: float = 30.0,
        overlap_mm: float = 15.0,
        fill_z: float = 6.0,
        fill_speed: float = 30.0,
    ) -> torch.Tensor:
        """Render overlapping rectangles to visualise N-layer stacking.

        Returns
        -------
        Tensor
            ``(1, 3, H, W)``.
        """
        canvas = self.reset(batch_size=1)
        margin = 10.0
        line_spacing = 0.5

        for cidx, (c, m, y) in enumerate(colors):
            x0 = margin + cidx * overlap_mm
            y0 = margin
            yy = y0
            while yy < y0 + rect_size_mm:
                stroke = {
                    "bezier": {
                        "p1": (x0, yy),
                        "p2": (x0 + rect_size_mm / 3, yy),
                        "p3": (x0 + 2 * rect_size_mm / 3, yy),
                        "p4": (x0 + rect_size_mm, yy),
                    },
                    "z_profile": {"z0": fill_z, "z1": fill_z},
                    "speed_profile": {"v0": fill_speed, "v1": fill_speed},
                    "color_cmy": {"c": c, "m": m, "y": y},
                }
                canvas = self.render_stroke(canvas, stroke)
                yy += line_spacing
        return canvas

    # ---- Measurement comparison --------------------------------------

    def compare_to_measurements(
        self,
    ) -> Dict[str, Any]:
        """Simulate every raw measurement and compute delta-E.

        Returns
        -------
        dict
            ``{block: {id: {target_rgb, sim_rgb, delta_e}}}``
        """
        results: Dict[str, Any] = {}
        rm = self._cal.raw_measurements

        # dots
        dot_results: Dict[str, Any] = {}
        for did, dot in rm.dots.items():
            sim_rgb = self._simulate_dot_rgb(
                dot.z_mm, dot.color_recipe_cmy, dot.diameter_mm,
            )
            target = torch.tensor(dot.center_rgb, device=self.device)
            de = self._delta_e(target, sim_rgb)
            dot_results[did] = {
                "target_rgb": list(dot.center_rgb),
                "sim_rgb": sim_rgb.tolist(),
                "delta_e": de,
            }
        results["dots"] = dot_results

        # lines
        line_results: Dict[str, Any] = {}
        for lid, line in rm.lines.items():
            sim_rgb = self._simulate_line_center_rgb(
                line.z_mm, line.speed_mm_s, line.color_recipe_cmy,
            )
            target = torch.tensor(line.center_rgb, device=self.device)
            de = self._delta_e(target, sim_rgb)
            line_results[lid] = {
                "target_rgb": list(line.center_rgb),
                "sim_rgb": sim_rgb.tolist(),
                "delta_e": de,
            }
        results["lines"] = line_results

        # swatches
        swatch_results: Dict[str, Any] = {}
        for sid, sw in rm.swatches.items():
            sim_rgb = _gpu_trilinear_color(
                self._color_lut,
                torch.tensor(sw.cmy_command, device=self.device),
                self._color_domain[0], self._color_domain[1],
            )
            target = torch.tensor(sw.interior_rgb, device=self.device)
            de = self._delta_e(target, sim_rgb)
            swatch_results[sid] = {
                "target_rgb": list(sw.interior_rgb),
                "sim_rgb": sim_rgb.tolist(),
                "delta_e": de,
            }
        results["swatches"] = swatch_results

        return results

    def _simulate_dot_rgb(
        self,
        z: float,
        cmy: Tuple[float, float, float],
        diameter_mm: float,
    ) -> torch.Tensor:
        """Simulate a single dot and return the centre pixel RGB."""
        small = self.reset(batch_size=1)
        cx = self._cal.render.work_area_mm[0] / 2
        cy = self._cal.render.work_area_mm[1] / 2
        tiny = 0.01
        stroke = {
            "bezier": {
                "p1": (cx, cy), "p2": (cx + tiny, cy),
                "p3": (cx + tiny, cy), "p4": (cx + tiny, cy),
            },
            "z_profile": {"z0": z, "z1": z},
            "speed_profile": {"v0": 60.0, "v1": 60.0},
            "color_cmy": {"c": cmy[0], "m": cmy[1], "y": cmy[2]},
        }
        small = self.render_stroke(small, stroke)
        px_x = int(cx * self._dpi_x)
        px_y = int(cy * self._dpi_y)
        px_x = min(max(px_x, 0), self._W - 1)
        px_y = min(max(px_y, 0), self._H - 1)
        return small[0, :, px_y, px_x]

    def _simulate_line_center_rgb(
        self,
        z: float,
        speed: float,
        cmy: Tuple[float, float, float],
    ) -> torch.Tensor:
        """Simulate a straight line and return centre-strip mean RGB."""
        small = self.reset(batch_size=1)
        wa_x = self._cal.render.work_area_mm[0]
        wa_y = self._cal.render.work_area_mm[1]
        cy = wa_y / 2
        x0, x1 = wa_x * 0.2, wa_x * 0.8
        stroke = {
            "bezier": {
                "p1": (x0, cy), "p2": (x0 + (x1 - x0) / 3, cy),
                "p3": (x0 + 2 * (x1 - x0) / 3, cy),
                "p4": (x1, cy),
            },
            "z_profile": {"z0": z, "z1": z},
            "speed_profile": {"v0": speed, "v1": speed},
            "color_cmy": {"c": cmy[0], "m": cmy[1], "y": cmy[2]},
        }
        small = self.render_stroke(small, stroke)
        py = int(cy * self._dpi_y)
        px_lo = int(wa_x * 0.35 * self._dpi_x)
        px_hi = int(wa_x * 0.65 * self._dpi_x)
        py = min(max(py, 0), self._H - 1)
        px_lo = max(px_lo, 0)
        px_hi = min(px_hi, self._W)
        strip = small[0, :, py, px_lo:px_hi]
        return strip.mean(dim=1)

    def _delta_e(
        self, rgb_a: torch.Tensor, rgb_b: torch.Tensor,
    ) -> float:
        """Compute CIEDE2000 between two linear-RGB vectors."""
        a = rgb_a.unsqueeze(0).cpu()
        b = rgb_b.unsqueeze(0).cpu()
        lab_a = color_utils.rgb_to_lab(a)
        lab_b = color_utils.rgb_to_lab(b)
        de = color_utils.delta_e2000(lab_a, lab_b)
        return float(de.item())

    # ---- Visualisation helpers ---------------------------------------

    def error_heatmap(
        self,
        measured_rgb: torch.Tensor,
        simulated_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """Per-pixel absolute RGB difference.

        Parameters
        ----------
        measured_rgb, simulated_rgb : Tensor
            ``(3, H, W)`` linear RGB images.

        Returns
        -------
        Tensor
            ``(H, W)`` mean-channel absolute error.
        """
        return (measured_rgb - simulated_rgb).abs().mean(dim=0)

    def radial_profile_plot(
        self,
        z: float,
        measured_center_mid_edge: Tuple[
            Tuple[float, float, float],
            Tuple[float, float, float],
            Tuple[float, float, float],
        ],
    ) -> plt.Figure:
        """Overlay measured zoned RGB vs simulated radial profile.

        Parameters
        ----------
        z : float
            Nozzle height (mm).
        measured_center_mid_edge : tuple of 3 RGB tuples
            (center_rgb, mid_rgb, edge_rgb) measured from scan.

        Returns
        -------
        matplotlib.Figure
        """
        R = _gpu_interp_1d(
            torch.tensor([z], device=self.device),
            self._z_grid, self._radius_lut,
        ).item()
        u = torch.linspace(0.0, self._profile_margin, 256, device=self.device)
        phi = self._sample_profile(u).cpu().numpy()
        r_mm = u.cpu().numpy() * R

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(r_mm, phi, "b-", label="simulated phi(r)")

        center, mid, edge = measured_center_mid_edge
        # luminance proxy for measured points
        def _lum(rgb: Tuple[float, float, float]) -> float:
            return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

        bg_lum = 1.0
        lum_vals = [1.0 - _lum(center) / bg_lum,
                    1.0 - _lum(mid) / bg_lum,
                    1.0 - _lum(edge) / bg_lum]
        r_positions = [0.0, R * 0.5, R * 0.9]
        ax.plot(r_positions, lum_vals, "ro", label="measured zones")

        ax.set_xlabel("r (mm)")
        ax.set_ylabel("relative intensity")
        ax.set_title(f"Radial profile  z={z:.1f} mm  R={R:.2f} mm")
        ax.legend()
        fig.tight_layout()
        return fig

    def line_cross_section_plot(
        self,
        z: float,
        speed: float,
        measured_center_shoulder_edge: Tuple[
            Tuple[float, float, float],
            Tuple[float, float, float],
            Tuple[float, float, float],
        ],
        color_cmy: Tuple[float, float, float] = (0.8, 0.2, 0.1),
    ) -> plt.Figure:
        """Cross-section intensity comparison: simulated vs measured.

        Parameters
        ----------
        z : float
            Nozzle height.
        speed : float
            Traverse speed.
        measured_center_shoulder_edge : tuple of 3 RGB tuples
        color_cmy : tuple
            CMY recipe used.

        Returns
        -------
        matplotlib.Figure
        """
        canvas = self.reset(batch_size=1)
        wa_x = self._cal.render.work_area_mm[0]
        wa_y = self._cal.render.work_area_mm[1]
        cy = wa_y / 2
        x0, x1 = wa_x * 0.2, wa_x * 0.8
        c, m, y = color_cmy
        stroke = {
            "bezier": {
                "p1": (x0, cy), "p2": (x0 + (x1 - x0) / 3, cy),
                "p3": (x0 + 2 * (x1 - x0) / 3, cy),
                "p4": (x1, cy),
            },
            "z_profile": {"z0": z, "z1": z},
            "speed_profile": {"v0": speed, "v1": speed},
            "color_cmy": {"c": c, "m": m, "y": y},
        }
        canvas = self.render_stroke(canvas, stroke)

        px_mid_x = int(wa_x * 0.5 * self._dpi_x)
        col_slice = canvas[0, :, :, px_mid_x].cpu().numpy()
        lum = 0.2126 * col_slice[0] + 0.7152 * col_slice[1] + 0.0722 * col_slice[2]
        y_px = np.arange(len(lum))
        y_mm = y_px / self._dpi_y

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(y_mm, 1.0 - lum, "b-", label="simulated")

        R = _gpu_interp_1d(
            torch.tensor([z], device=self.device),
            self._z_grid, self._radius_lut,
        ).item()
        center, shoulder, edge = measured_center_shoulder_edge

        def _lum_t(rgb: Tuple[float, float, float]) -> float:
            return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

        r_pos = [cy, cy - R * 0.5, cy - R * 0.9]
        m_vals = [1.0 - _lum_t(center),
                  1.0 - _lum_t(shoulder),
                  1.0 - _lum_t(edge)]
        ax.plot(r_pos, m_vals, "ro", label="measured zones")

        ax.set_xlabel("y (mm)")
        ax.set_ylabel("1 - luminance (darkness)")
        ax.set_title(f"Line cross-section  z={z:.1f}  v={speed:.0f}")
        ax.legend()
        fig.tight_layout()
        return fig

    def summary_table(self) -> Any:
        """Build a Rich Table comparing simulated vs measured RGB.

        Returns
        -------
        rich.table.Table
        """
        from rich.table import Table

        results = self.compare_to_measurements()
        table = Table(title="Calibration Validation Summary")
        table.add_column("Block", style="cyan")
        table.add_column("ID")
        table.add_column("Target RGB")
        table.add_column("Sim RGB")
        table.add_column("dE", justify="right")
        table.add_column("Status")

        for block_name, block_data in results.items():
            for mid, entry in block_data.items():
                de = entry["delta_e"]
                status = "PASS" if de < 3.0 else "WARN" if de < 5.0 else "FAIL"
                style = (
                    "green" if status == "PASS"
                    else "yellow" if status == "WARN"
                    else "red"
                )
                t_str = "({:.2f},{:.2f},{:.2f})".format(*entry["target_rgb"])
                s_str = "({:.2f},{:.2f},{:.2f})".format(*entry["sim_rgb"])
                table.add_row(
                    block_name, mid, t_str, s_str,
                    f"{de:.2f}", f"[{style}]{status}[/{style}]",
                )
        return table
