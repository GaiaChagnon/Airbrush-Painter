"""CPU reference renderer for ground-truth stroke simulation.

This is a deterministic, pure-CPU implementation using OpenCV distance transforms
to create realistic airbrush spray profiles with flat cores and Gaussian skirts.

Architecture:
    - Bézier → polyline with adaptive flattening (error control in mm)
    - Rasterize 1-px centerline with OpenCV
    - Distance transform to get radial distance from centerline
    - Build radial opacity profile: flat core + Gaussian skirt
    - Mass-per-mm scaling (not normalized Gaussians)
    - Speed-aware width & deposition
    - CMY → linear RGB via color_lut (trilinear interpolation)
    - Alpha-over compositing (linear RGB, clamped accumulation)
    - Visibility gates: skip imperceptible strokes

Invariants:
    - All geometry in mm; conversions to px happen at boundaries only
    - Linear RGB [0,1] throughout; sRGB conversion at I/O only
    - FP32 precision (no mixed precision, no autocast)
    - Deterministic (seeded noise for speckle)
    - Center of stroke MUST be visibly darker than paper

Usage:
    from src.airbrush_simulator.cpu_reference import CPUReferenceRenderer
    from src.utils import fs
    
    # Load configs and LUTs
    sim_cfg = fs.load_yaml("configs/sim/physics_v1.yaml")
    env_cfg = fs.load_yaml("configs/env_airbrush_v1.yaml")
    luts = load_toy_luts()  # or load calibrated LUTs
    
    # Initialize renderer
    renderer = CPUReferenceRenderer(sim_cfg, env_cfg, luts)
    
    # Render a stroke
    canvas = np.ones((H, W, 3), dtype=np.float32)
    alpha = np.zeros((H, W), dtype=np.float32)
    stroke_dict = {...}
    
    canvas_new, alpha_new = renderer.render_stroke(canvas, alpha, stroke_dict)

LUT format:
    - color_lut: (Nc, Nm, Ny, 3) FP32, CMY [0,1]³ → linear RGB [0,1]³
    - alpha_lut: (Nz, Nv) FP32, (z, v) → coverage [0,1] (legacy; not used in new model)
    - psf_lut: (Nz, Nv) FP32, (z, v) → σ_px (legacy; not used in new model)

Notes:
    - Sampling step: Δs ≤ max_step_mm from config
    - Visibility gates enforce min coverage & luminance drop
    - Width & mass controlled by renderer_cpu.v1.yaml config
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from src.utils import compute, geometry, color as color_utils, metrics, fs

logger = logging.getLogger(__name__)


class CPUReferenceRenderer:
    """CPU-based reference renderer using OpenCV distance-transform spray model.
    
    Provides deterministic, non-differentiable rendering with realistic airbrush
    physics: flat core + Gaussian skirt, speed-aware width and mass deposition.
    
    Attributes
    ----------
    sim_cfg : dict
        Physics config (LUT paths, CPU renderer config path)
    env_cfg : dict
        Environment config (work_area_mm, render_px, origin)
    luts : dict
        Lookup tables: {color_lut, alpha_lut, psf_lut} as numpy arrays
    cpu_cfg : dict
        CPU renderer config (width model, deposition, profile, visibility)
    stroke_bounds : dict
        Schema bounds for clamping: {x, y, z, speed, cmy}
    dpi : tuple
        Resolution in pixels/mm (px_per_mm_x, px_per_mm_y)
    """
    
    def __init__(
        self,
        sim_cfg: Dict,
        env_cfg: Dict,
        luts: Dict[str, np.ndarray]
    ):
        """Initialize CPU reference renderer.
        
        Parameters
        ----------
        sim_cfg : dict
            Physics config with LUT metadata and renderer_cpu_config path
        env_cfg : dict
            Environment config (work_area_mm, render_px, origin)
        luts : dict
            Lookup tables as numpy arrays:
            - color_lut: (Nc, Nm, Ny, 3) FP32
            - alpha_lut: (Nz, Nv) FP32 (legacy)
            - psf_lut: (Nz, Nv) FP32 (legacy)
        """
        self.sim_cfg = sim_cfg
        self.env_cfg = env_cfg
        self.luts = luts
        
        # Load CPU renderer config
        cpu_cfg_path = sim_cfg.get('renderer_cpu_config', 'configs/sim/renderer_cpu.v1.yaml')
        self.cpu_cfg = fs.load_yaml(Path(cpu_cfg_path))
        
        # Extract bounds from stroke schema
        self.stroke_bounds = {
            'x': (0.0, 210.0),
            'y': (0.0, 297.0),
            'z': (0.0, 30.0),
            'speed': (1.0, 300.0),
            'cmy': (0.0, 1.0)
        }
        
        # Compute resolution (pixels per mm)
        work_area_mm = env_cfg.get('work_area_mm', (210.0, 297.0))
        render_px = env_cfg.get('render_px', (908, 1280))
        self.dpi = (
            render_px[1] / work_area_mm[0],  # px_per_mm_x
            render_px[0] / work_area_mm[1]   # px_per_mm_y
        )
        self.dpi_avg = 0.5 * (self.dpi[0] + self.dpi[1])
        
        # Canvas dimensions
        self.canvas_h_px = render_px[0]
        self.canvas_w_px = render_px[1]
        
        # Validate LUTs
        self._validate_luts()
        
        # Noise generator (for speckle)
        self.rng = np.random.RandomState(self.cpu_cfg.get('randomness', {}).get('seed', 42))
        
        logger.info(
            f"CPUReferenceRenderer initialized (OpenCV distance-transform mode): "
            f"canvas={render_px}, work_area={work_area_mm} mm, "
            f"dpi={self.dpi[0]:.2f}×{self.dpi[1]:.2f} px/mm"
        )
    
    def _validate_luts(self):
        """Validate LUT shapes and dtypes."""
        required_keys = {'color_lut'}  # alpha_lut and psf_lut are legacy, not required
        missing = required_keys - set(self.luts.keys())
        if missing:
            raise ValueError(f"Missing required LUTs: {missing}")
        
        # Check dtypes (all FP32)
        for name, lut in self.luts.items():
            if lut.dtype != np.float32:
                raise TypeError(f"LUT {name} must be FP32, got {lut.dtype}")
        
        # Check color LUT shape
        color_shape = self.luts['color_lut'].shape
        if len(color_shape) != 4 or color_shape[-1] != 3:
            raise ValueError(
                f"color_lut must be (Nc, Nm, Ny, 3), got {color_shape}"
            )
        
        # Validate value ranges
        color_min, color_max = self.luts['color_lut'].min(), self.luts['color_lut'].max()
        if color_min < 0.0 or color_max > 1.0:
            logger.warning(f"color_lut values outside [0,1]: [{color_min}, {color_max}]")
    
    def project_params(self, stroke_dict: Dict) -> Dict:
        """Clamp stroke parameters to schema bounds (mm-space).
        
        Parameters
        ----------
        stroke_dict : dict
            Stroke dictionary (may have out-of-bounds values)
        
        Returns
        -------
        dict
            Clamped stroke dictionary
        """
        clamped = stroke_dict.copy()
        
        # Clamp control points
        x_min, x_max = self.stroke_bounds['x']
        y_min, y_max = self.stroke_bounds['y']
        
        for pt in ['p1', 'p2', 'p3', 'p4']:
            x, y = clamped['bezier'][pt]
            clamped['bezier'][pt] = (
                np.clip(x, x_min, x_max),
                np.clip(y, y_min, y_max)
            )
        
        # Clamp z profile
        z_min, z_max = self.stroke_bounds['z']
        clamped['z_profile']['z0'] = np.clip(
            clamped['z_profile']['z0'], z_min, z_max
        )
        clamped['z_profile']['z1'] = np.clip(
            clamped['z_profile']['z1'], z_min, z_max
        )
        
        # Clamp speed profile
        v_min, v_max = self.stroke_bounds['speed']
        clamped['speed_profile']['v0'] = np.clip(
            clamped['speed_profile']['v0'], v_min, v_max
        )
        clamped['speed_profile']['v1'] = np.clip(
            clamped['speed_profile']['v1'], v_min, v_max
        )
        
        # Clamp CMY
        c_min, c_max = self.stroke_bounds['cmy']
        for component in ['c', 'm', 'y']:
            clamped['color_cmy'][component] = np.clip(
                clamped['color_cmy'][component], c_min, c_max
            )
        
        return clamped
    
    def _width_mm(self, z: float, v: float) -> float:
        """Compute spray width in mm from Z and speed.
        
        Parameters
        ----------
        z : float
            Nozzle height (mm)
        v : float
            Speed (mm/s)
        
        Returns
        -------
        float
            Width in mm
        """
        width_model = self.cpu_cfg['width_model']
        
        # Interpolate min/max width at this Z
        z_knots = np.array(width_model['z_knots_mm'], dtype=np.float32)
        width_min = np.interp(z, z_knots, width_model['width_min_mm'])
        width_max = np.interp(z, z_knots, width_model['width_max_mm'])
        
        # Speed scaling factor
        v_knots = np.array(width_model['v_knots_mm_s'], dtype=np.float32)
        width_scale = np.interp(v, v_knots, width_model['width_scale'])
        
        # Compute width (geometric mean of min/max, scaled by speed)
        w = np.sqrt(width_min * width_max) * width_scale
        
        # Clamp to [min, max]
        return float(np.clip(w, width_min, width_max))
    
    def _mass_per_mm(self, z: float, v: float) -> float:
        """Compute mass (opacity) per mm of path length.
        
        Parameters
        ----------
        z : float
            Nozzle height (mm)
        v : float
            Speed (mm/s)
        
        Returns
        -------
        float
            Mass per mm (opacity units)
        """
        deposition = self.cpu_cfg['deposition']
        
        # Interpolate mass per second at this Z
        z_knots = np.array(deposition['z_knots_mm'], dtype=np.float32)
        mass_per_sec = np.interp(z, z_knots, deposition['mass_per_sec'])
        
        # Convert to per-mm via speed scaling
        speed_exp = deposition.get('speed_exponent', 1.0)
        mass_per_mm = mass_per_sec / max(v, 1e-6) ** abs(speed_exp)
        
        return float(mass_per_mm)
    
    def _build_radial_profile(
        self,
        dist_px: np.ndarray,
        width_mm: float
    ) -> np.ndarray:
        """Build radial opacity profile from distance map.
        
        Parameters
        ----------
        dist_px : np.ndarray
            Distance transform (pixels from centerline), shape (H, W)
        width_mm : float
            Spray width in mm
        
        Returns
        -------
        np.ndarray
            Radial profile phi(r), shape (H, W), range [0, 1]
        """
        profile_cfg = self.cpu_cfg['profile']
        
        # Convert distance to mm
        dist_mm = dist_px / self.dpi_avg
        
        # Compute radii
        r_core = profile_cfg['core_frac'] * 0.5 * width_mm
        sigma_skirt = profile_cfg['skirt_sigma_frac'] * 0.5 * width_mm
        r_max = 0.5 * width_mm * profile_cfg.get('margin_factor', 1.5)
        power = profile_cfg.get('skirt_power', 1.8)
        
        # Build profile: flat core + Gaussian skirt
        phi = np.ones_like(dist_mm, dtype=np.float32)
        
        # Skirt region (r > r_core)
        skirt_mask = dist_mm > r_core
        skirt_dist = dist_mm[skirt_mask] - r_core
        phi[skirt_mask] = np.exp(-np.power(skirt_dist / max(sigma_skirt, 1e-6), power))
        
        # Zero beyond r_max
        phi[dist_mm > r_max] = 0.0
        
        # Optional speckle (deterministic noise)
        if self.cpu_cfg.get('randomness', {}).get('speckle', False):
            gain = self.cpu_cfg.get('randomness', {}).get('speckle_gain', 0.08)
            scale = self.cpu_cfg.get('randomness', {}).get('speckle_scale', 2.0)
            noise = self._generate_speckle(phi.shape, scale)
            phi *= (1.0 + gain * noise)
            phi = np.clip(phi, 0.0, 1.0)
        
        return phi
    
    def _generate_speckle(self, shape: Tuple[int, int], scale: float) -> np.ndarray:
        """Generate deterministic speckle noise.
        
        Parameters
        ----------
        shape : tuple
            (H, W) shape
        scale : float
            Feature size in pixels
        
        Returns
        -------
        np.ndarray
            Noise in [-1, 1], shape (H, W)
        """
        h, w = shape
        # Generate low-res noise and upscale
        h_low = max(1, int(h / scale))
        w_low = max(1, int(w / scale))
        noise_low = self.rng.randn(h_low, w_low).astype(np.float32)
        noise = cv2.resize(noise_low, (w, h), interpolation=cv2.INTER_LINEAR)
        # Normalize to [-1, 1]
        noise = (noise - noise.mean()) / max(noise.std(), 1e-6)
        return np.clip(noise, -1.0, 1.0)
    
    def _check_visibility(
        self,
        canvas_before: np.ndarray,
        canvas_after: np.ndarray,
        alpha_delta: np.ndarray,
        core_mask: Optional[np.ndarray] = None
    ) -> bool:
        """Check if stroke is visible enough to keep.
        
        Parameters
        ----------
        canvas_before : np.ndarray
            Canvas before stroke, shape (H, W, 3)
        canvas_after : np.ndarray
            Canvas after stroke, shape (H, W, 3)
        alpha_delta : np.ndarray
            Change in alpha, shape (H, W)
        core_mask : np.ndarray, optional
            Mask for core region (for center luminance check)
        
        Returns
        -------
        bool
            True if stroke is visible, False if it should be skipped
        """
        vis_cfg = self.cpu_cfg['visibility']
        
        # Coverage check
        coverage = np.mean(alpha_delta)
        if coverage < vis_cfg['min_stroke_coverage']:
            logger.debug(f"Stroke skipped: coverage {coverage:.6f} < {vis_cfg['min_stroke_coverage']}")
            return False
        
        # Center luminance drop check (if core mask provided)
        if core_mask is not None and core_mask.any():
            # Compute luminance (Y from linear RGB)
            L_before = 0.2126 * canvas_before[..., 0] + 0.7152 * canvas_before[..., 1] + 0.0722 * canvas_before[..., 2]
            L_after = 0.2126 * canvas_after[..., 0] + 0.7152 * canvas_after[..., 1] + 0.0722 * canvas_after[..., 2]
            
            drop = float(L_before[core_mask].mean() - L_after[core_mask].mean())
            min_drop = vis_cfg.get('min_center_luminance_drop', 0.05)
            
            if drop < min_drop:
                logger.debug(f"Stroke skipped: center luminance drop {drop:.4f} < {min_drop}")
                return False
        
        return True
    
    def render_stroke(
        self,
        canvas: np.ndarray,
        alpha_map: np.ndarray,
        stroke_dict: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render a single stroke using OpenCV distance-transform model.
        
        Parameters
        ----------
        canvas : np.ndarray
            Current canvas, shape (H, W, 3), linear RGB [0,1], FP32
        alpha_map : np.ndarray
            Current alpha accumulation, shape (H, W), [0,1], FP32
        stroke_dict : dict
            Stroke definition matching stroke.v1.yaml schema
        
        Returns
        -------
        canvas_new : np.ndarray
            Updated canvas, shape (H, W, 3), linear RGB [0,1], FP32
        alpha_new : np.ndarray
            Updated alpha map, shape (H, W), [0,1], FP32
        
        Notes
        -----
        - Deterministic, non-differentiable
        - Modifies input arrays in-place and returns them
        - Uses distance-transform + radial profile (flat core + Gaussian skirt)
        - Visibility gates may skip imperceptible strokes
        """
        # Reset RNG for deterministic speckle (use stroke ID as seed if available)
        stroke_id = stroke_dict.get('id', 'unknown')
        seed = self.cpu_cfg.get('randomness', {}).get('seed', 42)
        # Hash stroke ID to get a unique but deterministic seed per stroke
        stroke_seed = seed + hash(stroke_id) % 10000
        self.rng = np.random.RandomState(stroke_seed)
        
        # Validate inputs
        if canvas.shape[:2] != (self.canvas_h_px, self.canvas_w_px):
            raise ValueError(
                f"Canvas shape {canvas.shape[:2]} != expected "
                f"({self.canvas_h_px}, {self.canvas_w_px})"
            )
        if alpha_map.shape != (self.canvas_h_px, self.canvas_w_px):
            raise ValueError(
                f"Alpha map shape {alpha_map.shape} != expected "
                f"({self.canvas_h_px}, {self.canvas_w_px})"
            )
        
        # Project parameters to bounds
        stroke_dict = self.project_params(stroke_dict)
        
        # Extract parameters
        bezier = stroke_dict['bezier']
        p1 = torch.tensor(bezier['p1'], dtype=torch.float32)
        p2 = torch.tensor(bezier['p2'], dtype=torch.float32)
        p3 = torch.tensor(bezier['p3'], dtype=torch.float32)
        p4 = torch.tensor(bezier['p4'], dtype=torch.float32)
        
        z0 = stroke_dict['z_profile']['z0']
        z1 = stroke_dict['z_profile']['z1']
        v0 = stroke_dict['speed_profile']['v0']
        v1 = stroke_dict['speed_profile']['v1']
        
        cmy = np.array([
            stroke_dict['color_cmy']['c'],
            stroke_dict['color_cmy']['m'],
            stroke_dict['color_cmy']['y']
        ], dtype=np.float32)
        
        # Convert CMY to linear RGB via color LUT
        paint_color_rgb = self._interpolate_color_lut(cmy)
        
        # Flatten Bézier to polyline
        polyline_mm = geometry.bezier_cubic_polyline(
            p1, p2, p3, p4,
            max_err_mm=0.25,
            max_depth=12
        ).cpu().numpy()  # (N, 2)
        
        if len(polyline_mm) < 2:
            logger.warning("Degenerate stroke (< 2 polyline points), skipping")
            return canvas, alpha_map
        
        # Use average z, v for width/mass (could interpolate per-sample if needed)
        z_avg = 0.5 * (z0 + z1)
        v_avg = 0.5 * (v0 + v1)
        
        # Compute width and mass
        width_mm = self._width_mm(z_avg, v_avg)
        mass_per_mm = self._mass_per_mm(z_avg, v_avg)
        
        # Convert polyline to pixels
        polyline_px = np.zeros_like(polyline_mm)
        polyline_px[:, 0] = polyline_mm[:, 0] * self.dpi[0]
        polyline_px[:, 1] = polyline_mm[:, 1] * self.dpi[1]
        polyline_px = polyline_px.astype(np.int32)
        
        # Compute ROI with margin
        margin_px = int(np.ceil(0.5 * width_mm * self.dpi_avg * self.cpu_cfg['profile'].get('margin_factor', 1.5)))
        x_min = max(0, polyline_px[:, 0].min() - margin_px)
        x_max = min(self.canvas_w_px, polyline_px[:, 0].max() + margin_px + 1)
        y_min = max(0, polyline_px[:, 1].min() - margin_px)
        y_max = min(self.canvas_h_px, polyline_px[:, 1].max() + margin_px + 1)
        
        if x_max <= x_min or y_max <= y_min:
            logger.warning("Stroke ROI outside canvas, skipping")
            return canvas, alpha_map
        
        roi_w = x_max - x_min
        roi_h = y_max - y_min
        
        # Shift polyline to ROI coordinates
        polyline_roi = polyline_px.copy()
        polyline_roi[:, 0] -= x_min
        polyline_roi[:, 1] -= y_min
        
        # Rasterize 1-px centerline
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.polylines(mask, [polyline_roi], isClosed=False, color=255, thickness=1)
        
        # Distance transform
        dist_px = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3).astype(np.float32)
        
        # Build radial profile
        phi = self._build_radial_profile(dist_px, width_mm)
        
        # Apply mass scaling
        k_mass = self.cpu_cfg['deposition'].get('k_mass', 2.5)
        alpha_profile = np.clip(k_mass * mass_per_mm * phi, 0.0, 1.0)
        
        # Zero out values below visibility threshold
        min_alpha = self.cpu_cfg['visibility']['min_alpha_visible']
        alpha_profile[alpha_profile < min_alpha] = 0.0
        
        # Extract ROI from canvas
        canvas_roi = canvas[y_min:y_max, x_min:x_max]  # (h, w, 3)
        alpha_roi = alpha_map[y_min:y_max, x_min:x_max]  # (h, w)
        
        # Alcohol ink layering model (instant dry)
        # Alcohol inks dry instantly on paper - no wet mixing.
        # Each layer is a semi-transparent filter placed on top of previous layers.
        # 
        # Physics: Light passes through multiple transparent colored filters.
        # Each filter absorbs some wavelengths (subtractive).
        # 
        # Model: Work in absorption space (1 - RGB), multiply absorptions, convert back.
        # This is how stacked colored glass or film works.
        
        alpha_stamp = alpha_profile  # (h, w)
        alpha_stamp_3d = alpha_stamp[:, :, np.newaxis]  # (h, w, 1)
        
        # Convert RGB to transmission (how much light passes through)
        # transmission = 1 means all light passes (white/transparent)
        # transmission = 0 means no light passes (black/opaque)
        canvas_transmission = canvas_roi  # (h, w, 3)
        paint_transmission = paint_color_rgb  # (3,)
        
        # New layer acts as a filter: reduces transmission by its own transmission
        # Where alpha=1, full filter effect; where alpha=0, no filter
        # transmission_new = transmission_old * lerp(1.0, paint_transmission, alpha)
        filter_effect = (1.0 - alpha_stamp_3d) + paint_transmission[np.newaxis, np.newaxis, :] * alpha_stamp_3d
        
        canvas_roi_new = canvas_transmission * filter_effect
        canvas_roi_new = np.clip(canvas_roi_new, 0.0, 1.0)
        
        # Update total alpha (accumulated opacity)
        alpha_roi_new = alpha_roi + alpha_stamp * (1.0 - alpha_roi)
        alpha_roi_new = np.clip(alpha_roi_new, 0.0, 1.0)
        
        # Visibility check
        core_radius_px = self.cpu_cfg['profile']['core_frac'] * 0.5 * width_mm * self.dpi_avg
        core_mask = dist_px <= core_radius_px
        
        alpha_delta = alpha_roi_new - alpha_roi
        
        if not self._check_visibility(canvas_roi, canvas_roi_new, alpha_delta, core_mask):
            return canvas, alpha_map  # Skip this stroke
        
        # Write back to canvas
        canvas[y_min:y_max, x_min:x_max] = canvas_roi_new
        alpha_map[y_min:y_max, x_min:x_max] = alpha_roi_new
        
        return canvas, alpha_map
    
    def render_strokes(
        self,
        canvas: np.ndarray,
        alpha_map: np.ndarray,
        strokes: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render multiple strokes in sequence.
        
        Parameters
        ----------
        canvas : np.ndarray
            Initial canvas, shape (H, W, 3), linear RGB [0,1], FP32
        alpha_map : np.ndarray
            Initial alpha map, shape (H, W), [0,1], FP32
        strokes : list of dict
            List of stroke dictionaries
        
        Returns
        -------
        canvas_new : np.ndarray
            Final canvas after all strokes
        alpha_new : np.ndarray
            Final alpha map
        """
        for stroke_dict in strokes:
            canvas, alpha_map = self.render_stroke(canvas, alpha_map, stroke_dict)
        return canvas, alpha_map
    
    def _interpolate_color_lut(self, cmy: np.ndarray) -> np.ndarray:
        """Trilinear interpolation on color LUT.
        
        Parameters
        ----------
        cmy : np.ndarray
            CMY values, shape (3,), range [0,1]
        
        Returns
        -------
        np.ndarray
            Linear RGB color, shape (3,), range [0,1]
        """
        color_lut = self.luts['color_lut']  # (Nc, Nm, Ny, 3)
        Nc, Nm, Ny, _ = color_lut.shape
        
        # Map CMY [0,1] to grid indices [0, N-1]
        c_idx = cmy[0] * (Nc - 1)
        m_idx = cmy[1] * (Nm - 1)
        y_idx = cmy[2] * (Ny - 1)
        
        # Integer and fractional parts
        c0 = int(np.floor(c_idx))
        m0 = int(np.floor(m_idx))
        y0 = int(np.floor(y_idx))
        
        c1 = min(c0 + 1, Nc - 1)
        m1 = min(m0 + 1, Nm - 1)
        y1 = min(y0 + 1, Ny - 1)
        
        fc = c_idx - c0
        fm = m_idx - m0
        fy = y_idx - y0
        
        # Trilinear interpolation
        c000 = color_lut[c0, m0, y0]
        c001 = color_lut[c0, m0, y1]
        c010 = color_lut[c0, m1, y0]
        c011 = color_lut[c0, m1, y1]
        c100 = color_lut[c1, m0, y0]
        c101 = color_lut[c1, m0, y1]
        c110 = color_lut[c1, m1, y0]
        c111 = color_lut[c1, m1, y1]
        
        c00 = (1 - fc) * c000 + fc * c100
        c01 = (1 - fc) * c001 + fc * c101
        c10 = (1 - fc) * c010 + fc * c110
        c11 = (1 - fc) * c011 + fc * c111
        
        c0_ = (1 - fm) * c00 + fm * c10
        c1_ = (1 - fm) * c01 + fm * c11
        
        rgb = (1 - fy) * c0_ + fy * c1_
        
        return np.clip(rgb, 0.0, 1.0)


def load_toy_luts(
    color_grid: Tuple[int, int, int] = (11, 11, 11),
    high_visibility: bool = True
) -> Dict[str, np.ndarray]:
    """Generate toy LUTs for testing (before calibration).
    
    Parameters
    ----------
    color_grid : tuple of int
        (Nc, Nm, Ny) grid size for color LUT
    high_visibility : bool
        If True, produce solid black for all CMY (easy verification).
        If False, use realistic subtractive model (default True).
    
    Returns
    -------
    dict
        LUTs: {color_lut} as numpy arrays (FP32)
    
    Notes
    -----
    The new renderer doesn't use alpha_lut or psf_lut; those are legacy.
    Width and mass come from renderer_cpu.v1.yaml config instead.
    """
    Nc, Nm, Ny = color_grid
    
    # Color LUT: simple subtractive model or black for high visibility
    c_vals = np.linspace(0, 1, Nc, dtype=np.float32)
    m_vals = np.linspace(0, 1, Nm, dtype=np.float32)
    y_vals = np.linspace(0, 1, Ny, dtype=np.float32)
    
    C, M, Y = np.meshgrid(c_vals, m_vals, y_vals, indexing='ij')
    
    if high_visibility:
        # Black ink for all CMY combinations (easy to see)
        color_lut = np.zeros((Nc, Nm, Ny, 3), dtype=np.float32)
    else:
        # Subtractive model: 1 - CMY → RGB
        color_lut = np.stack([
            1.0 - C,  # R = 1 - C
            1.0 - M,  # G = 1 - M
            1.0 - Y   # B = 1 - Y
        ], axis=-1).astype(np.float32)
    
    return {
        'color_lut': color_lut,
        # Legacy LUTs (not used by new renderer, but kept for compatibility)
        'alpha_lut': np.ones((8, 8), dtype=np.float32),
        'psf_lut': np.ones((8, 8), dtype=np.float32) * 2.0
    }
