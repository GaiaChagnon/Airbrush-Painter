"""CPU reference renderer for ground-truth stroke simulation.

This is a deterministic, pure-CPU implementation that serves as the oracle for
all stroke rendering behavior. It defines "correct" physics and compositing
before any GPU optimization or differentiability concerns.

Architecture:
    - Bézier → polyline with adaptive flattening (error control in mm)
    - Per-sample Z/V interpolation (linear profiles)
    - PSF splatting with Gaussian kernels (σ from LUT)
    - Alpha modulation from (z, v) via alpha_lut
    - CMY → linear RGB via color_lut (trilinear interpolation)
    - Alpha-over compositing (linear RGB, clamped accumulation)

Invariants:
    - All geometry in mm; conversions to px happen at boundaries only
    - Linear RGB [0,1] throughout; sRGB conversion at I/O only
    - FP32 precision (no mixed precision, no autocast)
    - Deterministic (no SIMD quirks, no threading)
    - Kernel cache uses quantized σ grid (0.1 px steps)

Usage:
    from src.airbrush_simulator.cpu_reference import CPUReferenceRenderer
    from src.utils import validators, fs
    
    # Load configs and LUTs
    sim_cfg = fs.load_yaml("configs/sim/physics_v1.yaml")
    env_cfg = fs.load_yaml("configs/env_airbrush_v1.yaml")
    luts = load_toy_luts()  # or load calibrated LUTs
    
    # Initialize renderer
    renderer = CPUReferenceRenderer(sim_cfg, env_cfg, luts)
    
    # Render a stroke
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    alpha = np.zeros((H, W), dtype=np.float32)
    stroke_dict = validators.StrokeV1(...).dict()
    
    canvas_new, alpha_new = renderer.render_stroke(canvas, alpha, stroke_dict)

LUT format:
    - color_lut: (Nc, Nm, Ny, 3) FP32, CMY [0,1]³ → linear RGB [0,1]³
    - alpha_lut: (Nz, Nv) FP32, (z, v) → coverage [0,1]
    - psf_lut: (Nz, Nv) FP32, (z, v) → σ_px (Gaussian width in pixels)
      OR (Nz, Nv, K, K) FP32 kernels (future extension)

Notes:
    - Sampling step: Δs ≤ 0.5·σ_px to avoid under-splatting
    - PSF kernel radius: 4σ (>99.99% mass captured)
    - Canvas boundaries: splats clipped at edges (no wraparound)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.utils import compute, geometry, validators, color as color_utils

logger = logging.getLogger(__name__)


class CPUReferenceRenderer:
    """CPU-based reference renderer for ground-truth stroke simulation.
    
    Provides deterministic, non-differentiable rendering using splat-based
    deposition with Gaussian PSFs. All operations in FP32 for stability.
    
    Attributes
    ----------
    sim_cfg : dict
        Physics config (LUT paths, PSF cache settings)
    env_cfg : dict
        Environment config (work_area_mm, render_px, origin)
    luts : dict
        Lookup tables: {color_lut, alpha_lut, psf_lut} as numpy arrays
    stroke_bounds : dict
        Schema bounds for clamping: {x, y, z, speed, cmy}
    dpi : float
        Resolution in pixels/mm (computed from render_px and work_area_mm)
    kernel_cache : dict
        Cached Gaussian kernels keyed by quantized σ
    """
    
    def __init__(
        self,
        sim_cfg: Dict,
        env_cfg: Dict,
        luts: Dict[str, np.ndarray],
        psf_cache_quantization: float = 0.1,
        bbox_margin_sigma: float = 4.0
    ):
        """Initialize CPU reference renderer.
        
        Parameters
        ----------
        sim_cfg : dict
            Physics config with LUT metadata
        env_cfg : dict
            Environment config (work_area_mm, render_px, origin)
        luts : dict
            Lookup tables as numpy arrays:
            - color_lut: (Nc, Nm, Ny, 3) FP32
            - alpha_lut: (Nz, Nv) FP32
            - psf_lut: (Nz, Nv) FP32 (σ values) or (Nz, Nv, K, K) (kernels)
        psf_cache_quantization : float
            Quantization step for kernel cache (default 0.1 px)
        bbox_margin_sigma : float
            PSF radius in units of σ for bbox calculation (default 4.0)
        """
        self.sim_cfg = sim_cfg
        self.env_cfg = env_cfg
        self.luts = luts
        self.psf_cache_quantization = psf_cache_quantization
        self.bbox_margin_sigma = bbox_margin_sigma
        
        # Extract bounds from stroke schema (hardcoded for now, matches stroke.v1.yaml)
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
        
        # Canvas dimensions
        self.canvas_h_px = render_px[0]
        self.canvas_w_px = render_px[1]
        
        # Kernel cache (quantized σ → kernel)
        self.kernel_cache: Dict[float, np.ndarray] = {}
        
        # Validate LUTs
        self._validate_luts()
        
        logger.info(
            f"CPUReferenceRenderer initialized: "
            f"canvas={render_px}, work_area={work_area_mm} mm, "
            f"dpi={self.dpi[0]:.2f}×{self.dpi[1]:.2f} px/mm"
        )
    
    def _validate_luts(self):
        """Validate LUT shapes and dtypes."""
        required_keys = {'color_lut', 'alpha_lut', 'psf_lut'}
        missing = required_keys - set(self.luts.keys())
        if missing:
            raise ValueError(f"Missing LUTs: {missing}")
        
        # Check dtypes (all FP32)
        for name, lut in self.luts.items():
            if lut.dtype != np.float32:
                raise TypeError(f"LUT {name} must be FP32, got {lut.dtype}")
        
        # Check shapes
        color_shape = self.luts['color_lut'].shape
        if len(color_shape) != 4 or color_shape[-1] != 3:
            raise ValueError(
                f"color_lut must be (Nc, Nm, Ny, 3), got {color_shape}"
            )
        
        alpha_shape = self.luts['alpha_lut'].shape
        if len(alpha_shape) != 2:
            raise ValueError(f"alpha_lut must be (Nz, Nv), got {alpha_shape}")
        
        psf_shape = self.luts['psf_lut'].shape
        if len(psf_shape) not in [2, 4]:
            raise ValueError(
                f"psf_lut must be (Nz, Nv) or (Nz, Nv, K, K), got {psf_shape}"
            )
        
        # Validate value ranges
        alpha_min, alpha_max = self.luts['alpha_lut'].min(), self.luts['alpha_lut'].max()
        if alpha_min < 0.0 or alpha_max > 1.0:
            logger.warning(f"alpha_lut values outside [0,1]: [{alpha_min}, {alpha_max}]")
        
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
    
    def render_stroke(
        self,
        canvas: np.ndarray,
        alpha_map: np.ndarray,
        stroke_dict: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render a single stroke onto canvas with alpha-over compositing.
        
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
        - Sampling step: Δs ≤ 0.5·σ_px
        - PSF kernel radius: 4σ (>99.99% mass)
        """
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
            max_err_mm=0.25,  # Fine subdivision
            max_depth=12
        ).cpu().numpy()  # (N, 2)
        
        if len(polyline_mm) < 2:
            logger.warning("Degenerate stroke (< 2 polyline points), skipping")
            return canvas, alpha_map
        
        # Arc-length parametrization
        arc_lengths = self._compute_arc_lengths(polyline_mm)
        total_length = arc_lengths[-1]
        
        if total_length < 1e-6:
            logger.warning("Zero-length stroke, skipping")
            return canvas, alpha_map
        
        # Compute average σ to determine sampling step
        z_avg = 0.5 * (z0 + z1)
        v_avg = 0.5 * (v0 + v1)
        sigma_avg_px = self._interpolate_psf_lut(z_avg, v_avg)
        
        # Sampling step: Δs ≤ 0.5·σ_px (in mm)
        # Convert σ_px to mm using average dpi
        dpi_avg = 0.5 * (self.dpi[0] + self.dpi[1])
        sigma_avg_mm = sigma_avg_px / dpi_avg
        step_mm = min(0.5 * sigma_avg_mm, 0.5)  # Cap at 0.5 mm
        
        num_samples = max(int(np.ceil(total_length / step_mm)), 2)
        sample_t = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
        
        # Interpolate along polyline
        sample_positions_mm = self._interpolate_polyline(
            polyline_mm, arc_lengths, sample_t
        )
        
        # Interpolate z and v linearly
        sample_z = z0 + sample_t * (z1 - z0)
        sample_v = v0 + sample_t * (v1 - v0)
        
        # Render each sample
        for i in range(num_samples):
            pos_mm = sample_positions_mm[i]
            z = sample_z[i]
            v = sample_v[i]
            
            # Get PSF width and alpha from LUTs
            sigma_px = self._interpolate_psf_lut(z, v)
            alpha_local = self._interpolate_alpha_lut(z, v)
            
            # Splat onto canvas
            self._splat_sample(
                canvas, alpha_map,
                pos_mm, sigma_px, alpha_local, paint_color_rgb
            )
        
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
    
    def _compute_arc_lengths(self, polyline: np.ndarray) -> np.ndarray:
        """Compute cumulative arc length along polyline.
        
        Parameters
        ----------
        polyline : np.ndarray
            Polyline vertices, shape (N, 2)
        
        Returns
        -------
        np.ndarray
            Cumulative arc lengths, shape (N,), starting at 0
        """
        deltas = np.diff(polyline, axis=0)  # (N-1, 2)
        segment_lengths = np.linalg.norm(deltas, axis=1)  # (N-1,)
        arc_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        return arc_lengths.astype(np.float32)
    
    def _interpolate_polyline(
        self,
        polyline: np.ndarray,
        arc_lengths: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """Interpolate positions along polyline at normalized arc-length t.
        
        Parameters
        ----------
        polyline : np.ndarray
            Polyline vertices, shape (N, 2)
        arc_lengths : np.ndarray
            Cumulative arc lengths, shape (N,)
        t : np.ndarray
            Normalized arc-length parameters [0,1], shape (M,)
        
        Returns
        -------
        np.ndarray
            Interpolated positions, shape (M, 2)
        """
        total_length = arc_lengths[-1]
        target_lengths = t * total_length
        
        # Linear interpolation along segments
        positions = np.empty((len(t), 2), dtype=np.float32)
        for i, s in enumerate(target_lengths):
            # Find segment
            idx = np.searchsorted(arc_lengths, s, side='right') - 1
            idx = np.clip(idx, 0, len(polyline) - 2)
            
            # Interpolate within segment
            s0 = arc_lengths[idx]
            s1 = arc_lengths[idx + 1]
            if s1 - s0 < 1e-9:
                local_t = 0.0
            else:
                local_t = (s - s0) / (s1 - s0)
            
            positions[i] = (1 - local_t) * polyline[idx] + local_t * polyline[idx + 1]
        
        return positions
    
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
    
    def _interpolate_alpha_lut(self, z: float, v: float) -> float:
        """Bilinear interpolation on alpha LUT.
        
        Parameters
        ----------
        z : float
            Nozzle height (mm)
        v : float
            Speed (mm/s)
        
        Returns
        -------
        float
            Alpha coverage [0,1]
        """
        alpha_lut = self.luts['alpha_lut']  # (Nz, Nv)
        Nz, Nv = alpha_lut.shape
        
        # Map z, v to grid indices
        z_min, z_max = self.stroke_bounds['z']
        v_min, v_max = self.stroke_bounds['speed']
        
        z_idx = (z - z_min) / (z_max - z_min) * (Nz - 1)
        v_idx = (v - v_min) / (v_max - v_min) * (Nv - 1)
        
        z0 = int(np.floor(z_idx))
        v0 = int(np.floor(v_idx))
        z1 = min(z0 + 1, Nz - 1)
        v1 = min(v0 + 1, Nv - 1)
        
        fz = z_idx - z0
        fv = v_idx - v0
        
        # Bilinear interpolation
        a00 = alpha_lut[z0, v0]
        a01 = alpha_lut[z0, v1]
        a10 = alpha_lut[z1, v0]
        a11 = alpha_lut[z1, v1]
        
        a0 = (1 - fz) * a00 + fz * a10
        a1 = (1 - fz) * a01 + fz * a11
        
        alpha = (1 - fv) * a0 + fv * a1
        
        return float(np.clip(alpha, 0.0, 1.0))
    
    def _interpolate_psf_lut(self, z: float, v: float) -> float:
        """Bilinear interpolation on PSF LUT (returns σ in pixels).
        
        Parameters
        ----------
        z : float
            Nozzle height (mm)
        v : float
            Speed (mm/s)
        
        Returns
        -------
        float
            PSF width σ in pixels
        """
        psf_lut = self.luts['psf_lut']
        
        if len(psf_lut.shape) == 2:
            # (Nz, Nv) → σ values directly
            Nz, Nv = psf_lut.shape
            
            z_min, z_max = self.stroke_bounds['z']
            v_min, v_max = self.stroke_bounds['speed']
            
            z_idx = (z - z_min) / (z_max - z_min) * (Nz - 1)
            v_idx = (v - v_min) / (v_max - v_min) * (Nv - 1)
            
            z0 = int(np.floor(z_idx))
            v0 = int(np.floor(v_idx))
            z1 = min(z0 + 1, Nz - 1)
            v1 = min(v0 + 1, Nv - 1)
            
            fz = z_idx - z0
            fv = v_idx - v0
            
            s00 = psf_lut[z0, v0]
            s01 = psf_lut[z0, v1]
            s10 = psf_lut[z1, v0]
            s11 = psf_lut[z1, v1]
            
            s0 = (1 - fz) * s00 + fz * s10
            s1 = (1 - fz) * s01 + fz * s11
            
            sigma = (1 - fv) * s0 + fv * s1
            
            return float(np.clip(sigma, 0.1, 50.0))  # Reasonable bounds
        else:
            # (Nz, Nv, K, K) → full kernels (future extension)
            raise NotImplementedError("Full kernel LUTs not yet supported")
    
    def _get_gaussian_kernel(self, sigma_px: float) -> np.ndarray:
        """Get or create cached Gaussian kernel.
        
        Parameters
        ----------
        sigma_px : float
            Gaussian width in pixels
        
        Returns
        -------
        np.ndarray
            Normalized Gaussian kernel, shape (K, K), sum=1
        """
        # Quantize sigma for cache
        sigma_quantized = round(sigma_px / self.psf_cache_quantization) * self.psf_cache_quantization
        
        if sigma_quantized in self.kernel_cache:
            return self.kernel_cache[sigma_quantized]
        
        # Create kernel
        radius = int(np.ceil(self.bbox_margin_sigma * sigma_quantized))
        size = 2 * radius + 1
        
        # Meshgrid
        y, x = np.meshgrid(
            np.arange(size, dtype=np.float32) - radius,
            np.arange(size, dtype=np.float32) - radius,
            indexing='ij'
        )
        
        # Gaussian
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma_quantized**2))
        kernel /= kernel.sum()  # Normalize
        
        self.kernel_cache[sigma_quantized] = kernel
        return kernel
    
    def _splat_sample(
        self,
        canvas: np.ndarray,
        alpha_map: np.ndarray,
        pos_mm: np.ndarray,
        sigma_px: float,
        alpha_local: float,
        paint_color: np.ndarray
    ):
        """Splat a single sample onto canvas (alpha-over compositing).
        
        Parameters
        ----------
        canvas : np.ndarray
            Canvas to modify in-place, shape (H, W, 3)
        alpha_map : np.ndarray
            Alpha map to modify in-place, shape (H, W)
        pos_mm : np.ndarray
            Position in mm, shape (2,)
        sigma_px : float
            PSF width in pixels
        alpha_local : float
            Local alpha coverage [0,1]
        paint_color : np.ndarray
            Paint color (linear RGB), shape (3,)
        """
        # Convert mm to pixels
        pos_px = np.array([
            pos_mm[0] * self.dpi[0],
            pos_mm[1] * self.dpi[1]
        ], dtype=np.float32)
        
        # Get kernel
        kernel = self._get_gaussian_kernel(sigma_px)
        kernel_h, kernel_w = kernel.shape
        radius = kernel_h // 2
        
        # Canvas region (integer pixel coordinates)
        cx = int(np.round(pos_px[0]))
        cy = int(np.round(pos_px[1]))
        
        y_min = max(0, cy - radius)
        y_max = min(self.canvas_h_px, cy + radius + 1)
        x_min = max(0, cx - radius)
        x_max = min(self.canvas_w_px, cx + radius + 1)
        
        if y_max <= y_min or x_max <= x_min:
            return  # Outside canvas
        
        # Kernel region (handle boundary clipping)
        ky_min = radius - (cy - y_min)
        ky_max = ky_min + (y_max - y_min)
        kx_min = radius - (cx - x_min)
        kx_max = kx_min + (x_max - x_min)
        
        kernel_roi = kernel[ky_min:ky_max, kx_min:kx_max]
        
        # Alpha stamp (weighted by kernel)
        alpha_stamp = alpha_local * kernel_roi  # (h, w)
        
        # Current canvas and alpha
        canvas_roi = canvas[y_min:y_max, x_min:x_max]  # (h, w, 3)
        alpha_roi = alpha_map[y_min:y_max, x_min:x_max]  # (h, w)
        
        # Alpha-over compositing
        # C_new = C_old * (1 - alpha_stamp) + C_paint * alpha_stamp
        # α_new = α_old + alpha_stamp * (1 - α_old)  (or clamp to 1)
        
        alpha_stamp_3d = alpha_stamp[:, :, np.newaxis]  # (h, w, 1)
        canvas_roi[:] = (
            canvas_roi * (1 - alpha_stamp_3d) +
            paint_color[np.newaxis, np.newaxis, :] * alpha_stamp_3d
        )
        
        # Accumulate alpha (simple clamp, no drying model yet)
        alpha_roi[:] = np.clip(alpha_roi + alpha_stamp, 0.0, 1.0)


def load_toy_luts(
    color_grid: Tuple[int, int, int] = (11, 11, 11),
    alpha_grid: Tuple[int, int] = (8, 8),
    psf_grid: Tuple[int, int] = (8, 8),
    sigma_range: Tuple[float, float] = (1.0, 5.0)
) -> Dict[str, np.ndarray]:
    """Generate toy LUTs for testing (before calibration).
    
    Parameters
    ----------
    color_grid : tuple of int
        (Nc, Nm, Ny) grid size for color LUT
    alpha_grid : tuple of int
        (Nz, Nv) grid size for alpha LUT
    psf_grid : tuple of int
        (Nz, Nv) grid size for PSF LUT
    sigma_range : tuple of float
        (σ_min, σ_max) range in pixels
    
    Returns
    -------
    dict
        LUTs: {color_lut, alpha_lut, psf_lut} as numpy arrays (FP32)
    
    Notes
    -----
    Toy models:
    - Color: 1 - CMY → RGB (subtractive approximation)
    - Alpha: inversely proportional to z, directly proportional to v
    - PSF: σ increases with z, decreases with v
    """
    Nc, Nm, Ny = color_grid
    Nz, Nv = alpha_grid
    
    # Color LUT: simple subtractive model (1 - CMY)
    c_vals = np.linspace(0, 1, Nc, dtype=np.float32)
    m_vals = np.linspace(0, 1, Nm, dtype=np.float32)
    y_vals = np.linspace(0, 1, Ny, dtype=np.float32)
    
    C, M, Y = np.meshgrid(c_vals, m_vals, y_vals, indexing='ij')
    color_lut = np.stack([
        1.0 - C,  # R = 1 - C
        1.0 - M,  # G = 1 - M
        1.0 - Y   # B = 1 - Y
    ], axis=-1).astype(np.float32)  # (Nc, Nm, Ny, 3)
    
    # Alpha LUT: α ∝ 1/z * v
    z_vals = np.linspace(0.5, 30.0, Nz, dtype=np.float32)  # Avoid z=0
    v_vals = np.linspace(1.0, 300.0, Nv, dtype=np.float32)
    Z, V = np.meshgrid(z_vals, v_vals, indexing='ij')
    
    alpha_lut = 0.3 * (1.0 / Z) * (V / 300.0)  # Scale factor 0.3
    alpha_lut = np.clip(alpha_lut, 0.0, 1.0).astype(np.float32)
    
    # PSF LUT: σ ∝ z / v (wider at high z, narrow at high v)
    sigma_min, sigma_max = sigma_range
    psf_lut = sigma_min + (sigma_max - sigma_min) * (Z / 30.0) * (1.0 - V / 300.0)
    psf_lut = np.clip(psf_lut, sigma_min, sigma_max).astype(np.float32)
    
    return {
        'color_lut': color_lut,
        'alpha_lut': alpha_lut,
        'psf_lut': psf_lut
    }
