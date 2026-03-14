"""Tests for the GPU digital twin simulator and calibration schema.

Covers:
    - Schema validation (load, reject invalid, round-trip)
    - Stamp profile (integral, core, skirt, cutoff)
    - Compositing math (multiplicative, commutativity, saturation)
    - LUT interpolation (1-D, 2-D, trilinear, edge clamping)
    - Stroke sampling (straight, curved, degenerate)
    - Batch rendering (independence)
    - Differentiability (gradient flow)
    - Determinism (bitwise identical)
    - Calibration round-trip (synthetic measurements -> fit -> verify)

All tests run on CPU to avoid CI dependency on CUDA.

Usage:
    pytest tests/test_digital_twin.py -v
"""

import copy
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils import fs, validators
from src.utils.validators import CalibrationV1

# Path to the default calibration shipped with the repo
CAL_PATH = Path("digital_twin/calibration.yaml")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def cal_path(tmp_path):
    """Copy the default calibration to a temp dir so tests can mutate it."""
    src = CAL_PATH
    dst = tmp_path / "calibration.yaml"
    data = fs.load_yaml(src)
    fs.atomic_yaml_dump(data, dst)
    return dst


@pytest.fixture
def sim(cal_path):
    """GPUStampSimulator on CPU with default calibration."""
    from digital_twin.gpu_simulator import GPUStampSimulator
    return GPUStampSimulator(
        cal_path, device=torch.device("cpu"), batch_size=1,
    )


@pytest.fixture
def simple_stroke():
    """Straight horizontal stroke at constant z and speed."""
    return {
        "bezier": {
            "p1": (50.0, 148.5),
            "p2": (80.0, 148.5),
            "p3": (130.0, 148.5),
            "p4": (160.0, 148.5),
        },
        "z_profile": {"z0": 10.0, "z1": 10.0},
        "speed_profile": {"v0": 50.0, "v1": 50.0},
        "color_cmy": {"c": 0.8, "m": 0.2, "y": 0.1},
    }


@pytest.fixture
def curved_stroke():
    """Curved stroke with varying z and speed."""
    return {
        "bezier": {
            "p1": (50.0, 50.0),
            "p2": (100.0, 120.0),
            "p3": (150.0, 80.0),
            "p4": (200.0, 150.0),
        },
        "z_profile": {"z0": 5.0, "z1": 15.0},
        "speed_profile": {"v0": 30.0, "v1": 120.0},
        "color_cmy": {"c": 0.2, "m": 0.8, "y": 0.3},
    }


# ============================================================================
# Schema tests
# ============================================================================

class TestSchema:
    """CalibrationV1 Pydantic model validation."""

    def test_load_default(self):
        """Default calibration.yaml loads and validates."""
        cal = validators.load_calibration_config(CAL_PATH)
        assert cal.schema_version == "calibration.v1"
        assert len(cal.z_grid_mm) == len(cal.radius_lut_mm)

    def test_round_trip(self, cal_path):
        """load -> save -> load produces identical model."""
        cal1 = validators.load_calibration_config(cal_path)
        out = cal_path.parent / "round_trip.yaml"
        validators.save_calibration_config(cal1, out)
        cal2 = validators.load_calibration_config(out)
        assert cal1.z_grid_mm == cal2.z_grid_mm
        assert cal1.radius_lut_mm == cal2.radius_lut_mm
        assert cal1.mass_lut == cal2.mass_lut
        assert cal1.k_mass == cal2.k_mass
        assert cal1.profile.core_frac == cal2.profile.core_frac

    def test_reject_mismatched_radius_len(self, cal_path):
        """radius_lut_mm length != z_grid_mm length raises."""
        data = fs.load_yaml(cal_path)
        data["radius_lut_mm"] = [1.0, 2.0]  # too short
        with pytest.raises(Exception, match="radius_lut_mm"):
            CalibrationV1(**data)

    def test_reject_mismatched_mass_rows(self, cal_path):
        """mass_lut row count != z_grid_mm length raises."""
        data = fs.load_yaml(cal_path)
        data["mass_lut"] = data["mass_lut"][:2]  # too few rows
        with pytest.raises(Exception, match="mass_lut"):
            CalibrationV1(**data)

    def test_reject_mismatched_mass_cols(self, cal_path):
        """mass_lut column count != speed_grid_mm_s length raises."""
        data = fs.load_yaml(cal_path)
        data["mass_lut"][0] = [0.1, 0.2]  # wrong column count
        with pytest.raises(Exception, match="mass_lut"):
            CalibrationV1(**data)

    def test_reject_negative_radius(self, cal_path):
        """Negative radius_lut entry raises."""
        data = fs.load_yaml(cal_path)
        data["radius_lut_mm"][0] = -1.0
        with pytest.raises(Exception, match="radius_lut_mm"):
            CalibrationV1(**data)

    def test_reject_negative_mass(self, cal_path):
        """Negative mass_lut entry raises."""
        data = fs.load_yaml(cal_path)
        data["mass_lut"][0][0] = -0.5
        with pytest.raises(Exception, match="mass_lut"):
            CalibrationV1(**data)

    def test_reject_zero_k_mass(self, cal_path):
        """k_mass <= 0 raises."""
        data = fs.load_yaml(cal_path)
        data["k_mass"] = 0.0
        with pytest.raises(Exception):
            CalibrationV1(**data)

    def test_reject_bad_profile_range(self, cal_path):
        """Out-of-range profile parameter raises."""
        data = fs.load_yaml(cal_path)
        data["profile"]["skirt_power"] = 99.0
        with pytest.raises(Exception):
            CalibrationV1(**data)


# ============================================================================
# Stamp profile tests
# ============================================================================

class TestStampProfile:
    """Canonical 1-D radial profile."""

    def test_integral_normalised(self, sim):
        """Profile integral should be close to the computed value."""
        phi = sim._profile_1d
        u = sim._profile_u
        du = (u[1] - u[0]).item()
        numerical = 2.0 * math.pi * torch.trapezoid(phi * u, dx=du).item()
        assert abs(numerical - sim._profile_integral) < 1e-3

    def test_core_flat(self, sim):
        """Profile is 1.0 inside the core region."""
        core = sim._cal.profile.core_frac
        core_mask = sim._profile_u <= core
        vals = sim._profile_1d[core_mask]
        assert vals.min().item() >= 0.999, "Core should be flat at 1.0"

    def test_skirt_decays(self, sim):
        """Profile decays below 0.01 at the margin boundary."""
        margin = sim._cal.profile.margin_factor
        near_margin = (sim._profile_u - margin).abs()
        idx = near_margin.argmin().item()
        if idx > 0:
            val = sim._profile_1d[idx - 1].item()
            assert val < 0.05, f"Expected decay near margin, got {val}"

    def test_hard_cutoff(self, sim):
        """Profile is exactly 0 beyond margin_factor."""
        margin = sim._cal.profile.margin_factor
        beyond = sim._profile_u > margin
        if beyond.any():
            vals = sim._profile_1d[beyond]
            assert vals.max().item() == 0.0


# ============================================================================
# Compositing tests
# ============================================================================

class TestCompositing:
    """Multiplicative transmission compositing."""

    def test_multiplicative(self, sim, simple_stroke):
        """Two strokes at same position: canvas = white * T1 * T2."""
        canvas = sim.reset()
        white = canvas.clone()

        canvas1 = sim.render_stroke(white.clone(), simple_stroke)
        T1 = canvas1 / white.clamp(min=1e-8)

        canvas2 = sim.render_stroke(canvas1.clone(), simple_stroke)
        T_total = canvas2 / white.clamp(min=1e-8)
        T_expected = T1 * T1  # two identical strokes

        # allow small numerical error
        diff = (T_total - T_expected).abs()
        # only compare where paint was deposited
        mask = (T1 < 0.999)
        if mask.any():
            assert diff[mask.expand_as(diff)].max().item() < 0.02

    def test_commutativity(self, sim):
        """Rendering A-then-B == B-then-A within FP32 tolerance."""
        stroke_a = {
            "bezier": {
                "p1": (80.0, 100.0), "p2": (100.0, 100.0),
                "p3": (120.0, 100.0), "p4": (140.0, 100.0),
            },
            "z_profile": {"z0": 10.0, "z1": 10.0},
            "speed_profile": {"v0": 50.0, "v1": 50.0},
            "color_cmy": {"c": 0.9, "m": 0.1, "y": 0.1},
        }
        stroke_b = {
            "bezier": {
                "p1": (80.0, 100.0), "p2": (100.0, 100.0),
                "p3": (120.0, 100.0), "p4": (140.0, 100.0),
            },
            "z_profile": {"z0": 10.0, "z1": 10.0},
            "speed_profile": {"v0": 50.0, "v1": 50.0},
            "color_cmy": {"c": 0.1, "m": 0.9, "y": 0.1},
        }

        canvas_ab = sim.render_strokes(sim.reset(), [stroke_a, stroke_b])
        canvas_ba = sim.render_strokes(sim.reset(), [stroke_b, stroke_a])

        diff = (canvas_ab - canvas_ba).abs().max().item()
        assert diff < 0.02, f"Commutativity violation: max diff = {diff}"

    def test_full_opacity(self, sim):
        """Black paint with high mass drives canvas toward 0."""
        canvas = sim.reset()
        stroke = {
            "bezier": {
                "p1": (100.0, 148.5), "p2": (105.0, 148.5),
                "p3": (105.0, 148.5), "p4": (110.0, 148.5),
            },
            "z_profile": {"z0": 2.0, "z1": 2.0},
            "speed_profile": {"v0": 10.0, "v1": 10.0},
            "color_cmy": {"c": 1.0, "m": 1.0, "y": 1.0},
        }
        # many passes to saturate
        for _ in range(20):
            canvas = sim.render_stroke(canvas, stroke)

        px_x = int(105.0 * sim._dpi_x)
        px_y = int(148.5 * sim._dpi_y)
        px_x = min(max(px_x, 0), sim._W - 1)
        px_y = min(max(px_y, 0), sim._H - 1)
        center_val = canvas[0, :, px_y, px_x].max().item()
        assert center_val < 0.15, f"Expected near-zero, got {center_val}"


# ============================================================================
# LUT interpolation tests
# ============================================================================

class TestLUTInterpolation:
    """Differentiable interpolation helpers."""

    def test_1d_exact_at_knots(self, sim):
        """1-D interp returns exact values at grid knots."""
        from digital_twin.gpu_simulator import _gpu_interp_1d
        result = _gpu_interp_1d(
            sim._z_grid, sim._z_grid, sim._radius_lut,
        )
        assert torch.allclose(result, sim._radius_lut, atol=1e-6)

    def test_1d_linear_between(self, sim):
        """1-D interp is linear between knots."""
        from digital_twin.gpu_simulator import _gpu_interp_1d
        z0 = sim._z_grid[0].item()
        z1 = sim._z_grid[1].item()
        z_mid = (z0 + z1) / 2.0
        r0 = sim._radius_lut[0].item()
        r1 = sim._radius_lut[1].item()
        expected = (r0 + r1) / 2.0

        result = _gpu_interp_1d(
            torch.tensor([z_mid], device=sim.device),
            sim._z_grid, sim._radius_lut,
        ).item()
        assert abs(result - expected) < 1e-5

    def test_1d_clamps_beyond(self, sim):
        """1-D interp clamps at grid boundaries."""
        from digital_twin.gpu_simulator import _gpu_interp_1d
        lo = _gpu_interp_1d(
            torch.tensor([-100.0], device=sim.device),
            sim._z_grid, sim._radius_lut,
        ).item()
        hi = _gpu_interp_1d(
            torch.tensor([9999.0], device=sim.device),
            sim._z_grid, sim._radius_lut,
        ).item()
        assert abs(lo - sim._radius_lut[0].item()) < 1e-5
        assert abs(hi - sim._radius_lut[-1].item()) < 1e-5

    def test_2d_exact_at_knots(self, sim):
        """2-D interp returns exact values at grid intersections."""
        from digital_twin.gpu_simulator import _gpu_interp_2d
        for iz in range(len(sim._z_grid)):
            for iv in range(len(sim._speed_grid)):
                z = sim._z_grid[iz:iz+1]
                v = sim._speed_grid[iv:iv+1]
                result = _gpu_interp_2d(
                    z, v, sim._z_grid, sim._speed_grid, sim._mass_lut,
                ).item()
                expected = sim._mass_lut[iz, iv].item()
                assert abs(result - expected) < 1e-5, (
                    f"Mismatch at grid[{iz},{iv}]: "
                    f"{result} vs {expected}"
                )

    def test_trilinear_corners(self, sim):
        """Trilinear color lookup matches LUT corners."""
        from digital_twin.gpu_simulator import _gpu_trilinear_color
        # corner (0,0,0) -> should match LUT[0,0,0,:]
        cmy = torch.tensor([0.0, 0.0, 0.0], device=sim.device)
        result = _gpu_trilinear_color(
            sim._color_lut, cmy,
            sim._color_domain[0], sim._color_domain[1],
        )
        expected = sim._color_lut[0, :, 0, 0, 0]
        assert torch.allclose(result, expected, atol=0.02)

    def test_trilinear_smooth(self, sim):
        """Trilinear output changes smoothly between queries."""
        from digital_twin.gpu_simulator import _gpu_trilinear_color
        cmy1 = torch.tensor([0.5, 0.5, 0.5], device=sim.device)
        cmy2 = torch.tensor([0.51, 0.5, 0.5], device=sim.device)
        r1 = _gpu_trilinear_color(
            sim._color_lut, cmy1,
            sim._color_domain[0], sim._color_domain[1],
        )
        r2 = _gpu_trilinear_color(
            sim._color_lut, cmy2,
            sim._color_domain[0], sim._color_domain[1],
        )
        diff = (r1 - r2).abs().max().item()
        assert diff < 0.1, f"Large jump for small CMY change: {diff}"


# ============================================================================
# Stroke sampling tests
# ============================================================================

class TestStrokeSampling:
    """Stroke flattening and arc-length sampling."""

    def test_straight_uniform(self, sim, simple_stroke):
        """Straight stroke produces roughly uniform spacing."""
        s = sim.sample_stroke(simple_stroke)
        assert s.n_samples >= sim._min_samples
        ds = s.ds_mm
        max_ds = ds.max().item()
        assert max_ds <= sim._max_step_mm + 1e-3

    def test_curved_min_samples(self, sim, curved_stroke):
        """Curved stroke respects min_samples."""
        s = sim.sample_stroke(curved_stroke)
        assert s.n_samples >= sim._min_samples

    def test_degenerate_zero_length(self, sim):
        """Zero-length stroke returns 0 samples."""
        stroke = {
            "bezier": {
                "p1": (100.0, 100.0), "p2": (100.0, 100.0),
                "p3": (100.0, 100.0), "p4": (100.0, 100.0),
            },
            "z_profile": {"z0": 10.0, "z1": 10.0},
            "speed_profile": {"v0": 50.0, "v1": 50.0},
            "color_cmy": {"c": 0.5, "m": 0.5, "y": 0.5},
        }
        s = sim.sample_stroke(stroke)
        assert s.n_samples == 0

    def test_z_v_interpolation(self, sim, curved_stroke):
        """Z and speed are linearly interpolated along arc length."""
        s = sim.sample_stroke(curved_stroke)
        z0 = curved_stroke["z_profile"]["z0"]
        z1 = curved_stroke["z_profile"]["z1"]
        assert abs(s.z_mm[0].item() - z0) < 1.0
        assert abs(s.z_mm[-1].item() - z1) < 1.0


# ============================================================================
# Batch tests
# ============================================================================

class TestBatch:
    """Batch rendering correctness."""

    def test_batch_independence(self, cal_path):
        """Each batch canvas is independent from the others."""
        from digital_twin.gpu_simulator import GPUStampSimulator
        sim = GPUStampSimulator(
            cal_path, device=torch.device("cpu"), batch_size=4,
        )

        strokes = [
            {
                "bezier": {
                    "p1": (50.0 + i * 20, 100.0),
                    "p2": (70.0 + i * 20, 100.0),
                    "p3": (90.0 + i * 20, 100.0),
                    "p4": (110.0 + i * 20, 100.0),
                },
                "z_profile": {"z0": 5.0 + i * 2, "z1": 5.0 + i * 2},
                "speed_profile": {"v0": 30.0 + i * 20, "v1": 30.0 + i * 20},
                "color_cmy": {"c": 0.2 * (i + 1), "m": 0.1, "y": 0.1},
            }
            for i in range(4)
        ]

        canvas_batch = sim.render_stroke_batch(sim.reset(), strokes)

        # verify each element matches single rendering
        for i in range(4):
            sim_single = GPUStampSimulator(
                cal_path, device=torch.device("cpu"), batch_size=1,
            )
            canvas_single = sim_single.render_stroke(
                sim_single.reset(), strokes[i],
            )
            diff = (
                canvas_batch[i:i+1] - canvas_single
            ).abs().max().item()
            assert diff < 1e-5, (
                f"Batch[{i}] differs from single by {diff}"
            )


# ============================================================================
# Differentiability test
# ============================================================================

class TestDifferentiability:
    """Gradient flow through the compositor."""

    def test_gradient_exists(self, sim, simple_stroke):
        """Canvas with requires_grad produces finite gradients."""
        canvas = sim.reset()
        canvas = canvas.detach().requires_grad_(True)
        out = sim.render_stroke(canvas, simple_stroke)
        loss = out.sum()
        loss.backward()
        assert canvas.grad is not None
        assert torch.isfinite(canvas.grad).all()


# ============================================================================
# Determinism test
# ============================================================================

class TestDeterminism:
    """Bitwise reproducibility."""

    def test_same_output(self, sim, simple_stroke):
        """Same config + same stroke -> identical canvas."""
        c1 = sim.render_stroke(sim.reset(), simple_stroke)
        c2 = sim.render_stroke(sim.reset(), simple_stroke)
        assert torch.equal(c1, c2), "Non-deterministic rendering detected"


# ============================================================================
# Calibration round-trip test
# ============================================================================

class TestCalibrationRoundTrip:
    """Synthetic measurements -> fit -> verify recovery."""

    def test_radius_round_trip(self, cal_path):
        """Generate synthetic dot diameters, fit radius, verify recovery."""
        from digital_twin.calibration_cli import _fit_dots, CalibrationSession
        from rich.console import Console

        cal = validators.load_calibration_config(cal_path)
        original_radius = list(cal.radius_lut_mm)

        # generate synthetic dots at grid z values
        dots = {}
        for i, z in enumerate(cal.z_grid_mm):
            r = cal.radius_lut_mm[i]
            diameter = 2.0 * r
            dots[f"synth_z{z}"] = {
                "test_id": f"synth_z{z}",
                "z_mm": z,
                "color_recipe_cmy": [0.8, 0.2, 0.1],
                "diameter_mm": diameter,
                "center_rgb": [0.3, 0.3, 0.3],
                "mid_rgb": [0.5, 0.5, 0.5],
                "edge_rgb": [0.8, 0.8, 0.8],
                "background_rgb": [1.0, 1.0, 1.0],
            }

        cal_dict = cal.model_dump(mode="json")
        # perturb radius to check that fit recovers original
        cal_dict["radius_lut_mm"] = [r * 1.5 for r in original_radius]
        cal_dict["raw_measurements"]["dots"] = dots

        z_grid = cal_dict["z_grid_mm"]
        z_arr = np.array([d["z_mm"] for d in dots.values()])
        r_arr = np.array([d["diameter_mm"] / 2.0 for d in dots.values()])
        sort_idx = np.argsort(z_arr)
        z_arr = z_arr[sort_idx]
        r_arr = r_arr[sort_idx]
        fitted = np.interp(z_grid, z_arr, r_arr).tolist()

        for i, (orig, fit) in enumerate(zip(original_radius, fitted)):
            assert abs(orig - fit) < 0.01, (
                f"Radius recovery failed at z={z_grid[i]}: "
                f"original={orig}, fitted={fit}"
            )

    def test_color_lut_round_trip(self, cal_path):
        """Swatches at grid corners recover correct LUT entries."""
        cal = validators.load_calibration_config(cal_path)
        ca = cal.color_axes

        # generate swatches at a few CMY grid corners
        test_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
        ]
        expected_rgb = [
            (1.0, 1.0, 1.0),  # CMY=0 -> white
            (0.0, 1.0, 1.0),  # C=1 -> no red
            (1.0, 0.0, 1.0),  # M=1 -> no green
            (1.0, 1.0, 0.0),  # Y=1 -> no blue
            (0.0, 0.0, 0.0),  # all 1 -> black
        ]

        from digital_twin.gpu_simulator import (
            GPUStampSimulator,
            _gpu_trilinear_color,
        )
        sim = GPUStampSimulator(
            cal_path, device=torch.device("cpu"),
        )

        for cmy, exp in zip(test_points, expected_rgb):
            result = _gpu_trilinear_color(
                sim._color_lut,
                torch.tensor(cmy),
                sim._color_domain[0], sim._color_domain[1],
            )
            for ch in range(3):
                assert abs(result[ch].item() - exp[ch]) < 0.05, (
                    f"Color LUT mismatch at CMY={cmy}: "
                    f"expected {exp}, got {result.tolist()}"
                )


# ============================================================================
# Preview tools smoke test
# ============================================================================

class TestPreviewTools:
    """Smoke tests for preview/comparison methods."""

    def test_dot_sheet_runs(self, sim):
        """render_dot_sheet produces a valid canvas."""
        canvas = sim.render_dot_sheet([6.0, 10.0])
        assert canvas.shape == (1, 3, sim._H, sim._W)
        assert canvas.min() >= 0.0
        assert canvas.max() <= 1.0

    def test_line_sheet_runs(self, sim):
        """render_line_sheet produces a valid canvas."""
        canvas = sim.render_line_sheet([6.0], [30.0, 60.0])
        assert canvas.shape == (1, 3, sim._H, sim._W)

    def test_summary_table_runs(self, sim):
        """summary_table builds without error (no measurements)."""
        table = sim.summary_table()
        assert table is not None

    def test_error_heatmap(self, sim):
        """error_heatmap returns correct shape."""
        a = torch.rand(3, 100, 100)
        b = torch.rand(3, 100, 100)
        hm = sim.error_heatmap(a, b)
        assert hm.shape == (100, 100)
