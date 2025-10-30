#!/usr/bin/env python3
"""Extended validation for newly implemented utils modules.

Tests validators, gcode_generator, gcode_vm, and mlflow_helpers.

Test coverage:
- validators: Schema validation, bounds checking, error messages
- gcode_generator: Coordinate transforms, linearization, soft limits
- gcode_vm: G-code parsing, time estimation, violation detection
- mlflow_helpers: Parameter logging, artifact management (no-op if MLflow unavailable)
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_validators():
    """Test pydantic validators for all schemas."""
    print("=" * 60)
    print("1. Testing Validators")
    print("=" * 60)
    
    try:
        from src.utils import validators, fs
        import torch
        
        # Test stroke bounds helpers
        bounds = validators.get_stroke_bounds()
        assert 'x' in bounds and 'y' in bounds and 'z' in bounds
        # Canvas bounds (now offset from machine origin)
        assert bounds['x'] == (20.0, 230.0)  # A4 width, centered at x=20mm
        assert bounds['y'] == (26.5, 323.5)  # A4 height, centered at y=26.5mm
        print("✅ get_stroke_bounds() returns A4 canvas dimensions (offset from machine origin)")
        
        # Test valid stroke
        stroke_data = {
            'id': 'test-stroke-12345',
            'bezier': {
                'p1': [10.0, 10.0],
                'p2': [50.0, 20.0],
                'p3': [100.0, 30.0],
                'p4': [150.0, 40.0],
            },
            'z_profile': {'z0': 10.0, 'z1': 15.0},
            'speed_profile': {'v0': 100.0, 'v1': 150.0},
            'color_cmy': {'c': 0.5, 'm': 0.3, 'y': 0.7},
        }
        stroke = validators.StrokeV1(**stroke_data)
        assert stroke.id == 'test-stroke-12345'
        print("✅ Valid stroke passes validation")
        
        # Test out-of-bounds stroke (should fail)
        bad_stroke_data = stroke_data.copy()
        bad_stroke_data['bezier'] = {
            'p1': [10.0, 10.0],
            'p2': [50.0, 20.0],
            'p3': [100.0, 30.0],
            'p4': [300.0, 40.0],  # X > 210, out of bounds
        }
        try:
            validators.StrokeV1(**bad_stroke_data)
            print("❌ Out-of-bounds stroke should have failed")
            return False
        except ValueError as e:
            assert 'out of bounds' in str(e).lower()
            print("✅ Out-of-bounds stroke rejected with clear message")
        
        # Test machine profile loading
        machine_cfg = validators.load_machine_profile(
            "configs/machine_grbl_airbrush_v1.yaml"
        )
        assert machine_cfg.schema_version == "machine.v1"
        assert machine_cfg.work_area_mm.x == 250.0  # Machine can travel 250mm
        assert machine_cfg.canvas_mm.x_min == 20.0  # Canvas offset
        assert machine_cfg.canvas_mm.x_max == 230.0  # 210mm wide A4
        assert machine_cfg.gcode_flavor == "grbl_1.1f"
        print("✅ load_machine_profile() works with real config (machine + canvas bounds)")
        
        # Test flatten_config_for_mlflow
        nested_cfg = {
            'agent': {
                'learning_rate': 0.001,
                'gamma': 0.99,
            },
            'env': {
                'render_px': 908,
            }
        }
        flat = validators.flatten_config_for_mlflow(nested_cfg)
        assert flat['agent.learning_rate'] == 0.001
        assert flat['agent.gamma'] == 0.99
        assert flat['env.render_px'] == 908
        print("✅ flatten_config_for_mlflow() correctly flattens nested dicts")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Validators test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gcode_generator():
    """Test G-code generation with coordinate transforms."""
    print("=" * 60)
    print("2. Testing G-code Generator")
    print("=" * 60)
    
    try:
        from src.utils import gcode_generator, validators, fs
        import torch
        
        # Load machine config
        machine_cfg = validators.load_machine_profile(
            "configs/machine_grbl_airbrush_v1.yaml"
        )
        
        # Test coordinate frame transform (critical boundary)
        pts_img = torch.tensor([
            [0.0, 0.0],       # Top-left corner of canvas (image frame)
            [210.0, 0.0],     # Top-right corner of canvas
            [0.0, 297.0],     # Bottom-left corner of canvas
            [210.0, 297.0],   # Bottom-right corner of canvas
            [105.0, 148.5],   # Center of canvas
        ])
        
        # Canvas bounds (x_min, x_max, y_min, y_max)
        canvas_bounds = (20.0, 230.0, 26.5, 323.5)
        pts_mach = gcode_generator.image_mm_to_machine_mm(
            pts_img, canvas_bounds, flip_y=True
        )
        
        # Image (0, 0) → Machine (20.0, 323.5) - top-left of canvas
        assert torch.allclose(pts_mach[0], torch.tensor([20.0, 323.5]), atol=1e-6)
        # Image (210, 0) → Machine (230.0, 323.5) - top-right of canvas
        assert torch.allclose(pts_mach[1], torch.tensor([230.0, 323.5]), atol=1e-6)
        # Image (0, 297) → Machine (20.0, 26.5) - bottom-left of canvas
        assert torch.allclose(pts_mach[2], torch.tensor([20.0, 26.5]), atol=1e-6)
        # Image (210, 297) → Machine (230.0, 26.5) - bottom-right of canvas
        assert torch.allclose(pts_mach[3], torch.tensor([230.0, 26.5]), atol=1e-6)
        # Center: Image (105, 148.5) → Machine (125, 175) - center of canvas
        assert torch.allclose(pts_mach[4], torch.tensor([125.0, 175.0]), atol=1e-6)
        print("✅ Coordinate frame transform (image → machine) correct with canvas offset")
        
        # Test header generation
        header = gcode_generator.generate_gcode_header(machine_cfg)
        header_str = ''.join(header)
        assert 'G21' in header_str  # mm units
        assert 'G90' in header_str  # absolute positioning
        assert 'grbl_1.1f' in header_str
        print("✅ G-code header contains required commands")
        
        # Test feed conversion
        feed_mm_min = gcode_generator.speed_mm_s_to_feed(100.0, "mm/min")
        assert feed_mm_min == 6000.0  # 100 mm/s * 60
        feed_mm_s = gcode_generator.speed_mm_s_to_feed(100.0, "mm/s")
        assert feed_mm_s == 100.0
        print("✅ Speed → feed conversion correct")
        
        # Test soft limit validation (machine limit is now 250mm, so 260 violates)
        try:
            gcode_generator.validate_soft_limits(260.0, 100.0, 10.0, machine_cfg)
            print("❌ Soft limit violation should have been caught")
            return False
        except ValueError as e:
            assert 'out of bounds' in str(e).lower()
            print("✅ Soft limit violation detected")
        
        # Test stroke linearization
        stroke_data = {
            'id': 'test-stroke-001',
            'bezier': {
                'p1': [50.0, 50.0],
                'p2': [70.0, 60.0],
                'p3': [90.0, 70.0],
                'p4': [110.0, 80.0],
            },
            'z_profile': {'z0': 10.0, 'z1': 15.0},
            'speed_profile': {'v0': 100.0, 'v1': 150.0},
            'color_cmy': {'c': 0.5, 'm': 0.3, 'y': 0.7},
        }
        stroke = validators.StrokeV1(**stroke_data)
        
        # Canvas bounds now implicit from machine_cfg
        pts_mach, z_interp, v_interp = gcode_generator.linearize_stroke(
            stroke, machine_cfg, max_err_mm=0.5
        )
        
        assert pts_mach.shape[0] >= 2  # At least start and end
        assert z_interp.shape[0] == pts_mach.shape[0]
        assert v_interp.shape[0] == pts_mach.shape[0]
        assert torch.all(z_interp >= 10.0) and torch.all(z_interp <= 15.0)
        assert torch.all(v_interp >= 100.0) and torch.all(v_interp <= 150.0)
        print("✅ Stroke linearization produces valid polyline with interpolated Z/speed")
        
        # Test full G-code generation (to temp file)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            strokes_file_data = {
                'schema': 'stroke.v1',
                'strokes': [stroke_data]
            }
            strokes = validators.StrokesFileV1(**strokes_file_data)
            
            gcode_path = tmpdir / "test.gcode"
            # Canvas bounds implicit from machine_cfg
            gcode_generator.generate_cmy_gcode(
                strokes, machine_cfg, gcode_path
            )
            
            assert gcode_path.exists()
            gcode_content = gcode_path.read_text()
            assert 'G21' in gcode_content  # mm units
            assert 'G1' in gcode_content   # linear moves
            assert 'STROKE_ID: test-stroke-001' in gcode_content
            print("✅ Full G-code generation writes valid file with stroke IDs")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ G-code generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gcode_vm():
    """Test G-code virtual machine (dry-run simulator)."""
    print("=" * 60)
    print("3. Testing G-code VM")
    print("=" * 60)
    
    try:
        from src.utils import gcode_vm, validators
        import math
        
        # Load machine config
        machine_cfg = validators.load_machine_profile(
            "configs/machine_grbl_airbrush_v1.yaml"
        )
        
        # Create VM
        vm = gcode_vm.GCodeVM(machine_cfg, purge_time_s=1.0, pen_time_s=0.2)
        
        # Test simple G-code
        test_gcode = """
; Test G-code
G21            ; mm units
G90            ; absolute positioning
G1 X10 Y10 Z5 F3000
G1 X20 Y20 Z5 F3000
G1 X30 Y20 Z5 F3000
"""
        vm.load_string(test_gcode)
        result = vm.run()
        
        assert result['move_count'] == 3
        assert result['time_estimate_s'] > 0
        assert len(result['violations']) == 0
        assert result['final_pos'][0] == 30.0
        assert result['final_pos'][1] == 20.0
        print("✅ VM executes simple G-code without violations")
        print(f"   Moves: {result['move_count']}, Time: {result['time_estimate_s']:.3f}s")
        
        # Test soft limit violation (machine limit is 250mm, so 260 violates)
        vm_strict = gcode_vm.GCodeVM(machine_cfg)
        bad_gcode = """
G21
G90
G1 X260 Y100 Z5 F3000
"""
        vm_strict.load_string(bad_gcode)
        result = vm_strict.run()
        
        assert len(result['violations']) > 0
        assert 'machine limit' in result['violations'][0].lower()
        print("✅ VM detects soft limit violations")
        
        # Test stroke ID tracking
        gcode_with_id = """
; STROKE_ID: stroke-abc-123
G1 X10 Y10 Z5 F3000
"""
        vm_id = gcode_vm.GCodeVM(machine_cfg)
        vm_id.load_string(gcode_with_id)
        result = vm_id.run()
        
        assert result['last_stroke_id'] == 'stroke-abc-123'
        print("✅ VM tracks stroke IDs from comments")
        
        # Test time estimation (trapezoidal motion)
        vm_trap = gcode_vm.GCodeVM(machine_cfg, accel_mm_s2=1000.0)
        move_gcode = """
G21
G90
G1 X100 Y0 Z0 F6000
"""
        vm_trap.load_string(move_gcode)
        result_trap = vm_trap.run()
        
        # 100mm at 100mm/s (6000mm/min) with accel should take roughly 1-2s
        assert 0.5 < result_trap['time_estimate_s'] < 5.0
        print(f"✅ VM time estimation with accel: {result_trap['time_estimate_s']:.3f}s for 100mm move")
        
        # Test trajectory extraction
        vm_traj = gcode_vm.GCodeVM(machine_cfg)
        vm_traj.load_string(test_gcode)
        trajectory = vm_traj.get_trajectory()
        
        assert len(trajectory) == 4  # Initial + 3 moves
        assert trajectory[0] == (0.0, 0.0, 0.0)
        assert trajectory[-1] == (30.0, 20.0, 5.0)
        print("✅ VM extracts trajectory correctly")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ G-code VM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mlflow_helpers():
    """Test MLflow helpers (graceful no-op if MLflow unavailable)."""
    print("=" * 60)
    print("4. Testing MLflow Helpers")
    print("=" * 60)
    
    try:
        from src.utils import mlflow_helpers, hashing
        import tempfile
        
        # Check if MLflow is available
        if not mlflow_helpers.MLFLOW_AVAILABLE:
            print("⚠️  MLflow not installed, testing graceful degradation")
            
            # All functions should be no-ops
            mlflow_helpers.setup_experiment("test_exp")
            with mlflow_helpers.start_run("test_run"):
                mlflow_helpers.log_params_flat({'test': 123})
                mlflow_helpers.log_metrics({'metric': 1.0}, step=0)
            
            print("✅ MLflow helpers degrade gracefully when MLflow unavailable")
            print()
            return True
        
        print("✅ MLflow available, testing full functionality")
        
        # Test parameter flattening (already tested in validators, but test here too)
        params = {
            'agent': {'lr': 0.001},
            'env': {'size': 100}
        }
        mlflow_helpers.log_params_flat(params)
        print("✅ log_params_flat() works")
        
        # Test resolution logging
        mlflow_helpers.log_resolutions(render_px=908, obs_px=128, reward_px=454)
        print("✅ log_resolutions() works")
        
        # Test LUT hash logging
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create dummy LUT file
            lut_path = tmpdir / "dummy_lut.pt"
            lut_path.write_bytes(b"dummy data")
            
            lut_paths = {'color_lut': lut_path}
            mlflow_helpers.log_lut_hashes(lut_paths)
            print("✅ log_lut_hashes() works")
        
        # Test reproducibility info
        mlflow_helpers.log_reproducibility_info(seed=123, cudnn_deterministic=True)
        print("✅ log_reproducibility_info() works")
        
        # Test artifact logging (with temp file)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            artifact_path = tmpdir / "artifact.txt"
            artifact_path.write_text("test content")
            
            mlflow_helpers.log_artifact(artifact_path)
            mlflow_helpers.log_artifact_with_hash(artifact_path, log_hash_as_param=True)
            print("✅ log_artifact() and log_artifact_with_hash() work")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ MLflow helpers test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between new modules."""
    print("=" * 60)
    print("5. Testing Integration (validators → gcode_gen → vm)")
    print("=" * 60)
    
    try:
        from src.utils import validators, gcode_generator, gcode_vm, fs
        import tempfile
        
        # Load machine config
        machine_cfg = validators.load_machine_profile(
            "configs/machine_grbl_airbrush_v1.yaml"
        )
        
        # Create valid strokes
        strokes_data = {
            'schema': 'stroke.v1',
            'strokes': [
                {
                    'id': f'stroke-{i:03d}',
                    'bezier': {
                        'p1': [20.0 + i*10, 50.0],
                        'p2': [30.0 + i*10, 60.0],
                        'p3': [40.0 + i*10, 70.0],
                        'p4': [50.0 + i*10, 80.0],
                    },
                    'z_profile': {'z0': 10.0, 'z1': 12.0},
                    'speed_profile': {'v0': 100.0, 'v1': 120.0},
                    'color_cmy': {'c': 0.3, 'm': 0.5, 'y': 0.7},
                }
                for i in range(3)
            ]
        }
        
        # Validate strokes
        strokes = validators.StrokesFileV1(**strokes_data)
        assert len(strokes.strokes) == 3
        print("✅ Created and validated 3 strokes")
        
        # Generate G-code
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            gcode_path = tmpdir / "integration_test.gcode"
            
            # Canvas bounds implicit from machine_cfg
            gcode_generator.generate_cmy_gcode(
                strokes, machine_cfg, gcode_path
            )
            
            assert gcode_path.exists()
            print("✅ Generated G-code file")
            
            # Validate with VM
            vm = gcode_vm.GCodeVM(machine_cfg)
            vm.load_file(gcode_path)
            result = vm.run()
            
            assert len(result['violations']) == 0, f"Unexpected violations: {result['violations']}"
            assert result['move_count'] > 3  # Should have multiple moves per stroke
            assert result['time_estimate_s'] > 0
            print(f"✅ VM validated G-code: {result['move_count']} moves, "
                  f"{result['time_estimate_s']:.2f}s, no violations")
            
            # Check stroke ID tracking
            gcode_content = gcode_path.read_text()
            for i in range(3):
                assert f'STROKE_ID: stroke-{i:03d}' in gcode_content
            print("✅ All stroke IDs present in G-code")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("=" * 60)
    print("6. Testing Edge Cases")
    print("=" * 60)
    
    try:
        from src.utils import validators, gcode_generator, gcode_vm
        import torch
        
        # Test empty strokes
        empty_strokes = validators.StrokesFileV1(schema='stroke.v1', strokes=[])
        assert len(empty_strokes.strokes) == 0
        print("✅ Empty strokes container is valid")
        
        # Test minimal stroke ID
        try:
            validators.StrokeV1(
                id='abc',  # Only 3 chars (min is 5)
                bezier={'p1': [0,0], 'p2': [1,1], 'p3': [2,2], 'p4': [3,3]},
                z_profile={'z0': 10, 'z1': 10},
                speed_profile={'v0': 100, 'v1': 100},
                color_cmy={'c': 0, 'm': 0, 'y': 0}
            )
            print("❌ Short stroke ID should have failed")
            return False
        except ValueError:
            print("✅ Short stroke ID rejected")
        
        # Test Z bounds
        try:
            validators.ZProfile(z0=-1.0, z1=10.0)
            print("❌ Negative Z should have failed")
            return False
        except ValueError:
            print("✅ Negative Z rejected")
        
        try:
            validators.ZProfile(z0=10.0, z1=50.0)  # Max is 30
            print("❌ Z > 30 should have failed")
            return False
        except ValueError:
            print("✅ Z > 30 rejected")
        
        # Test speed bounds
        try:
            validators.SpeedProfile(v0=0.5, v1=100.0)  # Min is 1.0
            print("❌ Speed < 1 should have failed")
            return False
        except ValueError:
            print("✅ Speed < 1 rejected")
        
        # Test CMY bounds
        try:
            validators.ColorCMY(c=1.5, m=0.5, y=0.5)  # Max is 1.0
            print("❌ CMY > 1 should have failed")
            return False
        except ValueError:
            print("✅ CMY > 1 rejected")
        
        # Test VM with empty G-code
        machine_cfg = validators.load_machine_profile(
            "configs/machine_grbl_airbrush_v1.yaml"
        )
        vm = gcode_vm.GCodeVM(machine_cfg)
        try:
            vm.run()  # No G-code loaded
            print("❌ Running VM without G-code should fail")
            return False
        except RuntimeError:
            print("✅ VM run() fails gracefully without loaded G-code")
        
        # Test coordinate transform with canvas_bounds (tuple format: x_min, x_max, y_min, y_max)
        pts = torch.tensor([[10.0, 20.0]])
        canvas_bounds = (0.0, 100.0, 0.0, 200.0)  # Canvas at origin
        pts_mach = gcode_generator.image_mm_to_machine_mm(
            pts, canvas_bounds, flip_y=False
        )
        # With flip_y=False and canvas at origin, only offset applied
        assert torch.allclose(pts, pts_mach)
        print("✅ Coordinate transform with canvas bounds correct")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vm_microfixes():
    """Test VM micro-fixes from review."""
    try:
        from src.utils import validators, gcode_vm
        
        machine_cfg = validators.load_machine_profile("configs/machine_grbl_airbrush_v1.yaml")
        
        # Test 1: Triangular profile (short move, high accel)
        vm = gcode_vm.GCodeVM(machine_cfg, accel_mm_s2=2000.0)
        vm.load_string("G21\nG90\nG1 X1.0 F6000\n")  # 1 mm move at 100 mm/s
        r = vm.run()
        # Expected ~ 2*sqrt(1/2000) = ~0.0447 s
        assert 0.03 < r['time_estimate_s'] < 0.06, f"Triangular profile time {r['time_estimate_s']} not in expected range"
        print("✅ Triangular profile time formula correct")
        
        # Test 2: Inches feed scaling
        vm = gcode_vm.GCodeVM(machine_cfg)
        vm.load_string("G20\nG90\nG1 X1.0 F60\n")  # 60 in/min should become 1524 mm/min
        vm.run()
        assert abs(vm.feed - 60*25.4) < 1e-3, f"Feed scaling failed: {vm.feed} != {60*25.4}"
        print("✅ Inches feed scaling correct")
        
        # Test 3: G0 rapid timing (faster than G1)
        vm = gcode_vm.GCodeVM(machine_cfg)
        vm.rapid_mm_s = 200.0
        vm.load_string("G21\nG90\nG0 X100\n")
        r_rapid = vm.run()
        
        vm.reset()
        vm.load_string("G21\nG90\nG1 X100 F6000\n")  # 100 mm/s
        r_normal = vm.run()
        
        assert r_rapid['time_estimate_s'] < r_normal['time_estimate_s'], \
               f"G0 should be faster than G1: {r_rapid['time_estimate_s']} vs {r_normal['time_estimate_s']}"
        print(f"✅ G0 rapid timing correct (rapid: {r_rapid['time_estimate_s']:.3f}s, G1: {r_normal['time_estimate_s']:.3f}s)")
        
        # Test 4: Parser robustness (X.5, G10 not a move)
        vm = gcode_vm.GCodeVM(machine_cfg)
        vm.load_string("G21\nG90\nG1 X.5 F6000\nG10 L2 P1 X0\n")  # G10 is not a move
        r = vm.run()
        assert r['move_count'] == 1, f"Expected 1 move, got {r['move_count']}"
        assert abs(vm.pos[0] - 0.5) < 1e-6, f"X should be 0.5, got {vm.pos[0]}"
        print("✅ Parser accepts X.5 and rejects G10 as move")
        
        # Test 5: Line numbers in violations (machine limit is 250mm, so 260 violates)
        vm = gcode_vm.GCodeVM(machine_cfg)
        vm.load_string("G21\nG90\nG1 X260 F6000\n")  # Out of machine bounds
        r = vm.run()
        assert len(r['violations']) > 0, f"Expected violation for X=260, got {r['violations']}"
        assert "line 3" in r['violations'][0], f"Expected line number in violation: {r['violations'][0]}"
        print("✅ Line numbers appear in violations")
        
        return True
    except Exception as e:
        print(f"❌ VM micro-fixes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all extended validation tests."""
    print("\n" + "=" * 60)
    print("EXTENDED UTILS VALIDATION")
    print("Testing: validators, gcode_generator, gcode_vm, mlflow_helpers")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run all tests
    results.append(("Validators", test_validators()))
    results.append(("G-code Generator", test_gcode_generator()))
    results.append(("G-code VM", test_gcode_vm()))
    results.append(("MLflow Helpers", test_mlflow_helpers()))
    results.append(("Integration", test_integration()))
    results.append(("Edge Cases", test_edge_cases()))
    
    # Additional micro-fix tests
    print("=" * 60)
    print("7. Testing VM Micro-Fixes")
    print("=" * 60)
    results.append(("VM Micro-Fixes", test_vm_microfixes()))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:30s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 60)
    if all_passed:
        print("🎉 All extended tests passed!")
        print("✅ New utils modules are validated and production-ready")
        return 0
    else:
        print("⚠️  Some tests failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

