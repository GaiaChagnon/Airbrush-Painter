#!/usr/bin/env python3
"""Quick validation script to test utils imports and basic functionality.

Run this to verify all utils modules are properly implemented and importable.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all utils modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.utils import (
            color,
            compute,
            fs,
            geometry,
            hashing,
            logging_config,
            metrics,
            profiler,
            strokes,
            torch_utils,
            validators
        )
        print("✅ All utils modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of key modules."""
    print("\nTesting basic functionality...")
    
    try:
        import torch
        from src.utils import compute, color, geometry, strokes
        
        # Test compute
        img = torch.rand(3, 100, 100)
        normalized = compute.normalize_img_for_lpips(img)
        assert normalized.min() >= -1.1 and normalized.max() <= 1.1, "LPIPS normalization out of range"
        print("✅ compute.normalize_img_for_lpips works")
        
        # Test color
        srgb = torch.rand(3, 10, 10)
        linear = color.srgb_to_linear(srgb)
        srgb_back = color.linear_to_srgb(linear)
        assert torch.allclose(srgb, srgb_back, atol=1e-5), "sRGB roundtrip failed"
        print("✅ color.srgb_to_linear/linear_to_srgb roundtrip works")
        
        # Test geometry
        p1 = torch.tensor([0.0, 0.0])
        p2 = torch.tensor([50.0, 0.0])
        p3 = torch.tensor([100.0, 50.0])
        p4 = torch.tensor([150.0, 50.0])
        polyline = geometry.bezier_cubic_polyline(p1, p2, p3, p4, max_err_px=0.5)
        assert polyline.shape[0] >= 2, "Polyline should have at least 2 points"
        print("✅ geometry.bezier_cubic_polyline works")
        
        # Test strokes
        vec_mm = torch.randn(15)
        stroke_dict = strokes.stroke_vec_to_yaml_dict(vec_mm)
        vec_back = strokes.stroke_yaml_dict_to_vec(stroke_dict)
        assert vec_mm.shape == vec_back.shape, "Stroke vec shape mismatch"
        print("✅ strokes vec↔YAML roundtrip works")
        
        # Test mm↔px conversion
        x_mm = torch.tensor([[50.0, 100.0], [100.0, 200.0]])
        work_area = (210.0, 297.0)
        render_px = (908, 1280)
        x_px = compute.mm_to_px(x_mm, work_area, render_px)
        x_mm_back = compute.px_to_mm(x_px, work_area, render_px)
        assert torch.allclose(x_mm, x_mm_back, atol=0.01), "mm↔px roundtrip failed"
        print("✅ compute.mm_to_px/px_to_mm roundtrip works")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logging():
    """Test logging configuration."""
    print("\nTesting logging...")
    
    try:
        from src.utils.logging_config import setup_logging, get_logger, push_context
        
        # Setup logging (to stderr only, no file)
        setup_logging(log_level="INFO", to_stderr=False, context={"app": "test"})
        
        logger = get_logger(__name__)
        push_context(test_id=42)
        
        # This should work without errors
        logger.info("Test log message")
        
        print("✅ Logging configuration works")
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Utils Module Validation")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test basic functionality
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Test logging
    results.append(("Logging", test_logging()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:30s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

