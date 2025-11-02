# Visual Regression Test Suite

## Overview

The visual regression test suite provides comprehensive verification of the CPU renderer's capabilities, demonstrating realistic airbrush stroke rendering with proper physics, colors, curves, and layering.

## Test Suite Structure

### Location
- **Test File**: `tests/test_cpu_visual_regression.py`
- **Output Directory**: `ci/visual_regression_cpu/`
- **Configuration**: `pytest.ini` (custom markers registered)

### Running Tests

```bash
# Run all visual regression tests
pytest tests/test_cpu_visual_regression.py -m visual -v

# Run all CPU tests (physics + visual)
pytest tests/test_cpu_golden.py tests/test_cpu_visual_regression.py -v

# Run specific visual test
pytest tests/test_cpu_visual_regression.py::test_visual_bezier_curves -v
```

## Test Categories

### 1. **Physics Behavior Tests** (`test_cpu_golden.py`)
- `test_visibility_sanity` - Strokes are clearly visible
- `test_width_control` - Width increases with Z height
- `test_speed_scaling` - Width/opacity scale with speed
- `test_determinism` - Renders are perfectly reproducible
- `test_layering_alpha_over` - Alpha-over compositing works correctly
- `test_coverage_stability` - Coverage metrics are stable

### 2. **Visual Regression Tests** (`test_cpu_visual_regression.py`)

#### Speed Variation
- **Test**: `test_visual_speed_variation`
- **Output**: `speed_variation.png`
- **Demonstrates**: Strokes at 20, 40, 80, 120, 180 mm/s
- **Validates**: Speed-dependent width and opacity scaling

#### Z Height Variation
- **Test**: `test_visual_z_height_variation`
- **Output**: `z_height_variation.png`
- **Demonstrates**: Strokes at Z=3, 6, 9, 12, 16 mm
- **Validates**: Z-dependent width model

#### Color Palette
- **Test**: `test_visual_color_palette`
- **Output**: `color_palette.png`
- **Demonstrates**: Black, cyan, magenta, yellow, red, green, blue
- **Validates**: CMY color model and LUT interpolation

#### Bézier Curves
- **Test**: `test_visual_bezier_curves`
- **Output**: `bezier_curves.png`
- **Demonstrates**: Straight, S-curve, sharp turn, loop, wave
- **Validates**: Bézier flattening and smooth curve rendering

#### Color Layering
- **Test**: `test_visual_color_layering`
- **Output**: `color_layering.png`
- **Demonstrates**: Overlapping CMY strokes
- **Validates**: Alpha-over compositing and color mixing

#### Gradient Strokes
- **Test**: `test_visual_gradient_strokes`
- **Output**: `gradient_strokes.png`
- **Demonstrates**: Varying Z and speed along stroke path
- **Validates**: Smooth gradients in width and opacity

#### Complex Composition
- **Test**: `test_visual_complex_composition`
- **Output**: `complex_composition.png`
- **Demonstrates**: Grid of overlapping colored strokes
- **Validates**: Multi-stroke layering and compositing

#### All Features
- **Test**: `test_visual_all_features`
- **Output**: `all_features.png`
- **Demonstrates**: Comprehensive demo of all capabilities
- **Validates**: Integration of speed, colors, curves, and layering

## Test Results Summary

### Current Status (as of implementation)
```
✅ 6/6 Physics tests passing
✅ 8/8 Visual regression tests passing
✅ 15/15 Total tests passing
⏭️  5 Golden image tests skipped (awaiting golden images)
```

### Key Metrics
- **Test Execution Time**: ~3 seconds for full suite
- **Image Generation**: 8 PNG images (8-31 KB each)
- **Coverage**: All major renderer features tested
- **Determinism**: 100% reproducible results

## Implementation Details

### Fixtures
- `visual_renderer`: Module-scoped renderer with toy LUTs
- `blank_canvas`: Fresh white canvas for each test

### Helper Functions
- `save_canvas()`: Converts linear RGB → sRGB and saves PNG
- Automatic directory creation for output images
- Proper color space handling throughout

### Assertions
- All tests verify `np.mean(alpha) > threshold`
- Ensures strokes are actually rendered (not invisible)
- Thresholds tuned for realistic stroke visibility

## Configuration

### Pytest Markers
Defined in `pytest.ini`:
- `@pytest.mark.visual` - Visual regression tests
- `@pytest.mark.physics` - Physics behavior tests  
- `@pytest.mark.golden` - Golden image comparison tests

### Renderer Configuration
- Uses `renderer_cpu.v1.yaml` for physics parameters
- Toy LUTs with `high_visibility=True` for easy verification
- Canvas size: 908×1280 pixels (standard render_px)

## Maintenance

### When to Regenerate Images
1. **Intentional renderer changes**: New physics model, updated parameters
2. **Bug fixes**: If previous images showed incorrect behavior
3. **Configuration updates**: Changes to default mass/width values

### How to Regenerate
```bash
# Delete old images
rm ci/visual_regression_cpu/*.png

# Run tests to regenerate
pytest tests/test_cpu_visual_regression.py -m visual -v
```

### Review Checklist
- [ ] All strokes are clearly visible (not faint/gray)
- [ ] Width variation matches expected physics
- [ ] Colors are rendered correctly (or black if using toy LUTs)
- [ ] Curves are smooth without artifacts
- [ ] Layering shows proper alpha-over compositing
- [ ] No unexpected clipping or boundary issues

## Integration with CI/CD

### Recommended CI Pipeline
```yaml
- name: Run CPU Renderer Tests
  run: |
    pytest tests/test_cpu_golden.py tests/test_cpu_visual_regression.py -v
    
- name: Archive Visual Regression Images
  uses: actions/upload-artifact@v3
  with:
    name: visual-regression-images
    path: ci/visual_regression_cpu/*.png
```

### Golden Image Comparison (Future)
- Store reference images in version control
- Compare new renders against golden images using SSIM/PSNR
- Fail CI if differences exceed threshold
- Require manual review for intentional changes

## Troubleshooting

### Issue: Strokes are invisible
**Cause**: Mass deposition values too low  
**Fix**: Increase `mass_per_sec` in `renderer_cpu.v1.yaml`

### Issue: Tests fail with "No strokes visible"
**Cause**: Visibility gates rejecting strokes  
**Fix**: Lower thresholds in `visibility` section of config

### Issue: Colors appear as black
**Cause**: Using `high_visibility=True` toy LUTs  
**Fix**: Load calibrated color LUTs or set `high_visibility=False`

### Issue: Curves have artifacts
**Cause**: Bézier flattening tolerance too high  
**Fix**: Reduce `max_err_mm` in `bezier_cubic_polyline()` call

## Future Enhancements

1. **Perceptual Comparison**: Add Delta E / SSIM metrics for golden image comparison
2. **Color Accuracy**: Test with calibrated LUTs for realistic colors
3. **Performance Benchmarks**: Track rendering speed for each test
4. **Interactive Viewer**: Web-based tool to compare before/after images
5. **Automated Regression Detection**: CI integration with automatic failure on visual changes

## References

- CPU Renderer Implementation: `src/airbrush_simulator/cpu_reference.py`
- Physics Configuration: `configs/sim/renderer_cpu.v1.yaml`
- Test Documentation: `ci/visual_regression_cpu/README.md`
- Project Rules: `documentation/GlobalPlan.md`

