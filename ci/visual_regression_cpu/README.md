# CPU Renderer Visual Regression Tests

This directory contains golden reference images for visual regression testing of the CPU renderer.

## Purpose

These tests serve dual purposes:
1. **Visual Verification**: Demonstrate that the renderer produces realistic, visible strokes
2. **Regression Prevention**: Ensure rendering quality remains consistent across code changes

## Test Categories

### 1. Speed Variation (`speed_variation.png`)
- Tests strokes at different speeds (20-180 mm/s)
- Verifies that faster speeds produce lighter/narrower strokes
- Validates speed-dependent width scaling

### 2. Z Height Variation (`z_height_variation.png`)
- Tests strokes at different nozzle heights (3-16 mm)
- Verifies that higher Z produces wider strokes
- Validates Z-dependent width model

### 3. Color Palette (`color_palette.png`)
- Tests different CMY color combinations
- Demonstrates black, cyan, magenta, yellow, red, green, blue
- Validates color LUT interpolation

### 4. Bézier Curves (`bezier_curves.png`)
- Tests various curve shapes: straight, S-curve, sharp turn, loop, wave
- Validates Bézier flattening and polyline rendering
- Ensures smooth curves without artifacts

### 5. Color Layering (`color_layering.png`)
- Tests overlapping strokes in different colors
- Validates alpha-over compositing
- Demonstrates color mixing behavior

### 6. Gradient Strokes (`gradient_strokes.png`)
- Tests strokes with varying Z and speed along the path
- Validates smooth transitions in width and opacity
- Demonstrates gradient capabilities

### 7. Complex Composition (`complex_composition.png`)
- Tests multiple overlapping strokes in a grid pattern
- Validates layering and compositing at scale
- Demonstrates realistic multi-stroke rendering

### 8. All Features (`all_features.png`)
- Comprehensive demo combining all renderer features
- Shows speed, colors, curves, and layering in one image
- Serves as a quick visual sanity check

## Running the Tests

```bash
# Run all visual regression tests
pytest tests/test_cpu_visual_regression.py -m visual -v

# Run specific test
pytest tests/test_cpu_visual_regression.py::test_visual_color_palette -v

# Regenerate all golden images (use with caution!)
pytest tests/test_cpu_visual_regression.py -m visual --regen-visual
```

## Image Specifications

- **Format**: PNG (lossless)
- **Color Space**: sRGB (converted from linear RGB)
- **Resolution**: 908×1280 pixels (matches render_px)
- **Background**: White (#FFFFFF)
- **Stroke Colors**: Black or CMY combinations

## Notes

- Images use `high_visibility=True` toy LUTs for easy verification
- Actual colors may appear as black due to simplified color model
- Real calibrated LUTs will produce accurate CMY colors
- All strokes use realistic physics parameters (Z, speed, width, mass)

## Maintenance

When updating the renderer:
1. Run tests to ensure no regressions
2. If intentional changes are made, regenerate golden images
3. Review images manually to verify quality
4. Commit updated images to version control

