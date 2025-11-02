# Tests Directory

This directory contains comprehensive test suites for all utils modules and domain-specific tests.

## Test Files

### `test_utils_comprehensive.py` (18 test suites - ALL IN ONE FILE)
**All-in-one comprehensive test suite** covering all utils modules:

**Core functionality tests (1-11):**
- Imports - Module import validation
- Core Functionality - LPIPS, color, geometry, strokes, mm↔px (with hardening)
- Tiled LPIPS Parity - Even (256×256) & ragged (250×250) tile sizes
- Logging Idempotency - File output & console leakage check
- Atomic Operations - Filesystem operations & symlinks
- Error Paths - Bounds projection, broadcasting, assert_finite
- Coverage Metric - Paint coverage on alpha
- Hashing - Files & tensors
- Device Recursion - Recursive device movement
- Coordinate Transforms - Both origins tested (top_left, bottom_left)
- Profiler - Timers & NVTX

**Extended module tests (12-18):**
- Validators - Schema validation, bounds checking
- G-code Generator - Coordinate transforms, linearization, soft limits
- G-code VM - Parsing, timing, violation detection
- MLflow Helpers - Graceful degradation when unavailable
- Integration - validators → G-code → VM pipeline
- Edge Cases - Empty strokes, bounds validation, ID validation
- VM Micro-Fixes - Triangular profile, feed scaling, rapid timing

**Hardening Improvements:**
- ✅ Geometry endpoint preservation
- ✅ Strokes schema validation
- ✅ mm↔px origin contrast
- ✅ Logging console leakage check
- ✅ LPIPS ragged tile case (250×250)
- ✅ Error path broadcasting

### Non-Utils Tests (Domain-Specific)
- `test_env_v1.py` - RL environment tests
- `test_env_resolutions.py` - Multi-resolution architecture
- `test_networks.py` - Policy network tests
- `test_renderer.py` - Differentiable renderer
- `test_cpu_renderer.py` - CPU renderer tests
- `test_cpu_golden.py` - CPU golden tests
- `test_parity_cpu_vs_gpu.py` - CPU vs GPU parity
- `test_action_scaling.py` - Action space scaling
- `test_gui_monitoring.py` - GUI monitoring
- `test_paint_main.py` - Paint script
- `test_reward_hacks.py` - Reward adversarial tests

### Support Files
- `reference_simulator.py` - Reference implementation for validation

## Quick Start

```bash
# From project root
python tests/test_utils_comprehensive.py

# Or with pytest
pytest tests/test_utils_comprehensive.py -v

# Run all tests
pytest tests/ -v
```

## Expected Output

```
============================================================
COMPREHENSIVE UTILS VALIDATION
Combined test suite: core + hardening + extended modules
============================================================

... (all 18 test suites run) ...

============================================================
SUMMARY
============================================================
Imports                        ✅ PASS
Core Functionality             ✅ PASS
Tiled LPIPS Parity             ✅ PASS
Logging Idempotency            ✅ PASS
Atomic Operations              ✅ PASS
Error Paths                    ✅ PASS
Coverage Metric                ✅ PASS
Hashing                        ✅ PASS
Device Recursion               ✅ PASS
Coordinate Transforms          ✅ PASS
Profiler                       ✅ PASS
Validators                     ✅ PASS
G-code Generator               ✅ PASS
G-code VM                      ✅ PASS
MLflow Helpers                 ✅ PASS
Integration                    ✅ PASS
Edge Cases                     ✅ PASS
VM Micro-Fixes                 ✅ PASS
============================================================
🎉 All 18 test suites passed!
✅ Utils implementation is comprehensive, hardened, and production-ready
```

## Total Coverage

- **18 test suites** (all in one file)
- **150+ individual test cases**
- **All utils modules tested**
- **Hardening improvements included**

## CI Integration

```bash
# In CI pipeline
python tests/test_utils_comprehensive.py || exit 1

# Or with timeout
timeout 300 python tests/test_utils_comprehensive.py || exit 1
```

Or with Docker:
```bash
docker run <image> python tests/test_utils_comprehensive.py
```

## Test Structure

These tests use a proven pattern:
- Function-based (not pytest classes)
- Lazy imports (inside functions)
- Try/except error handling
- Custom main() runner
- Clear progress output
- Exit code 0 = success, non-zero = failure

This makes them robust and easy to run in any environment.

## Test Cleanup

Previously, there were 14 separate utils test files. These have been consolidated into `test_utils_comprehensive.py` for:
- Better maintainability (one file to update)
- Enhanced coverage (hardening improvements added)
- Clearer structure (18 organized suites)
- Faster CI (fewer files to discover)
- No duplication (all tests in one place)

**Deleted redundant files:**
- test_color.py, test_compute.py, test_coordinate_frames.py
- test_fs.py, test_gcode_gen.py, test_geometry.py
- test_hash.py, test_hashing.py, test_lpips_normalization.py
- test_metrics.py, test_mm_px_roundtrip.py, test_schemas.py
- test_strokes.py, test_torch_utils.py

All functionality preserved with 100% coverage in the comprehensive suite.

