# Tests Directory

This directory contains a comprehensive test suite for all utils modules.

## Test Files

### `test_utils_comprehensive.py` (18 test suites combined)
**All-in-one comprehensive test suite** covering:

**Core functionality tests (1-11):**
- Imports
- Core functionality (LPIPS, color, geometry, strokes, mm↔px)
- Tiled LPIPS parity (even & ragged sizes)
- Logging idempotency (with console leakage check)
- Atomic filesystem operations
- Error paths (bounds projection, broadcasting, assert_finite)
- Coverage metrics
- Hashing (files & tensors)
- Device recursion
- Coordinate transforms (both origins tested)
- Profiler

**Extended module tests (12-18):**
- Validators (schema validation, bounds checking)
- G-code generator (coordinate transforms, linearization, soft limits)
- G-code VM (parsing, timing, violation detection)
- MLflow helpers (with graceful degradation)
- Integration (validators → G-code → VM pipeline)
- Edge cases (empty strokes, bounds validation, ID validation)
- VM micro-fixes (triangular profile, feed scaling, rapid timing)

## Quick Start

```bash
# From project root
python tests/test_utils_comprehensive.py

# Or with pytest (if installed)
pytest tests/test_utils_comprehensive.py -v
```

## Expected Output

```
============================================================
COMPREHENSIVE UTILS VALIDATION
Combined test suite: core + hardening + extended modules
============================================================

============================================================
1. Testing Imports
============================================================
✅ All utils modules imported successfully

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
- **Hardening improvements** (endpoint preservation, schema validation, origin contrast, console leakage, ragged tiles, broadcasting)

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

