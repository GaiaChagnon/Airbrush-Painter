# Improvement Plan

Comprehensive codebase audit performed 2026-03-20. Second-pass addendum
(Sections 16-17) added same date after verifying coverage gaps and
cross-referencing all directories. Every issue includes its location,
description, and rationale. **No code changes yet** -- each item requires
manual review before implementation.

---

## Table of Contents

1. [Critical / Bugs](#1-critical--bugs)
2. [Rule Violations: Exception Handling](#2-rule-violations-exception-handling)
3. [Rule Violations: print() Instead of Logging](#3-rule-violations-print-instead-of-logging)
4. [Rule Violations: Magic Numbers](#4-rule-violations-magic-numbers)
5. [Code Duplication](#5-code-duplication)
6. [Dead Code & Stubs](#6-dead-code--stubs)
7. [Type Hint Issues](#7-type-hint-issues)
8. [Resource Leaks & Safety](#8-resource-leaks--safety)
9. [Security](#9-security)
10. [Test Suite Gaps](#10-test-suite-gaps)
11. [Configuration & Build](#11-configuration--build)
12. [Documentation Gaps](#12-documentation-gaps)
13. [Simplification Opportunities](#13-simplification-opportunities)
14. [Potential Shared Utilities](#14-potential-shared-utilities)
15. [Feature Opportunities](#15-feature-opportunities)
16. [Missing Issues (Addendum)](#16-missing-issues-addendum-2026-03-20)
17. [File Deletion Candidates](#17-file-deletion-candidates)
18. [Risk Annotations](#18-risk-annotations)

---

## 1. Critical / Bugs

### 1.1 `validators.py` -- NameError in Pydantic validators
- **File:** `src/utils/validators.py:261, 268`
- **Issue:** Uses undefined variable `values` instead of `info.data` in
  Pydantic v2 field validators for `x_max` / `y_max`.
- **Impact:** Any validation trigger on these fields raises `NameError` at
  runtime, not the intended `ValueError`.
- **Fix:** Replace `values['x_min']` with `info.data['x_min']` (and same for
  `y_min`).

### 1.2 Missing LUT tensor files
- **File:** `configs/sim/luts/` (only `.gitkeep` exists)
- **Issue:** `physics_v1.yaml` and `luts.v1.yaml` reference
  `color_lut.pt`, `alpha_lut.pt`, `psf_lut.pt` which do not exist.
- **Impact:** Does **not** break `demo_alcohol_ink.py` -- `load_toy_luts()`
  generates LUTs in memory. However, any future code path that tries to load
  calibrated LUTs from disk will fail with no clear error message.
- **Fix:** Add an existence check with a clear error message in the loader, or
  document that these files are produced by the calibration pipeline.

---

## 2. Rule Violations: Exception Handling

CLAUDE.md rule 6: "Catch specific exception types. No bare `except` or lazy
catch-log-ignore."

| # | File | Line(s) | Current | Suggested |
|---|------|---------|---------|-----------|
| 2.1 | `src/utils/fs.py` | 404 | `except Exception: return False` | `except OSError` + log warning |
| 2.2 | `src/utils/mlflow_helpers.py` | 149, 253, 288 + 12 more | `except Exception` (15+ instances) | Catch specific MLflow / IO errors |
| 2.3 | `src/utils/gcode_generator.py` | 465-467 | `except Exception` | `except (ValueError, KeyError)` |
| 2.4 | `src/utils/gcode_vm.py` | 457, 497 | `except Exception` | Catch parse-specific errors |
| 2.5 | `robot_control/hardware/klipper_client.py` | 733 | `except (JSONDecodeError, Exception)` -- redundant | `except json.JSONDecodeError` only |
| 2.6 | `robot_control/hardware/klipper_client.py` | 739 | `except Exception  # noqa: BLE001` | `except (socket.error, RuntimeError)` |
| 2.7 | `robot_control/calibration/routines.py` | 1072 | `except Exception: pass` | `except (GCodeError, TimeoutError)` + log |
| 2.8 | `robot_control/scripts/cli/lineart_tracer.py` | 737, 744, 751 | `except Exception: pass` | `except (OSError, TimeoutError)` + log |
| 2.9 | `robot_control/scripts/cli/calibration.py` | 85, 150, 155, 473 | `except Exception: pass` | Catch specific + warn user |
| 2.10 | `robot_control/scripts/cli/connection.py` | 134, 140, 220, 226, 263, 278, 289 | `except Exception` (7 instances) | Catch socket/OS errors |
| 2.11 | `robot_control/scripts/cli/interactive_control.py` | 277-288 | `except Exception` for all gcode errors | Distinguish `TimeoutError` / `KlipperShutdown` / `OSError` |
| 2.12 | `digital_twin/gpu_simulator.py` | 422 | `except Exception` on torch.load | `except (FileNotFoundError, OSError, RuntimeError)` |
| 2.13 | `digital_twin/calibration_cli.py` | 1190, 1366, 1405 | `except Exception` | Catch `ConfigurationError`, `ValueError` |
| 2.14 | `src/data_pipeline/preprocess.py` | 151, 340, 479 | `except Exception` (3 instances) | `except (OSError, ValueError)` per context |
| 2.15 | `robot_control/scripts/cli/pump_controller.py` | 246, 1809, 1814 | `except Exception` | Catch specific pump/markup errors |

---

## 3. Rule Violations: print() Instead of Logging

CLAUDE.md: "No `print()` for runtime output."

| # | File | Approx Count | Notes |
|---|------|-------------|-------|
| 3.1 | `robot_control/calibration/routines.py` | ~80+ | Entire module uses `print()` for calibration output |
| 3.2 | `robot_control/hardware/pump_control.py` | ~15 | Hardware init/debug messages |
| 3.3 | `scripts/demo_alcohol_ink.py` | ~26 | All demo output |
| 3.4 | `scripts/preprocess_dataset.py` | ~7 | Error output via `print(..., file=sys.stderr)` |
| 3.5 | `scripts/verify_preprocessed.py` | ~3 | Error output |
| 3.6 | `src/utils/torch_utils.py` | 6 (lines 357-372) | `print_model_summary()` function |
| 3.7 | `src/utils/profiler.py` | 1 (line 63) | Timer fallback output |

**Note:** `robot_control/scripts/cli/` modules correctly use Rich `console.print()` --
those are compliant. The violations above are plain `print()` calls.

---

## 4. Rule Violations: Magic Numbers

CLAUDE.md rule 5: "No magic numbers/strings. Use named constants."

| # | File | Line(s) | Value | Suggested Constant |
|---|------|---------|-------|--------------------|
| 4.1 | `src/utils/gcode_vm.py` | 117 | `3000.0` | `DEFAULT_FEED_MM_MIN` |
| 4.2 | `src/utils/profiler.py` | 67 | `0x00AAFF` | `NVTX_DEFAULT_COLOR` |
| 4.3 | `robot_control/hardware/klipper_client.py` | 380, 560, 715 | `4096` (3 occurrences) | `_SOCKET_RECV_BUFFER_SIZE` |
| 4.4 | `robot_control/calibration/routines.py` | various | Timeout values `5.0`, `10.0`, `60.0` | `_QUERY_TIMEOUT_S`, `_MOVE_TIMEOUT_S`, `_HOME_TIMEOUT_S` |
| 4.5 | `robot_control/scripts/cli/interactive_control.py` | 284 | `msg[:77]` | `_STATUS_MSG_MAX_LEN` |
| 4.6 | `robot_control/scripts/cli/lineart_tracer.py` | 224, 235, 260 | `turdsize=10`, `blur_radius=10`, `flow_seed=42` | Named defaults at module level |
| 4.7 | `src/airbrush_simulator/cpu_reference.py` | 569-570 | `0.2126`, `0.7152`, `0.0722` (Rec.709 luma) | `_LUMA_R`, `_LUMA_G`, `_LUMA_B` |
| 4.8 | `digital_twin/calibration_cli.py` | 348, 532, 812, 1076, 1434 | Various thresholds and defaults | Module-level constants |
| 4.9 | `digital_twin/gpu_simulator.py` | 565, 763, 1610 | `0.25 * r_min`, `R * 0.5`, `R * 0.9` | Named scale factors |

---

## 5. Code Duplication

CLAUDE.md rule 2: "No code duplication."

### 5.1 `_ask_float()` / `_ask_int()` -- duplicated input helpers
- **Files:** `robot_control/scripts/cli/pump_controller.py:358-387` and
  `robot_control/scripts/cli/lineart_tracer.py:80-103`
- **Issue:** Nearly identical implementations. The lineart version has an
  unused `app` parameter.
- **Fix:** Extract to shared module (e.g., `robot_control/scripts/cli/input_helpers.py`).

### 5.2 `_load_jobs_config()` -- duplicated YAML loading
- **Files:** `robot_control/scripts/run_lineart_tracer.py:97-106`,
  `robot_control/scripts/run_tracer.py:87-95`
- **Issue:** Both reimplement YAML loading with CSafeLoader fallback.
- **Fix:** Replace with `src.utils.fs.load_yaml()`.

### 5.3 `image_mm_to_machine_mm()` -- duplicated coordinate transform
- **Files:** `src/utils/compute.py:205` and `src/utils/gcode_generator.py:54`
- **Issue:** Different signatures and logic for the same conceptual transform.
- **Fix:** Consolidate into single canonical version in `compute.py`.

### 5.4 `_lum()` -- duplicated luminance function
- **Files:** `digital_twin/calibration_cli.py:422` and
  `digital_twin/gpu_simulator.py:1670`
- **Issue:** Identical Rec.709 luminance calculation. Also appears as
  inline magic numbers in `cpu_reference.py:569-570`.
- **Fix:** Move to `src/utils/color.py` as a shared utility (see Section 14).

### 5.5 `TARGET_LONG_EDGE` / `TARGET_SHORT_EDGE` -- duplicated constants
- **Files:** `scripts/verify_preprocessed.py:18-19` (redefined) vs
  `src/data_pipeline/preprocess.py` (source of truth)
- **Fix:** Import from `src.data_pipeline.preprocess`.

---

## 6. Dead Code & Stubs

CLAUDE.md rule 3: "No dead code. If code isn't called, delete it."

### 6.1 `src/data_pipeline/calibrate.py` -- empty stub
- **Issue:** 35-line module docstring promising `generate_calibration_gcode()`
  and `build_luts_from_manual()` but zero lines of actual code.
- **Fix:** Either implement the functions or remove the file.

### 6.2 `tests/test_renderer.py` -- docstring-only test file
- **Issue:** Lists 6 planned tests in a docstring but contains zero test
  functions. The differentiable renderer is completely untested.
- **Fix:** Implement the listed tests or remove file if renderer is not ready.

### 6.3 Unused imports in `cpu_reference.py`
- **File:** `src/airbrush_simulator/cpu_reference.py:64`
- **Issue:** Imports `color as color_utils` and `metrics` but never uses them.
- **Fix:** Remove unused imports.

### 6.4 Unused import in `preprocess.py`
- **File:** `src/data_pipeline/preprocess.py:49`
- **Issue:** Imports `setup_logging` but never calls it.
- **Fix:** Remove import or add `setup_logging()` call.

### 6.5 Dead config section in `renderer_cpu.v1.yaml`
- **File:** `configs/sim/renderer_cpu.v1.yaml:56-60`
- **Issue:** `stamp_train:` section defined but mode is hardcoded to
  `"opencv_distance"` on line 10. Section is unreachable.
- **Fix:** Remove or document as future expansion.

### 6.6 `tests/test_parity_cpu_vs_gpu.py` -- all tests are xfail/skip
- **Issue:** 4 parity tests marked `@pytest.mark.xfail` with internal
  `pytest.skip()`. They will never actually run.
- **Fix:** Remove xfail or implement GPU renderer path.

---

## 7. Type Hint Issues

CLAUDE.md: "Python >= 3.10, full type hints."

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| 7.1 | `src/utils/gcode_vm.py` | 429 | Return type `Dict[str, any]` -- should be `Any` (capital A) |
| 7.2 | `src/utils/strokes.py` | 65 | `stroke_id: str = None` -- should be `Optional[str] = None` |
| 7.3 | `robot_control/calibration/measurement.py` | 227 | `prompt_fn: callable` -- should be `Callable[[float], bool]` |
| 7.4 | `robot_control/scripts/cli/app.py` | 42 | `type: ignore[attr-defined]` bypass |
| 7.5 | `robot_control/scripts/cli/pump_controller.py` | 132 | `type: ignore[assignment]` bypass |
| 7.6 | `src/airbrush_simulator/cpu_reference.py` | 760 | `strokes: list` -- should be `list[dict[str, Any]]` |
| 7.7 | `digital_twin/calibration_cli.py` | 1068, 1099 | `sim` parameter missing type hint |
| 7.8 | `digital_twin/calibration_cli.py` | 294-970 | `cal_dict: Dict` without generic params (5 functions) |
| 7.9 | `robot_control/scripts/cli/pump_controller.py` | 59 | `socket` in `TYPE_CHECKING` block but used at runtime |

---

## 8. Resource Leaks & Safety

CLAUDE.md rule 10: "No resource leaks. Use context managers."

### 8.1 Socket leak in `pump_control.py:wait_for_ready()`
- **File:** `robot_control/hardware/pump_control.py:236-245`
- **Issue:** Returns raw `socket.socket` with no context manager. Callers must
  remember to close.
- **Fix:** Document ownership or refactor to yield via context manager.

### 8.2 Global subprocess never terminated
- **File:** `robot_control/hardware/pump_control.py:36-195`
- **Issue:** `_spawned_klipper_proc` is a module-global `subprocess.Popen`
  with no cleanup. Process becomes orphan on exit.
- **Fix:** Register `atexit` handler to `terminate()` + `wait()`.

### 8.3 Socket without context manager in `connection.py`
- **File:** `robot_control/scripts/cli/connection.py:298-331`
- **Issue:** Socket created via `socket.socket()` and manually closed. If
  exception between create and close, socket leaks.
- **Fix:** Use `with socket.socket(...) as sock:`.

### 8.4 Thread safety race in `connection.py`
- **File:** `robot_control/scripts/cli/connection.py:252-266`
- **Issue:** `_poll_loop()` checks `self._client is not None` outside the
  lock. `disconnect()` could set it to `None` between check and use.
- **Fix:** Move the None-check inside the lock.

### 8.5 Global mutable state in `gcode_generator.py`
- **File:** `src/utils/gcode_generator.py:47`
- **Issue:** `_warned_macros: Set[str] = set()` is module-level mutable
  state. Thread-unsafe.
- **Fix:** Document thread-safety assumptions or use thread-local storage.

---

## 9. Security

### 9.1 `eval()` on user input
- **File:** `robot_control/scripts/cli/lineart_tracer.py:336`
- **Issue:** `eval(raw.strip())` on user-provided parameter values. Even with
  `# noqa: S307`, this is arbitrary code execution.
- **Fix:** Replace with `ast.literal_eval()`.

---

## 10. Test Suite Gaps

### 10.1 Empty / stub test files
- `tests/test_renderer.py` -- zero test functions (docstring only)
- `tests/test_parity_cpu_vs_gpu.py` -- all 4 tests xfail+skip (never run)

### 10.2 Skipped visual regression with no tracking
- **File:** `tests/test_cpu_visual_regression.py:302-303, 363-364`
- **Issue:** Two tests `@pytest.mark.skip("Needs adjustment for transparent
  alcohol ink model")` with no issue link or timeline.
- **Fix:** Create tracking issue or fix the tests.

### 10.3 Missing test coverage -- untested modules
| Module | Status |
|--------|--------|
| `src/airbrush_simulator/differentiable_renderer.py` | **Zero tests** |
| `robot_control/scripts/cli/*` | **Zero tests** for any CLI module |

### 10.4 Missing edge case tests
- `compute.py`: `mm_to_px()` / `px_to_mm()` with zero/negative values
- `fs.py`: `atomic_yaml_dump()` with permission errors, `load_yaml()` with
  malformed YAML
- `validators.py`: Reject out-of-range CMY values, mismatched calibration grids

### 10.5 Missing integration tests
- No end-to-end test for: image load -> preprocess -> render -> gcode
- No CLI workflow tests

### 10.6 Underused test markers
- `pytest.ini` already defines `slow`, `integration`, `golden`, `physics`,
  and `visual`. Only `@pytest.mark.smoke` is missing.
- No tests currently use `@pytest.mark.slow` or `@pytest.mark.integration`
  even though the markers are registered.

### 10.7 Inconsistent test path handling
- Some tests use `Path("relative/path")` (fragile, CWD-dependent)
- Others use `Path(__file__).parent.parent` (robust)
- **Fix:** Standardize all to `__file__`-relative or `tmp_path` fixture.

---

## 11. Configuration & Build

### 11.1 `requirements.txt` -- no upper version bounds
- **Issue:** `torch>=2.4.0` allows torch 3.0+ which may break. Same for all
  `>=` deps.
- **Fix:** Add upper bounds: `torch>=2.4.0,<3.0.0`.

### 11.2 Git dependencies not pinned to commit
- **File:** `requirements.txt:28-29`
- **Issue:** `hatched` and `vpype-flow-imager` point to latest `main`.
- **Fix:** Pin to specific commit hash.

### 11.3 Dockerfile layer order suboptimal
- **File:** `Dockerfile:45-49`
- **Issue:** nvdiffrast build happens after project COPY. Any source change
  invalidates the expensive build cache.
- **Fix:** Move nvdiffrast clone+install before `COPY . /workspace/`.

### 11.4 Manual calibration config has placeholder values
- **File:** `configs/sim/manual_calibration_results.yaml:28-29`
- **Issue:** Literal strings `"0.xx"` and `"x.x"` in YAML that would crash
  any parser expecting floats.
- **Fix:** Mark file clearly as template or validate before use.

### 11.5 No YAML schema version validation in code
- **Issue:** Each schema has `semver: "1.0.0"` but no runtime code checks
  version compatibility.
- **Fix:** Add version assertion in validators.

### 11.6 LUT axis/shape constraint not enforced
- **File:** `configs/schema/luts.v1.yaml`
- **Issue:** Shape `[5, 5]` and axis lists `[5 items]` are declared
  separately. Schema doesn't enforce `len(axes.z) == shape[0]`.
- **Fix:** Add Pydantic validator cross-checking axis length vs. shape.

### 11.7 No `machine.yaml` inline documentation
- **Issue:** Values like `pen_work_mm: 80.194` have no comments explaining
  their origin (empirically calibrated? from which routine?).
- **Fix:** Add inline YAML comments per cursor rule 04.

---

## 12. Documentation Gaps

### 12.1 Missing logging setup in `digital_twin/calibration_cli.py`
- **Line 39:** Creates `logger = logging.getLogger(__name__)` but never calls
  `setup_logging()`.
- **Fix:** Add `from src.utils.logging_config import setup_logging` and call
  it in `main()`.

### 12.2 Missing docstrings
| File | Function | Line |
|------|----------|------|
| `digital_twin/calibration_cli.py` | `_format_value()` | 127 |
| `robot_control/calibration/routines.py` | `_axis_stats()` return dict keys | 300 |
| `robot_control/hardware/pump_control.py` | Module-level global state docs | top |
| `scripts/verify_preprocessed.py` | `main()` | 85 |

---

## 13. Simplification Opportunities

### 13.1 Z-state lookup via if-chain -> dict
- **File:** `robot_control/configs/loader.py:469-481`
- **Issue:** `get_z_for_tool()` uses nested if/if instead of dict lookup.
- **Fix:** `z_map = {"pen": ..., "airbrush": ...}; return z_map[tool]`.

### 13.2 Inefficient polyline width computation
- **File:** `src/airbrush_simulator/cpu_reference.py:676-679`
- **Issue:** Computes `_width_mm()` at every vertex (could be hundreds) but
  only uses `min()`. Array is then discarded.
- **Fix:** Sample 4-10 key points instead of all vertices.

### 13.3 Unsupported gcode operation only logged, not raised
- **File:** `robot_control/gcode/generator.py:192`
- **Issue:** Unknown operation type logs a warning and silently continues,
  producing incomplete G-code.
- **Fix:** Raise `GCodeError(f"Unsupported operation: {type(op).__name__}")`.

### 13.4 `_lum()` / `_lum_tensor()` redundancy
- **File:** `digital_twin/gpu_simulator.py:1670, 1675`
- **Issue:** Two nearly identical functions. `_lum_tensor()` just calls
  `.item()` on each element.
- **Fix:** Single polymorphic utility or `_lum(rgb.tolist())`.

### 13.5 Encapsulation violation in calibration CLI helpers
- **File:** `digital_twin/calibration_cli.py:1071-1112`
- **Issue:** `_render_fill_band()` and `_sample_centre()` reach into
  `sim._cal`, `sim._dpi_x`, `sim._dpi_y`, `sim._W`, `sim._H`.
- **Fix:** Add public accessors on `GPUStampSimulator`.

---

## 14. Potential Shared Utilities

Patterns appearing in 3+ files that could be extracted to `src/utils/`.

### 14.1 Rec.709 luminance calculation
- **Appears in:** `cpu_reference.py` (inline), `calibration_cli.py` (`_lum`),
  `gpu_simulator.py` (`_lum`, `_lum_tensor`)
- **Proposed:** `src/utils/color.py::rec709_luminance(rgb)` accepting
  list/tuple/ndarray/tensor.

### 14.2 Validated numeric input prompt
- **Appears in:** `pump_controller.py` (`_ask_float`, `_ask_int`),
  `lineart_tracer.py` (same), `calibration.py` (inline int parsing)
- **Proposed:** `robot_control/scripts/cli/input_helpers.py` with
  `ask_float()`, `ask_int()`, `ask_int_bounded()`.

### 14.3 Low-level Klipper RPC (socket send/recv)
- **Appears in:** `run_tracer.py`, `run_lineart_tracer.py`, `test_motors.py`
- **Proposed:** `robot_control/hardware/klipper_rpc.py` with `raw_send()`,
  `raw_gcode()`, `wait_for_ready()`.

### 14.4 YAML loading with CSafeLoader fallback
- **Appears in:** `run_lineart_tracer.py`, `run_tracer.py`
- **Already exists:** `src/utils/fs.load_yaml()` -- just needs to be used.

---

## 15. Feature Opportunities

### 15.1 Klipper process cleanup on exit
- **File:** `robot_control/hardware/pump_control.py`
- **Proposed:** `atexit.register()` handler to terminate spawned Klipper
  subprocess.

### 15.2 Dockerfile HEALTHCHECK
- **File:** `Dockerfile`
- **Proposed:** Add `HEALTHCHECK CMD python -c "import torch; print('ok')"`
  for container orchestration.

### 15.3 Test markers for CI filtering
- `pytest.ini` already registers `slow`, `integration`, `golden`, `physics`,
  `visual` but no tests currently use them and `smoke` is missing.
- **Proposed:** Add `@pytest.mark.smoke` marker, then actually tag tests with
  `@pytest.mark.slow` / `@pytest.mark.integration` so CI can run subsets.

### 15.4 Interpolation grid size guard
- **File:** `digital_twin/gpu_simulator.py:178-179`
- **Proposed:** Assert `len(xp) > 1 and len(yp) > 1` in constructor to
  prevent single-point grids that break `searchsorted`.

---

## 16. Missing Issues (Addendum, 2026-03-20)

Items discovered during a second-pass audit that the original plan missed.

### 16.1 (HIGH) `strokes.py` -- YAML format mismatch in heatmap/count helpers
- **File:** `src/utils/strokes.py:242, 326`
- **Issue:** `fs.load_yaml()` returns a dict
  `{"schema": "stroke.v1", "strokes": [...]}` for v1 files. The code
  iterates the dict directly (`for stroke in strokes_list`), which iterates
  over **keys** (`"schema"`, `"strokes"`), not the actual stroke dicts.
  `stroke_bbox_mm()` then receives a string and crashes.
- **Impact:** Runtime crash on any v1 stroke YAML.
- **Fix:** Unwrap the dict:
  `data = fs.load_yaml(...); strokes_list = data.get('strokes', data) if isinstance(data, dict) else data`.

### 16.2 (HIGH) `tests/test_schemas.py` referenced but does not exist
- **File:** `.cursor/rules/03-config-and-schemas.mdc:10`
- **Issue:** Rule says *"update schema + validator + tests
  (`tests/test_schemas.py`)"*. That file does not exist anywhere in the repo.
- **Impact:** The entire schema validation testing contract in the rules is
  unfulfilled. Developers adding config fields have no test target.
- **Fix:** Create `tests/test_schemas.py` with round-trip validation tests for
  every schema, or update the rule to reference `tests/test_utils_comprehensive.py`
  if schema tests already live there.

### 16.3 (HIGH) No CI workflow files exist
- **Issue:** No `.github/workflows/` directory. `ci/golden_tests_cpu/README.md`
  references `.github/workflows/test.yml` which does not exist.
  `ci/golden_tests/compare.py` is a 46-line docstring with zero executable
  code. There is **no functional CI pipeline** at all.
- **Impact:** No automated testing on push/PR. Tests only run manually.
- **Fix:** Create `.github/workflows/test.yml` with at minimum
  `pytest tests/ -v` and `pytest robot_control/tests/ -v`.
- **Hardware safety:** Both test suites are hardware-independent and safe for
  CI. `robot_control/tests/test_klipper_client.py` uses `MockKlipperServer`
  (mock Unix socket), `test_calibration.py` tests pure math, and `tests/`
  has zero hardware imports. No test attempts to connect to the Octopus or
  any physical device.

### 16.4 (MEDIUM) `.cursor/rules` reference `io.*` instead of `fs.*`
- **Files:**
  - `.cursor/rules/00-global-architecture.mdc` (`src.utils.io.atomic_*`)
  - `.cursor/rules/01-utils-and-consistency.mdc` (`src.utils.io`)
  - `.cursor/rules/03-config-and-schemas.mdc` (`io.atomic_yaml_dump`, `io.load_yaml`)
  - `.cursor/rules/07-utilities-mandate.mdc` (`io.atomic_*`, `io.symlink_atomic`, `io.ensure_dir`)
- **Issue:** The actual module is `src.utils.fs`. AI assistants and developers
  following the rules get incorrect import paths.
- **Fix:** Replace all `io.*` references with `fs.*` in the four rule files.

### 16.5 (MEDIUM) `run_lineart_tracer.py` -- 198 `print()` calls
- **File:** `robot_control/scripts/run_lineart_tracer.py` (3128 lines)
- **Issue:** Largest `print()` violator in the entire repo (198 calls).
  Original plan (Section 3) lists 7 files but misses this one entirely.
- **Fix:** Replace with `logging.getLogger(__name__)` calls. The file is
  actively used by `cli/lineart_tracer.py` as a backend, so the fix is
  high-value.

### 16.6 (MEDIUM) `run_job.py` -- `logging.basicConfig()` instead of `setup_logging()`
- **File:** `robot_control/scripts/run_job.py:27-30`
- **Issue:** Uses `logging.basicConfig()` bypassing the project's structured
  logging (`logging_config.setup_logging()`). Also has 6 bare `print()` calls
  and uses `sys.path.insert(0, ...)` for imports.
- **Fix:** Replace with `setup_logging()`, replace `print()` with logger, use
  proper package imports.

### 16.7 (MEDIUM) `test_motors.py` -- 202 `print()` calls + 8 broad exceptions
- **File:** `robot_control/scripts/test_motors.py` (1230+ lines)
- **Issue:** 202 `print()` calls and 8 `except Exception` blocks. Not
  mentioned anywhere in the original plan.
- **Fix:** Replace `print()` with logging, narrow exception handling.

### 16.8 (MEDIUM) `session_log.py` -- file handle opened without context manager
- **File:** `robot_control/scripts/cli/session_log.py:65-66`
- **Issue:** `self._fh = open(self._path, "w", ...)` followed by
  `self._write_header()`. If `_write_header()` raises, the file handle
  leaks. The `# noqa: SIM115` suppresses the linter.
- **Fix:** Wrap in try/except that closes `_fh` on failure, or restructure
  to use a context manager pattern.

### 16.9 (MEDIUM) `patterns.py` -- silent exception fallback to A4
- **File:** `robot_control/calibration/patterns.py:36-43`
- **Issue:** `except Exception:` swallows **any** error loading config
  (including `ImportError`, `AttributeError`, real config bugs) and silently
  falls back to A4 dimensions (210x297). Could mask real configuration
  problems during calibration.
- **Fix:** Catch `(FileNotFoundError, KeyError)` specifically; log a warning
  when falling back.

### 16.10 (MEDIUM) `sys.path.insert()` hacks in standalone scripts
- **Files:** `robot_control/scripts/run_job.py:19`,
  `robot_control/scripts/run_lineart_tracer.py` (similar pattern)
- **Issue:** `sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))`
  is fragile and non-standard. Breaks if file is moved or symlinked.
- **Fix:** Use `-m` module invocation (`python -m robot_control.scripts.run_job`)
  or proper package installation.

### 16.11 (MEDIUM) `job_executor.py` -- broad exception not listed
- **File:** `robot_control/hardware/job_executor.py:125, 299`
- **Issue:** `except Exception as exc:  # noqa: BLE001` on progress and cancel
  callbacks. Same pattern as other files but missing from Section 2.
- **Fix:** Catch `(TypeError, RuntimeError)` or document why broad catch is
  intentional for user-supplied callbacks.

### 16.12 (MEDIUM) `test_preprocess.py` -- function-level duplication
- **File:** `tests/test_preprocess.py:305-369`
- **Issue:** `verify_preprocessed_directory()` reimplements the function from
  `scripts/verify_preprocessed.py` with minor differences. Plan item 5.6
  catches constant duplication but misses this larger function copy.
- **Fix:** Import `verify_preprocessed_directory` from the canonical source
  instead of re-defining it.

### 16.13 (LOW) `pump_control.py:516` -- additional broad exception
- **File:** `robot_control/hardware/pump_control.py:516`
- **Issue:** `stdin_has_data()` uses `except Exception: return False`. Plan
  Section 2 only lists `pump_controller.py` lines 246, 1809, 1814.
- **Fix:** `except (OSError, ValueError): return False`.

### 16.14 (LOW) `validators.py` -- deprecated Pydantic v2 `min_items`
- **File:** `src/utils/validators.py:413` (`JobConfig.passes` field, single occurrence)
- **Issue:** `min_items=1` is deprecated in Pydantic v2; should be
  `min_length=1`. Single occurrence; low urgency.
- **Fix:** Replace `min_items=1` with `min_length=1` on line 413.

### 16.15 (LOW) `run_job.py` -- `callable` lowercase type hint
- **File:** `robot_control/scripts/run_job.py:33`
- **Issue:** `PATTERN_MAP: dict[str, callable]` -- `callable` (lowercase)
  is the builtin, not a type hint. Should be
  `dict[str, Callable[..., list[Operation]]]`.
- **Fix:** Import `Callable` from `collections.abc` and use it.

### 16.16 (LOW) `gcode_vm.py` -- `callable` lowercase type hint
- **File:** `src/utils/gcode_vm.py` (PATTERN_MAP or similar)
- **Issue:** Same lowercase `callable` issue as 16.15.
- **Fix:** Use `Callable[..., Any]` from `collections.abc`.

---

## 17. File Deletion Candidates

Files that contain zero executable code (docstring-only stubs), are fully
orphaned, or serve no current purpose. Each entry explains **why** the file
can be removed.

### 17.1 `src/data_pipeline/calibrate.py` -- DELETE
- **Content:** 35-line module docstring promising `generate_calibration_gcode()`
  and `build_luts_from_manual()` but **zero lines of actual code**.
- **Imported by:** Nothing. Not called anywhere.
- **Why delete:** Dead stub violates CLAUDE.md rule 3 ("No dead code"). The
  calibration workflow it describes is handled by `digital_twin/calibration_cli.py`.
- **Post-deletion:** Update `src/data_pipeline/__init__.py` to remove the
  `calibrate` reference from its module docstring.

### 17.2 `tests/test_renderer.py` -- DELETE
- **Content:** 26-line docstring listing 6 planned test names. Zero
  `def test_*` functions. pytest collects nothing.
- **Imported by:** Nothing.
- **Why delete:** Dead test file. The differentiable renderer it would test
  has its own test stubs in `tests/test_parity_cpu_vs_gpu.py`. Keeping an
  empty file gives a false sense of coverage.
- **Post-deletion:** Update `tests/README.md` to remove references.

### 17.3 `tests/reference_simulator.py` -- DELETE
- **Content:** 27-line docstring describing a "Reference simulator for unit
  tests" with a `render_stroke()` API. **Zero implementation.**
- **Imported by:** Nothing. Its docstring claims use by `test_renderer.py`
  and `test_mm_px_roundtrip.py`, neither of which exist or import it.
- **Why delete:** Dead stub. The actual CPU reference renderer is
  `src/airbrush_simulator/cpu_reference.py`.
- **Post-deletion:** Update `tests/README.md` to remove references.

### 17.4 `ci/golden_tests/compare.py` -- DELETE (then remove `ci/golden_tests/`)
- **Content:** 46-line docstring describing a golden test CLI with `--golden`
  and `--all` flags, expected YAML format, exit codes. **Zero executable code.**
- **Imported by:** Nothing.
- **Why delete:** Spec document masquerading as a script. The directories it
  references (`ci/golden_tests/images/`, `ci/golden_tests/expected/`) do not
  exist. Functional golden tests live in `tests/test_cpu_golden.py` with
  thresholds in `ci/golden_tests_cpu/thresholds.yaml`.
- **Post-deletion:** Remove the now-empty `ci/golden_tests/` directory.
  `.dockerignore` entries referencing it become harmless no-ops.

### 17.5 `documentation/FINAL_UTILS_REPORT.md` -- DELETE (optional)
- **Content:** Historical report on utils implementation referencing
  `validate_utils_comprehensive.py` and `validate_utils_extended.py`.
- **Why delete:** Both referenced scripts were deleted (merged into
  `tests/test_utils_comprehensive.py` per `COMBINED_TEST_SUITE_SUMMARY.md`).
  The report is a historical snapshot that does not reflect current code.
- **Confidence:** MEDIUM -- safe to delete but low priority.

### 17.6 `documentation/COMBINED_TEST_SUITE_SUMMARY.md` -- DELETE (optional)
- **Content:** Documents the merge of `test_utils_extended.py` into
  `test_utils_comprehensive.py`. Both the source and target of the merge
  are done; the document is a completed one-time changelog entry.
- **Why delete:** One-time migration record with no ongoing value.
- **Confidence:** MEDIUM -- informational only.

### Files reviewed but NOT recommended for deletion

| File / Directory | Reason to keep |
|---|---|
| `robot_control/scripts/run_lineart_tracer.py` | Actively imported by `cli/lineart_tracer.py` as backend |
| `robot_control/scripts/run_tracer.py` | Executes pre-generated `pen_vectors.yaml`; no CLI equivalent |
| `robot_control/scripts/run_job.py` | Runs calibration/drawing patterns via `JobExecutor`; no CLI equivalent |
| `robot_control/scripts/test_motors.py` | Full hardware bring-up (motor spin, endstops, limits); CLI only partially overlaps |
| `robot_control/scripts/robot_cli.py` | Main entry point that loads CLI modules |
| `data/calibration_scans/` (.gitkeep) | Placeholder for future calibration scan data |
| `configs/sim/luts/` (.gitkeep) | Placeholder for generated LUT tensors |
| `ci/visual_regression_cpu/` | Used by `test_cpu_visual_regression.py` for output |
| `ci/golden_tests_cpu/` | Contains `thresholds.yaml` used by `test_cpu_golden.py` |
| `tests/test_parity_cpu_vs_gpu.py` | Has real test structure (xfail); fix or rewrite rather than delete |
| `documentation/GlobalPlan.md` | Main design document; needs reference updates, not deletion |
| `documentation/ALCOHOL_INK_RENDERER_FINAL.md` | Documents current CPU renderer physics model |
| `documentation/CPU_RENDERER_OPENCV_MODEL.md` | Documents current renderer implementation |

---

## 18. Risk Annotations

Items from this plan that can change operator-visible behavior if applied
blindly. Each should be preceded by tests covering the current behavior.

| Item(s) | Risk | Mitigation |
|---------|------|------------|
| Section 2 (mass `except Exception` narrowing) | Exposes previously swallowed runtime failures, especially in hardware/CLI paths. | Narrow one file at a time; add integration test for each path before narrowing. |
| Section 3 + 16.5-16.7 (mass `print()` to logging) | Changes operator UX in interactive scripts; log output may not appear on console without handler config. | Ensure `setup_logging()` includes a console handler; test scripts interactively after conversion. |
| 9.1 (`eval` to `ast.literal_eval`) | Safer, but removes expression-style inputs (e.g., `10 * 2.54`) if operators relied on them. | Document the change; add a `float()` fallback for simple arithmetic expressions if needed. |
| 5.3 (merge `image_mm_to_machine_mm()`) | Must preserve canvas-offset and 180-degree rotation semantics or G-code coordinates break. | Write parity tests comparing both implementations' outputs before consolidating. |
| 13.3 (raise on unknown G-code ops) | Stricter correctness, but may break forward-compatibility if new operation types are added. | Gate behind a `strict=True` flag defaulting to current behavior; add to CI. |

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Critical / Bugs | 3 (validators, LUTs, strokes.py) |
| Exception handling violations | 18+ locations |
| print() violations | ~530+ calls across 10 files |
| Magic numbers | ~20 distinct instances |
| Code duplication | 6 clusters |
| Dead code / stubs | 6 items |
| Type hint issues | 11 items |
| Resource leaks / safety | 5 items |
| Security | 1 (eval) |
| Test gaps | 8 categories |
| Config / build | 7 items |
| Documentation gaps | 4 categories |
| Simplification opportunities | 5 items |
| Shared utility candidates | 4 patterns |
| Feature opportunities | 4 proposals |
| CI pipeline | 1 (entirely missing) |
| File deletion candidates | 6 files (4 high confidence, 2 optional) |
