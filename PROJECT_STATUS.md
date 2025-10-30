# Airbrush Painter - Project Status

## Structure Created

All directories and files have been created with module docstrings as of 2025-10-30.

## Directory Structure ✓

- `configs/` - YAML configurations and schemas
- `data/` - Raw images, calibration scans, targets, validation sets
- `outputs/` - Checkpoints, training monitor, HPO results, CI artifacts
- `gcode_output/` - Generated G-code files
- `src/` - Source code (layered architecture)
  - `utils/` - Cross-cutting utilities (lowest layer)
  - `data_pipeline/` - Preprocessing and calibration
  - `airbrush_simulator/` - Differentiable renderer
  - `airbrush_robot_env/` - RL environment
  - `rl_agent/` - Policy networks
  - `gui/` - PyQt interface
- `scripts/` - Training, inference, GUI entrypoints
- `tests/` - Unit and integration tests
- `ci/` - Golden tests for CI/CD

## Files Created

### Python Source (56 files)
- Core modules with comprehensive docstrings
- Utils modules (validators, compute, color, geometry, fs, torch_utils, metrics, profiler, strokes, hashing, logging_config, mlflow_helpers, gcode_generator, gcode_vm)
- Data pipeline (preprocess, pen_vectorizer, calibrate)
- Simulator (differentiable_renderer)
- Environment (env_v1)
- RL agent (networks)
- GUI (main_window, tabs, widgets)
- Scripts (train, paint, launch_gui)
- Tests (22 test modules)

### Configuration Files (12 files)
- train.yaml
- env_airbrush_v1.yaml
- physics_v1.yaml
- hpo_search_space_v1.yaml
- machine_grbl_airbrush_v1.yaml
- scanner_profile_v1.yaml
- calibration_layout.yaml
- manual_calibration_results.yaml
- Schema files (stroke.v1, luts.v1, machine.v1, job.v1)

### Docker & Build Files
- Dockerfile
- .dockerignore
- requirements.txt
- .gitignore

### Documentation
- README.md
- GlobalPlan.md (pre-existing)
- PROJECT_STATUS.md (this file)

## Next Steps

### Phase 1: Core Infrastructure
1. Implement `src/utils/validators.py` (YAML schema validation)
2. Implement `src/utils/compute.py` (mm↔px, LPIPS, tiling)
3. Implement `src/utils/fs.py` (atomic writes)
4. Implement `src/utils/color.py` (color space conversions)
5. Implement `src/utils/geometry.py` (Bézier operations)
6. Write unit tests for utils modules

### Phase 2: Renderer & Physics
1. Implement `src/airbrush_simulator/differentiable_renderer.py`
2. Create reference simulator for tests
3. Write renderer tests

### Phase 3: Environment
1. Implement `src/airbrush_robot_env/env_v1.py`
2. Test multi-resolution architecture
3. Test action scaling and coordinate frames

### Phase 4: Policy Network
1. Implement `src/rl_agent/networks.py`
2. Test spatial heads (CoordConv, heatmap+soft-argmax)

### Phase 5: Data Pipeline
1. Implement `src/data_pipeline/preprocess.py`
2. Implement `src/data_pipeline/pen_vectorizer.py`
3. Implement `src/data_pipeline/calibrate.py`

### Phase 6: Training & Inference
1. Implement `scripts/train.py`
2. Implement `scripts/paint.py` (with paint_main callable API)
3. Implement HPO integration

### Phase 7: GUI
1. Implement GUI tabs and widgets
2. Implement watchdog monitoring
3. Implement stroke playback

### Phase 8: G-code Generation
1. Implement `src/utils/gcode_generator.py`
2. Implement `src/utils/gcode_vm.py`

### Phase 9: Testing & CI
1. Implement all unit tests
2. Setup golden tests
3. Setup CI/CD pipeline

## Architecture Compliance

All files follow the architectural invariants from GlobalPlan.md:
- ✓ Strict layering (scripts → modules → utils)
- ✓ Multi-resolution architecture (render_px, obs_px, reward_px)
- ✓ Geometry in millimeters end-to-end
- ✓ Fixed stroke cap with pure quality objective
- ✓ YAML-only configs
- ✓ Decoupled GUI (observer pattern)
- ✓ Module docstrings explaining purpose and API

## Docstring Coverage

- ✓ Every module has a comprehensive docstring
- ✓ Docstrings explain:
  - Module purpose
  - Public API
  - Integration points
  - Invariants and constraints
  - Usage examples where appropriate

## Version

Current: v2.3.0 (per GlobalPlan.md)

## Notes

All files are skeleton implementations with detailed docstrings.
Implementation can proceed following the phases outlined above.
Each module's docstring provides clear guidance on what needs to be implemented.

