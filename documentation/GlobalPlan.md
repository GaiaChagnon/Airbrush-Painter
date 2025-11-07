# **Hybrid AI Painter – Engineering Plan**

> ⚠️ **NOTE:** This is a **design and architecture document** that describes the planned system.  
> For **current implementation status**, see `PROJECT_STATUS.md`.  
> **As of 2025-11-02:** Foundation layer complete (utils, CPU renderer, pen tracer).  
> RL pipeline (GPU renderer, environment, networks, training, GUI) is in stub phase.

---

## **Key Design Principles**

- **Multi-resolution architecture** (physics/reward at high fidelity, observation downsampled for speed).
- **Fixed stroke cap** (you enforce it; no stroke penalties).
- **Pure quality objective** (no time/ink costs).
- **YAML only** (no JSON).
- **Containerized**.
- **MLflow‑integrated HPO with validation set** (Optuna + 10 fixed test images for objective evaluation).
- **Decoupled GUI for training visualization** (observer pattern via watchdog; on-demand stroke playback).
- **Adversarial target tests** to prevent reward hacking.
- **Scanner input is PNG/JPEG from a printer‑scanner** (consistent lighting).
- **DGX Spark optimized** (BF16, channels-last, full-frame compute).

---

## **0) Executive overview (what this system ships)**

**Inputs**

- A **high-quality A4 image** (sRGB PNG/JPEG, user-defined resolution).
- **Calibrated LUTs** (CMY→RGB color; PSF & alpha vs. Z & speed).
- A **machine profile** (work area, feeds, macros).

**Outputs**

- **<job>_cmy.gcode** (airbrush painting).
- **<job>_pen.gcode** (black pen layer).
- **<job>_manifest.yaml** (complete job description: inputs, limits, artifacts).

**Core architecture**

- **Multi-resolution pipeline:** physics grid (render_px), policy input (obs_px), reward scoring (reward_px)—all independent.
- **Geometry in mm:** all coordinates are millimeters; px conversions happen at boundaries.
- Fixed **stroke cap** (e.g., 1500).
- Reward = **LPIPS improvement only** (no time/ink penalties).
- Open‑loop (no camera during painting).

---

## **1) Repository & directories**

```
airbrush_painter/
airbrush_painter/
├─ configs/
│  ├─ train.yaml                       
│  ├─ calibration_layout.yaml 
│  ├─ env_airbrush_v1.yaml
│  ├─ hpo_search_space_v1.yaml 
│  ├─ machine_grbl_airbrush_v1.yaml
│  ├─ scanner_profile_v1.yaml
│  │
│  ├─ sim/
│  │  ├─ physics_v1.yaml
│  │  ├─ manual_calibration_results.yaml
│  │  └─ luts/
│  │     ├─ color_lut.pt
│  │     ├─ psf_lut.pt
│  │     └─ alpha_lut.pt
│  │
│  └─ schema/
│     ├─ stroke.v1.yaml
│     ├─ luts.v1.yaml
│     ├─ machine.v1.yaml
│     └─ job.v1.yaml
│
├─ data/
│  ├─ raw_images/
│  ├─ calibration_scans/
│  ├─ validation_images/
│  │  ├─ cmy_only/
│  │  └─ pen_layers/
│  └─ target_images/
│     ├─ cmy_only/{easy,medium,hard}/
│     └─ pen_layers/{easy,medium,hard}/
│
├─ outputs/
│  ├─ checkpoints/
│  ├─ training_monitor/
│  │  ├─ epoch_{N}/
│  │  │  ├─ target.png
│  │  │  ├─ canvas.png
│  │  │  ├─ strokes.yaml
│  │  │  └─ metadata.yaml
│  │  └─ latest/
│  ├─ final_paintings/
│  ├─ hpo_eval/
│  │  └─ trial_{N}/
│  ├─ studies/hpo.db
│  └─ ci/
│
├─ gcode_output/
│  └─ <job_name>/{_cmy.gcode,_pen.gcode,_manifest.yaml}
│
├─ src/
│  ├─ __init__.py
│  ├─ differentiable_renderer.py 
│  ├─ env_v1.py 
│  ├─ networks.py   
│  │
│  ├─ data_pipeline/ 
│  │  ├─ __init__.py
│  │  ├─ preprocess.py
│  │  ├─ pen_vectorizer.py
│  │  └─ calibrate.py
│  ├─ gui/   
│  │  ├─ __init__.py
│  │  ├─ main_window.py
│  │  ├─ tabs/
│  │  └─ widgets/
│  └─ utils/
│     ├─ __init__.py
│     ├─ validators.py
│     ├─ compute.py
│     ├─ gcode_generator.py
│     ├─ gcode_vm.py
│     ├─ color.py
│     ├─ geometry.py
│     ├─ io.py
│     ├─ torch_utils.py
│     ├─ metrics.py
│     ├─ profiler.py
│     ├─ strokes.py
│     ├─ hash.py
│     ├─ logging_config.py
│     └─ mlflow_helpers.py
│
├─ scripts/
│  ├─ train.py
│  ├─ paint.py
│  └─ launch_gui.py
│
├─ ci/golden_tests/
│  ├─ images/
│  ├─ expected/
│  └─ compare.py
│
├─ tests/
│  ├─ reference_simulator.py
│  ├─ test_schemas.py
│  ├─ test_renderer.py
│  ├─ test_env_v1.py
│  ├─ test_networks.py
│  ├─ test_gcode_gen.py
│  ├─ test_reward_hacks.py
│  ├─ test_compute.py
│  ├─ test_env_resolutions.py
│  ├─ test_mm_px_roundtrip.py
│  ├─ test_coordinate_frames.py
│  ├─ test_action_scaling.py
│  ├─ test_lpips_normalization.py
│  ├─ test_paint_main.py
│  ├─ test_gui_monitoring.py
│  ├─ test_color.py
│  ├─ test_geometry.py
│  ├─ test_io.py
│  ├─ test_torch_utils.py
│  ├─ test_metrics.py
│  ├─ test_strokes.py
│  └─ test_hash.py
│
├─ Dockerfile
└─ .dockerignore

```

---

## **2) YAML interfaces (single source of truth)**

### **2.1 Stroke schema –**

### **configs/schema/stroke.v1.yaml**

```
schema: stroke.v1
units: {xy: mm, z: mm, speed: mm/s}
bounds:
  x: [0, 210.0]
  y: [0, 297.0]
  z: [0.0, 30.0]
  speed: [1.0, 300.0]
fields:
  id: str
  bezier: {p1: [float, float], p2: [float, float], p3: [float, float], p4: [float, float]}
  z_profile: {z0: float, z1: float}
  speed_profile: {v0: float, v1: float}
  color_cmy: {c: [0.0, 1.0], m: [0.0, 1.0], y: [0.0, 1.0]}
metadata:
  sim_version: "1.0.0"
validators:
  - "all control points within bounds.xy"
  - "z0,z1 within bounds.z"
  - "v0,v1 within bounds.speed"
  - "0.0 <= c,m,y <= 1.0"

```

### **2.2 LUT metadata –**

### **configs/schema/luts.v1.yaml**

```
schema: luts.v1
color_lut:
  grid: {c: NC, m: NM, y: NY}
  domain: [0.0, 1.0]
  values_path: "configs/sim/luts/color_lut.pt"   # tensor [NC x NM x NY x 3], linear RGB
alpha_lut:
  axes: {z: [..], v: [..]}
  values_path: "configs/sim/luts/alpha_lut.pt"   # tensor [Nz x Nv]
psf_lut:
  axes: {z: [..], v: [..]}
  values_path: "configs/sim/luts/psf_lut.pt"     # tensor [Nz x Nv x K x K]
metadata:
  machine_profile: "grbl_airbrush.v1"
  date: "2025-10-27"
  semver: "1.0.0"

```

### **2.3 Machine profile –**

### **configs/machine/grbl_airbrush.v1.yaml**

```
schema: machine.v1
work_area_mm: {x: 210.0, y: 297.0, z: 30.0}
gcode_flavor: "grbl_1.1f"
units: "mm"
feed_units: "mm/min"
feeds:
  max_xy_mm_s: 300.0
  max_z_mm_s: 20.0
macros:
  include_dir: "gcode/macros/"
  purge: "PURGE.gcode"
  pen_up: "PEN_UP.gcode"
  pen_down: "PEN_DOWN.gcode"
safety:
  soft_limits: true
  purge_zone_mm: {x: [200, 209], y: [290, 296], z: [10, 15]}

```

### **2.4 Job manifest –**

### **configs/schema/job.v1.yaml**

```
schema: job.v1
machine_profile: "grbl_airbrush.v1"
inputs:
  target_image_path: "data/target_images/cmy_only/hard/sample_hd.png"
  pen_layer_path: "data/target_images/pen_layers/hard/sample_hd.yaml"
limits:
  max_strokes: 1500    # non-negotiable
  bbox_mm: {x: [0,210], y: [0,297], z: [0,30]}
passes:
  - name: "cmy"
    strokes_path: "outputs/strokes/sample_hd_strokes.yaml"   # list[stroke.v1]
  - name: "pen"
    vectors_path: "outputs/pen/sample_hd_pen.yaml"
artifacts:
  cmy_gcode_out: "gcode_output/sample/painting_cmy.gcode"
  pen_gcode_out: "gcode_output/sample/painting_pen.gcode"
  manifest_out:  "gcode_output/sample/painting_manifest.yaml"

```

**Validation**

src/utils/validators.py provides:

- load_job_config(path) -> JobV1
- load_machine_profile(path) -> MachineV1
- validate_strokes(path) -> None
- validate_luts(path) -> None

Fail fast with readable errors (line numbers, offending key).

---

## **3) Data pipeline (Multi-Resolution)**

This pipeline handles preparing target images for the AI and calibrating the simulator physics based on real-world measurements.

### **3.1 Preprocess (Standardize, Separate Layers, Vectorize)**

**Purpose:** Convert user-provided images (`data/raw_images/`) into the two components needed for training and painting: a color-only target for the AI (`cmy_only/`) and vector data for the pen (`pen_layers/`).

**Module:** `src/data_pipeline/preprocess.py`

**Public API:**

Python

`def preprocess_image(
raw_image_path: str,
output_dir_cmy: str,
output_dir_pen: str,
target_render_px: Tuple[int, int], # e.g., (908, 1280) from env config
pen_threshold_lab_l: int = 10
) -> Tuple[str, str]:
"""
Loads raw image, standardizes size, separates CMY and Pen layers,
vectorizes pen layer, and saves outputs.

```
Returns:
    Tuple[str, str]: Paths to the saved cmy_only.png and pen_layer.yaml
"""
# ... implementation ...`

```

**Workflow:**

1. **Load Raw Image:** Reads the input image (PNG/JPEG) from `data/raw_images/`.
2. **Standardize Resolution & Aspect Ratio:** Resizes and pads/crops the image to match the `target_render_px` (e.g., 908×1280) defined in `configs/env/airbrush_v1.yaml`, maintaining the A4 aspect ratio (portrait). This ensures all target images used in training have the same dimensions as the simulator's `render_px` grid. Converts to a standard format (e.g., 8-bit RGB PNG).
3. **Identify Pen Mask (K Layer):**
    - Converts the standardized image to the LAB color space.
    - Creates a binary mask (`pen_mask`) by thresholding the L (lightness) channel (e.g., `L < pen_threshold_lab_l`).
    - Applies small morphological operations (e.g., closing, opening) to clean up noise in the mask.
4. **Inpaint CMY Target:** Uses `cv2.inpaint()` with the `pen_mask` (and `cv2.INPAINT_TELEA` or `cv2.INPAINT_NS`) on the standardized RGB image. This fills in the black line areas with plausible surrounding colors, creating the **color-only target** for the AI.
5. **Save CMY Target:** Saves the inpainted image to the appropriate `data/target_images/cmy_only/{easy|medium|hard}/` directory as a PNG file.
6. **Vectorize Pen Mask:** Calls the `pen_vectorizer` module (see below) on the `pen_mask`.
7. **Save Pen Vectors:** Saves the resulting vector data (list of paths) as a YAML file in `data/target_images/pen_layers/{easy|medium|hard}/`.
8. Returns the paths to the two saved files.

**Module:** `src/data_pipeline/pen_vectorizer.py`

**Public API:**

Python

`def vectorize_mask_to_yaml(     pen_mask: np.ndarray,     output_yaml_path: str,     simplification_tolerance: float = 1.5 # Pixels ) -> None:     """     Vectorizes a binary pen mask using Potrace and saves simplified     centerline paths to a YAML file.     """     # ... implementation ...`

**Workflow:**

1. **Vectorize:** Uses `pypotrace` (or calls the `potrace` command-line tool) on the binary `pen_mask` to extract contour outlines.
2. **Simplify:** Applies the Ramer-Douglas-Peucker algorithm (`shapely.simplify` or equivalent) to reduce the number of points in each path while preserving the shape within the given `simplification_tolerance`.
3. **Format:** Organizes the simplified paths into a list of lists of `[x, y]` coordinates (in pixels, relative to `render_px`).
4. **Save:** Writes the path data to the specified YAML file, adhering to the `pen_vectors.v1` schema (implicitly defined by usage).

---

### **3.2 Calibration (Manual Measurement Workflow)**

**Purpose:** Generate the physical parameters (LUTs) that define the simulator's behavior based on manually measured real-world prints.

**Module:** `src/data_pipeline/calibrate.py`

**Public APIs:**

Python

`def generate_calibration_gcode(
machine_cfg_path: str,
calib_layout_cfg_path: str,
out_dir: str
) -> None:
"""Generates G-code files for printing calibration patterns with labels."""
# ... implementation ...

def build_luts_from_manual(
manual_results_yaml_path: str,
out_dir: str
) -> Dict[str, str]:
"""Reads manual measurements from YAML and generates .pt LUT files."""
# ... implementation ...
# Returns dictionary mapping LUT name to saved file path`

**Configuration:**

- `configs/calibration_layout.yaml`: Defines the grid layouts, step values (CMY, Z, V), patch sizes, origins, and **label prefixes** for the printed patterns. Includes fiducial marker positions.
- `configs/sim/manual_calibration_results.yaml`: The file **you edit** to input your measurements, organized by the labels printed on the sheet (e.g., "C123", "P00"). Contains fields for `rgb_linear_measured`, `alpha_measured`, `width_mm`, etc.
- `configs/machine/grbl_airbrush.v1.yaml`: Provides machine limits and G-code details.
- `configs/scanner/scanner_profile.v1.yaml`: Records scanner DPI for mm conversions.

**Workflow:**

1. **Generate G-Code (Automated):** Run `generate_calibration_gcode`. This reads the `calibration_layout.yaml` and `machine.v1.yaml` to produce `.gcode` files (e.g., `calibration_color_grid.gcode`, `calibration_psf_alpha.gcode`, `calibration_layering.gcode`, `calibration_fiducials.gcode`) in `gcode_output/calibration/`. These files include commands to draw fiducials and **pen-written labels** (like "C123", "P00") next to each test patch/dot.
2. **Print & Scan (Manual):** Execute the generated G-code files on the robot. Scan the resulting printouts and save them to `data/calibration_scans/`. Record the scanner DPI in `manual_calibration_results.yaml`. 📝
3. **Measure (Manual):**
    - Open the scanned images in an image editor configured for **linear RGB** display and **millimeter** measurements (using the recorded DPI).
    - For each labeled patch/dot on the scans:
        - **Color Patches ("Cxyz"):** Use the eyedropper (average, linear RGB) to measure `rgb_linear_measured`.
        - **PSF/Alpha Dots ("Pxy"):** Measure `alpha_measured` (relative luminance vs. paper white) and `width_mm` (diameter/characteristic size). Optionally note the `shape_description`.
        - **Layering Patches ("Lxy"):** Measure `base_rgb_linear_measured`, `top_rgb_linear_measured` (outside overlap), and `overlap_rgb_linear_measured`.
    - Enter all measurements into `configs/sim/manual_calibration_results.yaml` next to the corresponding label. 꼼꼼하게! (meticulously!)
4. **Build LUTs (Automated):** Run `build_luts_from_manual`. This script:
    - Loads your `manual_calibration_results.yaml`.
    - Uses **interpolation** (`scipy.interpolate`) to build functions from your sparse measurements:
        - `f(C, M, Y) -> (R, G, B)`
        - `g(Z, V) -> Alpha`
        - `h(Z, V) -> PSF` (e.g., generates a Gaussian kernel based on `width_mm`)
    - Evaluates these functions on dense grids and saves the resulting tensors as `color_lut.pt`, `alpha_lut.pt`, and `psf_lut.pt` (FP32) in `configs/sim/luts/`.
    - **Validates Layering:** Uses the generated LUTs (`f`, `g`) and the layering measurements to predict the overlap color and compares it (Delta E) to your `overlap_rgb_linear_measured`, printing the results. ✅

**Output:** Calibrated `.pt` LUT files ready to be loaded by the simulator (`differentiable_renderer.py`).

---

## **4) Multi-resolution compute plan (keystone)**

### **4.1 Three independent resolutions**

- **render_px:** Physics simulation grid (training default: 908×1280, portrait).
- **obs_px:** Policy network input (downsampled, e.g., 454×640 for speed).
- **reward_px:** LPIPS evaluation grid (can match render_px or be higher for final eval).
- **Geometry truth:** All stroke parameters in **mm**; conversions via mm_to_px/px_to_mm helpers.

### **4.2 Tiled LPIPS (flexible)**

src/utils/compute.py::TiledLPIPS

- Configurable tile_size (0 = full-frame on DGX Spark).
- For smaller GPUs: tiles of 1024–2048 with overlap; cosine (Hann) windows to avoid seam artifacts.
- Runs LPIPS in FP32 (inputs normalized to [-1,1]), with optional tiling to reduce memory.
- Unit tests compare against full‑frame (where feasible) for small images.

### **4.3 BF16 (mixed precision) & DGX Spark defaults**

- Wrap Strategist forward and renderer inside autocast(bf16).
- **Disable grad checkpointing** (DGX has headroom; use compute for speed).
- **Channels-last memory format** for CNNs (free perf).
- LUT tensors kept in **FP32** for stability.

### **4.4 Memory expectations & knobs**

- Observation: **9×Ho×Wo** tensor at obs_px (small for fast policy).
- Canvas/target: **3×Hr×Wr** at render_px (physics fidelity).
- Reward: computed at reward_px (quality metric).
- Strategist batch: **1–4 envs**; fewer envs, longer episodes.
- Technician: **10–20 steps**; **early‑stop** if ΔLPIPS < ε for K steps.

---

## **5) Environment (multi-resolution, fixed stroke cap)**

src/airbrush_robot_env/env_v1.py

- **Canvas/target storage:** kept at **render_px** (physics fidelity).
- **Obs:** downsample to obs_px → (Target, Canvas, |Target−Canvas|) → **(9, Ho, Wo)** FP32 (returned as np.float32; AMP happens inside model only).
- **Reward:** resample {canvas, target} to reward_px, compute LPIPS(old) − LPIPS(new) (dense improvement).
- **Terminate:** step_count >= stroke_cap (from env YAML).
- **Renderer:** uses render_stroke_tiled at render_px; only tiles overlapped by the stroke are recomputed.
- **Action validation:** out‑of‑bounds actions (in mm) are **projected** to valid ranges before rendering (hard constraint).
- **Geometry:** All actions in **mm**; mm↔px conversion happens at render boundary using work_area_mm + render_px.

YAML (configs/env/airbrush_v1.yaml)

```
units: "mm"
work_area_mm: {x: 210.0, y: 297.0}
resolutions:
  render_px: {w: 908, h: 1280}
  obs_px:    {w: 454, h: 640}
  reward_px: {w: 908, h: 1280}
stroke_cap: 1500
renderer:
  tile_size: 0
  tile_overlap: 0
  technician_steps: 12
  early_stop_delta: 1.0e-4
  project_params_each_step: true

```

---

## **6) Differentiable renderer (robust & tiled, mm-space)**

src/airbrush_simulator/differentiable_renderer.py

- **Units:** All stroke parameters in **mm**; converts to px internally via work_area_mm + render_px.
- Loads **LUT tensors** (color_lut.pt, alpha_lut.pt, psf_lut.pt) in **FP32**.
- **Param projection** per step: clamp XY, Z, speed, CMY to legal ranges (mm-space).
- **Primary path:**
    - Convert Bézier to a polyline ribbon.
    - Rasterize with **nvdiffrast CUDA** to get an **alpha mask** for tiles overlapped.
    - Convert CMY→RGB via color LUT (trilinear).
    - Modulate alpha via alpha_lut(z,v) and convolve with PSF (or sample PSF‑conditioned raster).
    - Composite over canvas (**alpha‑over**).
- **Fallback strategy (hardened):**
    - **render_stroke_tiled (env path):** if nvdiffrast fails, log + fallback to **splat** (Gaussian stamps) and proceed.
    - **fine_tune_stroke_technician (inference path):** if CUDA rasterizer or backward fails, **raise** and discard stroke—do not switch backends mid-optimization.
- **BF16** aware; logs Technician convergence (steps, ΔLPIPS).

---

## **7) Training & MLflow‑integrated HPO**

### **7.1 Training config**

configs/train.yaml

```
task: "AirbrushPainter-v0"
device: "cuda:0"
compute:
  precision: "bf16"           # DGX Spark default (Blackwell Tensor Cores)
  channels_last: true         # free CNN perf
  grad_checkpointing: false   # DGX has headroom; prioritize speed
  torch_compile:
    enabled: false            # enable after stability confirmed
    mode: "max-autotune"
env_config: "configs/env/airbrush_v1.yaml"
sim_config: "configs/sim/physics_v1.yaml"
lpips:
  net: "vgg"
  tile: {size: 0, overlap: 0}  # 0 = full-frame on DGX; reduce if needed
agent:
  backbone: "resnet34"
  spatial_head: "heatmap_soft_argmax"  # or "coordconv"
  softargmax_temp: 1.0                 # temperature for soft-argmax (lower=sharper)
  learning_rate: 0.0003
  entropy_coef: 0.001
  gamma: 0.995
  clip_param: 0.2
curriculum:
  stage: "hard"     # content curriculum only
dataloader:
  pin_memory: false  # UMA (unified memory architecture)
logging:
  mlflow_experiment: "airbrush_train_v2"
  run_name_prefix: "train"
  log_seeds: true    # log PYTHONHASHSEED, torch seeds, cudnn flags for reproducibility
  save_interval: 10  # save training monitor artifacts every N epochs for GUI visualization

```

### **7.2 HPO search space (YAML)**

configs/hpo/search_space.v1.yaml

```
objective:
  name: "avg_neg_final_lpips_validation"
  direction: "maximize"
budget:
  n_trials: 80
  timeout_minutes: 3600
sampler: {type: "tpe", seed: 123}
pruner: {type: "median", warmup_steps: 8}
search_space:
  agent.learning_rate: {dist: "loguniform", low: 3.0e-5, high: 3.0e-3}
  agent.entropy_coef:  {dist: "loguniform", low: 1.0e-4, high: 1.0e-2}
  agent.gamma:         {dist: "uniform", low: 0.94, high: 0.999}
  agent.clip_param:    {dist: "uniform", low: 0.1, high: 0.3}
  agent.softargmax_temp: {dist: "loguniform", low: 0.5, high: 3.0}
validation:
  image_dir: "data/validation_images/cmy_only/"
  stroke_cap: 1500
constraints:
  max_strokes: 1500
mlflow:
  experiment: "airbrush_hpo_v2"
  run_name_prefix: "trial"

```

### **7.3 HPO execution (scripts/train.py)**

**Objective function flow (per trial):**

1. **Start MLflow run** (nested under HPO study).
2. **Sample hyperparameters** from search_space.v1.yaml using trial.suggest_*.
3. **Patch config:** Merge sampled params into train.yaml (OmegaConf).
4. **Train subprocess:** Launch rl-games training on standard training set; returns checkpoint path and MLflow run_id.
5. **Load checkpoint:** Load trained PPO model from checkpoint.
6. **Validation inference loop:**
    - For each of 10 validation images in `data/validation_images/cmy_only/`:
        - Run full Hybrid loop (Strategist→Technician, stroke_cap steps) via [paint.py](http://paint.py/) logic.
        - Render final simulated canvas at reward_px.
        - Compute final LPIPS vs. validation target (FP32, [-1,1] normalized).
        - Save simulated painting to `outputs/hpo_eval/trial_<N>/<image>.png`.
7. **Calculate average LPIPS** across 10 validation images.
8. **Log to MLflow:** avg_final_lpips_validation, individual scores, generated images as artifacts.
9. **Return objective:** `avg_lpips` (Optuna maximizes; lower LPIPS is better).
10. **Best config:** After all trials, save best_config.yaml with winning hyperparameters.

---

## **7.4) HPO Validation Set Workflow (Detailed)**

This section provides the complete workflow for implementing HPO with a dedicated validation set.

### **Step 1: Create Validation Set**

**Manual curation:**

1. Select 10 diverse target images representing the full range of complexity:
2. Ensure these images are **excluded** from `data/target_images/{easy,medium,hard}/`.

**Preprocessing:**

### **Step 2: Refactor [paint.py](http://paint.py/)**

**Create callable paint_main function:**

### **Step 3: Modify scripts/train.py HPO Objective**

**Update objective function:**

### **Step 4: Update run_training_subprocess**

**Enhanced return signature:**

### **Step 5: Considerations & Tuning**

**Stroke cap for validation:**

- Use full `stroke_cap=1500` for production-representative evaluation.
- If trials are too slow, consider `stroke_cap=750` during HPO search, then final validation at 1500.

**Validation resolution:**

- Use `reward_px` from env config (default: 908×1280).
- Match training conditions for fair comparison.

**Pruning:**

- Optionally implement early stopping based on intermediate training metrics.
- Median pruner can terminate unpromising trials after 20-30% of training.

**Parallelization:**

- Each trial is independent; run multiple trials in parallel if resources allow.
- Ensure each trial writes to unique checkpoint/output paths.

---

## **8) Inference & production (Hybrid loop → G‑code)**

scripts/paint.py

**Refactored structure:**

- **paint_main(checkpoint_path, job_config, output_dir):** Core inference function callable from CLI or HPO.
    - Load checkpoint + config.
    - Build **CMY‑only target** and **pen vectors** from data_pipeline.
    - Init renderer; canvas = white.
    - **For i in 1..stroke_cap**:
        - Obs = concat(Target, Canvas, Error).
        - a_proposed = strategist(obs, deterministic=True).
        - a_final = technician_refine(a_proposed, canvas, target) (≤ N steps, early‑stop).
        - Append to strokes.yaml (validated against stroke.v1.yaml).
        - canvas = render_stroke_tiled(canvas, a_final).
    - Save final_sim.png.
    - Generate **G‑code**.
    - Write **job manifest** (job.v1.yaml, fully populated).
    - **Return:** {final_canvas, final_lpips, strokes_path, gcode_path}.
- **CLI entry point:** `if __name__ == "__main__":` parses args and calls paint_main.

src/utils/gcode_generator.py

- Read **machine profile** to determine units, feed conversions (mm/s → F mm/min), macros, purge zone.
- Before each stroke, emit **PURGE macro**; include ; STROKE_ID: <id> comment markers.
- Convert Bézier → linearized G1 segments with interpolated Z and feed.
- Pen pass emits PEN_UP / PEN_DOWN macros and G0/G1 moves along vectors.
- Validate all moves against **soft limits** (from profile).
- Save **_cmy.gcode**, **_pen.gcode**, **_manifest.yaml** in the job folder.

src/utils/gcode_vm.py (dry‑run simulator)

- Parse G‑code; enforce soft limits; compute a time estimate (feeds, distances, purges).
- Used by GUI “Dry Run”; **no device I/O** (you stated files are the final output).

---

## **9) GUI (Training Visualizer & Inference Tool)**

**Purpose:** Provide a graphical interface for monitoring training progress, analyzing results, performing calibration, and generating final G-code without interfering with the core training loop. The GUI runs as a **separate process** that reads data produced by the training script.

**Key Principles:**

1. **Decoupled:** The GUI is not part of scripts/train.py loop. Training runs fully headlessly.
2. **Observer:** The GUI acts as an observer, monitoring files and logs generated by the training process.
3. **HD-Aware:** Uses tiled image viewers (HDTiledImageViewer) to handle large images responsively.
4. **On-Demand Simulation:** For stroke playback, the GUI uses its own instance of DifferentiableRenderer to re-render paintings as needed.

### **9.1 Training Script Data Export**

To enable visualization, scripts/train.py periodically saves data via callback/observer:

**Trigger:** Every N epochs (configurable save_interval in train.yaml).

**Data Saved to** `outputs/training_monitor/epoch_{epoch_num}/`:

- target.png: HD render_px target image from highest-reward agent.
- canvas.png: Final HD render_px canvas from that agent.
- strokes.yaml: Complete list of stroke objects (stroke.v1 schema) for playback.
- metadata.yaml: Epoch number, final LPIPS score, stroke count.
- latest/ symlink: Points to most recent epoch for easy GUI access.

**Atomic writes:** Files written to .tmp names, fsync'd, then renamed atomically to prevent GUI from reading partial files.

**Filesystem Monitoring:** GUI uses watchdog library to detect new epochs.

### **9.2 Training Tab**

**Purpose:** Visualize progress of currently running or previously completed training sessions.

**Components:**

1. **Training Status Indicator:** Detects active [train.py](http://train.py/) process via lock file or PID check.
2. **Live Monitoring Section:**
    - Target vs. Canvas Viewer: HDTiledImageViewer with two panes monitoring outputs/training_monitor/latest/. Updates when new target.png and canvas.png appear.
    - Live Metrics Plot: pyqtgraph widget reading MLflow API or log CSVs; plots mean_reward, mean_LPIPS_score over epochs.
3. **Epoch Playback Section:**
    - Epoch Selector: Dropdown/slider populated by scanning epoch directories. User selects specific epoch (e.g., "Epoch 100").
    - Selected Epoch Viewer: HDTiledImageViewer showing target.png and final canvas.png for selected epoch; displays metadata from metadata.yaml.
    - Stroke Playback Controls:
        - Controls: First, Prev, Next, Last, Play/Pause buttons; slider showing "Stroke X / N".
        - Canvas Viewer (Playback): Dedicated HDTiledImageViewer for playback rendering.
        - Logic: GUI uses internal DifferentiableRenderer instance. When user navigates to stroke N, GUI re-renders strokes 1 through N from loaded strokes.yaml. Play button iterates with timer.

**Underlying Logic:** watchdog for live updates; on-demand rendering for playback using GUI-local simulator instance.

### **9.3 Inference Tab**

**Purpose:** Run trained model on new target image (post-training) and generate G-code.

**Components:**

1. **Load Controls:** Buttons for "Load Checkpoint (.pth)" and "Load Target Image (png/jpg)".
2. **Configuration Display:** Shows stroke cap, resolutions; allows print resolution override.
3. **Execution Controls:**
    - Run Step-by-Step: Executes one Strategist→Technician loop using loaded checkpoint and internal renderer. Updates canvas viewer.
    - Run to Completion: Executes full painting loop internally (like scripts/paint.py). Progressive canvas updates. Saves final PNG to outputs/final_paintings/.
    - Dry Run G-Code: Requires completion or loaded strokes.yaml. Runs gcode_vm.py; displays estimated time and soft-limit violations.
    - Generate G-Code Files: Saves _cmy.gcode, _pen.gcode, _manifest.yaml to gcode_output/.
4. **Canvas Viewer:** HDTiledImageViewer displaying simulated painting result.

### **9.4 Calibration Tab**

**Purpose:** Manage manual calibration workflow.

**Components:**

1. **G-Code Generation:** Button "Generate Calibration G-Code". Runs calibrate.py→generate_calibration_gcode. Reports save location.
2. **Manual Data Entry Aid:**
    - Displays expected layout from calibration_layout.yaml.
    - Provides fields/table mirroring manual_calibration_results.yaml for easier data entry.
    - Button "Save Manual Results YAML".
3. **LUT Building:** Button "Build LUTs from Manual Results". Runs calibrate.py→build_luts_from_manual. Shows validation results (Delta E) and confirms LUTs saved to configs/sim/luts/.
4. **LUT Viewer (Optional):** Widgets to visualize slices or plots of generated .pt LUTs for smoothness/correctness verification.

**Architecture:** GUI maintains its own renderer, LPIPS metric, and file watchers. Never shares state with training process. All communication via filesystem artifacts.

---

## **10) Testing & CI (HD)**

### **10.1 Unit & property tests**

- **Schemas:** validate example YAMLs (stroke/luts/machine/job).
- **Renderer:** param projection works; PSF normalization; no NaNs after randomized action fuzz; fallback path engaged when forced.
- **HD compute:** tiled LPIPS ≈ full LPIPS (on small frames where full fits); OOM guard triggers → recovers.
- **G‑code:** round‑trip small synthetic job; markers present; macros included; feed conversion correct; soft‑limit failures detected.

### **10.2 Adversarial target tests (reward hacking)**

tests/test_reward_hacks.py

- **All‑black HD** and **all‑white HD**: ensure policy does **not** flood paint needlessly (e.g., verify **very low total alpha** / blank strokes.yaml acceptable).
- **Checkerboard HD**: ensure no background wash “cheat” that reduces LPIPS but destroys structure; assert improvement ≤ small ε and stroke count under a threshold.

### **10.3 Golden image CI (HD)**

ci/golden_tests/

- **images/**: g1_hd.png … g5_hd.png (A4 HD).
- **expected/**: per image tolerances in YAML:

```
image: "ci/golden_tests/images/g1_hd.png"
tolerances:
  lpips_max: 0.085
  psnr_min: 26.0
  strokes_max: 1200
renderer:
  technician_steps: 12
  precision: "bf16"
  lpips_tile: {size: 0, overlap: 0}

```

- **Smoke (per push):** run 1 golden image; fail if out of tolerance.
- **Full (nightly):** run all 5; store diffs in outputs/ci/.

---

## **11) Containerization**

Dockerfile (outline)

- Base: CUDA‑compatible PyTorch image.
- Install: torch, rl-games, lpips, opencv, nvdiffrast/kaolin, PyQt5 (optional), mlflow, optuna, pydantic, potrace bindings.
- Set envs for CUDA/AMP; expose entrypoints:
    - scripts/train.py --config configs/train.yaml
    - scripts/paint.py --job configs/schema/job_sample.yaml
    - scripts/launch_gui.py

Everything runs in‑container for **deterministic** builds and CI parity.

---

## **12) Runtime observability**

- **MLflow metrics** (per episode / per stroke where feasible):
    - LPIPS (current, Δ), Technician steps & convergence rate, renderer fallback count, GPU memory high‑water mark, stroke utilization (used / cap).
- **HPO metrics** (per trial):
    - avg_final_lpips_validation (primary objective), individual LPIPS per validation image, training loss curves.
- **Artifacts**:
    - Training: save HD **target**, **final_sim**, and **stroke heatmaps** (over‑paint visualization) for debugging.
    - HPO: save 10 generated validation paintings per trial to `outputs/hpo_eval/trial_<N>/`; log as MLflow artifacts.

---

## **13) Roadmap (software milestones)**

1. **Schemas & validators** (YAML contracts; fail‑fast loaders).
2. **Renderer core** (tiled + AMP + projection + fallback; unit tests).
3. **Env HD** (9×HD obs; tiled LPIPS; stroke cap).
4. **Data pipeline** (preprocess + pen vectorizer YAML).
5. **Validation set creation** (10 diverse images; preprocess to data/validation_images/).
6. [**paint.py](http://paint.py/) refactor** (paint_main callable function returning dict; CLI preserved).
7. **G‑code generator + VM** (macros, markers, soft‑limit checks).
8. **HPO wiring** (Optuna↔MLflow with validation loop; train→infer→evaluate per trial).
9. **HD Golden CI & adversarial tests** (smoke+nightly).
10. **Training callback** (save_interval artifacts: target/canvas/strokes/metadata per epoch).
11. **GUI architecture** (decoupled observer with watchdog; tiled viewer; stroke playback; dry‑run).
12. **Calibration ingest** (scanner homography → LUTs).

---

Below is a **file‑by‑file blueprint** for your repository. For each item I explain **purpose**, **contents**, **public APIs**, **interactions**, **extensibility**, and **tests**. The design consciously prevents duplication: shared logic lives under src/utils/ and shared contracts live in configs/schema/. All modules load configuration **only from YAML**, pass around **typed objects** (pydantic models), and expose **stable functions/classes** so you can swap implementations without touching call‑sites.

Where relevant, knobs reflect your **DGX Spark** defaults (HD, BF16, unified memory).

---

## **Top‑level**

### **Dockerfile**

**Purpose:** Containerize training/inference/GUI in a reproducible Arm64 environment (DGX OS).

**Should contain:**

- Base image: Use ARG ARCH (aarch64 for Grace Hopper / x86_64 for standard DGX); select appropriate [nvcr.io/nvidia/cuda](http://nvcr.io/nvidia/cuda) base.
- System deps: build-essential, cmake, ninja, OpenGL/EGL headers for nvdiffrast offscreen context.
- Python deps: torch/vision (matching CUDA), rl-games, lpips, opencv-python, omegaconf, mlflow, optuna, pydantic, pyyaml, PyQt5 (if GUI inside container), pyqtgraph, watchdog, scikit-image, shapely, pypotrace (or potrace bindings), nvdiffrast (build from source).
- Environment variables for BF16, channels‑last default toggles.

**Interactions:** Everything runs inside the container, including CI jobs.

**Extensibility:** Add build args for versions; use multi‑stage builds to cache heavy wheels.

**Tests:** Build on CI; run scripts/train.py --help, scripts/paint.py --help inside the image.

---

### **.dockerignore**

**Purpose:** Keep image slim.

**Should contain:** data/, outputs/, gcode_output/, .git/, **pycache**/, *.pth, local virtualenvs, etc.

---

## **configs/**

## **(single source of truth)**

> All code reads settings via
> 
> 
> **OmegaConf**
> 
> **pydantic**
> 

### **configs/train.yaml**

**Purpose:** Global training & compute knobs.

**Should contain:**

- compute: { precision: "bf16", channels_last: true, grad_checkpointing: false, torch_compile: {enabled: false, mode: "max-autotune"} }
- lpips: { tile: {size: 0, overlap: 0} } (0 = full‑frame on DGX; you can still set 2048… if needed)
- env_config: path to env/airbrush_v1.yaml
- sim_config: path to sim/physics_v1.yaml
- agent: PPO hyperparams (learning rate, entropy, gamma, clip) + spatial_head: "heatmap_soft_argmax" or "coordconv"
- curriculum: { stage: "hard" } (content‑based buckets, not resolution)
- dataloader: { pin_memory: false } (UMA)
- logging: { mlflow_experiment: "airbrush_train_v2", run_name_prefix: "train" }

**Interactions:** Loaded by scripts/train.py and by env_v1.py.

**Extensibility:** Add nested blocks for Strategist/Technician toggles; keep unknown keys rejected by validators.

---

### **configs/env/airbrush_v1.yaml**

**Purpose:** Environment parameters that **never change per run** unless you edit this file.

**Should contain:**

```
units: "mm"
work_area_mm: {x: 210.0, y: 297.0}

resolutions:
  render_px: {w: 908, h: 1280}   # physics grid; portrait ratio matches 210/297
  obs_px:    {w: 454, h: 640}    # strategist input (downsampled for speed)
  reward_px: {w: 908, h: 1280}   # LPIPS scoring resolution (coupled to render_px)

coordinate_frames:
  image_origin: "top_left"       # +Y down (standard image convention)
  machine_origin: "bottom_left"  # +Y up (standard G-code convention)
  flip_y_for_gcode: true         # transform image→machine at G-code gen

stroke_cap: 1500

renderer:
  tile_size: 0            # 0 = full-frame on DGX Spark
  tile_overlap: 0
  technician_steps: 12
  early_stop_delta: 1.0e-4
  project_params_each_step: true

```

**Interactions:** Read by env_v1.py to set three independent resolutions, termination, and renderer behavior.

**Extensibility:** Add safety constraints (min/max Z, speed limits) used by parameter projection. Can raise render_px/reward_px for final print evaluation.

---

### **configs/sim/physics_v1.yaml**

**Purpose:** Paths and toggles for the digital twin.

**Should contain:**

```
luts_config: "configs/sim/luts/metadata.yaml"   # optional, or inline below
color_lut:  "configs/sim/luts/color_lut.pt"
alpha_lut:  "configs/sim/luts/alpha_lut.pt"
psf_lut:    "configs/sim/luts/psf_lut.pt"
rasterizer:
  backend: "cuda"   # nvdiffrast CUDA
  fallback: "splat"
color_space: "linear_rgb"

```

**Interactions:** Loaded in differentiable_renderer.py.

**Extensibility:** Add per‑nozzle profiles; version them.

---

### **configs/sim/luts/{color_lut.pt, psf_lut.pt, alpha_lut.pt}**

**Purpose:** Torch tensors saved by calibration.

**Contents:**

- color_lut.pt: shape [Nc, Nm, Ny, 3] linear RGB, dtype **fp32**.
- psf_lut.pt: shape [Nz, Nv, K, K] normalized kernels, dtype fp32.
- alpha_lut.pt: shape [Nz, Nv] scalar coverage factors, dtype fp32.

**Interactions:** Loaded by the renderer; interpolated with pure torch ops.

---

### **configs/hpo/search_space.v1.yaml**

**Purpose:** YAML‑defined Optuna/MLflow HPO.

**Contents:** Objective name, budget, sampler, pruner, search ranges for PPO params, plus constraints.max_strokes.

**Interactions:** Consumed by scripts/train.py objective.

---

### **configs/machine/grbl_airbrush.v1.yaml**

**Purpose:** Machine‑level constraints and post‑processor details.

**Contents:** work area, gcode flavor, feed units, max feeds, purge zone, macro include dir.

**Interactions:** Used by gcode_generator.py and gcode_vm.py for validation/time estimates.

---

### **configs/scanner/scanner_profile.v1.yaml**

**Purpose:** Deterministic scan ingestion (you provide PNG/JPEG from printer‑scanner).

**Contents:** { input_color_space: "srgb", linearize: true, deskew: true, homography: {enabled: true}, patch_sampling: {method: "median", roi_margin_px: 6} }

**Interactions:** Used by data_pipeline/calibrate.py.

---

### **configs/schema/*.yaml**

### **(stroke.v1.yaml,luts.v1.yaml,machine.v1.yaml,job.v1.yaml)**

**Purpose:** Contracts validated via pydantic.

**Contents:** Bounds, units, required fields, paths. See earlier drafts.

**Interactions:** Enforced by src/utils/validators.py in all scripts.

---

## **data/**

**(inputs)**

### **data/raw_images/**

Where you drop HD source images. [preprocess.py](http://preprocess.py/) consumes these.

### **data/calibration_scans/**

Where you drop scanner outputs for LUT building.

### **data/validation_images/**

**Purpose:** Fixed validation set for HPO evaluation.

- **cmy_only/**: 10 diverse preprocessed color-only targets at render_px.
- **pen_layers/**: Corresponding pen vector YAML files.

**Creation:** Manually select 10 representative images, preprocess with standard pipeline, place here. These images **must not** appear in training sets.

**Interactions:** Used only by HPO objective function in scripts/train.py for final LPIPS scoring per trial.

### **data/target_images/**

- cmy_only/*: HD color‑only targets produced by [preprocess.py](http://preprocess.py/).
- pen_layers/*: YAML vectors produced by pen_vectorizer.py.

**Interactions:** Env, paint script, and GUI read from here for training and production inference.

---

## **outputs/**

**(derived artifacts)**

- checkpoints/: PPO weights and best_config.yaml.
- training_monitor/: **tiled** PNGs for GUI during training.
- final_paintings/: final simulated HD PNGs from [paint.py](http://paint.py/).
- hpo_eval/trial_<N>/: Validation paintings generated during HPO trials (10 images per trial).
- studies/hpo.db: Optuna SQLite store.
- ci/: golden diffs/reports.

---

## **gcode_output/<job>/**

- <job>_cmy.gcode, <job>_pen.gcode, <job>_manifest.yaml.
- **Manifest** is a resolved snapshot of the job (paths, limits, machine profile id).

**Interactions:** GUI “Dry Run” loads these into gcode_vm.py.

---

## **src/**

### **src/airbrush_simulator/differentiable_renderer.py**

**Purpose:** The production GPU renderer + technician optimizer. MM-space geometry, BF16, parameter projection; CUDA rasterizer; hardened fallback.

**Public API:**

- class DifferentiableRenderer(cfg):
    - render_stroke_tiled(canvas: torch.Tensor, params_mm: torch.Tensor) -> torch.Tensor
    (params in mm; converts to px internally; fallback to splat on CUDA error)
    - fine_tune_stroke_technician(a_init_mm, canvas, target) -> torch.Tensor
    (runs N steps of gradient descent; early‑stop; raises on CUDA error—no fallback)
    - project_params(params_mm) -> params_mm (clamps xy/z/speed/cmy in mm-space)
    - load_luts(color_lut, alpha_lut, psf_lut) (LUTs in FP32)

**Interactions:**

- Called by env_v1.py during training and by [paint.py](http://paint.py/) during inference.
- Uses LUTs from configs/sim/physics_v1.yaml.
- Uses work_area_mm + render_px from env config for mm↔px conversion.

**Extensibility:**

- Swap rasterizer backend behind the same API (but respect fallback rules).
- Change PSF model without touching callers.
- Technician step count and tolerances driven solely by YAML.

**Tests:**

- Projection clamps out‑of‑bounds (mm-space).
- No NaNs under fuzzed actions.
- Fallback path (splat) renders within tolerance vs. primary.
- Technician raises (no fallback) on CUDA error.

---

### **src/airbrush_simulator/fast_simulator.py**

**Purpose:** Simple, CPU‑only renderer for unit tests and quick repros (reference implementation). **Kept under tests/ only**—not imported into training.

**Public API:** render_stroke(canvas_np, params_mm_np) -> canvas_np.

**Interactions:** Used only in tests or debugging.

**Extensibility:** Keep math mirrored to the GPU path for sanity checks.

**Tests:** Golden outputs on toy strokes; mm-space parameters.

---

### **src/airbrush_robot_env/env_v1.py**

**Purpose:** RL environment (rl‑games compatible) with multi-resolution architecture.

**Public API:**

- **init**(cfg) loads env/sim configs (including 3 resolutions), LUTs, Tiled LPIPS module.
- reset() -> obs initializes white canvas at render_px; picks random target from content bucket; returns obs downsampled to obs_px.
- step(action_normalized) -> (obs, reward, terminated, info) where:
    - action_normalized: [-1, 1]^15; env maps to mm using schema-driven scaling, then projects to bounds.
    - obs = cat(target, canvas, abs_diff) downsampled to (9, Ho, Wo) FP32 numpy (AMP stays inside model).
    - reward = LPIPS(old) − LPIPS(new) computed at reward_px (dense).
    - terminated when stroke_cap reached.
- _get_obs() -> downsample {target, canvas, error} from render_px to obs_px.

**Interactions:** Used by scripts/train.py via rl‑games.

**Extensibility:** Add optional observation channels or masks via YAML flags; adjust resolutions per experiment.

**Tests:** Shapes (obs at obs_px, canvas at render_px, reward at reward_px), dtype, deterministic reset under fixed seed; reward sign checks; mm↔px roundtrip.

---

### **src/data_pipeline/preprocess.py**

**Purpose:** Convert user images → color‑only target + pen mask at desired resolution.

**Public API:**

- separate_layers(img_path, out_dir, target_px=None) -> (cmy_png_path, pen_mask_path)
    - Standardizes/resizes to target_px (or A4 aspect ratio if None).
    - LAB threshold to find black lines; small morphology.
    - Inpaint CMY and save PNG.

**Interactions:** Called by GUI and scripts/paint.py; resolution driven by env config or user override.

**Extensibility:** Swap black‑line detection heuristics; configurable thresholds in YAML.

**Tests:** On fixtures, assert expected output size; mask sparsity; no alpha.

---

### **src/data_pipeline/pen_vectorizer.py**

**Purpose:** Turn pen_mask into centerline vectors.

**Public API:**

- vectorize_pen_mask(pen_mask_path, out_yaml_path) -> None
    - Potrace → polylines → simplify (Douglas‑Peucker) → YAML
    - Path ordering optional (nearest neighbor).

**Interactions:** Used by [paint.py](http://paint.py/) and GUI to generate the pen pass.

**Extensibility:** Pluggable simplifiers/orderers.

**Tests:** Path count/length; endpoints; YAML schema compliance.

---

### **src/data_pipeline/calibrate.py**

**Purpose:** Generate calibration jobs; write LUTs from manual measurements.

**Public API:**

- generate_calibration_gcode(machine_cfg, calib_layout_cfg, out_dir): Generates labeled calibration patterns.
- build_luts_from_manual(manual_results_yaml, out_dir): Reads manual measurements, interpolates, generates .pt LUTs.

**Calibration mode:** Manual measurement workflow (Section 3.2). Operator prints patterns, measures with calibrated tools, enters data into manual_calibration_results.yaml.

**Interactions:** Updates configs/sim/physics_v1.yaml to point at new LUTs; validates layering model via Delta E comparison.

**Extensibility:** Support scanner-based automatic mode via YAML switch (calibration.mode: "manual" or "scanner").

**Tests:** Manual input fixtures; LUT shapes; interpolation smoothness; Delta E validation.

---

### **src/rl_agent/networks_hd.py**

**Purpose:** Strategist network (ResNet‑34 backbone) with spatial-aware heads for coordinate prediction.

**Public API:**

- build_actor_critic(cfg) -> (actor, critic)
    - Applies channels_last if configured.
    - **Spatial head options:**
        - "coordconv": prepend 2 normalized (x,y) coord channels to 9-ch obs → 11 channels; standard ResNet + linear heads.
        - "heatmap_soft_argmax": last conv produces K heatmaps for Bézier (x,y) coords; soft-argmax yields sub-pixel mm coords; separate MLPs for {z,v,cmy}.
    - BF16 autocast in forward; FP32 in loss accumulation.
    - Optional torch.compile for the backbone only (disabled by default).

**Interactions:** Used by rl‑games policy builder; receives obs at obs_px, outputs action in mm-space.

**Extensibility:** Swap to other backbones via YAML (name → factory mapping); add new spatial head types.

**Tests:** Forward pass on dummy (9, Ho, Wo); parameter counts; BF16 path; coordinate prediction sanity (synthetic bright dot test).

---

### **src/gui/main_window.py, src/gui/tabs/, src/gui/widgets/**

**Purpose:** Decoupled PyQt GUI for training visualization, inference, and calibration.

**Architecture:** GUI runs as separate process; monitors filesystem artifacts; maintains own renderer instance.

**Key widgets:**

- HDTiledImageViewer: Lazy tile loading for HD images (canvas, target).
- TrainingTab: Monitors outputs/training_monitor/ via watchdog; plots MLflow metrics; provides epoch selector and stroke playback with on-demand re-rendering.
- InferenceTab: Loads checkpoint; runs Hybrid loop (step/full) using GUI-local renderer; Dry Run via gcode_vm; generates final G-code.
- CalibrationTab: Generates calibration G-code; provides manual data entry aid; builds LUTs; optional LUT visualization.

**Public API:**

- TrainingTab.load_epoch(epoch_num): Loads target/canvas/strokes for selected epoch.
- TrainingTab.render_to_stroke(stroke_idx): Re-renders strokes 1..N for playback.
- InferenceTab.run_painting_step(): Single Strategist→Technician iteration.
- InferenceTab.run_painting_full(): Complete painting loop with progress updates.

**Interactions:**

- Reads from outputs/training_monitor/, data/validation_images/, outputs/checkpoints/.
- Calls paint_main for inference (internally, not subprocess).
- Calls gcode_vm.py for dry runs.
- Never writes to or reads from training process memory/state.

**Dependencies:** PyQt5, pyqtgraph, watchdog, PIL/torchvision for image handling.

**Extensibility:** Tab plugins via interface; config-driven paths; renderer/LPIPS instances created per-tab as needed.

**Tests:**

- Smoke test: app launches; widgets render.
- Mock training data: verify epoch loading and playback rendering.
- Dry Run on synthetic job: confirm gcode_vm integration.
- Watchdog test: simulate new epoch directory appearing; verify GUI updates.

---

### **src/utils/validators.py**

**Purpose:** Centralized YAML schema validation.

**Public API:**

- load_job_config(path) -> JobV1
- load_machine_profile(path) -> MachineV1
- validate_strokes(path) -> None
- validate_luts(meta_or_paths) -> None

**Interactions:** Imported by **every** script and module that opens YAML.

**Extensibility:** Add new versions (*.v2.yaml) with side‑by‑side models.

**Tests:** Invalid examples raise with clear messages; valid pass.

---

### **src/utils/compute.py**

**Purpose:** Shared compute utilities including resolution conversions.

**Public API:**

- class TiledLPIPS(cfg): forward(img1, img2, tile_size, overlap) with cosine windowing; supports tile_size=0 = full‑frame.
- autocast_context(precision: "bf16"|"fp16"|"fp32") → context manager.
- mm_to_px(x_mm, work_area_mm, render_px) -> x_px (converts mm coords to pixel coords)
- px_to_mm(x_px, work_area_mm, render_px) -> x_mm (converts pixel coords to mm coords)
- choose_tile_size() (optional heuristic).
- OOM guard helpers (retry_with_smaller_tiles) (less needed on DGX).

**Interactions:** Used by env, renderer, and CI compare script; mm↔px used throughout for geometry conversions.

**Extensibility:** Plug in alternative metrics (e.g., SSIM) behind same interface.

**Tests:** Tiled ≈ full on small frames; BF16/FP32 parity bounds; mm↔px roundtrip is identity within tolerance.

---

### **src/utils/gcode_generator.py**

**Purpose:** Turn stroke/pen YAML into G‑code with proper coordinate frame transformation.

**Public API:**

- generate_all_gcode(strokes_yaml, pen_yaml, machine_cfg, env_cfg, out_dir)
    - Emits header with units/positioning/zero commands (from machine profile).
    - Transforms image-frame mm → machine-frame mm using env_cfg.coordinate_frames.
    - Includes macros (PURGE, PEN_UP/DOWN).
    - Emits ; STROKE_ID: <id> markers.
    - Converts mm/s → F mm/min; clamps to machine feeds.
    - Validates soft limits; raises on violation.

**Header template (GRBL-safe):**

```
; --- AIRBRUSH JOB START ---
G21            ; mm units
G90            ; absolute positioning
G92 X0 Y0 Z0   ; zero current position (optional, from machine profile)
; [PURGE/AIR macros loaded from machine profile - M7/M8 for coolant or custom]
; Note: M82 removed (RepRap-specific, not GRBL-compatible)

```

**Interactions:** Called by scripts/paint.py; GUI's "Generate G‑code".

**Extensibility:** Machine‑specific post‑processors via strategy classes; header from YAML template.

**Tests:** Soft‑limit detection; feed conversion; macro inclusion; coordinate frame transform (corner stroke test).

---

### **src/utils/gcode_vm.py**

**Purpose:** Offline G‑code simulator for **Dry Run**.

**Public API:**

- class GCodeVM(machine_cfg)
    - load_file(gcode_path)
    - run() -> {time_estimate, violations: [], last_stroke_id}
    - Simple kinematics (constant accel optional), feed clamping, purge time accounting.

**Interactions:** Used by GUI and CI to validate generated jobs quickly.

**Extensibility:** Plug motion models (trapezoidal vs. naive).

**Tests:** Synthetic jobs → expected times; detect out‑of‑bounds moves.

---

## **scripts/**

### **scripts/train.py**

**Purpose:** Launch PPO training, optionally with HPO.

**Flow:**

1. Load train.yaml and validate.
2. If --hpo, also load configs/hpo/search_space.v1.yaml.
3. **Without HPO:**
    - Build rl‑games Runner; register env class (env_v1); feed Strategist nets from networks_hd.py.
    - Log to MLflow; snapshot configs, LUT versions, **and {render_px, obs_px, reward_px}** into run params for traceability.
    - Save checkpoint + config.
4. **With HPO (Optuna study):**
    - Create Optuna study from search_space.v1.yaml.
    - For each trial:
        - **Train phase:** Sample hyperparameters, patch config, launch rl-games subprocess on training set; subprocess returns checkpoint_path and mlflow_run_id.
        - **Load checkpoint:** Load trained PPO model.
        - **Validation phase:** Run paint_main on each of 10 validation images from `data/validation_images/cmy_only/`; compute final LPIPS per image; save paintings to `outputs/hpo_eval/trial_<N>/`.
        - **Score:** Average LPIPS across validation set; return `avg_lpips` to Optuna.
        - **Log:** avg_final_lpips_validation, individual scores, generated validation paintings as MLflow artifacts.
    - After all trials, save best_config.yaml with winning hyperparameters.
5. Set torch_compile.enabled: false by default (enable after stability confirmed).

**Extensibility:** Add flags to export ONNX/TorchScript of actor; support pruning via intermediate training metrics.

**Tests:** Dry run with 1 episode and mock env; verify artifacts; confirm resolutions logged; HPO dry run with 2 trials and mock validation set.

---

### **scripts/paint.py**

**Purpose:** Headless Hybrid loop → strokes + G‑code.

**Refactored architecture:**

**Public API:**

- **paint_main(checkpoint_path, job_config, output_dir) -> dict:** Core inference function.
    - Returns: {final_canvas: Tensor, final_lpips: float, strokes_path: str, gcode_path: str}.
    - Callable from CLI or programmatically (HPO validation loop).

**Flow:**

1. Load checkpoint & config; read job config (target image, stroke_cap, etc.).
2. Optional --print_res_px W H flag: draft mode uses render_px; final mode uses higher print resolution.
3. preprocess.separate_layers at target resolution; pen_vectorizer.vectorize_pen_mask.
4. Loop to stroke_cap: Strategist → Technician (in mm-space) → append stroke → render at render_px (or print_px for final).
5. Save strokes.yaml (mm-space), final PNG; call gcode_generator.generate_all_gcode.
6. Compute final LPIPS and return with outputs.
7. Log {render_px, obs_px, reward_px, print_px} to MLflow params for traceability (if MLflow enabled).

**CLI entry point:** `if __name__ == "__main__":` parses args and calls paint_main.

**Extensibility:** Multi‑image batching (optional) without API change.

**Tests:** Tiny synthetic image; ensure stroke cap obeyed; outputs exist; mm-space strokes validated; callable API returns expected dict.

---

### **scripts/launch_gui.py**

**Purpose:** Start the PyQt GUI.

**Flow:** Build QApplication, instantiate MainWindow, pass default config paths.

**Tests:** Smoke test (headless CI uses QT_QPA_PLATFORM=offscreen).

---

## **ci/golden_tests/**

### **images/**

A4 goldens at various resolutions: g1.png … g5.png (configurable per test).

### **expected/**

YAML tolerances per image:

```
image: "ci/golden_tests/images/g1.png"
resolutions:
  render_px: {w: 908, h: 1280}  # portrait ratio matches A4
  reward_px: {w: 908, h: 1280}
tolerances: { lpips_max: 0.085, psnr_min: 26.0, strokes_max: 1200 }
renderer:   { technician_steps: 12, precision: "bf16", lpips_tile: {size: 0, overlap: 0} }

```

### [**compare.py**](http://compare.py/)

**Purpose:** Run [paint.py](http://paint.py/) on a golden (at specified resolution), compute LPIPS/PSNR (using utils.compute.TiledLPIPS at reward_px), check tolerances, emit diffs to outputs/ci/.

---

## **tests/**

### **test_schemas.py**

Load every example YAML; ensure validators accept good and reject bad with clear reasons.

### **test_renderer.py**

- Param projection clamps to bounds (mm-space).
- No NaNs under randomized actions (1000 trials).
- Fallback path (splat) parity within tolerance of primary.
- Technician raises (no fallback) on forced CUDA error.

### **test_gcode_gen.py**

- Soft‑limit violations detected.
- Feed conversions correct.
- Stroke markers present; macros inserted.

### **test_reward_hacks.py**

- All‑black/white: near‑zero paint; passes LPIPS sanity.
- Checkerboard: no background flooding; stroke count under threshold.

### **test_paint_main.py**

- Verify paint_main returns dict with expected keys (final_canvas, final_lpips, strokes_path, gcode_path).
- Synthetic tiny image: ensure callable API completes without MLflow.
- Verify return types: final_canvas is Tensor, final_lpips is float, paths are strings.

### **test_compute.py**

- Tiled LPIPS ≈ full on smaller frames.
- BF16/FP32 parity bounds.

### **test_env_resolutions.py**

- Obs shape equals obs_px.
- Canvas/target stored at render_px.
- Reward computed at reward_px.
- Resolutions independent and configurable.

### **test_mm_px_roundtrip.py**

- mm→px→mm is identity within tolerance.
- Edge cases (0, work_area_mm bounds).

### **test_spatial_head.py**

- Synthetic target with bright dot → predicted control points converge near dot after supervised steps.
- Both CoordConv and heatmap+soft-argmax heads tested.
- Verify backbone returns spatial map (not GAP'd vector) when using heatmap head.

### **test_coordinate_frames.py**

- Corner stroke test: stroke at (0, 0) in image frame → correct (0, work_area_mm.y) in machine frame.
- Corner stroke at (work_area_mm.x, work_area_mm.y) in image → (work_area_mm.x, 0) in machine.
- Roundtrip: image_mm → machine_mm → G-code → parsed back → matches original.

### **test_action_scaling.py**

- Verify action space is [-1, 1]^15.
- Denormalization: [-1, -1, ...] → [0, 0, ...] mm; [1, 1, ...] → [work_area_mm.x, work_area_mm.y, ...] (for XY coords).
- Project_params clamps denormalized actions to valid ranges.

### **test_lpips_normalization.py**

- LPIPS input range: verify inputs are in [-1, 1].
- LPIPS dtype: verify computation runs in FP32 even under BF16 autocast.
- Compare LPIPS(img, img) ≈ 0 and LPIPS(white, black) > threshold.

### **test_gui_monitoring.py**

- Mock training monitor directory structure.
- Create new epoch directory with artifacts; verify watchdog callback triggers.
- Test epoch selector population from filesystem scan.
- Test stroke playback: load strokes.yaml; verify GUI renderer produces expected canvas at stroke N.
- Verify GUI never attempts to read training process memory.

---

# **How this layout avoids duplication & stays versatile**

- **One renderer** (differentiable_renderer.py) serves both RL env and inference; there isn’t a second code path.
- **One LPIPS implementation** (utils.compute.TiledLPIPS) used by env, CI, and diagnostics.
- **One place for contracts** (configs/schema/*.yaml) and **one validator** (utils/validators.py) used everywhere.
- **One G‑code generator** and **one VM** used by both CLI and GUI.
- **Configurations are injected** (YAML → OmegaConf → pydantic objects) rather than hard‑coded constants, so you can change behavior without code edits.
- **Strategy/factory pattern** used where alternatives may exist later (rasterizer backend, vectorizer, motion model), but the **public API stays fixed**.

This is the complete, per‑file blueprint you can hand to engineers to implement cleanly, with minimal cross‑talk and maximum future flexibility.

---

# **Implementation code snippets**

## **Environment implementation (env_v1.py)**

```python
# src/airbrush_robot_env/env_v1.py (core gist)

class AirbrushEnvV1(gym.Env):
    def __init__(self, cfg):
        super().__init__()
        # Define action space as normalized [-1, 1]^15 for stability
        # 15-D: 4 control points (8 coords) + z0,z1 (2) + v0,v1 (2) + c,m,y (3)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(15,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(9, cfg.obs_px.h, cfg.obs_px.w), dtype=np.float32
        )
        # Action scaling: [-1,1] -> mm-space (schema-driven, no magic numbers)
        from src.utils.validators import load_stroke_schema
        schema = load_stroke_schema("configs/schema/stroke.v1.yaml")
        bounds = schema.bounds

        # Build per-dimension [low, high]
        action_lows = np.array(
            [0.0, 0.0] * 4 +  # x,y for 4 control points
            [bounds.z[0], bounds.z[0]] +
            [bounds.speed[0], bounds.speed[0]] +
            [0.0, 0.0, 0.0]
        )
        action_highs = np.array(
            [cfg.work_area_mm.x, cfg.work_area_mm.y] * 4 +
            [bounds.z[1], bounds.z[1]] +
            [bounds.speed[1], bounds.speed[1]] +
            [1.0, 1.0, 1.0]
        )
        self.action_scale = (action_highs - action_lows) / 2.0
        self.action_bias = (action_highs + action_lows) / 2.0
        # ... init renderer, LPIPS, resolutions ...

    def _denormalize_action(self, action_normalized):
        """Map action from [-1, 1] to mm-space."""
        return action_normalized * self.action_scale + self.action_bias

    def _get_obs(self):
        error = torch.abs(self.target_render - self.canvas_render)               # (3,Hr,Wr)
        state = torch.cat([self.target_render, self.canvas_render, error], 0)    # (9,Hr,Wr)
        obs = F.interpolate(state.unsqueeze(0),
                            size=(self.obs_h, self.obs_w),
                            mode='area', align_corners=False).squeeze(0)         # (9,Ho,Wo)
        # *** CRITICAL: return FP32 for RL stack compatibility ***
        return obs.float().cpu().numpy()

    def step(self, action_normalized):
        # 1) Denormalize action [-1,1] -> mm
        action_mm = self._denormalize_action(action_normalized)
        action_mm = torch.from_numpy(action_mm).float().to(self.device)
        # 2) Project action to valid ranges (mm)
        action_mm = self.renderer.project_params(action_mm)
        # 3) Render stroke on render grid (no grad needed in env)
        with torch.no_grad():
            self.canvas_render = self.renderer.render_stroke_tiled(self.canvas_render, action_mm)
        # 4) Reward on reward grid (LPIPS in FP32)
        can_r = F.interpolate(self.canvas_render.unsqueeze(0), size=(self.rwd_h, self.rwd_w),
                              mode='area', align_corners=False)
        tgt_r = F.interpolate(self.target_render.unsqueeze(0), size=(self.rwd_h, self.rwd_w),
                              mode='area', align_corners=False)
        # Normalize to [-1, 1] for LPIPS
        can_r_norm = 2.0 * can_r - 1.0
        tgt_r_norm = 2.0 * tgt_r - 1.0
        old, new = self.curr_lpips, self.lpips(can_r_norm, tgt_r_norm)
        reward = (old - new).item()
        self.curr_lpips = new
        # 5) Termination
        self.steps += 1
        done = self.steps >= self.stroke_cap
        return self._get_obs(), reward, done, {}

```

## **MM ↔ PX conversion helpers ([compute.py](http://compute.py/))**

```python
# src/utils/compute.py

def mm_to_px(x_mm: torch.Tensor, work_area_mm: Tuple[float,float], render_px: Tuple[int,int],
             image_origin: str = "top_left"):
    """Convert mm coordinates to pixel coordinates.

    Args:
        x_mm: (..., 2) tensor with (x, y) in mm
        work_area_mm: (width_mm, height_mm) tuple
        render_px: (width_px, height_px) tuple
        image_origin: "top_left" (image frame) or "bottom_left" (machine frame)

    Returns:
        x_px: (..., 2) tensor with (x, y) in pixels
    """
    sx = render_px[0] / work_area_mm[0]
    sy = render_px[1] / work_area_mm[1]
    x_px = x_mm.clone()
    x_px[..., 0] = x_mm[..., 0] * sx
    if image_origin == "top_left":
        # Image: +Y down; invert Y
        x_px[..., 1] = (work_area_mm[1] - x_mm[..., 1]) * sy
    else:
        # Machine: +Y up
        x_px[..., 1] = x_mm[..., 1] * sy
    return x_px

def px_to_mm(x_px: torch.Tensor, work_area_mm: Tuple[float,float], render_px: Tuple[int,int],
             image_origin: str = "top_left"):
    """Convert pixel coordinates to mm coordinates.

    Args:
        x_px: (..., 2) tensor with (x, y) in pixels
        work_area_mm: (width_mm, height_mm) tuple
        render_px: (width_px, height_px) tuple
        image_origin: "top_left" (image frame) or "bottom_left" (machine frame)

    Returns:
        x_mm: (..., 2) tensor with (x, y) in mm
    """
    sx = work_area_mm[0] / render_px[0]
    sy = work_area_mm[1] / render_px[1]
    x_mm = x_px.clone()
    x_mm[..., 0] = x_px[..., 0] * sx
    if image_origin == "top_left":
        # Image: +Y down; invert Y
        x_mm[..., 1] = work_area_mm[1] - (x_px[..., 1] * sy)
    else:
        # Machine: +Y up
        x_mm[..., 1] = x_px[..., 1] * sy
    return x_mm

def image_mm_to_machine_mm(x_img_mm: torch.Tensor, work_area_mm: Tuple[float,float],
                           flip_y: bool = True):
    """Transform image-frame mm to machine-frame mm.

    Args:
        x_img_mm: (..., 2) tensor with (x, y) in image frame (top-left, +Y down)
        work_area_mm: (width_mm, height_mm) tuple
        flip_y: True if machine uses bottom-left origin (+Y up)

    Returns:
        x_mach_mm: (..., 2) tensor with (x, y) in machine frame
    """
    x_mach_mm = x_img_mm.clone()
    if flip_y:
        x_mach_mm[..., 1] = work_area_mm[1] - x_img_mm[..., 1]
    return x_mach_mm

class TiledLPIPS(nn.Module):
    """LPIPS with optional tiling; always runs in FP32 with normalized inputs."""
    def __init__(self, net='vgg', tile_size=0, overlap=0):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net).eval()
        self.tile_size = tile_size
        self.overlap = overlap

    def forward(self, img1, img2):
        """Compute LPIPS. Inputs must be in [-1, 1] range.

        Args:
            img1, img2: (B, 3, H, W) tensors in [-1, 1]

        Returns:
            loss: scalar LPIPS distance
        """
        # Force FP32 for numerical accuracy
        img1 = img1.float()
        img2 = img2.float()

        if self.tile_size == 0:
            # Full-frame
            with torch.cuda.amp.autocast(enabled=False):
                return self.lpips(img1, img2).mean()
        else:
            # Tiled with Hann windowing
            # ... (tiling logic with cosine windows) ...
            pass

```

## **Spatial head options (networks_hd.py)**

```python
# src/rl_agent/networks_hd.py

# Option A: CoordConv
class StrategistCoordConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Prepend 2 coord channels to 9-ch obs → 11 channels
        self.backbone = resnet34(in_channels=11, pretrained=False)
        # Standard linear heads for 15-D action (8 xy coords + 4 z/v + 3 cmy)
        self.actor_head = nn.Linear(512, 15)
        self.critic_head = nn.Linear(512, 1)

    def forward(self, obs):
        # obs: (B, 9, H, W)
        B, _, H, W = obs.shape
        # Create normalized coordinate grids
        yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W))
        coords = torch.stack([xx, yy], 0).unsqueeze(0).expand(B, -1, -1, -1).to(obs.device)
        # Concatenate: (B, 11, H, W)
        obs_with_coords = torch.cat([obs, coords], 1)
        features = self.backbone(obs_with_coords)
        return self.actor_head(features), self.critic_head(features)

# Option B: Heatmap + soft-argmax (recommended)
class StrategistSpatial(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = resnet34(in_channels=9, pretrained=False, remove_gap=True)
        # *** CRITICAL: backbone must return spatial map (B, 512, H', W'), NOT GAP'd vector ***
        # K=4 heatmaps for 4 Bézier control points (15-D action total)
        self.heatmap_head = nn.Conv2d(512, 4, 1)
        # Separate heads operate on GAP'd features for non-spatial params
        self.z_head = nn.Linear(512, 2)  # z0, z1
        self.v_head = nn.Linear(512, 2)  # v0, v1
        self.cmy_head = nn.Linear(512, 3)  # c, m, y
        self.critic_head = nn.Linear(512, 1)
        self.softargmax_temp = cfg.softargmax_temp  # from YAML

    def soft_argmax_2d(self, heatmap, temp=1.0):
        """Convert heatmap to normalized (x, y) in [0, 1] via soft-argmax.

        Uses pixel-center convention: grid is [0.5, 1.5, ..., W-0.5].
        Temperature τ controls sharpness (lower = sharper peaks).
        """
        B, K, H, W = heatmap.shape
        # Apply temperature and normalize to probabilities
        flat = heatmap.view(B, K, H * W) / temp
        probs = F.softmax(flat, dim=-1).view(B, K, H, W)
        # Pixel-center grid: [0.5, 1.5, ..., W-0.5] normalized to [0, 1]
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32) + 0.5,
                                 torch.arange(W, dtype=torch.float32) + 0.5)
        xx = (xx / W).to(heatmap.device)
        yy = (yy / H).to(heatmap.device)
        x_norm = (probs * xx.view(1, 1, H, W)).sum(dim=[2, 3])
        y_norm = (probs * yy.view(1, 1, H, W)).sum(dim=[2, 3])
        return x_norm, y_norm  # (B, K) in [0, 1]

    def forward(self, obs):
        # obs: (B, 9, Ho, Wo) in FP32
        feat_map = self.backbone(obs)  # (B, 512, H', W') spatial map
        assert feat_map.ndim == 4, "Backbone must return spatial map before GAP"
        # Heatmaps for coordinates (operate on spatial map)
        heatmaps = self.heatmap_head(feat_map)  # (B, 4, H', W')
        x_norm, y_norm = self.soft_argmax_2d(heatmaps, self.softargmax_temp)  # (B, 4) in [0, 1]
        # Map [0, 1] to [-1, 1] for stable action space
        x_action = 2.0 * x_norm - 1.0
        y_action = 2.0 * y_norm - 1.0
        coords_action = torch.stack([x_action, y_action], -1).view(B, 8)  # 4 points × 2
        # Other parameters (use GAP for non-spatial)
        feat_vec = feat_map.mean(dim=[2, 3])  # (B, 512)
        z = torch.tanh(self.z_head(feat_vec))
        v = torch.tanh(self.v_head(feat_vec))
        cmy = torch.tanh(self.cmy_head(feat_vec))
        action_normalized = torch.cat([coords_action, z, v, cmy], -1)  # (B, 15) in [-1, 1]
        value = self.critic_head(feat_vec)
        return action_normalized, value

```

## **Renderer fallback strategy (differentiable_renderer.py)**

```python
# src/airbrush_simulator/differentiable_renderer.py

def render_stroke_tiled(self, canvas, params_mm):
    """Render a stroke on canvas. Params in mm-space.

    Fallback to splat on CUDA error (env path). Not backpropped in RL, so
    differentiability is harmless but not required.
    """
    try:
        # Primary: nvdiffrast CUDA rasterizer
        return self._render_cuda(canvas, params_mm)
    except Exception as e:
        logging.warning(f"CUDA rasterizer failed: {e}. Falling back to splat.")
        # Splat fallback (no grad needed for env, but keep it differentiable for consistency)
        return self._render_splat(canvas, params_mm)

def fine_tune_stroke_technician(self, a_init_mm, canvas, target):
    """Refine stroke via gradient descent. Raises on CUDA error.

    No fallback allowed (inference path) to prevent changing optimization landscape.
    """
    a = a_init_mm.clone().requires_grad_(True)
    prev_loss = float('inf')
    for step in range(self.technician_steps):
        try:
            canvas_new = self._render_cuda(canvas, a)
            # Normalize inputs to [-1, 1] for LPIPS
            canvas_norm = 2.0 * canvas_new - 1.0
            target_norm = 2.0 * target - 1.0
            # LPIPS in FP32
            with torch.cuda.amp.autocast(enabled=False):
                loss = self.lpips(canvas_norm.float(), target_norm.float())
            loss.backward()
            with torch.no_grad():
                a -= self.lr * a.grad
                a = self.project_params(a)
                a.grad.zero_()
            if step > 0 and abs(loss.item() - prev_loss) < self.early_stop_delta:
                break
            prev_loss = loss.item()
        except Exception as e:
            logging.error(f"Technician CUDA error at step {step}: {e}")
            raise  # Hard fail, no fallback
    return a.detach()

```

---

# **Summary of key changes from v1.0 (HD-only) to v2.0 (multi-resolution)**

## **1) Three independent resolutions**

- **render_px:** Physics simulation grid (training default: 908×1280, portrait).
- **obs_px:** Policy network input (downsampled, e.g., 454×640 for speed).
- **reward_px:** LPIPS evaluation grid (can match render_px or be higher for final eval).
- All configurable in `configs/env/airbrush_v1.yaml`.

## **2) Geometry truth: millimeters**

- All stroke parameters (Bézier control points, Z, speed) are in **mm-space**.
- Conversions to pixels happen at render boundaries via `mm_to_px()` and `px_to_mm()` helpers in `src/utils/compute.py`.
- Work area defined as `work_area_mm: {x: 210.0, y: 297.0}` in env config.

## **3) Spatial-aware policy head**

- **Option A (CoordConv):** Prepend 2 normalized (x,y) coord channels to 9-ch obs → 11 channels; standard ResNet + linear heads.
- **Option B (recommended - Heatmap+soft-argmax):** Last conv produces K heatmaps for Bézier (x,y) coords; soft-argmax yields sub-pixel mm coords; separate MLPs for {z,v,cmy}.
- Configured via `agent.spatial_head` in train.yaml.
- Keeps policy fast (smaller obs) while maintaining coordinate prediction accuracy.

## **4) Hardened Technician fallback rule**

- **render_stroke_tiled (env path):** If nvdiffrast CUDA fails, log + fallback to splat (Gaussian stamps) and proceed.
- **fine_tune_stroke_technician (inference path):** If CUDA rasterizer or backward fails, **raise** and discard stroke—do not switch backends mid-optimization.
- Prevents changing optimization landscape during refinement.

## **5) DGX Spark defaults (BF16, full-frame, UMA)**

```yaml
compute:
  precision: "bf16"           # Blackwell Tensor Cores
  channels_last: true         # free CNN perf
  grad_checkpointing: false   # DGX has headroom
  torch_compile:
    enabled: false            # enable after stability confirmed

lpips:
  tile: {size: 0, overlap: 0}  # 0 = full-frame; reduce only if needed

renderer:
  tile: {size: 0, overlap: 0}
  luts_precision: "fp32"       # keep LUTs high precision

dataloader:
  pin_memory: false            # UMA (unified memory architecture)

```

## **6) File renames**

- `src/airbrush_robot_env/env_hd.py` → `env_v1.py`
- `configs/env/airbrush_hd_v1.yaml` → `airbrush_v1.yaml`

## **7) New tests**

- `test_env_resolutions.py`: Verify obs at obs_px, canvas at render_px, reward at reward_px.
- `test_mm_px_roundtrip.py`: Verify mm↔px conversions are lossless.
- `test_spatial_head.py`: Synthetic bright dot test for coordinate prediction.

## **8) Enhanced traceability**

- MLflow logs now include `{render_px, obs_px, reward_px, print_px}` as run params.
- Job manifests include resolution metadata.
- Golden tests specify resolution per image.

---

**Result:** Fast RL training (small obs), high-fidelity physics and reward (large render/reward grids), flexible print resolution at inference, and DGX-optimized compute—without sacrificing final quality.

---

# **Critical implementation fixes (v2.1)**

## **1) Spatial feature map for policy head**

**Problem:** Original code used GAP'd features for heatmap head, destroying spatial information.

**Fix:** Backbone must return spatial map `(B, 512, H', W')` before any global pooling. Heatmap head operates on this spatial map. Only non-spatial parameters (z, v, cmy) use GAP'd features.

**Example:** Backbone returns spatial map; heatmap head produces K=4 heatmaps for 4 control points.

## **2) Observation dtype for RL compatibility**

**Problem:** Returning BF16/FP16 obs breaks rl-games and most RL stacks.

**Fix:** Env returns **FP32 numpy arrays**. Autocast happens inside the model, not at the env boundary.

## **3) Action space normalization**

**Problem:** Raw mm-space actions have wildly different scales (210 mm for x vs. 1.0 for CMY), making policy distributions unstable.

**Fix:** Declare action space as normalized `[-1, 1]^15` in env. Map to mm-space with linear bijection before `project_params`.

## **4) Coordinate frame conventions**

**Problem:** Image frame (top-left, +Y down) vs. machine frame (bottom-left, +Y up) mismatch causes flipped output.

**Fix:** Add `coordinate_frames` to env config. Implement `image_mm_to_machine_mm()` helper. Apply transform once at G-code generation. Validate with corner stroke unit test.

**Config:**

```yaml
coordinate_frames:
  image_origin: "top_left"
  machine_origin: "bottom_left"
  flip_y_for_gcode: true

```

## **5) LPIPS precision & normalization**

**Problem:** Running LPIPS under BF16 autocast introduces numerical noise; incorrect input range degrades metric.

**Fix:** Force FP32 for LPIPS. Normalize inputs to `[-1, 1]` (expected by pretrained VGG/AlexNet).

```python
with torch.cuda.amp.autocast(enabled=False):
    loss = self.lpips(img1.float(), img2.float())

```

## **6) Soft-argmax pixel-center convention**

**Problem:** Off-by-one errors when mapping indices to normalized coords.

**Fix:** Use pixel-center grid `[0.5, 1.5, ..., W-0.5]` divided by `W` (consistent numerator/denominator).

```python
xx = (torch.arange(W, dtype=torch.float32) + 0.5) / W

```

## **7) Resampling parameters**

**Fix:** All `F.interpolate` calls use `mode='area'` (good for downsampling) and `align_corners=False` (correct for pixel grids).

## **8) G-code header & determinism**

**Fix:** Emit header once per file with units, positioning mode, zero commands (loaded from machine profile macros). Ensures deterministic machine state.

```
G21            ; mm units
G90            ; absolute positioning
G92 X0 Y0 Z0   ; zero current position

```

## **9) Reproducibility logging**

**Enhancement:** Log to MLflow:

- `PYTHONHASHSEED`
- `torch.manual_seed()` value
- `torch.cuda.manual_seed_all()` value
- `torch.backends.cudnn.deterministic` flag
- `torch.backends.cudnn.benchmark` flag

Enables exact reproduction of runs.

---

**Impact:** These fixes prevent silent correctness bugs (flipped Y, wrong LPIPS normalization), improve training stability (normalized actions, FP32 obs), and ensure spatial prediction works correctly (feature map preservation).

---

# **Implementation checklist**

When implementing this plan, follow this order to minimize rework:

## **Phase 1: Core infrastructure (no ML)**

1. ✓ Validators (`src/utils/validators.py`) – YAML schema enforcement
2. ✓ Compute helpers (`src/utils/compute.py`) – mm↔px, coordinate frames, TiledLPIPS
3. ✓ Unit tests for coordinate transforms (`test_mm_px_roundtrip.py`, `test_coordinate_frames.py`)
4. ✓ G-code generator with header & transforms (`src/utils/gcode_generator.py`)
5. ✓ G-code VM (`src/utils/gcode_vm.py`)

## **Phase 2: Renderer & physics**

1. ✓ Differentiable renderer (`src/airbrush_simulator/differentiable_renderer.py`)
    - mm-space parameters
    - FP32 LUTs
    - Hardened fallback (env vs. Technician)
2. ✓ Renderer tests (`test_renderer.py`)
3. ✓ Data pipeline (`src/data_pipeline/`) – preprocess, vectorizer, calibrate

## **Phase 3: Environment**

1. ✓ Environment v1 (`src/airbrush_robot_env/env_v1.py`)
    - Multi-resolution (render/obs/reward)
    - Normalized action space [-1, 1]^21
    - FP32 numpy obs return
    - LPIPS normalization to [-1, 1]
2. ✓ Environment tests (`test_env_resolutions.py`, `test_action_scaling.py`, `test_lpips_normalization.py`)

## **Phase 4: Policy network**

1. ✓ Networks with spatial head (`src/rl_agent/networks_hd.py`)
    - Backbone returns spatial map (no premature GAP)
    - Heatmap + soft-argmax (pixel-center convention)
    - CoordConv alternative
2. ✓ Spatial head tests (`test_spatial_head.py`)

## **Phase 5: Training & HPO**

1. ✓ Inference script refactor (`scripts/paint.py`)
    - paint_main() callable function
    - Returns dict with canvas, LPIPS, paths
    - CLI entry point preserved
2. ✓ Validation set creation
    - 10 diverse images preprocessed to `data/validation_images/`
    - Excluded from training sets
3. ✓ Training script (`scripts/train.py`)
    - MLflow logging (resolutions, seeds, config)
    - BF16 defaults, channels-last
4. ✓ HPO integration (Optuna + MLflow + validation)
    - Train subprocess returns checkpoint path
    - Validation inference loop using paint_main
    - Average LPIPS across 10 validation images as objective
    - Generated validation paintings logged as artifacts
5. ✓ Adversarial tests (`test_reward_hacks.py`)

## **Phase 6: CI & golden tests**

1. ✓ Golden tests at multiple resolutions (`ci/golden_tests/`)
2. ✓ Smoke tests (per-push)
3. ✓ Full CI (nightly)

## **Phase 7: GUI & polish**

1. ✓ GUI architecture (`src/gui/`)
    - Decoupled observer pattern with watchdog
    - HDTiledImageViewer for responsive HD display
    - Training tab: live monitoring, epoch selector, stroke playback
    - Inference tab: step/full execution, dry run, G-code generation
    - Calibration tab: G-code generation, manual data entry, LUT building
    - GUI-local renderer instance for on-demand playback
2. ✓ Training script callback for monitoring artifacts (save_interval)
3. ✓ Documentation & README
4. ✓ Containerization (`Dockerfile`)

---

# **Version history**

- **v1.0 (deprecated):** HD-only approach; single resolution; no spatial head; BF16 everywhere including LPIPS.
- **v2.0:** Multi-resolution architecture; mm-space geometry; spatial-aware policy; DGX Spark defaults.
- **v2.1:** Critical fixes for spatial features, action scaling, coordinate frames, LPIPS precision, FP32 obs return, soft-argmax conventions.
- **v2.2:** Dimensional consistency (15-D actions), schema-driven scaling, GRBL header compliance, soft-argmax temperature, reward_px coupling.
- **v2.3 (current):** HPO validation set integration; [paint.py](http://paint.py/) refactored to callable paint_main; objective evaluates on 10 fixed validation images; validation paintings logged as MLflow artifacts; decoupled GUI architecture with watchdog-based training monitoring and on-demand stroke playback rendering.

---

# **Final hardening pass (v2.2)**

## **Surgical consistency fixes**

### **1) Action dimensionality: 15-D everywhere**

**Problem:** Schema defines 1 cubic Bézier (4 control points = 8 coords) + z0,z1 (2) + v0,v1 (2) + c,m,y (3) = **15 total**. Some sections incorrectly reference 8 control points or 21-D actions.

**Fix:**

- Action space: `shape=(15,)` in all locations
- Policy heatmaps: `K = 4` (one per Bézier control point)
- Update all docs/tests/comments to "4 (x,y) control points / 15-D action"
- If dual-segment strokes needed later, introduce `stroke.v2.yaml` schema

**Affected:** `env_v1.py`, `networks_hd.py`, all tests and documentation.

---

### **2) FP32 observations from env (RL stack compatibility)**

**Problem:** Some sections still mention BF16 env returns. rl-games and most RL toolkits expect FP32 numpy.

**Rule:** Env **always** returns `np.float32`; BF16/AMP happens **inside** the model only.

**Implementation:** `observation_space.dtype = np.float32`; `_get_obs()` returns `.astype(np.float32)`.

---

### **3) Compute action scaling from schema bounds (no magic numbers)**

**Problem:** Hard-coded scales like `150.5` drift from YAML bounds and cause maintenance issues.

**Fix:** Build scales/biases dynamically from validated schema:

```python
scale = (high - low) / 2.0
bias  = (high + low) / 2.0

```

Load bounds from `stroke.v1.yaml` (x/y from work_area_mm; z/speed/CMY from schema) for single source of truth.

**Benefit:** No drift; bounds changes propagate automatically.

---

### **4) Spatial feature map: explicit architecture constraint**

**Already corrected in v2.1, but emphasize:**

**Requirement:**

- Backbone **must** return `(B, 512, H', W')` spatial map
- Heatmap head operates on spatial map
- GAP applied **only** for scalar heads (z, v, cmy) and critic
- Test: `assert feat_map.ndim == 4` before heatmap head

**Anti-pattern:** Applying GAP before heatmaps destroys spatial info—avoid.

---

### **5) LPIPS: always FP32, inputs in [-1, 1]**

**Rule:** LPIPS never runs under autocast; inputs always normalized.

**Enforcement:** Normalize to `[-1, 1]`, wrap in `autocast(enabled=False)`, force `.float()`.

**Test:** Verify LPIPS input range is `[-1, 1] ± 0.01`.

---

### **6) Coordinate frames: single transform at G-code boundary**

**Principle:** Internal pipeline stays in **image frame** (top-left, +Y down). Transform to machine frame happens **once** in `gcode_generator.py`.

**Validation:** Corner tests verify:

- Stroke at `(0, 0)` in image → `(0, work_area_mm.y)` in machine (if flip_y=true)
- Stroke at `(work_area_mm.x, work_area_mm.y)` in image → `(work_area_mm.x, 0)` in machine

---

### **7) GRBL header: remove non-GRBL codes**

**Problem:** `M82` is Marlin/RepRap (absolute extrusion); GRBL doesn't recognize it.

**GRBL-safe header:**

```
G21            ; mm units
G90            ; absolute positioning
G92 X0 Y0 Z0   ; set current position as origin (optional)

```

**Air control:** Use coolant M-codes (`M7`/`M8`) if wired that way, or controller-specific macros loaded from machine profile. **Do not** include RepRap-specific codes.

---

### **8) Golden CI precision: match DGX defaults (BF16)**

**Fixed:** Golden test YAMLs use `precision: "bf16"` to match DGX default, with `lpips_tile: {size: 0, overlap: 0}` for full-frame on DGX Spark.

---

### **9) Soft-argmax temperature hyperparameter**

**Enhancement:** Add temperature τ to control peak sharpness.

**Config addition (configs/train.yaml):**

```yaml
agent:
  softargmax_temp: 1.0  # Lower = sharper, higher = smoother

```

**Benefit:** Tuneable via HPO; lower τ for crisp coords, higher for stable early learning. Log to MLflow as hyperparameter.

---

### **10) Reward resolution coupling**

**Principle:** Keep `reward_px == render_px` during training (default). Only decouple for final print evaluation.

**Config enforcement:** Default coupling documented in `airbrush_v1.yaml`.

**Override:** `paint.py --print_res_px W H` can raise reward/render for final eval without retraining.

---

## **Pre-implementation checklist**

Before coding, verify:

- [ ]  Action dimensionality is **15** in: env space, networks (K=4 heatmaps), tests, docs
- [ ]  Env observation space dtype is **`np.float32`**; `_get_obs()` returns `.astype(np.float32)`
- [ ]  Action scale/bias computed from **schema bounds** (no hard-coded 150.5, 30.0, etc.)
- [ ]  Backbone returns **spatial map** `(B, 512, H', W')`; heatmaps applied before GAP
- [ ]  LPIPS always **FP32** with inputs **[-1, 1]**; wrapped in `autocast(enabled=False)`
- [ ]  Coordinate transform applied **once** in `gcode_generator.py`; corner tests pass
- [ ]  GRBL header contains **only** `G21/G90/G92` (no M82 or RepRap codes)
- [ ]  CI golden configs use **`precision: "bf16"`** (or separate FP16 profile documented)
- [ ]  Soft-argmax temperature **logged to MLflow** as hyperparameter
- [ ]  `reward_px == render_px` default coupling documented
- [ ]  **Validation set:** 10 diverse images in `data/validation_images/`; **excluded** from training sets
- [ ]  [**paint.py](http://paint.py/) refactored:** `paint_main()` returns dict; callable without MLflow; CLI preserved
- [ ]  **HPO objective:** trains → validates on 10 images → returns `avg_lpips`
- [ ]  **run_training_subprocess:** returns `(checkpoint_path, mlflow_run_id)`

---

## **Files requiring updates**

### **Configuration**

- `configs/env/airbrush_v1.yaml`: Verify reward_px == render_px default
- `configs/train.yaml`: Add `agent.softargmax_temp: 1.0` and `logging.save_interval: 10`
- `configs/hpo/search_space.v1.yaml`: Add validation section, softargmax_temp range
- `ci/golden_tests/expected/*.yaml`: Change `precision: "bf16"`

### **Source code**

- `src/airbrush_robot_env/env_v1.py`: 15-D action space, FP32 obs, schema-driven scaling
- `src/rl_agent/networks_hd.py`: K=4 heatmaps, spatial map preservation, softargmax_temp
- `src/utils/gcode_generator.py`: GRBL-only header, coordinate transform once
- `src/utils/validators.py`: Add `load_stroke_schema()` helper returning bounds
- `scripts/paint.py`: Refactor to paint_main() callable function
- `scripts/train.py`: Add HPO validation loop, run_training_subprocess enhancements, save_interval callback for GUI monitoring
- `src/gui/`: Full implementation (main_window.py, tabs, widgets) with watchdog integration

### **Tests**

- `test_action_scaling.py`: Verify 15-D, bounds from schema
- `test_coordinate_frames.py`: Corner stroke validation
- `test_spatial_head.py`: Assert feat_map.ndim == 4
- `test_lpips_normalization.py`: Verify [-1,1] input range
- `test_paint_main.py`: Verify callable API, return dict structure
- `test_hpo_validation.py`: Mock validation loop, verify artifact logging
- `test_gui_monitoring.py`: Watchdog triggers, epoch loading, stroke playback rendering

### **Documentation**

- Replace all "8 control points / 21-D" with "4 control points / 15-D"
- Document soft-argmax temperature in HPO search space
- Note GRBL header restrictions (no RepRap codes)
- Add section 7.4 with complete HPO validation workflow
- Add section 9 with decoupled GUI architecture details

Other utils functions : 

# **Utilities Suite – Engineering Plan (Addendum)**

## **0) Executive summary (what this adds)**

Introduce a cohesive set of small, well‑tested utility modules under src/utils/ to centralize:

- **Numerics & tiling** (LPIPS normalization, tiled inference with Hann blending, bounds projection).
- **Color science** (sRGB↔linear, RGB↔Lab, ΔE2000, luminance).
- **Geometry** (cubic Bézier evaluation/flattening, polyline ops).
- **Atomic I/O** (safe file writes, YAML helpers, symlink management).
- **Torch ergonomics** (seeding, channels‑last, device moves, AMP contexts, GPU mem HWM).
- **Metrics** (PSNR/SSIM, coverage, edge preservation).
- **Profiling** (wall‑clock timers, NVTX ranges).
- **Stroke helpers** (stable IDs, vec↔YAML round‑trip, stroke heatmaps).
- **Hashing** (file/tensor SHA‑256 for provenance).
- *(Optional)* **Image helpers** (tensor↔PNG/JPEG round‑trips) if you want to keep I/O generic.

This reduces duplication, improves testability, and yields consistent behavior across training, inference, GUI, calibration, and CI.

---

## **1) Repository changes**

```
src/
  utils/
    compute.py          # numerics, tiling, ranges, bounds, OOM guards (extends existing)
    color.py            # sRGB↔linear, RGB↔Lab, ΔE2000, luminance
    geometry.py         # Bézier, polyline, bbox, length, clipping
    io.py               # atomic writes, YAML load/save, symlinks, path utils
    torch_utils.py      # seeding, device/dtype, channels-last, AMP, GPU mem HWM
    metrics.py          # PSNR, SSIM, coverage, edge preservation
    profiler.py         # timer() and nvtx_range()
    strokes.py          # stroke IDs, vec↔YAML, bbox, heatmaps
    hash.py             # SHA-256 helpers
    # optional
    image_utils.py      # tensor↔image helpers (only if you want to split from io.py)
```

Add corresponding tests in tests/utils_*/*.py (see Section 6).

---

## **2) Conventions & invariants (enforced by utils)**

- **Images (internal):** torch.FloatTensor, shape (3, H, W), **linear RGB in [0,1]** unless explicitly noted (e.g., LPIPS helper outputs **[-1,1]**).
- **LPIPS inputs:** always normalized by compute.normalize_img_for_lpips(); LPIPS computed in **FP32**.
- **Geometry:** all public stroke params in **mm** (schema authority). Pixel conversion only at env/renderer boundaries.
- **Atomic outputs:** GUI‑visible artifacts are written via io.atomic_* + symlink_atomic.
- **No JSON:** YAML is the only on‑disk config/manifest format.

---

## **3) Module specs (APIs you can copy into docstrings)**

### **3.1 src/utils/compute.py (extended)**

**Purpose:** numerics, image ranges, tiling, bounds, OOM guards.

```
# Ranges & normalization
def to_0_1(x: torch.Tensor, src_range: str = "uint8") -> torch.Tensor
def to_minus1_1(x: torch.Tensor, src_range: str = "0_1") -> torch.Tensor
def normalize_img_for_lpips(x: torch.Tensor, colorspace: str = "linear_rgb") -> torch.Tensor
# -> returns [-1,1] FP32; no autocast inside

# Tiling primitives
def hann2d(h: int, w: int, device=None, dtype=None) -> torch.Tensor
def tile_slices(H: int, W: int, tile: int, overlap: int) -> list[tuple[slice, slice]]
def stitch_weight_map(H: int, W: int, tile: int, overlap: int, device=None, dtype=None) -> torch.Tensor
def apply_tiled(fn, img: torch.Tensor, tile: int, overlap: int,
                pad_to_multiple: int = 1, **kwargs) -> torch.Tensor

# Bounds & safety
def project_to_bounds(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor
def clamp_finite(x: torch.Tensor, min_val: float = -1e30, max_val: float = 1e30) -> torch.Tensor
def assert_finite(x: torch.Tensor, name: str = "tensor") -> None

# Memory heuristics
def choose_tile_size_from_mem(H: int, W: int, bytes_per_px: int = 16, mem_margin: float = 0.8) -> int
def retry_cuda_oom(fn, *args, **kwargs) -> any

# Keep your existing:
# - class TiledLPIPS(...)
# - autocast_context(...)
# - mm_to_px(...), px_to_mm(...)
```

**Use:** env (reward), renderer (tiled ops), CI (golden LPIPS), GUI (huge images).

---

### **3.2 src/utils/color.py**

**Purpose:** consistent color conversions & ΔE for calibration/metrics.

```
def srgb_to_linear(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor    # [0,1] sRGB -> linear
def linear_to_srgb(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor    # linear -> [0,1] sRGB
def luminance_linear(img: torch.Tensor) -> torch.Tensor                     # Y from linear RGB
def rgb_to_lab(img_linear_rgb: torch.Tensor) -> torch.Tensor                # (3,H,W) -> (3,H,W)
def delta_e2000(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor     # mean or per-pixel ΔE
```

**Use:** calibration ΔE checks, pen mask thresholding (LAB), diagnostics.

---

### **3.3 src/utils/geometry.py**

**Purpose:** Bézier math & polylines for renderer/G‑code/tests.

```
def bezier_cubic_eval(p1, p2, p3, p4, t: torch.Tensor) -> torch.Tensor          # (...,2)
def bezier_cubic_polyline(p1, p2, p3, p4, max_err_px: float = 0.25) -> torch.Tensor  # (N,2)
def polyline_length(points: torch.Tensor) -> torch.Tensor
def polyline_bbox(points: torch.Tensor) -> tuple[float, float, float, float]    # xmin,ymin,xmax,ymax
def parametrize_by_arclength(points: torch.Tensor) -> torch.Tensor              # s in [0,1]
def clip_polyline_to_rect(points: torch.Tensor, rect_xyxy: tuple) -> torch.Tensor
```

**Use:** stroke rasterization tile culling, G‑code linearization, tests.

---

### **3.4 src/utils/io.py**

**Purpose:** safe file writes, YAML, symlinks.

```
from pathlib import Path
from typing import Any

def ensure_dir(p: Path) -> None
def atomic_write_bytes(path: Path, data: bytes, tmp_suffix: str = ".tmp") -> None
def atomic_save_image(img: "np.ndarray|torch.Tensor", path: Path, pil_kwargs: dict | None = None) -> None
def atomic_yaml_dump(obj: Any, path: Path) -> None
def load_yaml(path: Path) -> dict
def symlink_atomic(target: Path, link_path: Path) -> None
def make_latest_symlink(dir_to_link: Path, latest_link: Path) -> None
def find_latest_epoch(root: Path) -> int | None
```

**Use:** training monitor artifacts, GUI readers, CI outputs, manifests.

---

### **3.5 src/utils/torch_utils.py**

**Purpose:** reproducibility & device ergonomics.

```
def seed_everything(seed: int, deterministic_cudnn: bool = True) -> None
def to_device_recursive(obj, device: torch.device, non_blocking: bool = True)
def set_channels_last(model: torch.nn.Module) -> None
def get_gpu_mem_highwater(reset: bool = True) -> int
def no_autocast(): ...
def autocast_context(precision: str = "bf16"): ...  # alias to your existing
```

**Use:** train.py/paint.py seeding, MLflow mem metrics, model setup.

---

### **3.6 src/utils/metrics.py**

**Purpose:** CI & adversarial tests adjunct metrics.

```
def psnr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor
def ssim(a: torch.Tensor, b: torch.Tensor, window: int = 11) -> torch.Tensor
def paint_coverage(alpha_like: torch.Tensor) -> torch.Tensor           # fraction [0,1]
def edge_preservation_score(target: torch.Tensor, canvas: torch.Tensor) -> torch.Tensor
```

**Use:** golden thresholds, reward‑hack assertions (coverage + edges).

---

### **3.7 src/utils/profiler.py**

**Purpose:** cheap timings + Nsight markers.

```
from contextlib import contextmanager

@contextmanager
def timer(name: str, sink=None):  # sink: Callable[[str, float], None]
    ...

@contextmanager
def nvtx_range(msg: str, color: int = 0x00AAFF):
    ...
```

**Use:** measure Strategist/Technician/render/LPIPS; log to MLflow.

---

### **3.8 src/utils/strokes.py**

**Purpose:** ID, (de)serialization, visualization.

```
import uuid

def make_stroke_id(epoch: int, idx: int) -> str                     # "00012-00345-ab12cd34"
def stroke_vec_to_yaml_dict(vec_mm: torch.Tensor, bounds: dict) -> dict
def stroke_yaml_dict_to_vec(y: dict) -> torch.Tensor
def stroke_bbox_mm(stroke_yaml: dict) -> tuple[float, float, float, float]
def strokes_heatmap_mm(strokes_yaml_path: str, work_area_mm: tuple, H: int, W: int) -> torch.Tensor
```

> Note:
> 
> 
> **single place**
> 
> **normalized**
> 
> **mm**
> 

---

### **3.9 src/utils/hash.py**

**Purpose:** provenance hashes.

```
def sha256_file(path: "str|Path", chunk: int = 1 << 20) -> str
def sha256_tensor(t: torch.Tensor) -> str
```

**Use:** log LUT/dataset/checkpoint digests into MLflow params; cache keys.

---

### **3.10 (Optional) src/utils/image_utils.py**

**Purpose:** tensor↔image (kept small; no heavy deps).

```
def tensor_to_uint8(img_0_1: torch.Tensor) -> "np.ndarray"
def uint8_to_tensor(img_uint8: "np.ndarray") -> torch.Tensor
def resize_tensor_area(img: torch.Tensor, size_hw: tuple[int,int]) -> torch.Tensor
```

---

## **4) Integration points (who calls what)**

- **Env (env_v1.py)**
    - compute.normalize_img_for_lpips before LPIPS; project_to_bounds on actions; choose_tile_size_from_mem if needed; torch_utils.seed_everything.
- **Renderer (differentiable_renderer.py)**
    - geometry.bezier_cubic_polyline (CPU fallback & tests); compute.apply_tiled if you tile PSF later; compute.assert_finite after composites.
- **G‑code (gcode_generator.py)**
    - geometry.bezier_cubic_polyline, strokes.make_stroke_id, io.atomic_yaml_dump.
- **Calibration (data_pipeline/calibrate.py)**
    - color.srgb_to_linear, color.rgb_to_lab, color.delta_e2000, io.load_yaml/atomic_yaml_dump.
- **GUI**
    - io.atomic_save_image, io.symlink_atomic, io.find_latest_epoch, strokes.strokes_heatmap_mm, compute.apply_tiled for giant previews.
- **Training/HPO (scripts/train.py)**
    - torch_utils.seed_everything, torch_utils.get_gpu_mem_highwater, profiler.timer, hash.sha256_file on LUTs & validation set, record into MLflow params.
- **CI**
    - metrics.psnr/ssim, compute.normalize_img_for_lpips, io.atomic_*, hash.sha256_file.

---

## **5) Coding standards for utils**

- **No side effects:** pure functions; no global state.
- **No heavyweight deps:** only torch, numpy, PIL where truly needed (I/O).
- **Docstrings:** clear **I/O ranges, dtype, shape**, example use. Include invariants (e.g., linear RGB in [0,1]).
- **Errors:** raise ValueError/RuntimeError with actionable messages; never silently clamp except in the explicit projection helpers.
- **Determinism:** guard random ops via torch_utils.seed_everything.

---

## **6) Tests to add**

Create focused suites:

```
tests/
  utils_compute_test.py         # tiling parity, hann window properties, projection
  utils_color_test.py           # srgb↔linear roundtrip, ΔE known pairs
  utils_geometry_test.py        # cubic flattening max_err_px behavior, length, bbox
  utils_io_test.py              # atomic writes/symlink under concurrent reads
  utils_torch_utils_test.py     # seeding repeatability, channels-last no-op
  utils_metrics_test.py         # PSNR/SSIM sanity, coverage monotonicity
  utils_strokes_test.py         # vec↔YAML roundtrip, bbox of simple stroke
  utils_hash_test.py            # stable digests for file/tensor
```

**Property tests** (hypothesis or simple fuzz):

- apply_tiled(fn=id) ≈ identity.
- project_to_bounds idempotent (project(project(x))==project(x)).
- Bézier max_err_px ↓ → polyline length converges.

---

## **7) Implementation order (low‑risk)**

1. **Add modules** with docstrings + unit tests.
2. **Refactor call sites**:
    - LPIPS normalization → compute.normalize_img_for_lpips
    - Action projection → compute.project_to_bounds
    - Bézier polyline → geometry.bezier_cubic_polyline
    - Atomic emission of training artifacts → io.atomic_*
    - Seeding + mem HWM logging → torch_utils
3. **Wire MLflow** params:
    - Hashes (hash.sha256_file) of LUTs/validation set
    - GPU mem HWM after key phases
4. **CI**: run utils tests in the same job as existing unit tests.

---

## **8) Risk checklist & mitigations**

- **Range confusion ([0,1] vs [-1,1]):** one public entry—normalize_img_for_lpips()—and unit tests.
- **Mixed precision bleed into LPIPS:** utils force FP32 in LPIPS helper; test verifies dtype.
- **GUI partial reads:** io.atomic_* + symlink_atomic; test simulates read while write.
- **Stroke normalization drift:** keep vec↔YAML conversion in strokes.py; test round‑trip.

---

## **9) Example snippets (copy/paste into call sites)**

**LPIPS in env (reward)**

```
from src.utils import compute

can_r = F.interpolate(canvas.unsqueeze(0), size=(Hr, Wr), mode='area', align_corners=False)
tgt_r = F.interpolate(target.unsqueeze(0), size=(Hr, Wr), mode='area', align_corners=False)
can_lpips = compute.normalize_img_for_lpips(can_r)
tgt_lpips = compute.normalize_img_for_lpips(tgt_r)
with torch.cuda.amp.autocast(enabled=False):
    new = self.lpips(can_lpips.float(), tgt_lpips.float()).mean()
reward = (self.curr_lpips - new).item()
self.curr_lpips = new
```

**Atomic artifact drop (train → GUI)**

```
from pathlib import Path
from src.utils import io

epoch_dir = Path(f"outputs/training_monitor/epoch_{epoch}")
io.ensure_dir(epoch_dir)
io.atomic_save_image(canvas_np, epoch_dir / "canvas.png")
io.atomic_yaml_dump(meta_dict, epoch_dir / "metadata.yaml")
io.symlink_atomic(epoch_dir, Path("outputs/training_monitor/latest"))
```

**Stroke vec → YAML (paint.py)**

```
from src.utils import strokes

vec_mm = action_vec_mm.cpu()  # (15,)
stroke_yaml = strokes.stroke_vec_to_yaml_dict(vec_mm, bounds=schema.bounds)
# append to list and dump via io.atomic_yaml_dump later
```

---

## **File**

src/utils/logging_config.py

## **Purpose**

Provide a single, reliable way to configure logging across all entrypoints with:

- consistent formatting (console & file),
- optional rotation,
- structured fields (run/trial/epoch/etc.),
- JSON output switch (for tooling),
- warnings→logging capture,
- safe, idempotent setup (no duplicate handlers),
- optional multi‑process support (QueueHandler/QueueListener).

---

## **Public API**

```
def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    *,
    json: bool = False,
    color: bool = True,
    to_stderr: bool = True,
    rotate: dict | None = None,   # e.g. {"mode": "size", "max_bytes": 50_000_000, "backup_count": 5}
    tz: str = "UTC",
    capture_warnings: bool = True,
    quiet_libs: list[str] | None = None,
    context: dict | None = None,  # e.g. {"app": "train", "run_id": "...", "trial": 17}
    queue: "multiprocessing.Queue | None" = None,  # optional: mp-safe centralized logging
) -> dict:
    """
    Configure root logger once. Returns a dict with {"handlers": [...], "listener": optional}.
    Idempotent: repeated calls won't stack handlers.
    """

def get_logger(name: str) -> "logging.Logger":
    """Convenience accessor wrapping logging.getLogger(name)."""

def set_level(level: str) -> None:
    """Update root logger level at runtime (e.g., via CLI flag)."""

def push_context(**kw) -> None:
    """Attach contextual fields (e.g., trial=7, epoch=120) to all subsequent log records in this context."""

def pop_context(keys: list[str] | None = None) -> None:
    """Remove contextual fields."""

def install_excepthook() -> None:
    """Log uncaught exceptions via root logger with traceback, then re-raise."""

def route_warnings() -> None:
    """Redirect Python warnings to logging.warning()."""
```

> Context is implemented with contextvars so each thread/process can carry its own fields (e.g., {"app":"train","trial":3}) without global collisions.
> 

---

## **Format & Handlers**

- **Console (stderr)** by default:
    - **Human format** (timestamp, level, app/trial/run if present, message).
    - ANSI colors if color=True and TTY (no external dependency; optional colorlog if installed).
- **File handler** if log_file set:
    - Plain text by default; **JSON lines** if json=True.
    - Optional rotation:
        - rotate={"mode":"size","max_bytes":50_000_000,"backup_count":5} → RotatingFileHandler
        - rotate={"mode":"time","when":"D","interval":1,"backup_count":7} → TimedRotatingFileHandler
- **Queue logging (optional)**:
    - If queue provided, use QueueHandler in children and QueueListener in the main process to avoid interleaved lines across processes (useful for HPO trials or dataloader workers that log).
- **Warnings capture**: warnings.capturewarnings(True) → routing to logging.

**Standard human format** (UTC):

```
2025-10-28T13:45:12.345Z | INFO  | app=train trial=17 epoch=120 | Strategist step=12 Δlpips=0.0042
```

**Standard JSON line**:

```
{"t":"2025-10-28T13:45:12.345Z","lvl":"INFO","name":"train",
 "pid":12345,"msg":"Strategist step","trial":17,"epoch":120,"dlpips":0.0042}
```

---

## **YAML Configuration**

Extend your existing configs/train.yaml (and reuse the same schema for paint.py, calibrate.py, GUI):

```
logging:
  level: "INFO"
  json: false
  color: true
  file: "outputs/logs/train.log"
  rotate:
    mode: "size"          # "size" | "time"
    max_bytes: 50000000   # if mode=size
    backup_count: 5
  capture_warnings: true
  quiet_libs: ["matplotlib","PIL","numba","urllib3","OpenGL"]
  context: { app: "train" }   # extra fields merged into each log record
```

For HPO, add trial/run context dynamically:

- trial context: context={"app":"hpo","trial": trial.number}
- MLflow run_id: context={"run_id": mlflow.active_run().info.run_id}

---

## **Integration Examples**

**train.py**

```
from src.utils import logging_config as logc

def main():
    # Load YAML -> args.logging
    logc.setup_logging(**yaml_cfg.logging, context={"app":"train"})
    logc.install_excepthook()
    # later per-trial:
    logc.push_context(trial=trial.number, run_id=run.info.run_id)
    logger = logc.get_logger(__name__)
    logger.info("Starting trial")
```

**paint.py**

```
logc.setup_logging(log_level="INFO", log_file="outputs/logs/paint.log",
                   rotate={"mode":"time","when":"D","backup_count":7},
                   context={"app":"paint"})
```

**launch_gui.py**

```
logc.setup_logging(log_level="INFO", log_file="outputs/logs/gui.log",
                   json=False, color=True, context={"app":"gui"})
```

**calibrate.py**

```
logc.setup_logging(log_level="INFO", log_file="outputs/logs/calibrate.log",
                   context={"app":"calibrate"})
```

**Multiprocess HPO (optional)**

- Create a multiprocessing.Manager().Queue() in the parent.
- Call setup_logging(queue=q, ...) once in the parent (listener created).
- In child processes, call setup_logging(queue=q, to_stderr=False, log_file=None) so they emit via QueueHandler.

---

## **Tests**

Add tests/utils_logging_test.py:

- **Idempotency:** calling setup_logging() twice doesn’t duplicate handlers.
- **File write:** logs appear in path; rotation creates backups.
- **JSON mode:** entry parses as JSON; includes context fields.
- **Warnings routed:** warnings.warn("x") captured as logging.WARNING.
- **Context:** push_context(trial=3) adds trial=3 to subsequent records.
- **Excepthook:** raise a test exception in a subprocess and assert it logs.
- *(Optional)* **Queue mode:** parent+child processes produce ordered, non-interleaved logs.

---

## **Why this belongs in utils**

- Every entrypoint needs logging; centralizing avoids copy‑paste config sprawl.
- Contextual fields make MLflow/Optuna/GUI logs **navigable** (trial/run/epoch).
- JSON switch future‑proofs ingestion (ELK/Vector/etc.) without touching call sites.
- Queue logging eliminates interleaving in multi‑process HPO or dataloaders.

---