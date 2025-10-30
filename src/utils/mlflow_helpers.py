"""MLflow integration helpers for tracking and artifact management.

Provides:
    - Experiment setup with naming conventions
    - Nested runs for HPO (study → trials)
    - Bulk parameter/metric logging
    - Artifact logging with checksums
    - Run resume/continue logic

Used by:
    - scripts/train.py: Log hyperparams, metrics, checkpoints
    - HPO objective: Nested trial runs with validation metrics
    - CI: Archive golden test results

Tracked entities:
    Parameters:
        - Config (flattened YAML)
        - Resolutions (render_px, obs_px, reward_px)
        - LUT hashes (provenance, via src.utils.hashing)
        - Reproducibility (seeds, cudnn flags)
    Metrics:
        - Training: mean_reward, mean_lpips, gpu_mem_hwm
        - HPO: avg_final_lpips_validation (minimize directly, no negation)
    Artifacts:
        - Checkpoints (.pth)
        - Validation paintings (trial_<N>/*.png)
        - Configs (train.yaml, env.yaml, etc.)

Convention: experiment names use "airbrush_<task>_v<N>" format.
"""

