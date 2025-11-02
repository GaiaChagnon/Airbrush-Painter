"""Training script: PPO training with optional HPO.

Entrypoint for training the Strategist (PPO agent):
    - Standard training: Fixed hyperparameters from train.yaml
    - HPO mode: Optuna study with validation set evaluation

Training flow:
    1. Load and validate configs (train.yaml, env.yaml, sim.yaml)
    2. Setup logging and MLflow tracking
    3. Seed everything for reproducibility
    4. Build environment (AirbrushEnvV1) and agent (networks)
    5. Launch rl-games PPO trainer
    6. Save checkpoints and best_config.yaml
    7. Periodically export training monitor artifacts for GUI

HPO flow (with --hpo flag):
    1. Load search_space.v1.yaml (direction: minimize)
    2. Create Optuna study
    3. For each trial:
        a. Sample hyperparameters
        b. Patch config
        c. Train subprocess on training set → checkpoint
        d. Load checkpoint
        e. Run paint_main on 10 validation images
        f. Compute avg_final_lpips_validation (return avg_lpips directly)
        g. Log to MLflow with artifacts
    4. After all trials, save best_config.yaml

Training monitor artifacts (every save_interval epochs):
    outputs/training_monitor/epoch_{N}/
        - target.png
        - canvas.png
        - strokes.yaml
        - metadata.yaml
    outputs/training_monitor/latest/ → epoch_{N} (symlink)

CLI:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --hpo --trials 80

MLflow tracking:
    - Experiment: airbrush_train_v2 (or airbrush_hpo_v2)
    - Params: config (flattened), resolutions, LUT hashes (via src.utils.hashing), seeds
    - Metrics: mean_reward, mean_lpips, gpu_mem_hwm, avg_final_lpips_validation
    - Artifacts: checkpoints, validation paintings, configs
"""

