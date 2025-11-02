"""RL environment (rl-games compatible) with multi-resolution architecture.

Provides Gym-style environment for PPO training:
    - Observation: (Target, Canvas, Error) stacked, downsampled to obs_px
    - Action: Normalized [-1,1]^15 stroke parameters
    - Reward: LPIPS improvement (dense, per-step)
    - Termination: Fixed stroke cap (e.g., 1500)

Multi-resolution architecture:
    - render_px: Physics grid (canvas/target storage, e.g., 908×1280)
    - obs_px: Policy input (downsampled for speed, e.g., 454×640)
    - reward_px: LPIPS scoring (typically == render_px for training)

All resolutions configurable in configs/env/airbrush_v1.yaml.

Modules:
    - env_v1: AirbrushEnvV1 class

Invariants:
    - Observations returned as FP32 numpy (rl-games compatibility)
    - Actions denormalized: [-1,1] → mm-space using schema bounds
    - Reward = LPIPS(old) - LPIPS(new) computed at reward_px
    - No time/ink penalties (pure quality objective)
"""

