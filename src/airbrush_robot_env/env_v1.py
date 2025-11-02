"""AirbrushEnvV1: Multi-resolution RL environment.

Gym environment for training PPO agent to control robotic airbrush:
    - Observation space: Box(0,1, shape=(9, obs_h, obs_w), dtype=float32)
        * 9 channels: target(3) + canvas(3) + |error|(3)
        * Downsampled to obs_px for policy speed
    - Action space: Box(-1,1, shape=(15,), dtype=float32)
        * Normalized [-1,1]^15 for training stability
        * Denormalized to mm-space via schema-driven scaling
        * 15-D: 4 Bézier control points (8 xy) + z0,z1 (2) + v0,v1 (2) + c,m,y (3)
    - Reward: Dense LPIPS improvement computed at reward_px
    - Termination: step_count >= stroke_cap

Public API:
    env = AirbrushEnvV1(cfg)
    obs = env.reset()  # → (9, obs_h, obs_w) float32
    obs, reward, done, info = env.step(action_normalized)

Multi-resolution flow:
    1. Store canvas/target at render_px (physics fidelity)
    2. Render stroke at render_px (no grad needed for env)
    3. Resample to reward_px for LPIPS scoring
    4. Downsample to obs_px for policy input

Action denormalization:
    scale = (bounds_high - bounds_low) / 2.0
    bias = (bounds_high + bounds_low) / 2.0
    action_mm = action_normalized * scale + bias

LPIPS normalization:
    canvas/target [0,1] → [-1,1] via normalize_img_for_lpips()
    Computed in FP32 (no autocast) for numerical accuracy

Coordinate frame:
    Internal: image frame (top-left origin, +Y down)
    G-code: machine frame (bottom-left, +Y up)
    Transform happens once in gcode_generator, not here
"""

