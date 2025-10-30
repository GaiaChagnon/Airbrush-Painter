"""Test multi-resolution architecture in environment.

Validates that three resolutions are independent and configurable:
    - render_px: Canvas/target storage
    - obs_px: Observation returned to policy
    - reward_px: LPIPS scoring grid

Test cases:
    - test_obs_shape_matches_obs_px()
    - test_canvas_storage_at_render_px()
    - test_reward_computed_at_reward_px()
    - test_resolutions_independent()
    - test_resolution_override_from_config()

Test matrix:
    - render_px=908×1280, obs_px=454×640, reward_px=908×1280 (default)
    - render_px=1816×2560, obs_px=454×640, reward_px=1816×2560 (HD)
    - render_px=454×640, obs_px=227×320, reward_px=454×640 (debug)

Assertions:
    - obs.shape == (9, obs_h, obs_w)
    - env.canvas.shape == (3, render_h, render_w)
    - LPIPS computed on (reward_h, reward_w) tensors

Run:
    pytest tests/test_env_resolutions.py -v
"""

