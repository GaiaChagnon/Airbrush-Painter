"""Test AirbrushEnvV1 environment correctness.

Tests for src.airbrush_robot_env.env_v1:
    - Observation space shape (9, obs_h, obs_w) matches obs_px
    - Observation dtype is float32 (rl-games compatibility)
    - Action space shape is (15,) in [-1,1]
    - Canvas/target stored at render_px
    - Reward computed at reward_px
    - Deterministic reset under fixed seed
    - Reward sign (improvement → positive)
    - Termination at stroke_cap
    - Action denormalization: [-1,1] → mm-space

Test cases:
    - test_obs_space_shape_matches_obs_px()
    - test_obs_dtype_is_float32()
    - test_action_space_normalized()
    - test_canvas_stored_at_render_px()
    - test_reward_computed_at_reward_px()
    - test_deterministic_reset()
    - test_reward_improvement_positive()
    - test_termination_at_stroke_cap()
    - test_action_denormalization()

Fixtures:
    - minimal_env_config.yaml (small resolutions for speed)

Run:
    pytest tests/test_env_v1.py -v
"""

