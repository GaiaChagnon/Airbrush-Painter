"""Airbrush Painter: AI-powered robotic painting system.

This package contains the core modules for training an RL agent to control
a robotic airbrush system, simulating paint physics, and generating G-code
for physical execution.

Architecture layers (strict one-way dependency):
    scripts/ → src/{gui,rl_agent,airbrush_robot_env,airbrush_simulator,data_pipeline}/ → src/utils/

Key invariants:
    - Multi-resolution: render_px (physics), obs_px (policy), reward_px (LPIPS)
    - Geometry in millimeters end-to-end
    - Fixed stroke cap (e.g., 1500) with pure quality reward (LPIPS improvement only)
    - YAML-only configs, no JSON
    - All images are linear RGB [0,1] unless explicitly noted
"""

__version__ = "2.3.0"

