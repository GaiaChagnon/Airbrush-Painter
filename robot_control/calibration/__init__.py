"""
Calibration module.

Provides test pattern generators and interactive calibration routines for
machine setup and verification.
"""

from robot_control.calibration.patterns import (
    square,
    rectangle,
    cross,
    grid,
    circle,
    ruler_x,
    ruler_y,
    crosshair_grid,
    speed_test,
    backlash_test,
    z_touch_pattern,
)
from robot_control.calibration.routines import (
    calibrate_steps_per_mm,
    calibrate_tool_offset,
    calibrate_z_seesaw,
    calibrate_speed,
    verify_endstops,
)

__all__ = [
    "square",
    "rectangle",
    "cross",
    "grid",
    "circle",
    "ruler_x",
    "ruler_y",
    "crosshair_grid",
    "speed_test",
    "backlash_test",
    "z_touch_pattern",
    "calibrate_steps_per_mm",
    "calibrate_tool_offset",
    "calibrate_z_seesaw",
    "calibrate_speed",
    "verify_endstops",
]
