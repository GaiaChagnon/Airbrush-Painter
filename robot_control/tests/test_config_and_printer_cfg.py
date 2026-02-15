"""Tests for config loader schema and printer.cfg generation.

Validates that:
    - machine.yaml loads successfully with the updated schema
    - New fields (full_steps_per_rotation, pins, endstop_polarity,
      square_corner_velocity, mcu_serial) are parsed correctly
    - Validation catches mismatched microsteps / full_steps / pulses
    - printer_cfg.generate_printer_cfg produces valid Klipper config
    - Generated config contains all required sections and values
"""

from __future__ import annotations

import pytest

from robot_control.configs.loader import (
    ConfigError,
    MachineConfig,
    PinConfig,
    load_config,
)
from robot_control.configs.printer_cfg import generate_printer_cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> MachineConfig:
    """Load the default machine.yaml shipped with the package."""
    return load_config()


# ---------------------------------------------------------------------------
# Config loader: new schema fields
# ---------------------------------------------------------------------------


class TestLoaderNewFields:
    def test_full_steps_per_rotation(self, config: MachineConfig) -> None:
        assert config.steppers.full_steps_per_rotation == 400

    def test_microsteps_times_full_steps(self, config: MachineConfig) -> None:
        s = config.steppers
        assert s.klipper_microsteps * s.full_steps_per_rotation == s.driver_pulses_per_rev

    def test_square_corner_velocity(self, config: MachineConfig) -> None:
        assert config.motion.square_corner_velocity_mm_s == 8.0

    def test_mcu_serial(self, config: MachineConfig) -> None:
        assert "usb-Klipper" in config.connection.mcu_serial

    def test_axis_pins_y(self, config: MachineConfig) -> None:
        y = config.axes["y"]
        assert len(y.pins) == 1
        assert isinstance(y.pins[0], PinConfig)
        assert y.pins[0].step == "PF13"

    def test_axis_pins_x_dual(self, config: MachineConfig) -> None:
        x = config.axes["x"]
        assert len(x.pins) == 2
        assert x.pins[0].step == "PF11"
        assert x.pins[1].dir == "!PC1"

    def test_endstop_polarity(self, config: MachineConfig) -> None:
        assert config.axes["x"].endstop_polarity == "^!"
        assert config.axes["y"].endstop_polarity == "^!"

    def test_endstop_pins_swapped(self, config: MachineConfig) -> None:
        """Verify Y=PG9 (STOP_1), X=PG6 (STOP_0) -- the bring-up result."""
        assert config.axes["y"].endstop_pin == "PG9"
        assert config.axes["x"].endstop_pin == "PG6"

    def test_y_homing_side_max(self, config: MachineConfig) -> None:
        assert config.axes["y"].homing_side == "max"

    def test_x_homing_side_min(self, config: MachineConfig) -> None:
        assert config.axes["x"].homing_side == "min"

    def test_work_area_200x200x80(self, config: MachineConfig) -> None:
        assert config.work_area.x == 200.0
        assert config.work_area.y == 200.0
        assert config.work_area.z == 80.0

    def test_motion_limits(self, config: MachineConfig) -> None:
        assert config.motion.max_velocity_mm_s == 200.0
        assert config.motion.max_accel_mm_s2 == 1500.0
        assert config.motion.homing_speed_mm_s == 16.0
        assert config.motion.z_homing_speed_mm_s == 10.0

    def test_z_axis_real(self, config: MachineConfig) -> None:
        """Z axis is a real physical axis with endstop, not a dummy."""
        z = config.axes["z"]
        assert z.endstop_pin == "PG10"
        assert z.endstop_polarity == "^!"
        assert z.endstop_type == "NO_to_GND"
        assert z.homing_side == "max"
        assert z.pins[0].step == "PG0"

    def test_z_rotation_distance(self, config: MachineConfig) -> None:
        assert config.steppers.z_rotation_distance == 64.0

    def test_axis_motor_count_matches_pins(self, config: MachineConfig) -> None:
        for name, axis in config.axes.items():
            assert len(axis.pins) == axis.motors, (
                f"Axis '{name}': {axis.motors} motor(s) "
                f"but {len(axis.pins)} pin group(s)"
            )


# ---------------------------------------------------------------------------
# Printer.cfg generation
# ---------------------------------------------------------------------------


class TestPrinterCfgGeneration:
    def test_generates_string(self, config: MachineConfig) -> None:
        result = generate_printer_cfg(config)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_cartesian_kinematics(self, config: MachineConfig) -> None:
        result = generate_printer_cfg(config)
        assert "kinematics: cartesian" in result

    def test_stepper_x_section(self, config: MachineConfig) -> None:
        result = generate_printer_cfg(config)
        assert "[stepper_x]" in result
        assert "step_pin: PF11" in result

    def test_stepper_x1_section(self, config: MachineConfig) -> None:
        """stepper_x1 is the secondary X motor auto-synced to the X rail."""
        result = generate_printer_cfg(config)
        assert "[stepper_x1]" in result
        assert "step_pin: PG4" in result
        assert "dir_pin: !PC1" in result

    def test_stepper_y_section(self, config: MachineConfig) -> None:
        result = generate_printer_cfg(config)
        assert "[stepper_y]" in result
        assert "homing_positive_dir: True" in result
        assert "position_endstop: 200" in result

    def test_stepper_z_real(self, config: MachineConfig) -> None:
        """Z is a real axis with physical endstop, homing to max.

        Klipper hard limits are 0..80 (position_endstop must be inside
        this range).  The tighter 5..75 mm soft limits are enforced in
        the test script, not in printer.cfg.
        """
        result = generate_printer_cfg(config)
        assert "[stepper_z]" in result
        assert "endstop_pin: ^!PG10" in result
        assert "position_endstop: 80" in result
        assert "position_min: 0" in result
        assert "position_max: 80" in result
        assert "homing_speed: 10.0" in result
        assert "homing_positive_dir: True" in result

    def test_force_move_section(self, config: MachineConfig) -> None:
        result = generate_printer_cfg(config)
        assert "[force_move]" in result
        assert "enable_force_move: True" in result

    def test_gcode_arcs_section(self, config: MachineConfig) -> None:
        result = generate_printer_cfg(config)
        assert "[gcode_arcs]" in result

    def test_mcu_serial(self, config: MachineConfig) -> None:
        result = generate_printer_cfg(config)
        assert "serial:" in result
        assert "usb-Klipper" in result

    def test_step_pulse_duration_fixed_point(self, config: MachineConfig) -> None:
        """step_pulse_duration must be decimal, not scientific notation."""
        result = generate_printer_cfg(config)
        assert "step_pulse_duration: 0.000005" in result
        assert "5e-06" not in result
        assert "5e-6" not in result

    def test_microsteps_16(self, config: MachineConfig) -> None:
        result = generate_printer_cfg(config)
        assert "microsteps: 16" in result

    def test_full_steps_per_rotation(self, config: MachineConfig) -> None:
        result = generate_printer_cfg(config)
        assert "full_steps_per_rotation: 400" in result

    def test_endstop_polarity_prefix(self, config: MachineConfig) -> None:
        result = generate_printer_cfg(config)
        assert "endstop_pin: ^!PG6" in result   # X endstop
        assert "endstop_pin: ^!PG9" in result   # Y endstop
        assert "endstop_pin: ^!PG10" in result  # Z endstop

    def test_motion_params(self, config: MachineConfig) -> None:
        result = generate_printer_cfg(config)
        assert "max_velocity: 200" in result
        assert "max_accel: 1500" in result
        assert "square_corner_velocity: 8.0" in result
