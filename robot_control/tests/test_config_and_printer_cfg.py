"""Tests for config loader schema and printer.cfg generation.

Validates that:
    - machine.yaml loads successfully with the current schema
    - Structural invariants hold (types, pin assignments, polarity)
    - Consistency: microsteps * full_steps == driver_pulses_per_rev
    - printer_cfg.generate_printer_cfg produces valid Klipper config
    - Generated config values match machine.yaml (no hardcoded drift)

Tests deliberately avoid hardcoding machine.yaml *tunable* values
(workspace dimensions, speeds, microsteps) so they never break when
the operator edits the config.  Only pin assignments, polarity, and
structural invariants use literal assertions.
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


@pytest.fixture()
def printer_cfg(config: MachineConfig) -> str:
    """Generate printer.cfg text from the loaded config."""
    return generate_printer_cfg(config)


# ---------------------------------------------------------------------------
# Config loader: structural & consistency tests
# ---------------------------------------------------------------------------


class TestLoaderStructure:
    # -- Stepper invariant --------------------------------------------------

    def test_microsteps_times_full_steps(self, config: MachineConfig) -> None:
        s = config.steppers
        assert s.klipper_microsteps * s.full_steps_per_rotation == s.driver_pulses_per_rev

    def test_full_steps_per_rotation_positive(self, config: MachineConfig) -> None:
        assert config.steppers.full_steps_per_rotation > 0

    def test_rotation_distances_positive(self, config: MachineConfig) -> None:
        assert config.steppers.xy_rotation_distance > 0
        assert config.steppers.z_rotation_distance > 0

    # -- Work area ----------------------------------------------------------

    def test_work_area_positive(self, config: MachineConfig) -> None:
        assert config.work_area.x > 0
        assert config.work_area.y > 0
        assert config.work_area.z > 0

    # -- Motion limits ------------------------------------------------------

    def test_motion_limits_positive(self, config: MachineConfig) -> None:
        m = config.motion
        assert m.max_velocity_mm_s > 0
        assert m.max_accel_mm_s2 > 0
        assert m.homing_speed_mm_s > 0
        assert m.z_homing_speed_mm_s > 0
        assert m.square_corner_velocity_mm_s > 0

    def test_z_homing_slower_than_xy(self, config: MachineConfig) -> None:
        m = config.motion
        assert m.z_homing_speed_mm_s <= m.homing_speed_mm_s

    # -- MCU serial ---------------------------------------------------------

    def test_mcu_serial(self, config: MachineConfig) -> None:
        assert "usb-Klipper" in config.connection.mcu_serial

    # -- Pin assignments (hardware, not tunables) ---------------------------

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
        assert config.axes["z"].endstop_polarity == "^!"

    def test_endstop_pins(self, config: MachineConfig) -> None:
        """Verify physical pin assignments (Y=PG9, X=PG6, Z=PG10)."""
        assert config.axes["y"].endstop_pin == "PG9"
        assert config.axes["x"].endstop_pin == "PG6"
        assert config.axes["z"].endstop_pin == "PG10"

    def test_y_homing_side_min(self, config: MachineConfig) -> None:
        assert config.axes["y"].homing_side == "min"

    def test_x_homing_side_min(self, config: MachineConfig) -> None:
        assert config.axes["x"].homing_side == "min"

    def test_z_axis_real(self, config: MachineConfig) -> None:
        """Z axis is a real physical axis with endstop, not a dummy."""
        z = config.axes["z"]
        assert z.endstop_pin == "PG10"
        assert z.endstop_type == "NO_to_GND"
        assert z.homing_side == "max"
        assert z.pins[0].step == "PG0"

    def test_axis_motor_count_matches_pins(self, config: MachineConfig) -> None:
        for name, axis in config.axes.items():
            assert len(axis.pins) == axis.motors, (
                f"Axis '{name}': {axis.motors} motor(s) "
                f"but {len(axis.pins)} pin group(s)"
            )


# ---------------------------------------------------------------------------
# Printer.cfg generation: structure + config<->output agreement
# ---------------------------------------------------------------------------


class TestPrinterCfgGeneration:
    def test_generates_string(self, printer_cfg: str) -> None:
        assert isinstance(printer_cfg, str)
        assert len(printer_cfg) > 100

    def test_cartesian_kinematics(self, printer_cfg: str) -> None:
        assert "kinematics: cartesian" in printer_cfg

    # -- Stepper sections present -------------------------------------------

    def test_stepper_x_section(self, printer_cfg: str) -> None:
        assert "[stepper_x]" in printer_cfg
        assert "step_pin: PF11" in printer_cfg

    def test_stepper_x1_section(self, printer_cfg: str) -> None:
        assert "[stepper_x1]" in printer_cfg
        assert "step_pin: PG4" in printer_cfg
        assert "dir_pin: !PC1" in printer_cfg

    def test_stepper_y_section(self, printer_cfg: str) -> None:
        assert "[stepper_y]" in printer_cfg
        assert "homing_positive_dir: True" in printer_cfg

    def test_stepper_z_section(self, printer_cfg: str) -> None:
        assert "[stepper_z]" in printer_cfg
        assert "homing_positive_dir: True" in printer_cfg

    def test_force_move_section(self, printer_cfg: str) -> None:
        assert "[force_move]" in printer_cfg
        assert "enable_force_move: True" in printer_cfg

    def test_gcode_arcs_section(self, printer_cfg: str) -> None:
        assert "[gcode_arcs]" in printer_cfg

    def test_mcu_serial_in_cfg(self, printer_cfg: str) -> None:
        assert "serial:" in printer_cfg
        assert "usb-Klipper" in printer_cfg

    def test_step_pulse_duration_fixed_point(self, printer_cfg: str) -> None:
        """step_pulse_duration must be decimal, not scientific notation."""
        assert "step_pulse_duration: 0.000005" in printer_cfg
        assert "5e-06" not in printer_cfg
        assert "5e-6" not in printer_cfg

    # -- Config values propagate to printer.cfg correctly -------------------

    def test_microsteps_match_config(
        self, config: MachineConfig, printer_cfg: str,
    ) -> None:
        expected = f"microsteps: {config.steppers.klipper_microsteps}"
        assert expected in printer_cfg

    def test_full_steps_match_config(
        self, config: MachineConfig, printer_cfg: str,
    ) -> None:
        expected = f"full_steps_per_rotation: {config.steppers.full_steps_per_rotation}"
        assert expected in printer_cfg

    def test_endstop_polarity_prefix(
        self, config: MachineConfig, printer_cfg: str,
    ) -> None:
        for axis_name in ("x", "y", "z"):
            ax = config.axes[axis_name]
            expected = f"endstop_pin: {ax.endstop_polarity}{ax.endstop_pin}"
            assert expected in printer_cfg, (
                f"Axis {axis_name}: expected '{expected}' in printer.cfg"
            )

    def test_position_max_matches_work_area(
        self, config: MachineConfig, printer_cfg: str,
    ) -> None:
        """position_max for each stepper matches work_area dimensions."""
        wa = config.work_area
        assert f"position_max: {wa.x:.0f}" in printer_cfg
        assert f"position_max: {wa.y:.0f}" in printer_cfg
        assert f"position_max: {wa.z:.0f}" in printer_cfg

    def test_homing_speed_matches_config(
        self, config: MachineConfig, printer_cfg: str,
    ) -> None:
        m = config.motion
        assert f"homing_speed: {m.homing_speed_mm_s}" in printer_cfg
        assert f"homing_speed: {m.z_homing_speed_mm_s}" in printer_cfg

    def test_motion_params_match_config(
        self, config: MachineConfig, printer_cfg: str,
    ) -> None:
        m = config.motion
        assert f"max_velocity: {m.max_velocity_mm_s:.0f}" in printer_cfg
        assert f"max_accel: {m.max_accel_mm_s2:.0f}" in printer_cfg
        assert f"square_corner_velocity: {m.square_corner_velocity_mm_s}" in printer_cfg

    def test_y_position_endstop_at_zero(
        self, config: MachineConfig, printer_cfg: str,
    ) -> None:
        """Y homes to min, so position_endstop == 0."""
        assert "position_endstop: 0" in printer_cfg

    def test_z_position_endstop_equals_work_area(
        self, config: MachineConfig, printer_cfg: str,
    ) -> None:
        """Z homes to max, so position_endstop == work_area.z."""
        assert f"position_endstop: {config.work_area.z:.0f}" in printer_cfg
