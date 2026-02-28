"""Configuration loader for robot control.

Loads and validates ``machine.yaml`` into typed, frozen dataclasses.
All hardware values (stepper params, axis mapping, feed rates, travel
limits) come from the config -- nothing is hardcoded.

Feed rates are stored in **mm/s** throughout Python.  Conversion to the
G-code ``F`` parameter (mm/min) happens only in the G-code generator.

Usage::

    from robot_control.configs.loader import load_config
    cfg = load_config()                       # default path
    cfg = load_config("/custom/machine.yaml") # explicit path
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.fs import load_yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when configuration validation fails."""

    pass


# ---------------------------------------------------------------------------
# Dataclasses -- mirror the YAML structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectionConfig:
    """Klipper API connection settings."""

    socket_path: str
    timeout_s: float
    reconnect_attempts: int
    reconnect_interval_s: float
    auto_reconnect: bool
    mcu_serial: str = ""


@dataclass(frozen=True)
class SoftLimitsConfig:
    """Z-axis soft limits in mm.

    ``z_max`` is the normal operational ceiling (usually == endstop).
    ``z_overtravel_mm`` is the extra travel past the endstop that Klipper
    allows (used during calibration probing).  Klipper's ``position_max``
    is set to ``z_max + z_overtravel_mm``.
    """

    z_min: float
    z_max: float
    z_overtravel_mm: float = 0.0

    @property
    def z_max_with_overtravel(self) -> float:
        """Absolute Z ceiling including calibration overtravel."""
        return self.z_max + self.z_overtravel_mm


@dataclass(frozen=True)
class WorkAreaConfig:
    """Machine work-area dimensions and soft limits in mm."""

    x: float
    y: float
    z: float
    soft_limits: SoftLimitsConfig


@dataclass(frozen=True)
class CanvasConfig:
    """Canvas position and size within the work area (mm)."""

    offset_x_mm: float
    offset_y_mm: float
    width_mm: float
    height_mm: float


@dataclass(frozen=True)
class ZStatesConfig:
    """Z-axis seesaw states in mm."""

    travel_mm: float
    pen_work_mm: float
    airbrush_work_mm: float


@dataclass(frozen=True)
class ToolConfig:
    """Per-tool configuration.  All feeds in mm/s."""

    xy_offset_mm: tuple[float, float]
    feed_mm_s: float
    travel_feed_mm_s: float
    plunge_feed_mm_s: float
    spray_height_mm: float = 0.0


@dataclass(frozen=True)
class SteppersConfig:
    """Stepper / driver / belt / pulley hardware spec."""

    motor_type: str
    full_steps_per_rotation: int
    driver: str
    driver_pulses_per_rev: int
    wiring: str
    enable_pin_inverted: bool
    step_pulse_duration_s: float
    direction_reversal_pause_s: float

    klipper_microsteps: int
    xy_rotation_distance: float
    z_rotation_distance: float | None

    belt_type: str
    belt_pitch_mm: float
    pulley_teeth: int
    pulley_bore_mm: float


@dataclass(frozen=True)
class PinConfig:
    """Step/dir/enable pin assignment for one stepper."""

    step: str
    dir: str
    enable: str


@dataclass(frozen=True)
class AxisConfig:
    """Single-axis hardware mapping with pin assignments.

    For a dual-motor axis (e.g. X), ``pins`` is a tuple of two
    ``PinConfig`` objects.  For a single-motor axis, it is a tuple
    of one.
    """

    octopus_slot: str
    motors: int
    pins: tuple[PinConfig, ...]
    endstop_pin: str | None
    endstop_polarity: str | None
    endstop_type: str | None
    homing_side: str | None


@dataclass(frozen=True)
class MotionConfig:
    """Motion-planner limits."""

    max_velocity_mm_s: float
    max_accel_mm_s2: float
    square_corner_velocity_mm_s: float
    homing_speed_mm_s: float
    z_homing_speed_mm_s: float
    junction_deviation_mm: float
    z_second_homing_speed_mm_s: float = 0.0
    z_homing_retract_mm: float = 0.0
    idle_timeout_s: float = 30.0


@dataclass(frozen=True)
class InteractiveConfig:
    """Interactive-mode (jog / TUI) settings."""

    jog_increments_mm: tuple[float, ...]
    default_jog_increment_mm: float
    position_poll_interval_ms: int


@dataclass(frozen=True)
class FileExecutionConfig:
    """File-run (virtual_sdcard) settings."""

    gcode_directory: str


@dataclass(frozen=True)
class BedMeshConfig:
    """Klipper ``[bed_mesh]`` parameters for manual surface leveling.

    Parameters
    ----------
    mesh_min : tuple[float, float]
        XY of the first probe point in mm (canvas near corner).
    mesh_max : tuple[float, float]
        XY of the last probe point in mm (canvas far corner).
    probe_count : tuple[int, int]
        Grid dimensions ``(X, Y)``.  3x3 is the default.
    horizontal_move_z : float
        Z height (mm) for pen-retracted travel between probe points.
    speed : float
        XY travel speed between probe points in mm/s.
    mesh_pps : tuple[int, int]
        Interpolation points per mesh segment ``(X, Y)``.
    algorithm : str
        Interpolation algorithm: ``"lagrange"`` or ``"bicubic"``.
    calibrated_points : list[list[float]] | None
        2-D array of Z offsets (rows = Y, cols = X) written by the
        calibration routine.  ``None`` when not yet calibrated.
    """

    mesh_min: tuple[float, float]
    mesh_max: tuple[float, float]
    probe_count: tuple[int, int]
    horizontal_move_z: float
    speed: float
    mesh_pps: tuple[int, int]
    algorithm: str
    calibrated_points: list[list[float]] | None = None


# ---------------------------------------------------------------------------
# Pump configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PumpStepperConfig:
    """Motor / driver / lead-screw spec shared by all pump steppers.

    Parameters
    ----------
    motor_type : str
        Descriptive label, e.g. ``"1.8deg"``.
    full_steps_per_rotation : int
        Native full steps per revolution (200 for 1.8 deg motor).
    driver_pulses_per_rev : int
        DIP-switch micro-step setting on the driver.
    klipper_microsteps : int
        Klipper micro-step divisor.  Must satisfy
        ``full_steps_per_rotation * klipper_microsteps == driver_pulses_per_rev``.
    rotation_distance : float
        Lead-screw travel per revolution in mm.
    step_pulse_duration_s : float
        Minimum step pulse width required by the driver.
    enable_pin_inverted : bool
        True if the enable signal is active-low.
    direction_reversal_pause_s : float
        Dwell time in seconds between rapid direction changes.
    """

    motor_type: str
    full_steps_per_rotation: int
    driver_pulses_per_rev: int
    klipper_microsteps: int
    rotation_distance: float
    step_pulse_duration_s: float
    enable_pin_inverted: bool
    direction_reversal_pause_s: float


@dataclass(frozen=True)
class SyringeConfig:
    """Syringe geometry: volume capacity and plunger stroke.

    Parameters
    ----------
    volume_ml : float
        Total syringe capacity in ml.
    plunger_travel_mm : float
        Full plunger stroke from retracted to fully extended, in mm.
    """

    volume_ml: float
    plunger_travel_mm: float

    @property
    def volume_per_mm(self) -> float:
        """Dispensed volume per mm of plunger travel (ml/mm)."""
        return self.volume_ml / self.plunger_travel_mm

    @property
    def mm_per_ml(self) -> float:
        """Plunger travel per ml of dispensed volume (mm/ml)."""
        return self.plunger_travel_mm / self.volume_ml


@dataclass(frozen=True)
class PumpMotorConfig:
    """Per-pump motor wiring, endstop, homing, and speed config.

    Parameters
    ----------
    octopus_slot : str
        Board connector label, e.g. ``"Motor 3"``.
    pins : PinConfig
        Step / dir / enable pin assignments.
    endstop_pin : str
        GPIO pin for the limit switch.
    endstop_polarity : str
        Klipper polarity prefix, e.g. ``"^!"``.
    endstop_type : str
        Switch wiring description, e.g. ``"NO_to_GND"``.
    homing_direction : int
        ``+1`` if the limit switch is in the positive travel direction,
        ``-1`` if it is in the negative direction.
    homing_speed_mm_s : float
        Speed when homing toward the limit switch, in mm/s.
    home_backoff_mm : float
        Distance to back off from the switch after homing, in mm.
    max_dispense_speed_mm_s : float
        Maximum plunger push speed, in mm/s.
    max_retract_speed_mm_s : float
        Maximum plunger pull speed, in mm/s.
    syringe : SyringeConfig
        Syringe geometry (may override the shared default).
    """

    octopus_slot: str
    pins: PinConfig
    endstop_pin: str
    endstop_polarity: str
    endstop_type: str
    homing_direction: int
    homing_speed_mm_s: float
    home_backoff_mm: float
    max_dispense_speed_mm_s: float
    max_retract_speed_mm_s: float
    syringe: SyringeConfig


@dataclass(frozen=True)
class PumpsConfig:
    """Top-level pump subsystem configuration.

    Parameters
    ----------
    enabled : bool
        Whether the pump subsystem is active.
    stepper : PumpStepperConfig
        Shared motor/driver specs for all pump steppers.
    syringe_defaults : SyringeConfig
        Default syringe geometry (overridable per pump).
    motors : dict[str, PumpMotorConfig]
        Per-pump motor configs keyed by pump identifier.
    """

    enabled: bool
    stepper: PumpStepperConfig
    syringe_defaults: SyringeConfig
    motors: dict[str, PumpMotorConfig]


@dataclass(frozen=True)
class ServoConfig:
    """Servo motor hardware parameters.

    Parameters
    ----------
    name : str
        Klipper servo identifier (used in ``SET_SERVO SERVO=<name>``).
    pin : str
        MCU output pin (PWM-capable), e.g. ``"PB6"``.
    angle_range_deg : float
        Total mechanical travel in degrees (e.g. 270 for a 270-degree servo).
    min_pulse_width_s : float
        Pulse width at 0 degrees, in seconds (e.g. 0.0005 = 500 us).
    max_pulse_width_s : float
        Pulse width at max angle, in seconds (e.g. 0.0025 = 2500 us).
    neutral_pulse_width_s : float
        Pulse width for the neutral / centre position, in seconds
        (e.g. 0.0015 = 1500 us).
    """

    name: str
    pin: str
    angle_range_deg: float
    min_pulse_width_s: float
    max_pulse_width_s: float
    neutral_pulse_width_s: float

    @property
    def neutral_angle_deg(self) -> float:
        """Angle corresponding to the neutral pulse width."""
        pulse_range = self.max_pulse_width_s - self.min_pulse_width_s
        frac = (
            (self.neutral_pulse_width_s - self.min_pulse_width_s)
            / pulse_range
        )
        return frac * self.angle_range_deg


@dataclass(frozen=True)
class MachineConfig:
    """Complete machine configuration loaded from ``machine.yaml``.

    All linear dimensions are in **millimeters**.
    All feed rates are in **mm/s**.
    """

    connection: ConnectionConfig
    work_area: WorkAreaConfig
    canvas: CanvasConfig
    z_states: ZStatesConfig
    tools: dict[str, ToolConfig]
    steppers: SteppersConfig
    axes: dict[str, AxisConfig]
    motion: MotionConfig
    interactive: InteractiveConfig
    file_execution: FileExecutionConfig
    pumps: PumpsConfig | None = None
    bed_mesh: BedMeshConfig | None = None
    servo: ServoConfig | None = None
    endstop_phase_enabled: bool = False

    # -- Convenience helpers ------------------------------------------------

    def get_tool(self, name: str) -> ToolConfig:
        """Return tool config or raise ``ConfigError``."""
        if name not in self.tools:
            raise ConfigError(
                f"Unknown tool '{name}'. Available: {list(self.tools.keys())}"
            )
        return self.tools[name]

    def canvas_to_machine(
        self, x: float, y: float, tool: str = "pen",
    ) -> tuple[float, float]:
        """Convert canvas (top-left origin, +Y down) to machine coords.

        Parameters
        ----------
        x, y : float
            Canvas position in mm.
        tool : str
            Tool name for XY offset.

        Returns
        -------
        tuple[float, float]
            Absolute machine X, Y in mm.
        """
        tool_cfg = self.get_tool(tool)

        # Canvas offset (canvas origin -> machine origin)
        mx = x + self.canvas.offset_x_mm
        # Y-flip: image top-left -> machine bottom-left
        my = self.canvas.offset_y_mm + (self.canvas.height_mm - y)

        # Tool XY offset
        mx += tool_cfg.xy_offset_mm[0]
        my += tool_cfg.xy_offset_mm[1]
        return mx, my

    def get_z_for_tool(self, tool: str, state: str) -> float:
        """Return Z position for *tool* in *state* ('travel' or 'work')."""
        if state == "travel":
            return self.z_states.travel_mm
        if state == "work":
            if tool == "pen":
                return self.z_states.pen_work_mm
            if tool == "airbrush":
                return self.z_states.airbrush_work_mm
            raise ConfigError(f"Unknown tool '{tool}' for Z work state")
        raise ConfigError(
            f"Unknown Z state '{state}'. Expected 'travel' or 'work'"
        )


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------


def _parse_tool(name: str, data: dict[str, Any]) -> ToolConfig:
    """Parse a single tool section from raw YAML dict."""
    xy_offset = data.get("xy_offset_mm", [0.0, 0.0])
    if not isinstance(xy_offset, (list, tuple)) or len(xy_offset) != 2:
        raise ConfigError(
            f"Tool '{name}' xy_offset_mm must be a 2-element list, "
            f"got {xy_offset!r}"
        )
    return ToolConfig(
        xy_offset_mm=(float(xy_offset[0]), float(xy_offset[1])),
        feed_mm_s=float(data["feed_mm_s"]),
        travel_feed_mm_s=float(data["travel_feed_mm_s"]),
        plunge_feed_mm_s=float(data["plunge_feed_mm_s"]),
        spray_height_mm=float(data.get("spray_height_mm", 0.0)),
    )


def _parse_steppers(data: dict[str, Any]) -> SteppersConfig:
    """Parse the ``steppers`` section."""
    return SteppersConfig(
        motor_type=str(data["motor_type"]),
        full_steps_per_rotation=int(data["full_steps_per_rotation"]),
        driver=str(data["driver"]),
        driver_pulses_per_rev=int(data["driver_pulses_per_rev"]),
        wiring=str(data["wiring"]),
        enable_pin_inverted=bool(data["enable_pin_inverted"]),
        step_pulse_duration_s=float(data["step_pulse_duration_s"]),
        direction_reversal_pause_s=float(data["direction_reversal_pause_s"]),
        klipper_microsteps=int(data["klipper_microsteps"]),
        xy_rotation_distance=float(data["xy_rotation_distance"]),
        z_rotation_distance=(
            float(data["z_rotation_distance"])
            if data.get("z_rotation_distance") is not None
            else None
        ),
        belt_type=str(data["belt_type"]),
        belt_pitch_mm=float(data["belt_pitch_mm"]),
        pulley_teeth=int(data["pulley_teeth"]),
        pulley_bore_mm=float(data["pulley_bore_mm"]),
    )


def _parse_pins(raw: Any) -> tuple[PinConfig, ...]:
    """Parse axis pin assignments.

    Accepts a single dict (one motor) or a list of dicts (dual motor).
    """
    if isinstance(raw, dict):
        return (PinConfig(
            step=str(raw["step"]),
            dir=str(raw["dir"]),
            enable=str(raw["enable"]),
        ),)
    if isinstance(raw, list):
        return tuple(
            PinConfig(
                step=str(p["step"]),
                dir=str(p["dir"]),
                enable=str(p["enable"]),
            )
            for p in raw
        )
    raise ConfigError(f"axis pins must be a dict or list, got {type(raw)}")


def _parse_syringe(data: dict[str, Any]) -> SyringeConfig:
    """Parse a syringe specification from raw YAML dict."""
    return SyringeConfig(
        volume_ml=float(data["volume_ml"]),
        plunger_travel_mm=float(data["plunger_travel_mm"]),
    )


def _parse_pump_stepper(data: dict[str, Any]) -> PumpStepperConfig:
    """Parse the ``pumps.stepper`` section."""
    return PumpStepperConfig(
        motor_type=str(data["motor_type"]),
        full_steps_per_rotation=int(data["full_steps_per_rotation"]),
        driver_pulses_per_rev=int(data["driver_pulses_per_rev"]),
        klipper_microsteps=int(data["klipper_microsteps"]),
        rotation_distance=float(data["rotation_distance"]),
        step_pulse_duration_s=float(data["step_pulse_duration_s"]),
        enable_pin_inverted=bool(data.get("enable_pin_inverted", False)),
        direction_reversal_pause_s=float(
            data.get("direction_reversal_pause_s", 0.3)
        ),
    )


def _parse_homing_direction(pump_name: str, data: dict[str, Any]) -> int:
    """Parse and validate ``homing_direction`` (+1 or -1)."""
    raw = data.get("homing_direction", 1)
    val = int(raw)
    if val not in (1, -1):
        raise ConfigError(
            f"Pump '{pump_name}' homing_direction must be +1 or -1, "
            f"got {val}"
        )
    return val


def _parse_pump_motor(
    name: str,
    data: dict[str, Any],
    syringe_defaults: SyringeConfig,
) -> PumpMotorConfig:
    """Parse a single pump motor entry from ``pumps.motors``."""
    raw_pins = data.get("pins")
    if raw_pins is None:
        raise ConfigError(f"Pump '{name}' is missing 'pins' field")
    pins = PinConfig(
        step=str(raw_pins["step"]),
        dir=str(raw_pins["dir"]),
        enable=str(raw_pins["enable"]),
    )
    syringe_override = data.get("syringe")
    syringe = (
        _parse_syringe(syringe_override)
        if syringe_override
        else syringe_defaults
    )
    return PumpMotorConfig(
        octopus_slot=str(data["octopus_slot"]),
        pins=pins,
        endstop_pin=str(data["endstop_pin"]),
        endstop_polarity=str(data.get("endstop_polarity", "^!")),
        endstop_type=str(data.get("endstop_type", "NO_to_GND")),
        homing_direction=_parse_homing_direction(name, data),
        homing_speed_mm_s=float(data.get("homing_speed_mm_s", 2.0)),
        home_backoff_mm=float(data.get("home_backoff_mm", 0.5)),
        max_dispense_speed_mm_s=float(
            data.get("max_dispense_speed_mm_s", 5.0)
        ),
        max_retract_speed_mm_s=float(
            data.get("max_retract_speed_mm_s", 8.0)
        ),
        syringe=syringe,
    )


def _parse_pumps(data: dict[str, Any]) -> PumpsConfig | None:
    """Parse the top-level ``pumps`` section.

    Returns ``None`` when pumps are disabled or absent.
    """
    if not data or not data.get("enabled", False):
        return None

    stepper = _parse_pump_stepper(data["stepper"])
    syringe_defaults = _parse_syringe(data["syringe"])

    motors_data = data.get("motors", {})
    motors = {
        name: _parse_pump_motor(name, motor_data, syringe_defaults)
        for name, motor_data in motors_data.items()
    }

    return PumpsConfig(
        enabled=True,
        stepper=stepper,
        syringe_defaults=syringe_defaults,
        motors=motors,
    )


def _parse_bed_mesh(data: dict[str, Any]) -> BedMeshConfig | None:
    """Parse the optional ``bed_mesh`` section.

    Returns ``None`` when the section is absent or empty.
    """
    if not data:
        return None

    mesh_min_raw = data.get("mesh_min", [10.0, 10.0])
    mesh_max_raw = data.get("mesh_max", [440.0, 310.0])
    probe_count_raw = data.get("probe_count", [3, 3])
    mesh_pps_raw = data.get("mesh_pps", [2, 2])

    for label, val in [
        ("mesh_min", mesh_min_raw),
        ("mesh_max", mesh_max_raw),
        ("probe_count", probe_count_raw),
        ("mesh_pps", mesh_pps_raw),
    ]:
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ConfigError(
                f"bed_mesh.{label} must be a 2-element list, got {val!r}"
            )

    pc = (int(probe_count_raw[0]), int(probe_count_raw[1]))
    if pc[0] < 2 or pc[1] < 2:
        raise ConfigError(
            f"bed_mesh.probe_count must be >= [2, 2], got {list(pc)}"
        )

    cal_raw = data.get("calibrated_points")
    calibrated: list[list[float]] | None = None
    if cal_raw is not None:
        calibrated = [
            [float(v) for v in row] for row in cal_raw
        ]
        if len(calibrated) != pc[1]:
            raise ConfigError(
                f"bed_mesh.calibrated_points has {len(calibrated)} rows "
                f"but probe_count Y = {pc[1]}"
            )
        for i, row in enumerate(calibrated):
            if len(row) != pc[0]:
                raise ConfigError(
                    f"bed_mesh.calibrated_points row {i} has {len(row)} "
                    f"values but probe_count X = {pc[0]}"
                )

    algorithm = str(data.get("algorithm", "lagrange"))
    if algorithm not in ("lagrange", "bicubic"):
        raise ConfigError(
            f"bed_mesh.algorithm must be 'lagrange' or 'bicubic', "
            f"got '{algorithm}'"
        )

    return BedMeshConfig(
        mesh_min=(float(mesh_min_raw[0]), float(mesh_min_raw[1])),
        mesh_max=(float(mesh_max_raw[0]), float(mesh_max_raw[1])),
        probe_count=pc,
        horizontal_move_z=float(data.get("horizontal_move_z", 75.0)),
        speed=float(data.get("speed", 200.0)),
        mesh_pps=(int(mesh_pps_raw[0]), int(mesh_pps_raw[1])),
        algorithm=algorithm,
        calibrated_points=calibrated,
    )


def _parse_servo(data: dict[str, Any]) -> ServoConfig | None:
    """Parse the optional ``servos`` section.

    Returns ``None`` when servos are disabled or absent.
    """
    if not data or not data.get("enabled", False):
        return None

    name = str(data.get("name", "tool_servo"))
    pin = data.get("pin")
    if not pin:
        raise ConfigError("servos.pin is required when servos are enabled")

    angle_range = float(data.get("angle_range_deg", 180.0))
    if angle_range <= 0 or angle_range > 360:
        raise ConfigError(
            f"servos.angle_range_deg must be in (0, 360], got {angle_range}"
        )

    min_pw = float(data.get("min_pulse_width_s", 0.0005))
    max_pw = float(data.get("max_pulse_width_s", 0.0025))
    neutral_pw = float(data.get("neutral_pulse_width_s", 0.0015))

    if not (min_pw < neutral_pw < max_pw):
        raise ConfigError(
            f"Servo pulse widths must satisfy min < neutral < max: "
            f"{min_pw} < {neutral_pw} < {max_pw}"
        )

    return ServoConfig(
        name=name,
        pin=str(pin),
        angle_range_deg=angle_range,
        min_pulse_width_s=min_pw,
        max_pulse_width_s=max_pw,
        neutral_pulse_width_s=neutral_pw,
    )


def _parse_axis(name: str, data: dict[str, Any]) -> AxisConfig:
    """Parse a single axis entry from the ``axes`` section."""
    raw_pins = data.get("pins")
    if raw_pins is None:
        raise ConfigError(f"Axis '{name}' is missing 'pins' field")
    return AxisConfig(
        octopus_slot=str(data["octopus_slot"]),
        motors=int(data["motors"]),
        pins=_parse_pins(raw_pins),
        endstop_pin=data.get("endstop_pin"),
        endstop_polarity=data.get("endstop_polarity"),
        endstop_type=data.get("endstop_type"),
        homing_side=data.get("homing_side"),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_config(cfg: MachineConfig) -> None:
    """Validate cross-field consistency.

    Raises
    ------
    ConfigError
        On any invalid combination.
    """
    # -- Canvas fits inside work area ---------------------------------------
    canvas_max_x = cfg.canvas.offset_x_mm + cfg.canvas.width_mm
    canvas_max_y = cfg.canvas.offset_y_mm + cfg.canvas.height_mm
    if canvas_max_x > cfg.work_area.x:
        raise ConfigError(
            f"Canvas exceeds work area in X: "
            f"{canvas_max_x:.1f} > {cfg.work_area.x:.1f}"
        )
    if canvas_max_y > cfg.work_area.y:
        raise ConfigError(
            f"Canvas exceeds work area in Y: "
            f"{canvas_max_y:.1f} > {cfg.work_area.y:.1f}"
        )

    # -- Z states ordered for seesaw mechanism ------------------------------
    z = cfg.z_states
    if not (
        z.airbrush_work_mm <= z.travel_mm <= z.pen_work_mm
        or z.pen_work_mm <= z.travel_mm <= z.airbrush_work_mm
    ):
        logger.warning(
            "Z states may not be ordered correctly for seesaw: "
            "travel=%.1f, pen_work=%.1f, airbrush_work=%.1f",
            z.travel_mm,
            z.pen_work_mm,
            z.airbrush_work_mm,
        )

    # -- At least one tool defined -----------------------------------------
    if "pen" not in cfg.tools and "airbrush" not in cfg.tools:
        raise ConfigError(
            "Configuration must define at least 'pen' or 'airbrush' tool"
        )

    # -- Jog increments valid -----------------------------------------------
    for inc in cfg.interactive.jog_increments_mm:
        if inc <= 0:
            raise ConfigError(f"Jog increment must be positive, got {inc}")
    if (
        cfg.interactive.default_jog_increment_mm
        not in cfg.interactive.jog_increments_mm
    ):
        raise ConfigError(
            f"Default jog increment "
            f"{cfg.interactive.default_jog_increment_mm} not in "
            f"jog_increments_mm {cfg.interactive.jog_increments_mm}"
        )

    # -- Stepper sanity: microsteps * full_steps == driver_pulses_per_rev ---
    s = cfg.steppers
    expected_pulses = s.klipper_microsteps * s.full_steps_per_rotation
    if expected_pulses != s.driver_pulses_per_rev:
        raise ConfigError(
            f"klipper_microsteps ({s.klipper_microsteps}) * "
            f"full_steps_per_rotation ({s.full_steps_per_rotation}) = "
            f"{expected_pulses}, but driver_pulses_per_rev = "
            f"{s.driver_pulses_per_rev}. These must match."
        )

    # -- rotation_distance == belt_pitch_mm * pulley_teeth ------------------
    expected_rot = s.belt_pitch_mm * s.pulley_teeth
    if not math.isclose(s.xy_rotation_distance, expected_rot, rel_tol=1e-6):
        raise ConfigError(
            f"xy_rotation_distance ({s.xy_rotation_distance}) != "
            f"belt_pitch_mm ({s.belt_pitch_mm}) * pulley_teeth "
            f"({s.pulley_teeth}) = {expected_rot}"
        )

    # -- Tool feed rates do not exceed motion limits ------------------------
    for name, tool in cfg.tools.items():
        if tool.travel_feed_mm_s > cfg.motion.max_velocity_mm_s:
            raise ConfigError(
                f"Tool '{name}' travel_feed_mm_s ({tool.travel_feed_mm_s}) "
                f"exceeds max_velocity_mm_s ({cfg.motion.max_velocity_mm_s})"
            )
        if tool.feed_mm_s > cfg.motion.max_velocity_mm_s:
            raise ConfigError(
                f"Tool '{name}' feed_mm_s ({tool.feed_mm_s}) "
                f"exceeds max_velocity_mm_s ({cfg.motion.max_velocity_mm_s})"
            )

    # -- Pump stepper sanity (if enabled) ------------------------------------
    if cfg.pumps is not None:
        ps = cfg.pumps.stepper
        expected_pump_pulses = (
            ps.klipper_microsteps * ps.full_steps_per_rotation
        )
        if expected_pump_pulses != ps.driver_pulses_per_rev:
            raise ConfigError(
                f"Pump stepper: klipper_microsteps ({ps.klipper_microsteps}) "
                f"* full_steps_per_rotation ({ps.full_steps_per_rotation}) = "
                f"{expected_pump_pulses}, but driver_pulses_per_rev = "
                f"{ps.driver_pulses_per_rev}. These must match."
            )
        for pump_name, pump_motor in cfg.pumps.motors.items():
            sy = pump_motor.syringe
            if sy.volume_ml <= 0:
                raise ConfigError(
                    f"Pump '{pump_name}' syringe volume_ml must be > 0, "
                    f"got {sy.volume_ml}"
                )
            if sy.plunger_travel_mm <= 0:
                raise ConfigError(
                    f"Pump '{pump_name}' syringe plunger_travel_mm must "
                    f"be > 0, got {sy.plunger_travel_mm}"
                )

    # -- Bed mesh bounds within canvas --------------------------------------
    if cfg.bed_mesh is not None:
        bm = cfg.bed_mesh
        canvas_min_x = cfg.canvas.offset_x_mm
        canvas_min_y = cfg.canvas.offset_y_mm
        canvas_max_x = cfg.canvas.offset_x_mm + cfg.canvas.width_mm
        canvas_max_y = cfg.canvas.offset_y_mm + cfg.canvas.height_mm
        if bm.mesh_min[0] < canvas_min_x or bm.mesh_min[1] < canvas_min_y:
            raise ConfigError(
                f"bed_mesh.mesh_min {bm.mesh_min} is outside canvas "
                f"bounds ({canvas_min_x}, {canvas_min_y})"
            )
        if bm.mesh_max[0] > canvas_max_x or bm.mesh_max[1] > canvas_max_y:
            raise ConfigError(
                f"bed_mesh.mesh_max {bm.mesh_max} is outside canvas "
                f"bounds ({canvas_max_x}, {canvas_max_y})"
            )

    # -- Connection values positive -----------------------------------------
    c = cfg.connection
    if c.timeout_s <= 0:
        raise ConfigError(f"timeout_s must be > 0, got {c.timeout_s}")
    if c.reconnect_attempts < 0:
        raise ConfigError(
            f"reconnect_attempts must be >= 0, got {c.reconnect_attempts}"
        )
    if c.reconnect_interval_s < 0:
        raise ConfigError(
            f"reconnect_interval_s must be >= 0, got {c.reconnect_interval_s}"
        )

    # -- Axis pin counts match motor counts ---------------------------------
    for axis_name, axis in cfg.axes.items():
        if len(axis.pins) != axis.motors:
            raise ConfigError(
                f"Axis '{axis_name}' declares {axis.motors} motor(s) "
                f"but has {len(axis.pins)} pin group(s)"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> MachineConfig:
    """Load and validate machine configuration from YAML.

    Parameters
    ----------
    path : str | Path | None
        Path to ``machine.yaml``.  ``None`` loads the default shipped
        alongside this module.

    Returns
    -------
    MachineConfig
        Fully validated, frozen configuration object.

    Raises
    ------
    ConfigError
        If any field is missing or fails validation.
    FileNotFoundError
        If *path* does not exist.
    """
    if path is None:
        path = Path(__file__).parent / "machine.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    logger.info("Loading configuration from %s", path)

    data: dict[str, Any] = load_yaml(path)
    if data is None:
        raise ConfigError(f"Empty configuration file: {path}")

    try:
        # -- connection -----------------------------------------------------
        conn_data = data["connection"]
        connection = ConnectionConfig(
            socket_path=conn_data["socket_path"],
            timeout_s=float(conn_data["timeout_s"]),
            reconnect_attempts=int(conn_data["reconnect_attempts"]),
            reconnect_interval_s=float(conn_data["reconnect_interval_s"]),
            auto_reconnect=bool(conn_data.get("auto_reconnect", True)),
            mcu_serial=str(conn_data.get("mcu_serial", "")),
        )

        # -- work area ------------------------------------------------------
        wa = data["machine"]["work_area_mm"]
        sl_data = data["machine"].get("soft_limits", {})
        soft_limits = SoftLimitsConfig(
            z_min=float(sl_data.get("z_min", 2.0)),
            z_max=float(sl_data.get("z_max", wa["z"])),
            z_overtravel_mm=float(sl_data.get("z_overtravel_mm", 0.0)),
        )
        work_area = WorkAreaConfig(
            x=float(wa["x"]), y=float(wa["y"]), z=float(wa["z"]),
            soft_limits=soft_limits,
        )

        # -- canvas ---------------------------------------------------------
        cv = data["canvas"]
        canvas = CanvasConfig(
            offset_x_mm=float(cv["offset_x_mm"]),
            offset_y_mm=float(cv["offset_y_mm"]),
            width_mm=float(cv["width_mm"]),
            height_mm=float(cv["height_mm"]),
        )

        # -- z states -------------------------------------------------------
        zd = data["z_states"]
        z_states = ZStatesConfig(
            travel_mm=float(zd["travel_mm"]),
            pen_work_mm=float(zd["pen_work_mm"]),
            airbrush_work_mm=float(zd["airbrush_work_mm"]),
        )

        # -- tools ----------------------------------------------------------
        tools = {
            name: _parse_tool(name, cfg)
            for name, cfg in data["tools"].items()
        }

        # -- steppers -------------------------------------------------------
        steppers = _parse_steppers(data["steppers"])

        # -- axes -----------------------------------------------------------
        axes = {
            name: _parse_axis(name, cfg)
            for name, cfg in data["axes"].items()
        }

        # -- motion ---------------------------------------------------------
        md = data["motion"]
        motion = MotionConfig(
            max_velocity_mm_s=float(md["max_velocity_mm_s"]),
            max_accel_mm_s2=float(md["max_accel_mm_s2"]),
            square_corner_velocity_mm_s=float(
                md["square_corner_velocity_mm_s"]
            ),
            homing_speed_mm_s=float(md["homing_speed_mm_s"]),
            z_homing_speed_mm_s=float(
                md.get("z_homing_speed_mm_s", md["homing_speed_mm_s"])
            ),
            z_second_homing_speed_mm_s=float(
                md.get("z_second_homing_speed_mm_s", 0.0)
            ),
            z_homing_retract_mm=float(
                md.get("z_homing_retract_mm", 0.0)
            ),
            junction_deviation_mm=float(md["junction_deviation_mm"]),
            idle_timeout_s=float(md.get("idle_timeout_s", 30.0)),
        )

        # -- interactive ----------------------------------------------------
        id_ = data["interactive"]
        interactive = InteractiveConfig(
            jog_increments_mm=tuple(
                float(x) for x in id_["jog_increments_mm"]
            ),
            default_jog_increment_mm=float(id_["default_jog_increment_mm"]),
            position_poll_interval_ms=int(id_["position_poll_interval_ms"]),
        )

        # -- file execution -------------------------------------------------
        fe = data["file_execution"]
        file_execution = FileExecutionConfig(
            gcode_directory=fe["gcode_directory"],
        )

        # -- pumps (optional) -----------------------------------------------
        pumps = _parse_pumps(data.get("pumps", {}))

        # -- bed mesh (optional) --------------------------------------------
        bed_mesh = _parse_bed_mesh(data.get("bed_mesh", {}))

        # -- servo (optional) -----------------------------------------------
        servo = _parse_servo(data.get("servos", {}))

        # -- endstop phase (optional) ---------------------------------------
        ep_data = data.get("endstop_phase", {})
        endstop_phase_enabled = bool(
            ep_data.get("enabled", False) if ep_data else False
        )

        config = MachineConfig(
            connection=connection,
            work_area=work_area,
            canvas=canvas,
            z_states=z_states,
            tools=tools,
            steppers=steppers,
            axes=axes,
            motion=motion,
            interactive=interactive,
            file_execution=file_execution,
            pumps=pumps,
            bed_mesh=bed_mesh,
            servo=servo,
            endstop_phase_enabled=endstop_phase_enabled,
        )

        _validate_config(config)
        logger.info("Configuration loaded successfully")
        return config

    except KeyError as exc:
        raise ConfigError(
            f"Missing required configuration key: {exc}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Invalid configuration value: {exc}"
        ) from exc
