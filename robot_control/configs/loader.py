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
class WorkAreaConfig:
    """Machine work-area dimensions in mm."""

    x: float
    y: float
    z: float


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
    pumps_enabled: bool = False
    servos_enabled: bool = False

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
        work_area = WorkAreaConfig(
            x=float(wa["x"]), y=float(wa["y"]), z=float(wa["z"]),
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
            junction_deviation_mm=float(md["junction_deviation_mm"]),
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
            pumps_enabled=data.get("pumps", {}).get("enabled", False),
            servos_enabled=data.get("servos", {}).get("enabled", False),
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
