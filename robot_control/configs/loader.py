"""
Configuration loader.

Loads and validates machine configuration from YAML files.
Returns typed dataclasses for type-safe access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration validation fails."""

    pass


@dataclass(frozen=True)
class ConnectionConfig:
    """Klipper API connection settings."""

    socket_path: str
    timeout_s: float
    reconnect_attempts: int


@dataclass(frozen=True)
class WorkAreaConfig:
    """Machine work area dimensions in mm."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class CanvasConfig:
    """Canvas position and size within work area."""

    offset_x_mm: float
    offset_y_mm: float
    width_mm: float
    height_mm: float
    y_flip: bool


@dataclass(frozen=True)
class ZStatesConfig:
    """Z-axis seesaw states in mm."""

    travel_mm: float
    pen_work_mm: float
    airbrush_work_mm: float


@dataclass(frozen=True)
class ToolConfig:
    """Individual tool configuration."""

    xy_offset_mm: tuple[float, float]
    feed_mm_min: float
    travel_feed_mm_min: float
    plunge_feed_mm_min: float
    spray_height_mm: float = 0.0  # Only used for airbrush


@dataclass(frozen=True)
class MotionConfig:
    """Motion limits."""

    max_velocity_mm_s: float
    max_accel_mm_s2: float
    junction_deviation_mm: float


@dataclass(frozen=True)
class InteractiveConfig:
    """Interactive mode settings."""

    jog_increments_mm: tuple[float, ...]
    default_jog_increment_mm: float
    position_poll_interval_ms: int


@dataclass(frozen=True)
class FileExecutionConfig:
    """File-run mode settings."""

    gcode_directory: str


@dataclass(frozen=True)
class MachineConfig:
    """
    Complete machine configuration.

    Loaded from machine.yaml and validated for consistency.
    All dimensions are in millimeters unless otherwise noted.
    """

    connection: ConnectionConfig
    work_area: WorkAreaConfig
    canvas: CanvasConfig
    z_states: ZStatesConfig
    tools: dict[str, ToolConfig]
    motion: MotionConfig
    interactive: InteractiveConfig
    file_execution: FileExecutionConfig
    pumps_enabled: bool = False
    servos_enabled: bool = False

    def get_tool(self, name: str) -> ToolConfig:
        """Get tool configuration by name, raising ConfigError if not found."""
        if name not in self.tools:
            raise ConfigError(f"Unknown tool '{name}'. Available: {list(self.tools.keys())}")
        return self.tools[name]

    def canvas_to_machine(self, x: float, y: float, tool: str = "pen") -> tuple[float, float]:
        """
        Convert canvas coordinates to machine coordinates.

        Parameters
        ----------
        x : float
            X position in canvas coordinates (mm).
        y : float
            Y position in canvas coordinates (mm).
        tool : str
            Tool name for offset application.

        Returns
        -------
        tuple[float, float]
            Machine coordinates (x, y) in mm.
        """
        tool_cfg = self.get_tool(tool)

        # Apply canvas offset
        mx = x + self.canvas.offset_x_mm
        my = y + self.canvas.offset_y_mm

        # Apply Y flip if needed (image origin is top-left, machine origin is bottom-left)
        if self.canvas.y_flip:
            my = self.canvas.offset_y_mm + (self.canvas.height_mm - y)

        # Apply tool offset
        mx += tool_cfg.xy_offset_mm[0]
        my += tool_cfg.xy_offset_mm[1]

        return mx, my

    def get_z_for_tool(self, tool: str, state: str) -> float:
        """
        Get Z position for a tool in a given state.

        Parameters
        ----------
        tool : str
            Tool name ("pen" or "airbrush").
        state : str
            Z state ("travel" or "work").

        Returns
        -------
        float
            Z position in mm.
        """
        if state == "travel":
            return self.z_states.travel_mm
        elif state == "work":
            if tool == "pen":
                return self.z_states.pen_work_mm
            elif tool == "airbrush":
                return self.z_states.airbrush_work_mm
            else:
                raise ConfigError(f"Unknown tool '{tool}' for Z work state")
        else:
            raise ConfigError(f"Unknown Z state '{state}'. Expected 'travel' or 'work'")


def _parse_tool(name: str, data: dict[str, Any]) -> ToolConfig:
    """Parse a tool configuration section."""
    xy_offset = data.get("xy_offset_mm", [0.0, 0.0])
    if not isinstance(xy_offset, (list, tuple)) or len(xy_offset) != 2:
        raise ConfigError(f"Tool '{name}' xy_offset_mm must be a 2-element list")

    return ToolConfig(
        xy_offset_mm=(float(xy_offset[0]), float(xy_offset[1])),
        feed_mm_min=float(data["feed_mm_min"]),
        travel_feed_mm_min=float(data["travel_feed_mm_min"]),
        plunge_feed_mm_min=float(data["plunge_feed_mm_min"]),
        spray_height_mm=float(data.get("spray_height_mm", 0.0)),
    )


def _validate_config(cfg: MachineConfig) -> None:
    """
    Validate configuration consistency.

    Raises
    ------
    ConfigError
        If validation fails.
    """
    # Canvas must fit within work area
    canvas_max_x = cfg.canvas.offset_x_mm + cfg.canvas.width_mm
    canvas_max_y = cfg.canvas.offset_y_mm + cfg.canvas.height_mm

    if canvas_max_x > cfg.work_area.x:
        raise ConfigError(
            f"Canvas exceeds work area in X: {canvas_max_x:.1f} > {cfg.work_area.x:.1f}"
        )
    if canvas_max_y > cfg.work_area.y:
        raise ConfigError(
            f"Canvas exceeds work area in Y: {canvas_max_y:.1f} > {cfg.work_area.y:.1f}"
        )

    # Z states should be ordered sensibly
    z = cfg.z_states
    if not (z.airbrush_work_mm <= z.travel_mm <= z.pen_work_mm or
            z.pen_work_mm <= z.travel_mm <= z.airbrush_work_mm):
        logger.warning(
            "Z states may not be ordered correctly for seesaw mechanism: "
            f"travel={z.travel_mm}, pen_work={z.pen_work_mm}, airbrush_work={z.airbrush_work_mm}"
        )

    # Must have at least pen or airbrush tool
    if "pen" not in cfg.tools and "airbrush" not in cfg.tools:
        raise ConfigError("Configuration must define at least 'pen' or 'airbrush' tool")

    # Jog increments must be positive
    for inc in cfg.interactive.jog_increments_mm:
        if inc <= 0:
            raise ConfigError(f"Jog increment must be positive, got {inc}")

    # Default jog must be in the list
    if cfg.interactive.default_jog_increment_mm not in cfg.interactive.jog_increments_mm:
        raise ConfigError(
            f"Default jog increment {cfg.interactive.default_jog_increment_mm} "
            f"not in jog_increments_mm {cfg.interactive.jog_increments_mm}"
        )


def load_config(path: str | Path | None = None) -> MachineConfig:
    """
    Load and validate machine configuration from YAML file.

    Parameters
    ----------
    path : str | Path | None
        Path to configuration file. If None, loads from default location
        (robot_control/configs/machine.yaml).

    Returns
    -------
    MachineConfig
        Validated configuration object.

    Raises
    ------
    ConfigError
        If configuration is invalid or file not found.
    FileNotFoundError
        If configuration file does not exist.
    """
    if path is None:
        path = Path(__file__).parent / "machine.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    logger.info("Loading configuration from %s", path)

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ConfigError(f"Empty configuration file: {path}")

    try:
        # Parse connection
        conn_data = data["connection"]
        connection = ConnectionConfig(
            socket_path=conn_data["socket_path"],
            timeout_s=float(conn_data["timeout_s"]),
            reconnect_attempts=int(conn_data["reconnect_attempts"]),
        )

        # Parse work area
        wa_data = data["machine"]["work_area_mm"]
        work_area = WorkAreaConfig(
            x=float(wa_data["x"]),
            y=float(wa_data["y"]),
            z=float(wa_data["z"]),
        )

        # Parse canvas
        cv_data = data["canvas"]
        canvas = CanvasConfig(
            offset_x_mm=float(cv_data["offset_x_mm"]),
            offset_y_mm=float(cv_data["offset_y_mm"]),
            width_mm=float(cv_data["width_mm"]),
            height_mm=float(cv_data["height_mm"]),
            y_flip=bool(cv_data.get("y_flip", True)),
        )

        # Parse Z states
        z_data = data["z_states"]
        z_states = ZStatesConfig(
            travel_mm=float(z_data["travel_mm"]),
            pen_work_mm=float(z_data["pen_work_mm"]),
            airbrush_work_mm=float(z_data["airbrush_work_mm"]),
        )

        # Parse tools
        tools_data = data["tools"]
        tools = {name: _parse_tool(name, cfg) for name, cfg in tools_data.items()}

        # Parse motion
        motion_data = data["motion"]
        motion = MotionConfig(
            max_velocity_mm_s=float(motion_data["max_velocity_mm_s"]),
            max_accel_mm_s2=float(motion_data["max_accel_mm_s2"]),
            junction_deviation_mm=float(motion_data["junction_deviation_mm"]),
        )

        # Parse interactive
        int_data = data["interactive"]
        interactive = InteractiveConfig(
            jog_increments_mm=tuple(float(x) for x in int_data["jog_increments_mm"]),
            default_jog_increment_mm=float(int_data["default_jog_increment_mm"]),
            position_poll_interval_ms=int(int_data["position_poll_interval_ms"]),
        )

        # Parse file execution
        fe_data = data["file_execution"]
        file_execution = FileExecutionConfig(
            gcode_directory=fe_data["gcode_directory"],
        )

        # Build config
        config = MachineConfig(
            connection=connection,
            work_area=work_area,
            canvas=canvas,
            z_states=z_states,
            tools=tools,
            motion=motion,
            interactive=interactive,
            file_execution=file_execution,
            pumps_enabled=data.get("pumps", {}).get("enabled", False),
            servos_enabled=data.get("servos", {}).get("enabled", False),
        )

        # Validate
        _validate_config(config)

        logger.info("Configuration loaded successfully")
        return config

    except KeyError as e:
        raise ConfigError(f"Missing required configuration key: {e}") from e
    except (TypeError, ValueError) as e:
        raise ConfigError(f"Invalid configuration value: {e}") from e
