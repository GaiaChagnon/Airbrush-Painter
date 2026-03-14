"""Reusable Rich renderables for the robot CLI.

All widgets are plain ``Rich.ConsoleRenderable`` objects that can be
composed into any ``Layout`` or printed directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from robot_control.hardware.klipper_client import Position
    from robot_control.scripts.cli.connection import KlipperConnectionManager


# ======================================================================
# Status bar (persistent footer)
# ======================================================================


class StatusBar:
    """One-line footer showing connection state, position, and Klipper state.

    Parameters
    ----------
    conn : KlipperConnectionManager
        Connection manager for live data.
    """

    def __init__(self, conn: KlipperConnectionManager) -> None:
        self._conn = conn

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        pos = self._conn.get_position()
        state = self._conn.get_state()
        connected = self._conn.is_connected()

        conn_dot = "[green]●[/]" if connected else "[red]●[/]"
        state_color = {
            "ready": "green", "startup": "yellow",
            "shutdown": "red", "error": "red",
        }.get(state, "dim")

        px = f"{pos.x:.2f}" if pos else "---"
        py = f"{pos.y:.2f}" if pos else "---"
        pz = f"{pos.z:.2f}" if pos else "---"

        line = Text.from_markup(
            f" {conn_dot} Klipper: [{state_color}]{state}[/]"
            f"   X: [bold]{px}[/]  Y: [bold]{py}[/]  Z: [bold]{pz}[/]"
            f"   [dim]Ctrl+C to return[/]"
        )
        yield Panel(line, style="dim", height=3)


def render_status_bar(conn: KlipperConnectionManager) -> Panel:
    """Build a status bar panel from current connection state."""
    pos = conn.get_position()
    state = conn.get_state()
    connected = conn.is_connected()

    conn_dot = "[green]●[/]" if connected else "[red]●[/]"
    state_color = {
        "ready": "green", "startup": "yellow",
        "shutdown": "red", "error": "red",
    }.get(state, "dim")

    px = f"{pos.x:.2f}" if pos else "---"
    py = f"{pos.y:.2f}" if pos else "---"
    pz = f"{pos.z:.2f}" if pos else "---"

    line = (
        f" {conn_dot} Klipper: [{state_color}]{state}[/]"
        f"   X: [bold]{px}[/]  Y: [bold]{py}[/]  Z: [bold]{pz}[/]"
    )
    return Panel(Text.from_markup(line), style="dim", height=3)


# ======================================================================
# Keymap panel
# ======================================================================


def keymap_panel(
    bindings: dict[str, str],
    title: str = "Controls",
    columns: int = 2,
) -> Panel:
    """Render a compact keybinding reference panel.

    Parameters
    ----------
    bindings : dict
        ``{"key_label": "description", ...}``.
    title : str
        Panel title.
    columns : int
        Number of columns to arrange bindings in.
    """
    items: list[Text] = []
    for key, desc in bindings.items():
        items.append(Text.from_markup(f"[bold cyan]{key:<12}[/] {desc}"))

    col_renderables: list[Text] = []
    per_col = (len(items) + columns - 1) // columns
    for c in range(columns):
        chunk = items[c * per_col : (c + 1) * per_col]
        col_renderables.append(Text("\n").join(chunk))

    return Panel(
        Columns(col_renderables, equal=True, expand=True),
        title=f"[bold]{title}[/]",
        border_style="blue",
    )


# ======================================================================
# Info panel (generic key-value)
# ======================================================================


def info_panel(
    data: dict[str, str],
    title: str = "Info",
    border_style: str = "cyan",
) -> Panel:
    """Generic key-value information panel.

    Parameters
    ----------
    data : dict
        ``{"Label": "value", ...}``.
    """
    table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    table.add_column("key", style="bold", no_wrap=True)
    table.add_column("value")
    for k, v in data.items():
        table.add_row(k, v)
    return Panel(table, title=f"[bold]{title}[/]", border_style=border_style)


# ======================================================================
# Pump diagram
# ======================================================================


_SYRINGE_WIDTH = 20
_FLUID_COLORS = {
    "cyan": "cyan",
    "magenta": "magenta",
    "yellow": "yellow",
    "purge": "bright_white",
}


def pump_diagram(
    pump_states: dict[str, dict[str, Any]],
    valve_open: bool = False,
    needle_retracted: bool = False,
) -> Panel:
    """Visual diagram of syringe pumps, valve, and needle.

    Parameters
    ----------
    pump_states : dict
        ``{pump_id: {"fluid": str, "position_mm": float,
        "position_ml": float, "travel_mm": float,
        "travel_ml": float, "homed": bool,
        "speed_mm_s": float}}``.
    valve_open : bool
        Refill valve state.
    needle_retracted : bool
        Airbrush needle state.
    """
    lines: list[Text] = []

    # Valve and needle status line
    valve_indicator = (
        Text.from_markup("[green]■ OPEN[/]") if valve_open
        else Text.from_markup("[red]■ CLOSED[/]")
    )
    needle_indicator = (
        Text.from_markup("[green]■ RETRACTED[/]") if needle_retracted
        else Text.from_markup("[red]■ EXTENDED[/]")
    )
    header = Text("  Refill Valve: ")
    header.append_text(valve_indicator)
    header.append("    Needle: ")
    header.append_text(needle_indicator)
    lines.append(header)
    lines.append(Text(""))

    for pid, st in pump_states.items():
        fluid = st.get("fluid", "?")
        pos = st.get("position_mm", 0.0)
        travel = st.get("travel_mm", 16.5)
        pos_ml = st.get("position_ml", 0.0)
        travel_ml = st.get("travel_ml", 1.0)
        homed = st.get("homed", False)
        speed = st.get("speed_mm_s", 0.0)
        color = _FLUID_COLORS.get(fluid, "white")

        ratio = max(0.0, min(1.0, pos / travel)) if travel > 0 else 0.0
        filled = int(ratio * _SYRINGE_WIDTH)
        empty = _SYRINGE_WIDTH - filled
        bar = Text("[")
        bar.append("█" * filled, style=color)
        bar.append("░" * empty, style="dim")
        bar.append("]")

        homed_tag = "[green]H[/]" if homed else "[red]?[/]"

        label = Text.from_markup(
            f"  {pid:<8} {fluid:<8} {homed_tag} "
        )
        label.append_text(bar)
        label.append(f"  {pos_ml:.3f} / {travel_ml:.3f} ml")
        if speed > 0:
            label.append(f"  @ {speed:.1f} mm/s")
        lines.append(label)

    content = Text("\n").join(lines)
    return Panel(content, title="[bold]Pump Status[/]", border_style="magenta")


# ======================================================================
# Probe grid panel (bed mesh)
# ======================================================================


def probe_grid_panel(
    probe_points: list[tuple[float, float]],
    probe_count: tuple[int, int],
    results: list[float | None] | None = None,
    mean_z: float | None = None,
    title: str = "Probe Grid",
) -> Panel:
    """Visualise a bed-mesh probe grid with optional offset results.

    Parameters
    ----------
    probe_points : list of (x, y) tuples
        All probe point coordinates in grid order (Y rows, X cols).
    probe_count : (nx, ny)
        Grid dimensions.
    results : list of float or None
        Per-point contact Z values. ``None`` entries = not yet probed.
    mean_z : float or None
        Mean contact Z for computing offsets. If ``None``, offsets are
        not displayed.
    """
    nx, ny = probe_count
    table = Table(show_header=False, box=None, padding=(0, 1))
    for _ in range(nx + 1):
        table.add_column(justify="center", min_width=12)

    for row_idx in range(ny - 1, -1, -1):
        cells: list[str] = []
        y_val = probe_points[row_idx * nx][1]
        cells.append(f"[dim]Y={y_val:.0f}[/]")

        for col_idx in range(nx):
            flat_idx = row_idx * nx + col_idx
            x_val = probe_points[flat_idx][0]

            if results is None or results[flat_idx] is None:
                cells.append(f"[dim]({x_val:.0f}, {y_val:.0f})\n  ---[/]")
            else:
                z_val = results[flat_idx]
                if mean_z is not None:
                    offset = z_val - mean_z
                    if abs(offset) < 0.05:
                        color = "green"
                    elif abs(offset) < 0.1:
                        color = "yellow"
                    else:
                        color = "red"
                    cells.append(
                        f"({x_val:.0f}, {y_val:.0f})\n"
                        f"[{color}]{offset:+.3f} mm[/]"
                    )
                else:
                    cells.append(
                        f"({x_val:.0f}, {y_val:.0f})\n"
                        f"Z={z_val:.3f}"
                    )
        table.add_row(*cells)

    return Panel(table, title=f"[bold]{title}[/]", border_style="green")


# ======================================================================
# Position panel (large format for interactive control)
# ======================================================================


def position_panel(
    pos: Position | None,
    tool: str = "pen",
    tool_up: bool = True,
    jog_step: float = 1.0,
    homed: bool = False,
) -> Panel:
    """Large-format position display for interactive jog mode.

    Parameters
    ----------
    pos : Position or None
        Current machine position.
    tool : str
        Active tool name.
    tool_up : bool
        Whether the tool is raised.
    jog_step : float
        Current jog increment in mm.
    homed : bool
        Whether axes are homed.
    """
    px = f"{pos.x:8.3f}" if pos else "    ---"
    py = f"{pos.y:8.3f}" if pos else "    ---"
    pz = f"{pos.z:8.3f}" if pos else "    ---"

    tool_state = "[green]UP[/]" if tool_up else "[red]DOWN[/]"
    homed_tag = "[green]HOMED[/]" if homed else "[yellow]NOT HOMED[/]"

    content = Text.from_markup(
        f"  [bold]X[/]  [bold cyan]{px}[/] mm\n"
        f"  [bold]Y[/]  [bold cyan]{py}[/] mm\n"
        f"  [bold]Z[/]  [bold cyan]{pz}[/] mm\n"
        f"\n"
        f"  Tool:     [bold]{tool.upper()}[/] {tool_state}\n"
        f"  Jog Step: [bold]{jog_step}[/] mm\n"
        f"  Axes:     {homed_tag}"
    )
    return Panel(content, title="[bold]Position[/]", border_style="cyan")


# ======================================================================
# Digital outputs panel
# ======================================================================


def outputs_panel(
    output_states: dict[str, bool],
    active_pump: str = "",
    pump_fluid: str = "",
) -> Panel:
    """Digital output and pump selection display.

    Parameters
    ----------
    output_states : dict
        ``{"output_name": is_on, ...}``.
    active_pump : str
        Currently selected pump ID.
    pump_fluid : str
        Fluid type of the active pump.
    """
    lines: list[str] = []
    for name, is_on in output_states.items():
        short = name.replace("servo_", "").replace("_", " ").title()
        dot = "[green]●[/]" if is_on else "[red]●[/]"
        tag = "[green]ON[/]" if is_on else "[dim]OFF[/]"
        lines.append(f"  {dot} {short:<20} {tag}")

    if active_pump:
        lines.append("")
        lines.append(f"  Pump: [bold]{active_pump}[/]")
        if pump_fluid:
            lines.append(f"  Fluid: {pump_fluid}")

    content = "\n".join(lines)
    return Panel(
        Text.from_markup(content),
        title="[bold]Outputs & Pump[/]",
        border_style="yellow",
    )


# ======================================================================
# Banner
# ======================================================================


def banner() -> Panel:
    """Application banner shown at startup."""
    art = Text.from_markup(
        "[bold cyan]"
        "    ╔═══════════════════════════════════════╗\n"
        "    ║       AIRBRUSH PAINTER  ·  Robot CLI  ║\n"
        "    ╚═══════════════════════════════════════╝"
        "[/]"
    )
    return Panel(Align.center(art), border_style="bright_blue")
