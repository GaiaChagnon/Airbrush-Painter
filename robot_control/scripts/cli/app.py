"""Central coordinator for the unified robot CLI.

``RobotApp`` owns the Rich console, machine config, Klipper
connection manager, and session log.  It runs the main menu loop and
dispatches to mode modules.  Adding a new mode is one line in the
``_MODE_REGISTRY``.
"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path
from typing import Callable

import questionary
from rich.console import Console

from robot_control.configs.loader import MachineConfig, load_config
from robot_control.scripts.cli.connection import KlipperConnectionManager
from robot_control.scripts.cli.session_log import SessionLog
from robot_control.scripts.cli.widgets import banner, render_status_bar

logger = logging.getLogger(__name__)


# Mode registry populated lazily to avoid circular imports.
# Each entry: (display_label, import_path_function)
_MODE_REGISTRY: list[tuple[str, str]] = [
    ("Interactive Control", "robot_control.scripts.cli.interactive_control"),
    ("Pump Controller", "robot_control.scripts.cli.pump_controller"),
    ("Lineart Tracer", "robot_control.scripts.cli.lineart_tracer"),
    ("Calibration", "robot_control.scripts.cli.calibration"),
]


def _import_mode_runner(module_path: str) -> Callable[..., None]:
    """Dynamically import a mode module and return its ``run`` function."""
    import importlib
    mod = importlib.import_module(module_path)
    return mod.run  # type: ignore[attr-defined]


class RobotApp:
    """Top-level application object for the robot CLI.

    Parameters
    ----------
    config_path : str or None
        Path to ``machine.yaml``.  ``None`` uses the default.
    socket_override : str or None
        Override Klipper socket path.
    no_config_write : bool
        Skip printer.cfg regeneration on startup.
    """

    def __init__(
        self,
        config_path: str | None = None,
        socket_override: str | None = None,
        no_config_write: bool = False,
    ) -> None:
        self.console = Console()
        self.config: MachineConfig = load_config(config_path)
        self.session_log = SessionLog()
        self.connection = KlipperConnectionManager(
            config=self.config,
            console=self.console,
            session_log=self.session_log,
        )
        self.no_config_write = no_config_write
        self.socket_override = socket_override
        self._running = True
        self._command_history: dict[str, list[str]] = {}

        # Wire Ctrl+C to graceful shutdown
        signal.signal(signal.SIGINT, self._sigint_handler)

    # ------------------------------------------------------------------
    # Command history for prompted inputs
    # ------------------------------------------------------------------

    def get_history(self, key: str) -> list[str]:
        """Return the prompt history list for *key*, creating if needed."""
        return self._command_history.setdefault(key, [])

    def add_history(self, key: str, value: str) -> None:
        """Append a value to the named history list (deduped)."""
        hist = self.get_history(key)
        if value and (not hist or hist[-1] != value):
            hist.append(value)

    # ------------------------------------------------------------------
    # Confirmation for dangerous operations
    # ------------------------------------------------------------------

    def confirm_dangerous(self, action_name: str) -> bool:
        """Prompt for confirmation before a dangerous operation.

        Parameters
        ----------
        action_name : str
            Human-readable description of the action.

        Returns
        -------
        bool
            ``True`` if the user confirmed.
        """
        result = questionary.confirm(
            f"Confirm dangerous operation: {action_name}?",
            default=False,
        ).ask()
        if result:
            self.session_log.log_action("app", "confirm_dangerous", action_name)
        return bool(result)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the main menu loop until the user quits."""
        self.console.clear()
        self.console.print(banner())
        self.console.print()

        try:
            self.session_log.prompt_and_open()
        except KeyboardInterrupt:
            return

        self.session_log.log_action("app", "startup", f"config={self.config}")

        choices = [label for label, _ in _MODE_REGISTRY] + ["Quit"]

        while self._running:
            self.console.clear()
            self.console.print(banner())
            choice = questionary.select(
                "Select mode:",
                choices=choices,
                instruction="(use arrow keys)",
            ).ask()

            if choice is None or choice == "Quit":
                self._running = False
                break

            for label, module_path in _MODE_REGISTRY:
                if label == choice:
                    self._run_mode(label, module_path)
                    break

        self._cleanup()

    def _run_mode(self, label: str, module_path: str) -> None:
        """Import and execute a single mode, catching exceptions."""
        self.console.clear()
        self.console.rule(f"[bold]{label}[/]")
        self.session_log.log_action("app", "mode_enter", label)

        try:
            runner = _import_mode_runner(module_path)
            runner(self)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Returning to main menu...[/]")
        except Exception as exc:
            self.session_log.log_error(exc)
            self.console.print(f"\n[red]Error: {exc}[/]")
            logger.exception("Mode %s failed", label)

        self.session_log.log_action("app", "mode_exit", label)

    def _cleanup(self) -> None:
        """Graceful shutdown: disconnect, close log."""
        self.console.print("\n[dim]Shutting down...[/]")
        self.connection.disconnect()
        self.session_log.close()
        self.console.print("[green]Goodbye.[/]")

    def _sigint_handler(self, signum: int, frame: object) -> None:
        """Handle Ctrl+C: return to main menu or exit."""
        raise KeyboardInterrupt
