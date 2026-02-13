"""Job executor -- file-run and interactive execution modes.

Orchestrates job execution in two modes:

File-run (production)
    Generate complete ``.gcode`` file, write it to the ``virtual_sdcard``
    directory, and command Klipper to print it.  Klipper handles all
    internal buffering.  Host can disconnect without affecting the print.

Interactive (testing / calibration)
    Stroke-by-stroke execution with ``M400`` barriers.  At most one
    stroke is in flight at any time, so pause/cancel is responsive.

Mode transitions are **safe by construction**:
    - Pause: complete current stroke -> tool up + M400 -> paused state.
    - Cancel: stroke-boundary cancel preferred; if immediate, use
      ``emergency_stop()`` -> ``restart()`` -> re-home -> park.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable

from robot_control.configs.loader import MachineConfig
from robot_control.gcode.generator import GCodeGenerator
from robot_control.hardware.klipper_client import KlipperClient
from robot_control.job_ir.operations import (
    HomeXY,
    Operation,
    Stroke,
    ToolUp,
    operations_to_strokes,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ExecutorState(Enum):
    """Current job-executor state."""

    IDLE = auto()
    RUNNING_FILE = auto()
    RUNNING_INTERACTIVE = auto()
    PAUSED = auto()
    ERROR = auto()


@dataclass
class ExecutorProgress:
    """Execution progress snapshot."""

    state: ExecutorState
    total_strokes: int = 0
    completed_strokes: int = 0
    file_progress: float = 0.0  # 0..1 for file-run mode
    message: str = ""


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class JobExecutor:
    """Dual-mode job executor.

    Parameters
    ----------
    client : KlipperClient
        Connected Klipper API client.
    config : MachineConfig
        Machine configuration.
    """

    def __init__(
        self,
        client: KlipperClient,
        config: MachineConfig,
    ) -> None:
        self._client = client
        self._cfg = config
        self._gen = GCodeGenerator(config)

        self._state = ExecutorState.IDLE
        self._pause_flag = threading.Event()
        self._cancel_flag = threading.Event()
        self._progress_cb: Callable[[ExecutorProgress], None] | None = None
        self._progress = ExecutorProgress(state=ExecutorState.IDLE)

    # ------------------------------------------------------------------
    # Common
    # ------------------------------------------------------------------

    def get_state(self) -> ExecutorState:
        """Return current executor state."""
        return self._state

    def set_progress_callback(
        self, fn: Callable[[ExecutorProgress], None],
    ) -> None:
        """Register a callback invoked on progress updates."""
        self._progress_cb = fn

    def _notify(self, **kwargs: object) -> None:
        """Update internal progress and fire callback."""
        for k, v in kwargs.items():
            if hasattr(self._progress, k):
                setattr(self._progress, k, v)
        self._progress.state = self._state
        if self._progress_cb is not None:
            try:
                self._progress_cb(self._progress)
            except Exception as exc:  # noqa: BLE001
                logger.error("Progress callback error: %s", exc)

    # ------------------------------------------------------------------
    # File-run mode (production)
    # ------------------------------------------------------------------

    def run_file(
        self,
        operations: list[Operation],
        output_dir: str | Path | None = None,
        filename: str = "job.gcode",
    ) -> None:
        """Generate G-code file and start a file print.

        Parameters
        ----------
        operations : list[Operation]
            Flat list of Job IR operations.
        output_dir : str | Path | None
            Directory to write the ``.gcode`` file.  Defaults to the
            ``file_execution.gcode_directory`` from config.
        filename : str
            Output filename.

        Notes
        -----
        Returns immediately after starting the print.  Use
        ``get_file_progress()`` to monitor.
        """
        self._state = ExecutorState.RUNNING_FILE
        self._cancel_flag.clear()

        if output_dir is None:
            output_dir = Path(self._cfg.file_execution.gcode_directory)
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename

        gcode = self._gen.generate(operations)
        filepath.write_text(gcode, encoding="utf-8")
        logger.info("Wrote G-code to %s (%d bytes)", filepath, len(gcode))

        self._client.start_print(filename)
        self._notify(message=f"Printing {filename}")

    def get_file_progress(self) -> ExecutorProgress:
        """Query file-print progress from Klipper."""
        prog = self._client.get_print_progress()
        self._progress.file_progress = prog.get("progress", 0.0)
        self._progress.message = (
            f"{prog.get('state', 'unknown')} "
            f"({self._progress.file_progress * 100:.1f}%)"
        )
        if not prog.get("is_active", False) and self._state == ExecutorState.RUNNING_FILE:
            self._state = ExecutorState.IDLE
        self._progress.state = self._state
        return self._progress

    def pause_file(self) -> None:
        """Pause the current file print."""
        self._client.pause_print()
        self._state = ExecutorState.PAUSED
        self._notify(message="Paused (file)")

    def resume_file(self) -> None:
        """Resume a paused file print."""
        self._client.resume_print()
        self._state = ExecutorState.RUNNING_FILE
        self._notify(message="Resumed (file)")

    def cancel_file(self) -> None:
        """Cancel the current file print."""
        self._client.cancel_print()
        self._state = ExecutorState.IDLE
        self._notify(message="Cancelled (file)")

    # ------------------------------------------------------------------
    # Interactive mode (testing / calibration)
    # ------------------------------------------------------------------

    def run_interactive(
        self,
        operations: list[Operation],
        step_mode: bool = False,
    ) -> None:
        """Execute operations stroke-by-stroke.

        Parameters
        ----------
        operations : list[Operation]
            Flat list of Job IR operations (will be grouped into strokes).
        step_mode : bool
            If ``True``, pause after each stroke and wait for
            ``resume_interactive()`` before continuing.
        """
        strokes = operations_to_strokes(operations)
        total = len(strokes)

        self._state = ExecutorState.RUNNING_INTERACTIVE
        self._pause_flag.clear()
        self._cancel_flag.clear()
        self._notify(total_strokes=total, completed_strokes=0)

        logger.info("Starting interactive execution (%d strokes)", total)

        for idx, stroke in enumerate(strokes):
            # Check cancel
            if self._cancel_flag.is_set():
                logger.info("Interactive execution cancelled at stroke %d", idx)
                self._ensure_safe_state()
                self._state = ExecutorState.IDLE
                self._notify(message="Cancelled (interactive)")
                return

            # Check pause / step mode
            if step_mode or self._pause_flag.is_set():
                self._state = ExecutorState.PAUSED
                self._notify(
                    completed_strokes=idx,
                    message=f"Paused at stroke {idx}/{total}",
                )
                logger.info("Paused at stroke %d/%d", idx, total)
                # Block until resume or cancel
                while self._pause_flag.is_set() or (
                    step_mode and self._state == ExecutorState.PAUSED
                ):
                    if self._cancel_flag.is_set():
                        self._ensure_safe_state()
                        self._state = ExecutorState.IDLE
                        self._notify(message="Cancelled (interactive)")
                        return
                    time.sleep(0.05)
                self._state = ExecutorState.RUNNING_INTERACTIVE

            # Execute stroke
            gcode = self._gen.generate_stroke(stroke)
            self._client.send_gcode(gcode, timeout=30.0)
            self._notify(
                completed_strokes=idx + 1,
                message=f"Stroke {idx + 1}/{total}",
            )

        self._state = ExecutorState.IDLE
        self._notify(message="Complete")
        logger.info("Interactive execution complete")

    def pause_interactive(self) -> None:
        """Set pause flag -- takes effect at the next stroke boundary."""
        self._pause_flag.set()
        logger.info("Interactive pause requested")

    def resume_interactive(self) -> None:
        """Clear the pause flag so execution continues."""
        self._pause_flag.clear()
        self._state = ExecutorState.RUNNING_INTERACTIVE
        logger.info("Interactive resume requested")

    def cancel_interactive(self) -> None:
        """Set cancel flag -- aborts after the current stroke finishes."""
        self._cancel_flag.set()
        logger.info("Interactive cancel requested")

    def _ensure_safe_state(self) -> None:
        """Ensure the tool is up after cancel/pause."""
        try:
            tc = self._cfg.get_tool("pen")
            z_travel = self._cfg.get_z_for_tool("pen", "travel")
            f_val = tc.plunge_feed_mm_s * 60.0
            self._client.send_gcode(
                f"G0 Z{z_travel:.3f} F{f_val:.1f}\nM400",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not raise tool after cancel: %s", exc)
