"""
Job Executor.

Orchestrates job execution in both file-run and interactive modes.
Converts Job IR to G-code, manages execution state, provides progress feedback.

Execution Modes:
    - File-run (production): Generate complete G-code file, use virtual_sdcard
    - Interactive (testing): Stroke-by-stroke with M400 barriers
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
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


class ExecutorState(Enum):
    """Job executor state."""

    IDLE = auto()
    RUNNING_FILE = auto()
    RUNNING_INTERACTIVE = auto()
    PAUSED = auto()
    ERROR = auto()


@dataclass
class ExecutorProgress:
    """Execution progress information."""

    state: ExecutorState
    total_strokes: int
    completed_strokes: int
    current_stroke: int
    elapsed_time: float
    progress_percent: float


class JobExecutor:
    """
    Execute Job IR operations via Klipper.

    Provides two execution paths:
    - File-run: Generate G-code file, print via virtual_sdcard
    - Interactive: Stroke-by-stroke with pause support

    Parameters
    ----------
    client : KlipperClient
        Connected Klipper API client.
    config : MachineConfig
        Machine configuration.

    Examples
    --------
    >>> from robot_control.hardware.klipper_client import KlipperClient
    >>> from robot_control.configs.loader import load_config
    >>> config = load_config()
    >>> with KlipperClient(config.connection.socket_path) as client:
    ...     executor = JobExecutor(client, config)
    ...     executor.run_interactive(operations)
    """

    def __init__(self, client: KlipperClient, config: MachineConfig) -> None:
        self.client = client
        self.config = config
        self.generator = GCodeGenerator(config)

        self._state = ExecutorState.IDLE
        self._lock = threading.Lock()

        # Progress tracking
        self._total_strokes = 0
        self._completed_strokes = 0
        self._current_stroke = 0
        self._start_time: float | None = None

        # Control flags
        self._pause_flag = threading.Event()
        self._cancel_flag = threading.Event()

        # Callbacks
        self._progress_callback: Callable[[ExecutorProgress], None] | None = None

    @property
    def state(self) -> ExecutorState:
        """Current executor state."""
        with self._lock:
            return self._state

    def set_progress_callback(
        self,
        callback: Callable[[ExecutorProgress], None] | None,
    ) -> None:
        """
        Register callback for progress updates.

        Parameters
        ----------
        callback : callable | None
            Function called with ExecutorProgress on each stroke completion.
        """
        self._progress_callback = callback

    def get_progress(self) -> ExecutorProgress:
        """Get current execution progress."""
        with self._lock:
            elapsed = 0.0
            if self._start_time:
                elapsed = time.monotonic() - self._start_time

            percent = 0.0
            if self._total_strokes > 0:
                percent = 100.0 * self._completed_strokes / self._total_strokes

            return ExecutorProgress(
                state=self._state,
                total_strokes=self._total_strokes,
                completed_strokes=self._completed_strokes,
                current_stroke=self._current_stroke,
                elapsed_time=elapsed,
                progress_percent=percent,
            )

    def _set_state(self, state: ExecutorState) -> None:
        """Thread-safe state update."""
        with self._lock:
            self._state = state
        logger.debug("Executor state: %s", state.name)

    def _notify_progress(self) -> None:
        """Call progress callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(self.get_progress())
            except Exception as e:
                logger.error("Progress callback error: %s", e)

    # --- File-Run Mode ---

    def run_file(
        self,
        operations: list[Operation],
        output_dir: str | Path | None = None,
        filename: str = "job.gcode",
    ) -> str:
        """
        Execute job via file-run mode.

        Generates complete G-code file and prints via virtual_sdcard.
        Non-blocking - returns immediately after starting print.

        Parameters
        ----------
        operations : list[Operation]
            Job IR operations to execute.
        output_dir : str | Path | None
            Directory for G-code file. Uses config default if None.
        filename : str
            Output filename.

        Returns
        -------
        str
            Path to generated G-code file.
        """
        if self._state != ExecutorState.IDLE:
            raise RuntimeError(f"Cannot start job: executor is {self._state.name}")

        # Determine output path
        if output_dir is None:
            output_dir = Path(self.config.file_execution.gcode_directory)
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        # Generate G-code
        logger.info("Generating G-code file: %s", output_path)
        gcode = self.generator.generate(operations)

        with open(output_path, "w") as f:
            f.write(gcode)

        # Track strokes for progress
        strokes = operations_to_strokes(operations)
        self._total_strokes = len(strokes)
        self._completed_strokes = 0
        self._start_time = time.monotonic()

        # Start print
        self._set_state(ExecutorState.RUNNING_FILE)
        self.client.start_print(filename)

        logger.info(
            "Started file print: %s (%d strokes)",
            filename,
            self._total_strokes,
        )

        return str(output_path)

    def get_file_progress(self) -> dict:
        """
        Query file print progress from Klipper.

        Returns
        -------
        dict
            Progress info from virtual_sdcard.
        """
        return self.client.get_print_progress()

    def pause_file(self) -> None:
        """Pause file print."""
        if self._state != ExecutorState.RUNNING_FILE:
            logger.warning("Cannot pause: not in file-run mode")
            return

        self.client.pause_print()
        self._set_state(ExecutorState.PAUSED)
        logger.info("File print paused")

    def resume_file(self) -> None:
        """Resume file print."""
        if self._state != ExecutorState.PAUSED:
            logger.warning("Cannot resume: not paused")
            return

        self.client.resume_print()
        self._set_state(ExecutorState.RUNNING_FILE)
        logger.info("File print resumed")

    def cancel_file(self) -> None:
        """Cancel file print."""
        if self._state not in (ExecutorState.RUNNING_FILE, ExecutorState.PAUSED):
            logger.warning("Cannot cancel: no file print active")
            return

        self.client.cancel_print()
        self._set_state(ExecutorState.IDLE)
        self._reset_progress()
        logger.info("File print cancelled")

    # --- Interactive Mode ---

    def run_interactive(
        self,
        operations: list[Operation],
        *,
        step_mode: bool = False,
        step_callback: Callable[[], bool] | None = None,
    ) -> bool:
        """
        Execute job interactively, stroke by stroke.

        Blocking - runs until complete, paused, or cancelled.
        Pause/cancel takes effect at stroke boundaries.

        Parameters
        ----------
        operations : list[Operation]
            Job IR operations to execute.
        step_mode : bool
            If True, pause after each stroke and wait for confirmation.
        step_callback : callable | None
            Called after each stroke in step mode. Return True to continue,
            False to stop. If None in step mode, just pauses until resume.

        Returns
        -------
        bool
            True if completed successfully, False if cancelled.
        """
        if self._state != ExecutorState.IDLE:
            raise RuntimeError(f"Cannot start job: executor is {self._state.name}")

        # Group operations into strokes
        strokes = operations_to_strokes(operations)
        self._total_strokes = len(strokes)
        self._completed_strokes = 0
        self._current_stroke = 0
        self._start_time = time.monotonic()

        # Clear control flags
        self._pause_flag.clear()
        self._cancel_flag.clear()

        self._set_state(ExecutorState.RUNNING_INTERACTIVE)
        logger.info("Starting interactive execution: %d strokes", self._total_strokes)

        try:
            for i, stroke in enumerate(strokes):
                self._current_stroke = i

                # Check cancel flag
                if self._cancel_flag.is_set():
                    logger.info("Execution cancelled at stroke %d", i)
                    self._ensure_tool_up()
                    return False

                # Check pause flag
                while self._pause_flag.is_set():
                    if self._cancel_flag.is_set():
                        self._ensure_tool_up()
                        return False
                    self._set_state(ExecutorState.PAUSED)
                    time.sleep(0.1)
                self._set_state(ExecutorState.RUNNING_INTERACTIVE)

                # Execute stroke
                self._execute_stroke(stroke)
                self._completed_strokes = i + 1
                self._notify_progress()

                # Step mode handling
                if step_mode and i < len(strokes) - 1:
                    if step_callback:
                        if not step_callback():
                            logger.info("Step callback returned False, stopping")
                            return False
                    else:
                        # Pause until manually resumed
                        self._pause_flag.set()

            logger.info(
                "Interactive execution complete: %d strokes in %.1fs",
                self._total_strokes,
                time.monotonic() - self._start_time,
            )
            return True

        except Exception as e:
            logger.error("Execution error: %s", e)
            self._set_state(ExecutorState.ERROR)
            self._ensure_tool_up()
            raise

        finally:
            self._set_state(ExecutorState.IDLE)
            self._reset_progress()

    def _execute_stroke(self, stroke: Stroke) -> None:
        """
        Execute a single stroke.

        Generates G-code with M400 barrier and sends to Klipper.
        """
        gcode = self.generator.generate_stroke(stroke, include_barrier=True)
        self.client.send_gcode(gcode)

        # Wait for motion to complete (M400 is in the script)
        # The send_gcode returns after the script is processed,
        # but M400 blocks until motion is done

    def _ensure_tool_up(self) -> None:
        """Ensure tool is raised for safe pause/cancel."""
        try:
            z_travel = self.config.z_states.travel_mm
            self.client.send_gcode(f"G0 Z{z_travel:.3f} F300\nM400")
        except Exception as e:
            logger.error("Failed to raise tool: %s", e)

    def pause_interactive(self) -> None:
        """
        Set pause flag for interactive execution.

        Takes effect at next stroke boundary.
        """
        self._pause_flag.set()
        logger.info("Pause requested (will take effect at stroke boundary)")

    def resume_interactive(self) -> None:
        """Clear pause flag to resume interactive execution."""
        self._pause_flag.clear()
        logger.info("Resume requested")

    def cancel_interactive(self) -> None:
        """
        Set cancel flag for interactive execution.

        Takes effect at next stroke boundary.
        """
        self._cancel_flag.set()
        logger.info("Cancel requested (will take effect at stroke boundary)")

    def _reset_progress(self) -> None:
        """Reset progress tracking state."""
        self._total_strokes = 0
        self._completed_strokes = 0
        self._current_stroke = 0
        self._start_time = None

    # --- Direct Commands ---

    def home_xy(self) -> None:
        """Home X and Y axes."""
        logger.info("Homing X and Y axes")
        self.client.send_gcode("G28 X Y")

    def move_to(self, x: float, y: float, *, rapid: bool = True) -> None:
        """
        Move to canvas position.

        Parameters
        ----------
        x : float
            Canvas X coordinate in mm.
        y : float
            Canvas Y coordinate in mm.
        rapid : bool
            If True, use rapid move (G0). Otherwise, use linear move (G1).
        """
        mx, my = self.config.canvas_to_machine(x, y)
        tool_cfg = self.config.get_tool("pen")
        feed = tool_cfg.travel_feed_mm_min if rapid else tool_cfg.feed_mm_min
        cmd = "G0" if rapid else "G1"
        self.client.send_gcode(f"{cmd} X{mx:.3f} Y{my:.3f} F{feed:.0f}\nM400")

    def tool_up(self) -> None:
        """Raise tool to travel height."""
        z = self.config.z_states.travel_mm
        feed = self.config.get_tool("pen").plunge_feed_mm_min
        self.client.send_gcode(f"G0 Z{z:.3f} F{feed:.0f}\nM400")

    def tool_down(self, tool: str = "pen") -> None:
        """Lower tool to work height."""
        z = self.config.get_z_for_tool(tool, "work")
        feed = self.config.get_tool(tool).plunge_feed_mm_min
        self.client.send_gcode(f"G1 Z{z:.3f} F{feed:.0f}\nM400")

    def emergency_stop(self) -> None:
        """Trigger immediate emergency stop."""
        self._cancel_flag.set()
        self.client.emergency_stop()
        self._set_state(ExecutorState.ERROR)
