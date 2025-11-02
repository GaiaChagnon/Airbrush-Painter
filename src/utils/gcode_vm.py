"""Offline G-code simulator for dry-run validation.

Provides:
    - Dry-run execution: Parse G-code without hardware
    - Time estimation: Based on feeds, distances, purge delays
    - Soft-limit checking: Detect out-of-bounds moves
    - Kinematics: Simple constant-accel model (optional trapezoidal)

Used by:
    - GUI "Dry Run" button: Validate jobs before sending to machine
    - CI: Ensure generated G-code is well-formed
    - Debugging: Visualize toolpath without hardware

Public API:
    vm = GCodeVM(machine_cfg)
    vm.load_file(gcode_path)
    result = vm.run()  # → {time_estimate, violations: [], last_stroke_id}

Tracks:
    - Current position (X, Y, Z)
    - Feed rate (F in mm/min)
    - Macro execution (purges, pen up/down)

No actual serial communication or hardware control.

Usage:
    from src.utils import gcode_vm, validators
    
    machine_cfg = validators.load_machine_profile("machine.yaml")
    vm = gcode_vm.GCodeVM(machine_cfg)
    vm.load_file("job.gcode")
    result = vm.run()
    
    print(f"Estimated time: {result['time_estimate_s']:.1f}s")
    if result['violations']:
        print(f"Violations: {result['violations']}")
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
import logging
import math

from . import validators

logger = logging.getLogger(__name__)


# ============================================================================
# G-CODE VM
# ============================================================================

class GCodeVM:
    """Offline G-code virtual machine for dry-run validation.
    
    Parameters
    ----------
    machine_cfg : validators.MachineV1
        Machine configuration
    purge_time_s : float
        Time per purge operation (seconds), default 2.0
    pen_time_s : float
        Time per pen up/down operation (seconds), default 0.5
    accel_mm_s2 : Optional[float]
        Acceleration for trapezoidal motion (mm/s²), None for instant accel
    
    Attributes
    ----------
    pos : Tuple[float, float, float]
        Current position (X, Y, Z) in mm
    feed : float
        Current feed rate (mm/min)
    gcode_lines : List[str]
        Loaded G-code lines
    violations : List[str]
        Accumulated soft-limit violations
    last_stroke_id : Optional[str]
        Last encountered stroke ID comment
    total_time : float
        Accumulated time estimate (seconds)
    """
    
    def __init__(
        self,
        machine_cfg: validators.MachineV1,
        purge_time_s: float = 2.0,
        pen_time_s: float = 0.5,
        accel_mm_s2: Optional[float] = None
    ):
        """Initialize G-code VM.
        
        Parameters
        ----------
        machine_cfg : validators.MachineV1
            Machine configuration (includes acceleration, canvas bounds)
        purge_time_s : float
            Time estimate for purge operations (s), default 2.0
        pen_time_s : float
            Time estimate for pen up/down (s), default 0.5
        accel_mm_s2 : Optional[float]
            Acceleration override (mm/s²), None to use machine_cfg.acceleration.max_xy_mm_s2
        
        Notes
        -----
        Acceleration defaults to machine config, not hardcoded 1000.
        Checks both machine and canvas limits.
        """
        self.machine_cfg = machine_cfg
        self.purge_time_s = purge_time_s
        self.pen_time_s = pen_time_s
        # Use acceleration from config if not overridden
        self.accel_mm_s2 = accel_mm_s2 if accel_mm_s2 is not None else machine_cfg.acceleration.max_xy_mm_s2
        
        # VM state
        self.pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.feed: float = 3000.0  # Default feed (mm/min), modal
        self.rapid_mm_s: float = getattr(machine_cfg.feeds, "rapid_mm_s", machine_cfg.feeds.max_xy_mm_s)
        self.absolute_mode: bool = True  # G90=absolute, G91=relative
        self.mm_units: bool = True  # G21=mm, G20=inch
        self.units_scale: float = 1.0  # 1.0 for mm, 25.4 for inch
        
        # Execution tracking
        self.gcode_lines: List[str] = []
        self.violations: List[str] = []
        self.last_stroke_id: Optional[str] = None
        self.total_time: float = 0.0
        self.move_count: int = 0
    
    def load_file(self, path: Union[str, Path]) -> None:
        """Load G-code file for execution.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to G-code file
        
        Raises
        ------
        FileNotFoundError
            If file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"G-code file not found: {path}")
        
        with open(path, 'r') as f:
            self.gcode_lines = f.readlines()
        
        logger.info(f"Loaded {len(self.gcode_lines)} G-code lines from {path}")
    
    def load_string(self, gcode: str) -> None:
        """Load G-code from string.
        
        Parameters
        ----------
        gcode : str
            G-code string (multiple lines)
        """
        self.gcode_lines = gcode.splitlines(keepends=True)
        logger.info(f"Loaded {len(self.gcode_lines)} G-code lines from string")
    
    def reset(self) -> None:
        """Reset VM state to initial position."""
        self.pos = (0.0, 0.0, 0.0)
        self.feed = 3000.0
        self.absolute_mode = True
        self.mm_units = True
        self.units_scale = 1.0
        self.violations = []
        self.last_stroke_id = None
        self.total_time = 0.0
        self.move_count = 0
    
    def check_soft_limits(self, x: float, y: float, z: float, line_idx: Optional[int] = None) -> None:
        """Check if position violates machine or canvas limits.
        
        Parameters
        ----------
        x, y, z : float
            Position to check (absolute machine coordinates, mm)
        line_idx : Optional[int]
            Source line index (0-based), None to use move_count
        
        Notes
        -----
        Violations are accumulated in self.violations.
        Checks MACHINE physical limits (hard violations).
        Warns if outside CANVAS bounds (soft warnings, allows purge zones).
        """
        if not self.machine_cfg.safety.soft_limits:
            return
        
        work = self.machine_cfg.work_area_mm
        canvas = self.machine_cfg.canvas_mm
        where = f"line {line_idx+1}" if line_idx is not None else f"move {self.move_count}"
        
        # Check MACHINE physical limits (hard violations)
        if not (0 <= x <= work.x):
            msg = f"X={x:.2f} exceeds MACHINE limit [0, {work.x}] at {where}"
            self.violations.append(msg)
            logger.warning(msg)
        
        if not (0 <= y <= work.y):
            msg = f"Y={y:.2f} exceeds MACHINE limit [0, {work.y}] at {where}"
            self.violations.append(msg)
            logger.warning(msg)
        
        if not (0 <= z <= work.z):
            msg = f"Z={z:.2f} exceeds MACHINE limit [0, {work.z}] at {where}"
            self.violations.append(msg)
            logger.warning(msg)
        
        # Check CANVAS bounds (soft warnings, allows off-canvas purge zones)
        if not (canvas.x_min <= x <= canvas.x_max):
            logger.debug(f"X={x:.2f} outside CANVAS [{canvas.x_min}, {canvas.x_max}] at {where}")
        
        if not (canvas.y_min <= y <= canvas.y_max):
            logger.debug(f"Y={y:.2f} outside CANVAS [{canvas.y_min}, {canvas.y_max}] at {where}")
    
    def estimate_move_time(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        feed_mm_min: float
    ) -> float:
        """Estimate time for a single move.
        
        Parameters
        ----------
        start : Tuple[float, float, float]
            Start position (X, Y, Z) in mm
        end : Tuple[float, float, float]
            End position (X, Y, Z) in mm
        feed_mm_min : float
            Feed rate (mm/min)
        
        Returns
        -------
        float
            Estimated time (seconds)
        
        Notes
        -----
        Uses constant velocity if accel_mm_s2 is None.
        Uses trapezoidal motion profile if accel_mm_s2 is set.
        """
        # Compute distance
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if dist < 1e-6:
            return 0.0
        
        # Convert feed to mm/s
        feed_mm_s = feed_mm_min / 60.0
        
        if self.accel_mm_s2 is None:
            # Constant velocity
            return dist / feed_mm_s
        else:
            # Trapezoidal motion profile
            # Time to reach target speed
            t_accel = feed_mm_s / self.accel_mm_s2
            # Distance covered during accel/decel
            d_accel = 0.5 * self.accel_mm_s2 * t_accel * t_accel
            
            if 2 * d_accel >= dist:
                # Triangle profile (never reach target speed)
                # d = a * t_acc^2  =>  t_acc = sqrt(d / a); total time = 2 * t_acc
                t_acc = math.sqrt(dist / self.accel_mm_s2)
                return 2.0 * t_acc
            else:
                # Trapezoid profile
                d_const = dist - 2 * d_accel
                t_const = d_const / feed_mm_s
                return 2 * t_accel + t_const
    
    def parse_coordinates(self, line: str) -> Dict[str, float]:
        """Parse X, Y, Z, F from G-code line.
        
        Parameters
        ----------
        line : str
            G-code line
        
        Returns
        -------
        Dict[str, float]
            Parsed values (keys: 'X', 'Y', 'Z', 'F')
        
        Notes
        -----
        Accepts numbers like X.5 (leading decimal point).
        """
        coords = {}
        
        # Match: optional sign, digits.digits OR .digits OR digits
        for axis in ['X', 'Y', 'Z', 'F']:
            match = re.search(rf'{axis}\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))', line, re.IGNORECASE)
            if match:
                coords[axis] = float(match.group(1))
        
        return coords
    
    def execute_line(self, line: str, line_idx: Optional[int] = None) -> None:
        """Execute a single G-code line.
        
        Parameters
        ----------
        line : str
            G-code line
        line_idx : Optional[int]
            Source line index (0-based) for error reporting
        
        Notes
        -----
        Updates VM state (position, feed, time estimate).
        Detects violations and stroke IDs.
        """
        # Strip whitespace and comments
        line = line.strip()
        if not line or line.startswith(';'):
            # Check for stroke ID comment
            if '; STROKE_ID:' in line:
                stroke_id = line.split(':', 1)[1].strip()
                self.last_stroke_id = stroke_id
                logger.debug(f"Stroke ID: {stroke_id}")
            return
        
        # Remove inline comments
        if ';' in line:
            line = line.split(';', 1)[0].strip()
        
        # Parse command
        line_upper = line.upper()
        
        # G0/G1: Linear move (use regex to avoid misclassifying G10 as G1)
        is_g0 = bool(re.match(r'^(?:G0|G00)\b', line_upper))
        is_g1 = bool(re.match(r'^(?:G1|G01)\b', line_upper))
        
        if is_g0 or is_g1:
            coords = self.parse_coordinates(line)
            
            # Update feed if specified (modal - persists to next move)
            # Convert to mm/min if in inches
            if 'F' in coords:
                self.feed = coords['F'] * (25.4 if not self.mm_units else 1.0)
            
            # Apply unit scaling
            scaled_x = coords.get('X', 0.0 if not self.absolute_mode else None)
            scaled_y = coords.get('Y', 0.0 if not self.absolute_mode else None)
            scaled_z = coords.get('Z', 0.0 if not self.absolute_mode else None)
            
            if scaled_x is not None:
                scaled_x *= self.units_scale
            if scaled_y is not None:
                scaled_y *= self.units_scale
            if scaled_z is not None:
                scaled_z *= self.units_scale
            
            # Compute new position
            if self.absolute_mode:
                new_x = scaled_x if scaled_x is not None else self.pos[0]
                new_y = scaled_y if scaled_y is not None else self.pos[1]
                new_z = scaled_z if scaled_z is not None else self.pos[2]
            else:
                # Relative mode (G91)
                new_x = self.pos[0] + (scaled_x if scaled_x is not None else 0.0)
                new_y = self.pos[1] + (scaled_y if scaled_y is not None else 0.0)
                new_z = self.pos[2] + (scaled_z if scaled_z is not None else 0.0)
            
            new_pos = (new_x, new_y, new_z)
            
            # Check soft limits
            self.check_soft_limits(new_x, new_y, new_z, line_idx=line_idx)
            
            # Estimate time (use rapid for G0, modal feed for G1)
            feed_mm_min = (self.rapid_mm_s * 60.0) if is_g0 else self.feed
            move_time = self.estimate_move_time(self.pos, new_pos, feed_mm_min)
            self.total_time += move_time
            
            # Update position
            self.pos = new_pos
            self.move_count += 1
        
        # G21: mm units
        elif line_upper.startswith('G21'):
            self.mm_units = True
            self.units_scale = 1.0
        
        # G20: inch units (convert to mm internally)
        elif line_upper.startswith('G20'):
            self.mm_units = False
            self.units_scale = 25.4  # 1 inch = 25.4 mm
            logger.warning("Inch units (G20) detected, converting to mm internally")
        
        # G90: Absolute positioning
        elif line_upper.startswith('G90'):
            self.absolute_mode = True
        
        # G91: Relative positioning
        elif line_upper.startswith('G91'):
            self.absolute_mode = False
        
        # G92: Set position (coordinate system offset)
        elif line_upper.startswith('G92'):
            coords = self.parse_coordinates(line)
            # Scale coordinates if in inches, then reset position
            sx = coords.get('X')
            sy = coords.get('Y')
            sz = coords.get('Z')
            new_x = (sx * self.units_scale) if sx is not None else self.pos[0]
            new_y = (sy * self.units_scale) if sy is not None else self.pos[1]
            new_z = (sz * self.units_scale) if sz is not None else self.pos[2]
            self.pos = (new_x, new_y, new_z)
            logger.debug(f"G92: Position set to {self.pos}")
        
        # M codes (macros, ignore but estimate time)
        elif line_upper.startswith('M'):
            # Assume purge or pen operations
            if 'PURGE' in line_upper or 'M7' in line_upper or 'M8' in line_upper:
                self.total_time += self.purge_time_s
            elif 'PEN' in line_upper:
                self.total_time += self.pen_time_s
    
    def run(self) -> Dict[str, any]:
        """Execute loaded G-code and return results.
        
        Returns
        -------
        Dict[str, any]
            Results dictionary with keys:
                - time_estimate_s: float (estimated execution time)
                - violations: List[str] (soft-limit violations)
                - last_stroke_id: Optional[str] (last stroke ID)
                - move_count: int (number of moves)
                - final_pos: Tuple[float, float, float] (final position)
        
        Raises
        ------
        RuntimeError
            If no G-code loaded
        """
        if not self.gcode_lines:
            raise RuntimeError("No G-code loaded, call load_file() or load_string() first")
        
        self.reset()
        
        logger.info(f"Running VM on {len(self.gcode_lines)} lines...")
        
        for i, line in enumerate(self.gcode_lines):
            try:
                self.execute_line(line, line_idx=i)
            except Exception as e:
                msg = f"Error at line {i+1}: {line.strip()}: {e}"
                logger.error(msg)
                self.violations.append(msg)
        
        logger.info(f"VM execution complete: {self.move_count} moves, {self.total_time:.1f}s estimated")
        
        if self.violations:
            logger.warning(f"Found {len(self.violations)} violations")
        
        return {
            'time_estimate_s': self.total_time,
            'violations': self.violations,
            'last_stroke_id': self.last_stroke_id,
            'move_count': self.move_count,
            'final_pos': self.pos,
        }
    
    def get_trajectory(self) -> List[Tuple[float, float, float]]:
        """Extract full trajectory from G-code.
        
        Returns
        -------
        List[Tuple[float, float, float]]
            List of (X, Y, Z) positions in order
        
        Notes
        -----
        Useful for visualization/debugging.
        Resets VM state before execution.
        """
        self.reset()
        trajectory = [self.pos]
        
        for line in self.gcode_lines:
            try:
                self.execute_line(line)
                # Record position after each move
                if self.pos != trajectory[-1]:
                    trajectory.append(self.pos)
            except Exception as e:
                logger.warning(f"Failed to parse line for trajectory: {e}")
        
        return trajectory


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def validate_gcode_file(
    gcode_path: Union[str, Path],
    machine_cfg: validators.MachineV1
) -> Dict[str, any]:
    """Validate G-code file and return results.
    
    Parameters
    ----------
    gcode_path : Union[str, Path]
        Path to G-code file
    machine_cfg : validators.MachineV1
        Machine configuration
    
    Returns
    -------
    Dict[str, any]
        VM execution results (see GCodeVM.run())
    
    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If validation fails (soft-limit violations)
    """
    vm = GCodeVM(machine_cfg)
    vm.load_file(gcode_path)
    result = vm.run()
    
    if result['violations']:
        raise ValueError(f"G-code validation failed: {result['violations']}")
    
    return result


def estimate_job_time(
    gcode_path: Union[str, Path],
    machine_cfg: validators.MachineV1,
    purge_time_s: float = 2.0,
    pen_time_s: float = 0.5
) -> float:
    """Estimate total job execution time.
    
    Parameters
    ----------
    gcode_path : Union[str, Path]
        Path to G-code file
    machine_cfg : validators.MachineV1
        Machine configuration
    purge_time_s : float
        Time per purge (seconds), default 2.0
    pen_time_s : float
        Time per pen operation (seconds), default 0.5
    
    Returns
    -------
    float
        Estimated time (seconds)
    """
    vm = GCodeVM(machine_cfg, purge_time_s=purge_time_s, pen_time_s=pen_time_s)
    vm.load_file(gcode_path)
    result = vm.run()
    return result['time_estimate_s']
