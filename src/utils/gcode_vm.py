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
"""

