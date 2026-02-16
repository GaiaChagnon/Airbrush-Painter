#!/usr/bin/env python3
"""Drive the 3-axis robot to trace pen vector paths from the pen tracer.

Loads a ``pen_vectors.yaml`` produced by the pen tracer pipeline,
transforms image-frame coordinates (top-left, +Y down) to machine-frame
coordinates (bottom-left, +Y up), and sends G-code commands via the
Klipper UDS API.

A4 paper in **landscape** orientation (297 mm on X, 210 mm on Y) is
placed with the top-left corner near the endstops (X=0, Y=max).
Portrait images are automatically rotated 90 degrees to fill the paper.
A configurable margin (default 10 mm) is applied on all sides, and a
border rectangle is drawn at the margin boundary before the image.

Alignment workflow:
    1. Home all axes (runs in parallel with YAML loading).
    2. Draw a small circle at each of the 4 border corners (pen down).
    3. Operator aligns the paper to the corner marks.
    4. Press Enter to continue.
    5. Draw the border rectangle.
    6. Draw all pen paths (edges first, then hatching).
    7. Retract and return to safe position.

Usage::

    # List available traced images:
    python robot_control/scripts/run_tracer.py --list

    # Trace a specific image:
    python robot_control/scripts/run_tracer.py --image "peakpx (4)"

    # Dry-run (no robot, just stats):
    python robot_control/scripts/run_tracer.py --image "peakpx (4)" --dry-run

    # Custom Z and speeds:
    python robot_control/scripts/run_tracer.py --image "peakpx (4)" \\
        --z-contact 70.0 --z-retract 1.5 --draw-speed 400
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
import threading
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from robot_control.configs.loader import load_config
from robot_control.configs.printer_cfg import generate_printer_cfg

# ---------------------------------------------------------------------------
# Machine config -- loaded once, all constants derived from it
# ---------------------------------------------------------------------------

_CFG = load_config()

SOCKET_PATH = "/tmp/klippy_uds"
PRINTER_CFG_PATH = Path.home() / "printer.cfg"
ETX = b"\x03"

WORKSPACE_X_MM = _CFG.work_area.x  # 450
WORKSPACE_Y_MM = _CFG.work_area.y  # 320
WORKSPACE_Z_MM = _CFG.work_area.z  # 80

Z_BUFFER_MM = 5.0
Z_MIN_SAFE = Z_BUFFER_MM
Z_MAX_SAFE = WORKSPACE_Z_MM - Z_BUFFER_MM

# A4 paper dimensions (mm) -- landscape: long side on X, short side on Y.
A4_W = 297.0   # long side along X
A4_H = 210.0   # short side along Y

# Default pen traces output directory
DEFAULT_TRACES_DIR = Path(_PROJECT_ROOT) / "outputs" / "pen_traces_hard"


# ---------------------------------------------------------------------------
# Klipper low-level comms (reuse from test_motors.py)
# ---------------------------------------------------------------------------

_next_id = 1


def _raw_send(sock: socket.socket, method: str, params: dict, timeout: float = 10.0) -> dict:
    """Send one JSON request and return the matching response dict."""
    global _next_id
    req_id = _next_id
    _next_id += 1

    payload = json.dumps({"id": req_id, "method": method, "params": params}).encode() + ETX
    sock.sendall(payload)
    sock.settimeout(timeout)

    buf = b""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        while ETX not in buf and time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            sock.settimeout(remaining)
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
            except socket.timeout:
                break

        while ETX in buf:
            idx = buf.index(ETX)
            frame = buf[:idx]
            buf = buf[idx + 1:]
            try:
                msg = json.loads(frame.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if msg.get("id") == req_id:
                return msg

        if ETX not in buf and time.monotonic() >= deadline:
            break

    return {}


def _drain_socket(sock: socket.socket, duration: float = 0.2) -> None:
    """Read and discard any pending data in the socket buffer."""
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        sock.settimeout(max(0.05, deadline - time.monotonic()))
        try:
            data = sock.recv(4096)
            if not data:
                break
        except socket.timeout:
            break


def _raw_gcode(sock: socket.socket, script: str, timeout: float = 30.0) -> bool:
    """Send G-code and return True on success."""
    resp = _raw_send(sock, "gcode/script", {"script": script}, timeout)
    if "error" in resp:
        err = resp["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        print(f"  !! G-code error: {msg[:120]}")
        return False
    return True


def _wait_for_ready(timeout: float = 30.0) -> socket.socket:
    """Connect to Klipper UDS and wait until state is 'ready'."""
    deadline = time.monotonic() + timeout
    restart_attempted = False

    while time.monotonic() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(SOCKET_PATH)
            resp = _raw_send(sock, "info", {})
            result = resp.get("result", {})
            state = result.get("state", "unknown")
            state_msg = result.get("state_message", "")

            if state == "ready":
                return sock

            if state in ("error", "shutdown") and not restart_attempted:
                restart_attempted = True
                print(f"  Klipper state: {state} -- {state_msg[:80]}")
                print("  Attempting FIRMWARE_RESTART...")
                try:
                    _raw_gcode(sock, "FIRMWARE_RESTART")
                except Exception:
                    pass
                time.sleep(3.0)
                continue

            sock.close()
        except OSError:
            pass
        time.sleep(1.0)

    raise RuntimeError(f"Klipper did not become ready within {timeout}s")


def restart_klipper() -> None:
    """Send RESTART via the API and wait for Klipper to come back."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(SOCKET_PATH)
        _raw_gcode(sock, "RESTART")
        sock.close()
    except OSError:
        pass
    time.sleep(3.0)


# ---------------------------------------------------------------------------
# Pen vectors loader (lightweight, no pydantic validation overhead)
# ---------------------------------------------------------------------------


def load_pen_vectors_raw(yaml_path: Path) -> dict:
    """Load pen_vectors.yaml and return the raw dict.

    Uses the C YAML loader for speed on large files (5+ MB).
    """
    import yaml
    try:
        from yaml import CSafeLoader as Loader
    except ImportError:
        from yaml import SafeLoader as Loader

    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=Loader)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")

    schema = data.get("schema", "")
    if schema != "pen_vectors.v1":
        raise ValueError(f"Unsupported schema '{schema}', expected 'pen_vectors.v1'")

    return data


# ---------------------------------------------------------------------------
# Coordinate transform (image-frame -> machine-frame)
# ---------------------------------------------------------------------------


class PaperTransform:
    """Map image coordinates (mm, top-left origin, Y-down) to machine
    coordinates (mm, bottom-left origin, Y-up).

    Parameters
    ----------
    workspace_x, workspace_y : float
        Machine workspace dimensions (mm).
    paper_w, paper_h : float
        Physical paper dimensions as placed on the bed (mm).
        Landscape A4 (default): paper_w=297, paper_h=210.
    margin : float
        Inset from paper edge on all sides (mm).
    image_w, image_h : float
        Image coordinate space dimensions (mm) from pen_vectors.
    paper_origin : tuple[float, float] | None
        Bottom-left corner of the paper in machine coords.
        If None, centres the paper on the workspace.
    """

    def __init__(
        self,
        workspace_x: float,
        workspace_y: float,
        paper_w: float,
        paper_h: float,
        margin: float,
        image_w: float,
        image_h: float,
        paper_origin: tuple[float, float] | None = None,
    ) -> None:
        if paper_origin is not None:
            self.paper_left = paper_origin[0]
            self.paper_bottom = paper_origin[1]
        else:
            self.paper_left = (workspace_x - paper_w) / 2.0
            self.paper_bottom = (workspace_y - paper_h) / 2.0

        # Bounds check
        paper_right = self.paper_left + paper_w
        paper_top = self.paper_bottom + paper_h
        if paper_right > workspace_x or paper_top > workspace_y:
            print(f"  WARNING: paper extends to ({paper_right:.1f}, {paper_top:.1f}) "
                  f"but workspace is ({workspace_x:.1f}, {workspace_y:.1f}). "
                  f"Paths will be clamped.")

        # Drawable area within paper (inset by margin)
        self.draw_left = self.paper_left + margin
        self.draw_bottom = self.paper_bottom + margin
        self.draw_width = paper_w - 2.0 * margin
        self.draw_height = paper_h - 2.0 * margin

        # Uniform scale to fit image into drawable area
        self.scale = min(self.draw_width / image_w, self.draw_height / image_h)
        self.scaled_w = image_w * self.scale
        self.scaled_h = image_h * self.scale

        # Centre offset within drawable area
        self.offset_x = (self.draw_width - self.scaled_w) / 2.0
        self.offset_y = (self.draw_height - self.scaled_h) / 2.0

        # Precompute absolute origin of the scaled image in machine coords
        self.origin_x = self.draw_left + self.offset_x
        self.origin_y = self.draw_bottom + self.offset_y

        # Store for reporting
        self.margin = margin
        self.paper_w = paper_w
        self.paper_h = paper_h
        self.image_w = image_w
        self.image_h = image_h

    def image_to_machine(self, x_img: float, y_img: float) -> tuple[float, float]:
        """Convert image-frame (mm) to machine-frame (mm).

        Parameters
        ----------
        x_img, y_img : float
            Position in image frame (top-left origin, +Y down), in mm.

        Returns
        -------
        tuple[float, float]
            Position in machine frame (bottom-left origin, +Y up), in mm.
        """
        x_scaled = x_img * self.scale
        y_scaled = y_img * self.scale

        # Flip Y: image Y-down -> machine Y-up
        y_flipped = self.scaled_h - y_scaled

        x_mach = self.origin_x + x_scaled
        y_mach = self.origin_y + y_flipped
        return x_mach, y_mach

    def get_border_corners(self) -> list[tuple[float, float]]:
        """Return the 4 corners of the image boundary in machine coords.

        Order: bottom-left, bottom-right, top-right, top-left (CCW).
        """
        corners_img = [
            (0.0, self.image_h),
            (self.image_w, self.image_h),
            (self.image_w, 0.0),
            (0.0, 0.0),
        ]
        return [self.image_to_machine(x, y) for x, y in corners_img]

    def get_margin_corners(self) -> list[tuple[float, float]]:
        """Return the 4 corners of the margin rectangle on the paper.

        Order: bottom-left, bottom-right, top-right, top-left (CCW).
        """
        return [
            (self.draw_left, self.draw_bottom),
            (self.draw_left + self.draw_width, self.draw_bottom),
            (self.draw_left + self.draw_width, self.draw_bottom + self.draw_height),
            (self.draw_left, self.draw_bottom + self.draw_height),
        ]

    def report(self) -> None:
        """Print a human-readable summary of the transform."""
        orient = "portrait" if self.paper_h > self.paper_w else "landscape"
        print(f"  Paper: {self.paper_w:.0f} x {self.paper_h:.0f} mm (A4 {orient})")
        print(f"  Paper bottom-left corner: ({self.paper_left:.1f}, "
              f"{self.paper_bottom:.1f}) mm")
        paper_top_right = (self.paper_left + self.paper_w,
                           self.paper_bottom + self.paper_h)
        print(f"  Paper top-right corner:   ({paper_top_right[0]:.1f}, "
              f"{paper_top_right[1]:.1f}) mm")
        print(f"  Margin: {self.margin:.1f} mm on all sides")
        print(f"  Drawable area: {self.draw_width:.1f} x {self.draw_height:.1f} mm")
        print(f"  Image source: {self.image_w:.1f} x {self.image_h:.1f} mm")
        print(f"  Scale factor: {self.scale:.4f}")
        print(f"  Scaled image: {self.scaled_w:.1f} x {self.scaled_h:.1f} mm")
        print(f"  Image origin on bed: ({self.origin_x:.1f}, {self.origin_y:.1f}) mm")


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------


def pen_up(
    sock: socket.socket,
    z_travel: float,
    z_feedrate: float,
) -> None:
    """Retract pen to travel height."""
    z = max(Z_MIN_SAFE, min(z_travel, Z_MAX_SAFE))
    _raw_gcode(sock, f"G1 Z{z:.2f} F{z_feedrate:.0f}")


def pen_down(
    sock: socket.socket,
    z_contact: float,
    z_feedrate: float,
) -> None:
    """Lower pen to contact height."""
    z = max(Z_MIN_SAFE, min(z_contact, Z_MAX_SAFE))
    _raw_gcode(sock, f"G1 Z{z:.2f} F{z_feedrate:.0f}")


def travel_to(
    sock: socket.socket,
    x: float,
    y: float,
    travel_feedrate: float,
) -> None:
    """Rapid move to (x, y) at travel speed.  Assumes pen is up."""
    x = max(0.0, min(x, WORKSPACE_X_MM))
    y = max(0.0, min(y, WORKSPACE_Y_MM))
    _raw_gcode(sock, f"G1 X{x:.2f} Y{y:.2f} F{travel_feedrate:.0f}")


def draw_to(
    sock: socket.socket,
    x: float,
    y: float,
    draw_feedrate: float,
) -> None:
    """Draw move to (x, y) at drawing speed.  Assumes pen is down."""
    x = max(0.0, min(x, WORKSPACE_X_MM))
    y = max(0.0, min(y, WORKSPACE_Y_MM))
    _raw_gcode(sock, f"G1 X{x:.2f} Y{y:.2f} F{draw_feedrate:.0f}")


def draw_circle_at(
    sock: socket.socket,
    cx: float,
    cy: float,
    radius: float,
    z_contact: float,
    z_travel: float,
    z_down_feedrate: float,
    z_up_feedrate: float,
    draw_feedrate: float,
    travel_feedrate: float,
) -> None:
    """Draw a small circle at (cx, cy) in machine coords."""
    start_x = cx + radius
    start_y = cy
    travel_to(sock, start_x, start_y, travel_feedrate)
    pen_down(sock, z_contact, z_down_feedrate)
    _raw_gcode(
        sock,
        f"G2 X{cx - radius:.2f} Y{cy:.2f} I{-radius:.2f} J0 F{draw_feedrate:.0f}",
    )
    _raw_gcode(
        sock,
        f"G2 X{cx + radius:.2f} Y{cy:.2f} I{radius:.2f} J0 F{draw_feedrate:.0f}",
    )
    pen_up(sock, z_travel, z_up_feedrate)


def draw_rectangle(
    sock: socket.socket,
    corners: list[tuple[float, float]],
    z_contact: float,
    z_travel: float,
    z_down_feedrate: float,
    z_up_feedrate: float,
    draw_feedrate: float,
    travel_feedrate: float,
) -> None:
    """Draw a closed rectangle through the 4 corners."""
    if len(corners) < 4:
        return
    travel_to(sock, corners[0][0], corners[0][1], travel_feedrate)
    pen_down(sock, z_contact, z_down_feedrate)
    for x, y in corners[1:]:
        draw_to(sock, x, y, draw_feedrate)
    draw_to(sock, corners[0][0], corners[0][1], draw_feedrate)
    pen_up(sock, z_travel, z_up_feedrate)


# ---------------------------------------------------------------------------
# Image listing
# ---------------------------------------------------------------------------


def list_available_images(traces_dir: Path) -> list[str]:
    """Return names of available traced images."""
    if not traces_dir.exists():
        return []
    names = []
    for subdir in sorted(traces_dir.iterdir()):
        if subdir.is_dir() and (subdir / "pen_vectors.yaml").exists():
            names.append(subdir.name)
    return names


# ---------------------------------------------------------------------------
# Main trace execution
# ---------------------------------------------------------------------------


def trace_image(
    sock: socket.socket,
    pen_vectors: dict,
    transform: PaperTransform,
    z_contact: float,
    z_retract: float,
    draw_speed_mm_s: float,
    travel_speed_mm_s: float,
    z_plunge_speed_mm_s: float,
    z_retract_speed_mm_s: float,
    accel_mm_s2: float,
    corner_circle_radius: float,
    dry_run: bool = False,
    skip_corners: bool = False,
) -> dict:
    """Execute the full tracing sequence.

    Returns
    -------
    dict
        Execution statistics (path count, travel distance, etc.).
    """
    z_travel = z_contact - z_retract
    draw_feedrate = draw_speed_mm_s * 60.0
    travel_feedrate = travel_speed_mm_s * 60.0
    z_down_feedrate = z_plunge_speed_mm_s * 60.0
    z_up_feedrate = z_retract_speed_mm_s * 60.0

    paths = pen_vectors.get("paths", [])
    corners_raw = transform.get_margin_corners()
    # get_margin_corners returns: BL, BR, TR, TL (CCW from bottom-left).
    # Reorder so corners[0] is nearest home (X=0, Y=YMAX) = top-left.
    corners = [corners_raw[3], corners_raw[2], corners_raw[1], corners_raw[0]]

    # Compute total drawing distance
    total_draw_mm = 0.0
    total_travel_mm = 0.0
    total_pen_lifts = 0

    print()
    print("=" * 60)
    print("  TRACE EXECUTION PLAN")
    print("=" * 60)
    print()
    print(f"  Paths to draw: {len(paths)}")
    print(f"  Z contact: {z_contact:.1f} mm")
    print(f"  Z travel:  {z_travel:.1f} mm  (retract {z_retract:.1f} mm)")
    print(f"  Draw speed: {draw_speed_mm_s:.0f} mm/s")
    print(f"  Travel speed: {travel_speed_mm_s:.0f} mm/s")
    print(f"  Z plunge speed: {z_plunge_speed_mm_s:.0f} mm/s (down)")
    print(f"  Z retract speed: {z_retract_speed_mm_s:.0f} mm/s (up)")
    print(f"  Acceleration:  {accel_mm_s2:.0f} mm/s^2")
    print()

    # Pre-compute all machine-frame paths and stats
    machine_paths: list[list[tuple[float, float]]] = []
    for path in paths:
        pts_mm = path.get("points_mm", [])
        if len(pts_mm) < 2:
            continue
        mpts = [transform.image_to_machine(pt[0], pt[1]) for pt in pts_mm]
        machine_paths.append(mpts)

        for i in range(1, len(mpts)):
            dx = mpts[i][0] - mpts[i - 1][0]
            dy = mpts[i][1] - mpts[i - 1][1]
            total_draw_mm += math.sqrt(dx * dx + dy * dy)

    # Travel distance (between paths)
    prev_end: Optional[tuple[float, float]] = None
    for mpts in machine_paths:
        if prev_end is not None:
            dx = mpts[0][0] - prev_end[0]
            dy = mpts[0][1] - prev_end[1]
            total_travel_mm += math.sqrt(dx * dx + dy * dy)
        prev_end = mpts[-1]
        total_pen_lifts += 1

    # ------------------------------------------------------------------
    # Time estimates using trapezoidal motion profile
    # ------------------------------------------------------------------
    def _trapezoidal_time(dist_mm: float, v_max: float, accel: float) -> float:
        """Time for a move under trapezoidal velocity profile."""
        if dist_mm <= 0 or v_max <= 0:
            return 0.0
        d_ramp = v_max * v_max / accel
        if dist_mm >= d_ramp:
            return dist_mm / v_max + v_max / accel
        return 2.0 * math.sqrt(dist_mm / accel)

    # Per-path drawing time: Klipper look-ahead maintains speed across
    # consecutive G1 junctions, so model each path as one continuous move.
    draw_time_s = 0.0
    for mpts in machine_paths:
        path_len = 0.0
        for i in range(1, len(mpts)):
            dx = mpts[i][0] - mpts[i - 1][0]
            dy = mpts[i][1] - mpts[i - 1][1]
            path_len += math.sqrt(dx * dx + dy * dy)
        draw_time_s += _trapezoidal_time(path_len, draw_speed_mm_s, accel_mm_s2)

    # Travel time between paths
    travel_time_s = 0.0
    prev_end_est: Optional[tuple[float, float]] = None
    for mpts in machine_paths:
        if prev_end_est is not None:
            dx = mpts[0][0] - prev_end_est[0]
            dy = mpts[0][1] - prev_end_est[1]
            seg_len = math.sqrt(dx * dx + dy * dy)
            travel_time_s += _trapezoidal_time(seg_len, travel_speed_mm_s, accel_mm_s2)
        prev_end_est = mpts[-1]

    # Pen up/down
    plunge_time_s = total_pen_lifts * (
        _trapezoidal_time(z_retract, z_plunge_speed_mm_s, accel_mm_s2)
        + _trapezoidal_time(z_retract, z_retract_speed_mm_s, accel_mm_s2)
    )

    # Communication overhead: draw segments batched in groups of 64
    batch_size = 64
    total_sends = sum(
        3 + max(1, math.ceil(max(len(mp) - 1, 1) / batch_size))
        for mp in machine_paths
    )
    comm_overhead_s = total_sends * 0.002

    total_time_s = draw_time_s + travel_time_s + plunge_time_s + comm_overhead_s

    print(f"  Drawing distance:  {total_draw_mm:.0f} mm "
          f"({draw_time_s / 60.0:.1f} min)")
    print(f"  Travel distance:   {total_travel_mm:.0f} mm "
          f"({travel_time_s / 60.0:.1f} min)")
    print(f"  Pen lifts:         {total_pen_lifts}")
    print(f"  Plunge time:       {plunge_time_s / 60.0:.1f} min")
    print(f"  Comms overhead:    {comm_overhead_s / 60.0:.1f} min "
          f"({total_sends} socket sends)")
    print(f"  Estimated total:   {total_time_s / 60.0:.1f} min "
          f"({total_time_s / 3600.0:.1f} hours)")
    print()

    stats = {
        "paths": len(machine_paths),
        "draw_mm": total_draw_mm,
        "travel_mm": total_travel_mm,
        "pen_lifts": total_pen_lifts,
        "estimated_time_min": total_time_s / 60.0,
    }

    if dry_run:
        print("  [DRY RUN] -- no G-code sent.")
        return stats

    # Apply acceleration limit
    _raw_gcode(sock, f"SET_VELOCITY_LIMIT ACCEL={accel_mm_s2:.0f}")
    _raw_gcode(sock, "SET_VELOCITY_LIMIT SQUARE_CORNER_VELOCITY=3")

    # ======================================================================
    # PHASE 0: Pen placement check at first corner
    # ======================================================================
    print("=" * 60)
    print("  PHASE 0: PEN PLACEMENT CHECK")
    print("=" * 60)
    print()

    first_cx, first_cy = corners[0]
    print(f"  Moving to first corner: ({first_cx:.1f}, {first_cy:.1f}) mm")

    pen_up(sock, z_travel, z_up_feedrate)
    travel_to(sock, first_cx, first_cy, travel_feedrate)
    _raw_gcode(sock, "M400")

    print(f"  Lowering Z to contact position ({z_contact:.1f} mm)...")
    pen_down(sock, z_contact, z_down_feedrate)
    _raw_gcode(sock, "M400")

    print()
    print("  >> Insert the pen and verify it touches the paper.")
    print("  >> Adjust --z-contact if needed (higher = pen closer to bed).")
    print()
    input("  Press ENTER when pen depth is correct to continue...")

    pen_up(sock, z_travel, z_up_feedrate)

    # ======================================================================
    # PHASE 1: Alignment circles (skippable)
    # ======================================================================
    if not skip_corners:
        print()
        print("=" * 60)
        print("  PHASE 1: CORNER ALIGNMENT CIRCLES")
        print("=" * 60)
        print()

        for i, (cx, cy) in enumerate(corners):
            label = ["top-left", "top-right", "bottom-right", "bottom-left"][i]
            print(f"  Corner {i + 1}/4 ({label}): ({cx:.1f}, {cy:.1f}) mm")
            draw_circle_at(
                sock, cx, cy, corner_circle_radius,
                z_contact, z_travel, z_down_feedrate, z_up_feedrate,
                draw_feedrate, travel_feedrate,
            )

        _raw_gcode(sock, "M400")
        print()
        print("  All 4 corner circles drawn.")
        print("  >> Align the A4 paper so its corners match the circles.")
        print()
        input("  Press ENTER when paper is aligned to continue...")
    else:
        print()
        print("  [Skipping corner alignment circles]")

    # ======================================================================
    # PHASE 2: Border rectangle
    # ======================================================================
    print()
    print("=" * 60)
    print("  PHASE 2: BORDER RECTANGLE")
    print("=" * 60)
    print()

    draw_rectangle(
        sock, corners,
        z_contact, z_travel, z_down_feedrate, z_up_feedrate,
        draw_feedrate, travel_feedrate,
    )
    _raw_gcode(sock, "M400")
    print("  Border rectangle drawn.")
    print()

    # ======================================================================
    # PHASE 3: Draw all paths
    # ======================================================================
    print("=" * 60)
    print("  PHASE 3: DRAWING IMAGE PATHS")
    print("=" * 60)
    print()

    t_start = time.monotonic()
    prev_pct = -1

    # Batch size: send multiple G1 commands per socket message
    DRAW_BATCH = 64

    for path_idx, mpts in enumerate(machine_paths):
        pct = int(100.0 * (path_idx + 1) / len(machine_paths))
        if pct >= prev_pct + 5 or path_idx == len(machine_paths) - 1:
            elapsed = time.monotonic() - t_start
            sys.stdout.write(
                f"\r  Path {path_idx + 1}/{len(machine_paths)} "
                f"({pct}%)  elapsed: {elapsed / 60.0:.1f} min    "
            )
            sys.stdout.flush()
            prev_pct = pct

        # Travel to start
        travel_to(sock, mpts[0][0], mpts[0][1], travel_feedrate)

        # Pen down
        pen_down(sock, z_contact, z_down_feedrate)

        # Draw polyline in batches
        draw_pts = mpts[1:]
        for batch_start in range(0, len(draw_pts), DRAW_BATCH):
            batch = draw_pts[batch_start:batch_start + DRAW_BATCH]
            lines = []
            for x, y in batch:
                xc = max(0.0, min(x, WORKSPACE_X_MM))
                yc = max(0.0, min(y, WORKSPACE_Y_MM))
                lines.append(f"G1 X{xc:.2f} Y{yc:.2f} F{draw_feedrate:.0f}")
            _raw_gcode(sock, "\n".join(lines))

        # Pen up
        pen_up(sock, z_travel, z_up_feedrate)

    _raw_gcode(sock, "M400")

    elapsed = time.monotonic() - t_start
    print()
    print()
    print(f"  Drawing complete in {elapsed / 60.0:.1f} min "
          f"({elapsed / 3600.0:.1f} hours)")

    # ======================================================================
    # PHASE 4: Return to safe position
    # ======================================================================
    print()
    print("  Returning to safe position...")
    cx = WORKSPACE_X_MM / 2.0
    cy = WORKSPACE_Y_MM / 2.0
    travel_to(sock, cx, cy, travel_feedrate)
    _raw_gcode(sock, "M400")
    print("  Done.")

    stats["actual_time_min"] = elapsed / 60.0
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace pen vector paths on the 3-axis robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image", "-i", type=str, default=None,
        help="Name of the traced image (subdirectory in traces dir)",
    )
    parser.add_argument(
        "--vectors", type=str, default=None,
        help="Direct path to pen_vectors.yaml (overrides --image)",
    )
    parser.add_argument(
        "--traces-dir", type=str, default=str(DEFAULT_TRACES_DIR),
        help=f"Directory containing traced image outputs (default: {DEFAULT_TRACES_DIR})",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available traced images and exit",
    )

    # Z-axis parameters
    z_group = parser.add_argument_group("Z-axis parameters")
    z_group.add_argument(
        "--z-contact", type=float, default=70.0,
        help="Z position where pen touches paper, mm (default: 70.0, near endstop at 80)",
    )
    z_group.add_argument(
        "--z-retract", type=float, default=1.5,
        help="How far to retract pen above contact, mm (default: 1.5)",
    )

    # Speed parameters
    speed_group = parser.add_argument_group("Speed parameters")
    speed_group.add_argument(
        "--draw-speed", type=float, default=150.0,
        help="Drawing speed, mm/s (default: 150)",
    )
    speed_group.add_argument(
        "--travel-speed", type=float, default=400.0,
        help="Travel (non-drawing) speed, mm/s (default: 400)",
    )
    speed_group.add_argument(
        "--z-plunge-speed", type=float, default=20.0,
        help="Z plunge (pen down) speed, mm/s (default: 20)",
    )
    speed_group.add_argument(
        "--z-retract-speed", type=float, default=50.0,
        help="Z retract (pen up) speed, mm/s (default: 50)",
    )
    speed_group.add_argument(
        "--accel", type=float, default=2000.0,
        help="XY acceleration limit, mm/s^2 (default: 2000)",
    )

    # Paper and alignment
    paper_group = parser.add_argument_group("Paper and alignment")
    paper_group.add_argument(
        "--paper-origin", type=float, nargs=2, default=[25.0, 25.0],
        metavar=("X", "Y"),
        help="Offset from limit switches (X from X_min, Y from Y_max), mm (default: 25 25)",
    )
    paper_group.add_argument(
        "--margin", type=float, default=10.0,
        help="Margin from paper edge on all sides, mm (default: 10.0)",
    )
    paper_group.add_argument(
        "--corner-radius", type=float, default=3.0,
        help="Radius of corner alignment circles, mm (default: 3.0)",
    )

    # Operational flags
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute stats without sending G-code to robot",
    )
    parser.add_argument(
        "--no-config-write", action="store_true",
        help="Don't overwrite printer.cfg (assume correct config is loaded)",
    )
    parser.add_argument(
        "--skip-corners", action="store_true",
        help="Skip corner alignment circles (if paper is already placed)",
    )

    args = parser.parse_args()

    # --- List mode ---------------------------------------------------------
    traces_dir = Path(args.traces_dir)
    if args.list:
        images = list_available_images(traces_dir)
        if not images:
            print(f"No traced images found in {traces_dir}")
            print("Run the pen tracer first:")
            print(f"  python run_pen_tracer_test.py --input data/raw_images/hard "
                  f"--output {traces_dir}")
        else:
            print(f"Available traced images in {traces_dir}:")
            for name in images:
                vectors_path = traces_dir / name / "pen_vectors.yaml"
                size_mb = vectors_path.stat().st_size / (1024 * 1024)
                print(f"  {name}  ({size_mb:.1f} MB)")
        return

    # --- Resolve pen vectors path ------------------------------------------
    if args.vectors:
        vectors_path = Path(args.vectors)
    elif args.image:
        vectors_path = traces_dir / args.image / "pen_vectors.yaml"
    else:
        parser.error("Specify --image NAME or --vectors PATH (use --list to see available)")
        return

    if not vectors_path.exists():
        print(f"ERROR: pen_vectors.yaml not found at {vectors_path}")
        available = list_available_images(traces_dir)
        if available:
            print(f"Available images: {', '.join(available)}")
        sys.exit(1)

    # --- Start loading pen vectors in background ----------------------------
    print("=" * 60)
    print("  PEN TRACER - ROBOT EXECUTION")
    print("=" * 60)
    print()
    print(f"  Loading: {vectors_path}")

    load_result: dict = {}
    load_error: list = []

    def _load_vectors_thread() -> None:
        try:
            t0 = time.monotonic()
            data = load_pen_vectors_raw(vectors_path)
            elapsed = time.monotonic() - t0
            load_result["data"] = data
            load_result["elapsed_s"] = elapsed
        except Exception as exc:
            load_error.append(exc)

    loader_thread = threading.Thread(target=_load_vectors_thread, daemon=True)
    loader_thread.start()

    # --- Robot setup runs concurrently with YAML loading -------------------
    sock: Optional[socket.socket] = None
    if not args.dry_run:
        print("  [Robot setup running in parallel with YAML loading]")
        print()

        print("  Robot workspace:")
        print(f"    X: 0 .. {WORKSPACE_X_MM:.0f} mm")
        print(f"    Y: 0 .. {WORKSPACE_Y_MM:.0f} mm")
        print(f"    Z: {Z_MIN_SAFE:.0f} .. {Z_MAX_SAFE:.0f} mm (safe range)")
        print()

        if not args.no_config_write:
            if PRINTER_CFG_PATH.exists():
                backup = PRINTER_CFG_PATH.with_suffix(".cfg.bak")
                PRINTER_CFG_PATH.rename(backup)
                print(f"  Backed up existing printer.cfg -> {backup}")

            config_text = generate_printer_cfg(_CFG)
            PRINTER_CFG_PATH.write_text(config_text)
            print(f"  Wrote printer.cfg to {PRINTER_CFG_PATH}")
            print()

            print("  Restarting Klipper...")
            restart_klipper()

        print("  Waiting for Klipper to become ready...")
        sock = _wait_for_ready(timeout=30.0)
        print("  [OK] Klipper is ready")
        print()

        print("  Homing X, Y...")
        ok_xy = _raw_gcode(sock, "G28 X Y", timeout=60.0)
        if not ok_xy:
            print("  ERROR: XY homing failed!")
            sock.close()
            sys.exit(1)
        print(f"    Homed OK  X=0  Y={WORKSPACE_Y_MM:.0f}")

        print("  Homing Z...")
        ok_z = _raw_gcode(sock, "G28 Z", timeout=60.0)
        if not ok_z:
            print("  ERROR: Z homing failed!")
            sock.close()
            sys.exit(1)
        print(f"    Homed OK  Z={WORKSPACE_Z_MM:.0f}")

        z_travel_val = args.z_contact - args.z_retract
        z_travel_val = max(Z_MIN_SAFE, min(z_travel_val, Z_MAX_SAFE))
        z_up_fr = args.z_retract_speed * 60.0
        pen_up(sock, z_travel_val, z_up_fr)
        _raw_gcode(sock, "M400")
        print(f"    Z retracted to {z_travel_val:.1f} mm")
        print()

    # --- Wait for YAML loading to finish -----------------------------------
    loader_thread.join()
    if load_error:
        print(f"  ERROR loading pen vectors: {load_error[0]}")
        if sock:
            sock.close()
        sys.exit(1)

    pen_vectors = load_result["data"]
    load_elapsed = load_result.get("elapsed_s", 0)
    work_area = pen_vectors["work_area_mm"]
    paths = pen_vectors.get("paths", [])

    print(f"  Vectors loaded in {load_elapsed:.1f}s")
    print(f"  Image work area: {work_area[0]:.0f} x {work_area[1]:.0f} mm")
    print(f"  Total paths: {len(paths)}")
    n_edges = sum(1 for p in paths if p.get("role") == "outline")
    n_hatch = sum(1 for p in paths if p.get("role") == "hatch")
    print(f"    Edges: {n_edges}  |  Hatching: {n_hatch}")
    print()

    # --- Rotate portrait image to landscape if needed -----------------------
    # Pen vectors are generated in A4 portrait (210 x 297 mm).  The paper
    # is placed landscape (297 on X, 210 on Y).  Rotate the image 90 deg CW
    # so it fills the landscape paper instead of being letter-boxed.
    img_w, img_h = work_area[0], work_area[1]
    if img_h > img_w:
        print("  Rotating image 90\u00b0 CW to match landscape paper")
        for path in paths:
            pts = path["points_mm"]
            path["points_mm"] = [[pt[1], img_w - pt[0]] for pt in pts]
        work_area = [img_h, img_w]  # now [297, 210]
        print(f"  Rotated work area: {work_area[0]:.0f} x {work_area[1]:.0f} mm")

    # --- Build transform ---------------------------------------------------
    # Paper is placed in LANDSCAPE: long side (297 mm) along X, short
    # (210 mm) along Y.  --paper-origin gives the offset from endstops:
    #   X endstop is at X=0        -> paper_left = ox
    #   Y endstop is at Y=Y_MAX   -> paper_top  = workspace_y - oy
    #   Paper extends right and downward from that corner.
    ox, oy = args.paper_origin
    paper_left = ox
    paper_top = WORKSPACE_Y_MM - oy
    paper_bottom = paper_top - A4_H  # 210 mm (short side on Y)

    # Auto-clamp: if paper extends below Y=0, shift it up
    if paper_bottom < 0:
        paper_bottom = 0.0
        paper_top = A4_H
        effective_oy = WORKSPACE_Y_MM - paper_top
        print(f"  NOTE: Y offset auto-adjusted to {effective_oy:.0f} mm "
              f"(requested {oy:.0f} exceeds workspace for {A4_H:.0f} mm paper)")

    paper_origin = (paper_left, paper_bottom)

    xform = PaperTransform(
        workspace_x=WORKSPACE_X_MM,
        workspace_y=WORKSPACE_Y_MM,
        paper_w=A4_W,           # 297 mm on X (long side)
        paper_h=A4_H,           # 210 mm on Y (short side)
        margin=args.margin,
        image_w=work_area[0],
        image_h=work_area[1],
        paper_origin=paper_origin,
    )

    print("  Coordinate transform:")
    xform.report()
    print()

    # --- Validate Z parameters ---------------------------------------------
    z_travel = args.z_contact - args.z_retract
    if z_travel < Z_MIN_SAFE:
        print(f"  WARNING: Z travel ({z_travel:.1f}) below safe minimum "
              f"({Z_MIN_SAFE:.1f}), clamping.")
    if args.z_contact > Z_MAX_SAFE:
        print(f"  WARNING: Z contact ({args.z_contact:.1f}) above safe maximum "
              f"({Z_MAX_SAFE:.1f}), clamping.")
        args.z_contact = Z_MAX_SAFE

    # --- Validate margins fit on bed ---------------------------------------
    margin_corners = xform.get_margin_corners()
    for cx, cy in margin_corners:
        if cx < 0 or cx > WORKSPACE_X_MM or cy < 0 or cy > WORKSPACE_Y_MM:
            print(f"  ERROR: Margin corner ({cx:.1f}, {cy:.1f}) outside workspace!")
            if sock:
                sock.close()
            sys.exit(1)

    # --- Dry run -----------------------------------------------------------
    if args.dry_run:
        trace_image(
            sock=None,  # type: ignore[arg-type]
            pen_vectors=pen_vectors,
            transform=xform,
            z_contact=args.z_contact,
            z_retract=args.z_retract,
            draw_speed_mm_s=args.draw_speed,
            travel_speed_mm_s=args.travel_speed,
            z_plunge_speed_mm_s=args.z_plunge_speed,
            z_retract_speed_mm_s=args.z_retract_speed,
            accel_mm_s2=args.accel,
            corner_circle_radius=args.corner_radius,
            dry_run=True,
            skip_corners=args.skip_corners,
        )
        return

    # --- Robot execution (homing already done above in parallel) ------------
    assert sock is not None, "Socket should be connected for non-dry-run"

    try:
        stats = trace_image(
            sock=sock,
            pen_vectors=pen_vectors,
            transform=xform,
            z_contact=args.z_contact,
            z_retract=args.z_retract,
            draw_speed_mm_s=args.draw_speed,
            travel_speed_mm_s=args.travel_speed,
            z_plunge_speed_mm_s=args.z_plunge_speed,
            z_retract_speed_mm_s=args.z_retract_speed,
            accel_mm_s2=args.accel,
            corner_circle_radius=args.corner_radius,
            dry_run=False,
            skip_corners=args.skip_corners,
        )

        _raw_gcode(sock, "M18")

        print()
        print("=" * 60)
        print("  TRACE COMPLETE")
        print("=" * 60)
        print()
        print(f"  Paths drawn:     {stats['paths']}")
        print(f"  Drawing dist:    {stats['draw_mm']:.0f} mm")
        print(f"  Travel dist:     {stats['travel_mm']:.0f} mm")
        print(f"  Pen lifts:       {stats['pen_lifts']}")
        if "actual_time_min" in stats:
            print(f"  Actual time:     {stats['actual_time_min']:.1f} min")
        print()

    except KeyboardInterrupt:
        print("\n\n  Interrupted -- retracting pen and disabling motors...")
        z_travel_val = args.z_contact - args.z_retract
        z_travel_val = max(Z_MIN_SAFE, min(z_travel_val, Z_MAX_SAFE))
        z_up_fr = args.z_retract_speed * 60.0
        try:
            pen_up(sock, z_travel_val, z_up_fr)
            _raw_gcode(sock, "M400")
            payload = (
                json.dumps({
                    "id": 9999,
                    "method": "gcode/script",
                    "params": {"script": "M18"},
                }).encode() + ETX
            )
            sock.settimeout(1.0)
            sock.sendall(payload)
        except Exception:
            pass

    finally:
        sock.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        try:
            _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            _sock.settimeout(2.0)
            _sock.connect(SOCKET_PATH)
            _payload = (
                json.dumps({
                    "id": 9998,
                    "method": "gcode/script",
                    "params": {"script": "M18"},
                }).encode() + ETX
            )
            _sock.sendall(_payload)
            _sock.close()
        except OSError:
            pass
        sys.exit(1)
