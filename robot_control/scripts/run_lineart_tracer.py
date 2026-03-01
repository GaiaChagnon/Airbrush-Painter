#!/usr/bin/env python3
"""Standalone line-art tracer, hatched-fill, and flow-imager for the robot pen plotter.

Three operating modes:

**line_tracing** (default)
    Potrace outlines with no fill -- faithful B&W replication of
    schematics, text, and engineering drawings.

**hatched**
    Density-based hatching with gradient support -- converts photos
    and grayscale art into plotter-friendly hatch patterns.

**flow_imager**
    Flow-field streamlines driven by image brightness -- artistic
    line-art rendering with configurable density and field type.

Usage::

    # Line-tracing dry-run:
    python robot_control/scripts/run_lineart_tracer.py \\
        --image "Syringe Pump Drawing v1.png" --dry-run --save-preview

    # Hatched mode with gradient fill:
    python robot_control/scripts/run_lineart_tracer.py \\
        --mode hatched --image-path data/raw_images/hard/peakpx.jpg \\
        --dry-run --save-preview

    # Flow-imager mode:
    python robot_control/scripts/run_lineart_tracer.py \\
        --mode flow_imager --image-path data/raw_images/hard/peakpx.jpg \\
        --dry-run --save-preview

    # Execute on robot:
    python robot_control/scripts/run_lineart_tracer.py \\
        --image "Syringe Pump Drawing v1.png"
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import potrace
import vpype
from hatched.hatched import _build_hatch, _load_image
from PIL import Image
from vpype_flow_imager.vpype_flow_imager import draw_image as flow_draw_image

# ---------------------------------------------------------------------------
# Project root and config loading
# ---------------------------------------------------------------------------

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from robot_control.configs.loader import load_config
from robot_control.configs.printer_cfg import generate_printer_cfg

# ---------------------------------------------------------------------------
# Machine config -- loaded once, all constants derived from it
# ---------------------------------------------------------------------------

_CFG = load_config()

SOCKET_PATH = _CFG.connection.socket_path
PRINTER_CFG_PATH = Path.home() / "printer.cfg"
ETX = b"\x03"

WORKSPACE_X_MM = _CFG.work_area.x
WORKSPACE_Y_MM = _CFG.work_area.y
WORKSPACE_Z_MM = _CFG.work_area.z
Z_MIN_SAFE = _CFG.work_area.soft_limits.z_min
Z_MAX_SAFE = _CFG.work_area.soft_limits.z_max_with_overtravel

DEFAULT_IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
DEFAULT_PREVIEW_DIR = DEFAULT_IMAGES_DIR / "previews"
DEFAULT_OUTPUT_DIR = Path(_PROJECT_ROOT) / "outputs" / "lineart_traces"

# ---------------------------------------------------------------------------
# Job config (jobs.yaml)
# ---------------------------------------------------------------------------

_JOBS_YAML = Path(__file__).resolve().parent.parent / "configs" / "jobs.yaml"


def _load_jobs_config() -> dict:
    """Load jobs.yaml and return the raw dict."""
    import yaml
    try:
        from yaml import CSafeLoader as Loader
    except ImportError:
        from yaml import SafeLoader as Loader
    with open(_JOBS_YAML, "r") as f:
        return yaml.load(f, Loader=Loader)


_JOBS = _load_jobs_config()
_PAPER = _JOBS.get("paper", {})
_PEN = _JOBS.get("pen_tracer", {})

# ---------------------------------------------------------------------------
# Lineart tracer config (lineart.yaml)
# ---------------------------------------------------------------------------

_LINEART_YAML = Path(__file__).resolve().parent.parent / "configs" / "lineart.yaml"


def _load_lineart_config() -> dict:
    """Load lineart.yaml and return the raw dict."""
    import yaml
    try:
        from yaml import CSafeLoader as Loader
    except ImportError:
        from yaml import SafeLoader as Loader
    if not _LINEART_YAML.exists():
        return {}
    with open(_LINEART_YAML, "r") as f:
        return yaml.load(f, Loader=Loader) or {}


_LINEART = _load_lineart_config()
_LINEART_LT = _LINEART.get("line_tracing", {})
_LINEART_HATCH = _LINEART.get("hatched", {})
_LINEART_HTNG = _LINEART.get("hatching", {})
_LINEART_FLOW = _LINEART.get("flow_imager", {})
_LINE_WIDTH_MM = float(_LINEART.get("line_width_mm", 0.3))


# ===========================================================================
# Klipper low-level communication (inlined from run_tracer.py)
# ===========================================================================

_next_id = 1


def _raw_send(
    sock: socket.socket, method: str, params: dict, timeout: float = 10.0,
) -> dict:
    """Send one JSON-RPC request to Klipper and return the response."""
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


def _query_mesh_status(sock: socket.socket) -> dict:
    """Query Klipper's bed_mesh status object.

    Returns the 'result' dict from Klipper's objects/query endpoint,
    or an empty dict on failure.  The status includes 'profile_name',
    'mesh_min', 'mesh_max', and 'probed_matrix' when a mesh is active.
    """
    resp = _raw_send(
        sock, "objects/query",
        {"objects": {"bed_mesh": None}},
        timeout=5.0,
    )
    status = resp.get("result", {}).get("status", {}).get("bed_mesh", {})
    return status


def _verify_mesh_loaded(
    sock: socket.socket,
    corners: list[tuple[float, float]],
    z_contact: float,
) -> bool:
    """Check that a bed mesh is active and print its compensation at corners.

    Parameters
    ----------
    sock : socket.socket
        Connected Klipper UDS socket.
    corners : list[tuple[float, float]]
        Corner positions to report mesh offsets for (BL, BR, TR, TL).
    z_contact : float
        Nominal contact Z (mm).

    Returns
    -------
    bool
        True if mesh is active, False otherwise.
    """
    status = _query_mesh_status(sock)
    profile = status.get("profile_name", "")

    if not profile:
        print("    BED MESH: ** NOT ACTIVE **")
        print("    No mesh compensation will be applied.")
        return False

    mesh_min = status.get("mesh_min", [0.0, 0.0])
    mesh_max = status.get("mesh_max", [0.0, 0.0])
    probed = status.get("probed_matrix", [])
    mesh_matrix = status.get("mesh_matrix", [])

    print(f"    BED MESH: active  profile='{profile}'")
    print(f"    Mesh range: ({mesh_min[0]:.1f}, {mesh_min[1]:.1f}) -> "
          f"({mesh_max[0]:.1f}, {mesh_max[1]:.1f})")

    if probed:
        flat = [v for row in probed for v in row]
        print(f"    Probed range: [{min(flat):.4f}, {max(flat):.4f}] mm  "
              f"({len(probed)}x{len(probed[0])} grid)")

    if mesh_matrix:
        flat_m = [v for row in mesh_matrix for v in row]
        print(f"    Interpolated range: [{min(flat_m):.4f}, {max(flat_m):.4f}] mm  "
              f"({len(mesh_matrix)}x{len(mesh_matrix[0])} grid)")

    labels = ["BL", "BR", "TR", "TL"]
    if mesh_matrix and len(mesh_matrix) > 0:
        ny = len(mesh_matrix)
        nx = len(mesh_matrix[0])
        x_min, y_min = mesh_min
        x_max, y_max = mesh_max
        x_dist = (x_max - x_min) / max(nx - 1, 1)
        y_dist = (y_max - y_min) / max(ny - 1, 1)

        print(f"    Corner mesh offsets (Z_adjusted = {z_contact:.3f} + offset):")
        for label, (cx, cy) in zip(labels, corners):
            # Bilinear interpolation matching Klipper's _get_linear_index
            xi_raw = (cx - x_min) / x_dist if x_dist > 0 else 0.0
            yi_raw = (cy - y_min) / y_dist if y_dist > 0 else 0.0
            xi = max(0, min(int(xi_raw), nx - 2))
            yi = max(0, min(int(yi_raw), ny - 2))
            tx = max(0.0, min(xi_raw - xi, 1.0))
            ty = max(0.0, min(yi_raw - yi, 1.0))
            z0 = mesh_matrix[yi][xi] * (1 - tx) + mesh_matrix[yi][xi + 1] * tx
            z1 = mesh_matrix[yi + 1][xi] * (1 - tx) + mesh_matrix[yi + 1][xi + 1] * tx
            offset = z0 * (1 - ty) + z1 * ty
            inside = "inside" if (x_min <= cx <= x_max and y_min <= cy <= y_max) else "OUTSIDE"
            print(f"      {label} ({cx:.0f}, {cy:.0f}): offset={offset:+.4f} mm  "
                  f"-> Z={z_contact + offset:.3f}  [{inside} mesh]")

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


# ===========================================================================
# Image processing
# ===========================================================================


# ===========================================================================
# Path processing
# ===========================================================================


def simplify_paths(
    paths: list[np.ndarray],
    tolerance_px: float,
) -> list[np.ndarray]:
    """Apply Douglas-Peucker simplification to each polyline.

    Parameters
    ----------
    paths : list[np.ndarray]
        Polylines as ``(N, 2)`` arrays (pixel or mm coords).
    tolerance_px : float
        Maximum deviation (same units as *paths*).

    Returns
    -------
    list[np.ndarray]
        Simplified polylines.
    """
    out: list[np.ndarray] = []
    for pts in paths:
        if len(pts) < 3:
            out.append(pts)
            continue
        approx = cv2.approxPolyDP(
            pts.astype(np.float32).reshape(-1, 1, 2),
            epsilon=tolerance_px,
            closed=False,
        )
        simplified = approx.reshape(-1, 2).astype(np.float64)
        if len(simplified) >= 2:
            out.append(simplified)
    return out

# ===========================================================================
# Coordinate transform (inlined from run_tracer.py)
# ===========================================================================


class PaperTransform:
    """Map image coords (mm, top-left, Y-down) to machine coords (mm,
    bottom-left, Y-up).

    Parameters
    ----------
    workspace_x, workspace_y : float
        Machine workspace dimensions (mm).
    paper_w, paper_h : float
        Physical paper on the bed (mm), landscape A4 = 297 x 210.
    margin : float
        Inset from paper edge on all sides (mm).
    image_w, image_h : float
        Image coordinate space dimensions (mm).
    paper_origin : tuple[float, float] or None
        Bottom-left corner of the paper in machine coords.
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
        mirror_x: bool = False,
    ) -> None:
        if paper_origin is not None:
            self.paper_left = paper_origin[0]
            self.paper_bottom = paper_origin[1]
        else:
            self.paper_left = (workspace_x - paper_w) / 2.0
            self.paper_bottom = (workspace_y - paper_h) / 2.0

        paper_right = self.paper_left + paper_w
        paper_top = self.paper_bottom + paper_h
        if paper_right > workspace_x or paper_top > workspace_y:
            print(
                f"  WARNING: paper extends to ({paper_right:.1f}, "
                f"{paper_top:.1f}) but workspace is ({workspace_x:.1f}, "
                f"{workspace_y:.1f}).  Paths will be clamped."
            )

        self.draw_left = self.paper_left + margin
        self.draw_bottom = self.paper_bottom + margin
        self.draw_width = paper_w - 2.0 * margin
        self.draw_height = paper_h - 2.0 * margin

        self.scale = min(self.draw_width / image_w, self.draw_height / image_h)
        self.scaled_w = image_w * self.scale
        self.scaled_h = image_h * self.scale

        self.offset_x = (self.draw_width - self.scaled_w) / 2.0
        self.offset_y = (self.draw_height - self.scaled_h) / 2.0

        self.origin_x = self.draw_left + self.offset_x
        self.origin_y = self.draw_bottom + self.offset_y

        self.margin = margin
        self.paper_w = paper_w
        self.paper_h = paper_h
        self.image_w = image_w
        self.image_h = image_h
        self.mirror_x = mirror_x

    def image_to_machine(
        self, x_img: float, y_img: float,
    ) -> tuple[float, float]:
        """Convert image-frame mm to machine-frame mm.

        Image frame: top-left origin, +Y down.
        Machine frame: bottom-left origin, +Y up.
        When ``mirror_x`` is True, the X axis is flipped so the image
        is horizontally mirrored on paper.
        """
        x_scaled = x_img * self.scale
        y_scaled = y_img * self.scale
        if self.mirror_x:
            x_scaled = self.scaled_w - x_scaled
        y_flipped = self.scaled_h - y_scaled
        return self.origin_x + x_scaled, self.origin_y + y_flipped

    def get_paper_corners(self) -> list[tuple[float, float]]:
        """4 paper corners in machine coords (BL, BR, TR, TL -- CCW)."""
        return [
            (self.paper_left, self.paper_bottom),
            (self.paper_left + self.paper_w, self.paper_bottom),
            (self.paper_left + self.paper_w, self.paper_bottom + self.paper_h),
            (self.paper_left, self.paper_bottom + self.paper_h),
        ]

    def get_margin_corners(self) -> list[tuple[float, float]]:
        """4 margin-rectangle corners in machine coords (BL, BR, TR, TL)."""
        return [
            (self.draw_left, self.draw_bottom),
            (self.draw_left + self.draw_width, self.draw_bottom),
            (self.draw_left + self.draw_width, self.draw_bottom + self.draw_height),
            (self.draw_left, self.draw_bottom + self.draw_height),
        ]

    def report(self) -> None:
        """Print a human-readable summary."""
        orient = "portrait" if self.paper_h > self.paper_w else "landscape"
        print(f"  Paper: {self.paper_w:.0f} x {self.paper_h:.0f} mm (A4 {orient})")
        print(f"  Paper bottom-left: ({self.paper_left:.1f}, {self.paper_bottom:.1f}) mm")
        ptr = (self.paper_left + self.paper_w, self.paper_bottom + self.paper_h)
        print(f"  Paper top-right:   ({ptr[0]:.1f}, {ptr[1]:.1f}) mm")
        print(f"  Margin: {self.margin:.1f} mm")
        print(f"  Drawable area: {self.draw_width:.1f} x {self.draw_height:.1f} mm")
        print(f"  Image source: {self.image_w:.1f} x {self.image_h:.1f} mm")
        print(f"  Scale: {self.scale:.6f}")
        print(f"  Scaled image: {self.scaled_w:.1f} x {self.scaled_h:.1f} mm")
        print(f"  Image origin on bed: ({self.origin_x:.1f}, {self.origin_y:.1f}) mm")


# ===========================================================================
# Motion helpers
# ===========================================================================


def pen_up(sock: socket.socket, z_travel: float, z_feedrate: float) -> None:
    """Retract pen to travel height."""
    z = max(Z_MIN_SAFE, min(z_travel, Z_MAX_SAFE))
    _raw_gcode(sock, f"G1 Z{z:.3f} F{z_feedrate:.0f}")


def pen_down(sock: socket.socket, z_contact: float, z_feedrate: float) -> None:
    """Lower pen to contact height."""
    z = max(Z_MIN_SAFE, min(z_contact, Z_MAX_SAFE))
    _raw_gcode(sock, f"G1 Z{z:.3f} F{z_feedrate:.0f}")


def travel_to(
    sock: socket.socket, x: float, y: float, feedrate: float,
) -> None:
    """Rapid XY move (pen assumed up)."""
    x = max(0.0, min(x, WORKSPACE_X_MM))
    y = max(0.0, min(y, WORKSPACE_Y_MM))
    _raw_gcode(sock, f"G1 X{x:.2f} Y{y:.2f} F{feedrate:.0f}")


def draw_to(
    sock: socket.socket, x: float, y: float, feedrate: float,
) -> None:
    """Drawing XY move (pen assumed down)."""
    x = max(0.0, min(x, WORKSPACE_X_MM))
    y = max(0.0, min(y, WORKSPACE_Y_MM))
    _raw_gcode(sock, f"G1 X{x:.2f} Y{y:.2f} F{feedrate:.0f}")


def draw_circle_at(
    sock: socket.socket,
    cx: float,
    cy: float,
    radius: float,
    z_contact: float,
    z_travel: float,
    z_down_fr: float,
    z_up_fr: float,
    draw_fr: float,
    travel_fr: float,
) -> None:
    """Draw a small alignment circle at (cx, cy) in machine coords."""
    start_x = cx + radius
    travel_to(sock, start_x, cy, travel_fr)
    pen_down(sock, z_contact, z_down_fr)
    _raw_gcode(
        sock,
        f"G2 X{cx - radius:.2f} Y{cy:.2f} I{-radius:.2f} J0 F{draw_fr:.0f}",
    )
    _raw_gcode(
        sock,
        f"G2 X{cx + radius:.2f} Y{cy:.2f} I{radius:.2f} J0 F{draw_fr:.0f}",
    )
    pen_up(sock, z_travel, z_up_fr)


def draw_rectangle(
    sock: socket.socket,
    corners: list[tuple[float, float]],
    z_contact: float,
    z_travel: float,
    z_down_fr: float,
    z_up_fr: float,
    draw_fr: float,
    travel_fr: float,
) -> None:
    """Draw a closed rectangle through 4 corners."""
    if len(corners) < 4:
        return
    travel_to(sock, corners[0][0], corners[0][1], travel_fr)
    pen_down(sock, z_contact, z_down_fr)
    for x, y in corners[1:]:
        draw_to(sock, x, y, draw_fr)
    draw_to(sock, corners[0][0], corners[0][1], draw_fr)
    pen_up(sock, z_travel, z_up_fr)


# ===========================================================================
# Preview / export helpers
# ===========================================================================


def save_preview(
    paths_px: list[np.ndarray],
    H: int,
    W: int,
    output_path: Path,
) -> None:
    """Render vectorised paths onto a white canvas and save as PNG.

    Parameters
    ----------
    paths_px : list[np.ndarray]
        Polylines in **(x, y)** pixel coordinates.
    H, W : int
        Canvas dimensions.
    output_path : Path
        Destination file.
    """
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    for pts in paths_px:
        ipts = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [ipts], isClosed=False, color=(0, 0, 0), thickness=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"  Preview saved: {output_path}")


def render_machine_preview(
    machine_paths: list[list[tuple[float, float]]],
    transform: PaperTransform,
    output_path: Path,
    dpi: float = 10.0,
    stats: dict | None = None,
) -> None:
    """Render a to-scale preview in machine coordinates showing paper,
    margins, and drawing paths exactly as they will appear on the bed.

    Parameters
    ----------
    machine_paths : list[list[tuple[float, float]]]
        Polylines in machine-frame mm.
    transform : PaperTransform
        Provides paper/margin geometry.
    output_path : Path
        Destination PNG file.
    dpi : float
        Pixels per mm in the preview image.
    stats : dict or None
        If provided, execution stats are annotated on the image.
    """
    # Canvas covers the full paper (not the whole workspace)
    pw = transform.paper_w
    ph = transform.paper_h
    pl = transform.paper_left
    pb = transform.paper_bottom

    canvas_w = int(math.ceil(pw * dpi))
    canvas_h = int(math.ceil(ph * dpi))
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    def mm_to_canvas(x_mm: float, y_mm: float) -> tuple[int, int]:
        # machine mm -> paper-relative mm -> canvas px (Y-flip for image coords)
        cx = int(round((x_mm - pl) * dpi))
        cy = int(round((pb + ph - y_mm) * dpi))  # flip Y
        return cx, cy

    # Scale factor relative to the 10 px/mm baseline
    sf = dpi / 10.0
    line_w = max(1, int(round(sf)))

    # Draw paper border (light grey)
    paper_corners = transform.get_paper_corners()
    paper_pts = np.array(
        [mm_to_canvas(x, y) for x, y in paper_corners], dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.polylines(canvas, [paper_pts], isClosed=True,
                  color=(200, 200, 200), thickness=max(2, int(2 * sf)))

    # Draw margin boundary (medium grey)
    margin_corners = transform.get_margin_corners()
    margin_pts = np.array(
        [mm_to_canvas(x, y) for x, y in margin_corners], dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.polylines(canvas, [margin_pts], isClosed=True,
                  color=(160, 160, 160), thickness=line_w)

    # Draw all vectorised paths (black)
    for mpts in machine_paths:
        if len(mpts) < 2:
            continue
        pts = np.array(
            [mm_to_canvas(x, y) for x, y in mpts], dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False,
                      color=(0, 0, 0), thickness=line_w)

    # Annotate dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4 * sf
    font_color = (120, 120, 120)
    font_thick = max(1, int(sf))
    line_spacing = int(14 * sf)

    # Paper size label (bottom-left)
    label = f"{pw:.0f} x {ph:.0f} mm"
    cv2.putText(canvas, label, (int(5 * sf), canvas_h - int(8 * sf)),
                font, font_scale, font_color, font_thick)

    # Margin label (inside margin top-left)
    mx, my = mm_to_canvas(
        transform.draw_left + 1.0,
        transform.draw_bottom + transform.draw_height - 1.0,
    )
    cv2.putText(
        canvas, f"margin {transform.margin:.0f} mm",
        (mx, my + int(12 * sf)), font, font_scale * 0.9,
        font_color, font_thick,
    )

    # Stats + config annotation (top-right area inside margin)
    if stats:
        info_lines: list[str] = []

        mode = stats.get("mode", "?")
        info_lines.append(f"Mode: {mode}")

        overrides = stats.get("cli_overrides", [])
        if overrides:
            info_lines.append("CLI overrides:")
            for ov in overrides:
                info_lines.append(f"  {ov}")

        info_lines.append("")
        info_lines.append(f"Paths: {stats.get('paths', '?')}")
        info_lines.append(f"Draw: {stats.get('draw_mm', 0):.0f} mm")
        info_lines.append(f"Travel: {stats.get('travel_mm', 0):.0f} mm")
        info_lines.append(f"Pen lifts: {stats.get('pen_lifts', '?')}")
        info_lines.append(f"Est. time: {stats.get('estimated_time_min', 0):.1f} min")

        sx, sy = mm_to_canvas(
            transform.draw_left + transform.draw_width - 50.0,
            transform.draw_bottom + transform.draw_height - 2.0,
        )
        for k, line_txt in enumerate(info_lines):
            cv2.putText(
                canvas, line_txt,
                (sx, sy + k * line_spacing + line_spacing),
                font, font_scale * 0.85, font_color, font_thick,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"  Preview saved: {output_path}")


def save_vectors_yaml(
    paths_mm: list[np.ndarray],
    image_w_mm: float,
    image_h_mm: float,
    W: int,
    H: int,
    output_path: Path,
) -> None:
    """Save paths in ``pen_vectors.v1`` YAML for use with ``run_tracer.py``.

    Parameters
    ----------
    paths_mm : list[np.ndarray]
        Polylines in image-frame mm ``(N, 2)``.
    image_w_mm, image_h_mm : float
        Image extent in mm.
    W, H : int
        Original image pixel dimensions.
    output_path : Path
        Destination YAML file.
    """
    import yaml

    data = {
        "schema": "pen_vectors.v1",
        "render_px": [int(W), int(H)],
        "work_area_mm": [round(float(image_w_mm), 3), round(float(image_h_mm), 3)],
        "paths": [],
        "metadata": {
            "tool_name": "lineart_tracer",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    for i, p in enumerate(paths_mm):
        data["paths"].append({
            "id": f"la-{i:06d}",
            "kind": "polyline",
            "role": "outline",
            "tip_diameter_mm": 0.3,
            "z_mm": 0.0,
            "feed_mm_s": 150.0,
            "points_mm": [
                [round(float(x), 4), round(float(y), 4)] for x, y in p
            ],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)
    print(f"  Vectors saved: {output_path}")


# ===========================================================================
# Zone-bounded hatching pipeline
# ===========================================================================


def _hatch_mask_at_angle(
    mask: np.ndarray, pitch_px: float, angle_deg: float,
) -> list[np.ndarray]:
    """Generate hatching lines inside *mask* at a given angle and pitch.

    Algorithm:
    1. Pad mask into a square canvas large enough that no pixels are
       clipped after rotation.
    2. Rotate the padded canvas around its centre.
    3. Extract horizontal scanline runs from the rotated mask.
    4. Map the run coordinates back to the *original* (un-padded) image
       frame using ``cv2.invertAffineTransform``.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask, shape ``(H, W)``.
    pitch_px : float
        Perpendicular spacing between hatch lines (px).
    angle_deg : float
        Hatch angle in degrees (0 = horizontal, 45 = diagonal).

    Returns
    -------
    list[np.ndarray]
        Polylines in original image coordinates, shape ``(N, 2)`` each.
    """
    H, W = mask.shape
    diag = int(math.ceil(math.sqrt(H * H + W * W)))

    # Step 1: Pad mask into a diag x diag canvas, centred
    pad_x = (diag - W) // 2
    pad_y = (diag - H) // 2
    padded_mask = np.zeros((diag, diag), dtype=np.uint8)
    padded_mask[pad_y:pad_y + H, pad_x:pad_x + W] = mask.astype(np.uint8)

    # Step 2: Rotate around the centre of the padded canvas
    centre = (diag / 2.0, diag / 2.0)
    fwd_mat = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    rot_mask = cv2.warpAffine(
        padded_mask, fwd_mat, (diag, diag),
        flags=cv2.INTER_NEAREST, borderValue=0,
    )

    # Step 3 & 4: Extract scanlines and map back to original coords
    # invertAffineTransform gives us M_inv: rotated coords -> padded coords
    inv_mat = cv2.invertAffineTransform(fwd_mat)
    i00, i01, i02 = inv_mat[0]
    i10, i11, i12 = inv_mat[1]

    segments: list[np.ndarray] = []
    step = max(int(round(pitch_px)), 1)
    row_idx = 0
    for row in range(0, diag, step):
        line = rot_mask[row, :]
        if not line.any():
            row_idx += 1
            continue

        # Find contiguous runs of True pixels
        bordered = np.concatenate([[0], line, [0]])
        d = np.diff(bordered.astype(np.int8))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]

        row_segs: list[np.ndarray] = []
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            xs_rot = np.arange(s, e, dtype=np.float64)
            ys_rot = np.full_like(xs_rot, row, dtype=np.float64)

            # Rotated -> padded coords via inverse matrix
            xs_pad = i00 * xs_rot + i01 * ys_rot + i02
            ys_pad = i10 * xs_rot + i11 * ys_rot + i12

            # Padded -> original image coords (subtract padding offset)
            xs_orig = xs_pad - pad_x
            ys_orig = ys_pad - pad_y

            row_segs.append(np.column_stack((xs_orig, ys_orig)))

        # Alternate direction per row for zigzag connectivity
        if row_idx % 2 == 1:
            row_segs = [seg[::-1] for seg in reversed(row_segs)]

        segments.extend(row_segs)
        row_idx += 1

    return segments


def _zigzag_connect(
    segments: list[np.ndarray], max_gap_px: float = 5.0,
) -> list[np.ndarray]:
    """Connect nearby line segments into continuous zigzag paths.

    Segments from adjacent scanlines whose endpoints are within
    *max_gap_px* are concatenated to reduce pen lifts.

    Parameters
    ----------
    segments : list[np.ndarray]
        Input polylines, shape ``(N, 2)`` each.
    max_gap_px : float
        Maximum pixel gap to bridge between segment endpoints.

    Returns
    -------
    list[np.ndarray]
        Connected polylines.
    """
    if not segments:
        return []

    chains: list[list[np.ndarray]] = [[segments[0]]]
    for seg in segments[1:]:
        last_chain = chains[-1]
        last_end = last_chain[-1][-1]
        seg_start = seg[0]
        seg_end = seg[-1]

        dist_fwd = np.linalg.norm(seg_start - last_end)
        dist_rev = np.linalg.norm(seg_end - last_end)

        if dist_fwd <= max_gap_px:
            last_chain.append(seg)
        elif dist_rev <= max_gap_px:
            last_chain.append(seg[::-1])
        else:
            chains.append([seg])

    result: list[np.ndarray] = []
    for chain in chains:
        combined = np.concatenate(chain, axis=0)
        if len(combined) >= 2:
            result.append(combined)
    return result


def _auto_thresholds(gray: np.ndarray, n_zones: int) -> list[int]:
    """Compute *n_zones - 1* thresholds that split the histogram into zones.

    Uses percentiles of the non-white pixel distribution so that each
    zone contains a roughly equal number of dark pixels.  The lightest
    zone (above the last threshold) receives no hatching.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image, dtype uint8, shape ``(H, W)``.
    n_zones : int
        Number of distinct darkness zones (including the white/no-hatch zone).

    Returns
    -------
    list[int]
        Sorted thresholds (ascending), length ``n_zones - 1``.
    """
    pixels = gray[gray < 245].astype(np.float64)
    if len(pixels) == 0:
        return list(np.linspace(64, 200, n_zones - 1).astype(int))

    pcts = np.linspace(0, 100, n_zones + 1)[1:-1]
    thresholds = np.percentile(pixels, pcts).astype(int).tolist()
    return sorted(set(max(1, min(254, t)) for t in thresholds))


def _process_hatching(
    image_path: Path,
    paper_w: float,
    paper_h: float,
    margin: float,
    *,
    n_zones: int = 4,
    pen_width_mm: float = 0.3,
    angles: list[float] | None = None,
    blur_radius: int = 5,
    image_scale: float = 1.0,
    include_outlines: bool = True,
    outline_turdsize: int = 10,
    merge_tolerance_px: float = 5.0,
    simplify_tol_mm: float = 0.05,
    min_path_mm: float = 0.3,
    connect_gap_px: float = 5.0,
    t0: float = 0.0,
) -> tuple[list[np.ndarray], list[np.ndarray], float, float, float, int, int]:
    """Zone-bounded hatching with auto-level detection and cross-hatching.

    Unlike the ``hatched`` mode (which uses the ``hatched`` library and
    can bleed across zone boundaries), this mode:

    1. Auto-detects *n_zones* gray levels from the image histogram.
    2. Creates a binary mask for each zone.
    3. Generates parallel scanlines clipped to each zone's mask.
    4. Connects adjacent scanlines with zigzag patterns for continuity.
    5. Supports cross-hatching (multiple angles per zone).

    The darkest zone gets the densest fill (pitch = pen_width, effectively
    solid).  Lighter zones get progressively sparser hatching.  The lightest
    zone (white) receives no hatching at all.

    Parameters
    ----------
    n_zones : int
        Number of gray zones (including white).  4 = 3 hatched zones + white.
    pen_width_mm : float
        Physical pen trace width (mm).  Densest zone uses this as pitch.
    angles : list[float] or None
        Hatching angles in degrees.  ``[45]`` = single direction,
        ``[45, -45]`` = cross-hatch, ``[0, 60, 120]`` = triple cross.
        If None, defaults to ``[45]``.
    blur_radius : int
        Gaussian blur before zone detection; smooths transitions.
    image_scale : float
        Scale image before processing (reduce for speed).
    include_outlines : bool
        If True, also trace potrace outlines on top.
    connect_gap_px : float
        Max pixel gap for zigzag connection within zones.

    Returns
    -------
    Same 7-tuple as ``process_image``.
    """
    if angles is None:
        angles = [45.0]

    print("  [1/6] Loading and preprocessing image...")
    pil_img = Image.open(str(image_path)).convert("L")
    orig_w, orig_h = pil_img.size

    if image_scale != 1.0:
        pil_img = pil_img.resize(
            (int(orig_w * image_scale), int(orig_h * image_scale)),
            Image.LANCZOS,
        )

    gray = np.array(pil_img, dtype=np.uint8)
    H, W = gray.shape

    if blur_radius > 0:
        ksize = blur_radius * 2 + 1
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # Compute drawable area and scaling
    drawable_w = paper_w - 2.0 * margin
    drawable_h = paper_h - 2.0 * margin
    mm_per_px = min(drawable_w / W, drawable_h / H)
    image_w_mm = W * mm_per_px
    image_h_mm = H * mm_per_px
    pen_width_px = pen_width_mm / mm_per_px

    print(f"        Image: {W} x {H} px -> {image_w_mm:.1f} x {image_h_mm:.1f} mm")
    print(f"        Pen width: {pen_width_mm:.2f} mm = {pen_width_px:.1f} px")
    print(f"        Zones: {n_zones}, angles: {angles}")

    # [2/6] Auto-detect zone thresholds
    print("  [2/6] Auto-detecting gray level thresholds...")
    thresholds = _auto_thresholds(gray, n_zones)
    print(f"        Thresholds: {thresholds}")

    # Build individual zone masks.
    # zone 0 = darkest (below first threshold) -- gets densest fill.
    # zone n-2 = lightest hatched -- gets sparsest fill.
    # zone n-1 (above last threshold) = white -- no hatching.
    n_hatched_zones = len(thresholds)
    zone_masks: list[np.ndarray] = []
    for i in range(n_hatched_zones):
        upper = thresholds[i]
        if i == 0:
            mask = gray < upper
        else:
            lower = thresholds[i - 1]
            mask = (gray >= lower) & (gray < upper)
        zone_masks.append(mask)

    # Erode masks slightly to prevent hatching from bleeding at edges
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for i in range(len(zone_masks)):
        zone_masks[i] = cv2.erode(
            zone_masks[i].astype(np.uint8), erode_kernel, iterations=1,
        ).astype(bool)

    zone_pixels = [int(m.sum()) for m in zone_masks]
    print(f"        Zone pixels: {zone_pixels}")

    # [3/6] Generate hatching for each zone
    print("  [3/6] Generating zone-bounded hatching lines...")
    all_hatch_segments: list[np.ndarray] = []

    for zone_idx in range(n_hatched_zones):
        # Pitch: zone 0 (darkest) = pen_width  (solid fill).
        #        zone n-1 (lightest hatched) = pen_width * n.
        density_factor = zone_idx + 1
        pitch_px = pen_width_px * density_factor

        if pitch_px < 1.0:
            pitch_px = 1.0

        mask = zone_masks[zone_idx]
        if not mask.any():
            continue

        # Adjacent diagonal scanlines are up to pitch * sqrt(2) apart;
        # use that as the minimum connect gap.
        effective_gap = max(connect_gap_px, pitch_px * 1.5)

        # Light zones (upper half of gradient) use single-direction
        # hatching; only the darker half gets the full cross-hatch.
        is_light = zone_idx >= n_hatched_zones / 2.0
        zone_angles = [angles[0]] if (is_light and len(angles) > 1) else angles

        zone_segments: list[np.ndarray] = []
        for angle in zone_angles:
            clipped = _hatch_mask_at_angle(mask, pitch_px, angle)
            connected = _zigzag_connect(clipped, max_gap_px=effective_gap)
            zone_segments.extend(connected)

        n_segs = len(zone_segments)
        total_pts = sum(len(s) for s in zone_segments)
        cross_label = "cross" if len(zone_angles) > 1 else "single"
        print(f"        Zone {zone_idx} (thr<{thresholds[zone_idx]}): "
              f"pitch={pitch_px:.1f}px {cross_label}, "
              f"{n_segs} paths, {total_pts} pts")
        all_hatch_segments.extend(zone_segments)

    print(f"        Total: {len(all_hatch_segments)} hatch paths")

    # [4/6] Optionally add potrace outlines
    outline_paths: list[np.ndarray] = []
    if include_outlines:
        print("  [4/6] Tracing outlines (potrace)...")
        binary = (gray < thresholds[0]).astype(np.uint8)
        bmp = potrace.Bitmap(binary)
        traced = bmp.trace(turdsize=outline_turdsize, alphamax=1.0,
                           opticurve=True)
        for curve in traced.curves:
            pts: list[np.ndarray] = [np.array(curve.start_point)]
            for seg in curve.segments:
                if seg.is_corner:
                    pts.append(np.array(seg.c))
                    pts.append(np.array(seg.end_point))
                else:
                    p0 = pts[-1]
                    p1 = np.array(seg.c1)
                    p2 = np.array(seg.c2)
                    p3 = np.array(seg.end_point)
                    pts.extend(_flatten_bezier(p0, p1, p2, p3, tol=0.5))
            pts.append(np.array(curve.start_point))
            outline_paths.append(np.array(pts, dtype=np.float64))
        print(f"        {len(outline_paths)} outline contours")
    else:
        print("  [4/6] Outlines disabled, skipping.")

    all_paths_px = all_hatch_segments + outline_paths

    if not all_paths_px:
        return [], [], image_w_mm, image_h_mm, mm_per_px, H, W

    # [5/6] vpype merge + sort
    print(f"  [5/6] Merging & sorting ({len(all_paths_px)} paths)...")
    lc = vpype.LineCollection()
    for arr in all_paths_px:
        cline = arr[:, 0] + 1j * arr[:, 1]
        lc.append(cline)

    lc.merge(tolerance=merge_tolerance_px, flip=True)
    print(f"        {len(lc)} paths after merge")

    li = vpype.LineIndex(lc.lines, reverse=True)
    sorted_lines: list[np.ndarray] = []
    if len(li) > 0:
        first = li.pop_front()
        sorted_lines.append(first)
        while len(li) > 0:
            last_pt = sorted_lines[-1][-1]
            idx, rev = li.find_nearest(last_pt)
            path = li.pop(idx)
            if path is None:
                break
            if rev:
                path = path[::-1]
            sorted_lines.append(path)

    # Convert complex -> (N,2) in pixel coords, scale to mm, filter
    paths_px: list[np.ndarray] = []
    paths_mm: list[np.ndarray] = []
    for cline in sorted_lines:
        arr = np.column_stack((cline.real, cline.imag)).astype(np.float64)
        if len(arr) < 2:
            continue
        p_mm = arr * mm_per_px
        length = float(np.sum(np.hypot(*np.diff(p_mm, axis=0).T)))
        if length >= min_path_mm:
            paths_px.append(arr)
            paths_mm.append(p_mm)

    paths_mm = simplify_paths(paths_mm, simplify_tol_mm)
    paths_px = [p / mm_per_px for p in paths_mm]

    elapsed = time.monotonic() - t0
    total_draw_mm = sum(
        float(np.sum(np.hypot(*np.diff(p, axis=0).T))) for p in paths_mm
    )
    print(f"  [6/6] Done.")
    print(f"        Paths: {len(paths_mm)}")
    print(f"        Total drawing distance: {total_draw_mm:.0f} mm")
    print(f"        Processing time: {elapsed:.1f} s")

    return paths_mm, paths_px, image_w_mm, image_h_mm, mm_per_px, H, W


# ===========================================================================
# Flow-imager processing pipeline
# ===========================================================================


def _process_flow_imager(
    image_path: Path,
    paper_w: float,
    paper_h: float,
    margin: float,
    *,
    noise_coeff: float = 0.001,
    n_fields: int = 1,
    min_sep: float = 0.8,
    max_sep: float = 10.0,
    min_length: float = 0.0,
    max_length: float = 40.0,
    max_size: int = 800,
    seed: int = 42,
    flow_seed: int = 42,
    search_ef: int = 50,
    test_frequency: float = 2.0,
    field_type: str = "noise",
    edge_field_multiplier: float | None = None,
    dark_field_multiplier: float | None = None,
    merge_tolerance_px: float = 5.0,
    simplify_tol_mm: float = 0.05,
    min_path_mm: float = 0.3,
    t0: float = 0.0,
) -> tuple[list[np.ndarray], list[np.ndarray], float, float, float, int, int]:
    """Generate flow-field streamline art from an image.

    Uses vpype-flow-imager's ``draw_image`` to produce density-modulated
    streamlines whose spacing follows image brightness.

    Parameters
    ----------
    image_path : Path
        Input image (colour or grayscale).
    paper_w, paper_h, margin : float
        Paper geometry in mm.
    noise_coeff : float
        Simplex noise frequency multiplier; smaller = smoother flow.
    n_fields : int
        Number of rotated copies of the flow field (1 = smooth,
        3/4/6 = triangular/rectangular/hexagonal patterns).
    min_sep, max_sep : float
        Flowline separation range in px at max_size resolution.
    min_length, max_length : float
        Flowline length range in px.
    max_size : int
        Image is resized so its largest side is at most this many px.
    seed, flow_seed : int
        PRNG seeds for reproducibility.
    field_type : str
        ``"noise"`` (opensimplex) or ``"curl_noise"`` (curly).
    edge_field_multiplier, dark_field_multiplier : float or None
        If set, blend edge-following or dark-curling fields into the
        noise field (try 1.0 as a starting point).

    Returns
    -------
    Same 7-tuple as ``process_image``.
    """
    print("  [1/4] Loading image for flow-field generation...")
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    H_orig, W_orig = img.shape[:2]
    drawable_w = paper_w - 2.0 * margin
    drawable_h = paper_h - 2.0 * margin
    mm_per_px = min(drawable_w / W_orig, drawable_h / H_orig)
    image_w_mm = W_orig * mm_per_px
    image_h_mm = H_orig * mm_per_px

    alpha = None
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]

    print(f"        Image: {W_orig} x {H_orig} px -> "
          f"{image_w_mm:.1f} x {image_h_mm:.1f} mm")
    print(f"        flow_seed={flow_seed}, field_type={field_type}, "
          f"n_fields={n_fields}")
    print(f"        sep=[{min_sep}, {max_sep}], "
          f"length=[{min_length}, {max_length}], max_size={max_size}")

    print("  [2/4] Generating flow-field streamlines...")
    try:
        from vpype_flow_imager.vpype_flow_imager import HNSWSearcher
        searcher_class = HNSWSearcher
    except ImportError:
        from vpype_flow_imager.kdtree import KDTSearcher
        searcher_class = KDTSearcher

    numpy_paths = flow_draw_image(
        img, alpha,
        mult=noise_coeff,
        n_fields=n_fields,
        min_sep=min_sep,
        max_sep=max_sep,
        min_length=min_length,
        max_length=max_length,
        max_img_size=max_size,
        flow_seed=flow_seed,
        search_ef=search_ef,
        test_frequency=test_frequency,
        field_type=field_type,
        transparent_val=127,
        transparent_mask=False,
        edge_field_multiplier=edge_field_multiplier,
        dark_field_multiplier=dark_field_multiplier,
        searcher_class=searcher_class,
        rotate=0,
        flow_image_data=None,
    )
    print(f"        {len(numpy_paths)} streamlines generated")

    if not numpy_paths:
        return [], [], image_w_mm, image_h_mm, mm_per_px, H_orig, W_orig

    # Convert to vpype LineCollection for merge + sort
    print(f"  [3/4] Merging & sorting ({len(numpy_paths)} paths)...")
    lc = vpype.LineCollection()
    for path in numpy_paths:
        cline = path[:, 0] + 1j * path[:, 1]
        lc.append(cline)

    lc.merge(tolerance=merge_tolerance_px, flip=True)
    print(f"        {len(lc)} paths after merge")

    li = vpype.LineIndex(lc.lines, reverse=True)
    sorted_lines: list[np.ndarray] = []
    if len(li) > 0:
        first = li.pop_front()
        sorted_lines.append(first)
        while len(li) > 0:
            last_pt = sorted_lines[-1][-1]
            idx, rev = li.find_nearest(last_pt)
            path = li.pop(idx)
            if path is None:
                break
            if rev:
                path = path[::-1]
            sorted_lines.append(path)

    # Convert complex -> (N,2) float64 in pixel coords
    paths_px: list[np.ndarray] = []
    for cline in sorted_lines:
        arr = np.column_stack((cline.real, cline.imag)).astype(np.float64)
        if len(arr) >= 2:
            paths_px.append(arr)

    # Scale to mm and filter
    paths_mm: list[np.ndarray] = []
    for p in paths_px:
        p_mm = p * mm_per_px
        length = float(np.sum(np.hypot(*np.diff(p_mm, axis=0).T)))
        if length >= min_path_mm:
            paths_mm.append(p_mm)

    paths_mm = simplify_paths(paths_mm, simplify_tol_mm)
    paths_px = [p / mm_per_px for p in paths_mm]

    elapsed = time.monotonic() - t0
    total_draw_mm = sum(
        float(np.sum(np.hypot(*np.diff(p, axis=0).T))) for p in paths_mm
    )
    print(f"  [4/4] Done.")
    print(f"        Paths: {len(paths_mm)}")
    print(f"        Total drawing distance: {total_draw_mm:.0f} mm")
    print(f"        Processing time: {elapsed:.1f} s")

    return paths_mm, paths_px, image_w_mm, image_h_mm, mm_per_px, H_orig, W_orig


# ===========================================================================
# Full image processing pipeline
# ===========================================================================


def process_image(
    image_path: Path,
    paper_w: float,
    paper_h: float,
    margin: float,
    mode: str = "line_tracing",
    merge_tolerance_px: float = 5.0,
    turdsize: int = 10,
    threshold: int | None = None,
    simplify_tol_mm: float = 0.05,
    min_path_mm: float = 0.5,
    hatch_levels: tuple[int, ...] = (64, 128, 192),
    hatch_pitch_mm: float = 0.6,
    hatch_angle_deg: float = 45.0,
    hatch_blur_radius: int = 10,
    hatch_image_scale: float = 1.0,
    hatch_circular: bool = False,
    hatch_invert: bool = False,
    hatch_include_outlines: bool = True,
    hatch_outline_turdsize: int = 10,
    flow_noise_coeff: float = 0.001,
    flow_n_fields: int = 1,
    flow_min_sep: float = 0.8,
    flow_max_sep: float = 10.0,
    flow_min_length: float = 0.0,
    flow_max_length: float = 40.0,
    flow_max_size: int = 800,
    flow_seed: int = 42,
    flow_flow_seed: int = 42,
    flow_search_ef: int = 50,
    flow_test_frequency: float = 2.0,
    flow_field_type: str = "noise",
    flow_edge_field_mult: float | None = None,
    flow_dark_field_mult: float | None = None,
    hatching_n_zones: int = 4,
    hatching_pen_width_mm: float = 0.3,
    hatching_angles: list[float] | None = None,
    hatching_blur_radius: int = 5,
    hatching_image_scale: float = 1.0,
    hatching_include_outlines: bool = True,
    hatching_outline_turdsize: int = 10,
    hatching_connect_gap_px: float = 5.0,
) -> tuple[list[np.ndarray], list[np.ndarray], float, float, float, int, int]:
    """Run the full vectorisation pipeline on an image.

    Four operating modes:

    - **line_tracing**: potrace outlines + vpype merge/sort.
    - **hatched**: density-based hatch fill (``hatched`` library).
    - **hatching**: zone-bounded cross-hatching with auto-levels.
    - **flow_imager**: flow-field streamlines.

    Returns
    -------
    paths_mm : list[np.ndarray]
        Final polylines in image-frame mm, shape ``(N, 2)``.
    paths_px : list[np.ndarray]
        Corresponding polylines in pixel coordinates (for preview).
    image_w_mm, image_h_mm : float
        Image dimensions in mm after fitting to drawable area.
    mm_per_px : float
        Conversion factor.
    H, W : int
        Image pixel dimensions.
    """
    t0 = time.monotonic()

    if mode == "hatching":
        return _process_hatching(
            image_path, paper_w, paper_h, margin,
            n_zones=hatching_n_zones,
            pen_width_mm=hatching_pen_width_mm,
            angles=hatching_angles,
            blur_radius=hatching_blur_radius,
            image_scale=hatching_image_scale,
            include_outlines=hatching_include_outlines,
            outline_turdsize=hatching_outline_turdsize,
            merge_tolerance_px=merge_tolerance_px,
            simplify_tol_mm=simplify_tol_mm,
            min_path_mm=min_path_mm,
            connect_gap_px=hatching_connect_gap_px,
            t0=t0,
        )

    if mode == "hatched":
        return _process_hatched(
            image_path, paper_w, paper_h, margin,
            levels=hatch_levels,
            hatch_pitch_mm=hatch_pitch_mm,
            hatch_angle_deg=hatch_angle_deg,
            blur_radius=hatch_blur_radius,
            image_scale=hatch_image_scale,
            circular=hatch_circular,
            invert=hatch_invert,
            include_outlines=hatch_include_outlines,
            outline_turdsize=hatch_outline_turdsize,
            merge_tolerance_px=merge_tolerance_px,
            simplify_tol_mm=simplify_tol_mm,
            min_path_mm=min_path_mm,
            t0=t0,
        )

    if mode == "flow_imager":
        return _process_flow_imager(
            image_path, paper_w, paper_h, margin,
            noise_coeff=flow_noise_coeff,
            n_fields=flow_n_fields,
            min_sep=flow_min_sep,
            max_sep=flow_max_sep,
            min_length=flow_min_length,
            max_length=flow_max_length,
            max_size=flow_max_size,
            seed=flow_seed,
            flow_seed=flow_flow_seed,
            search_ef=flow_search_ef,
            test_frequency=flow_test_frequency,
            field_type=flow_field_type,
            edge_field_multiplier=flow_edge_field_mult,
            dark_field_multiplier=flow_dark_field_mult,
            merge_tolerance_px=merge_tolerance_px,
            simplify_tol_mm=simplify_tol_mm,
            min_path_mm=min_path_mm,
            t0=t0,
        )

    # -- line_tracing mode (potrace + vpype) --
    thr = threshold if threshold is not None else 128
    return _process_vpype(
        image_path, paper_w, paper_h, margin, thr,
        merge_tolerance_px, simplify_tol_mm, min_path_mm,
        turdsize, t0,
    )



def _flatten_bezier(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
    tol: float = 0.5,
) -> list[np.ndarray]:
    """Adaptively flatten a cubic Bezier into polyline points.

    Recursively subdivides until the control-point deviation from the
    chord is below *tol* pixels.  Returns a list of intermediate/end
    points (excludes *p0*).
    """
    chord = p3 - p0
    chord_len = max(np.linalg.norm(chord), 1e-10)
    d1 = abs(float(np.cross(chord, p1 - p0))) / chord_len
    d2 = abs(float(np.cross(chord, p2 - p0))) / chord_len
    if max(d1, d2) <= tol:
        return [p3]
    m01 = (p0 + p1) * 0.5
    m12 = (p1 + p2) * 0.5
    m23 = (p2 + p3) * 0.5
    m012 = (m01 + m12) * 0.5
    m123 = (m12 + m23) * 0.5
    mid = (m012 + m123) * 0.5
    return (_flatten_bezier(p0, m01, m012, mid, tol)
            + _flatten_bezier(mid, m123, m23, p3, tol))


def _process_vpype(
    image_path: Path,
    paper_w: float,
    paper_h: float,
    margin: float,
    threshold: int,
    merge_tolerance_px: float,
    simplify_tol_mm: float,
    min_path_mm: float,
    turdsize: int,
    t0: float,
) -> tuple[list[np.ndarray], list[np.ndarray], float, float, float, int, int]:
    """Potrace + vpype pipeline: trace -> flatten -> merge -> sort.

    Uses **potrace** (via pypotrace) to produce smooth Bezier-curve
    outlines of each black region, then vpype for endpoint merging and
    nearest-neighbour path sorting.

    Potrace advantages over vectrace:
    - Smooth curves (no pixel staircase on diagonals).
    - Connected closed contours (no fragmented pixel-edge paths).
    - Tunable noise filtering via *turdsize*.

    Parameters
    ----------
    threshold : int
        Grayscale threshold (0-255).  Pixels darker than this become
        black (traced).  Default 128.
    merge_tolerance_px : float
        Max pixel distance for vpype endpoint merging.
    turdsize : int
        Potrace speckle filter: discard contours with area below this
        many pixels^2.  Default 10.
    """
    print("  [1/4] Tracing outlines (potrace)...")
    image = Image.open(str(image_path)).convert("L")
    W, H = image.size

    binary = (np.array(image) < threshold).astype(np.uint8)
    bmp = potrace.Bitmap(binary)
    traced = bmp.trace(
        turdsize=turdsize,
        alphamax=1.0,
        opticurve=True,
    )

    # Flatten Bezier curves to polylines (adaptive, 0.5 px tolerance).
    # pypotrace outputs (x, y) in image convention (y=0 at top).
    raw_paths: list[np.ndarray] = []
    for curve in traced.curves:
        pts: list[np.ndarray] = [np.array(curve.start_point)]
        for seg in curve.segments:
            if seg.is_corner:
                pts.append(np.array(seg.c))
                pts.append(np.array(seg.end_point))
            else:
                p0 = pts[-1]
                p1 = np.array(seg.c1)
                p2 = np.array(seg.c2)
                p3 = np.array(seg.end_point)
                pts.extend(_flatten_bezier(p0, p1, p2, p3, tol=0.5))
        # Close contour
        pts.append(np.array(curve.start_point))
        raw_paths.append(np.array(pts, dtype=np.float64))

    n_raw = len(raw_paths)
    n_pts = sum(len(p) for p in raw_paths)
    print(f"        {n_raw} contours, {n_pts} total points")

    # Load into vpype for merge + sort
    lc = vpype.LineCollection()
    for arr in raw_paths:
        cline = arr[:, 0] + 1j * arr[:, 1]
        lc.append(cline)

    print(f"  [2/4] Merging endpoints (tol={merge_tolerance_px:.0f} px)...")
    lc.merge(tolerance=merge_tolerance_px, flip=True)
    print(f"        {len(lc)} paths after merge")

    print("  [3/4] Sorting paths (nearest-neighbour)...")
    li = vpype.LineIndex(lc.lines, reverse=True)
    sorted_lines: list[np.ndarray] = []
    if len(li) > 0:
        first = li.pop_front()
        sorted_lines.append(first)
        while len(li) > 0:
            last_pt = sorted_lines[-1][-1]
            idx, rev = li.find_nearest(last_pt)
            path = li.pop(idx)
            if path is None:
                break
            if rev:
                path = path[::-1]
            sorted_lines.append(path)

    paths_px: list[np.ndarray] = []
    for cline in sorted_lines:
        arr = np.column_stack((cline.real, cline.imag)).astype(np.float64)
        paths_px.append(arr)

    drawable_w = paper_w - 2.0 * margin
    drawable_h = paper_h - 2.0 * margin
    mm_per_px = min(drawable_w / W, drawable_h / H)
    image_w_mm = W * mm_per_px
    image_h_mm = H * mm_per_px

    paths_mm = [p * mm_per_px for p in paths_px]

    if min_path_mm > 0:
        paths_mm = [p for p in paths_mm
                     if float(np.sum(np.hypot(*np.diff(p, axis=0).T)))
                     >= min_path_mm]

    if simplify_tol_mm > 0:
        paths_mm = simplify_paths(paths_mm, simplify_tol_mm)

    paths_px = [p / mm_per_px for p in paths_mm]

    total_draw_mm = sum(
        float(np.sum(np.hypot(*np.diff(p, axis=0).T))) for p in paths_mm
    )
    elapsed = time.monotonic() - t0
    print(f"  [4/4] Done.")
    print(f"        Paths: {len(paths_mm)}")
    print(f"        Total drawing distance: {total_draw_mm:.0f} mm")
    print(f"        Pen lifts: {len(paths_mm)}")
    print(f"        Processing time: {elapsed:.1f} s")

    return paths_mm, paths_px, image_w_mm, image_h_mm, mm_per_px, H, W


def _process_hatched(
    image_path: Path,
    paper_w: float,
    paper_h: float,
    margin: float,
    levels: tuple[int, ...],
    hatch_pitch_mm: float,
    hatch_angle_deg: float,
    blur_radius: int,
    image_scale: float,
    circular: bool,
    invert: bool,
    include_outlines: bool,
    outline_turdsize: int,
    merge_tolerance_px: float,
    simplify_tol_mm: float,
    min_path_mm: float,
    t0: float,
) -> tuple[list[np.ndarray], list[np.ndarray], float, float, float, int, int]:
    """Hatched-fill pipeline: image -> density hatching -> vpype sort.

    Uses the ``hatched`` library to convert a grayscale image into
    hatch lines whose density reflects the local darkness.  Optionally
    overlays potrace outlines for crisper edges.

    Parameters
    ----------
    levels : tuple[int, ...]
        Grayscale thresholds (0-255) separating density zones.
        3 thresholds = 4 zones (black / dark / light / white).
    hatch_pitch_mm : float
        Densest hatching pitch in mm (converted to pixels internally).
    hatch_angle_deg : float
        Angle for diagonal hatches (degrees).
    blur_radius : int
        Blur kernel applied to image before thresholding.
    image_scale : float
        Scale factor applied to image before processing.
    circular : bool
        Use concentric circles instead of diagonal lines.
    invert : bool
        Invert image before processing.
    include_outlines : bool
        When True, potrace outlines are overlaid on the hatched fill.
    outline_turdsize : int
        Potrace turdsize for outline generation.
    merge_tolerance_px : float
        Max pixel distance for vpype endpoint merging.
    """
    drawable_w = paper_w - 2.0 * margin
    drawable_h = paper_h - 2.0 * margin

    # [1/5] Load and prepare image
    print("  [1/5] Loading image for hatching...")
    img = _load_image(
        file_path=str(image_path),
        blur_radius=blur_radius,
        image_scale=image_scale,
        interpolation=cv2.INTER_LINEAR,
        h_mirror=False,
        invert=invert,
    )
    H, W = img.shape
    mm_per_px = min(drawable_w / W, drawable_h / H)
    image_w_mm = W * mm_per_px
    image_h_mm = H * mm_per_px

    hatch_pitch_px = hatch_pitch_mm / mm_per_px
    print(f"        Image: {W} x {H} px  ({image_w_mm:.1f} x {image_h_mm:.1f} mm)")
    print(f"        Levels: {list(levels)}  ({len(levels) + 1} zones)")
    print(f"        Pitch: {hatch_pitch_mm:.2f} mm = {hatch_pitch_px:.1f} px")

    # [2/5] Generate hatched fill
    print("  [2/5] Generating hatched fill...")
    mls, *_contours = _build_hatch(
        img,
        hatch_pitch=hatch_pitch_px,
        levels=levels,
        circular=circular,
        invert=invert,
        hatch_angle=hatch_angle_deg,
    )

    # Extract polylines from Shapely MultiLineString
    hatch_paths_px: list[np.ndarray] = []
    for geom in mls.geoms:
        coords = np.array(geom.coords, dtype=np.float64)
        if len(coords) >= 2:
            hatch_paths_px.append(coords)

    print(f"        {len(hatch_paths_px)} hatch lines generated")

    # [3/5] Optionally add potrace outlines
    outline_paths_px: list[np.ndarray] = []
    if include_outlines:
        print("  [3/5] Tracing outlines (potrace)...")
        pil_img = Image.open(str(image_path)).convert("L")
        orig_w, orig_h = pil_img.size
        if image_scale != 1.0:
            pil_img = pil_img.resize(
                (int(orig_w * image_scale), int(orig_h * image_scale)),
                Image.LANCZOS,
            )
        binary = (np.array(pil_img) < 128).astype(np.uint8)
        bmp = potrace.Bitmap(binary)
        traced = bmp.trace(
            turdsize=outline_turdsize,
            alphamax=1.0,
            opticurve=True,
        )
        for curve in traced.curves:
            pts: list[np.ndarray] = [np.array(curve.start_point)]
            for seg in curve.segments:
                if seg.is_corner:
                    pts.append(np.array(seg.c))
                    pts.append(np.array(seg.end_point))
                else:
                    p0 = pts[-1]
                    p1 = np.array(seg.c1)
                    p2 = np.array(seg.c2)
                    p3 = np.array(seg.end_point)
                    pts.extend(_flatten_bezier(p0, p1, p2, p3, tol=0.5))
            pts.append(np.array(curve.start_point))
            outline_paths_px.append(np.array(pts, dtype=np.float64))
        print(f"        {len(outline_paths_px)} outline contours")
    else:
        print("  [3/5] Outlines disabled, skipping.")

    # Combine hatch + outlines
    all_paths_px = hatch_paths_px + outline_paths_px

    if not all_paths_px:
        print("        No paths generated.")
        return [], [], image_w_mm, image_h_mm, mm_per_px, H, W

    # [4/5] vpype merge + sort
    print(f"  [4/5] Merging & sorting ({len(all_paths_px)} paths)...")
    lc = vpype.LineCollection()
    for arr in all_paths_px:
        cline = arr[:, 0] + 1j * arr[:, 1]
        lc.append(cline)

    lc.merge(tolerance=merge_tolerance_px, flip=True)
    print(f"        {len(lc)} paths after merge")

    li = vpype.LineIndex(lc.lines, reverse=True)
    sorted_lines: list[np.ndarray] = []
    if len(li) > 0:
        first = li.pop_front()
        sorted_lines.append(first)
        while len(li) > 0:
            last_pt = sorted_lines[-1][-1]
            idx, rev = li.find_nearest(last_pt)
            path = li.pop(idx)
            if path is None:
                break
            if rev:
                path = path[::-1]
            sorted_lines.append(path)

    paths_px_out: list[np.ndarray] = []
    for cline in sorted_lines:
        arr = np.column_stack((cline.real, cline.imag)).astype(np.float64)
        paths_px_out.append(arr)

    # Scale to mm, filter, simplify
    paths_mm = [p * mm_per_px for p in paths_px_out]

    if min_path_mm > 0:
        paths_mm = [
            p for p in paths_mm
            if float(np.sum(np.hypot(*np.diff(p, axis=0).T))) >= min_path_mm
        ]

    if simplify_tol_mm > 0:
        paths_mm = simplify_paths(paths_mm, simplify_tol_mm)

    paths_px_final = [p / mm_per_px for p in paths_mm]

    # [5/5] Stats
    total_draw_mm = sum(
        float(np.sum(np.hypot(*np.diff(p, axis=0).T))) for p in paths_mm
    )
    elapsed = time.monotonic() - t0
    print(f"  [5/5] Done.")
    print(f"        Paths: {len(paths_mm)}")
    print(f"        Total drawing distance: {total_draw_mm:.0f} mm")
    print(f"        Pen lifts: {len(paths_mm)}")
    print(f"        Processing time: {elapsed:.1f} s")

    return paths_mm, paths_px_final, image_w_mm, image_h_mm, mm_per_px, H, W


def execute_on_robot(
    sock: socket.socket,
    machine_paths: list[list[tuple[float, float]]],
    transform: PaperTransform,
    z_contact: float,
    z_retract: float,
    draw_speed_mm_s: float,
    travel_speed_mm_s: float,
    z_plunge_speed_mm_s: float,
    z_retract_speed_mm_s: float,
    accel_mm_s2: float,
    square_corner_velocity_mm_s: float,
    corner_circle_radius: float,
    dry_run: bool = False,
    skip_corners: bool = False,
) -> dict:
    """Execute the full drawing sequence on the robot.

    Parameters
    ----------
    sock : socket.socket
        Connected Klipper UDS socket (ignored in dry-run).
    machine_paths : list[list[tuple[float, float]]]
        Polylines in machine-frame mm.
    transform : PaperTransform
        For alignment corners and border rectangle.
    z_contact, z_retract : float
        Pen Z positions (mm).
    draw_speed_mm_s, travel_speed_mm_s : float
        XY speeds (mm/s).
    z_plunge_speed_mm_s, z_retract_speed_mm_s : float
        Z speeds (mm/s).
    accel_mm_s2 : float
        XY acceleration limit (mm/s^2).
    square_corner_velocity_mm_s : float
        Klipper junction speed limit (mm/s).
    corner_circle_radius : float
        Alignment circle radius (mm).
    dry_run : bool
        If True, compute stats without sending G-code.
    skip_corners : bool
        If True, skip alignment circles and border.

    Returns
    -------
    dict
        Execution statistics.
    """
    z_travel = z_contact - z_retract
    draw_fr = draw_speed_mm_s * 60.0
    travel_fr = travel_speed_mm_s * 60.0
    z_down_fr = z_plunge_speed_mm_s * 60.0
    z_up_fr = z_retract_speed_mm_s * 60.0

    margin_corners = transform.get_margin_corners()
    paper_corners = transform.get_paper_corners()

    # ---- Compute stats ----
    total_draw_mm = 0.0
    total_travel_mm = 0.0
    total_pen_lifts = 0

    for mpts in machine_paths:
        for i in range(1, len(mpts)):
            dx = mpts[i][0] - mpts[i - 1][0]
            dy = mpts[i][1] - mpts[i - 1][1]
            total_draw_mm += math.sqrt(dx * dx + dy * dy)

    prev_end: Optional[tuple[float, float]] = None
    for mpts in machine_paths:
        if prev_end is not None:
            dx = mpts[0][0] - prev_end[0]
            dy = mpts[0][1] - prev_end[1]
            total_travel_mm += math.sqrt(dx * dx + dy * dy)
        prev_end = mpts[-1]
        total_pen_lifts += 1

    # ---- Time estimates (trapezoidal motion model) ----
    _sqrt = math.sqrt

    def _trap_time(dist: float, v_max: float, accel: float) -> float:
        if dist <= 0 or v_max <= 0:
            return 0.0
        d_ramp = v_max * v_max / accel
        if dist >= d_ramp:
            return dist / v_max + v_max / accel
        return 2.0 * _sqrt(dist / accel)

    scv = square_corner_velocity_mm_s
    _SQRT2_M1 = _sqrt(2.0) - 1.0
    _V_MAX = draw_speed_mm_s
    _V_MAX2 = _V_MAX * _V_MAX
    _A = accel_mm_s2

    draw_time_s = 0.0
    total_segments = 0

    for mpts in machine_paths:
        n = len(mpts)
        if n < 2:
            continue

        dxs = [mpts[i][0] - mpts[i - 1][0] for i in range(1, n)]
        dys = [mpts[i][1] - mpts[i - 1][1] for i in range(1, n)]
        lens = [_sqrt(dxs[j] ** 2 + dys[j] ** 2) for j in range(n - 1)]

        jvs: list[float] = []
        for j in range(len(dxs) - 1):
            l1, l2 = lens[j], lens[j + 1]
            if l1 < 1e-6 or l2 < 1e-6:
                jvs.append(0.0)
                continue
            cos_t = (dxs[j] * dxs[j + 1] + dys[j] * dys[j + 1]) / (l1 * l2)
            if cos_t > 0.9999:
                jvs.append(_V_MAX)
            elif cos_t < -0.9999:
                jvs.append(0.0)
            else:
                sin_hd = _sqrt(0.5 * (1.0 - cos_t))
                factor = _SQRT2_M1 * sin_hd / (1.0 - sin_hd)
                vj = scv * _sqrt(factor) if factor > 0.0 else 0.0
                jvs.append(min(vj, _V_MAX))

        for j in range(len(dxs)):
            seg_len = lens[j]
            if seg_len < 1e-6:
                continue
            v_entry = jvs[j - 1] if j > 0 else 0.0
            v_exit = jvs[j] if j < len(jvs) else 0.0
            vj = min(v_entry, v_exit)

            if vj >= _V_MAX:
                draw_time_s += seg_len / _V_MAX
            else:
                d_ramp = (_V_MAX2 - vj * vj) / _A
                if seg_len >= d_ramp:
                    draw_time_s += (
                        2.0 * (_V_MAX - vj) / _A + (seg_len - d_ramp) / _V_MAX
                    )
                else:
                    v_peak = _sqrt(vj * vj + _A * seg_len)
                    draw_time_s += 2.0 * (v_peak - vj) / _A
            total_segments += 1

    travel_time_s = 0.0
    prev_end_t: Optional[tuple[float, float]] = None
    for mpts in machine_paths:
        if prev_end_t is not None:
            dx = mpts[0][0] - prev_end_t[0]
            dy = mpts[0][1] - prev_end_t[1]
            travel_time_s += _trap_time(
                _sqrt(dx * dx + dy * dy), travel_speed_mm_s, accel_mm_s2,
            )
        prev_end_t = mpts[-1]

    plunge_time_s = total_pen_lifts * (
        _trap_time(z_retract, z_plunge_speed_mm_s, accel_mm_s2)
        + _trap_time(z_retract, z_retract_speed_mm_s, accel_mm_s2)
    )

    batch_size = 64
    total_sends = sum(
        1 + max(1, math.ceil(max(len(mp) - 1, 1) / batch_size))
        for mp in machine_paths
    )
    comm_overhead_s = total_sends * 0.005

    total_time_s = draw_time_s + travel_time_s + plunge_time_s + comm_overhead_s

    print()
    print("=" * 60)
    print("  EXECUTION PLAN")
    print("=" * 60)
    print()
    print(f"  Paths:             {len(machine_paths)}")
    print(f"  Drawing distance:  {total_draw_mm:.0f} mm "
          f"({draw_time_s / 60.0:.1f} min, {total_segments} segments)")
    print(f"  Travel distance:   {total_travel_mm:.0f} mm "
          f"({travel_time_s / 60.0:.1f} min)")
    print(f"  Pen lifts:         {total_pen_lifts}")
    print(f"  Plunge time:       {plunge_time_s / 60.0:.1f} min")
    print(f"  Comms overhead:    {comm_overhead_s / 60.0:.1f} min "
          f"({total_sends} sends)")
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

    # ---- Apply motion limits ----
    _raw_gcode(sock, f"SET_VELOCITY_LIMIT ACCEL={accel_mm_s2:.0f}")
    _raw_gcode(
        sock,
        f"SET_VELOCITY_LIMIT SQUARE_CORNER_VELOCITY={scv:.1f}",
    )

    # ---- Phase 0: pen placement check ----
    print("=" * 60)
    print("  PHASE 0: PEN PLACEMENT CHECK")
    print("=" * 60)
    print()

    first_cx, first_cy = paper_corners[0]
    print(f"  Moving to paper edge corner: ({first_cx:.1f}, {first_cy:.1f}) mm")
    pen_up(sock, z_travel, z_up_fr)
    travel_to(sock, first_cx, first_cy, travel_fr)
    _raw_gcode(sock, "M400")

    print(f"  Lowering Z to contact ({z_contact:.3f} mm)...")
    pen_down(sock, z_contact, z_down_fr)
    _raw_gcode(sock, "M400")

    print()
    print("  >> Insert the pen and verify it touches the paper.")
    print("  >> Adjust --z-contact if needed (higher = pen closer to bed).")
    print()
    input("  Press ENTER when pen depth is correct to continue...")
    pen_up(sock, z_travel, z_up_fr)

    # ---- Phase 1: alignment circles ----
    if not skip_corners:
        print()
        print("=" * 60)
        print("  PHASE 1: CORNER ALIGNMENT CIRCLES")
        print("=" * 60)
        print()

        for i, (cx, cy) in enumerate(margin_corners):
            label = ["bottom-left", "bottom-right", "top-right", "top-left"][i]
            print(f"  Corner {i + 1}/4 ({label}): ({cx:.1f}, {cy:.1f}) mm")
            draw_circle_at(
                sock, cx, cy, corner_circle_radius,
                z_contact, z_travel, z_down_fr, z_up_fr, draw_fr, travel_fr,
            )

        _raw_gcode(sock, "M400")
        print()
        print("  All 4 corner circles drawn.")
        print("  >> Align paper so its corners match the circles.")
        print()
        input("  Press ENTER when paper is aligned to continue...")
    else:
        print()
        print("  [Skipping corner alignment]")

    # ---- Phase 2: border rectangle ----
    print()
    print("=" * 60)
    print("  PHASE 2: BORDER RECTANGLE")
    print("=" * 60)
    print()

    draw_rectangle(
        sock, margin_corners,
        z_contact, z_travel, z_down_fr, z_up_fr, draw_fr, travel_fr,
    )
    _raw_gcode(sock, "M400")
    print("  Border rectangle drawn.")
    print()

    # ---- Phase 3: draw all paths ----
    print("=" * 60)
    print("  PHASE 3: DRAWING LINE ART")
    print("=" * 60)
    print()

    t_start = time.monotonic()
    last_report_time = t_start
    prev_pct = -1
    DRAW_BATCH = 64
    REPORT_INTERVAL_S = 60.0  # update at least every minute

    z_travel_c = max(Z_MIN_SAFE, min(z_travel, Z_MAX_SAFE))
    z_contact_c = max(Z_MIN_SAFE, min(z_contact, Z_MAX_SAFE))

    for path_idx, mpts in enumerate(machine_paths):
        pct = int(100.0 * (path_idx + 1) / len(machine_paths))
        now = time.monotonic()
        time_since_report = now - last_report_time
        is_milestone = pct >= prev_pct + 5
        is_timer = time_since_report >= REPORT_INTERVAL_S
        is_last = path_idx == len(machine_paths) - 1

        if is_milestone or is_timer or is_last:
            elapsed = now - t_start
            frac = (path_idx + 1) / len(machine_paths)
            eta_s = (elapsed / frac - elapsed) if frac > 0.01 else 0.0
            draw_done_mm = sum(
                math.sqrt(
                    (mp[i][0] - mp[i-1][0])**2
                    + (mp[i][1] - mp[i-1][1])**2
                )
                for mp in machine_paths[:path_idx + 1]
                for i in range(1, len(mp))
            ) if path_idx < 20 else total_draw_mm * frac
            speed_actual = draw_done_mm / elapsed if elapsed > 1.0 else 0.0
            sys.stdout.write(
                f"\r  Path {path_idx + 1}/{len(machine_paths)} "
                f"({pct}%)  elapsed: {elapsed / 60.0:.1f} min  "
                f"ETA: {eta_s / 60.0:.1f} min  "
                f"speed: {speed_actual:.0f} mm/s    "
            )
            sys.stdout.flush()
            prev_pct = pct
            last_report_time = now

        x0 = max(0.0, min(mpts[0][0], WORKSPACE_X_MM))
        y0 = max(0.0, min(mpts[0][1], WORKSPACE_Y_MM))
        transition = (
            f"G1 Z{z_travel_c:.3f} F{z_up_fr:.0f}\n"
            f"G1 X{x0:.2f} Y{y0:.2f} F{travel_fr:.0f}\n"
            f"G1 Z{z_contact_c:.3f} F{z_down_fr:.0f}"
        )
        _raw_gcode(sock, transition)

        draw_pts = mpts[1:]
        for bs in range(0, len(draw_pts), DRAW_BATCH):
            batch = draw_pts[bs : bs + DRAW_BATCH]
            lines = []
            for x, y in batch:
                xc = max(0.0, min(x, WORKSPACE_X_MM))
                yc = max(0.0, min(y, WORKSPACE_Y_MM))
                lines.append(f"G1 X{xc:.2f} Y{yc:.2f} F{draw_fr:.0f}")
            _raw_gcode(sock, "\n".join(lines))

    # Final pen up
    pen_up(sock, z_travel, z_up_fr)
    _raw_gcode(sock, "M400")

    elapsed = time.monotonic() - t_start
    print()
    print()
    print(f"  Drawing complete in {elapsed / 60.0:.1f} min")

    # ---- Return to safe position ----
    print("  Returning to safe position...")
    travel_to(sock, WORKSPACE_X_MM / 2.0, WORKSPACE_Y_MM / 2.0, travel_fr)
    _raw_gcode(sock, "M400")
    print("  Done.")

    stats["actual_time_min"] = elapsed / 60.0
    return stats


# ===========================================================================
# CLI entry point
# ===========================================================================


def main() -> None:
    _default_mode = _LINEART.get("mode", "line_tracing")

    parser = argparse.ArgumentParser(
        description="Line-art tracer and hatched-fill generator for the "
                    "robot pen plotter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Image selection ----
    parser.add_argument(
        "--image", "-i", type=str, default=None,
        help="Image filename in robot_control/images/",
    )
    parser.add_argument(
        "--image-path", type=str, default=None,
        help="Direct path to any image file (overrides --image)",
    )
    parser.add_argument(
        "--images-dir", type=str, default=str(DEFAULT_IMAGES_DIR),
        help=f"Directory containing input images (default: {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available images and exit",
    )

    # ---- Operating mode ----
    parser.add_argument(
        "--mode", type=str, default=_default_mode,
        choices=["line_tracing", "hatched", "hatching", "flow_imager"],
        help="Operating mode: 'line_tracing' (B&W outlines), "
             "'hatched' (colour hatching, good for photos), "
             "'hatching' (zone-bounded cross-hatching with auto-levels), or "
             "'flow_imager' (flow-field streamlines)",
    )

    # ---- Potrace / vpype parameters (line_tracing mode) ----
    parser.add_argument(
        "--merge-tolerance", type=float,
        default=_LINEART_LT.get("merge_tolerance_px", 5.0),
        help="Max pixel distance for vpype endpoint merging "
             "(default: 5.0)",
    )
    parser.add_argument(
        "--turdsize", type=int,
        default=_LINEART_LT.get("turdsize", 10),
        help="Potrace speckle filter: discard contours with area below "
             "this many pixels^2 (default: 10)",
    )

    # ---- Hatched mode parameters ----
    hatch_group = parser.add_argument_group(
        "Hatched mode parameters (only used when --mode hatched)")
    hatch_group.add_argument(
        "--levels", type=int, nargs="+",
        default=_LINEART_HATCH.get("levels", [64, 128, 192]),
        help="Grayscale thresholds (0-255) dividing the image into "
             "density zones.  3 values = 4 zones (black/dark/light/white). "
             "More values = smoother gradient (default: 64 128 192)",
    )
    hatch_group.add_argument(
        "--hatch-pitch", type=float,
        default=_LINEART_HATCH.get("hatch_pitch_mm", 0.6),
        help="Densest hatching pitch in mm; smaller = denser fill, "
             "min ~= pen width (default: 0.6)",
    )
    hatch_group.add_argument(
        "--hatch-angle", type=float,
        default=_LINEART_HATCH.get("hatch_angle_deg", 45.0),
        help="Diagonal hatching angle in degrees (default: 45)",
    )
    hatch_group.add_argument(
        "--blur-radius", type=int,
        default=_LINEART_HATCH.get("blur_radius", 10),
        help="Blur radius applied to image before thresholding; "
             "smooths zone transitions (default: 10)",
    )
    hatch_group.add_argument(
        "--image-scale", type=float,
        default=_LINEART_HATCH.get("image_scale", 1.0),
        help="Scale factor applied to image before hatching; "
             "reduce for faster processing (default: 1.0)",
    )
    hatch_group.add_argument(
        "--circular", action="store_true",
        default=_LINEART_HATCH.get("circular", False),
        help="Use concentric circles instead of diagonal hatches",
    )
    hatch_group.add_argument(
        "--invert", action="store_true",
        default=_LINEART_HATCH.get("invert", False),
        help="Invert image before processing (dark becomes light)",
    )
    hatch_group.add_argument(
        "--no-outlines", action="store_true",
        help="Disable potrace outline overlay in hatched mode",
    )

    # ---- Hatching mode parameters (zone-bounded) ----
    htng_group = parser.add_argument_group(
        "Hatching mode parameters (only used when --mode hatching)")
    htng_group.add_argument(
        "--n-zones", type=int,
        default=_LINEART_HTNG.get("n_zones", 4),
        help="Number of gray zones including white (4 = 3 hatched + white). "
             "Auto-detected from image histogram (default: 4)",
    )
    htng_group.add_argument(
        "--hatching-angles", type=float, nargs="+",
        default=_LINEART_HTNG.get("angles", [45.0]),
        help="Hatching angles in degrees.  Single value = one direction, "
             "multiple = cross-hatch (e.g., 45 -45 for X-pattern) "
             "(default: 45)",
    )
    htng_group.add_argument(
        "--hatching-blur", type=int,
        default=_LINEART_HTNG.get("blur_radius", 5),
        help="Gaussian blur radius before zone detection (default: 5)",
    )
    htng_group.add_argument(
        "--hatching-scale", type=float,
        default=_LINEART_HTNG.get("image_scale", 1.0),
        help="Image scale factor; reduce for speed (default: 1.0)",
    )
    htng_group.add_argument(
        "--hatching-no-outlines", action="store_true",
        help="Disable potrace outline overlay in hatching mode",
    )
    htng_group.add_argument(
        "--connect-gap", type=float,
        default=_LINEART_HTNG.get("connect_gap_px", 5.0),
        help="Max pixel gap for zigzag connection within zones (default: 5)",
    )

    # ---- Flow-imager mode parameters ----
    flow_group = parser.add_argument_group(
        "Flow-imager mode parameters (only used when --mode flow_imager)")
    flow_group.add_argument(
        "--noise-coeff", type=float,
        default=_LINEART_FLOW.get("noise_coeff", 0.001),
        help="Simplex noise frequency; smaller = smoother flow (default: 0.001)",
    )
    flow_group.add_argument(
        "--n-fields", type=int,
        default=_LINEART_FLOW.get("n_fields", 1),
        help="Rotated copies of the flow field; 1=smooth, "
             "3/4/6=triangular/rectangular/hexagonal (default: 1)",
    )
    flow_group.add_argument(
        "--flow-min-sep", type=float,
        default=_LINEART_FLOW.get("min_sep", 0.8),
        help="Minimum flowline separation in px (default: 0.8)",
    )
    flow_group.add_argument(
        "--flow-max-sep", type=float,
        default=_LINEART_FLOW.get("max_sep", 10.0),
        help="Maximum flowline separation in px (default: 10)",
    )
    flow_group.add_argument(
        "--flow-min-length", type=float,
        default=_LINEART_FLOW.get("min_length", 0.0),
        help="Minimum flowline length in px (default: 0)",
    )
    flow_group.add_argument(
        "--flow-max-length", type=float,
        default=_LINEART_FLOW.get("max_length", 40.0),
        help="Maximum flowline length in px (default: 40)",
    )
    flow_group.add_argument(
        "--flow-max-size", type=int,
        default=_LINEART_FLOW.get("max_size", 800),
        help="Resize image so largest side <= this (default: 800)",
    )
    flow_group.add_argument(
        "--flow-seed", type=int,
        default=_LINEART_FLOW.get("flow_seed", 42),
        help="PRNG seed for flow field (default: 42)",
    )
    flow_group.add_argument(
        "--field-type", type=str,
        default=_LINEART_FLOW.get("field_type", "noise"),
        choices=["noise", "curl_noise"],
        help="Flow field type: 'noise' or 'curl_noise' (default: noise)",
    )
    flow_group.add_argument(
        "--edge-field-mult", type=float,
        default=_LINEART_FLOW.get("edge_field_multiplier"),
        help="Blend edge-following field (try 1.0); omit to disable",
    )
    flow_group.add_argument(
        "--dark-field-mult", type=float,
        default=_LINEART_FLOW.get("dark_field_multiplier"),
        help="Blend dark-curling field (try 1.0); omit to disable",
    )

    # ---- Image processing parameters ----
    proc_group = parser.add_argument_group("Image processing")
    proc_group.add_argument(
        "--threshold", type=int,
        default=_LINEART_LT.get("threshold"),
        help="Manual binary threshold 0-255 (default: 128)",
    )
    proc_group.add_argument(
        "--simplify-tol", type=float,
        default=_LINEART_LT.get("simplify_tol_mm", 0.05),
        help="Douglas-Peucker tolerance, mm (default: 0.05)",
    )
    proc_group.add_argument(
        "--min-path-mm", type=float,
        default=_LINEART_LT.get("min_path_mm", 0.5),
        help="Drop paths shorter than this, mm (default: 0.5)",
    )

    # ---- Z-axis ----
    z_group = parser.add_argument_group("Z-axis parameters")
    z_group.add_argument(
        "--z-contact", type=float,
        default=_PEN.get("z_contact_mm", 70.0),
        help="Z position where pen touches paper, mm",
    )
    z_group.add_argument(
        "--z-retract", type=float,
        default=_PEN.get("z_retract_mm", 1.5),
        help="How far to retract above contact, mm",
    )

    # ---- Speeds ----
    speed_group = parser.add_argument_group("Speed parameters")
    speed_group.add_argument(
        "--draw-speed", type=float,
        default=_PEN.get("draw_speed_mm_s", 150.0),
        help="Drawing speed, mm/s",
    )
    speed_group.add_argument(
        "--travel-speed", type=float,
        default=_PEN.get("travel_speed_mm_s", 400.0),
        help="Travel speed, mm/s",
    )
    speed_group.add_argument(
        "--z-plunge-speed", type=float,
        default=_PEN.get("z_plunge_speed_mm_s", 20.0),
        help="Z plunge (pen down) speed, mm/s",
    )
    speed_group.add_argument(
        "--z-retract-speed", type=float,
        default=_PEN.get("z_retract_speed_mm_s", 40.0),
        help="Z retract (pen up) speed, mm/s",
    )
    speed_group.add_argument(
        "--accel", type=float,
        default=_PEN.get("accel_mm_s2", 2000.0),
        help="XY acceleration limit, mm/s^2",
    )
    speed_group.add_argument(
        "--square-corner-velocity", type=float,
        default=_PEN.get("square_corner_velocity_mm_s", 3.0),
        help="Max speed through sharp corners, mm/s",
    )

    # ---- Paper ----
    _paper_size = _PAPER.get("size_mm", [297.0, 210.0])
    _paper_origin = _PAPER.get("origin_mm", [25.0, 25.0])
    paper_group = parser.add_argument_group("Paper and alignment")
    paper_group.add_argument(
        "--paper-size", type=float, nargs=2, default=_paper_size,
        metavar=("W", "H"),
        help="Paper size [X, Y] in mm",
    )
    paper_group.add_argument(
        "--paper-origin", type=float, nargs=2, default=_paper_origin,
        metavar=("X", "Y"),
        help="Paper bottom-left corner offset from endstops, mm",
    )
    paper_group.add_argument(
        "--margin", type=float,
        default=_PAPER.get("margin_mm", 10.0),
        help="Margin from paper edge on all sides, mm",
    )
    paper_group.add_argument(
        "--corner-radius", type=float,
        default=_PEN.get("corner_circle_radius_mm", 3.0),
        help="Alignment circle radius, mm",
    )

    # ---- Operational flags ----
    parser.add_argument(
        "--mirror-x", action="store_true",
        help="Mirror the image horizontally on paper",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute stats without sending G-code",
    )
    parser.add_argument(
        "--no-config-write", action="store_true",
        help="Don't overwrite printer.cfg",
    )
    parser.add_argument(
        "--skip-corners", action="store_true",
        help="Skip corner alignment circles",
    )
    parser.add_argument(
        "--save-preview", action="store_true",
        help="Save a vectorised preview image",
    )
    parser.add_argument(
        "--save-vectors", action="store_true",
        help="Save pen_vectors.yaml (compatible with run_tracer.py)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for previews/vectors (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    # ---- List mode ----
    images_dir = Path(args.images_dir)
    if args.list:
        if not images_dir.exists():
            print(f"Images directory not found: {images_dir}")
            return
        exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
        found = sorted(
            f.name for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in exts
        )
        if not found:
            print(f"No images found in {images_dir}")
        else:
            print(f"Available images in {images_dir}:")
            for name in found:
                sz = (images_dir / name).stat().st_size
                print(f"  {name}  ({sz / 1024:.0f} KB)")
        return

    # ---- Resolve image path ----
    if args.image_path:
        image_path = Path(args.image_path)
    elif args.image:
        image_path = images_dir / args.image
    else:
        parser.error("Specify --image NAME or --image-path PATH (use --list to see available)")
        return

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    # ---- Banner ----
    _mode_banners = {
        "line_tracing": "PEN PLOTTER - LINE ART TRACER",
        "hatched": "PEN PLOTTER - COLOUR HATCHING (hatched library)",
        "hatching": "PEN PLOTTER - ZONE-BOUNDED CROSS-HATCHING",
        "flow_imager": "PEN PLOTTER - FLOW FIELD IMAGER",
    }
    print("=" * 60)
    print(f"  {_mode_banners.get(args.mode, args.mode)}")
    print("=" * 60)
    print()
    print(f"  Image:       {image_path}")
    print(f"  Mode:        {args.mode}")

    if args.mode == "line_tracing":
        print("  Method:      potrace + vpype (smooth Bezier outlines + merge/sort)")
    elif args.mode == "hatched":
        hatch_style = "circular" if args.circular else f"diagonal {args.hatch_angle:.0f} deg"
        print(f"  Hatch style: {hatch_style}")
        print(f"  Levels:      {list(args.levels)} ({len(args.levels) + 1} zones)")
        print(f"  Pitch:       {args.hatch_pitch:.2f} mm")
        print(f"  Outlines:    {'yes' if not args.no_outlines else 'no'}")
    elif args.mode == "hatching":
        angle_str = ", ".join(f"{a:.0f}" for a in args.hatching_angles)
        cross = "cross-hatch" if len(args.hatching_angles) > 1 else "single"
        print(f"  Zones:       {args.n_zones} (auto-detected levels)")
        print(f"  Angles:      [{angle_str}] deg ({cross})")
        print(f"  Pen width:   {_LINE_WIDTH_MM:.2f} mm")
        print(f"  Blur:        {args.hatching_blur}")
        print(f"  Outlines:    {'yes' if not args.hatching_no_outlines else 'no'}")
        print(f"  Connect gap: {args.connect_gap:.1f} px")
    elif args.mode == "flow_imager":
        print(f"  Field type:  {args.field_type}")
        print(f"  Flow seed:   {args.flow_seed}")
        print(f"  Separation:  [{args.flow_min_sep}, {args.flow_max_sep}] px")
        print(f"  Length:      [{args.flow_min_length}, {args.flow_max_length}] px")
        print(f"  N fields:    {args.n_fields}")
        if args.edge_field_mult is not None:
            print(f"  Edge blend:  {args.edge_field_mult}")
        if args.dark_field_mult is not None:
            print(f"  Dark blend:  {args.dark_field_mult}")

    print(f"  Machine cfg: {Path(_CFG._source_path).resolve()}"
          if hasattr(_CFG, '_source_path') else
          f"  Machine cfg: {_JOBS_YAML.parent / 'machine.yaml'}")
    print(f"  Jobs cfg:    {_JOBS_YAML.resolve()}")
    if _LINEART_YAML.exists():
        print(f"  Lineart cfg: {_LINEART_YAML.resolve()}")
    print(f"  Line width:  {_LINE_WIDTH_MM:.2f} mm")
    print()

    # ---- Process image ----
    paper_w, paper_h = args.paper_size

    _include_outlines = (
        _LINEART_HATCH.get("include_outlines", True)
        and not args.no_outlines
    )

    paths_mm, paths_px, image_w_mm, image_h_mm, mm_per_px, H, W = process_image(
        image_path=image_path,
        paper_w=paper_w,
        paper_h=paper_h,
        margin=args.margin,
        mode=args.mode,
        merge_tolerance_px=args.merge_tolerance,
        turdsize=args.turdsize,
        threshold=args.threshold,
        simplify_tol_mm=args.simplify_tol,
        min_path_mm=args.min_path_mm,
        hatch_levels=tuple(args.levels),
        hatch_pitch_mm=args.hatch_pitch,
        hatch_angle_deg=args.hatch_angle,
        hatch_blur_radius=args.blur_radius,
        hatch_image_scale=args.image_scale,
        hatch_circular=args.circular,
        hatch_invert=args.invert,
        hatch_include_outlines=_include_outlines,
        hatch_outline_turdsize=_LINEART_HATCH.get("outline_turdsize", 10),
        flow_noise_coeff=args.noise_coeff,
        flow_n_fields=args.n_fields,
        flow_min_sep=args.flow_min_sep,
        flow_max_sep=args.flow_max_sep,
        flow_min_length=args.flow_min_length,
        flow_max_length=args.flow_max_length,
        flow_max_size=args.flow_max_size,
        flow_seed=args.flow_seed,
        flow_flow_seed=args.flow_seed,
        flow_search_ef=_LINEART_FLOW.get("search_ef", 50),
        flow_test_frequency=_LINEART_FLOW.get("test_frequency", 2.0),
        flow_field_type=args.field_type,
        flow_edge_field_mult=args.edge_field_mult,
        flow_dark_field_mult=args.dark_field_mult,
        hatching_n_zones=args.n_zones,
        hatching_pen_width_mm=_LINE_WIDTH_MM,
        hatching_angles=args.hatching_angles,
        hatching_blur_radius=args.hatching_blur,
        hatching_image_scale=args.hatching_scale,
        hatching_include_outlines=not args.hatching_no_outlines,
        hatching_outline_turdsize=_LINEART_HTNG.get("outline_turdsize", 10),
        hatching_connect_gap_px=args.connect_gap,
    )

    if not paths_mm:
        print("  No paths extracted -- nothing to draw.")
        return

    # ---- Optional raw-pixel preview (explicit flag only) ----
    stem = image_path.stem
    out_dir = Path(args.output_dir) / stem

    if args.save_preview:
        save_preview(paths_px, H, W, out_dir / "preview.png")

    if args.save_vectors:
        save_vectors_yaml(paths_mm, image_w_mm, image_h_mm, W, H,
                          out_dir / "pen_vectors.yaml")

    # ---- Handle image rotation for paper orientation ----
    img_is_portrait = image_h_mm > image_w_mm
    paper_is_landscape = paper_w > paper_h

    if (img_is_portrait and paper_is_landscape) or (
        not img_is_portrait and not paper_is_landscape
    ):
        print(f"  Rotating paths 90\u00b0 CW to match paper orientation")
        for i in range(len(paths_mm)):
            pts = paths_mm[i]
            # 90 deg CCW in image space: new_x = image_h - old_y, new_y = old_x
            rotated = np.column_stack([image_h_mm - pts[:, 1], pts[:, 0]])
            paths_mm[i] = rotated
        image_w_mm, image_h_mm = image_h_mm, image_w_mm
        print(f"  Rotated work area: {image_w_mm:.1f} x {image_h_mm:.1f} mm")

    # ---- Build coordinate transform ----
    ox, oy = args.paper_origin
    xform = PaperTransform(
        workspace_x=WORKSPACE_X_MM,
        workspace_y=WORKSPACE_Y_MM,
        paper_w=paper_w,
        paper_h=paper_h,
        margin=args.margin,
        image_w=image_w_mm,
        image_h=image_h_mm,
        paper_origin=(ox, oy),
        mirror_x=args.mirror_x,
    )

    print()
    print("  Coordinate transform:")
    xform.report()
    print()

    # ---- Convert paths to machine coordinates ----
    machine_paths: list[list[tuple[float, float]]] = []
    for p in paths_mm:
        if len(p) < 2:
            continue
        mpts = [xform.image_to_machine(float(pt[0]), float(pt[1])) for pt in p]
        machine_paths.append(mpts)

    # ---- Validate Z ----
    z_travel = args.z_contact - args.z_retract
    if z_travel < Z_MIN_SAFE:
        print(f"  WARNING: Z travel ({z_travel:.1f}) below safe min ({Z_MIN_SAFE:.1f})")
    if args.z_contact > Z_MAX_SAFE:
        print(f"  WARNING: Z contact ({args.z_contact:.1f}) above safe max ({Z_MAX_SAFE:.1f})")
        args.z_contact = Z_MAX_SAFE

    # ---- Validate margins ----
    margin_corners = xform.get_margin_corners()
    for cx, cy in margin_corners:
        if cx < 0 or cx > WORKSPACE_X_MM or cy < 0 or cy > WORKSPACE_Y_MM:
            print(f"  ERROR: Margin corner ({cx:.1f}, {cy:.1f}) outside workspace!")
            sys.exit(1)

    # ---- Config summary ----
    print("  Config values (from jobs.yaml / lineart.yaml, "
          "CLI overrides shown with *):")
    _show: list[tuple[str, str, bool]] = [
        ("paper_size", f"{paper_w:.0f} x {paper_h:.0f} mm",
         args.paper_size != _PAPER.get("size_mm", [297.0, 210.0])),
        ("paper_origin", f"({ox:.1f}, {oy:.1f}) mm",
         args.paper_origin != _PAPER.get("origin_mm", [25.0, 25.0])),
        ("margin", f"{args.margin:.1f} mm",
         args.margin != _PAPER.get("margin_mm", 10.0)),
        ("draw_speed", f"{args.draw_speed:.0f} mm/s",
         args.draw_speed != _PEN.get("draw_speed_mm_s", 150.0)),
        ("z_contact", f"{args.z_contact:.1f} mm",
         args.z_contact != _PEN.get("z_contact_mm", 70.0)),
    ]
    if args.mode == "hatched":
        _show.extend([
            ("hatch_pitch", f"{args.hatch_pitch:.2f} mm",
             args.hatch_pitch != _LINEART_HATCH.get("hatch_pitch_mm", 0.6)),
            ("hatch_angle", f"{args.hatch_angle:.0f} deg",
             args.hatch_angle != _LINEART_HATCH.get("hatch_angle_deg", 45.0)),
            ("levels", str(list(args.levels)),
             list(args.levels) != _LINEART_HATCH.get("levels", [64, 128, 192])),
            ("blur_radius", str(args.blur_radius),
             args.blur_radius != _LINEART_HATCH.get("blur_radius", 10)),
        ])
    elif args.mode == "hatching":
        _show.extend([
            ("n_zones", str(args.n_zones),
             args.n_zones != _LINEART_HTNG.get("n_zones", 4)),
            ("hatching_angles", str(args.hatching_angles),
             args.hatching_angles != _LINEART_HTNG.get("angles", [45.0])),
            ("hatching_blur", str(args.hatching_blur),
             args.hatching_blur != _LINEART_HTNG.get("blur_radius", 5)),
            ("hatching_scale", str(args.hatching_scale),
             args.hatching_scale != _LINEART_HTNG.get("image_scale", 1.0)),
            ("connect_gap", f"{args.connect_gap:.1f}",
             args.connect_gap != _LINEART_HTNG.get("connect_gap_px", 5.0)),
        ])
    elif args.mode == "flow_imager":
        _show.extend([
            ("field_type", args.field_type,
             args.field_type != _LINEART_FLOW.get("field_type", "noise")),
            ("flow_seed", str(args.flow_seed),
             args.flow_seed != _LINEART_FLOW.get("flow_seed", 42)),
            ("n_fields", str(args.n_fields),
             args.n_fields != _LINEART_FLOW.get("n_fields", 1)),
            ("flow_min_sep", str(args.flow_min_sep),
             args.flow_min_sep != _LINEART_FLOW.get("min_sep", 0.8)),
            ("flow_max_sep", str(args.flow_max_sep),
             args.flow_max_sep != _LINEART_FLOW.get("max_sep", 10.0)),
        ])
    elif args.mode == "line_tracing":
        thr_val = args.threshold if args.threshold is not None else 128
        thr_default = _LINEART_LT.get("threshold", None)
        thr_default_val = thr_default if thr_default is not None else 128
        _show.extend([
            ("threshold", str(thr_val),
             thr_val != thr_default_val),
            ("turdsize", str(args.turdsize),
             args.turdsize != _LINEART_LT.get("turdsize", 10)),
        ])

    # Collect CLI overrides for the preview annotation
    _cli_overrides: list[str] = []
    for name, val, overridden in _show:
        flag = " *" if overridden else ""
        print(f"    {name}: {val}{flag}")
        if overridden:
            _cli_overrides.append(f"{name}={val}")
    print()

    # ---- Robot setup or dry-run ----
    sock: Optional[socket.socket] = None

    if not args.dry_run:
        print("  Robot workspace:")
        print(f"    X: 0 .. {WORKSPACE_X_MM:.0f} mm")
        print(f"    Y: 0 .. {WORKSPACE_Y_MM:.0f} mm")
        print(f"    Z: {Z_MIN_SAFE:.0f} .. {Z_MAX_SAFE:.0f} mm")
        print()

        if not args.no_config_write:
            if PRINTER_CFG_PATH.exists():
                backup = PRINTER_CFG_PATH.with_suffix(".cfg.bak")
                PRINTER_CFG_PATH.rename(backup)
                print(f"  Backed up printer.cfg -> {backup}")

            config_text = generate_printer_cfg(_CFG)
            PRINTER_CFG_PATH.write_text(config_text)
            print(f"  Wrote printer.cfg to {PRINTER_CFG_PATH}")
            print()

            print("  Restarting Klipper...")
            restart_klipper()

        print("  Waiting for Klipper...")
        sock = _wait_for_ready(timeout=30.0)
        print("  [OK] Klipper is ready")
        print()

        # Extend idle timeout so motors stay energised during
        # interactive phases (pen check, corner alignment, etc.)
        _raw_gcode(sock, "SET_IDLE_TIMEOUT TIMEOUT=3600", timeout=5.0)

        print("  Homing all axes...")
        if not _raw_gcode(sock, "G28", timeout=60.0):
            print("  ERROR: Homing failed!")
            sock.close()
            sys.exit(1)
        print(f"    Homed OK  X=0  Y=0  Z={WORKSPACE_Z_MM:.0f}")

        z_travel_val = max(Z_MIN_SAFE, min(z_travel, Z_MAX_SAFE))
        pen_up(sock, z_travel_val, args.z_retract_speed * 60.0)
        _raw_gcode(sock, "M400")
        print(f"    Z retracted to {z_travel_val:.1f} mm")

        # Ensure absolute positioning mode
        _raw_gcode(sock, "G90")

        # Load bed mesh profile for dynamic Z compensation
        if _raw_gcode(sock, "BED_MESH_PROFILE LOAD=default"):
            margin_corners = xform.get_margin_corners()
            _verify_mesh_loaded(sock, margin_corners, args.z_contact)
        else:
            print("    WARNING: No bed mesh profile -- Z will be flat")
        print()

    # ---- Execute ----
    try:
        stats = execute_on_robot(
            sock=sock,  # type: ignore[arg-type]
            machine_paths=machine_paths,
            transform=xform,
            z_contact=args.z_contact,
            z_retract=args.z_retract,
            draw_speed_mm_s=args.draw_speed,
            travel_speed_mm_s=args.travel_speed,
            z_plunge_speed_mm_s=args.z_plunge_speed,
            z_retract_speed_mm_s=args.z_retract_speed,
            accel_mm_s2=args.accel,
            square_corner_velocity_mm_s=args.square_corner_velocity,
            corner_circle_radius=args.corner_radius,
            dry_run=args.dry_run,
            skip_corners=args.skip_corners,
        )

        # Inject mode/config info for the preview annotation
        stats["mode"] = args.mode
        stats["cli_overrides"] = _cli_overrides

        # In dry-run, always produce a machine-coordinate preview
        if args.dry_run:
            preview_path = DEFAULT_PREVIEW_DIR / f"{stem}_preview.png"
            render_machine_preview(
                machine_paths, xform, preview_path, dpi=20.0, stats=stats,
            )

        if not args.dry_run:
            assert sock is not None
            # Restore default idle timeout before disabling motors
            _raw_gcode(sock, "SET_IDLE_TIMEOUT TIMEOUT=30", timeout=5.0)
            _raw_gcode(sock, "M18")

            print()
            print("=" * 60)
            print("  TRACE COMPLETE")
            print("=" * 60)
            print()
            print(f"  Paths drawn:   {stats['paths']}")
            print(f"  Draw dist:     {stats['draw_mm']:.0f} mm")
            print(f"  Travel dist:   {stats['travel_mm']:.0f} mm")
            print(f"  Pen lifts:     {stats['pen_lifts']}")
            if "actual_time_min" in stats:
                print(f"  Actual time:   {stats['actual_time_min']:.1f} min")
            print()

    except KeyboardInterrupt:
        print("\n\n  Interrupted -- retracting pen...")
        if sock:
            z_travel_val = max(Z_MIN_SAFE, min(z_travel, Z_MAX_SAFE))
            try:
                pen_up(sock, z_travel_val, args.z_retract_speed * 60.0)
                _raw_gcode(sock, "M400")
                _raw_gcode(sock, "SET_IDLE_TIMEOUT TIMEOUT=30", timeout=5.0)
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
        if sock:
            try:
                _raw_gcode(sock, "SET_IDLE_TIMEOUT TIMEOUT=30", timeout=2.0)
            except Exception:
                pass
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
