"""Lineart tracer mode -- interactive image selection and execution.

Wraps ``run_lineart_tracer.py``'s ``process_image`` and ``execute_on_robot``
with an interactive parameter editor, image browser, and Rich progress.

Each step clears the terminal and renders its panels at the top,
keeping the display persistent and uncluttered.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from robot_control.scripts.cli.widgets import info_panel

if TYPE_CHECKING:
    from robot_control.scripts.cli.app import RobotApp

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_JOBS_YAML = _CONFIGS_DIR / "jobs.yaml"
_LINEART_YAML = _CONFIGS_DIR / "lineart.yaml"
_IMAGES_DIR = Path(__file__).resolve().parents[2] / "images"
_RAW_IMAGES_DIR = _PROJECT_ROOT / "data" / "raw_images" / "hard"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "lineart_traces"
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def _load_yaml_safe(path: Path) -> dict:
    if not path.exists():
        return {}
    from src.utils.fs import load_yaml
    data = load_yaml(path)
    return data if isinstance(data, dict) else {}


def _find_images() -> list[tuple[str, Path]]:
    """Scan image directories and return (display_label, path) pairs."""
    images: list[tuple[str, Path]] = []
    seen: set[str] = set()

    for dir_path, dir_label in [
        (_RAW_IMAGES_DIR, "data/raw_images/hard"),
        (_IMAGES_DIR, "robot_control/images"),
    ]:
        if not dir_path.exists():
            continue
        for f in sorted(dir_path.iterdir()):
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
                key = f.name.lower()
                if key not in seen:
                    seen.add(key)
                    sz = f.stat().st_size
                    label = f"{f.name}  ({sz / 1024:.0f} KB)  [{dir_label}]"
                    images.append((label, f))

    return images


def _ask_float(app: RobotApp, prompt: str, default: float) -> float:
    result = questionary.text(f"{prompt} [{default:g}]:", default=str(default)).ask()
    if result is None:
        raise KeyboardInterrupt
    val = result.strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _ask_int(app: RobotApp, prompt: str, default: int) -> int:
    result = questionary.text(f"{prompt} [{default}]:", default=str(default)).ask()
    if result is None:
        raise KeyboardInterrupt
    val = result.strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def run(app: RobotApp) -> None:
    """Entry point for lineart tracer mode."""
    console = app.console
    log = app.session_log

    sys.path.insert(0, str(_PROJECT_ROOT))

    from robot_control.scripts.run_lineart_tracer import (
        PaperTransform,
        execute_on_robot,
        process_image,
        save_preview,
        save_vectors_yaml,
    )

    lineart = _load_yaml_safe(_LINEART_YAML)
    jobs = _load_yaml_safe(_JOBS_YAML)
    paper_cfg = jobs.get("paper", {})
    pen_cfg = jobs.get("pen_tracer", {})
    lt_cfg = lineart.get("line_tracing", {})
    hatch_cfg = lineart.get("hatched", {})
    htng_cfg = lineart.get("hatching", {})
    flow_cfg = lineart.get("flow_imager", {})

    # === Step 1: Image selection ===
    console.clear()
    console.print(
        Panel(
            Text.from_markup("[bold]Step 1[/]  Select Image"),
            border_style="cyan",
        )
    )

    images = _find_images()
    if images:
        choices = [label for label, _ in images] + ["Enter custom path"]
        selected = questionary.select("Select image:", choices=choices).ask()
        if selected is None:
            return

        if selected == "Enter custom path":
            custom = questionary.path("Image path:").ask()
            if custom is None:
                return
            image_path = Path(custom)
        else:
            image_path = next(p for label, p in images if label == selected)
    else:
        console.print(
            "[yellow]No images found in data/raw_images/hard/ or robot_control/images/[/]"
        )
        custom = questionary.path("Image path:").ask()
        if custom is None:
            return
        image_path = Path(custom)

    if not image_path.exists():
        console.print(f"[red]Image not found: {image_path}[/]")
        return

    log.log_action("tracer", "image_select", str(image_path))

    # === Step 2: Mode selection ===
    console.clear()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]Step 2[/]  Select Mode\n\n"
                f"  Image: [bold]{image_path.name}[/]"
            ),
            border_style="cyan",
        )
    )

    default_mode = lineart.get("mode", "line_tracing")
    mode_choices = {
        "line_tracing": "Line Tracing  (potrace outlines -- schematics, text, drawings)",
        "hatched": "Hatched  (density hatching -- photos, portraits, grayscale art)",
        "hatching": "Hatching  (zone-bounded cross-hatching -- maps, diagrams)",
        "flow_imager": "Flow Imager  (flow-field streamlines -- artistic line art)",
    }
    mode_list = list(mode_choices.values())
    default_idx = (
        list(mode_choices.keys()).index(default_mode)
        if default_mode in mode_choices
        else 0
    )
    mode_selected = questionary.select(
        "Select vectorization mode:",
        choices=mode_list,
        default=mode_list[default_idx],
    ).ask()
    if mode_selected is None:
        return
    mode = [k for k, v in mode_choices.items() if v == mode_selected][0]
    log.log_action("tracer", "mode_select", mode)

    # === Step 3: Parameters ===
    paper_size = paper_cfg.get("size_mm", [297.0, 210.0])
    paper_origin = paper_cfg.get("origin_mm", [25.0, 25.0])

    params: dict[str, Any] = {
        "z_contact": pen_cfg.get("z_contact_mm", 70.0),
        "z_retract": pen_cfg.get("z_retract_mm", 1.5),
        "draw_speed": pen_cfg.get("draw_speed_mm_s", 150.0),
        "travel_speed": pen_cfg.get("travel_speed_mm_s", 400.0),
        "z_plunge_speed": pen_cfg.get("z_plunge_speed_mm_s", 20.0),
        "z_retract_speed": pen_cfg.get("z_retract_speed_mm_s", 40.0),
        "accel": pen_cfg.get("accel_mm_s2", 2000.0),
        "square_corner_velocity": pen_cfg.get("square_corner_velocity_mm_s", 3.0),
        "paper_w": float(paper_size[0]),
        "paper_h": float(paper_size[1]),
        "paper_origin_x": float(paper_origin[0]),
        "paper_origin_y": float(paper_origin[1]),
        "margin": paper_cfg.get("margin_mm", 10.0),
        "corner_radius": pen_cfg.get("corner_circle_radius_mm", 3.0),
        "merge_tolerance": lt_cfg.get("merge_tolerance_px", 5.0),
        "turdsize": lt_cfg.get("turdsize", 10),
        "threshold": lt_cfg.get("threshold"),
        "simplify_tol": lt_cfg.get("simplify_tol_mm", 0.05),
        "min_path_mm": lt_cfg.get("min_path_mm", 0.5),
    }

    if mode == "hatched":
        params.update({
            "levels": hatch_cfg.get("levels", [64, 128, 192]),
            "hatch_pitch": hatch_cfg.get("hatch_pitch_mm", 0.6),
            "hatch_angle": hatch_cfg.get("hatch_angle_deg", 45.0),
            "blur_radius": hatch_cfg.get("blur_radius", 10),
            "image_scale": hatch_cfg.get("image_scale", 1.0),
            "circular": hatch_cfg.get("circular", False),
            "invert": hatch_cfg.get("invert", False),
            "include_outlines": hatch_cfg.get("include_outlines", True),
        })
    elif mode == "hatching":
        params.update({
            "n_zones": htng_cfg.get("n_zones", 4),
            "hatching_angles": htng_cfg.get("angles", [45.0, -45.0]),
            "hatching_blur": htng_cfg.get("blur_radius", 7),
            "hatching_scale": htng_cfg.get("image_scale", 1.0),
            "hatching_outlines": htng_cfg.get("include_outlines", True),
            "connect_gap": htng_cfg.get("connect_gap_px", 5.0),
        })
    elif mode == "flow_imager":
        params.update({
            "noise_coeff": flow_cfg.get("noise_coeff", 0.001),
            "n_fields": flow_cfg.get("n_fields", 1),
            "field_type": flow_cfg.get("field_type", "noise"),
            "flow_min_sep": flow_cfg.get("min_sep", 0.8),
            "flow_max_sep": flow_cfg.get("max_sep", 10.0),
            "flow_min_length": flow_cfg.get("min_length", 0.0),
            "flow_max_length": flow_cfg.get("max_length", 40.0),
            "flow_max_size": flow_cfg.get("max_size", 800),
            "flow_seed": flow_cfg.get("flow_seed", 42),
        })

    param_table = Table(title=f"Parameters ({mode})")
    param_table.add_column("Parameter", style="bold")
    param_table.add_column("Value")
    for k, v in params.items():
        param_table.add_row(k.replace("_", " ").title(), str(v))

    console.clear()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]Step 3[/]  Parameters\n\n"
                f"  Image: [bold]{image_path.name}[/]   Mode: [bold]{mode}[/]"
            ),
            border_style="cyan",
        )
    )
    console.print(param_table)

    edit = questionary.confirm("Edit parameters?", default=False).ask()
    if edit is None:
        return
    if edit:
        console.print("[dim]Press Enter to keep default, or type new value.[/]")
        for k, v in list(params.items()):
            if isinstance(v, bool):
                params[k] = questionary.confirm(f"{k}?", default=v).ask()
            elif isinstance(v, float):
                params[k] = _ask_float(app, k.replace("_", " ").title(), v)
            elif isinstance(v, int):
                params[k] = _ask_int(app, k.replace("_", " ").title(), v)
            elif isinstance(v, list):
                raw = questionary.text(f"{k} {v}:", default=str(v)).ask()
                if raw and raw.strip():
                    try:
                        params[k] = eval(raw.strip())  # noqa: S307
                    except Exception:
                        pass

    # === Step 4: Execution options ===
    console.clear()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]Step 4[/]  Options\n\n"
                f"  Image: [bold]{image_path.name}[/]   Mode: [bold]{mode}[/]"
            ),
            border_style="cyan",
        )
    )

    exec_options = questionary.checkbox(
        "Select options:",
        choices=[
            questionary.Choice("Dry run (no G-code)", value="dry_run"),
            questionary.Choice("Save preview image", value="save_preview"),
            questionary.Choice("Save pen_vectors.yaml", value="save_vectors"),
            questionary.Choice("Skip corner alignment", value="skip_corners"),
            questionary.Choice("Mirror X", value="mirror_x"),
        ],
    ).ask()
    if exec_options is None:
        return

    dry_run = "dry_run" in exec_options
    do_save_preview = "save_preview" in exec_options
    do_save_vectors = "save_vectors" in exec_options
    skip_corners = "skip_corners" in exec_options
    mirror_x = "mirror_x" in exec_options

    # === Step 5: Process image ===
    console.clear()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]Step 5[/]  Processing Image\n\n"
                f"  Image: [bold]{image_path.name}[/]   Mode: [bold]{mode}[/]"
            ),
            border_style="cyan",
        )
    )
    log.log_action("tracer", "process_start", f"mode={mode} image={image_path.name}")

    import numpy as np

    line_width_mm = float(lineart.get("line_width_mm", 0.3))

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing image...", total=None)

        try:
            paths_mm, paths_px, image_w_mm, image_h_mm, mm_per_px, H, W = process_image(
                image_path=image_path,
                paper_w=params["paper_w"],
                paper_h=params["paper_h"],
                margin=params["margin"],
                mode=mode,
                merge_tolerance_px=params["merge_tolerance"],
                turdsize=params["turdsize"],
                threshold=params.get("threshold"),
                simplify_tol_mm=params["simplify_tol"],
                min_path_mm=params["min_path_mm"],
                hatch_levels=tuple(params.get("levels", [64, 128, 192])),
                hatch_pitch_mm=params.get("hatch_pitch", 0.6),
                hatch_angle_deg=params.get("hatch_angle", 45.0),
                hatch_blur_radius=params.get("blur_radius", 10),
                hatch_image_scale=params.get("image_scale", 1.0),
                hatch_circular=params.get("circular", False),
                hatch_invert=params.get("invert", False),
                hatch_include_outlines=params.get("include_outlines", True),
                hatch_outline_turdsize=hatch_cfg.get("outline_turdsize", 10),
                flow_noise_coeff=params.get("noise_coeff", 0.001),
                flow_n_fields=params.get("n_fields", 1),
                flow_min_sep=params.get("flow_min_sep", 0.8),
                flow_max_sep=params.get("flow_max_sep", 10.0),
                flow_min_length=params.get("flow_min_length", 0.0),
                flow_max_length=params.get("flow_max_length", 40.0),
                flow_max_size=params.get("flow_max_size", 800),
                flow_seed=params.get("flow_seed", 42),
                flow_flow_seed=params.get("flow_seed", 42),
                flow_search_ef=flow_cfg.get("search_ef", 50),
                flow_test_frequency=flow_cfg.get("test_frequency", 2.0),
                flow_field_type=params.get("field_type", "noise"),
                flow_edge_field_mult=flow_cfg.get("edge_field_multiplier"),
                flow_dark_field_mult=flow_cfg.get("dark_field_multiplier"),
                hatching_n_zones=params.get("n_zones", 4),
                hatching_pen_width_mm=line_width_mm,
                hatching_angles=params.get("hatching_angles", [45.0]),
                hatching_blur_radius=params.get("hatching_blur", 5),
                hatching_image_scale=params.get("hatching_scale", 1.0),
                hatching_include_outlines=params.get("hatching_outlines", True),
                hatching_outline_turdsize=htng_cfg.get("outline_turdsize", 10),
                hatching_connect_gap_px=params.get("connect_gap", 5.0),
            )
        except Exception as exc:
            progress.stop()
            console.print(f"[red]Processing failed: {exc}[/]")
            log.log_error(exc)
            return

        progress.update(task, description="Processing complete", completed=1, total=1)

    if not paths_mm:
        console.print("[yellow]No paths extracted -- nothing to draw.[/]")
        return

    total_draw_mm = sum(
        float(np.sum(np.hypot(*np.diff(p, axis=0).T))) for p in paths_mm
    )

    console.clear()
    console.print(
        info_panel(
            {
                "Paths": str(len(paths_mm)),
                "Total draw distance": f"{total_draw_mm:.0f} mm",
                "Image dimensions": f"{image_w_mm:.1f} x {image_h_mm:.1f} mm",
                "Pixels": f"{W} x {H}",
                "mm/px": f"{mm_per_px:.4f}",
            },
            title="Vectorization Results",
        )
    )

    stem = image_path.stem
    out_dir = _DEFAULT_OUTPUT_DIR / stem

    if do_save_preview:
        save_preview(paths_px, H, W, out_dir / "preview.png")
        console.print(f"  Preview saved: {out_dir / 'preview.png'}")

    if do_save_vectors:
        save_vectors_yaml(paths_mm, image_w_mm, image_h_mm, W, H, out_dir / "pen_vectors.yaml")
        console.print(f"  Vectors saved: {out_dir / 'pen_vectors.yaml'}")

    img_is_portrait = image_h_mm > image_w_mm
    paper_is_landscape = params["paper_w"] > params["paper_h"]
    if (img_is_portrait and paper_is_landscape) or (
        not img_is_portrait and not paper_is_landscape
    ):
        console.print("  Rotating paths 90 deg CW to match paper orientation")
        for i in range(len(paths_mm)):
            pts = paths_mm[i]
            rotated = np.column_stack([image_h_mm - pts[:, 1], pts[:, 0]])
            paths_mm[i] = rotated
        image_w_mm, image_h_mm = image_h_mm, image_w_mm

    cfg = app.config
    xform = PaperTransform(
        workspace_x=cfg.work_area.x,
        workspace_y=cfg.work_area.y,
        paper_w=params["paper_w"],
        paper_h=params["paper_h"],
        margin=params["margin"],
        image_w=image_w_mm,
        image_h=image_h_mm,
        paper_origin=(params["paper_origin_x"], params["paper_origin_y"]),
        mirror_x=mirror_x,
    )

    console.print(
        info_panel(
            {
                "Paper": f"{params['paper_w']:.0f} x {params['paper_h']:.0f} mm",
                "Origin": f"({params['paper_origin_x']:.0f}, {params['paper_origin_y']:.0f})",
                "Margin": f"{params['margin']:.0f} mm",
                "Scale": f"{xform.scale:.4f}",
            },
            title="Coordinate Transform",
        )
    )

    machine_paths: list[list[tuple[float, float]]] = []
    for p in paths_mm:
        if len(p) < 2:
            continue
        mpts = [xform.image_to_machine(float(pt[0]), float(pt[1])) for pt in p]
        machine_paths.append(mpts)

    if dry_run:
        console.print("\n[bold cyan]DRY RUN[/] -- no G-code sent to robot.")
        log.log_action("tracer", "dry_run_complete", f"{len(machine_paths)} paths")
        questionary.press_any_key_to_continue("Press any key to continue...").ask()
        return

    # === Execute on robot ===
    console.clear()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]Executing on Robot[/]\n\n"
                f"  {len(machine_paths)} paths   {total_draw_mm:.0f} mm total"
            ),
            border_style="green",
        )
    )

    if not app.connection.is_connected():
        console.print("[yellow]Connecting to Klipper...[/]")
        if not app.no_config_write:
            app.connection.regenerate_printer_cfg(force=True)
        app.connection.connect()

    from robot_control.hardware.pump_control import wait_for_ready as _hw_wait
    from robot_control.scripts.run_lineart_tracer import _raw_gcode

    socket_path = app.config.connection.socket_path
    console.print("  Connecting to Klipper for drawing...")
    sock = _hw_wait(socket_path, timeout=60.0)

    try:
        stats = execute_on_robot(
            sock=sock,
            machine_paths=machine_paths,
            transform=xform,
            z_contact=params["z_contact"],
            z_retract=params["z_retract"],
            draw_speed_mm_s=params["draw_speed"],
            travel_speed_mm_s=params["travel_speed"],
            z_plunge_speed_mm_s=params["z_plunge_speed"],
            z_retract_speed_mm_s=params["z_retract_speed"],
            accel_mm_s2=params["accel"],
            square_corner_velocity_mm_s=params["square_corner_velocity"],
            corner_circle_radius=params["corner_radius"],
            dry_run=False,
            skip_corners=skip_corners,
        )

        if stats:
            console.clear()
            console.print(
                info_panel(
                    {k: str(v) for k, v in stats.items()},
                    title="Execution Stats",
                    border_style="green",
                )
            )
        log.log_action(
            "tracer", "execute_complete", f"{len(machine_paths)} paths drawn"
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Drawing interrupted -- pen up and disable.[/]")
        try:
            _raw_gcode(sock, "G0 Z40 F1200\nM400\nM18")
        except Exception:
            pass
    except Exception as exc:
        console.print(f"[red]Execution error: {exc}[/]")
        log.log_error(exc)
        try:
            _raw_gcode(sock, "M18")
        except Exception:
            pass
    finally:
        try:
            sock.close()
        except Exception:
            pass

    console.print("[green]Lineart tracer complete.[/]")
    questionary.press_any_key_to_continue("Press any key to continue...").ask()
