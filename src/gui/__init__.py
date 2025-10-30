"""Decoupled PyQt GUI for training visualization and inference.

Provides graphical interface with three main tabs:
    1. Training Tab: Monitor live training, replay epochs, stroke playback
    2. Inference Tab: Run paint_main(), dry-run G-code, generate final output
    3. Calibration Tab: Generate calibration G-code, enter measurements, build LUTs

Architecture (decoupled observer):
    - GUI runs as separate process from training
    - Monitors filesystem artifacts via watchdog (no shared state)
    - Maintains own renderer instance for on-demand stroke playback
    - Never reads from or writes to training process memory
    - All communication via atomic file writes

Modules:
    - main_window: QMainWindow with tab container
    - tabs/: TrainingTab, InferenceTab, CalibrationTab
    - widgets/: HDTiledImageViewer, MetricsPlot, StrokePlayback, etc.

Key features:
    - HD-aware: Lazy tile loading for responsive display of large images
    - Live monitoring: Watchdog detects new epoch artifacts atomically
    - Stroke playback: On-demand re-rendering of strokes 1..N
    - Dry run: gcode_vm integration for pre-flight validation
    - Calibration workflow: Guided manual measurement entry

Data sources:
    - outputs/training_monitor/: Training artifacts (target, canvas, strokes, metadata)
    - outputs/checkpoints/: Model checkpoints
    - data/validation_images/: Fixed validation set
    - gcode_output/: Generated G-code files

No Qt in training/inference scripts (optional GUI-free operation).
"""

