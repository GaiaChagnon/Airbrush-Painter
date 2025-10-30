"""GUI tab modules for different workflows.

Tabs:
    - TrainingTab: Live training monitoring, epoch selector, stroke playback
    - InferenceTab: Checkpoint loading, paint_main execution, dry-run, G-code generation
    - CalibrationTab: Calibration G-code generation, manual measurement entry, LUT building

Each tab is a QWidget with self-contained UI and logic.
Tabs communicate with main window via Qt signals (loose coupling).

Common patterns:
    - Long operations run in QThread workers (non-blocking UI)
    - File I/O via src.utils.fs (atomic reads)
    - Rendering via tab-local DifferentiableRenderer instance
    - Errors displayed via QMessageBox or status bar
"""

