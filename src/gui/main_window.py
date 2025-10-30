"""Main GUI window with tabbed interface.

QMainWindow containing:
    - TrainingTab: Live monitoring and epoch replay
    - InferenceTab: Paint_main execution and G-code generation
    - CalibrationTab: Calibration workflow management

Public API:
    app = QApplication(sys.argv)
    window = MainWindow(config_paths)
    window.show()
    sys.exit(app.exec_())

Features:
    - Tab navigation for different workflows
    - Shared status bar for global messages
    - Menu bar: File (load checkpoint, open target), Help (about, docs)
    - Non-blocking operations via QThread workers

Configuration:
    Loads GUI settings from configs/gui.yaml (optional):
        - Default paths (checkpoints, targets, output dirs)
        - Tile sizes for HD viewer
        - Update intervals for live monitoring

Lifecycle:
    - Spawns watchdog file observer on startup
    - Cleans up threads on close (QThread.quit(), wait())
    - Saves window geometry/state to ~/.airbrush_painter/gui_state.yaml

No training or inference logic here (pure presentation layer).
"""

