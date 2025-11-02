"""GUI launcher script.

Starts PyQt graphical interface for:
    - Training monitoring and epoch replay
    - Inference execution and G-code generation
    - Calibration workflow management

GUI runs as separate process (decoupled from training):
    - Monitors filesystem artifacts via watchdog
    - Maintains own renderer for stroke playback
    - Never shares state with training process

CLI:
    python scripts/launch_gui.py
    python scripts/launch_gui.py --config configs/gui.yaml

Optional config (configs/gui.yaml):
    - Default paths (checkpoints, targets, outputs)
    - Tile sizes for HD viewer
    - Update intervals for live monitoring
    - Window geometry/state persistence

Requirements:
    - PyQt5 (GUI framework)
    - pyqtgraph (live plotting)
    - watchdog (filesystem monitoring)

Headless CI:
    - Set QT_QPA_PLATFORM=offscreen for smoke tests
    - GUI smoke test verifies window launches without crash

No training or inference logic here (launches MainWindow only).
"""

