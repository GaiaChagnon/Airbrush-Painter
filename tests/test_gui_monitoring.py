"""Test GUI training monitor and watchdog integration.

Validates decoupled GUI observes training artifacts correctly:
    - Watchdog detects new epoch directories
    - Epoch selector populates from filesystem scan
    - Stroke playback renders correctly
    - GUI never reads training process memory
    - Atomic file writes prevent partial reads

Test cases:
    - test_watchdog_detects_new_epoch()
    - test_epoch_selector_population()
    - test_stroke_playback_rendering()
    - test_load_epoch_artifacts()
    - test_no_process_memory_access()

Mock training monitor structure:
    outputs/training_monitor/
        epoch_0/
            target.png
            canvas.png
            strokes.yaml
            metadata.yaml
        epoch_10/
            ...
        latest/ → epoch_10

Workflow simulation:
    1. Create mock epoch directory
    2. Write artifacts atomically
    3. Symlink latest
    4. Verify GUI callback triggers
    5. Verify GUI loads correct data

Run:
    pytest tests/test_gui_monitoring.py -v
"""

