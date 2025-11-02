"""Reusable Qt widgets for GUI.

Widgets:
    - HDTiledImageViewer: Lazy tile-loading image viewer for large canvases
    - MetricsPlotWidget: pyqtgraph-based live metric plotting
    - StrokePlaybackControls: First/Prev/Next/Last/Play/Pause, slider
    - FilePickerWidget: Path input with browse button
    - ConfigEditorWidget: YAML editor with validation
    - LogViewer: Tail log files with syntax highlighting

HDTiledImageViewer:
    - Loads image tiles on-demand (viewport-based culling)
    - Smooth pan/zoom (QGraphicsView)
    - Handles images larger than GPU memory
    - Configurable tile size (default: 512×512)

MetricsPlotWidget:
    - Reads MLflow CSVs or live metrics.json
    - Multiple series (reward, LPIPS, loss, etc.)
    - Zoom, pan, legend toggle
    - Export to PNG/SVG

All widgets use Qt Model/View pattern where applicable (separation of data/presentation).
"""

