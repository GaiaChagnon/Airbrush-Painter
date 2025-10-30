"""Test paint_main() callable API.

Validates that scripts.paint.paint_main() is callable without MLflow:
    - Returns dict with expected keys
    - final_canvas is torch.Tensor
    - final_lpips is float
    - strokes_path, gcode_path are strings
    - Stroke cap respected
    - Outputs exist and are valid

Test cases:
    - test_paint_main_callable()
    - test_paint_main_return_dict_structure()
    - test_paint_main_without_mlflow()
    - test_paint_main_respects_stroke_cap()
    - test_paint_main_outputs_exist()
    - test_paint_main_synthetic_tiny_image()

Synthetic test:
    - Tiny 32×32 target
    - Small stroke cap (e.g., 10)
    - Verify completes without errors
    - Check outputs are well-formed

Used by HPO validation loop (must not require MLflow active run).

Run:
    pytest tests/test_paint_main.py -v
"""

