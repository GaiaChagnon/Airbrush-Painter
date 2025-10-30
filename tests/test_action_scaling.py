"""Test action space normalization and denormalization.

Validates that actions are correctly scaled between [-1,1] and mm-space:
    - Action space is [-1,1]^15
    - Denormalization: [-1,-1,...] → [0, 0, ...] mm (mins)
    - Denormalization: [1,1,...] → [work_area_mm.x, work_area_mm.y, ...] mm (maxs)
    - project_params() clamps denormalized actions to valid ranges
    - Scaling computed from schema bounds (no hard-coded magic numbers)

Test cases:
    - test_action_space_shape()
    - test_action_space_range()
    - test_denormalize_min_action()
    - test_denormalize_max_action()
    - test_denormalize_zero_action()
    - test_project_params_clamps()
    - test_scaling_from_schema_bounds()

Schema bounds (stroke.v1.yaml):
    - x,y: [0, work_area_mm.x], [0, work_area_mm.y]
    - z: [0.0, 30.0] mm
    - speed: [1.0, 300.0] mm/s
    - c,m,y: [0.0, 1.0]

Run:
    pytest tests/test_action_scaling.py -v
"""

