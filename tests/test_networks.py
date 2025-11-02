"""Test policy networks and spatial heads.

Tests for src.rl_agent.networks:
    - Forward pass on dummy observation
    - Output shape (15,) in [-1,1]
    - Parameter counts reasonable
    - BF16 autocast path (no errors)
    - Spatial head: feature map preservation (not GAP'd prematurely)
    - Soft-argmax coordinate extraction accuracy

Test cases:
    - test_coordconv_forward_pass()
    - test_heatmap_soft_argmax_forward_pass()
    - test_spatial_feature_map_preserved()
    - test_soft_argmax_pixel_center_convention()
    - test_bf16_autocast_no_errors()
    - test_action_output_normalized()

Synthetic coordinate prediction test:
    - Create synthetic target with bright dot at known location
    - Supervised training step
    - Verify predicted coordinates converge near dot

Run:
    pytest tests/test_networks.py -v
"""

