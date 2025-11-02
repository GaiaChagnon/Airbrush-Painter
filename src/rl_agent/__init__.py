"""RL agent: policy networks and training integration.

Provides PPO-compatible actor-critic networks with spatial-aware heads:
    - Strategist: Proposes stroke parameters from observation
    - Critic: Value function for advantage estimation

Spatial head options (configurable):
    - CoordConv: Prepend normalized (x,y) coord channels to input
    - Heatmap + soft-argmax: Convolutional heatmaps → differentiable coordinates

Modules:
    - networks: Actor/critic builders, spatial head implementations

Architecture:
    - Backbone: ResNet-34 (or configurable alternative)
    - Input: (9, obs_h, obs_w) stacked observation
    - Output: 15-D action in normalized [-1,1] space
    - Precision: BF16 autocast for forward, FP32 for loss accumulation

Training integration:
    - Compatible with rl-games PPO trainer
    - Observation at obs_px (downsampled for speed)
    - Action mapped to mm-space by environment

No G-code generation, rendering, or reward computation here (separation of concerns).
"""

