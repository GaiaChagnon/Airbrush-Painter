"""Policy networks with spatial-aware coordinate prediction.

Provides two spatial head options for coordinate prediction:

Option A - CoordConv:
    - Prepend 2 normalized (x,y) coord channels to 9-ch obs → 11 channels
    - Standard ResNet backbone + linear heads for all 15 action dims
    - Simple, effective baseline

Option B - Heatmap + soft-argmax (recommended):
    - Backbone returns spatial feature map (B, 512, H', W') - NO GAP!
    - Heatmap head: Conv2d produces K=4 heatmaps (one per Bézier control point)
    - Soft-argmax: Differentiable coordinate extraction from heatmaps
    - Separate MLP heads for non-spatial params (z, v, cmy) using GAP'd features
    - Temperature-controlled sharpness (configurable via softargmax_temp)

Public API:
    build_actor_critic(cfg) → (actor, critic)
        Selects head type based on cfg.agent.spatial_head

Key implementation details:
    - Soft-argmax uses pixel-center convention: [0.5, 1.5, ..., W-0.5] / W
    - Output action in [-1,1]^15 (environment handles denormalization)
    - Backbone must NOT apply GAP before heatmap head (spatial info required)
    - Channels-last memory format for CNN efficiency (free on DGX)
    - BF16 autocast in forward; FP32 in loss/optimizer

Hyperparameters (from train.yaml):
    - agent.backbone: "resnet34" | "resnet18" | ...
    - agent.spatial_head: "coordconv" | "heatmap_soft_argmax"
    - agent.softargmax_temp: float (lower = sharper, higher = smoother)
    - agent.learning_rate, entropy_coef, gamma, clip_param

Used by:
    - scripts/train.py: rl-games PPO training
    - HPO: Temperature tuning for coordinate prediction accuracy
"""

