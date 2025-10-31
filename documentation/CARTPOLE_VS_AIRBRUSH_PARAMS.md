# Parameter Comparison: Cartpole vs Airbrush Painter

## Overview

This document compares the PPO hyperparameters from your cartpole project with the adapted values for the Airbrush Painter project, explaining the rationale for each adaptation.

## Side-by-Side Comparison

| Parameter | Cartpole Value | Airbrush Value | Rationale for Change |
|-----------|----------------|----------------|----------------------|
| **Algorithm** | | | |
| `ppo` | `True` | `True` | ✓ Same (PPO recommended for both) |
| `algo.name` | `a2c_continuous` | `a2c_continuous` | ✓ Same (rl_games convention) |
| `model.name` | `continuous_a2c_logstd` | `continuous_a2c_logstd` | ✓ Same (continuous actions) |
| **Core Hyperparameters** | | | |
| `learning_rate` | `5.0e-5` | `3.0e-4` | **Higher** - Visual features are richer than cart state |
| `gamma` | `0.99` | `0.995` | **Higher** - Longer horizon (1500 strokes vs stabilization) |
| `tau` | `0.985` | `0.95` | **Lower** - Standard PPO value, less aggressive GAE |
| `entropy_coef` | `0.01` | `0.001` | **Lower** - Dense visual obs needs less exploration |
| `e_clip` | `0.2` | `0.2` | ✓ Same (standard PPO) |
| `kl_threshold` | *Not specified* | `0.008` | **Added** - Adaptive LR stability |
| **Batch Parameters** | | | |
| `horizon_length` | `320` | `256` | **Shorter** - Faster updates for visual feedback |
| `minibatch_size` | `32768` | `8192` | **Smaller** - Fewer parallel envs (memory constraint) |
| `mini_epochs` | `8` | `4` | **Fewer** - Visual diversity reduces overfitting risk |
| **Value Function** | | | |
| `critic_coef` | `7` | `2.0` | **Lower** - Dense LPIPS reward needs less value emphasis |
| `clip_value` | `True` | `True` | ✓ Same (stabilizes training) |
| `normalize_value` | `True` | `True` | ✓ Same (standard PPO) |
| **Gradient & Regularization** | | | |
| `grad_norm` | `1.0` | `1.0` | ✓ Same (standard clipping) |
| `truncate_grads` | `True` | `True` | ✓ Same (enable clipping) |
| `bounds_loss_coef` | `0.008` | `0.0001` | **Lower** - Less aggressive boundary penalty |
| **Normalization** | | | |
| `normalize_input` | `True` | `True` | ✓ Same (stabilizes CNN inputs) |
| `normalize_advantage` | `True` | `True` | ✓ Same (standard PPO) |
| **Training Loop** | | | |
| `max_epochs` | `1250` | `10000` | **Much higher** - Complex visual task needs more training |
| `save_best_after` | `50` | `50` | ✓ Same |
| `save_frequency` | `10` | `10` | ✓ Same |
| **Environment** | | | |
| `clip_observations` | `35.0` | `10.0` | **Lower** - Images in [0,1], less clipping needed |
| `clip_actions` | `1.0` | `1.0` | ✓ Same (actions in [-1,1]) |
| **Network Architecture** | | | |
| `network.mlp.units` | `[512, 512, 512]` | `[512, 512, 512]` | ✓ Same (3-layer MLP) |
| `network.mlp.activation` | `elu` | `elu` | ✓ Same |
| `network.fixed_sigma` | `True` | `True` | ✓ Same (fixed exploration noise) |
| **Device & Precision** | | | |
| `device` | `cuda:0` | `cuda:0` | ✓ Same |
| `mixed_precision` | `False` | `False` | ✓ Same (handled externally via BF16) |
| **Cartpole-Specific** | | | |
| `env_cfg.rew_scale_alive` | `2.5` | *N/A* | Removed - No survival bonus in painting |
| `env_cfg.rew_scale_pole_angle` | `6` | *N/A* | Removed - Task-specific |
| `env_cfg.rew_scale_cart_vel` | `-0.001` | *N/A* | Removed - No velocity penalty |
| `env_cfg.rew_bonus_*` | Various | *Template only* | Commented out - LPIPS-only reward currently |

## Key Adaptations Explained

### 1. Learning Rate: 5e-5 → 3e-4 (6× Higher)

**Cartpole**: Tiny state space (4D: position, velocity, angle, angular velocity)  
**Airbrush**: Rich visual space (9 channels × 454×640 = 2.6M dimensions)

**Rationale**: CNNs with visual inputs benefit from higher learning rates because:
- Feature extraction is more complex
- Gradient signals are distributed across many parameters
- Modern ResNet + BF16 + Adam handle higher LR well

### 2. Gamma: 0.99 → 0.995 (More Farsighted)

**Cartpole**: Episode ends when pole falls (~200-500 steps)  
**Airbrush**: Fixed horizon of 1500 strokes

**Rationale**: Painting requires long-term planning:
- Early strokes affect later composition
- LPIPS improvement compounds over time
- Higher gamma = better credit assignment for early decisions

### 3. Tau: 0.985 → 0.95 (Less Aggressive GAE)

**Cartpole**: Very high tau (close to Monte Carlo)  
**Airbrush**: Standard PPO tau

**Rationale**: 
- Cartpole has very clean dynamics (physics simulator)
- Airbrush has noisier reward (LPIPS fluctuations)
- Lower tau = more bias toward value function = smoother training

### 4. Entropy: 0.01 → 0.001 (10× Less Exploration)

**Cartpole**: Needs aggressive exploration to discover balance strategies  
**Airbrush**: Dense visual feedback guides exploration naturally

**Rationale**:
- Visual observations are information-rich (9 channels)
- LPIPS provides dense reward signal every stroke
- Too much entropy = random painting, not useful

### 5. Horizon Length: 320 → 256 (Faster Updates)

**Cartpole**: Long rollouts for stable cart dynamics  
**Airbrush**: Shorter rollouts for responsive visual feedback

**Rationale**:
- Images change significantly each stroke
- Faster updates = quicker adaptation to target
- 256 is still long enough for credit assignment

### 6. Minibatch Size: 32768 → 8192 (4× Smaller)

**Cartpole**: Many parallel envs, small state vector (4D)  
**Airbrush**: Fewer parallel envs, huge state (2.6M dimensions)

**Rationale**:
- GPU memory constraint: 454×640×9×FP32 = ~10MB per observation
- 32 envs × 256 rollout = 8192 samples (perfect fit)
- Still large enough for stable gradient estimates

### 7. Mini Epochs: 8 → 4 (Less Replay)

**Cartpole**: Clean dynamics = high-quality replay  
**Airbrush**: Stochastic renderer + visual diversity = less replay value

**Rationale**:
- Each target image is different (high diversity)
- Overfitting to old rollouts less useful than in cartpole
- 4 epochs standard for PPO with rich observations

### 8. Critic Coef: 7 → 2.0 (3.5× Lower)

**Cartpole**: Sparse reward (pole falling is catastrophic)  
**Airbrush**: Dense reward (LPIPS every stroke)

**Rationale**:
- Dense rewards = policy can learn directly from environment
- Less need for accurate value estimates
- Lower critic coef = more emphasis on policy improvement

### 9. Clip Observations: 35.0 → 10.0

**Cartpole**: State includes velocities (can be very large)  
**Airbrush**: Images in [0,1], errors in [-1,1]

**Rationale**:
- Visual observations are naturally bounded
- 10.0 is safety margin for any normalization artifacts
- Too high clip = no effect, too low = distorts inputs

### 10. Max Epochs: 1250 → 10000

**Cartpole**: Simple balancing task, converges quickly  
**Airbrush**: Complex visual reasoning, needs extensive training

**Rationale**:
- Learning to paint 1500-stroke compositions is hard
- ResNet feature learning takes time
- HPO will find optimal stopping point

## Reward Structure Comparison

### Cartpole: Multi-Component Shaped Reward
```python
total_reward = (
    2.5 * alive_bonus +                    # Survival incentive
    6.0 * pole_angle_reward +              # Primary objective
    -0.001 * cart_velocity_penalty +       # Efficiency
    -0.01 * pole_velocity_penalty +        # Smoothness
    0.25 * pole_straight_reward +          # Alignment
    0.4 * cart_center_reward +             # Centering
    milestone_bonuses                      # 500, 1000, 2000
)
```

**Why complex?**
- Physics simulation has clean ground truth
- Multiple competing objectives (balance, center, minimize movement)
- Sparse base reward (pole falling)

### Airbrush: Single Dense Reward
```python
total_reward = lpips_improvement  # Current - Previous LPIPS
```

**Why simple?**
- LPIPS is comprehensive perceptual metric
- Already dense (computed every stroke)
- Multi-scale (captures low & high-freq errors)

**Future extensions** (commented in config):
- Stroke efficiency penalty
- Color accuracy bonus
- Coverage bonus
- Edge preservation bonus
- Milestone bonuses (LPIPS < 0.5, 0.2, 0.1)

## Network Architecture Comparison

### Cartpole: Simple MLP
```yaml
network:
  mlp:
    units: [512, 512, 512]
    activation: elu
  input: 4D vector (pos, vel, angle, ang_vel)
```

### Airbrush: ResNet34 + Spatial Head
```yaml
agent:
  backbone: resnet34         # CNN for visual features
  spatial_head: heatmap_soft_argmax  # Differentiable coordinates
network:
  mlp:
    units: [512, 512, 512]   # Same MLP size (for non-spatial params)
    activation: elu
  input: 9×454×640 image (target, canvas, error)
```

**Key differences**:
- Airbrush adds ResNet backbone for spatial feature extraction
- Airbrush uses heatmap→soft-argmax for coordinate prediction
- Same MLP head size (after spatial pooling)

## Action Space Comparison

### Cartpole: 1D Continuous
- Force applied to cart (-1 to +1)

### Airbrush: 15D Continuous
- 8D: Bézier control points (4 xy pairs)
- 2D: Z-height (z0, z1)
- 2D: Velocity (v0, v1)
- 3D: Color (cyan, magenta, yellow)

**Implications**:
- Airbrush needs spatial reasoning (CNN backbone)
- Airbrush needs separate heads for coordinates vs scalars
- Both normalize to [-1,1] for training stability

## Observation Space Comparison

### Cartpole: 4D Vector
```
[cart_position, cart_velocity, pole_angle, pole_angular_velocity]
```
- Total: 4 floats
- Fully observable
- Markovian

### Airbrush: 9-Channel Image
```
[target_r, target_g, target_b,    # What we want
 canvas_r, canvas_g, canvas_b,    # What we have
 error_r, error_g, error_b]       # Difference
```
- Total: 2.6 million floats (454×640×9)
- Fully observable
- Markovian (includes canvas history)

## When to Use Each Set of Hyperparameters

### Use Cartpole-Style Params When:
✓ State space is low-dimensional (< 100D)  
✓ Dynamics are deterministic and simple  
✓ Reward is sparse (needs shaping)  
✓ Episodes are short (< 1000 steps)  
✓ Exploration is critical (multimodal solution space)  

**Examples**: Classic control (cartpole, pendulum), robotic reaching

### Use Airbrush-Style Params When:
✓ State space is high-dimensional (images)  
✓ Dynamics are complex or stochastic  
✓ Reward is dense (already shaped)  
✓ Episodes are long (> 1000 steps)  
✓ Observations are information-rich  

**Examples**: Image-based tasks, vision-based robotics, dense reward games

## Tuning Recommendations

### If Training is Unstable:
- **Reduce `learning_rate`**: Try 1e-4 (Airbrush) or 3e-5 (Cartpole)
- **Increase `grad_norm`**: Try 1.5 or 2.0
- **Reduce `e_clip`**: Try 0.15 or 0.1
- **Increase `critic_coef`**: Try 4.0 (Airbrush) or 10.0 (Cartpole)

### If Training is Too Slow:
- **Increase `learning_rate`**: Try 5e-4 (Airbrush) or 1e-4 (Cartpole)
- **Increase `mini_epochs`**: Try 6 or 8
- **Increase `entropy_coef`**: Try 0.005 (Airbrush) or 0.02 (Cartpole)

### If Policy is Too Random:
- **Reduce `entropy_coef`**: Try 0.0005 (Airbrush) or 0.005 (Cartpole)
- **Increase `mini_epochs`**: More exploitation of collected data
- **Reduce `e_clip`**: Smaller policy updates

### If Policy is Too Deterministic (Stuck in Local Optimum):
- **Increase `entropy_coef`**: Try 0.005 (Airbrush) or 0.02 (Cartpole)
- **Increase `learning_rate`**: Escape local minima faster
- **Reduce `mini_epochs`**: Less overfitting to current data

## Summary

The Airbrush Painter config adapts your cartpole hyperparameters for a fundamentally different task:

1. **Higher learning rate** (6×) for visual complexity
2. **More farsighted** (higher gamma) for long horizon
3. **Less exploration** (lower entropy) for dense reward
4. **Smaller batches** (4×) for memory constraints
5. **Less value emphasis** (lower critic coef) for dense reward
6. **Longer training** (8×) for visual complexity

Both configs follow rl_games conventions and PPO best practices, adapted to their respective task characteristics.

---

**Config Philosophy**: Start with proven hyperparameters (cartpole), adapt systematically for new task properties (visual, dense reward, long horizon), tune via HPO.

