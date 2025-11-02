"""Differentiable airbrush physics simulator.

Provides GPU-accelerated rendering with learned lookup tables (LUTs):
    - Color LUT: CMY → linear RGB
    - Alpha LUT: (Z, speed) → coverage factor
    - PSF LUT: (Z, speed) → point spread function kernel

Architecture:
    - Primary backend: nvdiffrast CUDA rasterizer (differentiable)
    - Fallback: Gaussian splat rasterizer (no hardware deps)
    - Technician optimizer: Gradient descent on stroke params

Modules:
    - differentiable_renderer: Production renderer + Technician refinement

Invariants:
    - All stroke params in millimeters (mm)
    - LUTs kept in FP32 for numerical stability
    - Networks run in BF16 (DGX Spark default)
    - Parameter projection each step: clamp to valid ranges

Used by:
    - env_v1.py: Forward rendering (no grad) for RL training
    - paint.py: Strategist → Technician loop for inference
    - GUI: On-demand stroke playback rendering
"""

