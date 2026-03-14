"""GPU digital twin for the airbrush painting robot.

Provides a config-driven, differentiable, batch-capable stamp-splatting
renderer calibrated against real scanned measurements.

Modules
-------
gpu_simulator
    GPUStampSimulator -- the core rendering engine.
calibration_cli
    Block-based calibration wizard (Rich + questionary).
"""

from digital_twin.gpu_simulator import GPUStampSimulator

__all__ = ["GPUStampSimulator"]
