"""
Configuration module.

Loads and validates machine configuration from YAML files.
"""

from robot_control.configs.loader import MachineConfig, load_config

__all__ = ["MachineConfig", "load_config"]
