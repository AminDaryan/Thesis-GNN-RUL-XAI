"""Trains the network, and records the settings that were used.

Needs PyTorch: pip install -e ".[model]"

Read README.md in this folder. It explains everything these files do.
"""

from .config import TrainingConfig, load_training_config
from .loop import TrainingHistory, evaluate_model, predict, resolve_device, train

__all__ = [
    "TrainingConfig",
    "TrainingHistory",
    "evaluate_model",
    "load_training_config",
    "predict",
    "resolve_device",
    "train",
]
