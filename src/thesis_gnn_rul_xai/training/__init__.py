"""
Training
========

The loop that fits the network of Table 4 using the hyper-parameters of
Table 6, together with the settings object that records them.

Needs PyTorch; install it with the ``model`` extra (``pip install -e ".[model]"``).
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
