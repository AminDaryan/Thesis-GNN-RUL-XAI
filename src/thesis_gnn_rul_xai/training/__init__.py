"""Trains the network, and records the settings that were used.

Needs PyTorch: pip install -e ".[model]"

Read README.md in this folder. It explains everything these files do.
"""

from .config import TrainingConfig, load_training_config
from .run import TrainingHistory, evaluate_model, predict, resolve_device, train
from .stored import StoredRun, find_runs, load_seed, read_run

__all__ = [
    "StoredRun",
    "TrainingConfig",
    "TrainingHistory",
    "evaluate_model",
    "find_runs",
    "load_seed",
    "load_training_config",
    "predict",
    "read_run",
    "resolve_device",
    "train",
]
