"""The network itself.

Needs PyTorch: pip install -e ".[model]"

Read README.md in this folder. It explains everything these files do.
"""

from .config import ModelConfig, build_model, load_model_config
from .estgnn import ESTGNN
from .layers import GraphConvolution
from .temporal import (
    LastCycle,
    LSTMStage,
    MeanWindow,
    RawWindow,
    TemporalStage,
    WindowSummary,
    build_temporal_stage,
)

__all__ = [
    "ESTGNN",
    "GraphConvolution",
    "LSTMStage",
    "LastCycle",
    "MeanWindow",
    "ModelConfig",
    "RawWindow",
    "TemporalStage",
    "WindowSummary",
    "build_model",
    "build_temporal_stage",
    "load_model_config",
]
