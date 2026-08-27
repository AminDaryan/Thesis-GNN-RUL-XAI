"""
Explainability
==============

Dynamic SHAP: Shapley values for the 14 sensor nodes, recomputed at every
cycle of an engine's life, as Section 2.4 of Wen et al. (2026) describes.

With only 14 players the Shapley values can be enumerated exactly rather than
sampled, so the numbers here are Equation 15 itself and not an estimate of it.

Needs PyTorch; install it with the ``model`` extra (``pip install -e ".[model]"``).
"""

from .shap import (
    MAX_EXACT_NODES,
    coalition_weights,
    dynamic_shap,
    efficiency_error,
    mean_window_baseline,
    shap_summary,
    shap_values,
)

__all__ = [
    "MAX_EXACT_NODES",
    "coalition_weights",
    "dynamic_shap",
    "efficiency_error",
    "mean_window_baseline",
    "shap_summary",
    "shap_values",
]
