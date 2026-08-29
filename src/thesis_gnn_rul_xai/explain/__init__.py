"""Works out how much each sensor moved a prediction, at every cycle.

Needs PyTorch: pip install -e ".[model]"

Read README.md in this folder. It explains what the numbers mean, how to read
them, and what they must not be compared with.
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
