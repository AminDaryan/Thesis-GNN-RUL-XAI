"""Works out how much each sensor, and each walk through the sensor graph, moved a prediction.

Needs PyTorch: pip install -e ".[model]"

Read README.md in this folder. It explains what the numbers mean, how to read
them, and what they must not be compared with.
"""

from .connections import connection_removal, connections_of
from .occlusion import occlusion_values
from .plots import plot_contributions_over_time
from .shap import (
    MAX_EXACT_NODES,
    coalition_weights,
    dynamic_shap,
    efficiency_error,
    mean_window_baseline,
    shap_summary,
    shap_values,
)
from .walks import (
    MAX_WALKS_PER_EXAMPLE,
    WalkRelevance,
    connection_relevance,
    conservation_error,
    crossing_relevance,
    cycle_relevance,
    indistinguishable_sensors,
    node_relevance,
    step_relevance,
    top_walks,
    walk_relevance,
)

__all__ = [
    "MAX_EXACT_NODES",
    "MAX_WALKS_PER_EXAMPLE",
    "WalkRelevance",
    "coalition_weights",
    "connection_relevance",
    "connection_removal",
    "connections_of",
    "conservation_error",
    "crossing_relevance",
    "cycle_relevance",
    "dynamic_shap",
    "efficiency_error",
    "indistinguishable_sensors",
    "mean_window_baseline",
    "node_relevance",
    "occlusion_values",
    "plot_contributions_over_time",
    "shap_summary",
    "shap_values",
    "step_relevance",
    "top_walks",
    "walk_relevance",
]
