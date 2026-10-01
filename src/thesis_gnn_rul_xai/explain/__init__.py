"""Works out how much each sensor, and each walk through the sensor graph, moved a prediction.

Needs PyTorch: pip install -e ".[model]"

Read README.md in this folder. It explains what the numbers mean, how to read
them, and what they must not be compared with.
"""

from .connections import connection_removal, connections_of
from .flipping import (
    FLIPPING_TASKS,
    RANDOM_ORDERS,
    aufc,
    node_flipping,
    subgraph_relevance,
    subgraph_scores,
)
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
    PAPER_GAMMA_START,
    WalkRelevance,
    connection_relevance,
    conservation_error,
    crossing_relevance,
    cycle_relevance,
    indistinguishable_sensors,
    node_relevance,
    paper_gammas,
    step_relevance,
    top_walks,
    walk_relevance,
)

__all__ = [
    "FLIPPING_TASKS",
    "MAX_EXACT_NODES",
    "MAX_WALKS_PER_EXAMPLE",
    "PAPER_GAMMA_START",
    "RANDOM_ORDERS",
    "WalkRelevance",
    "aufc",
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
    "node_flipping",
    "node_relevance",
    "occlusion_values",
    "paper_gammas",
    "plot_contributions_over_time",
    "shap_summary",
    "shap_values",
    "step_relevance",
    "subgraph_relevance",
    "subgraph_scores",
    "top_walks",
    "walk_relevance",
]
