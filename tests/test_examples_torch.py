"""The worked examples that need PyTorch. See test_examples.py for the rest.

These used to be doctests inside models/layers.py, models/estgnn.py and
explain/shap.py. They were moved here when the explanation was moved into the
folder README files. Without PyTorch installed they skip rather than fail.
"""

from __future__ import annotations

from math import comb, factorial

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from thesis_gnn_rul_xai.explain.shap import coalition_weights
from thesis_gnn_rul_xai.graph.adjacency import (
    build_physics_adjacency,
    normalize_adjacency,
)
from thesis_gnn_rul_xai.models import ESTGNN, GraphConvolution

# --- models/layers.py ---------------------------------------------------


def test_example_graph_convolution():
    """Two sensors joined by one edge, with a weight of 1, so each ends up
    holding the average of the two: (2 + 4) / 2."""
    layer = GraphConvolution(1, 1)
    with torch.no_grad():
        layer.weight.fill_(1.0)

    adjacency = torch.tensor(normalize_adjacency([[0.0, 1.0], [1.0, 0.0]]))
    features = torch.tensor([[[2.0], [4.0]]])

    assert layer(features, adjacency).squeeze().tolist() == [3.0, 3.0]


# --- models/estgnn.py ---------------------------------------------------


def test_example_estgnn():
    """Build the network on the paper's own graph, push four examples through
    it, and confirm it has exactly the number of weights Table 4 implies."""
    raw, _ = build_physics_adjacency()
    model = ESTGNN(normalize_adjacency(raw))

    batch = torch.zeros(4, 14, 1)

    assert model(batch).shape == torch.Size([4])
    assert sum(p.numel() for p in model.parameters()) == 61729


# --- explain/shap.py ----------------------------------------------------


def test_example_coalition_weights():
    """With three sensors, the empty group and the full group each count 1/3,
    and each single-sensor group counts 1/6."""
    assert coalition_weights(3).round(6).tolist() == [0.333333, 0.166667, 0.333333]


def test_example_coalition_weights_add_up_to_one():
    """The weights cover every ordering of the sensors exactly once, so
    multiplying each by the number of groups of that size gives 1."""
    n = 14
    counts = np.array([comb(n - 1, size) for size in range(n)])
    assert round(float(np.sum(counts * coalition_weights(n))), 9) == 1.0
    assert coalition_weights(3)[0] == factorial(0) * factorial(2) / factorial(3)
