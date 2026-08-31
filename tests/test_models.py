"""Tests for the graph network of Table 4.

The most important tests here are the two that check the architecture against
the paper rather than against itself:

- `test_parameter_counts_match_table_4` compares every layer's parameter
  count with the weight shapes the paper prints.
- `test_three_layers_cannot_reach_four_hops` checks that message passing
  respects the sensor graph, using a pair of sensors the graph places four
  edges apart.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from thesis_gnn_rul_xai.graph.adjacency import (
    build_fully_connected_adjacency,
    build_physics_adjacency,
    normalize_adjacency,
)
from thesis_gnn_rul_xai.models import ESTGNN, GraphConvolution


@pytest.fixture(scope="module")
def adjacency() -> np.ndarray:
    """The paper's sensor graph, normalised for graph convolution."""
    raw, _ = build_physics_adjacency()
    return normalize_adjacency(raw)


@pytest.fixture(scope="module")
def symbols() -> tuple[str, ...]:
    return build_physics_adjacency()[1]


# --- the architecture, checked against the paper ------------------------


def test_parameter_counts_match_table_4(adjacency):
    """Every weight matrix must have the shape Table 4 prints.

    The paper gives W(1) in R(1x32), W(2) in R(32x64), W(3) in R(64x32),
    W_fc1 in R(448x128) and W_fc2 in R(128x1). The graph layers carry no bias
    (Equations 8 to 10 show none); the dense layers do (Equations 12 and 13).
    """
    model = ESTGNN(adjacency, in_features=1)
    summary = {row["layer"]: row for row in model.layer_summary()}

    assert summary["GCN-1"]["parameters"] == 1 * 32
    assert summary["GCN-2"]["parameters"] == 32 * 64
    assert summary["GCN-3"]["parameters"] == 64 * 32
    assert summary["Flatten"]["out"] == "(B, 448)"
    assert summary["FC-1"]["parameters"] == 448 * 128 + 128
    assert summary["FC-2"]["parameters"] == 128 * 1 + 1
    assert model.count_parameters() == 61_729


def test_activations_and_shapes_match_table_4(adjacency):
    model = ESTGNN(adjacency, in_features=1)
    rows = model.layer_summary()
    assert [row["activation"] for row in rows] == [
        "-", "ReLU", "ReLU", "ReLU", "-", "ReLU", "Sigmoid"
    ]
    assert [row["out"] for row in rows] == [
        "(B, 14, 1)", "(B, 14, 32)", "(B, 14, 64)", "(B, 14, 32)",
        "(B, 448)", "(B, 128)", "(B, 1)",
    ]


def test_the_default_width_reproduces_table_4(adjacency):
    """The default in_features=1 is Table 4's printed input shape."""
    table4 = ESTGNN(adjacency)
    assert table4.in_features == 1
    assert table4.count_parameters() == 61729


def test_a_wider_first_layer_only_changes_w1(adjacency):
    """in_features stays general; widening it touches W(1) and nothing else."""
    table4 = ESTGNN(adjacency, in_features=1)
    wider = ESTGNN(adjacency, in_features=30)
    assert wider.count_parameters() - table4.count_parameters() == (30 - 1) * 32


# --- what the network produces ------------------------------------------


def test_output_is_one_number_per_example_inside_the_unit_interval(adjacency):
    model = ESTGNN(adjacency, in_features=30)
    batch = torch.randn(8, 14, 30)
    output = model(batch)
    assert output.shape == (8,)
    assert bool((output >= 0).all() and (output <= 1).all())


def test_the_prediction_can_never_exceed_the_ceiling(adjacency):
    """The architectural fact that assumption A-02 rests on.

    The final layer is a sigmoid scaled by ``rul_max``, so no input can make
    the model predict a longer remaining life than the ceiling. Capping the
    test labels is therefore not a convenience: leaving them uncapped would
    charge the model for an error it is structurally unable to avoid.
    """
    model = ESTGNN(adjacency, in_features=30, rul_max=125)
    for scale in (0.0, 1.0, 50.0, 1e4):
        cycles = model.predict_cycles(torch.full((4, 14, 30), scale))
        assert bool((cycles >= 0).all() and (cycles <= 125).all())


def test_the_graph_is_not_a_learned_parameter(adjacency):
    """The physics prior is knowledge put in, not something fitted."""
    model = ESTGNN(adjacency, in_features=30)
    assert "adjacency" in dict(model.named_buffers())
    assert "adjacency" not in dict(model.named_parameters())
    np.testing.assert_allclose(model.adjacency.numpy(), adjacency, rtol=1e-6)


def test_an_unnormalised_adjacency_is_rejected(adjacency):
    """A zero diagonal means normalize_adjacency() was not called."""
    raw, _ = build_physics_adjacency()
    # Match the condition, not the wording of the explanation: the message is
    # prose and gets rewritten, the zero diagonal is what the check is about.
    with pytest.raises(ValueError, match="zero on its diagonal"):
        ESTGNN(raw, in_features=30)


# --- message passing actually follows the graph -------------------------


def test_three_layers_cannot_reach_four_hops(adjacency, symbols):
    """A sensor four edges away is unreachable in three message-passing steps.

    The paper's graph has a diameter of 4, and four sensor pairs sit at that
    distance: P15, Nf, NRf and BPR are each four edges from W32. With the three
    graph convolution layers of Table 4, changing one member of such a pair
    cannot change the other's representation at all.

    This is a property of the published architecture, worth recording because
    it bounds what the model can express, and it doubles as a check that the
    layers really are using the adjacency matrix rather than mixing everything
    together.
    """
    model = ESTGNN(adjacency, in_features=30)
    order = list(symbols)
    far_source, far_target = order.index("P15"), order.index("W32")

    def three_graph_layers(features):
        hidden = features
        for layer in model.graph_layers:
            hidden = torch.relu(layer(hidden, model.adjacency))
        return hidden

    base = torch.randn(1, 14, 30)
    perturbed = base.clone()
    perturbed[0, far_source] += 100.0

    with torch.no_grad():
        before = three_graph_layers(base)
        after = three_graph_layers(perturbed)

    torch.testing.assert_close(before[0, far_target], after[0, far_target])
    # ...and a direct neighbour of P15 must change, or the test proves nothing.
    neighbour = order.index("Nf")
    assert not torch.allclose(before[0, neighbour], after[0, neighbour])


def test_a_fully_connected_graph_reaches_everything_in_one_hop(symbols):
    """The 'without physics prior' ablation removes the structure entirely."""
    dense, _ = build_fully_connected_adjacency(14)
    model = ESTGNN(normalize_adjacency(dense), in_features=30)
    order = list(symbols)

    def one_graph_layer(features):
        return torch.relu(model.graph_layers[0](features, model.adjacency))

    base = torch.randn(1, 14, 30)
    perturbed = base.clone()
    perturbed[0, order.index("P15")] += 100.0
    with torch.no_grad():
        before, after = one_graph_layer(base), one_graph_layer(perturbed)
    assert not torch.allclose(before[0, order.index("W32")], after[0, order.index("W32")])


# --- the network can be trained -----------------------------------------


def test_one_optimiser_step_reduces_the_loss(adjacency):
    """Gradients reach every layer, including the first graph convolution."""
    torch.manual_seed(0)
    model = ESTGNN(adjacency, in_features=30)
    features = torch.randn(16, 14, 30)
    targets = torch.rand(16)

    loss_function = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)

    loss = loss_function(model(features), targets)
    before = loss.detach().item()
    loss.backward()
    assert model.graph_layers[0].weight.grad is not None
    assert model.graph_layers[0].weight.grad.abs().sum().item() > 0
    optimiser.step()
    optimiser.zero_grad()

    with torch.no_grad():
        after = loss_function(model(features), targets).item()
    assert after < before


def test_building_the_model_is_repeatable(adjacency):
    torch.manual_seed(7)
    first = ESTGNN(adjacency, in_features=30)
    torch.manual_seed(7)
    second = ESTGNN(adjacency, in_features=30)
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b)


# --- the layer on its own -----------------------------------------------


def test_layer_averages_over_neighbours():
    """Two nodes, one edge, identity weights: each node gets the pair's mean."""
    layer = GraphConvolution(1, 1)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    matrix = torch.tensor(normalize_adjacency(np.array([[0.0, 1.0], [1.0, 0.0]])))
    output = layer(torch.tensor([[[2.0], [4.0]]]), matrix)
    torch.testing.assert_close(output.squeeze(), torch.tensor([3.0, 3.0]))


def test_layer_rejects_mismatched_shapes(adjacency):
    layer = GraphConvolution(30, 32)
    matrix = torch.tensor(adjacency)
    with pytest.raises(ValueError, match="expects 30 feature"):
        layer(torch.randn(2, 14, 7), matrix)
    with pytest.raises(ValueError, match="node"):
        layer(torch.randn(2, 9, 30), matrix)
    with pytest.raises(ValueError, match="batch, nodes, in_features"):
        layer(torch.randn(14, 30), matrix)
