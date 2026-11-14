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
from thesis_gnn_rul_xai.models import (
    ESTGNN,
    GraphConvolution,
    LastCycle,
    LSTMStage,
    RawWindow,
)


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
    model = ESTGNN(adjacency)
    summary = {row["layer"]: row for row in model.layer_summary()}

    assert summary["Input"]["parameters"] == 0      # Table 4 prints a dash
    assert summary["GCN-1"]["parameters"] == 1 * 32
    assert summary["GCN-2"]["parameters"] == 32 * 64
    assert summary["GCN-3"]["parameters"] == 64 * 32
    assert summary["Flatten"]["out"] == "(B, 448)"
    assert summary["FC-1"]["parameters"] == 448 * 128 + 128
    assert summary["FC-2"]["parameters"] == 128 * 1 + 1
    assert model.count_parameters() == 61_729


def test_activations_and_shapes_match_table_4(adjacency):
    """Table 4's seven rows, with the temporal stage folded into its Input row.

    Table 4 begins at "Input | Node features | [batch size,14,1]", where the
    paper's recurrent stage has already finished, and prints no row for that
    stage. The summary counts the stage on the Input row (decided by the author
    on 2026-10-02), so with a one-cycle window and the default stage, which
    learns nothing, every row has Table 4's shapes and activation.
    """
    model = ESTGNN(adjacency)
    rows = model.layer_summary()
    assert [row["layer"] for row in rows] == [
        "Input", "GCN-1", "GCN-2", "GCN-3", "Flatten", "FC-1", "FC-2"
    ]
    assert [row["type"] for row in rows] == [
        "Node features", "GCN", "GCN", "GCN", "Tensor reshape",
        "Fully connected", "Fully connected",
    ]
    assert rows[0]["parameters"] == 0 and rows[0]["from"] == ""   # LastCycle learns nothing
    assert [row["activation"] for row in rows] == [
        "-", "ReLU", "ReLU", "ReLU", "-", "ReLU", "Sigmoid"
    ]
    assert [row["in"] for row in rows] == [
        "(B, 14, 1)", "(B, 14, 1)", "(B, 14, 32)", "(B, 14, 64)", "(B, 14, 32)",
        "(B, 448)", "(B, 128)",
    ]
    assert [row["out"] for row in rows] == [
        "(B, 14, 1)", "(B, 14, 32)", "(B, 14, 64)", "(B, 14, 32)",
        "(B, 448)", "(B, 128)", "(B, 1)",
    ]


def test_the_default_width_reproduces_table_4(adjacency):
    """The default stage gives Table 4's printed input shape."""
    table4 = ESTGNN(adjacency)
    assert table4.in_features == 1
    assert table4.count_parameters() == 61729


def test_a_wider_first_layer_only_changes_w1(adjacency):
    """Widening what reaches the graph layer touches W(1) and nothing else."""
    table4 = ESTGNN(adjacency, temporal=LastCycle(30))
    wider = ESTGNN(adjacency, temporal=RawWindow(30),
                   allow_time_in_the_graph_layer=True)
    assert wider.count_parameters() - table4.count_parameters() == (30 - 1) * 32


def test_raw_cycles_reaching_the_graph_layer_are_refused_unless_asked_for():
    """The refusal that makes the 2026-08-28 mistake impossible to repeat.

    A stage that widens a node without holding any parameters is handing the
    graph layer raw cycles, which is a temporal encoder inside a layer that is
    not named one. The model works that out from the stage rather than
    trusting it, so a hand-written passthrough cannot slip past either.
    """
    matrix, _ = build_physics_adjacency()
    normalized = normalize_adjacency(matrix, method="row")
    with pytest.raises(ValueError, match="without learning anything across"):
        ESTGNN(normalized, temporal=RawWindow(30))
    allowed = ESTGNN(normalized, temporal=RawWindow(30),
                     allow_time_in_the_graph_layer=True)
    assert allowed.time_folded_into_graph_layer is True


def test_the_window_summary_is_built_only_when_named():
    """Three numbers made from the cycles, learning nothing: refused unless asked for by name."""
    from thesis_gnn_rul_xai.models import ModelConfig, WindowSummary, build_model

    normalized = normalize_adjacency(build_physics_adjacency()[0], method="row")
    with pytest.raises(ValueError, match="without learning anything across"):
        ESTGNN(normalized, temporal=WindowSummary(30))
    named = build_model(normalized, ModelConfig(temporal="window_summary"),
                        window_size=30, rul_max=125)
    assert named.in_features == 3
    assert tuple(named.graph_layers[0].weight.shape) == (3, 32)
    assert tuple(named(torch.zeros(2, 14, 30)).shape) == (2,)


def test_a_learned_stage_is_not_treated_as_raw_cycles():
    """An LSTM holds parameters, so widening past one number is not the trap."""
    matrix, _ = build_physics_adjacency()
    stage = LSTMStage(window_size=30, n_nodes=14, hidden_size=8, out_features=4)
    model = ESTGNN(normalize_adjacency(matrix, method="row"), temporal=stage)
    assert model.in_features == 4
    assert model.time_folded_into_graph_layer is False


def test_the_paper_gives_every_sensor_its_own_network():
    """Section 2.2: "Each sensor s is associated with a dedicated LSTM network".

    Fourteen cells, not one shared, so the count scales with the sensors.
    """
    per_sensor = LSTMStage(window_size=30, n_nodes=14, hidden_size=32)
    shared = LSTMStage(window_size=30, n_nodes=14, hidden_size=32,
                       shared_across_sensors=True)
    assert len(per_sensor.cells) == 14
    assert len(shared.cells) == 1
    per = sum(p.numel() for p in per_sensor.parameters())
    one = sum(p.numel() for p in shared.parameters())
    assert per == one + 13 * sum(p.numel() for p in shared.cells[0].parameters())


# --- what the network produces ------------------------------------------


def test_output_is_one_number_per_example_inside_the_unit_interval(adjacency):
    model = ESTGNN(adjacency, temporal=LastCycle(30))
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
    model = ESTGNN(adjacency, temporal=LastCycle(30), rul_max=125)
    for scale in (0.0, 1.0, 50.0, 1e4):
        cycles = model.predict_cycles(torch.full((4, 14, 30), scale))
        assert bool((cycles >= 0).all() and (cycles <= 125).all())


def test_the_graph_is_not_a_learned_parameter(adjacency):
    """The physics prior is knowledge put in, not something fitted."""
    model = ESTGNN(adjacency, temporal=LastCycle(30))
    assert "adjacency" in dict(model.named_buffers())
    assert "adjacency" not in dict(model.named_parameters())
    np.testing.assert_allclose(model.adjacency.numpy(), adjacency, rtol=1e-6)


def test_an_unnormalised_adjacency_is_rejected(adjacency):
    """A zero diagonal means normalize_adjacency() was not called."""
    raw, _ = build_physics_adjacency()
    # Match the condition, not the wording of the explanation: the message is
    # prose and gets rewritten, the zero diagonal is what the check is about.
    with pytest.raises(ValueError, match="zero on its diagonal"):
        ESTGNN(raw, temporal=LastCycle(30))


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
    model = ESTGNN(adjacency, temporal=RawWindow(30), allow_time_in_the_graph_layer=True)
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
    model = ESTGNN(normalize_adjacency(dense), temporal=RawWindow(30), allow_time_in_the_graph_layer=True)
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
    model = ESTGNN(adjacency, temporal=LastCycle(30))
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
    first = ESTGNN(adjacency, temporal=LastCycle(30))
    torch.manual_seed(7)
    second = ESTGNN(adjacency, temporal=LastCycle(30))
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


# --- the temporal stage folded into Table 4's Input row (2026-10-03) ------


@pytest.mark.parametrize(
    "name", ["last_cycle", "mean_window", "window_summary", "raw_window", "lstm", "gru"]
)
def test_the_input_row_carries_the_stage_and_every_learned_number_appears_once(adjacency, name):
    """The Input row counts the stage and names it only when it learns; the rows add up to the model."""
    from thesis_gnn_rul_xai.models import build_temporal_stage

    stage = build_temporal_stage(name, window_size=30, n_nodes=14, hidden_size=8)
    # The two stages that widen a node are refused unless asked for in words.
    model = ESTGNN(adjacency, temporal=stage,
                   allow_time_in_the_graph_layer=name in ("window_summary", "raw_window"))
    rows = model.layer_summary()
    first = rows[0]
    learned = sum(p.numel() for p in stage.parameters())
    assert (learned > 0) == (name in ("lstm", "gru"))
    assert len(rows) == 7 and first["layer"] == "Input"
    assert first["in"] == "(B, 14, 30)"
    assert first["out"] == f"(B, 14, {stage.out_features})"
    assert first["parameters"] == learned
    assert first["from"] == {"lstm": "LSTM", "gru": "GRU"}.get(name, "")
    assert sum(int(row["parameters"]) for row in rows) == model.count_parameters()


# --- spatial="none": the paper's "ESTGNN w/o GCN" as assumption A-42 reads it ---


def _built(adjacency, temporal: str, spatial: str):
    """A network from a ModelConfig, on the paper's graph, for 30-cycle windows."""
    from thesis_gnn_rul_xai.models import ModelConfig, build_model

    return build_model(adjacency, ModelConfig(temporal=temporal, spatial=spatial),
                       window_size=30, rul_max=125)


def test_without_graph_layers_each_sensors_number_goes_straight_to_the_dense_layers(adjacency):
    """(B, 14, 30) -> stage -> (B, 14, 1) -> flatten (B, 14) -> FC-1 (B, 128) -> FC-2 (B,)."""
    model = _built(adjacency, "last_cycle", "none")
    assert model.has_graph_layers is False and len(model.graph_layers) == 0
    assert model.flattened_size == 14 * 1
    assert tuple(model.fc1.weight.shape) == (128, 14)
    assert model(torch.zeros(5, 14, 30)).shape == torch.Size([5])
    rows = model.layer_summary()
    assert [row["layer"] for row in rows] == ["Input", "Flatten", "FC-1", "FC-2"]
    assert rows[1]["in"] == "(B, 14, 1)" and rows[1]["out"] == "(B, 14)"
    assert sum(int(row["parameters"]) for row in rows) == model.count_parameters()
    assert _built(adjacency, "last_cycle", "gcn").has_graph_layers is True


def test_the_four_networks_count_what_their_layers_hold(adjacency):
    """Default 61,729; the same without graph layers 2,049; recurrent 124,482 and 64,802 without."""
    counts = {(t, s): _built(adjacency, t, s).count_parameters()
              for t in ("last_cycle", "lstm") for s in ("gcn", "none")}
    graph_layers = 1 * 32 + 32 * 64 + 64 * 32             # 4,128, no offsets (A-12)
    assert counts[("last_cycle", "gcn")] == 61_729
    # FC-1 shrinks from 448 x 128 + 128 to 14 x 128 + 128, and the graph layers go.
    assert counts[("last_cycle", "none")] == 14 * 128 + 128 + 128 * 1 + 1 == 2_049
    assert counts[("last_cycle", "gcn")] - counts[("last_cycle", "none")] == graph_layers + (448 - 14) * 128
    assert counts[("lstm", "gcn")] == 124_482
    assert counts[("lstm", "none")] == 2_049 + 62_753 == 64_802


def test_without_graph_layers_the_graph_is_kept_but_never_read(adjacency):
    """The graph stays in the saved weights, so a stored network loads unchanged, and the prediction ignores it."""
    model = _built(adjacency, "lstm", "none").eval()
    assert "adjacency" in model.state_dict()
    window = torch.linspace(-1, 1, 14 * 30).reshape(1, 14, 30)
    with torch.no_grad():
        before = model(window)
        model.adjacency.fill_(float("nan"))
        after = model(window)
    assert torch.isfinite(after).all() and torch.equal(before, after)


def test_the_spatial_setting_is_checked_and_its_default_changes_no_record():
    """Every stored name and digest was made before the setting existed, so the default must not move them."""
    from thesis_gnn_rul_xai.models import ModelConfig

    with pytest.raises(ValueError, match=r"spatial must be one of \['gcn', 'none'\], got 'gat'"):
        ModelConfig(spatial="gat")
    with pytest.raises(ValueError, match="there are none"):
        ModelConfig(spatial="none", gcn_hidden=(16,))
    with pytest.raises(ValueError, match="there are none"):
        ModelConfig(spatial="none", gcn_bias=True)
    with pytest.raises(ValueError, match="no graph layers to give one to"):
        ESTGNN(normalize_adjacency(build_physics_adjacency()[0]), spatial="none", gcn_bias=True)

    assert "spatial" not in ModelConfig().to_dict()
    assert ModelConfig(spatial="gcn") == ModelConfig()
    assert ModelConfig().digest() == "4e49f2"
    assert ModelConfig(temporal="lstm").digest() == "a4cd38"
    removed = ModelConfig(spatial="none")
    assert removed.to_dict()["spatial"] == "none" and removed.has_graph_layers is False
    assert removed.digest() not in {"4e49f2", ModelConfig(temporal="lstm", spatial="none").digest()}


def test_a_network_without_graph_layers_saves_and_loads_back(adjacency, tmp_path):
    """model_config.yaml says spatial: none, and the weights go back into the network it rebuilds."""
    from thesis_gnn_rul_xai.models import ModelConfig, build_model, load_model_config

    saved = ModelConfig(temporal="lstm", spatial="none").save(tmp_path / "model_config.yaml")
    assert "spatial: none" in saved.read_text(encoding="utf-8")
    config = load_model_config(saved)
    assert config == ModelConfig(temporal="lstm", spatial="none")
    torch.manual_seed(1)
    trained = build_model(adjacency, config, window_size=30, rul_max=125)
    torch.manual_seed(2)
    rebuilt = build_model(adjacency, config, window_size=30, rul_max=125)
    rebuilt.load_state_dict(trained.state_dict())
    window = torch.zeros(2, 14, 30)
    assert torch.equal(rebuilt(window), trained(window))


def test_raw_cycles_reaching_the_dense_layer_are_refused_unless_asked_for(adjacency):
    """With no graph layers 30 raw cycles per sensor land in FC-1, a temporal encoder not named one."""
    with pytest.raises(ValueError, match=r"first fully connected weight matrix becomes \(420, 128\)"):
        ESTGNN(adjacency, temporal=RawWindow(30), spatial="none")
    allowed = ESTGNN(adjacency, temporal=RawWindow(30), spatial="none",
                     allow_time_in_the_graph_layer=True)
    assert allowed.time_folded_into_graph_layer is True
    assert tuple(allowed.fc1.weight.shape) == (128, 420)


def test_the_default_networks_start_from_the_weights_they_started_from_before_spatial_existed(adjacency):
    """Building the graph layers must draw the same random numbers as before; a fact about PyTorch 2.15 only.

    The digests were taken on 2026-11-13 before spatial was added, under
    PyTorch 2.15.0, on the row-rescaled Table 3 graph after torch.manual_seed(42).
    Another PyTorch version may draw other numbers, so it is skipped there; the
    second half, gcn spelled out equals gcn left out, holds on any version.
    """
    import hashlib

    from thesis_gnn_rul_xai.models import ModelConfig, build_model

    graph = normalize_adjacency(build_physics_adjacency()[0], method="row")

    def digest(config):
        torch.manual_seed(42)
        model = build_model(graph, config, window_size=30, rul_max=125)
        h = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            h.update(name.encode())
            h.update(tensor.detach().cpu().numpy().astype(np.float32).tobytes())
        return h.hexdigest()

    for temporal in ("last_cycle", "lstm"):
        assert digest(ModelConfig(temporal=temporal, spatial="gcn")) == digest(ModelConfig(temporal=temporal))
    if not torch.__version__.startswith("2.15."):
        pytest.skip(f"digests measured under PyTorch 2.15.0, this is {torch.__version__}")
    assert digest(ModelConfig()) == "50960d99e96174c2ea4672c99b8219589899245f8cefc89fb50e6f3d3b2be14b"
    assert digest(ModelConfig(temporal="lstm")) == "2bd20188f1d791ad22e2d0a7bc9b8a75dd14bf39b7806b8b64cef75920262011"
