"""Tests for the walk relevances of explain/walks.py (GNN-LRP, not in the paper).

A walk relevance is a share of the score the network computes before its
sigmoid. These tests check the properties the split is defined by, rather than
whether the walks it names look plausible:

- **conservation**: the walks plus the layers' constant must give back the
  score, to rounding noise, for every temporal stage and with or without the
  graph layers' constants;
- **agreement with the gradient**: each sensor's total must equal what
  PyTorch's own differentiation gives, reading times its derivative, which is
  computed by a path that shares no code with the walks;
- **structure**: a connection the sensor graph does not have must carry exactly
  zero, at every layer;
- **the network as it is**: a network that computes anything the file does not
  model must be refused rather than explained.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.explain import (
    MAX_WALKS_PER_EXAMPLE,
    conservation_error,
    crossing_relevance,
    indistinguishable_sensors,
    node_relevance,
    step_relevance,
    top_walks,
    walk_relevance,
)
from thesis_gnn_rul_xai.models import ESTGNN, build_temporal_stage


@pytest.fixture(scope="module")
def prepared(raw_dir):
    return run_preprocessing(
        PreprocessingConfig(window_size=20, validation_fraction=0.25, seed=7),
        data_dir=raw_dir,
    )


@pytest.fixture(scope="module")
def model(prepared):
    """An untrained network. Conservation is arithmetic, so it holds regardless of fit."""
    torch.manual_seed(0)
    return ESTGNN(
        prepared.adjacency_normalized,
        window_size=prepared.train.features.shape[2],
        rul_max=prepared.config.rul_max,
    )


@pytest.fixture(scope="module")
def windows(prepared):
    return prepared.test.features[:12]


def _network(prepared, stage: str, *, bias: bool) -> ESTGNN:
    torch.manual_seed(3)
    width = prepared.train.features.shape[2]
    return ESTGNN(
        prepared.adjacency_normalized,
        temporal=build_temporal_stage(stage, window_size=width, n_nodes=14),
        rul_max=prepared.config.rul_max,
        gcn_bias=bias,
        allow_time_in_the_graph_layer=True,
    )


# --- conservation -------------------------------------------------------


def test_walks_and_constant_give_back_the_score(model, windows):
    result = walk_relevance(model, windows)
    assert conservation_error(result) < 1e-9


@pytest.mark.parametrize(
    "stage", ["last_cycle", "mean_window", "window_summary", "raw_window", "lstm", "gru"]
)
@pytest.mark.parametrize("bias", [False, True])
def test_conservation_holds_for_every_stage_with_and_without_graph_constants(
    prepared, windows, stage, bias
):
    result = walk_relevance(_network(prepared, stage, bias=bias), windows[:4])
    assert conservation_error(result) < 1e-9


def test_the_constant_is_the_dense_layers_own_when_the_graph_layers_have_none(model, windows):
    """Without graph-layer constants, only fc1's and fc2's added numbers remain."""
    result = walk_relevance(model, windows)
    with torch.no_grad():
        nodes = model.temporal(torch.from_numpy(windows.astype(np.float32)))
        hidden = nodes
        for layer in model.graph_layers:
            hidden = torch.relu(layer(hidden, model.adjacency))
        before = model.fc1(torch.flatten(hidden, start_dim=1))
        on = (before > 0).double()
        expected = (on * model.fc1.bias.double()) @ model.fc2.weight.double()[0]
        expected = expected + model.fc2.bias.double()[0]
    np.testing.assert_allclose(result.constant, expected.numpy(), atol=1e-6)


def test_the_score_turns_into_the_networks_own_prediction(model, windows):
    result = walk_relevance(model, windows)
    with torch.no_grad():
        own = model.predict_cycles(torch.from_numpy(windows.astype(np.float32))).numpy()
    walks = result.relevance.reshape(len(windows), -1).sum(axis=1)
    rebuilt = model.rul_max / (1.0 + np.exp(-(walks + result.constant)))
    np.testing.assert_allclose(rebuilt, own, atol=1e-4)
    np.testing.assert_allclose(result.predicted, own, atol=1e-6)


# --- agreement with an independent calculation --------------------------


def test_each_sensors_total_is_its_reading_times_its_derivative(model, windows):
    """PyTorch differentiates the score with respect to the numbers entering the
    first graph layer; multiplied by those numbers, that must be each sensor's
    total over the walks that start at it."""
    result = walk_relevance(model, windows)
    double = ESTGNN(
        model.adjacency.numpy(),
        window_size=model.temporal.window_size,
        rul_max=model.rul_max,
    ).double()
    double.load_state_dict(model.state_dict())
    nodes = double.temporal(torch.from_numpy(windows.astype(np.float64))).detach()
    nodes.requires_grad_(True)
    hidden = nodes
    for layer in double.graph_layers:
        hidden = torch.relu(layer(hidden, double.adjacency))
    score = double.fc2(torch.relu(double.fc1(torch.flatten(hidden, start_dim=1))))
    score.sum().backward()
    by_gradient = (nodes.grad * nodes).sum(dim=2).detach().numpy()
    np.testing.assert_allclose(node_relevance(result), by_gradient, atol=1e-9)


def _double_copy(model):
    import copy

    return copy.deepcopy(model).double()


def test_each_ending_sensors_total_is_its_last_layer_numbers_times_their_derivative(
    model, windows
):
    """The same check at the other end of the walks: added up by where they end,
    the shares must equal the last graph layer's numbers times how fast the score
    changes with each."""
    result = walk_relevance(model, windows)
    net = _double_copy(model)
    hidden = net.temporal(torch.from_numpy(windows.astype(np.float64)))
    for layer in net.graph_layers:
        hidden = torch.relu(layer(hidden, net.adjacency))
    last = hidden.detach().requires_grad_(True)
    score = net.fc2(torch.relu(net.fc1(torch.flatten(last, start_dim=1))))
    score.sum().backward()
    by_gradient = (last.grad * last).sum(dim=2).detach().numpy()
    by_end = result.relevance.sum(axis=(1, 2, 3))
    np.testing.assert_allclose(by_end, by_gradient, atol=1e-9)


def test_what_passes_along_a_connection_is_its_graph_entry_times_the_scores_derivative(
    model, windows
):
    """Added over the three layers, what passed from T24 to T30 equals the graph's
    entry for that pair times how fast the score changes when that one entry
    changes, the same entry being used by all three layers."""
    result = walk_relevance(model, windows[0])
    net = _double_copy(model)
    adjacency = net.adjacency.clone().requires_grad_(True)
    hidden = net.temporal(torch.from_numpy(windows[:1].astype(np.float64)))
    for layer in net.graph_layers:
        hidden = torch.relu(adjacency @ hidden @ layer.weight)
    score = net.fc2(torch.relu(net.fc1(torch.flatten(hidden, start_dim=1))))
    score.sum().backward()
    by_gradient = (adjacency.grad * adjacency).detach().numpy().T
    passed = sum(step_relevance(result, step)[0] for step in range(1, 4))
    np.testing.assert_allclose(passed, by_gradient, atol=1e-9)


def test_scaling_the_graph_input_leaves_every_graph_relu_as_it_was(model, windows):
    """The graph layers add no constant, so a window's numbers scaled by 0.37 give
    every graph layer's numbers scaled by 0.37: no ReLU there switches. Only fc1's
    can, which is all that makes the split depend on this one window."""
    net = _double_copy(model)
    nodes = net.temporal(torch.from_numpy(windows.astype(np.float64))).detach()
    full, scaled = nodes, 0.37 * nodes
    for layer in net.graph_layers:
        full = torch.relu(layer(full, net.adjacency))
        scaled = torch.relu(layer(scaled, net.adjacency))
        assert torch.equal(full > 0, scaled > 0)
        np.testing.assert_allclose(scaled.detach().numpy(), 0.37 * full.detach().numpy(), atol=1e-12)


def test_the_score_at_zero_is_the_network_on_a_window_of_training_averages(model):
    """For the last-cycle stage a window of zeros hands the graph layers zeros, and
    under z-score scaling zero is each sensor's training average."""
    result = walk_relevance(model, np.zeros((1, 14, model.temporal.window_size), np.float32))
    with torch.no_grad():
        own = float(model(torch.zeros(1, 14, model.temporal.window_size)))
    assert 1.0 / (1.0 + np.exp(-result.score_at_zero)) == pytest.approx(own, abs=1e-6)
    assert float(result.score[0]) == pytest.approx(result.score_at_zero, abs=1e-12)


def test_the_absolute_total_is_never_below_the_size_of_the_walks_sum(model, windows):
    table = crossing_relevance(walk_relevance(model, windows))
    assert np.all(table["absolute_total"] >= np.abs(table["stays"] + table["moves"]) - 1e-12)


def test_pooled_walks_add_up_the_ones_the_graph_cannot_tell_apart(model, prepared, windows):
    """Pooling renames every position after the first by its group and adds the
    shares, so the total over all walks is unchanged and fewer walks remain."""
    result = walk_relevance(model, windows[:2])
    groups = indistinguishable_sensors(model, prepared.sensor_symbols)
    raw = top_walks(result, prepared.sensor_symbols, count=40_000)
    pooled = top_walks(result, prepared.sensor_symbols, count=40_000, pool=groups)
    assert len(pooled) < len(raw) == 14**4
    assert pooled["mean"].sum() == pytest.approx(raw["mean"].sum(), abs=1e-12)
    assert any(walk.startswith("P30 -> P30/Ps30") for walk in pooled["walk"])
    assert not any(walk.startswith("P30/Ps30") for walk in pooled["walk"])


def _walk_by_masked_gradient(model, window, walk, gammas) -> float:
    """One walk the authors' way: rewrite each graph layer, mask it to the walk,
    differentiate once, and multiply by the starting numbers (Schnake et al.,
    Section 3.2.1)."""
    import copy

    net = copy.deepcopy(model).double()
    adjacency = net.adjacency
    n_nodes = adjacency.shape[0]
    nodes = net.temporal(torch.from_numpy(window[None].astype(np.float64))).detach()
    nodes.requires_grad_(True)
    hidden = nodes
    for step, (layer, gamma) in enumerate(zip(net.graph_layers, gammas, strict=True)):
        mixed = adjacency @ hidden
        weight, bias = layer.weight, layer.bias
        before = mixed @ weight + (bias if bias is not None else 0.0)
        ruled = mixed @ (weight + gamma * weight.clamp(min=0))
        if bias is not None:
            ruled = ruled + bias + gamma * bias.clamp(min=0)
        after = torch.relu(before)
        on = after > 0
        ratio = torch.where(on, after / torch.where(on, ruled, torch.ones_like(ruled)), 0.0)
        passed = ruled * ratio.detach()
        mask = torch.zeros(1, n_nodes, 1, dtype=torch.float64)
        mask[0, walk[step + 1]] = 1.0
        hidden = passed * mask + passed.detach() * (1.0 - mask)
    score = net.fc2(torch.relu(net.fc1(torch.flatten(hidden, start_dim=1))))
    score.sum().backward()
    return float((nodes.grad[0, walk[0]] * nodes[0, walk[0]]).sum().detach())


@pytest.mark.parametrize("gamma", [(0.0, 0.0, 0.0), (0.0, 0.5, 0.25)])
@pytest.mark.parametrize("bias", [False, True])
def test_every_walk_matches_the_authors_masked_gradient(prepared, windows, gamma, bias):
    """Thirty walks along real connections, each computed both ways."""
    network = _network(prepared, "last_cycle", bias=bias)
    result = walk_relevance(network, windows[0], gamma=gamma)
    assert conservation_error(result) < 1e-9
    assert result.gamma == gamma
    allowed = network.adjacency.numpy().T != 0
    walks = [
        (a, b, c, d)
        for a in range(14) for b in range(14) for c in range(14) for d in range(14)
        if allowed[a, b] and allowed[b, c] and allowed[c, d]
    ]
    assert len(walks) == 5178
    for walk in walks[::173]:
        expected = _walk_by_masked_gradient(network, windows[0], walk, gamma)
        assert result.relevance[(0, *walk)] == pytest.approx(expected, abs=1e-12)


def test_a_positive_gamma_is_given_to_every_layer_but_the_first(model, windows):
    result = walk_relevance(model, windows[:2], gamma=0.5)
    assert result.gamma == (0.0, 0.5, 0.5)
    assert conservation_error(result) < 1e-9


def test_gamma_at_the_first_layer_is_refused(model, windows):
    with pytest.raises(ValueError, match="first graph layer"):
        walk_relevance(model, windows[:1], gamma=(0.5, 0.5, 0.5))


def test_a_negative_or_misshapen_gamma_is_refused(model, windows):
    with pytest.raises(ValueError, match="negative"):
        walk_relevance(model, windows[:1], gamma=(0.0, -1.0, 0.0))
    with pytest.raises(ValueError, match="one value per graph layer"):
        walk_relevance(model, windows[:1], gamma=(0.0, 0.5))


# --- structure ----------------------------------------------------------


def test_a_connection_the_graph_lacks_carries_exactly_zero(model, windows):
    result = walk_relevance(model, windows)
    missing = model.adjacency.numpy().T == 0
    assert missing.sum() == 14 * 14 - 96
    for step in range(1, result.n_steps + 1):
        assert np.all(step_relevance(result, step)[:, missing] == 0.0)


def test_the_papers_graph_has_5178_walks_of_three_steps(model, windows):
    """41 edges both ways plus 14 self-connections: 96 entries that are not zero."""
    assert int((model.adjacency.numpy() != 0).sum()) == 96
    result = walk_relevance(model, windows[:1])
    assert int((result.relevance[0] != 0).sum()) <= 5178
    allowed = (model.adjacency.numpy().T != 0).astype(np.int64)
    assert int(np.einsum("ab,bc,cd->", allowed, allowed, allowed)) == 5178


def test_every_layer_passes_on_the_same_total(model, windows):
    result = walk_relevance(model, windows)
    total = node_relevance(result).sum(axis=1)
    for step in range(1, result.n_steps + 1):
        np.testing.assert_allclose(step_relevance(result, step).sum(axis=(1, 2)), total, atol=1e-12)


def test_staying_and_moving_add_up_to_every_walk(model, windows):
    result = walk_relevance(model, windows)
    table = crossing_relevance(result)
    np.testing.assert_allclose(
        table["stays"] + table["moves"] + table["constant"], table["score"], atol=1e-9
    )


# --- sensors the graph cannot tell apart --------------------------------


def test_the_papers_graph_has_three_groups_of_indistinguishable_sensors(model, prepared):
    """Identical rows of the rescaled graph, under row rescaling as under symmetric."""
    assert indistinguishable_sensors(model, prepared.sensor_symbols) == [
        ("P15", "Nf", "NRf", "BPR"),
        ("P30", "Ps30"),
        ("Nc", "W31"),
    ]


def test_walks_differing_only_by_an_indistinguishable_sensor_in_the_middle_are_equal(
    model, prepared, windows
):
    """After the first graph layer P30 and Ps30 hold the same numbers, so a walk
    through one carries exactly what the same walk through the other carries."""
    result = walk_relevance(model, windows[:3])
    p30, ps30 = (list(prepared.sensor_symbols).index(s) for s in ("P30", "Ps30"))
    np.testing.assert_allclose(
        result.relevance[:, :, p30], result.relevance[:, :, ps30], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        result.relevance[:, :, :, p30], result.relevance[:, :, :, ps30], rtol=0, atol=1e-15
    )


def test_training_never_changes_the_difference_between_their_fc1_weights(prepared):
    """fc1 sees the same numbers at P30 and Ps30, so both of their blocks of fc1
    weights receive the same gradient and Adam moves them by the same amount:
    their difference stays what the random start made it. That is why where a
    walk ends within such a group says nothing about the engine."""
    torch.manual_seed(5)
    net = ESTGNN(
        prepared.adjacency_normalized,
        window_size=prepared.train.features.shape[2],
        rul_max=prepared.config.rul_max,
    )
    p30, ps30 = (list(prepared.sensor_symbols).index(s) for s in ("P30", "Ps30"))

    def blocks():
        weights = net.fc1.weight.detach().reshape(net.fc1.out_features, 14, -1)
        return weights[:, p30].clone(), weights[:, ps30].clone()

    first_p30, first_ps30 = blocks()
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)
    inputs = torch.from_numpy(prepared.train.features[:256].astype(np.float32))
    targets = torch.from_numpy(prepared.train.targets[:256].astype(np.float32)) / 125.0
    for _ in range(20):
        optimiser.zero_grad()
        loss = torch.mean((net(inputs) - targets) ** 2)
        loss.backward()
        optimiser.step()
    last_p30, last_ps30 = blocks()

    assert float((last_p30 - first_p30).abs().max()) > 1e-3
    np.testing.assert_allclose(
        (last_p30 - last_ps30).numpy(), (first_p30 - first_ps30).numpy(), rtol=0, atol=1e-6
    )


def test_indistinguishable_sensors_rejects_a_mismatched_label_list(model):
    with pytest.raises(ValueError, match="symbol"):
        indistinguishable_sensors(model, ["T24"])


# --- shapes and refusals ------------------------------------------------


def test_one_window_gives_one_row_of_every_array(model, windows):
    result = walk_relevance(model, windows[0])
    assert result.relevance.shape == (1, 14, 14, 14, 14)
    assert result.constant.shape == result.score.shape == result.predicted.shape == (1,)


def test_the_batch_size_does_not_change_the_answer(model, windows):
    """Only the order of additions changes, which moves the last few digits."""
    small = walk_relevance(model, windows, batch_size=5)
    large = walk_relevance(model, windows, batch_size=64)
    np.testing.assert_allclose(small.relevance, large.relevance, rtol=0, atol=1e-15)


def test_a_network_computing_something_else_is_refused(prepared, windows):
    class Shifted(ESTGNN):
        def forward(self, features):
            return super().forward(features) * 0.9

    torch.manual_seed(0)
    shifted = Shifted(
        prepared.adjacency_normalized,
        window_size=prepared.train.features.shape[2],
        rul_max=prepared.config.rul_max,
    )
    with pytest.raises(RuntimeError, match="does not model"):
        walk_relevance(shifted, windows[:2])


def test_too_many_walks_is_refused_rather_than_sampled(prepared):
    torch.manual_seed(0)
    deep = ESTGNN(prepared.adjacency_normalized, window_size=1, gcn_hidden=(4, 4, 4, 4, 4))
    assert MAX_WALKS_PER_EXAMPLE < 14**6
    with pytest.raises(ValueError, match="does not sample"):
        walk_relevance(deep, np.zeros((14, 1), dtype=np.float32))


def test_the_top_walks_are_named_and_ordered_by_size(model, prepared, windows):
    result = walk_relevance(model, windows)
    table = top_walks(result, prepared.sensor_symbols, count=5)
    assert list(table.columns) == ["walk", "mean", "mean_abs"]
    assert len(table) == 5
    assert table["mean_abs"].is_monotonic_decreasing
    assert all(walk.count(" -> ") == 3 for walk in table["walk"])


def test_top_walks_rejects_a_mismatched_label_list(model, windows):
    result = walk_relevance(model, windows[:1])
    with pytest.raises(ValueError, match="symbol"):
        top_walks(result, ["T24"])


def test_a_step_outside_the_network_is_rejected(model, windows):
    result = walk_relevance(model, windows[:1])
    with pytest.raises(ValueError, match="step"):
        step_relevance(result, 4)


# --- shares over the cycles of a window ----------------------------------


@pytest.mark.parametrize("stage", ["last_cycle", "mean_window", "window_summary", "raw_window"])
def test_cycle_shares_add_up_to_each_sensors_walk_total(prepared, windows, stage):
    from thesis_gnn_rul_xai.explain import cycle_relevance

    network = _network(prepared, stage, bias=False)
    shares = cycle_relevance(network, windows[:4])
    assert shares.shape == windows[:4].shape
    np.testing.assert_allclose(
        shares.sum(axis=2), node_relevance(walk_relevance(network, windows[:4])), atol=1e-9
    )


def test_the_last_cycle_stage_puts_every_share_on_the_last_cycle(model, windows):
    from thesis_gnn_rul_xai.explain import cycle_relevance

    shares = cycle_relevance(model, windows[:3])
    assert np.all(shares[:, :, :-1] == 0.0)


def test_cycle_shares_are_refused_for_a_stage_that_learns(prepared, windows):
    from thesis_gnn_rul_xai.explain import cycle_relevance

    with pytest.raises(ValueError, match="learns its own numbers"):
        cycle_relevance(_network(prepared, "lstm", bias=False), windows[:1])


# --- removing a connection ------------------------------------------------


def test_removal_gives_one_column_per_connection_of_the_raw_graph(model, prepared, windows):
    from thesis_gnn_rul_xai.explain import connection_removal

    table = connection_removal(
        model, windows, prepared.adjacency_raw, prepared.sensor_symbols,
        method=prepared.config.adjacency_normalization,
    )
    assert table.shape == (len(windows), 41)
    assert "T24-T30" in table.columns and "T30-T24" not in table.columns


def test_removal_matches_a_network_built_on_the_graph_without_that_connection(
    model, prepared, windows
):
    from thesis_gnn_rul_xai.explain import connection_removal
    from thesis_gnn_rul_xai.graph.adjacency import normalize_adjacency

    method = prepared.config.adjacency_normalization
    table = connection_removal(
        model, windows, prepared.adjacency_raw, prepared.sensor_symbols, method=method
    )
    symbols = list(prepared.sensor_symbols)
    i, j = symbols.index("T30"), symbols.index("Ps30")
    removed = prepared.adjacency_raw.copy()
    removed[i, j] = removed[j, i] = 0
    other = ESTGNN(
        normalize_adjacency(removed, method=method),
        window_size=model.temporal.window_size,
        rul_max=model.rul_max,
    )
    other.load_state_dict({k: v for k, v in model.state_dict().items() if k != "adjacency"}, strict=False)
    inputs = torch.from_numpy(windows.astype(np.float32))
    with torch.no_grad():
        expected = (model.predict_cycles(inputs) - other.predict_cycles(inputs)).numpy()
    np.testing.assert_allclose(table["T30-Ps30"].to_numpy(), expected, atol=1e-4)
    assert torch.equal(model.adjacency, torch.from_numpy(prepared.adjacency_normalized))


def test_removal_refuses_a_raw_graph_the_network_was_not_trained_on(model, prepared, windows):
    from thesis_gnn_rul_xai.explain import connection_removal

    other = "symmetric" if prepared.config.adjacency_normalization == "row" else "row"
    with pytest.raises(ValueError, match="not the graph this network"):
        connection_removal(
            model, windows[:1], prepared.adjacency_raw, prepared.sensor_symbols, method=other
        )


# --- pooled groups --------------------------------------------------------


def test_pooled_staying_counts_a_group_as_one_sensor(model, prepared, windows):
    """P30 -> P30 -> P30 -> Ps30 stays once P30 and Ps30 are one group, and the
    walks' total, staying plus moving, does not change."""
    result = walk_relevance(model, windows[:3])
    groups = indistinguishable_sensors(model, prepared.sensor_symbols)
    plain = crossing_relevance(result)
    pooled = crossing_relevance(result, prepared.sensor_symbols, pool=groups)
    np.testing.assert_allclose(
        pooled["stays"] + pooled["moves"], plain["stays"] + plain["moves"], atol=1e-12
    )
    assert np.all(pooled["absolute_stays"] >= plain["absolute_stays"] - 1e-15)
    names = list(prepared.sensor_symbols)
    p30, ps30 = names.index("P30"), names.index("Ps30")
    extra = result.relevance[:, p30, p30, p30, ps30]
    assert np.all(np.abs(extra) > 0)


def test_connections_keep_their_total_and_split_a_groups_last_arrivals_equally(
    model, prepared, windows
):
    from thesis_gnn_rul_xai.explain import connection_relevance

    result = walk_relevance(model, windows[:3])
    groups = indistinguishable_sensors(model, prepared.sensor_symbols)
    plain = connection_relevance(result, prepared.sensor_symbols)
    pooled = connection_relevance(result, prepared.sensor_symbols, pool=groups)
    np.testing.assert_allclose(
        plain, sum(step_relevance(result, step) for step in range(1, 4)), atol=1e-15
    )
    total = result.relevance.reshape(3, -1).sum(axis=1)
    np.testing.assert_allclose(pooled.sum(axis=(1, 2)), 3 * total, atol=1e-12)
    names = list(prepared.sensor_symbols)
    p30, ps30 = names.index("P30"), names.index("Ps30")
    np.testing.assert_allclose(pooled[:, :, p30], pooled[:, :, ps30], atol=1e-15)
    assert not np.allclose(plain[:, :, p30], plain[:, :, ps30])
