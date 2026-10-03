"""Tests for the exact Shapley values of Section 2.4.

Attribution methods are unusually easy to get subtly wrong, because a wrong
answer still looks like a plausible ranking of sensors. These tests therefore
check the mathematical properties a Shapley value is *defined* by, rather than
checking that the output looks sensible:

- **efficiency**: the contributions plus the baseline must equal the prediction
  exactly (the paper's Equation 14);
- **dummy**: a sensor that cannot change the prediction must receive exactly
  zero;
- **symmetry**: two sensors that are interchangeable must receive equal values.

If those three hold, the numbers are Shapley values. If any fails, they are not,
whatever else they may be.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.explain import (
    MAX_EXACT_NODES,
    coalition_weights,
    dynamic_shap,
    efficiency_error,
    mean_window_baseline,
    shap_summary,
    shap_values,
)
from thesis_gnn_rul_xai.graph.adjacency import (
    build_fully_connected_adjacency,
    normalize_adjacency,
)
from thesis_gnn_rul_xai.models import ESTGNN


@pytest.fixture(scope="module")
def prepared(raw_dir):
    return run_preprocessing(
        PreprocessingConfig(window_size=20, validation_fraction=0.25, seed=7),
        data_dir=raw_dir,
    )


@pytest.fixture(scope="module")
def model(prepared):
    """An untrained network. The Shapley properties hold regardless of fit."""
    torch.manual_seed(0)
    return ESTGNN(
        prepared.adjacency_normalized,
        window_size=prepared.train.features.shape[2],
        rul_max=prepared.config.rul_max,
    )


@pytest.fixture(scope="module")
def baseline(prepared):
    return mean_window_baseline(prepared.train)


def _tiny_model(n_nodes: int = 3, window: int = 4) -> ESTGNN:
    """A small complete-graph network, so 2**n coalitions is trivial."""
    torch.manual_seed(1)
    dense, _ = build_fully_connected_adjacency(n_nodes)
    return ESTGNN(normalize_adjacency(dense), window_size=window, rul_max=125)


# --- the weighting factor of Equation 15 --------------------------------


def test_coalition_weights_account_for_every_ordering_once():
    """Summed over all subsets of the other sensors, the weights give 1."""
    for n in (2, 3, 14):
        weights = coalition_weights(n)
        total = sum(comb(n - 1, s) * weights[s] for s in range(n))
        assert total == pytest.approx(1.0)


def test_coalition_weights_are_symmetric():
    """A subset of size s and one of size n-1-s carry the same weight."""
    weights = coalition_weights(14)
    np.testing.assert_allclose(weights, weights[::-1], rtol=1e-12)


# --- the three defining properties --------------------------------------


def test_efficiency_holds_to_machine_precision(model, prepared, baseline):
    """The paper's Equation 14: the parts must add up to the whole.

    This is the property that makes an attribution trustworthy. A method that
    fails it is assigning credit that does not correspond to anything the model
    did.
    """
    features = prepared.test.features[:4]
    values = shap_values(model, features, baseline)
    assert efficiency_error(model, features, baseline, values) < 1e-4


def test_a_sensor_that_cannot_matter_receives_exactly_zero(model, prepared):
    """The dummy property.

    If a sensor's readings are identical to the stand-in used for its absence,
    then including it changes nothing, in every coalition. Its Shapley value
    must be exactly zero, not merely small.
    """
    baseline = mean_window_baseline(prepared.train)
    features = prepared.test.features[0].copy()
    silent = 3                                   # P15's position in the graph
    features[silent] = baseline[silent]          # make it carry no information

    values = shap_values(model, features, baseline)
    assert values[silent] == pytest.approx(0.0, abs=1e-9)
    assert np.abs(values).sum() > 0               # the others still contribute


def test_interchangeable_sensors_receive_equal_values():
    """The symmetry property.

    On a complete graph every node occupies the same position, so two nodes
    given identical readings must be credited identically.
    """
    tiny = _tiny_model(n_nodes=3, window=4)
    baseline = np.zeros((3, 4), dtype=np.float32)
    features = np.array(
        [[0.9, 0.8, 0.7, 0.6],
         [0.9, 0.8, 0.7, 0.6],      # identical to the first
         [0.1, 0.2, 0.3, 0.4]],
        dtype=np.float32,
    )
    values = shap_values(tiny, features, baseline)
    assert values[0] == pytest.approx(values[1], abs=1e-9)


# --- occlusion, the cheaper method that is not in the paper --------------


def test_occlusion_is_the_change_from_replacing_one_sensor(model, prepared, baseline):
    """Each value is the full prediction minus the prediction with that one
    sensor's window swapped for the stand-in, and nothing else."""
    from thesis_gnn_rul_xai.explain import occlusion_values

    window = prepared.test.features[0]
    values = occlusion_values(model, window, baseline)
    with torch.no_grad():
        full = float(model.predict_cycles(torch.from_numpy(window[None]))[0])
        for sensor in (0, 3, 13):
            replaced = window.copy()
            replaced[sensor] = baseline[sensor]
            without = float(model.predict_cycles(torch.from_numpy(replaced[None]))[0])
            assert values[sensor] == pytest.approx(full - without, abs=1e-4)


def test_occlusion_gives_one_row_per_window_whatever_the_batch(model, prepared, baseline):
    from thesis_gnn_rul_xai.explain import occlusion_values

    windows = prepared.test.features[:7]
    small = occlusion_values(model, windows, baseline, batch_size=3)
    large = occlusion_values(model, windows, baseline, batch_size=512)
    assert small.shape == (7, 14)
    np.testing.assert_allclose(small, large, rtol=1e-6, atol=1e-6)


# --- the figure of contributions over an engine's life -------------------


def test_the_figure_puts_the_largest_average_contribution_first(model, prepared, baseline):
    import matplotlib

    matplotlib.use("Agg")
    from thesis_gnn_rul_xai.explain import plot_contributions_over_time

    engine = int(prepared.validation.units[0])
    table = dynamic_shap(model, prepared.validation, baseline, unit=engine,
                         symbols=prepared.sensor_symbols)
    sensors = list(prepared.sensor_symbols)
    figure = plot_contributions_over_time(table, sensors)

    panels = [axis for axis in figure.axes[1:] if axis.axison]
    titles = [axis.get_title(loc="left").split()[0] for axis in panels]
    expected = table[sensors].abs().mean().sort_values(ascending=False, kind="stable")
    assert titles == expected.index.tolist()
    assert len({axis.get_ylim() for axis in panels}) == 1   # one scale for every panel

    with pytest.raises(ValueError, match="no column"):
        plot_contributions_over_time(table.drop(columns="rul"), sensors)
    matplotlib.pyplot.close("all")


# --- shapes, inputs and guards ------------------------------------------


def test_a_single_window_returns_one_value_per_sensor(model, prepared, baseline):
    values = shap_values(model, prepared.test.features[0], baseline)
    assert values.shape == (14,)


def test_several_windows_return_a_matrix(model, prepared, baseline):
    values = shap_values(model, prepared.test.features[:5], baseline)
    assert values.shape == (5, 14)


def test_the_batch_size_does_not_change_the_answer(model, prepared, baseline):
    """It controls memory only, so the numbers must be identical."""
    window = prepared.test.features[0]
    small = shap_values(model, window, baseline, batch_size=256)
    large = shap_values(model, window, baseline, batch_size=16384)
    np.testing.assert_allclose(small, large, rtol=1e-6, atol=1e-9)


def test_a_mismatched_baseline_is_rejected(model, prepared):
    with pytest.raises(ValueError, match="baseline must be shaped"):
        shap_values(model, prepared.test.features[0], np.zeros((14, 5)))


def test_too_many_nodes_is_refused_rather_than_approximated():
    """This module computes Equation 15 exactly or not at all."""
    n = MAX_EXACT_NODES + 1
    dense, _ = build_fully_connected_adjacency(n)
    big = ESTGNN(normalize_adjacency(dense), window_size=2, rul_max=125)
    with pytest.raises(ValueError, match="deliberately does not provide"):
        shap_values(big, np.zeros((n, 2), dtype=np.float32), np.zeros((n, 2), dtype=np.float32))


def test_the_baseline_must_come_from_a_non_empty_dataset(prepared):
    empty = prepared.train.subset_by_units([])
    with pytest.raises(ValueError, match="empty dataset"):
        mean_window_baseline(empty)


# --- the dynamic part, Equation 16 --------------------------------------


def test_dynamic_shap_covers_one_engine_in_cycle_order(model, prepared, baseline):
    """Equation 16 is Equation 15 repeated at every time step."""
    unit = int(prepared.validation.units[0])
    table = dynamic_shap(
        model, prepared.validation, baseline,
        unit=unit, symbols=prepared.sensor_symbols,
    )
    expected = int((prepared.validation.units == unit).sum())
    assert len(table) == expected
    assert table["end_cycle"].is_monotonic_increasing
    assert list(table.columns[:4]) == ["end_cycle", "rul", "predicted", "baseline"]
    assert list(table.columns[4:]) == list(prepared.sensor_symbols)


def test_dynamic_shap_rows_each_satisfy_efficiency(model, prepared, baseline):
    unit = int(prepared.validation.units[0])
    table = dynamic_shap(
        model, prepared.validation, baseline,
        unit=unit, symbols=prepared.sensor_symbols,
    )
    contributions = table[list(prepared.sensor_symbols)].to_numpy().sum(axis=1)
    gap = table["predicted"].to_numpy() - table["baseline"].to_numpy()
    np.testing.assert_allclose(contributions, gap, atol=1e-4)


def test_an_unknown_engine_is_reported_clearly(model, prepared, baseline):
    with pytest.raises(ValueError, match="is not in this dataset"):
        dynamic_shap(
            model, prepared.validation, baseline,
            unit=999999, symbols=prepared.sensor_symbols,
        )


# --- the summary --------------------------------------------------------


def test_summary_ranks_by_average_size_of_contribution(prepared):
    values = np.array([[3.0, -1.0, 0.0], [1.0, -5.0, 0.0]])
    table = shap_summary(values, ["a", "b", "c"])
    assert table["sensor"].tolist() == ["b", "a", "c"]
    assert table.loc[0, "mean_abs"] == pytest.approx(3.0)
    assert table.loc[0, "mean"] == pytest.approx(-3.0)     # consistently negative
    assert table.loc[2, "mean_abs"] == 0.0


def test_summary_rejects_a_mismatched_label_list():
    with pytest.raises(ValueError, match="symbol"):
        shap_summary(np.zeros((2, 3)), ["a", "b"])


# --- the temporal stage runs once per window ----------------------------


def _every_group_through_the_whole_network(network, window, baseline):
    """Equation 15 with the whole network run on every one of the 16,384 mixed windows."""
    from thesis_gnn_rul_xai.explain.shap import _masked_batch

    n = window.shape[0]
    masks = ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)
    inputs = torch.from_numpy(_masked_batch(window, baseline, masks))
    with torch.no_grad():
        predictions = np.concatenate([
            network.predict_cycles(inputs[start:start + 2048]).double().numpy()
            for start in range(0, len(inputs), 2048)
        ])
    weights, sizes = coalition_weights(n), masks.sum(axis=1)
    values = np.empty(n)
    for i in range(n):
        without = np.flatnonzero(~masks[:, i])
        values[i] = np.sum(weights[sizes[without]]
                           * (predictions[without | (1 << i)] - predictions[without]))
    return values


@pytest.mark.parametrize(
    "stage", ["last_cycle", "mean_window", "window_summary", "raw_window", "lstm", "gru"]
)
def test_running_the_stage_once_per_window_changes_no_value(prepared, baseline, stage):
    """Each stage reads only its own sensor's window, so the shortcut is exact."""
    from thesis_gnn_rul_xai.models.temporal import build_temporal_stage

    torch.manual_seed(3)
    network = ESTGNN(
        prepared.adjacency_normalized,
        temporal=build_temporal_stage(stage, window_size=20, n_nodes=14),
        rul_max=125,
        allow_time_in_the_graph_layer=stage in ("window_summary", "raw_window"),
    )
    window = prepared.test.features[0]
    np.testing.assert_allclose(
        shap_values(network, window, baseline),
        _every_group_through_the_whole_network(network, window, baseline),
        rtol=0, atol=1e-5,
    )


def test_a_stage_that_mixes_sensors_is_refused(prepared, baseline):
    """The shortcut would be wrong for it, so no values are returned at all."""
    from thesis_gnn_rul_xai.models.temporal import TemporalStage

    class AddsTheAverageSensor(TemporalStage):
        """Each sensor's last reading plus the average of every sensor's last reading."""

        def __init__(self) -> None:
            super().__init__()
            self.window_size, self.out_features = 20, 1

        def forward(self, windows):
            last = windows[:, :, -1:]
            return last + last.mean(dim=1, keepdim=True)

    torch.manual_seed(3)
    network = ESTGNN(prepared.adjacency_normalized, temporal=AddsTheAverageSensor(), rul_max=125)
    with pytest.raises(ValueError, match="reads more than each sensor's own window"):
        shap_values(network, prepared.test.features[0], baseline)


# --- the figure's grey line is the capped label (added 2026-10-03) -------------


def test_the_figure_names_its_grey_line_the_capped_label(model, prepared, baseline):
    """Held-back engine 2 ran 287 cycles; its label is the remaining life capped at 125.

    Its first window ends at cycle 20 with 267 cycles truly left and the label
    125, and the label stays at 125 until cycle 163, where 124 remain. So in
    142 of its 268 windows the line sits below the remaining life the engine
    truly had, and the legend must not call it the true remaining life.
    """
    import matplotlib

    matplotlib.use("Agg")
    from thesis_gnn_rul_xai.explain import plot_contributions_over_time

    engine = int(prepared.validation.units[0])
    assert engine == 2
    table = dynamic_shap(model, prepared.validation, baseline, unit=engine,
                         symbols=prepared.sensor_symbols)
    line = plot_contributions_over_time(table, list(prepared.sensor_symbols)).axes[0].get_lines()[0]
    assert line.get_label() == "label: remaining life, capped"
    np.testing.assert_array_equal(line.get_ydata(), table["rul"].to_numpy())

    # A training engine runs to failure, so its last window ends at the failure
    # cycle, and the life truly left at a window is the cycles from its end to that.
    truly_left = table["end_cycle"].max() - table["end_cycle"]
    assert (table["end_cycle"].iloc[0], truly_left.iloc[0], table["rul"].iloc[0]) == (20, 267, 125)
    assert table["rul"].iloc[-1] == 0
    np.testing.assert_array_equal(table["rul"], np.minimum(truly_left, 125))
    assert table.loc[table["rul"] < 125, "end_cycle"].iloc[0] == 163
    assert (len(table), int((table["rul"] < truly_left).sum())) == (268, 142)
    matplotlib.pyplot.close("all")
