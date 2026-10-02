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
from thesis_gnn_rul_xai.models import (
    ESTGNN,
    GraphConvolution,
    LastCycle,
    LSTMStage,
    MeanWindow,
    WindowSummary,
)

# --- models/layers.py ---------------------------------------------------


def test_example_architecture_overview_follows_a_real_fd001_window(raw_dir):
    """The illustrated sizes and learned counts agree with a real window's path."""
    import xml.etree.ElementTree as ET

    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
    from thesis_gnn_rul_xai.docsite.figures import render_figure

    config = PreprocessingConfig()
    prepared = run_preprocessing(config, data_dir=raw_dir)
    model = ESTGNN(prepared.adjacency_normalized, window_size=config.window_size)
    with torch.no_grad():
        numbers = model.temporal(torch.from_numpy(prepared.train.features[:1]))
        shapes = [tuple(numbers.shape[1:])]
        for layer in model.graph_layers:
            numbers = torch.relu(layer(numbers, model.adjacency))
            shapes.append(tuple(numbers.shape[1:]))
        numbers = numbers.flatten(start_dim=1)
        shapes.append(tuple(numbers.shape[1:]))
        numbers = torch.relu(model.fc1(numbers))
        shapes.append(tuple(numbers.shape[1:]))
        numbers = torch.sigmoid(model.fc2(numbers))
        shapes.append(tuple(numbers.shape[1:]))
    illustrated = ET.fromstring('<div>' + render_figure('architecture-overview') + '</div>')
    layers = illustrated.findall('.//*[@data-architecture-layer]')
    assert [layer.attrib['data-shape'] for layer in layers] == [
        '(' + ', '.join(map(str, shape)) + ')' for shape in shapes
    ]
    summary = model.layer_summary()
    expected_counts = [summary[0]['parameters'], *(row['parameters'] for row in summary[2:])]
    assert [int(layer.attrib['data-parameters']) for layer in layers] == expected_counts
    assert sum(expected_counts) == model.count_parameters()
    assert shapes[3] == (14, 32) and shapes[4] == (14 * 32,)
    assert 'Only some entries' in ''.join(illustrated.itertext())
    matrix, symbols = build_physics_adjacency()
    expected_edges = {f'{symbols[a]} {symbols[b]}' for a in range(len(symbols))
                      for b in range(a + 1, len(symbols)) if matrix[a, b] or matrix[b, a]}
    for layer in layers[1:4]:
        assert {node.attrib['data-sensor'] for node in layer.findall('.//*[@data-sensor]')} == set(symbols)
        assert {edge.attrib['data-graph-edge'] for edge in layer.findall('.//*[@data-graph-edge]')} == expected_edges
        assert 'ReLU' in ''.join(layer.itertext())
    assert 'ReLU' in ''.join(layers[-2].itertext())
    assert 'Sigmoid' in ''.join(layers[-1].itertext())


def test_example_documentation_trace_reaches_the_saved_models_prediction():
    """The displayed arithmetic uses one real training window and the saved weights."""
    from thesis_gnn_rul_xai.docsite.worked_example import draw, trace

    try:
        example = trace()
    except FileNotFoundError:
        pytest.skip("The named stored run and FD001 files are required.")
    assert example["raw"][:3] == [641.82, 642.15, 642.35]
    assert example["last_cycle"] == 192
    assert (example["start"], example["end"]) == (1, 30)
    assert example["target"] == min(192 - example["end"], example["ceiling"])
    assert example["prediction"] == pytest.approx(example["reference"], abs=1e-6)
    assert example["steps"][0]["output"][0] == pytest.approx(example["scaled"][-1], abs=1e-6)
    assert len(example["steps"][5]["terms"]) == example["steps"][5]["input_shape"][-1]
    assert len(example["temporal_values"]) == 14
    for sensor in example["sensor_examples"]:
        alpha = sensor["alpha"]
        assert sensor["smoothed"][0] == sensor["raw"][0]
        for i in (1, 2):
            assert sensor["smoothed"][i] == pytest.approx(
                alpha * sensor["raw"][i] + (1 - alpha) * sensor["smoothed"][i - 1])
        assert sensor["scaled"] == pytest.approx(
            (np.array(sensor["smoothed"]) - sensor["mean"]) / sensor["spread"])
    for step in example["steps"]:
        for term in step["terms"]:
            assert term["product"] == pytest.approx(term["input"] * term["multiplier"], abs=1e-6)
        for addition in step["sums"]:
            for term in addition["terms"]:
                assert term["product"] == pytest.approx(term["input"] * term["multiplier"], abs=1e-6)
            assert sum(term["product"] for term in addition["terms"]) + addition["offset"] == pytest.approx(
                addition["total"], abs=2e-5)
    html = draw()
    assert example["run"] in html
    assert "Separate capped answer" in html


def test_example_engine_connections_come_from_the_selected_saved_network(tmp_path):
    """The correlation graph drawn over the engine equals the saved seed's matrix."""
    from pathlib import Path

    from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_SYMBOLS
    from thesis_gnn_rul_xai.docsite.playground.runs import _explanation_edges

    root = Path(__file__).resolve().parents[1]
    saved = next(iter(sorted((root / "reports").glob("*correlation*/seed_*/model.pt"))), None)
    if saved is None:
        pytest.skip("A stored run on a correlation graph is needed for this example.")
    seed = int(saved.parent.name.removeprefix("seed_"))
    edges = _explanation_edges(saved.parents[1], {"seed": seed})
    matrix = torch.load(saved, map_location="cpu", weights_only=True)["adjacency"].numpy()
    symbols = PAPER_SENSOR_SYMBOLS
    expected = [[a, b] for i, a in enumerate(symbols) for j, b in enumerate(symbols)
                if i < j and (matrix[i, j] != 0 or matrix[j, i] != 0)]
    assert edges == expected
    physics, _ = build_physics_adjacency()
    assert any((matrix[i, j] != 0) != (physics[i, j] != 0)
               for i in range(len(symbols)) for j in range(i + 1, len(symbols)))
    assert _explanation_edges(tmp_path, {"seed": seed}) is None
    assert _explanation_edges(saved.parents[1], {}) is None


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


# --- models/config.py ---------------------------------------------------


def test_example_a_run_name_gets_six_characters_that_change_with_any_network_setting():
    """Why a run directory name ends in something like `net4e49f2`, and how it is made.

    The problem: two runs that differ only in, say, the width of the first fully
    connected layer must never get the same directory name, or the second would
    be mistaken for the first. Spelling every setting out in the name would make
    it hundreds of characters long.

    The answer is a short fingerprint of all the settings together, made in
    three steps:

    1. Write every setting of the network out as one line of text. A
       `ModelConfig` is the object holding those settings. The keys are put in
       alphabetical order so the same settings always give the same line,
       whatever order they were typed in.
    2. Feed that line to SHA-256, a standard function that turns any text into
       64 hexadecimal characters. The same text always gives the same 64; text
       that differs by one character gives 64 unrelated ones.
    3. Keep the first six.

    Below, the default settings give "4e49f2". Changing one setting, the first
    fully connected layer from 128 numbers wide to 64, gives "f98ee3": a
    different fingerprint, so a different directory.
    """
    import hashlib
    import json

    from thesis_gnn_rul_xai.models import ModelConfig

    default = ModelConfig()

    # Step 1: the settings as one line of text, keys in alphabetical order.
    text = json.dumps(default.to_dict(), sort_keys=True)
    assert text == (
        '{"fc_hidden": 128, "gcn_bias": false, "gcn_hidden": [32, 64, 32], '
        '"shared_across_sensors": false, "temporal": "last_cycle", '
        '"temporal_hidden": 32, "temporal_out_features": 1}'
    )

    # Steps 2 and 3: SHA-256 of that text, first six characters. digest() is
    # the method that does both, and it gives the same answer as doing it by hand.
    by_hand = hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]
    assert by_hand == default.digest() == "4e49f2"

    # One setting changed, the fingerprint changed.
    assert ModelConfig(fc_hidden=64).digest() == "f98ee3"


def test_example_load_model_config_names_the_unknown_key(tmp_path):
    from thesis_gnn_rul_xai.models import ModelConfig, load_model_config

    saved = ModelConfig().save(tmp_path / "model_config.yaml")
    assert load_model_config(saved) == ModelConfig()

    (tmp_path / "bad.yaml").write_text("temporal: lstm\nhidden: 64\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Unknown model setting\(s\) in .*bad\.yaml: \['hidden'\]"):
        load_model_config(tmp_path / "bad.yaml")


def test_example_how_many_numbers_the_network_learns_with_each_temporal_stage():
    """The same graph layers, four different first stages, and what each costs.

    "Learns" means the numbers training adjusts: the entries of the weight
    matrices and biases. `count_parameters()` adds them up. Every model below is
    built on the paper's sensor graph, for windows of 30 cycles.

    The temporal stage sits in front of the graph layers and decides what each
    sensor hands to the first one, which is called W(1). W(1) is a table of
    multipliers with one row per number a sensor hands over and one column per
    output (32 columns, from Table 4). So it is 1 x 32 = 32 multipliers when
    each sensor hands over one number, and 30 x 32 = 960 when it hands over 30.

    - last_cycle keeps the most recent of the 30 readings. It learns nothing
      and hands over one number: 61,729 in all.
    - mean_window averages the 30 readings. It also learns nothing and hands
      over one number, so it is the same 61,729.
    - lstm is the paper's stage: 14 recurrent networks, one per sensor. Each
      has 4,480 learned numbers (its four gates each hold 32 weights for the
      input, 32 x 32 for the previous state and two sets of 32 biases:
      4 x 1,120). Then 33 more turn its 32 outputs into one number
      (32 weights and 1 bias). 61,729 + 14 x 4,480 + 33 = 124,482.
    - raw_window learns nothing but hands over all 30 readings, so W(1) grows
      from 32 multipliers to 960: 61,729 - 32 + 960 = 62,657.
    - window_summary learns nothing and hands over three numbers, so W(1) is
      3 x 32 = 96 multipliers: 61,729 - 32 + 96 = 61,793.
    """
    from thesis_gnn_rul_xai.models import ModelConfig, build_model

    graph = normalize_adjacency(build_physics_adjacency()[0], method="row")
    sizes = {
        stage: build_model(graph, ModelConfig(temporal=stage),
                           window_size=30, rul_max=125).count_parameters()
        for stage in ("last_cycle", "mean_window", "lstm", "raw_window", "window_summary")
    }

    assert sizes["last_cycle"] == 61_729
    assert sizes["mean_window"] == 61_729
    assert sizes["lstm"] == 61_729 + 14 * 4 * (32 + 32 * 32 + 32 + 32) + (32 + 1) == 124_482
    assert sizes["raw_window"] == 61_729 - 1 * 32 + 30 * 32 == 62_657
    assert sizes["window_summary"] == 61_729 - 1 * 32 + 3 * 32 == 61_793


# --- explain/shap.py ----------------------------------------------------


def test_example_xai_overview_keeps_prediction_and_explanation_units_distinct(raw_dir, monkeypatch):
    """Recompute every visible cell and both reconstructions on the same real window."""
    from test_examples import _check_design_tutorial

    example = _check_design_tutorial("03_explainability.md", monkeypatch, minimum_tables=4)
    assert example["readings"].Cycle.tolist() == [96, 97, 98]
    assert example["window"].shape == (3, 3)
    assert example["reconstruction"].Unit.tolist() == ["cycles", "internal score"]
    assert example["shap_total"] == pytest.approx(example["predicted"], abs=1e-5)
    assert example["walk_total"] == pytest.approx(example["score"], abs=1e-6)
    assert example["phi"][1] > 0 > example["routes"][1]


def test_example_dynamic_shap_tutorial_traces_every_group_and_window(raw_dir, monkeypatch):
    """Verify all printed samples and complete group enumeration on real sensor histories."""
    from test_examples import _check_design_tutorial

    example = _check_design_tutorial("10_dynamic_shap.md", monkeypatch, minimum_tables=13)
    assert example["features"].shape == (3, 3)
    assert example["masked_windows"].shape == (8, 3, 3)
    np.testing.assert_array_equal(example["masked_windows"][0], example["baseline"])
    np.testing.assert_array_equal(example["masked_windows"][-1], example["features"])
    assert example["values"].shape == (3,)
    assert example["trajectory_features"].shape == (3, 3, 3)
    assert example["trajectory"]["end_cycle"].tolist() == [30, 31, 32]
    assert example["residual"] < 1e-4


def test_example_gnn_lrp_tutorial_traces_routes_and_checks_changed_graphs(raw_dir, monkeypatch):
    """Verify every printed route and each sensor-addition and removal result."""
    from test_examples import _check_design_tutorial

    example = _check_design_tutorial("11_gnn_lrp.md", monkeypatch, minimum_tables=13)
    assert example["window"].shape == (3, 3)
    assert len(example["walk_table"]) == 9
    assert len(example["subset_table"]) == 8
    np.testing.assert_allclose(example["walks"].relevance[0], example["manual_shares"], atol=1e-12)
    assert example["walks"].gamma == (0.0,)
    assert example["adding_order"] == [1, 0, 2]
    assert example["removing_order"] == [0, 1, 2]
    assert example["adding_curve"]["Sensors present"].tolist() == [
        "None", "T30", "T24, T30", "T24, T30, T50",
    ]
    assert example["removing_curve"]["Sensors present"].tolist() == [
        "T24, T30, T50", "T30, T50", "T50", "None",
    ]


def test_example_coalition_weights():
    """With three sensors, the empty group and the full group each count 1/3,
    and each single-sensor group counts 1/6."""
    assert coalition_weights(3).round(6).tolist() == [0.333333, 0.166667, 0.333333]


def test_example_occlusion_is_the_shapley_term_where_the_sensor_arrives_last():
    """Three sensors on a complete graph. Occlusion credits sensor 0 with
    f(all three) - f(sensors 1 and 2): the one term of the Shapley sum in which
    sensor 0 joins last. The Shapley value averages that term with the three
    others, weighted 1/3, 1/6, 1/6 and 1/3."""
    from thesis_gnn_rul_xai.explain import occlusion_values, shap_values
    from thesis_gnn_rul_xai.graph.adjacency import build_fully_connected_adjacency

    torch.manual_seed(1)
    complete, _ = build_fully_connected_adjacency(3)
    model = ESTGNN(normalize_adjacency(complete), window_size=4, rul_max=125)
    baseline = np.zeros((3, 4), dtype=np.float32)
    features = np.array(
        [[0.9, 0.8, 0.7, 0.6], [0.2, 0.4, 0.6, 0.8], [0.1, 0.2, 0.3, 0.4]],
        dtype=np.float32,
    )

    def f(*present: int) -> float:
        keep = np.isin(np.arange(3), present)[:, None]
        window = np.where(keep, features, baseline)[None]
        with torch.no_grad():
            return float(model.predict_cycles(torch.from_numpy(window))[0])

    assert coalition_weights(3).tolist() == pytest.approx([1 / 3, 1 / 6, 1 / 3])
    by_hand = (
        (f(0) - f()) / 3
        + (f(0, 1) - f(1)) / 6
        + (f(0, 2) - f(2)) / 6
        + (f(0, 1, 2) - f(1, 2)) / 3
    )
    assert occlusion_values(model, features, baseline)[0] == pytest.approx(
        f(0, 1, 2) - f(1, 2), abs=1e-4
    )
    assert shap_values(model, features, baseline)[0] == pytest.approx(by_hand, abs=1e-4)


def test_example_coalition_weights_add_up_to_one():
    """The weights cover every ordering of the sensors exactly once, so
    multiplying each by the number of groups of that size gives 1."""
    n = 14
    counts = np.array([comb(n - 1, size) for size in range(n)])
    assert round(float(np.sum(counts * coalition_weights(n))), 9) == 1.0
    assert coalition_weights(3)[0] == factorial(0) * factorial(2) / factorial(3)


# --- models/temporal/ ---------------------------------------------------


@pytest.fixture(scope="module")
def playground_temporal_example(raw_dir):
    """Load the tutorial's real FD001 engine-1 window and its T50 position."""
    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing

    prepared = run_preprocessing(PreprocessingConfig(seed=42), data_dir=raw_dir)
    row = int(np.flatnonzero(prepared.train.units == 1)[0])
    assert prepared.train.end_cycles[row] == 30
    windows = torch.from_numpy(prepared.train.features[row:row + 1])
    sensor = list(prepared.sensor_symbols).index("T50")
    return windows, sensor


def _playground_temporal_stage(kind, shared=False, outputs=1):
    """Build the tutorial's untrained arithmetic demonstration on real readings."""
    stage = LSTMStage(window_size=30, n_nodes=14, hidden_size=1,
                      cell=kind, shared_across_sensors=shared, out_features=outputs)
    with torch.no_grad():
        for cell in stage.cells:
            cell.weight_ih_l0.fill_(0.2)
            cell.weight_hh_l0.fill_(0.1)
            cell.bias_ih_l0.zero_()
            cell.bias_hh_l0.zero_()
        stage.project.weight.fill_(0.5)
        stage.project.bias.fill_(0.1)
        if outputs == 2:
            stage.project.weight[1].fill_(-0.25)
            stage.project.bias[1].fill_(0.2)
    return stage


def test_example_playground_temporal_fixed_summaries(playground_temporal_example):
    """Check each displayed fixed summary against arithmetic on the real window."""
    from thesis_gnn_rul_xai.models.temporal import RawWindow

    windows, sensor = playground_temporal_example
    readings = windows[0, sensor]
    assert tuple(windows.shape) == (1, 14, 30)
    assert [round(float(x), 6) for x in readings[:3]] == [-0.99521, -0.8415, -0.700499]
    assert [round(float(x), 6) for x in windows[0, 0, :3]] == [-1.987346, -1.6039, -1.179785]
    assert [round(float(x), 6) for x in windows[0, 1, :3]] == [-0.146319, 0.060766, -0.209812]

    positions = torch.arange(30, dtype=windows.dtype) - 14.5
    terms = readings * positions
    numerator = terms.sum()
    denominator = positions.square().sum()
    slope = numerator / denominator
    assert [round(float(x), 6) for x in terms[:3]] == [14.430549, 11.360254, 8.756233]
    assert round(float(readings.sum()), 6) == -31.346590
    assert round(float(numerator), 6) == -27.887846
    assert float(denominator) == 2247.5
    assert round(float(slope), 9) == -0.012408385
    assert round(float(readings[-1] - readings[0]), 6) == -0.298558

    hand_slope = sum(float(t) * float(x) for t, x in zip(positions, readings, strict=True))
    hand_slope /= sum(float(t) ** 2 for t in positions)
    hand_summary = [float(readings[-1]), sum(map(float, readings)) / 30, hand_slope * 29]
    summary = WindowSummary(30)(windows)
    np.testing.assert_allclose(summary[0, sensor].numpy(), hand_summary, atol=1e-6)
    assert [round(float(x), 6) for x in summary[0, sensor]] == [-1.293769, -1.044886, -0.359843]
    torch.testing.assert_close(LastCycle(30)(windows), summary[:, :, :1])
    torch.testing.assert_close(MeanWindow(30)(windows), summary[:, :, 1:2])
    assert tuple(summary.shape) == (1, 14, 3)
    assert RawWindow(30)(windows) is windows
    print("Real T50 summaries:", [round(float(x), 6) for x in summary[0, sensor]])


@pytest.mark.parametrize("kind", ["lstm", "gru"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("outputs", [1, 2])
def test_example_playground_temporal_memory_and_projection(
    playground_temporal_example, kind, shared, outputs
):
    """Rebuild every memory update and projection independently on real T50 data."""
    windows, sensor = playground_temporal_example
    stage = _playground_temporal_stage(kind, shared, outputs)
    h = torch.zeros((), dtype=windows.dtype)
    c = torch.zeros((), dtype=windows.dtype)
    hidden_values, candidate_values, gates, cell_values = [], [], [], []
    for reading in windows[0, sensor]:
        gate = torch.sigmoid(0.2 * reading + 0.1 * h)
        if kind == "lstm":
            candidate = torch.tanh(0.2 * reading + 0.1 * h)
            c = gate * c + gate * candidate
            h = gate * torch.tanh(c)
        else:
            candidate = torch.tanh(0.2 * reading + gate * 0.1 * h)
            h = gate * h + (1 - gate) * candidate
        gates.append(float(gate))
        candidate_values.append(float(candidate))
        cell_values.append(float(c))
        hidden_values.append(h)

    with torch.no_grad():
        if shared:
            histories, _ = stage.cells[0](windows.reshape(14, 30, 1))
            history = histories[sensor, :, 0]
            assert tuple(histories.shape) == (14, 30, 1)
        else:
            histories, _ = stage.cells[sensor](windows[:, sensor, :, None])
            history = histories[0, :, 0]
            assert tuple(histories.shape) == (1, 30, 1)
        actual = stage(windows)
        repeated = stage(windows)
    torch.testing.assert_close(history, torch.stack(hidden_values), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(actual, repeated)
    assert tuple(actual.shape) == (1, 14, outputs)
    expected = [0.5 * float(h) + 0.1]
    if outputs == 2:
        expected.append(-0.25 * float(h) + 0.2)
    np.testing.assert_allclose(actual[0, sensor].numpy(), expected, atol=1e-7)

    def rounded(values):
        """Match the six-decimal values printed by the tutorial."""
        return [round(float(x), 6) for x in values]
    if kind == "lstm":
        assert rounded(gates[:3]) == [0.450403, 0.457037, 0.463692]
        assert rounded(candidate_values[:3]) == [-0.196455, -0.170591, -0.144468]
        assert rounded(cell_values[:3]) == [-0.088484, -0.118407, -0.121893]
        assert rounded(history[:3]) == [-0.03975, -0.053865, -0.056243]
        assert round(float(history[-1]), 6) == -0.081576
        assert rounded(actual[0, sensor]) == [0.059212, 0.220394][:outputs]
        assert rounded(actual[0, :3, 0]) == [0.062085, 0.081797, 0.059212]
    else:
        assert rounded(gates[:3]) == [0.450403, 0.455345, 0.461487]
        assert rounded(candidate_values[:3]) == [-0.196455, -0.171505, -0.145636]
        assert rounded(history[:3]) == [-0.107971, -0.142575, -0.144224]
        assert round(float(history[-1]), 6) == -0.246909
        assert rounded(actual[0, sensor]) == [-0.023454, 0.261727][:outputs]
        assert rounded(actual[0, :3, 0]) == [-0.010399, 0.056611, -0.023454]
    print(kind, "shared" if shared else "dedicated", "T50 hidden:", round(float(h), 6),
          "graph input:", rounded(actual[0, sensor]))


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_example_playground_temporal_sharing_keeps_histories_separate(
    playground_temporal_example, kind
):
    """Verify separate sensor memories and the difference between shared and dedicated weights."""
    windows, sensor = playground_temporal_example
    dedicated = _playground_temporal_stage(kind)
    shared = _playground_temporal_stage(kind, shared=True)
    with torch.no_grad():
        before = dedicated(windows)
        torch.testing.assert_close(before, shared(windows))
        dedicated.cells[0].weight_ih_l0.fill_(0.3)
        after = dedicated(windows)
        assert not torch.allclose(after[:, 0], before[:, 0])
        torch.testing.assert_close(after[:, 1:], before[:, 1:])

        reverse_sensors = windows.flip(dims=(1,))
        torch.testing.assert_close(shared(reverse_sensors).flip(dims=(1,)), shared(windows))
        separate = [windows[:, node, :, None] for node in range(14)]
        flat = windows.reshape(14, 30, 1)
        assert len(separate) == 14
        assert tuple(separate[sensor].shape) == (1, 30, 1)
        torch.testing.assert_close(separate[sensor][0], flat[sensor])


@pytest.mark.parametrize("filename,minimum_tables", [
    ("09_temporal.md", 30), ("05_the_model.md", 28),
])
def test_example_design_model_tutorials(raw_dir, monkeypatch, filename, minimum_tables):
    """Execute every teaching step and check every displayed input, calculation and output."""
    from test_examples import _check_design_tutorial

    namespace = _check_design_tutorial(filename, monkeypatch, minimum_tables)
    if filename == "09_temporal.md":
        assert tuple(namespace["windows"].shape) == (1, 14, 4)
        assert namespace["source"].shape == (4, 15)
    else:
        assert tuple(namespace["window"].shape) == (1, 14, 30)
        with torch.no_grad():
            torch.testing.assert_close(namespace["cycles"], namespace["model"].predict_cycles(namespace["window"]))


def test_example_the_stage_decides_the_width_of_the_first_graph_matrix(raw_dir):
    """On the real FD001 windows: same data in, two different networks out.

    The pipeline hands over all 30 cycles. What reaches the graph layers, and
    therefore how wide W(1) is, is decided by the stage and nothing else --
    which is the whole point of A-07 living in the model.
    """
    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing

    prepared = run_preprocessing(PreprocessingConfig(), data_dir=raw_dir)
    assert prepared.test.features.shape == (100, 14, 30)

    batch = torch.from_numpy(prepared.test.features[:8])
    graph = normalize_adjacency(build_physics_adjacency()[0], method="row")

    stand_in = ESTGNN(graph, temporal=LastCycle(30))
    assert stand_in.in_features == 1
    assert stand_in.count_parameters() == 61_729
    assert tuple(stand_in(batch).shape) == (8,)

    paper = ESTGNN(graph, temporal=LSTMStage(window_size=30, n_nodes=14,
                                             hidden_size=32))
    assert paper.in_features == 1
    # Fourteen dedicated cells at hidden size 32, plus the projection to one
    # number, on top of the same 61,729-parameter graph stage.
    assert paper.count_parameters() == 61_729 + 62_753
    assert tuple(paper(batch).shape) == (8,)


def test_example_window_summary_on_one_real_sensor_window(raw_dir):
    """T50 of engine 1 over its cycles 1 to 30: the three numbers, worked by hand.

    The readings are after smoothing and z-score scaling. The slope is the best
    straight line's rise per cycle, with the cycles numbered from the window's
    middle so that the average drops out: slope = sum(t * reading) / sum(t^2),
    where t runs -14.5, -13.5, ..., +14.5 and sum(t^2) = 2247.5. The change is the
    slope times the 29 cycles from the first reading to the last.
    """
    from dataclasses import replace

    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing

    prepared = run_preprocessing(replace(PreprocessingConfig(), seed=42), data_dir=raw_dir)
    first = int(np.where(prepared.train.units == 1)[0][0])
    sensor = list(prepared.sensor_symbols).index("T50")
    readings = prepared.train.features[first, sensor].astype(np.float64)
    assert prepared.train.end_cycles[first] == 30

    steps = [cycle - 14.5 for cycle in range(30)]
    assert sum(step * step for step in steps) == 2247.5
    slope = sum(step * reading for step, reading in zip(steps, readings, strict=True)) / 2247.5
    by_hand = [readings[-1], sum(readings) / 30, slope * 29]
    assert [round(value, 4) for value in by_hand] == [-1.2938, -1.0449, -0.3598]
    assert round(slope, 6) == -0.012408

    stage = WindowSummary(30)(torch.from_numpy(prepared.train.features[first:first + 1]))
    np.testing.assert_allclose(stage[0, sensor].numpy(), by_hand, atol=1e-5)


# --- explain/walks.py ---------------------------------------------------


def test_example_walks_on_three_real_sensors_worked_by_hand(raw_dir):
    """Test engine 5's last readings of T24, T30 and T50, split into walks by hand.

    The three sensors form a path in Table 3: T24 - T30 - T50, with no T24 - T50
    edge. Row rescaling with each sensor joined to itself gives the rows
    T24 [1/2, 1/2, 0], T30 [1/3, 1/3, 1/3], T50 [0, 1/2, 1/2]. One graph layer
    of width 1 with weight 1, then fc1 with weights [2, 1, -1] and constant 0.5,
    then fc2 with weight 1 and constant 0.

    The graph layer: T24 gets (-0.3734 + 0.1146) / 2 = -0.1294, below zero, so
    its ReLU passes nothing on. T30 gets (-0.3734 + 0.1146 + 0.8176) / 3 =
    0.1863 and T50 gets (0.1146 + 0.8176) / 2 = 0.4661. fc1 then gives
    2 x 0 + 1 x 0.1863 - 1 x 0.4661 + 0.5 = 0.2202, above zero, and that is the
    score. 125 / (1 + e^-0.2202) = 69.35 cycles; the true remaining life is 91.

    A walk here is one step, from a sensor to a sensor, and its share is the
    reading it starts from, times the graph's entry for the step, times what one
    unit at the arrival sensor is worth to the score (0 at T24, 1 at T30, -1 at
    T50). The five walks along an edge that exists, and the constant:

        T24 -> T30   -0.3734 x 1/3 x  1 = -0.1245
        T30 -> T30    0.1146 x 1/3 x  1 =  0.0382
        T50 -> T30    0.8176 x 1/3 x  1 =  0.2725
        T30 -> T50    0.1146 x 1/2 x -1 = -0.0573
        T50 -> T50    0.8176 x 1/2 x -1 = -0.4088
        constant      0.5 x 1           =  0.5
                                    total  0.2202

    T24 -> T24 carries 0 because T24's ReLU was off, and T24 -> T50 and
    T50 -> T24 carry 0 because the graph has no such edge. T24 still has a share,
    -0.1245, through the walk that leaves it.
    """
    from dataclasses import replace

    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
    from thesis_gnn_rul_xai.explain import node_relevance, walk_relevance

    prepared = run_preprocessing(replace(PreprocessingConfig(), seed=42), data_dir=raw_dir)
    row = int(np.where(prepared.test.units == 5)[0][0])
    chosen = [list(prepared.sensor_symbols).index(s) for s in ("T24", "T30", "T50")]
    window = prepared.test.features[row, chosen]
    assert [round(float(v), 4) for v in window[:, -1]] == [-0.3734, 0.1146, 0.8176]

    path = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    model = ESTGNN(
        normalize_adjacency(path, method="row"), window_size=30, gcn_hidden=(1,), fc_hidden=1
    )
    with torch.no_grad():
        model.graph_layers[0].weight.fill_(1.0)
        model.fc1.weight.copy_(torch.tensor([[2.0, 1.0, -1.0]]))
        model.fc1.bias.fill_(0.5)
        model.fc2.weight.fill_(1.0)
        model.fc2.bias.fill_(0.0)

    result = walk_relevance(model, window)
    from_to = result.relevance[0]
    by_hand = [
        [0.0, -0.3734 / 3, 0.0],
        [0.0, 0.1146 / 3, -0.1146 / 2],
        [0.0, 0.8176 / 3, -0.8176 / 2],
    ]
    np.testing.assert_allclose(from_to, by_hand, atol=1e-4)
    assert float(result.constant[0]) == pytest.approx(0.5)
    assert round(float(result.score[0]), 4) == 0.2202
    assert round(float(result.predicted[0]), 2) == 69.35
    assert [round(float(v), 4) for v in node_relevance(result)[0]] == [-0.1245, -0.0191, -0.1363]


def test_example_removing_a_connection_by_hand_on_the_same_three_sensors(raw_dir):
    """The network of the walk example, with each of its two connections removed.

    Removing T24 - T30 deletes it from the raw graph and rescales: T24's row
    becomes [1, 0, 0] and T30's [0, 1/2, 1/2], so T30 now weighs T50 by 1/2
    instead of 1/3 -- a change to entries nobody removed. T24 gets -0.3734, still
    below zero; T30 and T50 both get (0.1146 + 0.8176) / 2 = 0.4661; fc1 gives
    2 x 0 + 0.4661 - 0.4661 + 0.5 = 0.5, a prediction of 125 / (1 + e^-0.5) =
    77.81 cycles. With the connection the prediction was 69.35, so the connection
    is worth 69.35 - 77.81 = -8.45 cycles.

    Removing T30 - T50 instead: T30 gets (-0.3734 + 0.1146) / 2 = -0.1294 and
    switches off, T50 keeps only its own 0.8176, fc1 gives -0.8176 + 0.5 = -0.3176
    and switches off, the score is fc2's constant 0, a prediction of 62.5 cycles.
    That connection is worth 69.35 - 62.5 = +6.85 cycles.
    """
    from dataclasses import replace

    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
    from thesis_gnn_rul_xai.explain import connection_removal

    prepared = run_preprocessing(replace(PreprocessingConfig(), seed=42), data_dir=raw_dir)
    row = int(np.where(prepared.test.units == 5)[0][0])
    chosen = [list(prepared.sensor_symbols).index(s) for s in ("T24", "T30", "T50")]
    window = prepared.test.features[row, chosen]

    path = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    model = ESTGNN(
        normalize_adjacency(path, method="row"), window_size=30, gcn_hidden=(1,), fc_hidden=1
    )
    with torch.no_grad():
        model.graph_layers[0].weight.fill_(1.0)
        model.fc1.weight.copy_(torch.tensor([[2.0, 1.0, -1.0]]))
        model.fc1.bias.fill_(0.5)
        model.fc2.weight.fill_(1.0)
        model.fc2.bias.fill_(0.0)

    assert normalize_adjacency(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]), method="row")[
        1
    ].tolist() == [0.0, 0.5, 0.5]
    table = connection_removal(model, window, path, ["T24", "T30", "T50"], method="row")
    assert list(table.columns) == ["T24-T30", "T30-T50"]
    assert round(125 / (1 + np.exp(-0.5)), 2) == 77.81
    assert [round(float(v), 2) for v in table.iloc[0]] == [-8.45, 6.85]


# --- explain/flipping.py ------------------------------------------------


def test_example_node_flipping_by_hand_on_the_same_three_sensors(raw_dir):
    """The GNN-LRP paper's own check (its Section 4.1) on the walk example's network.

    What the walks say a set of sensors is worth is every walk lying wholly
    inside it: {T30} is worth T30 -> T30 = 0.0382, {T50} is worth T50 -> T50 =
    -0.4088, {T24} is worth nothing (its own walk carries 0). Adding sensors
    greedily (Equation 14) takes T30 first; then {T30, T24} is worth
    0.0382 - 0.1245 = -0.0862 and {T30, T50} is worth
    0.0382 + 0.2725 - 0.0573 - 0.4088 = -0.1554, so T24 comes second and T50 last.

    The network on each of those graphs, with the missing sensors' rows and
    columns of the rescaled graph set to zero: with no sensor, fc1 gives its
    constant 0.5, the score; with T30 alone, T30 holds 0.1146 / 3 = 0.0382 and
    the score is 0.5382; adding T24, T30 holds (-0.3734 + 0.1146) / 3 < 0 and
    switches off, the score is 0.5 again; with all three, 0.2202. The area under
    that curve is its mean, (0.5 + 0.5382 + 0.5 + 0.2202) / 4 = 0.4396.
    """
    from dataclasses import replace

    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
    from thesis_gnn_rul_xai.explain import (
        node_flipping,
        subgraph_relevance,
        walk_relevance,
    )

    prepared = run_preprocessing(replace(PreprocessingConfig(), seed=42), data_dir=raw_dir)
    row = int(np.where(prepared.test.units == 5)[0][0])
    chosen = [list(prepared.sensor_symbols).index(s) for s in ("T24", "T30", "T50")]
    window = prepared.test.features[row, chosen]

    path = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    model = ESTGNN(
        normalize_adjacency(path, method="row"), window_size=30, gcn_hidden=(1,), fc_hidden=1
    )
    with torch.no_grad():
        model.graph_layers[0].weight.fill_(1.0)
        model.fc1.weight.copy_(torch.tensor([[2.0, 1.0, -1.0]]))
        model.fc1.bias.fill_(0.5)
        model.fc2.weight.fill_(1.0)
        model.fc2.bias.fill_(0.0)

    walks = walk_relevance(model, window)
    worth = {
        name: round(subgraph_relevance(walks.relevance[0], np.array(mask, dtype=bool)), 4)
        for name, mask in (("T24", [1, 0, 0]), ("T30", [0, 1, 0]), ("T50", [0, 0, 1]),
                           ("T24 T30", [1, 1, 0]), ("T30 T50", [0, 1, 1]))
    }
    assert worth == {"T24": 0.0, "T30": 0.0382, "T50": -0.4088,
                     "T24 T30": -0.0862, "T30 T50": -0.1554}

    table = node_flipping(model, window, {"GNN-LRP": walks}, random_orders=0)
    adding = table[(table["task"] == "activation") & (table["ranking"] == "GNN-LRP, walks")].iloc[0]
    assert adding["order"] == "1 0 2"
    curve = [round(float(adding[f"step_{k}"]), 4) for k in range(4)]
    assert curve == [0.5, 0.5382, 0.5, 0.2202]
    assert round(float(adding["aufc"]), 4) == 0.4396


# --- docs/resources/: the papers the project builds on ---------------


def test_example_lstm_resource_page(raw_dir, monkeypatch):
    """The LSTM page reproduces every printed cell; its hand-written cells match the papers' rules and nn.LSTM."""
    from test_examples import _check_design_tutorial

    page = _check_design_tutorial("lstm.md", monkeypatch, minimum_tables=11, folder="resources")
    # Eq. 2 on real readings: every product shrinks with q; the carousel's stays exactly 1.
    weak = page["products"](page["weak"])
    assert np.all(np.diff(weak) < 0) and weak[-1] < 1e-30
    np.testing.assert_array_equal(page["carousel"], np.ones(len(page["steps"])))
    # The 1997 cell keeps its whole state: s(192) = s(191) + input gate x squashed cell input.
    trace = page["trace_1997"].set_index("Quantity")
    assert trace.loc["Internal state s", "Cycle 192"] == pytest.approx(
        trace.loc["Internal state s", "Cycle 191"]
        + trace.loc["Input gate", "Cycle 192"] * trace.loc["Squashed cell input g", "Cycle 192"])
    # The 1997 rule (Eqs. 23, 27, 28) is automatic differentiation with the previous output detached.
    assert page["w_cell"].grad.item() == pytest.approx(page["rule_1997"].iloc[1, -1], abs=1e-12)
    # Gers et al.'s Eq. 15 equals the cut derivative, and the full derivative differs from both.
    values = page["derivatives"].iloc[:, 1].to_numpy()
    assert values[0] == pytest.approx(values[1], abs=1e-8)
    assert values[2] == pytest.approx(values[3], abs=1e-12) and values[1] - values[2] > 0.005
    # Without a forget gate the state rises at every engine boundary shown; with one it stays bounded.
    assert np.all(np.diff(page["stream_table"]["Internal state, no forget gate"]) > 0)
    assert np.abs(page["with_forget"]).max() < page["largest_add"] / (1 - page["keep"].max())
    # PyTorch's starting forget gates sit near 0.5, below Gers et al.'s first-block 0.62.
    start = page["starting"].set_index("Quantity")["Value"]
    assert 0.41 < start.iloc[2] < 0.5 < start.iloc[3] < 0.59 < start.iloc[4]


def test_reference_gnn_lrp_page_reproduces_every_walk_and_the_papers_procedure(raw_dir, monkeypatch):
    """The GNN-LRP resource page: the hand split, the paper's masked backward passes and the
    project's walk_relevance agree on every walk of a real FD001 window."""
    from test_examples import _check_design_tutorial

    page = _check_design_tutorial("gnn_lrp.md", monkeypatch, minimum_tables=16, folder="resources")
    assert page["end_cycle"] == 88
    assert len(page["walk_table"]) == 17
    assert page["along_connections"] == 5178
    # Gradient x entry per connection counts every walk once per layer: twice the score.
    assert page["edge_scores"]["Slope × entry"].sum() == pytest.approx(2 * page["score"], abs=1e-8)  # noqa: RUF001
    # Eq. 6 bag term equals the sum of its two walks (Proposition 2).
    assert page["bag_table"]["Value"].iloc[3] == pytest.approx(page["bag_table"]["Value"].iloc[6], abs=1e-8)
    assert page["project_check"]["Largest gap from the hand split"].max() < 1e-6
    np.testing.assert_allclose(page["gi"].sum(), page["score"], atol=1e-12)
    np.testing.assert_allclose(page["lrp"].sum(), page["score"], atol=1e-12)
    # At gamma 0 the walk estimate is the rerun network wherever no switch flips.
    table = page["subset_table"]
    no_flip = ~table["Sensors connected"].isin(["T50", "T30, T50"])
    np.testing.assert_allclose(
        table.loc[no_flip, "Walk estimate, GNN-GI"], table.loc[no_flip, "Network's score"], atol=1e-12
    )
    # Table 1's weight-sign gamma enlarges a negative share when the input is negative.
    two = page["two_numbers"]
    assert two["Share, Table 1's form"].iloc[3] < two["Contribution"].iloc[3] < 0


# --- models/temporal/README.md: the worked example ---------------------------


def test_example_temporal_readme_on_two_real_sensors(raw_dir, playground_temporal_example):
    """T24 and T50 of FD001 engine 1, cycles 1 to 4: every number the README prints, by hand."""
    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing

    prepared = run_preprocessing(PreprocessingConfig(seed=42, window_size=4), data_dir=raw_dir)
    row = int(np.flatnonzero(prepared.train.units == 1)[0])
    assert prepared.train.end_cycles[row] == 4
    names = list(prepared.sensor_symbols)
    pair = [names.index("T24"), names.index("T50")]
    assert pair == [0, 2]
    windows = torch.from_numpy(prepared.train.features[row:row + 1, pair])
    assert tuple(windows.shape) == (1, 2, 4)
    readings = [[round(float(x), 6) for x in sensor] for sensor in windows[0]]
    assert readings == [[-1.987346, -1.6039, -1.179785, -0.967728],
                        [-0.99521, -0.8415, -0.700499, -0.770999]]
    # The same four readings open engine 1's thirty-cycle window.
    thirty, _ = playground_temporal_example
    torch.testing.assert_close(windows, thirty[:, pair, :4], atol=0.0, rtol=0.0)

    last = LastCycle(4)(windows)
    assert tuple(last.shape) == (1, 2, 1)
    assert [round(float(x), 6) for x in last[0, :, 0]] == [-0.967728, -0.770999]
    assert [round(float(x), 4) for x in last[0, :, 0]] == [-0.9677, -0.771]

    totals = [round(sum(sensor), 6) for sensor in readings]
    assert totals == [-5.738759, -3.308208]
    assert [round(total / 4, 6) for total in totals] == [-1.43469, -0.827052]
    mean = MeanWindow(4)(windows)
    assert tuple(mean.shape) == (1, 2, 1)
    assert [round(float(x), 6) for x in mean[0, :, 0]] == [-1.43469, -0.827052]
    assert [round(float(x), 4) for x in mean[0, :, 0]] == [-1.4347, -0.8271]
    # T24 rises through all four cycles; T50 peaks at cycle 3.
    assert readings[0] == sorted(readings[0])
    assert max(readings[1]) == readings[1][2]

    stage = LSTMStage(window_size=4, n_nodes=2, hidden_size=8)
    assert len(stage.cells) == 2
    assert tuple(stage(windows).shape) == (1, 2, 1)

    # Seed 42 keeps 80 of the 100 training engines; another seed keeps a
    # different 80, so the rescaled readings move, and under seeds 44 and 46
    # engine 1 is held back and the README's code finds no window for it.
    kept, held = np.unique(prepared.train.units), np.unique(prepared.validation.units)
    assert (len(kept), len(held), len(np.union1d(kept, held))) == (80, 20, 100)
    for seed, first in ((43, -2.032225), (44, None), (46, None)):
        other = run_preprocessing(PreprocessingConfig(seed=seed, window_size=4), data_dir=raw_dir)
        rows = np.flatnonzero(other.train.units == 1)
        if first is None:
            assert rows.size == 0 and 1 in other.validation.units
        else:
            assert round(float(other.train.features[rows[0], pair[0], 0]), 6) == first
