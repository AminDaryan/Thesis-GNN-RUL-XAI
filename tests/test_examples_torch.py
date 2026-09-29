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
    from thesis_gnn_rul_xai.docsite.playground import _explanation_edges

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


def test_example_temporal_stages_on_two_hand_checkable_sensors():
    """Two sensors, four cycles each, so every number can be checked by hand.

    The three stages differ only in what they leave on a node, and all three
    hand the graph layers the same shape, which is what makes them swappable.
    """
    windows = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                             [10.0, 20.0, 30.0, 40.0]]])

    last = LastCycle(4)(windows)
    torch.testing.assert_close(last, torch.tensor([[[4.0], [40.0]]]))

    mean = MeanWindow(4)(windows)
    # (1 + 2 + 3 + 4) / 4 = 2.5 and (10 + 20 + 30 + 40) / 4 = 25.0
    torch.testing.assert_close(mean, torch.tensor([[[2.5], [25.0]]]))

    summary = WindowSummary(4)(windows)
    # Cycles numbered from the middle: t = -1.5, -0.5, +0.5, +1.5, and sum t^2 = 5.
    # Sensor 0: sum t * reading = -1.5 - 1.0 + 1.5 + 6.0 = 5, slope 5 / 5 = 1 per
    # cycle, and the change from the first cycle to the last is 1 x 3 = 3.
    # Sensor 1 is the same series times ten, so every number is ten times larger.
    torch.testing.assert_close(summary, torch.tensor([[[4.0, 2.5, 3.0],
                                                      [40.0, 25.0, 30.0]]]))

    learned = LSTMStage(window_size=4, n_nodes=2, hidden_size=8)
    assert tuple(learned(windows).shape) == (1, 2, 1)
    assert last.shape == mean.shape == learned(windows).shape


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
