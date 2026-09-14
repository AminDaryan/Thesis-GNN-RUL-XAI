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
)

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
    """
    from thesis_gnn_rul_xai.models import ModelConfig, build_model

    graph = normalize_adjacency(build_physics_adjacency()[0], method="row")
    sizes = {
        stage: build_model(graph, ModelConfig(temporal=stage),
                           window_size=30, rul_max=125).count_parameters()
        for stage in ("last_cycle", "mean_window", "lstm", "raw_window")
    }

    assert sizes["last_cycle"] == 61_729
    assert sizes["mean_window"] == 61_729
    assert sizes["lstm"] == 61_729 + 14 * 4 * (32 + 32 * 32 + 32 + 32) + (32 + 1) == 124_482
    assert sizes["raw_window"] == 61_729 - 1 * 32 + 30 * 32 == 62_657


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
