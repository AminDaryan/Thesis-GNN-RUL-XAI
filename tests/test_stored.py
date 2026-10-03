"""Tests for finding a stored run and loading its trained network back.

A network explained from a stored run must be the network that run trained,
fed the data that run saw. Three things can silently break that: loading the
weights into a network built differently; rebuilding the data under the wrong
seed, which would hand it a different set of held-back engines; and rebuilding
a sensor graph, the table of which sensors pass numbers to which, other than
the one the network was trained on, as recorded in the seed's graph.json or
saved inside its model.pt. All three are checked here against runs written by
the tests themselves, so the suite does not depend on anything in reports/.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.models import ModelConfig, build_model
from thesis_gnn_rul_xai.training import find_runs, load_seed, read_run

SEEDS = (7, 8)


@pytest.fixture(scope="module")
def settings() -> PreprocessingConfig:
    return PreprocessingConfig(window_size=20, validation_fraction=0.25, seed=SEEDS[0])


@pytest.fixture(scope="module")
def reports(tmp_path_factory, raw_dir, settings):
    """A folder holding one finished two-seed run, written the way train.py writes one."""
    root = tmp_path_factory.mktemp("reports")
    run = root / "fd001_physics_row_cap_2seed_varsplit_test"
    run.mkdir()
    settings.save(run / "preprocessing_config.yaml")
    ModelConfig().save(run / "model_config.yaml")
    pd.DataFrame({"seed": list(SEEDS), "rmse": [20.0, 21.0]}).to_csv(
        run / "results.csv", index=False
    )
    (run / "summary.json").write_text(json.dumps({
        "seeds": len(SEEDS), "split": "varies_with_seed",
        "temporal_stage": "last_cycle", "finished_utc": "20260914T000000Z",
    }), encoding="utf-8")
    for seed in SEEDS:
        prepared = run_preprocessing(replace(settings, seed=seed), data_dir=raw_dir)
        torch.manual_seed(seed)
        model = build_model(prepared.adjacency_normalized, ModelConfig(),
                            window_size=20, rul_max=125)
        (run / f"seed_{seed}").mkdir()
        torch.save(model.state_dict(), run / f"seed_{seed}" / "model.pt")
    (root / "not_a_run").mkdir()
    return root


def test_a_seed_loads_back_with_its_own_weights_and_its_own_split(reports, raw_dir, settings):
    run = read_run(reports / "fd001_physics_row_cap_2seed_varsplit_test")
    model, prepared = load_seed(run, 8, data_dir=raw_dir)

    torch.manual_seed(8)
    expected_data = run_preprocessing(replace(settings, seed=8), data_dir=raw_dir)
    expected = build_model(expected_data.adjacency_normalized, ModelConfig(),
                           window_size=20, rul_max=125)
    for name, tensor in expected.state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor), name

    np.testing.assert_array_equal(
        np.unique(prepared.validation.units), np.unique(expected_data.validation.units)
    )
    first_seed_data = run_preprocessing(settings, data_dir=raw_dir)
    assert set(np.unique(prepared.validation.units)) != set(
        np.unique(first_seed_data.validation.units)
    )


def test_runs_are_found_by_their_settings_and_anything_else_is_skipped(reports, settings):
    assert [run.name for run in find_runs(root=reports, preprocessing=settings)] == [
        "fd001_physics_row_cap_2seed_varsplit_test"
    ]
    assert find_runs(root=reports, preprocessing=replace(settings, graph="complete")) == []
    assert find_runs(root=reports, preprocessing=settings, seeds=5) == []
    assert find_runs(root=reports, split="fixed") == []
    assert len(find_runs(root=reports, with_weights=True)) == 1


def test_a_seed_the_run_never_trained_is_refused(reports, raw_dir):
    run = read_run(reports / "fd001_physics_row_cap_2seed_varsplit_test")
    with pytest.raises(ValueError, match="not 99"):
        load_seed(run, 99, data_dir=raw_dir)


def test_a_folder_that_is_not_a_run_is_refused(reports):
    with pytest.raises(FileNotFoundError, match="not a finished run"):
        read_run(reports / "not_a_run")


# --- a reload must rebuild the graph the network was trained on --------------


def _rewired_run(root, raw_dir, *, with_record: bool):
    """A one-seed run on a rewired graph, its network and graph.json written the way scripts/train.py writes them."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_train_script_stored", Path(__file__).resolve().parents[1] / "scripts" / "train.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    settings = PreprocessingConfig(graph="rewired", window_size=20, validation_fraction=0.25, seed=42)
    run = root / "fd001_rewired_row_cap_1seed_varsplit_test"
    (run / "seed_42").mkdir(parents=True)
    settings.save(run / "preprocessing_config.yaml")
    ModelConfig().save(run / "model_config.yaml")
    pd.DataFrame({"seed": [42], "rmse": [20.0]}).to_csv(run / "results.csv", index=False)
    (run / "summary.json").write_text(json.dumps({
        "seeds": 1, "split": "varies_with_seed", "commit": "abc1234",
        "temporal_stage": "last_cycle", "finished_utc": "20261003T000000Z",
    }), encoding="utf-8")
    prepared = run_preprocessing(settings, data_dir=raw_dir)
    torch.manual_seed(42)
    model = build_model(prepared.adjacency_normalized, ModelConfig(), window_size=20, rul_max=125)
    torch.save(model.state_dict(), run / "seed_42" / "model.pt")
    if with_record:
        (run / "seed_42" / "graph.json").write_text(
            json.dumps(script.graph_record(prepared)), encoding="utf-8")
    return read_run(run)


@pytest.mark.parametrize("with_record", [True, False], ids=["graph.json", "older run, model.pt only"])
def test_a_reload_that_rebuilds_another_graph_is_refused(tmp_path, raw_dir, monkeypatch, with_record):
    """Rewiring seed 42 with 5,000 proposals instead of 4,000 keeps 22 of the 41 connections; the reload must say so."""
    from functools import partial

    import thesis_gnn_rul_xai.data.pipeline as pipeline
    from thesis_gnn_rul_xai.graph.controls import build_rewired_adjacency

    run = _rewired_run(tmp_path, raw_dir, with_record=with_record)
    model, prepared = load_seed(run, 42, data_dir=raw_dir)
    np.testing.assert_array_equal(prepared.adjacency_raw, build_rewired_adjacency(42)[0])
    assert torch.equal(model.adjacency, torch.as_tensor(prepared.adjacency_normalized))

    # As if the code had changed its number of proposals since the run was made.
    monkeypatch.setattr(pipeline, "build_rewired_adjacency", partial(build_rewired_adjacency, proposals=5000))
    source = "recorded in seed_42/graph.json" if with_record else "saved inside seed_42/model.pt"
    with pytest.raises(ValueError, match=f"not the one {source}") as refused:
        load_seed(run, 42, data_dir=raw_dir)
    message = str(refused.value)
    assert "19 of its 41 connections are missing from the rebuild" in message
    assert "which joins 19 other pair(s) instead" in message
    assert "abc1234" in message


def test_a_graph_record_of_another_kind_or_seed_is_refused(tmp_path, raw_dir):
    """graph.json names the kind of graph and the seed it was drawn from; either one differing is refused."""
    run = _rewired_run(tmp_path, raw_dir, with_record=True)
    record_path = run.path / "seed_42" / "graph.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))

    record_path.write_text(json.dumps({**record, "drawn_from_seed": 43}), encoding="utf-8")
    with pytest.raises(ValueError, match="drawn from seed 43, but the rebuild drew it from seed 42"):
        load_seed(run, 42, data_dir=raw_dir)
    record_path.write_text(json.dumps({**record, "graph": "random"}), encoding="utf-8")
    with pytest.raises(ValueError, match="records a 'random' graph, but the run's settings rebuild a 'rewired' one"):
        load_seed(run, 42, data_dir=raw_dir)


def test_a_reload_whose_graph_was_rescaled_differently_is_refused(tmp_path, raw_dir, monkeypatch):
    """Same pairs, other rescaling: graph.json cannot see it, the table saved in model.pt can, and names the gap."""
    import thesis_gnn_rul_xai.data.pipeline as pipeline
    from thesis_gnn_rul_xai.graph.adjacency import normalize_adjacency
    from thesis_gnn_rul_xai.graph.controls import build_rewired_adjacency

    run = _rewired_run(tmp_path, raw_dir, with_record=True)
    assert run.preprocessing.adjacency_normalization == "row"

    # As if the code now rescaled every graph symmetrically, whatever the settings say.
    def symmetric(adjacency, **options):
        return normalize_adjacency(adjacency, **{**options, "method": "symmetric"})

    monkeypatch.setattr(pipeline, "normalize_adjacency", symmetric)
    graph = build_rewired_adjacency(42)[0]
    gap = float(np.abs(normalize_adjacency(graph, method="row").astype(np.float64)
                       - normalize_adjacency(graph, method="symmetric").astype(np.float64)).max())
    with pytest.raises(ValueError, match=r"joins the same sensors as the one saved inside seed_42/model\.pt") as refused:
        load_seed(run, 42, data_dir=raw_dir)
    assert f"differ from the saved ones by up to {gap:.3g}, so it was rescaled differently" in str(refused.value)


def test_a_saved_graph_of_another_size_is_refused(tmp_path, raw_dir):
    """A table saved for 13 sensors cannot be the graph of a rebuild that has 14."""
    run = _rewired_run(tmp_path, raw_dir, with_record=True)
    saved = torch.load(run.weights(42), map_location="cpu")
    # As if the network had been trained on 13 sensors: keep the saved table's first 13 rows and columns.
    saved["adjacency"] = saved["adjacency"][:13, :13].clone()
    torch.save(saved, run.weights(42))
    with pytest.raises(ValueError, match=r"seed_42/model\.pt joins 13 sensors, but the rebuilt one joins 14"):
        load_seed(run, 42, data_dir=raw_dir)


def test_a_rebuild_that_only_adds_connections_names_what_it_adds(tmp_path, raw_dir):
    """A record one connection short: nothing is missing from the rebuild, so the refusal names the pair it adds."""
    run = _rewired_run(tmp_path, raw_dir, with_record=True)
    record_path = run.path / "seed_42" / "graph.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    first, *rest = record["connections"]
    record_path.write_text(json.dumps({**record, "connections": rest}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"not the one recorded in seed_42/graph\.json") as refused:
        load_seed(run, 42, data_dir=raw_dir)
    assert f"the rebuild joins 1 pair(s) that graph does not ({first[0]}-{first[1]})." in str(refused.value)
