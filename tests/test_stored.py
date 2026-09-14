"""Tests for finding a stored run and loading its trained network back.

A network explained from a stored run must be the network that run trained,
fed the data that run saw. Two things can silently break that: loading the
weights into a network built differently, and rebuilding the data under the
wrong seed, which would hand it a different set of held-back engines. Both are
checked here against a run written by the test itself, so the suite does not
depend on anything in reports/.
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
