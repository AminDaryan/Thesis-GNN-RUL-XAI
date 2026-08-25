"""End-to-end tests for the complete preprocessing pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from thesis_gnn_rul_xai.config import PreprocessingConfig, load_config
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing


@pytest.fixture(scope="module")
def prepared(request):
    directory = request.getfixturevalue("raw_dir")
    config = PreprocessingConfig(
        window_size=20, validation_fraction=0.25, seed=7
    )
    return run_preprocessing(config, data_dir=directory)


def test_all_three_splits_are_produced(prepared):
    assert len(prepared.train) > 0
    assert len(prepared.validation) > 0
    assert len(prepared.test) > 0


def test_training_and_validation_engines_never_overlap(prepared):
    """The core guarantee of an honest validation split."""
    train_engines = set(prepared.train.units.tolist())
    validation_engines = set(prepared.validation.units.tolist())
    assert train_engines.isdisjoint(validation_engines)


def test_validation_engines_never_influence_the_scaling(prepared):
    """The scaler must be fitted on the training engines alone.

    If the statistics were learned before the validation engines were held
    out, a held-out engine could define a minimum or a maximum. The visible
    consequence is this: with min-max scaling, the *training* windows span
    exactly [0, 1] only when the extremes belong to training engines. If a
    validation engine owned one of them, the training range would fall
    strictly inside [0, 1].

    This is the validation-side counterpart of
    ``test_normalization.py::test_test_data_never_influences_the_scaling``.
    """
    assert prepared.config.normalization.startswith("minmax")
    assert len(prepared.validation) > 0, "nothing is being held out"

    # Checked per sensor, not over the whole array. Globally the extremes of
    # one sensor mask another's, so a global range of [0, 1] would pass even
    # when a held-out engine owns a particular sensor's minimum.
    features = prepared.train.features                    # (examples, sensors, cycles)
    lowest = features.min(axis=(0, 2))
    highest = features.max(axis=(0, 2))

    offenders = [
        (symbol, float(low), float(high))
        for symbol, low, high in zip(prepared.sensor_symbols, lowest, highest)
        if abs(low) > 1e-6 or abs(high - 1.0) > 1e-6
    ]
    assert not offenders, (
        "These sensors do not span exactly [0, 1] across the training windows, "
        "which means a validation engine defined their scale: " + str(offenders)
    )


def test_an_empty_validation_split_is_usable(raw_dir):
    """A validation fraction of zero must produce an empty split, not a crash."""
    prepared = run_preprocessing(
        PreprocessingConfig(
            window_size=20, validation_fraction=0.0
        ),
        data_dir=raw_dir,
    )
    assert len(prepared.validation) == 0
    assert "empty" in repr(prepared.validation)      # must not raise
    assert prepared.summary().loc["validation", "examples"] == 0


def test_one_test_example_per_engine(prepared):
    assert len(prepared.test) == len(np.unique(prepared.test.units))


def test_labels_respect_the_ceiling(prepared):
    ceiling = prepared.config.rul_max
    for dataset in (prepared.train, prepared.validation, prepared.test):
        assert dataset.targets.max() <= ceiling
        assert dataset.targets.min() >= 0


def test_feature_axis_matches_the_graph(prepared):
    """The sensor axis and the adjacency matrix must have the same length.

    If these ever disagree, the graph would connect the wrong sensors.
    """
    n_nodes = prepared.adjacency_raw.shape[0]
    assert prepared.train.features.shape[1] == n_nodes
    assert len(prepared.sensor_symbols) == n_nodes
    assert len(prepared.train.sensor_columns) == n_nodes


def test_no_missing_or_infinite_values(prepared):
    for dataset in (prepared.train, prepared.validation, prepared.test):
        assert np.isfinite(dataset.features).all()
        assert np.isfinite(dataset.targets).all()


def test_diagnostics_record_the_important_facts(prepared):
    diagnostics = prepared.diagnostics
    assert diagnostics["graph"]["n_edges"] == 41
    assert diagnostics["graph"]["is_connected"] is True
    assert diagnostics["n_operating_conditions"] == 1
    assert diagnostics["subset"] == "FD001"
    assert diagnostics["kept_but_uninformative"] == ["P15"]   # finding F-01


def test_the_run_is_repeatable(raw_dir):
    """The same settings must give bit-for-bit identical arrays."""
    config = PreprocessingConfig(
        window_size=20, validation_fraction=0.25, seed=7
    )
    first = run_preprocessing(config, data_dir=raw_dir)
    second = run_preprocessing(config, data_dir=raw_dir)
    np.testing.assert_array_equal(first.train.features, second.train.features)
    np.testing.assert_array_equal(first.validation.units, second.validation.units)


def test_a_different_seed_changes_the_split(raw_dir):
    base = PreprocessingConfig(
        window_size=20, validation_fraction=0.25, seed=7
    )
    other = PreprocessingConfig(
        window_size=20, validation_fraction=0.25, seed=8
    )
    first = run_preprocessing(base, data_dir=raw_dir)
    second = run_preprocessing(other, data_dir=raw_dir)
    assert set(first.validation.units.tolist()) != set(second.validation.units.tolist())


def test_saving_writes_every_expected_file(prepared, tmp_path):
    directory = prepared.save(tmp_path / "processed")
    expected = {
        "windows.npz",
        "adjacency.npz",
        "normalization_statistics.csv",
        "config.yaml",
        "diagnostics.json",
    }
    assert expected <= {p.name for p in directory.iterdir()}

    # The saved settings must reload into an identical configuration object.
    assert load_config(directory / "config.yaml") == prepared.config

    # The saved arrays must match what is in memory.
    with np.load(directory / "windows.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["train_features"], prepared.train.features)
        np.testing.assert_array_equal(archive["test_targets"], prepared.test.targets)

    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    assert diagnostics["subset"] == "FD001"


def test_disabling_smoothing_changes_the_features(raw_dir):
    smoothed = run_preprocessing(
        PreprocessingConfig(window_size=20, smoothing_alpha=0.5),
        data_dir=raw_dir,
    )
    unsmoothed = run_preprocessing(
        PreprocessingConfig(window_size=20, smoothing_alpha=None),
        data_dir=raw_dir,
    )
    assert not np.allclose(smoothed.train.features, unsmoothed.train.features)


def test_last_step_mode_produces_single_feature_nodes(raw_dir):
    prepared = run_preprocessing(
        PreprocessingConfig(
            window_size=20, node_feature_mode="last_step"
        ),
        data_dir=raw_dir,
    )
    assert prepared.train.features.shape[1:] == (14, 1)
