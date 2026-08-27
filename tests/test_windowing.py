"""Tests for cutting histories into windows and splitting engines."""

from __future__ import annotations

import numpy as np
import pytest

from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_COLUMNS
from thesis_gnn_rul_xai.data.windowing import (
    make_last_windows,
    make_windows,
    split_units_by_id,
)

COLUMNS = list(PAPER_SENSOR_COLUMNS)


def test_window_shape_is_nodes_then_time(loaded):
    """The array must be (examples, sensors, time) so sensors are the nodes."""
    dataset = make_windows(loaded.train, COLUMNS, window_size=20)
    assert dataset.features.ndim == 3
    assert dataset.features.shape[1] == 14
    assert dataset.features.shape[2] == 20
    assert dataset.features.dtype == np.float32


def test_last_step_mode_gives_one_feature_per_sensor(loaded):
    dataset = make_windows(
        loaded.train, COLUMNS, window_size=20, node_feature_mode="last_step"
    )
    assert dataset.features.shape[1:] == (14, 1)


def test_the_number_of_windows_follows_the_formula(loaded):
    """For each engine there are (length - window + 1) windows at stride 1."""
    window = 20
    dataset = make_windows(loaded.train, COLUMNS, window_size=window, stride=1)
    lengths = loaded.train.groupby("unit").size()
    expected = int((lengths - window + 1).clip(lower=1).sum())
    assert len(dataset) == expected


def test_the_label_belongs_to_the_last_cycle_of_the_window(loaded):
    """Each window is labelled with the remaining life at its final cycle."""
    dataset = make_windows(loaded.train, COLUMNS, window_size=20)
    lookup = loaded.train.set_index(["unit", "cycle"])["rul"]
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        key = (int(dataset.units[index]), int(dataset.end_cycles[index]))
        assert dataset.targets[index] == pytest.approx(float(lookup.loc[key]))


def test_window_content_matches_the_source_rows(loaded):
    """The numbers inside a window must be the actual sensor readings."""
    dataset = make_windows(loaded.train, COLUMNS, window_size=20)
    unit = int(dataset.units[5])
    end_cycle = int(dataset.end_cycles[5])
    history = loaded.train[loaded.train["unit"] == unit]
    expected = (
        history[history["cycle"].between(end_cycle - 19, end_cycle)][COLUMNS]
        .to_numpy(dtype=np.float32)
        .T
    )
    np.testing.assert_allclose(dataset.features[5], expected, rtol=1e-6)


def test_test_split_yields_exactly_one_window_per_engine(loaded):
    dataset = make_last_windows(loaded.test, COLUMNS, window_size=20)
    assert len(dataset) == loaded.test["unit"].nunique()
    assert len(np.unique(dataset.units)) == len(dataset)


def test_a_short_history_is_padded_rather_than_dropped(loaded):
    """An engine shorter than the window must still produce one example."""
    short = loaded.test[loaded.test["unit"] == 1].head(5).copy()
    dataset = make_last_windows(short, COLUMNS, window_size=20)
    assert len(dataset) == 1
    assert dataset.features.shape[2] == 20


def test_engine_split_is_disjoint_and_complete():
    train, validation = split_units_by_id(range(1, 101), validation_fraction=0.2, seed=1)
    assert len(train) == 80
    assert len(validation) == 20
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(range(1, 101))


def test_engine_split_is_repeatable():
    first = split_units_by_id(range(1, 101), validation_fraction=0.2, seed=3)
    second = split_units_by_id(range(1, 101), validation_fraction=0.2, seed=3)
    np.testing.assert_array_equal(first[1], second[1])


def test_zero_fraction_gives_an_empty_validation_set():
    train, validation = split_units_by_id(range(1, 11), validation_fraction=0.0)
    assert len(train) == 10
    assert len(validation) == 0


def test_no_engine_appears_in_both_splits_after_subsetting(loaded):
    """The end-to-end guarantee: windows never cross the split boundary."""
    dataset = make_windows(loaded.train, COLUMNS, window_size=20)
    train_units, validation_units = split_units_by_id(
        np.unique(dataset.units), validation_fraction=0.25, seed=0
    )
    train_part = dataset.subset_by_units(train_units)
    validation_part = dataset.subset_by_units(validation_units)

    assert set(train_part.units.tolist()).isdisjoint(validation_part.units.tolist())
    assert len(train_part) + len(validation_part) == len(dataset)
