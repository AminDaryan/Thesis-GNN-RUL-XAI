"""Tests for cutting histories into windows and splitting engines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_COLUMNS
from thesis_gnn_rul_xai.data.windowing import (
    make_last_windows,
    make_windows,
    split_units_by_id,
)

COLUMNS = list(PAPER_SENSOR_COLUMNS)


def test_window_shape_is_nodes_then_every_cycle(loaded):
    """The array is (examples, sensors, window) -- one block of cycles per sensor.

    Table 4's [batch, 14, 1] is what the FIRST GRAPH LAYER sees, not what the
    pipeline produces. Reducing the window to one number is the temporal
    stage's job, in models/temporal/, because which cycle a node carries is a
    question about the model. See A-07.
    """
    dataset = make_windows(loaded.train, COLUMNS, window_size=20)
    assert dataset.features.ndim == 3
    assert dataset.features.shape[1] == 14
    assert dataset.features.shape[2] == 20
    assert dataset.features.dtype == np.float32


def test_the_last_column_is_the_last_cycle_of_the_window(loaded):
    """The window ends where it should, so LastCycle picks the right reading.

    A-07's stand-in takes features[:, :, -1]. This checks the pipeline puts the
    end cycle there, which is what makes that stage mean what it says.
    """
    window = 20
    dataset = make_windows(loaded.train, COLUMNS, window_size=window, stride=1)
    first_engine = loaded.train.sort_values(["unit", "cycle"])
    first_engine = first_engine[first_engine["unit"] == dataset.units[0]]
    expected = first_engine[COLUMNS].to_numpy(np.float32)[window - 1]
    np.testing.assert_allclose(dataset.features[0, :, -1], expected, rtol=1e-6)


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
    """Every cycle in the block must be the real reading from that flight."""
    window = 20
    dataset = make_windows(loaded.train, COLUMNS, window_size=window)
    unit = int(dataset.units[5])
    end_cycle = int(dataset.end_cycles[5])
    history = loaded.train[loaded.train["unit"] == unit].sort_values("cycle")
    rows = history[
        (history["cycle"] > end_cycle - window) & (history["cycle"] <= end_cycle)
    ]
    # The pipeline stores a sensor's cycles along the last axis, so the source
    # rows -- one per cycle -- are the transpose of what it holds.
    expected = rows[COLUMNS].to_numpy(dtype=np.float32).T
    np.testing.assert_allclose(dataset.features[5], expected, rtol=1e-6)


def test_test_split_yields_exactly_one_window_per_engine(loaded):
    dataset = make_last_windows(loaded.test, COLUMNS, window_size=20)
    assert len(dataset) == loaded.test["unit"].nunique()
    assert len(np.unique(dataset.units)) == len(dataset)


def test_a_short_history_is_dropped_rather_than_padded(loaded):
    """An engine shorter than the window is left out, not filled in.

    ASSUMPTION A-23. Padding it would put readings into the record that the
    engine never produced. Dida, Cheriet and Belhadj exclude such engines too.
    """
    short = loaded.test[loaded.test["unit"] == 1].head(5).copy()
    with pytest.raises(ValueError, match="left out under A-23"):
        make_last_windows(short, COLUMNS, window_size=20)


def test_a_short_engine_is_dropped_while_the_others_survive(loaded):
    """Only the too-short engine goes; the rest of the frame is untouched."""
    two = loaded.test[loaded.test["unit"].isin([1, 2])].copy()
    trimmed = pd.concat([
        two[two["unit"] == 1].head(5),      # 5 rows, below the window
        two[two["unit"] == 2],              # full history
    ])

    dataset = make_last_windows(trimmed, COLUMNS, window_size=20)

    assert set(np.unique(dataset.units)) == {2}
    assert len(dataset) == 1


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
