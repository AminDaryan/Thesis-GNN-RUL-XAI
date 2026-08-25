"""Tests for the exponentially weighted moving average filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_COLUMNS
from thesis_gnn_rul_xai.data.smoothing import apply_ewma, ewma_series


def test_matches_the_recursive_formula_by_hand():
    """Check the first three values against the formula in the paper."""
    values = [10.0, 20.0, 30.0]
    alpha = 0.5
    expected = [10.0, 0.5 * 20 + 0.5 * 10, 0.5 * 30 + 0.5 * 15]
    assert ewma_series(values, alpha).tolist() == pytest.approx(expected)


def test_alpha_of_one_leaves_the_signal_unchanged():
    """With alpha = 1 the filter keeps only the current reading."""
    values = np.array([3.0, -1.0, 7.5, 2.0])
    assert ewma_series(values, 1.0).tolist() == pytest.approx(values.tolist())


def test_smoothing_reduces_variation():
    """A smoothed noisy signal must vary less than the raw one."""
    rng = np.random.default_rng(0)
    noisy = rng.normal(size=500)
    assert ewma_series(noisy, 0.2).std() < noisy.std()


def test_the_filter_only_looks_backwards(loaded):
    """Changing a future reading must not change any earlier smoothed value.

    This is the property that keeps the experiment honest. A filter that used
    future readings would leak information about the future into the present
    and make the model look better than it is.
    """
    columns = list(PAPER_SENSOR_COLUMNS)
    original = loaded.train.copy()

    modified = original.copy()
    unit_one = modified["unit"] == 1
    last_index = modified.index[unit_one][-1]
    modified.loc[last_index, columns[0]] += 1000.0

    smoothed_original = apply_ewma(original, columns, alpha=0.5)
    smoothed_modified = apply_ewma(modified, columns, alpha=0.5)

    # Everything except the very last row of engine 1 must be identical.
    mask = smoothed_original.index != last_index
    np.testing.assert_allclose(
        smoothed_original.loc[mask, columns[0]].to_numpy(),
        smoothed_modified.loc[mask, columns[0]].to_numpy(),
    )


def test_engines_are_smoothed_independently(loaded):
    """The first reading of an engine must survive the filter untouched.

    If the filter ran across engine boundaries, the first value of engine 2
    would be blended with the last value of engine 1.
    """
    columns = list(PAPER_SENSOR_COLUMNS)
    smoothed = apply_ewma(loaded.train, columns, alpha=0.3)
    for unit, history in loaded.train.groupby("unit"):
        raw_first = history[columns[0]].iloc[0]
        smoothed_first = smoothed[smoothed["unit"] == unit][columns[0]].iloc[0]
        assert smoothed_first == pytest.approx(raw_first)


def test_labels_are_not_touched(loaded):
    columns = list(PAPER_SENSOR_COLUMNS)
    smoothed = apply_ewma(loaded.train, columns, alpha=0.5)
    pd.testing.assert_series_equal(
        smoothed.sort_index()["rul"], loaded.train.sort_index()["rul"]
    )


def test_none_disables_the_filter(loaded):
    columns = list(PAPER_SENSOR_COLUMNS)
    unchanged = apply_ewma(loaded.train, columns, alpha=None)
    pd.testing.assert_frame_equal(unchanged, loaded.train)


def test_invalid_alpha_is_rejected():
    with pytest.raises(ValueError, match=r"alpha must lie in \(0, 1\]"):
        ewma_series([1.0, 2.0], alpha=0.0)
