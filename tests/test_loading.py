"""Tests for reading raw files and attaching Remaining Useful Life labels."""

from __future__ import annotations

import pandas as pd
import pytest

from thesis_gnn_rul_xai.data.loading import (
    RAW_COLUMNS,
    attach_rul,
    read_cmapss_frame,
    read_rul_file,
)


def test_loaded_frame_has_the_expected_columns(loaded):
    assert list(loaded.train.columns) == [*RAW_COLUMNS, "rul"]


def test_training_labels_reach_zero_at_the_end_of_every_engine(loaded):
    """Training histories run to failure, so the last label must be zero."""
    last = loaded.train.groupby("unit")["rul"].last()
    assert (last == 0).all()


def test_training_labels_decrease_by_one_each_cycle(loaded):
    """Between consecutive cycles the remaining life drops by exactly one."""
    for _, history in loaded.train.groupby("unit"):
        differences = history["rul"].diff().dropna().unique()
        assert differences.tolist() == [-1]


def test_test_labels_match_the_official_file_at_the_last_cycle(loaded):
    """At the final recorded cycle, the computed label must equal the label file."""
    last = loaded.test.groupby("unit")["rul"].last()
    pd.testing.assert_series_equal(
        last.sort_index(),
        loaded.test_rul.sort_index(),
        check_names=False,
        check_dtype=False,
    )


def test_missing_file_gives_an_instructive_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Getting the data"):
        read_cmapss_frame(tmp_path / "train_FD001.txt")


def test_attach_rul_rejects_an_incomplete_label_file(loaded):
    """A label file that omits an engine must fail loudly, not silently."""
    incomplete = loaded.test_rul.iloc[:-1]
    with pytest.raises(ValueError, match="No Remaining Useful Life label"):
        attach_rul(loaded.test.drop(columns="rul"), rul_at_last_cycle=incomplete)


def test_rul_file_is_indexed_from_one(synthetic_dir):
    series = read_rul_file(synthetic_dir / "RUL_FD001.txt")
    assert series.index[0] == 1
    assert series.index[-1] == len(series)
