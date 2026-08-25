"""Tests for the sensor definitions and the sensor selection."""

from __future__ import annotations

import pytest

from thesis_gnn_rul_xai.data.sensors import (
    PAPER_EXCLUDED_SYMBOLS,
    PAPER_SENSOR_COLUMNS,
    PAPER_SENSOR_SYMBOLS,
    SENSORS,
    column_to_symbol,
    detect_uninformative_sensors,
    symbol_to_column,
)


def test_there_are_twenty_one_sensors():
    """C-MAPSS defines exactly 21 sensors, numbered 1 to 21 without gaps."""
    assert len(SENSORS) == 21
    assert [s.index for s in SENSORS] == list(range(1, 22))


def test_kept_and_excluded_sets_partition_all_sensors():
    """Every sensor is either kept or excluded, and none is in both groups."""
    kept, excluded = set(PAPER_SENSOR_SYMBOLS), set(PAPER_EXCLUDED_SYMBOLS)
    assert len(kept) == 14
    assert len(excluded) == 7
    assert kept.isdisjoint(excluded)
    assert kept | excluded == {s.symbol for s in SENSORS}


def test_kept_sensor_order_matches_the_paper_table():
    """The node order must match Table 3, because the graph is indexed by it.

    If this test fails, the sensor axis of the data no longer lines up with the
    rows of the adjacency matrix, and every graph operation silently connects
    the wrong sensors.
    """
    assert PAPER_SENSOR_SYMBOLS[0] == "T24"
    assert PAPER_SENSOR_SYMBOLS[-1] == "W32"
    assert PAPER_SENSOR_COLUMNS[0] == "sensor_2"
    assert PAPER_SENSOR_COLUMNS[-1] == "sensor_21"


def test_symbol_and_column_translation_round_trips():
    for symbol in PAPER_SENSOR_SYMBOLS:
        assert column_to_symbol(symbol_to_column(symbol)) == symbol


def test_unknown_symbol_raises_a_helpful_error():
    with pytest.raises(KeyError, match="Unknown sensor symbol"):
        symbol_to_column("not_a_sensor")


def test_constant_sensors_are_detected(loaded):
    """The synthetic data writes the seven excluded sensors as flat lines.

    Detecting exactly those seven confirms that the diagnostic works, and gives
    the notebook a meaningful check to run against the real dataset.
    """
    report = detect_uninformative_sensors(loaded.train)
    assert set(report["constant"]) == set(PAPER_EXCLUDED_SYMBOLS)
    assert set(report["informative"]) == set(PAPER_SENSOR_SYMBOLS)
