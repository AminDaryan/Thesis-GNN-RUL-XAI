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

# Doctests in sensors.py already cover the pure-logic branches of
# detect_uninformative_sensors; the tests here check what it reports on the
# real measurements, which is where the findings for the thesis come from.


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


def test_six_of_the_seven_dropped_sensors_are_constant(loaded):
    """Assumption A-10, checked against the data.

    Wen et al. (2026) state that "seven sensors are removed" and then name
    only six. Exactly those six are perfectly constant on FD001. The seventh,
    NRc, is not constant at all, so it was dropped for a different reason --
    redundancy with the physical core speed Nc, which they keep.
    """
    report = detect_uninformative_sensors(loaded.train)
    named_in_the_paper = {"T2", "P2", "epr", "farB", "Nf_dmd", "PCNfR_dmd"}

    assert set(report["constant"]) == named_in_the_paper
    assert "NRc" in report["informative"], (
        "NRc is not constant, so the paper's sentence conflates two different "
        "reasons for excluding a sensor. See A-10."
    )
    assert loaded.train["sensor_14"].nunique() > 1000     # NRc, in fact 6078


def test_p15_is_kept_by_the_paper_but_carries_no_information(loaded):
    """Finding F-01, checked against the data.

    The bypass-duct pressure P15 is one of the 14 sensors in the paper's
    adjacency matrix, yet it takes only two distinct values across the whole
    FD001 training set. A two-level signal cannot represent gradual wear.

    It is kept regardless, because the adjacency matrix is defined over the
    paper's fixed 14-sensor set and dropping a node would change the graph.
    This test exists so the observation cannot be lost, and so that the
    explainability stage has a sensor whose attribution should be near zero.
    """
    report = detect_uninformative_sensors(loaded.train)
    assert report["low_resolution"] == ["P15"]
    assert "P15" in PAPER_SENSOR_SYMBOLS
    assert loaded.train["sensor_6"].nunique() == 2
