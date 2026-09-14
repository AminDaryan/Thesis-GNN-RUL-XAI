"""One worked example of each function, small enough to read at a glance.

These used to live inside the source files as doctests. They were moved here
when the explanation was moved into the folder README files, so that the code
carries only a one-line description of what each function does. Nothing was
dropped: every example below is the one that used to sit in the docstring of
the function it tests.

The three examples that need PyTorch are in test_examples_torch.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_gnn_rul_xai.data.labeling import apply_piecewise_rul, piecewise_rul_curve
from thesis_gnn_rul_xai.data.sensors import (
    SENSOR_COLS,
    detect_uninformative_sensors,
)
from thesis_gnn_rul_xai.data.smoothing import ewma_series
from thesis_gnn_rul_xai.data.windowing import split_units_by_id
from thesis_gnn_rul_xai.evaluation.metrics import (
    asue,
    evaluate,
    nasa_score,
    prediction_error,
    r_squared,
    rmse,
)
from thesis_gnn_rul_xai.graph.adjacency import (
    adjacency_report,
    build_correlation_adjacency,
    build_physics_adjacency,
    normalize_adjacency,
    pairs_beyond_reach,
)

# --- data/labeling/ ---------------------------------------------------


def test_example_apply_piecewise_rul():
    """An engine with 300 cycles left is labelled 125; one with 47 keeps 47."""
    frame = pd.DataFrame({"rul": [300, 130, 125, 47, 0]})
    capped = apply_piecewise_rul(frame, rul_max=125)["rul"].tolist()
    assert capped == [125, 125, 125, 47, 0]


def test_example_piecewise_rul_curve():
    """A six-cycle engine with a ceiling of 3 is flat, then counts down."""
    assert piecewise_rul_curve(6, rul_max=3).tolist() == [3, 3, 3, 2, 1, 0]


# --- data/sensors/ ----------------------------------------------------


def test_example_detect_uninformative_sensors():
    """Sensor 1 is a flat line and sensor 2 varies, so only sensor 2 is kept."""
    frame = pd.DataFrame({column: [1.0, 1.0, 1.0] for column in SENSOR_COLS})
    frame["sensor_2"] = [1.0, 2.0, 3.0]

    report = detect_uninformative_sensors(frame)

    assert report["informative"] == ["T24"]
    assert "T2" in report["constant"]


# --- data/smoothing/ --------------------------------------------------


def test_example_ewma_series():
    """Half of each new reading, half of the smoothed value before it."""
    assert ewma_series([0.0, 1.0, 1.0], alpha=0.5).tolist() == [0.0, 0.5, 0.75]


# --- data/windowing/ --------------------------------------------------


def test_example_split_units_by_id():
    """Ten engines, one fifth held back: eight and two, sharing nothing."""
    train, held_back = split_units_by_id(
        range(1, 11), validation_fraction=0.2, seed=0
    )

    assert (len(train), len(held_back)) == (8, 2)
    assert set(train) & set(held_back) == set()


# --- evaluation/metrics.py ----------------------------------------------


def test_example_prediction_error():
    """Ten cycles short, then ten cycles long."""
    assert prediction_error([100, 50], [90, 60]).tolist() == [-10.0, 10.0]


def test_example_rmse():
    """Two errors of ten cycles and one of zero."""
    assert round(rmse([100, 50, 20], [110, 40, 20]), 4) == 8.165


def test_example_nasa_score():
    """Ten cycles late costs more than ten cycles early; perfect scores zero."""
    assert round(nasa_score([100], [110]), 4) == 1.7183
    assert round(nasa_score([100], [90]), 4) == 1.1581
    assert nasa_score([100, 50], [100, 50]) == 0.0


def test_example_r_squared():
    """A perfect fit scores 1; always predicting the mean scores 0."""
    assert r_squared([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert round(r_squared([1, 2, 3, 4], [2.5, 2.5, 2.5, 2.5]), 4) == 0.0


def test_example_asue():
    """Only the two pessimistic engines count: (10 + 20) / 2."""
    assert asue([100, 50, 30], [90, 60, 10]) == 15.0
    assert asue([100], [110]) == 0.0


def test_example_evaluate():
    """All four measures at once, plus how many engines were overestimated."""
    result = evaluate([100, 50, 30], [95, 55, 30])

    assert round(result["rmse"], 4) == 4.0825
    assert result["n_late"] == 2


# --- graph/adjacency.py -------------------------------------------------


def test_example_normalize_adjacency():
    """Two sensors joined by one edge: every entry becomes 1 / sqrt(2 * 2)."""
    result = normalize_adjacency(np.array([[0.0, 1.0], [1.0, 0.0]]))

    assert result.tolist() == [[0.5, 0.5], [0.5, 0.5]]


def test_example_normalize_adjacency_row_form_differs():
    """The two forms agree only while every sensor has the same number of
    neighbours. On a star they part company: the rows of the row form always
    add to 1, and the rows of the other form do not."""
    star = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    by_row = normalize_adjacency(star, method="row").sum(axis=1)
    symmetric = normalize_adjacency(star).sum(axis=1)

    assert [round(float(value), 6) for value in by_row] == [1.0, 1.0, 1.0]
    assert [round(float(value), 3) for value in symmetric] == [1.15, 0.908, 0.908]


def test_example_correlation_graph_keeps_the_strongest_pairs():
    """x and y rise together exactly (correlation 1); z moves against both at
    -0.4. Asked for one edge, x-y is kept. Asked for two, the tie between x-z
    and y-z goes to x-z, the pair that comes first in column order."""
    frame = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [2.0, 4.0, 6.0, 8.0],
        "z": [4.0, 1.0, 3.0, 2.0],
    })
    assert frame.corr().loc["x", "z"] == pytest.approx(-0.4)

    one, _ = build_correlation_adjacency(frame, ["x", "y", "z"], n_edges=1, symbols="xyz")
    two, _ = build_correlation_adjacency(frame, ["x", "y", "z"], n_edges=2, symbols="xyz")
    cut, _ = build_correlation_adjacency(frame, ["x", "y", "z"], threshold=0.39, symbols="xyz")

    assert one.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    assert two.tolist() == [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
    assert cut.tolist() == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]


def test_example_adjacency_report():
    """The paper's own sensor graph, measured."""
    matrix, symbols = build_physics_adjacency()

    report = adjacency_report(matrix, symbols)

    assert (report["n_edges"], report["diameter"]) == (41, 4)
    assert report["articulation_points"] == ["T24"]


def test_example_pairs_beyond_reach():
    """Three layers leave four pairs of sensors unable to reach each other."""
    matrix, symbols = build_physics_adjacency()

    assert pairs_beyond_reach(matrix, symbols, n_layers=3) == [
        ("BPR", "W32", 4),
        ("NRf", "W32", 4),
        ("Nf", "W32", 4),
        ("P15", "W32", 4),
    ]
    assert pairs_beyond_reach(matrix, symbols, n_layers=4) == []

