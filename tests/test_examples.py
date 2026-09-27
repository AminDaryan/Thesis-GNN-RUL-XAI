"""One worked example of each function, small enough to read at a glance.

These used to live inside the source files as doctests. They were moved here
when the explanation was moved into the folder README files, so that the code
carries only a one-line description of what each function does. Nothing was
dropped: every example below is the one that used to sit in the docstring of
the function it tests.

The three examples that need PyTorch are in test_examples_torch.py.
"""

from __future__ import annotations

import re

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
from thesis_gnn_rul_xai.evaluation.published import SUBSETS, published_row
from thesis_gnn_rul_xai.graph.adjacency import (
    adjacency_report,
    build_correlation_adjacency,
    build_physics_adjacency,
    normalize_adjacency,
    pairs_beyond_reach,
)
from thesis_gnn_rul_xai.utils.provenance import current_commit, utc_stamp

# --- docsite/ ---------------------------------------------------------


@pytest.mark.parametrize("changes", [{}, {"smoothing_alpha": 0.25},
                                    {"smoothing": False, "window_size": 25, "window_stride": 2,
                                        "validation_fraction": 0.0, "rul_max": 150, "seed": 7}])
def test_example_preprocessing_story(loaded, changes):
    """The diagram's rows, changed readings and overlapping windows come from the real FD001 files."""
    from thesis_gnn_rul_xai.docsite import playground

    settings = playground.validate_settings(changes)
    answer = playground.stage_detail("preprocessing", settings)
    opening = answer["steps"][0]
    assert opening["handoff"] == "20,631 rows · 27 columns"
    sample = opening["example"]["rows"]
    assert sample == [{"engine": 1, "cycle": 1, "reading": 641.82, "life": 191},
                      {"engine": 1, "cycle": 2, "reading": 642.15, "life": 190},
                      {"engine": 1, "cycle": 3, "reading": 642.35, "life": 189}]
    assert sample[1]["reading"] == loaded.train.loc[1, "sensor_2"]
    stages = {step.get("picture"): step for step in answer["steps"]}
    assert stages["cap"]["example"]["after"] == f"{min(190, settings['rul_max'])} cycles"
    if settings["smoothing"]:
        alpha = settings["smoothing_alpha"]
        expected = alpha * 642.15 + (1 - alpha) * 641.82
        assert float(stages["smooth"]["example"]["after"]) == pytest.approx(expected)
        assert stages["smooth"]["example"]["note"] == (
            f"{alpha:g} x current reading + {1 - alpha:g} x previous smoothed reading")
        assert stages["rescale"]["example"]["before"] == f"{expected:.3f}"
        if alpha == 0.5:
            assert stages["rescale"]["example"]["after"] == "-1.6039"
    else:
        assert "smooth" not in stages
        assert stages["rescale"]["example"]["before"] == "642.15"
    groups = stages["hold"]["example"]["items"]
    assert sum(int(group["detail"].split()[0].replace(",", "")) for group in groups) == len(loaded.train)
    windows = stages["window"]["example"]["windows"]
    for index, item in enumerate(windows):
        start = 1 + index * settings["window_stride"]
        end = start + settings["window_size"] - 1
        assert item == {"start": start, "end": end, "life": min(192 - end, settings["rul_max"])}


def test_example_preprocessing_without_files(monkeypatch):
    """Missing NASA files produce descriptions without invented readings or counts."""
    from thesis_gnn_rul_xai.docsite import playground

    monkeypatch.setattr(playground, "_files_present", lambda subset: False)
    answer = playground.stage_detail("preprocessing", playground.validate_settings({}))
    assert "not in data/raw/" in answer["note"]
    assert all("example" not in step for step in answer["steps"])
    assert answer["steps"][0]["handoff"] == "Rows · 27 columns"


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


# --- evaluation/published.py --------------------------------------------


def test_example_published_row_prints_only_what_the_table_prints():
    """Table 11 gives the model without its LSTM stage an RMSE and nothing else.

    Its average across the four sub-datasets and its difference from the full
    model are the two figures Table 11 prints beside it.
    """
    without = published_row("without_lstm", "FD001")
    assert (without.rmse, without.score, without.r2, without.asue) == (14.93, None, None, None)

    rows = [published_row("without_lstm", subset).rmse for subset in SUBSETS]
    assert rows == [14.93, 23.34, 16.25, 24.21]
    assert round(sum(rows) / 4, 2) == 19.68
    assert round(without.rmse - published_row("full", "FD001").rmse, 2) == 3.15

    with pytest.raises(ValueError, match="no published row called 'transformer'"):
        published_row("transformer", "FD001")


# --- utils/provenance.py ------------------------------------------------


def test_example_provenance_stamps_have_the_form_folder_names_use():
    """A short commit, marked -dirty when the files differ from it; UTC to the second."""
    assert re.fullmatch(r"[0-9a-f]{7,}(-dirty)?|unknown", current_commit())
    stamp = utc_stamp()
    assert re.fullmatch(r"\d{8}T\d{6}Z", stamp)
    assert ":" not in stamp


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

