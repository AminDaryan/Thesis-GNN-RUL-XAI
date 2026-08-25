"""Tests for the sensor graph.

The graph is the one part of Wen et al. (2026) that is fully specified, so
these tests check the transcription of their Table 3 rather than any assumption
of ours. A transcription error here would corrupt every later result while
leaving the code apparently working.
"""

from __future__ import annotations

import numpy as np
import pytest

from thesis_gnn_rul_xai.graph.adjacency import (
    adjacency_report,
    adjacency_to_frame,
    build_correlation_adjacency,
    build_fully_connected_adjacency,
    build_physics_adjacency,
    normalize_adjacency,
)


def test_the_graph_has_fourteen_nodes():
    matrix, symbols = build_physics_adjacency()
    assert matrix.shape == (14, 14)
    assert len(symbols) == 14


def test_the_graph_is_symmetric():
    """"Sensor i relates to sensor j" must mean the same in both directions."""
    matrix, _ = build_physics_adjacency()
    np.testing.assert_array_equal(matrix, matrix.T)


def test_the_graph_has_no_self_loops_before_normalisation():
    """Table 3 has a zero diagonal; self-loops are added later by the formula."""
    matrix, _ = build_physics_adjacency()
    assert np.all(np.diag(matrix) == 0)


def test_the_graph_contains_only_zeros_and_ones():
    """The paper states binary encoding: an edge is present or it is not."""
    matrix, _ = build_physics_adjacency()
    assert set(np.unique(matrix).tolist()) <= {0.0, 1.0}


def test_the_graph_matches_the_published_edge_count():
    """41 undirected edges, independently counted from the printed table."""
    matrix, symbols = build_physics_adjacency()
    report = adjacency_report(matrix, symbols)
    assert report["n_edges"] == 41
    assert report["density"] == pytest.approx(41 * 2 / (14 * 13))


def test_every_sensor_is_reachable_from_every_other():
    """A disconnected graph would split the sensors into isolated groups."""
    matrix, symbols = build_physics_adjacency()
    report = adjacency_report(matrix, symbols)
    assert report["is_connected"]
    assert report["isolated_nodes"] == []
    assert min(report["degrees"].values()) >= 1


def test_selected_edges_follow_the_stated_physical_rules():
    """Spot-check edges that the paper justifies explicitly in its Section 2.3."""
    matrix, symbols = build_physics_adjacency()
    frame = adjacency_to_frame(matrix, symbols)

    # Same engine station: total and static pressure at the HPC outlet.
    assert frame.loc["P30", "Ps30"] == 1
    # Speed correction: physical fan speed and corrected fan speed.
    assert frame.loc["Nf", "NRf"] == 1
    # Thermodynamic chain along the gas path.
    assert frame.loc["T24", "T30"] == 1
    assert frame.loc["T30", "T50"] == 1
    # Cooling system: the two turbine coolant bleeds.
    assert frame.loc["W31", "W32"] == 1
    # A pair with no direct physical relationship stated.
    assert frame.loc["T24", "T50"] == 0


def test_normalisation_matches_the_formula_on_a_tiny_graph():
    """Two nodes, one edge. With self-loops each degree is 2, so each entry is 0.5."""
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(
        normalize_adjacency(matrix), np.full((2, 2), 0.5), rtol=1e-6
    )


def test_normalisation_adds_self_loops():
    """Every diagonal entry must be positive after normalisation."""
    matrix, _ = build_physics_adjacency()
    assert np.all(np.diag(normalize_adjacency(matrix)) > 0)


def test_normalisation_keeps_the_matrix_symmetric_and_finite():
    matrix, _ = build_physics_adjacency()
    normalized = normalize_adjacency(matrix)
    np.testing.assert_allclose(normalized, normalized.T, rtol=1e-6)
    assert np.isfinite(normalized).all()


def test_normalisation_does_not_create_new_edges():
    """Only entries that were already non-zero, or on the diagonal, may be non-zero."""
    matrix, _ = build_physics_adjacency()
    normalized = normalize_adjacency(matrix)
    allowed = (matrix > 0) | np.eye(14, dtype=bool)
    assert np.all(normalized[~allowed] == 0)


def test_normalisation_rejects_an_asymmetric_matrix():
    asymmetric = np.array([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="must be symmetric"):
        normalize_adjacency(asymmetric)


def test_fully_connected_graph_has_every_possible_edge():
    matrix, symbols = build_fully_connected_adjacency(14)
    assert adjacency_report(matrix, symbols)["n_edges"] == 14 * 13 // 2


def test_correlation_graph_responds_to_the_threshold(loaded):
    from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_COLUMNS

    columns = list(PAPER_SENSOR_COLUMNS)
    loose, _ = build_correlation_adjacency(loaded.train, columns, threshold=0.1)
    strict, _ = build_correlation_adjacency(loaded.train, columns, threshold=0.95)
    assert loose.sum() >= strict.sum()
    assert np.all(np.diag(loose) == 0)
