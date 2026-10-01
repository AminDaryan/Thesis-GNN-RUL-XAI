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
    connections_of,
    normalize_adjacency,
    pairs_beyond_reach,
)


def test_connections_are_listed_once_in_sensor_order():
    matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    assert connections_of(matrix) == [(0, 1), (0, 2)]


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


def test_the_matrix_matches_the_papers_own_construction_rules_except_one():
    """Finding F-07, pinned down.

    Section 2.3.1 gives five rules for building the matrix, each with a worked
    example. Twenty-three of the twenty-four edges those examples require are
    present. The exception is ``phi``-``W32``: the text says phi "subsequently
    affects htBleed, W31, and W32", but the entry is 0 in Table 3 and in the
    Fig. 3 colour grid alike.

    This project follows the matrix rather than the text, because the matrix is
    what the model consumes. The test exists so that the discrepancy is a
    recorded, checkable fact rather than a footnote that could drift.
    """
    matrix, symbols = build_physics_adjacency()
    frame = adjacency_to_frame(matrix, symbols)

    required = [
        ("P30", "Ps30"), ("T30", "P30"), ("T30", "Ps30"),      # rule 1
        ("P15", "BPR"), ("W31", "W32"),
        ("Nf", "NRf"),                                          # rule 2
        ("T24", "T30"), ("T30", "T50"),                         # rule 3a
        ("P15", "Nf"), ("P15", "NRf"), ("P15", "BPR"),          # rule 3b
        ("Nf", "NRf"), ("Nf", "BPR"), ("NRf", "BPR"),
        ("phi", "T30"), ("phi", "P30"), ("phi", "Ps30"),        # rule 4
        ("phi", "htBleed"), ("phi", "W31"), ("phi", "W32"),
        ("W31", "W32"), ("W31", "htBleed"), ("W32", "htBleed"), # rule 5
    ]
    missing = [(a, b) for a, b in required if frame.loc[a, b] != 1]

    assert missing == [("phi", "W32")], (
        "The set of rule-implied edges absent from the matrix has changed. "
        f"Expected only phi-W32, got {missing}. See finding F-07."
    )


def test_rule_3b_makes_the_fan_sensors_a_complete_subgraph():
    """All six pairs among P15, NF, NRf and BPR are connected.

    The paper's third rule mentions "pressure connections between P15, NF, NRf,
    and BPR" without saying they form a clique, but the matrix makes them one.
    It is why those four sensors behave as a single tightly coupled group.
    """
    matrix, symbols = build_physics_adjacency()
    frame = adjacency_to_frame(matrix, symbols)
    group = ["P15", "Nf", "NRf", "BPR"]
    for i, a in enumerate(group):
        for b in group[i + 1:]:
            assert frame.loc[a, b] == 1, f"{a}-{b} is missing from the fan group"


def test_the_graph_has_a_diameter_of_four():
    """Two sensors are four edges apart, which bounds what three layers can do."""
    matrix, symbols = build_physics_adjacency()
    report = adjacency_report(matrix, symbols)
    assert report["diameter"] == 4
    assert report["distance_histogram"] == {1: 41, 2: 25, 3: 21, 4: 4}
    assert sum(report["distance_histogram"].values()) == 14 * 13 // 2


def test_three_layers_leave_four_pairs_unreachable():
    """The architectural consequence of that diameter.

    Table 4 stacks three graph convolution layers, and one layer moves
    information one edge. Every pair further apart than three therefore cannot
    influence each other, whatever the network learns. On this graph that is
    four pairs, all of them involving W32.
    """
    matrix, symbols = build_physics_adjacency()
    beyond = pairs_beyond_reach(matrix, symbols, n_layers=3)
    assert beyond == [
        ("BPR", "W32", 4), ("NRf", "W32", 4), ("Nf", "W32", 4), ("P15", "W32", 4),
    ]
    assert pairs_beyond_reach(matrix, symbols, n_layers=4) == []


def test_t24_is_the_only_articulation_point():
    """Removing T24 would split the sensors into a fan group and a core group.

    Every message between the two halves therefore passes through it. That is
    a fact about the published graph, and the kind of structure a graph-native
    explanation can express but node-level attribution cannot.
    """
    matrix, symbols = build_physics_adjacency()
    report = adjacency_report(matrix, symbols)
    assert report["articulation_points"] == ["T24"]
    assert 0 < report["algebraic_connectivity"] < 1      # connected, but weakly


def test_a_disconnected_graph_reports_no_diameter():
    """Distance is undefined between components, so the fields are None."""
    matrix = np.zeros((4, 4))
    matrix[0, 1] = matrix[1, 0] = 1
    matrix[2, 3] = matrix[3, 2] = 1
    report = adjacency_report(matrix, ["a", "b", "c", "d"])
    assert report["is_connected"] is False
    assert report["diameter"] is None
    assert report["algebraic_connectivity"] is None


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


def test_correlation_graph_keeps_exactly_the_strongest_pairs_asked_for(loaded):
    """A-27: as many edges as asked, and every kept pair at least as strong as every dropped one."""
    from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_COLUMNS

    columns = list(PAPER_SENSOR_COLUMNS)
    matrix, _ = build_correlation_adjacency(loaded.train, columns, n_edges=41)
    assert int(matrix.sum() // 2) == 41
    np.testing.assert_array_equal(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0)

    strength = np.abs(np.nan_to_num(loaded.train[columns].corr().to_numpy()))
    upper = np.triu(np.ones((14, 14), dtype=bool), k=1)
    kept = strength[upper & (matrix == 1)]
    dropped = strength[upper & (matrix == 0)]
    assert kept.min() >= dropped.max()


def test_correlation_graph_needs_exactly_one_rule(loaded):
    from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_COLUMNS

    columns = list(PAPER_SENSOR_COLUMNS)
    with pytest.raises(ValueError, match="exactly one"):
        build_correlation_adjacency(loaded.train, columns)
    with pytest.raises(ValueError, match="exactly one"):
        build_correlation_adjacency(loaded.train, columns, threshold=0.5, n_edges=41)
    with pytest.raises(ValueError, match="n_edges"):
        build_correlation_adjacency(loaded.train, columns, n_edges=92)
