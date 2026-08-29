"""The table of which sensors are wired to which, and the tools to inspect it.

Read README.md in this folder. It explains everything this file does.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd

from ..data.sensors import PAPER_SENSOR_SYMBOLS
from ..utils.logging_setup import get_logger

__all__ = [
    "PHYSICS_ADJACENCY_ROWS",
    "AdjacencyNormalization",
    "adjacency_report",
    "adjacency_to_frame",
    "build_correlation_adjacency",
    "build_fully_connected_adjacency",
    "build_physics_adjacency",
    "normalize_adjacency",
    "pairs_beyond_reach",
]

logger = get_logger(__name__)

# The two ways of rescaling the table before the network uses it. README.md.
AdjacencyNormalization = Literal["symmetric", "row"]

# Table 3 of Wen et al. (2026), copied out row by row. The row order is the
# sensor order in PAPER_SENSOR_SYMBOLS and must not change.
PHYSICS_ADJACENCY_ROWS: dict[str, tuple[int, ...]] = {
    #          T24 T30 T50 P15 P30  Nf  Nc Ps30 phi NRf BPR htB W31 W32
    "T24":     (0,  1,  0,  1,  1,  1,  0,  1,  0,  1,  1,  0,  0,  0),
    "T30":     (1,  0,  1,  0,  1,  0,  1,  1,  1,  0,  0,  1,  1,  0),
    "T50":     (0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1),
    "P15":     (1,  0,  0,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0,  0),
    "P30":     (1,  1,  0,  0,  0,  0,  1,  1,  1,  0,  0,  1,  1,  0),
    "Nf":      (1,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0),
    "Nc":      (0,  1,  1,  0,  1,  0,  0,  1,  1,  0,  0,  1,  1,  1),
    "Ps30":    (1,  1,  0,  0,  1,  0,  1,  0,  1,  0,  0,  1,  1,  0),
    "phi":     (0,  1,  0,  0,  1,  0,  1,  1,  0,  0,  0,  1,  1,  0),
    "NRf":     (1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0),
    "BPR":     (1,  0,  0,  1,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0),
    "htBleed": (0,  1,  0,  0,  1,  0,  1,  1,  1,  0,  0,  0,  1,  1),
    "W31":     (0,  1,  1,  0,  1,  0,  1,  1,  1,  0,  0,  1,  0,  1),
    "W32":     (0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  1,  1,  0),
}


def build_physics_adjacency() -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the paper's own sensor graph, and the sensor name of each row."""
    symbols = PAPER_SENSOR_SYMBOLS
    matrix = np.array(
        [PHYSICS_ADJACENCY_ROWS[symbol] for symbol in symbols],
        dtype=np.float32,
    )
    return matrix, symbols


def build_fully_connected_adjacency(
    n_nodes: int = 14,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return a graph where every sensor is connected to every other one.

    ASSUMPTION A-19: use this as the stand-in when the paper's experiment
    "without physics prior" is repeated. The paper never says what it put in
    place of its own graph there. Explained in README.md in this folder;
    written up in docs/ASSUMPTIONS.md.
    """
    matrix = np.ones((n_nodes, n_nodes), dtype=np.float32)
    np.fill_diagonal(matrix, 0.0)
    return matrix, PAPER_SENSOR_SYMBOLS[:n_nodes]


def build_correlation_adjacency(
    frame: pd.DataFrame,
    sensor_columns: Sequence[str],
    *,
    threshold: float = 0.5,
    symbols: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Connect sensors whose readings move together. Pass training data only."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}.")

    correlation = frame[list(sensor_columns)].corr(method="pearson").to_numpy()
    matrix = (np.abs(correlation) >= threshold).astype(np.float32)
    np.fill_diagonal(matrix, 0.0)

    labels = tuple(symbols) if symbols is not None else PAPER_SENSOR_SYMBOLS
    logger.info(
        "Built a correlation graph with threshold %.2f: %d undirected edge(s).",
        threshold, int(matrix.sum() // 2),
    )
    return matrix, labels


def normalize_adjacency(
    adjacency: np.ndarray,
    *,
    add_self_loops: bool = True,
    method: AdjacencyNormalization = "symmetric",
) -> np.ndarray:
    """Rescale the table so the network can use it. Adds each sensor to itself.

    ASSUMPTION A-21: choose between the two rescaling formulas below. The paper
    describes the step but never prints the formula, and both fit what it says.
    Explained in README.md in this folder; written up in docs/ASSUMPTIONS.md.
    """
    matrix = np.asarray(adjacency, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape {matrix.shape}.")
    if not np.allclose(matrix, matrix.T):
        raise ValueError(
            "Adjacency must be symmetric. An asymmetric matrix would mean "
            "sensor i is a neighbour of j but not the other way round, which "
            "the symmetric normalisation formula does not support."
        )

    if add_self_loops:
        matrix = matrix + np.eye(matrix.shape[0])

    degrees = matrix.sum(axis=1)
    if np.any(degrees <= 0):
        isolated = np.where(degrees <= 0)[0].tolist()
        raise ValueError(
            f"Node(s) {isolated} have no connections at all, so the "
            "normalisation would divide by zero. Add self-loops or edges."
        )

    if method == "symmetric":
        inverse_sqrt = np.diag(1.0 / np.sqrt(degrees))
        normalized = inverse_sqrt @ matrix @ inverse_sqrt
    elif method == "row":
        normalized = matrix / degrees[:, None]
    else:
        raise ValueError(
            f"method must be 'symmetric' or 'row', got {method!r}."
        )
    return normalized.astype(np.float32)


def to_networkx(
    adjacency: np.ndarray,
    symbols: Sequence[str],
) -> nx.Graph:
    """Hand the graph to the networkx library, with the sensors named."""
    matrix = np.asarray(adjacency, dtype=float).copy()
    np.fill_diagonal(matrix, 0.0)
    graph = nx.from_numpy_array(matrix)
    return nx.relabel_nodes(graph, dict(enumerate(symbols)))


def adjacency_report(
    adjacency: np.ndarray,
    symbols: Sequence[str],
) -> dict[str, object]:
    """Measure the shape of a graph. Returns a dictionary; README.md lists it."""
    matrix = np.asarray(adjacency)
    n_nodes = matrix.shape[0]
    degrees = matrix.sum(axis=1).astype(int)
    graph = to_networkx(matrix, symbols)

    connected = nx.is_connected(graph) if n_nodes else False
    n_possible = n_nodes * (n_nodes - 1)

    report: dict[str, object] = {
        "n_nodes": n_nodes,
        "n_edges": graph.number_of_edges(),
        "density": float(matrix.sum() / n_possible) if n_possible else 0.0,
        "is_symmetric": bool(np.allclose(matrix, matrix.T)),
        "has_self_loops": bool(np.any(np.diag(matrix) != 0)),
        "is_connected": connected,
        "degrees": dict(zip(symbols, degrees.tolist(), strict=True)),
        "isolated_nodes": [symbols[i] for i in np.flatnonzero(degrees == 0)],
        "articulation_points": sorted(nx.articulation_points(graph)),
    }

    if connected:
        lengths = dict(nx.all_pairs_shortest_path_length(graph))
        counts: dict[int, int] = {}
        for source, reachable in lengths.items():
            for target, distance in reachable.items():
                if source < target:
                    counts[distance] = counts.get(distance, 0) + 1
        report["diameter"] = nx.diameter(graph)
        report["distance_histogram"] = dict(sorted(counts.items()))
        report["algebraic_connectivity"] = float(nx.algebraic_connectivity(graph))
    else:
        report["diameter"] = None
        report["distance_histogram"] = None
        report["algebraic_connectivity"] = None

    return report


def pairs_beyond_reach(
    adjacency: np.ndarray,
    symbols: Sequence[str],
    n_layers: int,
) -> list[tuple[str, str, int]]:
    """Sensor pairs too far apart for this many layers to connect at all."""
    if n_layers < 1:
        raise ValueError(f"n_layers must be at least 1, got {n_layers}.")

    graph = to_networkx(adjacency, symbols)
    out: list[tuple[str, str, int]] = []
    for source, reachable in nx.all_pairs_shortest_path_length(graph):
        for target, distance in reachable.items():
            if source < target and distance > n_layers:
                out.append((source, target, distance))
    return sorted(out, key=lambda item: (item[2], item[0], item[1]))


def adjacency_to_frame(
    adjacency: np.ndarray,
    symbols: Sequence[str],
) -> pd.DataFrame:
    """Return the matrix as a labelled table, for display in a notebook."""
    return pd.DataFrame(
        np.asarray(adjacency), index=list(symbols), columns=list(symbols)
    )


# Verified at import time. A silent transcription error in Table 3 would
# corrupt every downstream result while leaving the code apparently working,
# so it is checked here rather than left to a test that might not be run.
_matrix, _symbols = build_physics_adjacency()
assert _matrix.shape == (14, 14), "The physics graph must have 14 nodes."
assert np.allclose(_matrix, _matrix.T), (
    "The transcribed Table 3 is not symmetric: at least one edge was entered "
    "in one direction only. Check PHYSICS_ADJACENCY_ROWS against the paper."
)
assert np.all(np.diag(_matrix) == 0), "Table 3 has a zero diagonal."
assert adjacency_report(_matrix, _symbols)["is_connected"], (
    "The physics graph must be connected."
)
del _matrix, _symbols
