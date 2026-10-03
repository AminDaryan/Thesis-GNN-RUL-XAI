"""Sensor graphs with the paper's 41 connections but none of its engineering knowledge.

Read README.md in this folder. It explains everything this file does.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np

from ..data.sensors import PAPER_SENSOR_SYMBOLS
from ..utils.logging_setup import get_logger
from .adjacency import build_physics_adjacency, connections_of, to_networkx

__all__ = [
    "GRAPH_DRAW_TAG",
    "REWIRING_PROPOSALS",
    "build_random_adjacency",
    "build_rewired_adjacency",
    "graph_generator",
    "swap_ends",
]

logger = get_logger(__name__)

# ASSUMPTION A-41: a control graph is drawn from the run's seed together with
# this tag, not from the seed alone, because the seed alone already starts the
# generator that chooses the held-back engines; sharing it would make the graph
# and the split two readings of the same random numbers.
GRAPH_DRAW_TAG = 1

# ASSUMPTION A-41: how many swaps the rewiring proposes before it stops. Chosen
# by measuring how many of Table 3's connections survive: the count stops
# falling well before this many proposals. README.md gives the measurement.
REWIRING_PROPOSALS = 4000


def graph_generator(seed: int) -> np.random.Generator:
    """The random number generator a control graph is drawn from, for one seed."""
    return np.random.default_rng([int(seed), GRAPH_DRAW_TAG])


def _matrix_from(edges: Sequence[tuple[int, int]], n_nodes: int) -> np.ndarray:
    """A symmetric grid of zeros and ones with a one at both ends of every connection."""
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i, j in edges:
        matrix[i, j] = matrix[j, i] = 1.0
    return matrix


def _is_connected(matrix: np.ndarray) -> bool:
    """Whether every sensor can reach every other one along connections."""
    return nx.is_connected(to_networkx(matrix, list(range(matrix.shape[0]))))


def build_random_adjacency(
    seed: int,
    *,
    n_edges: int | None = None,
    n_nodes: int = 14,
    connected: bool = True,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Join a number of sensor pairs chosen at random, by default as many as Table 3 joins."""
    if n_edges is None:
        n_edges = int(build_physics_adjacency()[0].sum() // 2)
    rows, cols = np.triu_indices(n_nodes, k=1)
    if not 0 <= n_edges <= len(rows):
        raise ValueError(
            f"n_edges must lie in [0, {len(rows)}] for {n_nodes} sensors, got {n_edges}."
        )
    if connected and n_edges < n_nodes - 1:
        raise ValueError(
            f"{n_edges} connections cannot join {n_nodes} sensors into one piece; "
            f"that needs at least {n_nodes - 1}."
        )
    generator = graph_generator(seed)
    draws = 0
    while True:
        draws += 1
        chosen = generator.choice(len(rows), size=n_edges, replace=False)
        matrix = _matrix_from(zip(rows[chosen], cols[chosen], strict=True), n_nodes)
        # ASSUMPTION A-41: a draw that leaves some sensors unable to reach the
        # others is thrown away and drawn again, because Table 3 joins every
        # sensor into one piece and the control should differ from it only in
        # which pairs are joined.
        if not connected or _is_connected(matrix):
            break
    logger.info("Drew a random graph of %d connection(s) from seed %d in %d draw(s).",
                n_edges, seed, draws)
    return matrix, PAPER_SENSOR_SYMBOLS[:n_nodes]


def swap_ends(
    first: tuple[int, int], second: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Exchange the second ends of two connections: (a, b) and (c, d) become (a, d) and (c, b)."""
    (a, b), (c, d) = first, second
    return (min(a, d), max(a, d)), (min(c, b), max(c, b))


def build_rewired_adjacency(
    seed: int,
    *,
    adjacency: np.ndarray | None = None,
    proposals: int = REWIRING_PROPOSALS,
    connected: bool = True,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Rewire a graph at random, keeping every sensor's number of connections. Default: Table 3."""
    if adjacency is None:
        adjacency = build_physics_adjacency()[0]
    matrix = np.asarray(adjacency)
    if not np.allclose(matrix, matrix.T) or np.any(np.diag(matrix) != 0):
        raise ValueError("Rewiring needs a symmetric graph with no sensor joined to itself.")
    if proposals < 1:
        raise ValueError(f"proposals must be at least 1, got {proposals}.")
    n_nodes = matrix.shape[0]
    edges = connections_of(matrix)
    if len(edges) < 2:
        raise ValueError("Rewiring needs at least two connections to swap.")
    if connected and (np.any(matrix.sum(axis=1) == 0) or len(edges) < n_nodes - 1):
        raise ValueError("No rewiring of this graph can be in one piece: a sensor has no "
                         "connection, or there are fewer than one less than the sensors.")
    present = set(edges)
    generator = graph_generator(seed)
    rounds = accepted = 0
    while True:
        rounds += 1
        # ASSUMPTION A-41: every proposal counts as a step, including one that
        # is refused. Counting only the accepted ones would favour graphs from
        # which many swaps are possible, so the graphs drawn would no longer be
        # equally likely.
        picks = generator.integers(len(edges), size=(proposals, 2))
        reverse = generator.integers(2, size=proposals)
        for (i, j), flip in zip(picks.tolist(), reverse.tolist(), strict=True):
            if i == j:
                continue
            first, second = edges[i], edges[j]
            if flip:
                second = (second[1], second[0])
            new_first, new_second = swap_ends(first, second)
            # Refused: a sensor joined to itself, or a pair joined twice.
            if (new_first[0] == new_first[1] or new_second[0] == new_second[1]
                    or new_first in present or new_second in present):
                continue
            present.difference_update((edges[i], edges[j]))
            present.update((new_first, new_second))
            edges[i], edges[j] = new_first, new_second
            accepted += 1
        result = _matrix_from(edges, n_nodes)
        # ASSUMPTION A-41: a rewiring that leaves the sensors in more than one
        # piece is carried on for another round of proposals, for the reason the
        # random graph is drawn again.
        if not connected or _is_connected(result):
            break
        if rounds == 1000:
            raise RuntimeError("1,000 rounds of proposals never gave a rewiring in one piece.")
    logger.info("Rewired the graph from seed %d in %d round(s) of %d proposal(s); %d swap(s) accepted.",
                seed, rounds, proposals, accepted)
    return result, PAPER_SENSOR_SYMBOLS[:n_nodes]
