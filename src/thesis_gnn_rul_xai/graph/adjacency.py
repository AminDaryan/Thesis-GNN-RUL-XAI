"""
The sensor graph and its adjacency matrix
=========================================

What a graph is, in this context
--------------------------------
A **graph** is a set of points, called **nodes**, together with a set of
connections between them, called **edges**. In this project each of the 14
retained sensors is one node. An edge between two sensors means "these two
measurements are physically related, so the model is allowed to pass
information directly between them".

An **adjacency matrix** is the standard way to write a graph as a table of
numbers. For a graph with 14 nodes it is a 14-by-14 table ``A`` where
``A[i, j] = 1`` if there is an edge between node ``i`` and node ``j``, and
``A[i, j] = 0`` otherwise. Because "sensor i is related to sensor j" means the
same as "sensor j is related to sensor i", the matrix is **symmetric**: it
equals its own mirror image across the diagonal.

Why a Graph Neural Network needs this
-------------------------------------
A Graph Neural Network (GNN) works by **message passing**: each node repeatedly
collects the current feature vectors of its neighbours, combines them, and
updates its own features (Kipf & Welling, 2017; Corso et al., 2024). The
adjacency matrix is what defines "neighbour". Stacking three such layers, as
Wen et al. (2026) do, lets information travel up to three edges away: the first
layer reaches direct neighbours, the second reaches neighbours of neighbours,
and the third reaches three steps out.

Where this particular graph comes from
--------------------------------------
Most work on this dataset builds the graph from the data, usually by connecting
sensors whose readings are strongly correlated. Wen et al. (2026) instead
design the graph by hand from engineering knowledge of how a turbofan engine
works, and print the result as their Table 3. Their five stated rules are:

1. Sensors at the same physical location in the engine are connected, because
   they measure the same gas at the same point. Example: ``P30`` and ``Ps30``,
   the total and static pressure at the high-pressure compressor outlet.
2. A raw speed is connected to its corrected counterpart, because one is
   computed from the other. Example: ``Nf`` and ``NRf``.
3. Sensors are connected along the path the gas actually takes through the
   engine. Example: the temperature chain ``T24`` to ``T30`` to ``T50``, which
   follows the flow from the low-pressure compressor through the high-pressure
   compressor to the low-pressure turbine.
4. Fuel and control quantities are connected to what they influence. Example:
   ``phi``, the ratio of fuel flow to static pressure, is connected to the
   temperatures and pressures that burning more fuel changes.
5. Cooling quantities are connected to each other, because they jointly manage
   the temperature of the turbine section.

This module reproduces their Table 3 exactly, as printed, and the
transcription has been checked against both of the paper's own renderings of
the matrix: the numbers in Table 3 and the colour grid in Fig. 3 agree with
each other and with the array below, in all 196 cells.

Two things about it are **not** fully specified, despite the matrix itself
being printed twice:

* The paper's own fourth rule says `phi` influences `htBleed`, `W31` **and
  `W32`**, but the entry for `phi`-`W32` is 0 in Table 3 and in Fig. 3. The
  matrix is followed here, because it is what the model consumes, and the
  conflict is recorded as finding F-07.
* The paper never prints the formula used to normalise the matrix. See
  assumption A-21 and :func:`normalize_adjacency`.

Normalising the matrix before use
---------------------------------
A graph convolution layer does not use the raw matrix ``A``. It uses::

    A_hat = D^(-1/2) * (A + I) * D^(-1/2)

Two things happen in this formula, both from Kipf and Welling (2017):

* Adding the identity matrix ``I`` puts a **self-loop** on every node. Without
  it a node would be updated only from its neighbours and would forget its own
  current value.
* Dividing by the square roots of the **degrees** (the number of edges at each
  node, stored on the diagonal of ``D``) stops well-connected nodes from
  dominating. Without this scaling, repeatedly summing over neighbours makes
  feature values grow layer after layer, and training becomes unstable.

References
----------
Corso, G., Stark, H., Jegelka, S., Jaakkola, T., & Barzilay, R. (2024). Graph
neural networks. *Nature Reviews Methods Primers*, 4, 17.
https://doi.org/10.1038/s43586-024-00294-7

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph
convolutional networks. *International Conference on Learning Representations
(ICLR)*. https://arxiv.org/abs/1609.02907

Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
168, 110875. https://doi.org/10.1016/j.ast.2025.110875
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

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
]

logger = get_logger(__name__)

#: How the adjacency matrix is normalised before a graph convolution uses it.
#:
#: ``"symmetric"``
#:     ``D^(-1/2) (A + I) D^(-1/2)``, the form defined by Kipf and Welling
#:     (2017). Each edge is divided by the geometric mean of the two degrees it
#:     joins, so a message between two well-connected sensors is damped more
#:     than one between two sparsely connected sensors.
#: ``"row"``
#:     ``D^(-1) (A + I)``, which simply averages over each node's own
#:     neighbourhood. Rows sum to exactly 1, so every node receives the mean of
#:     its neighbours regardless of how many it has.
#:
#: Both add self-loops and both "balance feature aggregation across different
#: node degrees", which is all Wen et al. (2026) say about it. See A-21.
AdjacencyNormalization = Literal["symmetric", "row"]

#: Table 3 of Wen et al. (2026), transcribed row by row. The keys and the order
#: of the values both follow :data:`PAPER_SENSOR_SYMBOLS`. The matrix is checked
#: for symmetry when this module is imported, which catches any transcription
#: error immediately rather than letting it corrupt every later result.
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
    """Return the hand-designed sensor graph of Wen et al. (2026), Table 3.

    Returns
    -------
    tuple
        ``(adjacency, symbols)`` where ``adjacency`` is a 14-by-14 array of
        zeros and ones with a zero diagonal, and ``symbols`` gives the sensor
        symbol belonging to each row and column, in order.
    """
    symbols = PAPER_SENSOR_SYMBOLS
    matrix = np.array(
        [PHYSICS_ADJACENCY_ROWS[symbol] for symbol in symbols],
        dtype=np.float32,
    )
    return matrix, symbols


def build_fully_connected_adjacency(
    n_nodes: int = 14,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return a graph in which every sensor is connected to every other one.

    This is the comparison graph for the ablation that Wen et al. (2026) call
    "without physics prior": it removes all engineering knowledge while
    keeping the rest of the architecture identical, so any change in accuracy
    can be attributed to the graph structure alone.

    ASSUMPTION A-19: the paper names that ablation but never says what
    replaces the hand-designed matrix. A graph with no edges would ablate
    message passing itself, which the paper reports separately as "w/o GCN",
    so a complete graph is used here.
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
    """Build a graph by connecting sensors whose readings move together.

    Two sensors are connected when the absolute value of their Pearson
    correlation coefficient is at least ``threshold``. The **Pearson
    correlation coefficient** measures how closely two quantities follow a
    straight-line relationship; it runs from -1 (one rises exactly as the other
    falls) through 0 (no straight-line relationship) to +1 (they rise and fall
    together). The absolute value is used because a strong inverse relationship
    is just as informative as a strong direct one.

    This is the data-driven alternative to the hand-designed graph, and is the
    approach used by most other work on this dataset. It is provided so that
    the two kinds of graph can be compared directly.

    Parameters
    ----------
    frame : pandas.DataFrame
        Data to compute the correlations from. This must be **training data
        only**, for the same reason that scaling statistics come from training
        data only: using test data here would leak information.
    sensor_columns : sequence of str
        Columns to correlate, in node order.
    threshold : float
        Minimum absolute correlation for an edge, between 0 and 1.
    symbols : sequence of str, optional
        Node labels. Defaults to the paper's 14 sensor symbols.

    Returns
    -------
    tuple
        ``(adjacency, symbols)``.
    """
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
    """Apply the symmetric normalisation used by graph convolution layers.

    Computes ``D^(-1/2) (A + I) D^(-1/2)`` as defined by Kipf and Welling
    (2017). See the module docstring for what each part does and why.

    ASSUMPTION A-21: Wen et al. (2026) never print this formula. Their
    Equation 8 says only that "A denotes the normalized adjacency matrix
    incorporating self-loops", and that the normalisation "ensures stable
    gradient propagation and balanced feature aggregation across different
    node degrees". Symmetric normalisation satisfies that description, and
    so does the row normalisation ``D^(-1) (A + I)``. The symmetric form is
    used because it is the one defined by the layer the paper cites.

    Parameters
    ----------
    adjacency : numpy.ndarray
        A square, symmetric matrix of edge indicators.
    add_self_loops : bool
        Whether to add the identity matrix before normalising. This should stay
        ``True`` for graph convolution; the option exists so the effect can be
        demonstrated.
    method : str
        ``"symmetric"`` for ``D^(-1/2) (A + I) D^(-1/2)``, or ``"row"`` for
        ``D^(-1) (A + I)``. See :data:`AdjacencyNormalization` and assumption
        A-21: the paper describes the normalisation but never prints it, and
        both forms fit the description.

    Returns
    -------
    numpy.ndarray
        The normalised matrix, dtype ``float32``.

    Raises
    ------
    ValueError
        If the matrix is not square or not symmetric.

    Examples
    --------
    Two nodes joined by one edge. After adding self-loops each node has degree
    2, so every entry becomes ``1 / sqrt(2 * 2) = 0.5``:

    >>> import numpy as np
    >>> normalize_adjacency(np.array([[0.0, 1.0], [1.0, 0.0]]))
    array([[0.5, 0.5],
           [0.5, 0.5]], dtype=float32)

    On that symmetric two-node graph the row form agrees, because both degrees
    are equal. The two differ as soon as the degrees do:

    >>> star = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> rows = normalize_adjacency(star, method="row").sum(axis=1)
    >>> [round(float(v), 6) for v in rows]
    [1.0, 1.0, 1.0]
    >>> symmetric = normalize_adjacency(star).sum(axis=1)
    >>> [round(float(v), 3) for v in symmetric]
    [1.15, 0.908, 0.908]
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


def adjacency_report(
    adjacency: np.ndarray,
    symbols: Sequence[str],
) -> dict[str, object]:
    """Describe a graph, and check that it is usable.

    Returns
    -------
    dict
        Keys: ``n_nodes``, ``n_edges``, ``density``, ``is_symmetric``,
        ``has_self_loops``, ``is_connected``, ``degrees`` and
        ``isolated_nodes``.

        ``density`` is the fraction of all possible edges that are present.
        ``is_connected`` says whether every node can be reached from every
        other node by following edges; a graph that is not connected would
        split the sensors into groups that can never exchange information, no
        matter how many layers the network has.
    """
    matrix = np.asarray(adjacency)
    n_nodes = matrix.shape[0]
    degrees = matrix.sum(axis=1).astype(int)

    # Breadth-first search from node 0 to test connectivity.
    reached = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbour in np.flatnonzero(matrix[current]):
            if int(neighbour) not in reached:
                reached.add(int(neighbour))
                frontier.append(int(neighbour))

    n_possible = n_nodes * (n_nodes - 1)
    return {
        "n_nodes": n_nodes,
        "n_edges": int(matrix.sum() // 2),
        "density": float(matrix.sum() / n_possible) if n_possible else 0.0,
        "is_symmetric": bool(np.allclose(matrix, matrix.T)),
        "has_self_loops": bool(np.any(np.diag(matrix) != 0)),
        "is_connected": len(reached) == n_nodes,
        "degrees": dict(zip(symbols, degrees.tolist(), strict=True)),
        "isolated_nodes": [symbols[i] for i in np.flatnonzero(degrees == 0)],
    }


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
