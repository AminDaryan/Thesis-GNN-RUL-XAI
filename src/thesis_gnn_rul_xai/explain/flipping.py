"""The GNN-LRP paper's own check of an explanation: add or remove sensors in the order it ranks them and follow the score. Not in the main paper.

Read README.md in this folder. It explains everything this file does.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
import torch

from ..models.estgnn import ESTGNN
from .walks import WalkRelevance, _check_model, node_relevance

__all__ = [
    "FLIPPING_TASKS",
    "RANDOM_ORDERS",
    "aufc",
    "node_flipping",
    "subgraph_relevance",
    "subgraph_scores",
]

# Schnake et al., Section 4.1: the activation task adds sensors to an empty
# graph, the pruning task removes them from the full one.
FLIPPING_TASKS: tuple[str, ...] = ("activation", "pruning")

# ASSUMPTION A-33: the random ordering the paper sets beside each explanation is
# averaged over this many random orders, each drawn from a generator seeded by
# the window's position. The paper reports a random baseline but not how many
# orders make it. Written up in docs/assumptions/A-33.md.
RANDOM_ORDERS: int = 20

# How far the full graph's score here may sit from the walks' own score before
# the two are judged to describe different networks or windows.
_SCORE_TOLERANCE: float = 1e-6


def subgraph_relevance(relevance: np.ndarray, present: np.ndarray) -> float:
    """What the walks say a set of sensors is worth: every walk lying wholly inside it, added up."""
    # Schnake et al., Section 4.1: R_G is the sum of R_B over the bags B whose
    # edges all lie in G. For walks between the sensors of G that is every
    # walk whose sensors are all present: one contraction per walk position.
    mask = present.astype(np.float64)
    total = np.asarray(relevance, dtype=np.float64)
    for _ in range(total.ndim):
        total = total @ mask
    return float(total)


def _greedy(estimate: Callable[[np.ndarray], float], n_nodes: int, task: str) -> list[int]:
    """The order the paper's Equations 14 and 15 pick, one sensor at a time."""
    order: list[int] = []
    if task == "activation":
        present = np.zeros(n_nodes, dtype=bool)
        for _ in range(n_nodes):
            # Equation 14: add the sensor whose subgraph the explanation rates highest.
            candidates = np.flatnonzero(~present)
            values = [estimate(_with(present, v, True)) for v in candidates]
            chosen = int(candidates[int(np.argmax(values))])
            order.append(chosen)
            present[chosen] = True
    elif task == "pruning":
        present = np.ones(n_nodes, dtype=bool)
        full = estimate(present)
        for _ in range(n_nodes):
            # Equation 15: remove the sensor whose loss the explanation rates smallest.
            candidates = np.flatnonzero(present)
            values = [abs(full - estimate(_with(present, v, False))) for v in candidates]
            chosen = int(candidates[int(np.argmin(values))])
            order.append(chosen)
            present[chosen] = False
    else:
        raise ValueError(f"task must be one of {FLIPPING_TASKS}, got {task!r}.")
    return order


def _with(present: np.ndarray, node: int, value: bool) -> np.ndarray:
    """A copy of the present sensors with one sensor switched."""
    changed = present.copy()
    changed[node] = value
    return changed


def _masks(order: list[int], n_nodes: int, task: str) -> np.ndarray:
    """The sensors present before the first flip and after each one, shaped (steps + 1, nodes)."""
    present = np.zeros(n_nodes, dtype=bool) if task == "activation" else np.ones(n_nodes, dtype=bool)
    masks = [present.copy()]
    for node in order:
        present[node] = task == "activation"
        masks.append(present.copy())
    return np.stack(masks)


@torch.no_grad()
def subgraph_scores(model: ESTGNN, window: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """The network's score before the sigmoid with only the sensors of each mask in its graph."""
    _check_model(model)
    return _scores(model, _stage_output(model, window), masks)


@torch.no_grad()
def _stage_output(model: ESTGNN, window: np.ndarray) -> torch.Tensor:
    """What the temporal stage hands the first graph layer for one window, computed once."""
    device = next(model.parameters()).device
    return model.temporal(torch.from_numpy(window[None].astype(np.float32)).to(device)).double()


@torch.no_grad()
def _scores(model: ESTGNN, nodes: torch.Tensor, masks: np.ndarray) -> np.ndarray:
    """The score for each mask, from the stage output of one window."""
    keep = torch.from_numpy(masks.astype(np.float64)).to(nodes.device)
    # ASSUMPTION A-33: a sensor is taken out of the graph by zeroing its row and
    # its column of the rescaled graph the network was trained with, including
    # its connection to itself, and nothing is rescaled again. The paper runs
    # the network on the smaller graph (its Fig. 6) without saying how a graph
    # that was rescaled loses a node; the authors' code, when it masks a graph,
    # multiplies the mask into the already rescaled one. Its readings stay in
    # the window; with no connection left, nothing of them reaches the score
    # through the graph layers. Written up in docs/assumptions/A-33.md.
    adjacency = model.adjacency.double()[None] * keep[:, :, None] * keep[:, None, :]
    hidden = nodes.expand(len(masks), -1, -1)
    for layer in model.graph_layers:
        before = adjacency @ hidden @ layer.weight.double()
        if layer.bias is not None:
            before = before + layer.bias.double()
        hidden = torch.relu(before)
    dense = torch.flatten(hidden, start_dim=1) @ model.fc1.weight.double().T
    dense = torch.relu(dense + model.fc1.bias.double())
    score = dense @ model.fc2.weight.double()[0] + model.fc2.bias.double()[0]
    return score.cpu().numpy()


def _curve(scores: np.ndarray, task: str) -> np.ndarray:
    """What the paper plots: the score while adding, its distance from the full graph's while pruning."""
    if task == "activation":
        return scores
    return np.abs(scores - scores[0])


def node_flipping(
    model: ESTGNN,
    features: np.ndarray,
    explanations: Mapping[str, WalkRelevance],
    *,
    random_orders: int = RANDOM_ORDERS,
) -> pd.DataFrame:
    """One row per window, task and ranking: the order sensors were flipped and the curve the paper plots."""
    single = features.ndim == 2
    batch = features[None, ...] if single else features
    if not explanations:
        raise ValueError("give at least one walk explanation to check.")
    for name, result in explanations.items():
        if len(result.score) != len(batch):
            raise ValueError(
                f"explanation {name!r} covers {len(result.score)} window(s), the features {len(batch)}."
            )
    n_nodes = model.n_nodes
    rows: list[dict[str, object]] = []
    _check_model(model)
    for index, window in enumerate(batch):
        nodes = _stage_output(model, window)
        full = _scores(model, nodes, np.ones((1, n_nodes), dtype=bool))[0]
        rankings: dict[str, Callable[[np.ndarray], float]] = {}
        for name, result in explanations.items():
            if abs(full - float(result.score[index])) > _SCORE_TOLERANCE:
                raise ValueError(
                    f"window {index}: the network's score is {full:.6f} but explanation "
                    f"{name!r} split {float(result.score[index]):.6f}; they describe "
                    "different networks or windows."
                )
            walks = result.relevance[index]
            starts = node_relevance(result)[index]
            rankings[f"{name}, walks"] = lambda present, w=walks: subgraph_relevance(w, present)
            # The paper's node-level comparison: its explanation pooled by the
            # first sensor of each walk, which it says matches Pope et al.
            rankings[f"{name}, starting sensors"] = (
                lambda present, s=starts: float(s[present].sum())
            )
        generator = np.random.default_rng(index)
        shuffles = [list(generator.permutation(n_nodes)) for _ in range(random_orders)]
        for task in FLIPPING_TASKS:
            for ranking, estimate in rankings.items():
                order = _greedy(estimate, n_nodes, task)
                curve = _curve(_scores(model, nodes, _masks(order, n_nodes, task)), task)
                rows.append(_row(index, task, ranking, order, curve))
            if random_orders:
                masks = np.concatenate([_masks(order, n_nodes, task) for order in shuffles])
                scores = _scores(model, nodes, masks).reshape(len(shuffles), n_nodes + 1)
                curves = [_curve(row, task) for row in scores]
                rows.append(_row(index, task, "random", [], np.mean(curves, axis=0)))
    return pd.DataFrame(rows)


def _row(index: int, task: str, ranking: str, order: list[int], curve: np.ndarray) -> dict[str, object]:
    """One flipping curve as a table row: the order, then one column per step."""
    row: dict[str, object] = {
        "window": index,
        "task": task,
        "ranking": ranking,
        "order": " ".join(str(node) for node in order),
        "aufc": float(np.mean(curve)),
    }
    for step, value in enumerate(curve):
        row[f"step_{step}"] = float(value)
    return row


def aufc(table: pd.DataFrame) -> pd.DataFrame:
    """The paper's area under the flipping curve, averaged over windows, one row per task and ranking."""
    return (
        table.groupby(["task", "ranking"], sort=False)["aufc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
