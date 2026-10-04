"""How much of one prediction travelled along each walk through the sensor graph (GNN-LRP). Not in the paper.

Read README.md in this folder. It explains everything this file does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..models.estgnn import ESTGNN
from ..models.layers import GraphConvolution
from ..models.temporal import LastCycle, MeanWindow, RawWindow, WindowSummary
from ..utils.logging_setup import get_logger

__all__ = [
    "MAX_WALKS_PER_EXAMPLE",
    "PAPER_GAMMA_START",
    "WalkRelevance",
    "connection_relevance",
    "conservation_error",
    "crossing_relevance",
    "cycle_relevance",
    "indistinguishable_sensors",
    "node_relevance",
    "paper_gammas",
    "step_relevance",
    "top_walks",
    "walk_relevance",
]

logger = get_logger(__name__)

# Every walk is stored, so a window costs nodes ** (layers + 1) numbers: 38,416
# for 14 sensors and three graph layers. A fifth layer would cost 7.5 million.
MAX_WALKS_PER_EXAMPLE: int = 1_000_000

# The gamma the authors' own implementation gives the first graph layer when it
# is given none, falling in equal steps to 0 at the last: linspace(3, 0, depth)
# in `explain` of their synthetic-graph code (Schnake et al., 2022). The paper
# itself sets gamma per network, and says what the layers after the graph part
# get only once: 0 in VGG-16's top-level classifier (its Section 5.3).
PAPER_GAMMA_START: float = 3.0

# How far the prediction rebuilt here may sit from the network's own, in cycles,
# before the network is judged to compute something this file does not model.
_REBUILD_TOLERANCE: float = 1e-3


@dataclass(frozen=True, eq=False)
class WalkRelevance:
    """Every walk's share of the score before the sigmoid, for one or more windows."""

    relevance: np.ndarray
    constant: np.ndarray
    score: np.ndarray
    predicted: np.ndarray
    gamma: tuple[float, ...]
    score_at_zero: float

    @property
    def n_nodes(self) -> int:
        """How many sensors the graph has."""
        return int(self.relevance.shape[1])

    @property
    def n_steps(self) -> int:
        """How many graph layers, which is how many steps every walk takes."""
        return int(self.relevance.ndim - 2)


def _check_model(model: ESTGNN) -> None:
    """Refuse a network this file does not know how to split into walks."""
    if not isinstance(model, ESTGNN):
        raise TypeError(f"walk_relevance explains an ESTGNN, got {type(model).__name__}.")
    for index, layer in enumerate(model.graph_layers, start=1):
        if type(layer) is not GraphConvolution:
            raise TypeError(
                f"graph layer {index} is a {type(layer).__name__}, not a "
                "GraphConvolution. The split into walks relies on every graph layer "
                "mixing its neighbours with fixed weights; a layer whose weights "
                "depend on the input, such as attention, needs a rule this file "
                "does not have."
            )


class _Layer:
    """One graph layer as the walks see it: its weights, its constant and its pass-on factors."""

    def __init__(self, weight: Tensor, bias: Tensor | None, factor: Tensor) -> None:
        self.weight = weight
        self.bias = bias
        self.factor = factor


def _rule_forward(
    model: ESTGNN, nodes: Tensor, gammas: Sequence[float]
) -> tuple[list[_Layer], Tensor, Tensor]:
    """Rebuild the forward pass, keeping what each graph layer passes on under the rule."""
    adjacency = model.adjacency.double()
    hidden = nodes
    layers: list[_Layer] = []
    for layer, gamma in zip(model.graph_layers, gammas, strict=True):
        weight = layer.weight.double()
        bias = layer.bias.double() if layer.bias is not None else None
        mixed = adjacency @ hidden
        before = mixed @ weight + (bias if bias is not None else 0.0)
        after = torch.relu(before)
        # The LRP-gamma rule of Schnake et al., Table 1: add gamma times each
        # positive weight, then rescale so the layer still passes on exactly
        # what it computed. Table 1 has no constant; a positive constant gets
        # gamma too, as in the authors' code (`lrpforward` in their
        # synthetic-graph modules); graph layers have one only with gcn_bias.
        # With gamma = 0 the factor is 1 where the ReLU is on and 0 where it is
        # off. Which gamma each layer gets is decided in _gammas, below.
        ruled_weight = weight + gamma * weight.clamp(min=0)
        ruled_bias = bias + gamma * bias.clamp(min=0) if bias is not None else None
        ruled = mixed @ ruled_weight + (ruled_bias if ruled_bias is not None else 0.0)
        safe = torch.where(after > 0, ruled, torch.ones_like(ruled))
        factor = torch.where(after > 0, after / safe, torch.zeros_like(after))
        layers.append(_Layer(ruled_weight, ruled_bias, factor))
        hidden = after
    dense_before = torch.flatten(hidden, start_dim=1) @ model.fc1.weight.double().T
    dense_before = dense_before + model.fc1.bias.double()
    dense_gate = (dense_before > 0).double()
    score = torch.relu(dense_before) @ model.fc2.weight.double()[0] + model.fc2.bias.double()[0]
    return layers, dense_gate, score


def _chunk(
    model: ESTGNN, windows: Tensor, gammas: Sequence[float]
) -> tuple[Tensor, Tensor, Tensor]:
    """Walks, constant and score for one batch of windows, in double precision."""
    adjacency = model.adjacency.double()
    n_nodes = model.n_nodes
    batch = windows.shape[0]
    nodes = model.temporal(windows).double()
    layers, dense_gate, score = _rule_forward(model, nodes, gammas)

    # What one unit of each last-layer number is worth to the score, with the
    # dense layers' ReLUs held as they were: fc2's weights, through the dense
    # ReLUs that were on, back through fc1. Shaped (batch, nodes, last width).
    # The dense layers always pass relevance on with gamma = 0, as the authors'
    # own code does at the output.
    last_width = model.graph_layers[-1].out_features
    worth = (dense_gate * model.fc2.weight.double()[0]) @ model.fc1.weight.double()
    worth = worth.reshape(batch, n_nodes, last_width)

    # ASSUMPTION A-29: split the score before the sigmoid, not the prediction in
    # cycles, into one share per walk. The paper explains only sensors, never
    # walks. Written up in docs/assumptions/A-29.md.
    #
    # carried[b, p, j, c]: how much of channel c at node j came from the walk
    # prefix p, which lists the nodes visited before j. There is one prefix
    # before the first layer, and each layer multiplies their number by nodes.
    carried = nodes[:, None, :, :]
    for index, layer in enumerate(layers):
        if index < len(layers) - 1:
            mixed = carried @ layer.weight
            # Node j sends to node k with the strength adjacency[k, j]; node k's
            # factor decides how much of it is passed on, and it is 0 where the
            # ReLU is off.
            carried = (
                mixed[:, :, :, None, :]
                * adjacency.T[None, None, :, :, None]
                * layer.factor[:, None, None, :, :]
            )
            prefixes = carried.shape[1] * n_nodes
            carried = carried.reshape(batch, prefixes, n_nodes, layer.weight.shape[1])
        else:
            # The last layer: fold its weights, its factors and what each number
            # is worth into one (batch, nodes, width) array first, so the widest
            # array never holds a channel axis.
            reach = (layer.factor * worth) @ layer.weight.T
            ends = carried @ reach.transpose(1, 2)[:, None, :, :]
            ends = ends * adjacency.T[None, None, :, :]
            walks = ends.reshape(batch, *([n_nodes] * (len(layers) + 1)))

    # The layers' added constants belong to no sensor. Each is carried forward
    # through the same factors and the same graph, so the part of the score they
    # make up is known separately from the walks, not as a remainder.
    constant_stream = torch.zeros(
        batch, n_nodes, nodes.shape[2], dtype=torch.float64, device=nodes.device
    )
    for layer in layers:
        constant_stream = adjacency @ constant_stream @ layer.weight
        if layer.bias is not None:
            constant_stream = constant_stream + layer.bias
        constant_stream = layer.factor * constant_stream
    constant = (constant_stream * worth).sum(dim=(1, 2))
    constant = constant + (dense_gate * model.fc1.bias.double()) @ model.fc2.weight.double()[0]
    constant = constant + model.fc2.bias.double()[0]
    return walks, constant, score


def paper_gammas(n_layers: int) -> list[float]:
    """The authors' default gamma for each graph layer, with 0 where its input can be negative."""
    if n_layers < 1:
        raise ValueError(f"n_layers must be at least 1, got {n_layers}.")
    values = [float(v) for v in np.linspace(PAPER_GAMMA_START, 0.0, n_layers)]
    # ASSUMPTION A-29: gamma is 0 at the first graph layer. The rule's derivation
    # needs every number entering the layer to be at least 0 (Schnake et al.,
    # appendix C), and where a network lacks that ReLU-then-dense-then-ReLU
    # structure the authors fall back to gamma 0 themselves, as for their
    # SchNet model. This network's first graph layer receives rescaled
    # readings, which are negative below their training average. With one
    # number per sensor, as from the last cycle, the window average or the
    # default recurrent stage, gamma there cancels exactly and this changes
    # nothing; it matters for the three-number summary and the raw window.
    values[0] = 0.0
    return values


def _gammas(model: ESTGNN, gamma: float | Sequence[float] | None) -> list[float]:
    """One gamma per graph layer, the first always 0. README.md says why."""
    n_layers = len(model.graph_layers)
    if gamma is None:
        values = paper_gammas(n_layers)
    elif isinstance(gamma, (int, float)):
        values = [0.0] + [float(gamma)] * (n_layers - 1)
    else:
        values = [float(value) for value in gamma]
        if len(values) != n_layers:
            raise ValueError(
                f"gamma needs one value per graph layer ({n_layers}), got {len(values)}."
            )
        if values[0] != 0.0:
            raise ValueError(
                "gamma must be 0 at the first graph layer. The rule that uses it "
                "assumes a layer's inputs are never negative, and the first layer "
                "receives rescaled readings, which can be (Schnake et al., appendix C)."
            )
    if min(values) < 0:
        raise ValueError(f"gamma cannot be negative, got {values}.")
    if max(values) > 0 and bool((model.adjacency < 0).any()):
        raise ValueError("a positive gamma needs a sensor graph with no negative entries.")
    return values


@torch.no_grad()
def walk_relevance(
    model: ESTGNN,
    features: np.ndarray,
    *,
    gamma: float | Sequence[float] | None = None,
    batch_size: int = 16,
    device: torch.device | str | None = None,
) -> WalkRelevance:
    """Split each window's score before the sigmoid into one number per walk."""
    _check_model(model)
    gammas = _gammas(model, gamma)
    single = features.ndim == 2
    batch = features[None, ...] if single else features
    if batch.ndim != 3:
        raise ValueError(
            "features must be (nodes, window) or (examples, nodes, window), got "
            f"{features.shape}."
        )
    n_nodes = model.n_nodes
    n_steps = len(model.graph_layers)
    if n_nodes ** (n_steps + 1) > MAX_WALKS_PER_EXAMPLE:
        raise ValueError(
            f"{n_nodes} sensors and {n_steps} graph layers make {n_nodes} ** "
            f"{n_steps + 1} = {n_nodes ** (n_steps + 1):,} walks per window, above "
            f"the limit of {MAX_WALKS_PER_EXAMPLE:,}. Every walk is stored, and "
            "this file does not sample them."
        )

    if any(p.numel() for p in model.temporal.parameters()):
        logger.warning(
            "%s learns its own numbers, so the walks start at its output and say "
            "nothing about which cycles of the window mattered; zero at its "
            "output is not any real window.",
            type(model.temporal).__name__,
        )

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device).eval()

    zero = torch.zeros(
        1, n_nodes, model.in_features, dtype=torch.float64, device=device
    )
    score_at_zero = float(_rule_forward(model, zero, gammas)[2][0])

    walks, constants, scores = [], [], []
    for start in range(0, len(batch), batch_size):
        chunk = torch.from_numpy(batch[start : start + batch_size].astype(np.float32))
        chunk_walks, chunk_constant, chunk_score = _chunk(model, chunk.to(device), gammas)
        walks.append(chunk_walks.cpu().numpy())
        constants.append(chunk_constant.cpu().numpy())
        scores.append(chunk_score.cpu().numpy())
    score = np.concatenate(scores)

    # The rebuilt forward pass must give the network's own prediction. If the
    # network ever computes something else -- a new layer, a different output
    # -- this refuses, rather than explaining a network that does not exist.
    predicted = (
        model.predict_cycles(torch.from_numpy(batch.astype(np.float32)).to(device))
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    rebuilt = model.rul_max / (1.0 + np.exp(-score))
    gap = float(np.max(np.abs(rebuilt - predicted)))
    if gap > _REBUILD_TOLERANCE:
        raise RuntimeError(
            f"The forward pass rebuilt for the walks differs from the network's own "
            f"prediction by {gap:.6f} cycles. The network computes something this "
            "file does not model, so its walks would describe a different network."
        )

    result = WalkRelevance(
        relevance=np.concatenate(walks),
        constant=np.concatenate(constants),
        score=score,
        predicted=predicted,
        gamma=tuple(gammas),
        score_at_zero=score_at_zero,
    )
    logger.info(
        "Split %d window(s) into %d walk(s) of %d step(s) each.",
        len(batch), n_nodes ** (n_steps + 1), n_steps,
    )
    return result


def conservation_error(result: WalkRelevance) -> float:
    """The largest amount by which walks plus constant miss the score. Rounding noise only."""
    total = result.relevance.reshape(len(result.score), -1).sum(axis=1)
    return float(np.max(np.abs(total + result.constant - result.score)))


def node_relevance(result: WalkRelevance) -> np.ndarray:
    """Each sensor's share of the score: every walk that starts at it, added up."""
    axes = tuple(range(2, result.relevance.ndim))
    return result.relevance.sum(axis=axes)


def step_relevance(result: WalkRelevance, step: int) -> np.ndarray:
    """What passed along each connection at one graph layer, shaped (examples, from, to)."""
    if not 1 <= step <= result.n_steps:
        raise ValueError(f"step must lie in [1, {result.n_steps}], got {step}.")
    keep = (step, step + 1)
    axes = tuple(axis for axis in range(1, result.relevance.ndim) if axis not in keep)
    return result.relevance.sum(axis=axes)


def _groups(
    result: WalkRelevance, symbols: Sequence[str], pool: Sequence[Sequence[str]]
) -> np.ndarray:
    """For each sensor, the index of the first sensor of its group; itself when in none."""
    names = list(symbols)
    if len(names) != result.n_nodes:
        raise ValueError(
            f"the walks run over {result.n_nodes} node(s) but {len(names)} symbol(s) were given."
        )
    label = np.arange(len(names))
    for group in pool:
        unknown = set(group) - set(names)
        if unknown:
            raise ValueError(f"pool names sensor(s) not in symbols: {sorted(unknown)}.")
        for symbol in group:
            label[names.index(symbol)] = names.index(group[0])
    return label


def crossing_relevance(
    result: WalkRelevance,
    symbols: Sequence[str] = (),
    *,
    pool: Sequence[Sequence[str]] = (),
) -> pd.DataFrame:
    """Split each score into walks that stay on their sensor's group, walks that move, and the constant."""
    label = (
        _groups(result, symbols, pool) if pool else np.arange(result.n_nodes)
    )
    # A walk stays when every sensor it reaches is in its starting sensor's
    # group; with no pool every sensor is its own group, so it stays on one
    # sensor. README.md says why a group must count as one sensor here.
    shape = result.relevance.shape[1:]
    stays_mask = np.ones(shape, dtype=bool)
    start = label.reshape((-1,) + (1,) * (len(shape) - 1))
    for axis in range(1, len(shape)):
        other = label.reshape((1,) * axis + (-1,) + (1,) * (len(shape) - axis - 1))
        stays_mask &= start == other
    flat = result.relevance.reshape(len(result.score), -1)
    mask = stays_mask.reshape(-1)
    stays = flat[:, mask].sum(axis=1)
    return pd.DataFrame(
        {
            "stays": stays,
            "moves": flat.sum(axis=1) - stays,
            "constant": result.constant,
            "score": result.score,
            "predicted": result.predicted,
            "absolute_stays": np.abs(flat[:, mask]).sum(axis=1),
            "absolute_total": np.abs(flat).sum(axis=1),
        }
    )


def connection_relevance(
    result: WalkRelevance,
    symbols: Sequence[str],
    *,
    pool: Sequence[Sequence[str]] = (),
) -> np.ndarray:
    """What passed along each connection over every graph layer, shaped (examples, from, to)."""
    _groups(result, symbols, pool)
    passed = sum(step_relevance(result, step) for step in range(1, result.n_steps))
    last = step_relevance(result, result.n_steps)
    names = list(symbols)
    for group in pool:
        members = [names.index(symbol) for symbol in group]
        # ASSUMPTION A-29: what arrives at the last graph layer inside a group of
        # sensors the graph cannot tell apart is split equally among them. How it
        # really splits follows fc1 weights that training never separated
        # (finding F-18), so only the group's total means anything; an equal
        # split keeps that total and draws it without inventing a difference.
        last[:, :, members] = last[:, :, members].mean(axis=2, keepdims=True)
    return passed + last


def cycle_relevance(
    model: ESTGNN,
    features: np.ndarray,
    *,
    gamma: float | Sequence[float] | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Each sensor's walk share under the same rule, split over the cycles of its window."""
    _check_model(model)
    gammas = _gammas(model, gamma)
    # Only a stage that is a fixed weighted sum of the readings, with nothing
    # added, hands the graph layers numbers whose shares divide exactly among
    # the cycles they were made from.
    if not isinstance(model.temporal, (LastCycle, MeanWindow, WindowSummary, RawWindow)):
        raise ValueError(
            f"{type(model.temporal).__name__} learns its own numbers, so a share of "
            "what it hands the graph layers cannot be divided among the cycles "
            "exactly; this file has no rule for it."
        )
    single = features.ndim == 2
    batch = features[None, ...] if single else features
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device).eval()

    windows = torch.from_numpy(batch.astype(np.float64)).to(device).requires_grad_(True)
    hidden = model.temporal(windows)
    adjacency = model.adjacency.double()
    with torch.no_grad():
        layers, _, _ = _rule_forward(model, hidden.detach(), gammas)
    # The authors' own way of running the rule through automatic
    # differentiation (Schnake et al., Section 3.2.1): each graph layer's output
    # is rebuilt as its rule-weighted sum times a held factor, which leaves its
    # value unchanged and makes the derivative follow the rule. Times the
    # reading, that derivative hands each sensor's share back to its cycles
    # through the stage's fixed weights.
    for layer in layers:
        ruled = adjacency @ hidden @ layer.weight
        if layer.bias is not None:
            ruled = ruled + layer.bias
        hidden = ruled * layer.factor
    flat = torch.flatten(hidden, start_dim=1)
    score = torch.relu(flat @ model.fc1.weight.double().T + model.fc1.bias.double())
    score = score @ model.fc2.weight.double()[0] + model.fc2.bias.double()[0]
    (gradient,) = torch.autograd.grad(score.sum(), windows)
    shares = (gradient * windows).detach().cpu().numpy()
    return shares[0] if single else shares


def indistinguishable_sensors(model: ESTGNN, symbols: Sequence[str]) -> list[tuple[str, ...]]:
    """Groups of sensors with identical rows in the rescaled graph, which the graph layers cannot tell apart."""
    if len(symbols) != model.n_nodes:
        raise ValueError(
            f"the network has {model.n_nodes} node(s) but {len(symbols)} symbol(s) were given."
        )
    rows = model.adjacency.detach().cpu().numpy()
    groups: dict[bytes, list[str]] = {}
    for index, symbol in enumerate(symbols):
        groups.setdefault(rows[index].tobytes(), []).append(symbol)
    return [tuple(group) for group in groups.values() if len(group) > 1]


def top_walks(
    result: WalkRelevance,
    symbols: Sequence[str],
    *,
    count: int = 20,
    pool: Sequence[Sequence[str]] = (),
) -> pd.DataFrame:
    """The walks that carried the most, named by sensor. README.md lists the columns."""
    if len(symbols) != result.n_nodes:
        raise ValueError(
            f"the walks run over {result.n_nodes} node(s) but {len(symbols)} "
            "symbol(s) were given."
        )
    # Every position after the first is named by its group, so walks that the
    # graph cannot tell apart are added together; the start keeps its own name,
    # because each member's own reading enters there. README.md explains why.
    label = {symbol: symbol for symbol in symbols}
    for group in pool:
        unknown = set(group) - set(symbols)
        if unknown:
            raise ValueError(f"pool names sensor(s) not in symbols: {sorted(unknown)}.")
        for symbol in group:
            label[symbol] = "/".join(group)
    shape = result.relevance.shape[1:]
    positions = np.array(np.unravel_index(np.arange(int(np.prod(shape))), shape)).T
    names = [
        " -> ".join(
            [symbols[walk[0]], *(label[symbols[node]] for node in walk[1:])]
        )
        for walk in positions
    ]
    flat = pd.DataFrame(result.relevance.reshape(len(result.score), -1).T)
    pooled = flat.groupby(np.array(names), sort=False).sum()
    mean = pooled.mean(axis=1)
    mean_abs = pooled.abs().mean(axis=1)
    order = np.argsort(-mean_abs.to_numpy(), kind="stable")[:count]
    return pd.DataFrame(
        {
            "walk": pooled.index[order],
            "mean": mean.to_numpy()[order],
            "mean_abs": mean_abs.to_numpy()[order],
        }
    )
