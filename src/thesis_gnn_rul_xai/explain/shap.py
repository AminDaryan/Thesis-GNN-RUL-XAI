"""How much each of the 14 sensors moved one prediction, in flight cycles.

Every combination of sensors is tried, so these are exact numbers rather than
an estimate.

Read README.md in this folder. It explains everything this file does.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import factorial

import numpy as np
import pandas as pd
import torch

from ..data.windowing import WindowedDataset
from ..models.estgnn import ESTGNN
from ..utils.logging_setup import get_logger

__all__ = [
    "MAX_EXACT_NODES",
    "coalition_weights",
    "dynamic_shap",
    "efficiency_error",
    "mean_window_baseline",
    "shap_summary",
    "shap_values",
]

logger = get_logger(__name__)

# Above this many sensors, trying every combination stops being reasonable:
# 20 sensors is about a million runs per window. This project has 14.
MAX_EXACT_NODES: int = 20


def coalition_weights(n_nodes: int) -> np.ndarray:
    """How much a group of each size counts, from the paper's Equation 15."""
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be at least 1, got {n_nodes}.")
    total = factorial(n_nodes)
    return np.array(
        [factorial(s) * factorial(n_nodes - s - 1) / total for s in range(n_nodes)],
        dtype=np.float64,
    )


def mean_window_baseline(dataset: WindowedDataset) -> np.ndarray:
    """The average reading of each sensor. Pass the training windows only.

    ASSUMPTION A-20: when a sensor is left out of the calculation, put its
    average training reading in its place. The paper never says what it puts
    there. Explained in README.md in this folder; written up in
    docs/ASSUMPTIONS.md.
    """
    if len(dataset) == 0:
        raise ValueError("cannot build a baseline from an empty dataset.")
    return dataset.features.mean(axis=0).astype(np.float32)


def _masked_batch(
    features: np.ndarray, baseline: np.ndarray, masks: np.ndarray
) -> np.ndarray:
    """Build one input per group: keep the chosen sensors, replace the rest."""
    # masks: (n_masks, nodes) of true/false, one entry per sensor and none per
    # cycle. Having no time axis is the point: one decision covers a sensor's
    # whole window, so it is kept entirely or replaced entirely, never in part.
    keep = masks[:, :, None]
    return np.where(keep, features[None, :, :], baseline[None, :, :])


def _after_stage(model: ESTGNN, nodes: torch.Tensor) -> torch.Tensor:
    """The prediction in cycles from what the temporal stage hands the graph layers."""
    # The steps of ESTGNN.forward after its temporal stage, in the same order;
    # _check_shortcut refuses a network for which the two stop agreeing.
    hidden = nodes
    for layer in model.graph_layers:
        hidden = torch.relu(layer(hidden, model.adjacency))
    dense = torch.relu(model.fc1(torch.flatten(hidden, start_dim=1)))
    return torch.sigmoid(model.fc2(dense)).squeeze(-1) * float(model.rul_max)


def _group_predictions(
    model: ESTGNN,
    example: np.ndarray,
    stand_in: torch.Tensor,
    masks: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """The prediction for every group, running the temporal stage once on the real window."""
    # Every temporal stage reads each sensor's own window and nothing else, so
    # in a window mixing real and stand-in readings, a sensor's stage output is
    # its output on the real window or on the stand-in, whichever the group
    # gives it. The stage then runs once per window rather than once per group.
    real = model.temporal(torch.from_numpy(example[None]).to(device))[0]
    keep = torch.from_numpy(masks[:, :, None]).to(device)
    predictions = np.empty(len(masks), dtype=np.float64)
    for start in range(0, len(masks), batch_size):
        nodes = torch.where(keep[start : start + batch_size], real, stand_in)
        predictions[start : start + batch_size] = _after_stage(model, nodes).cpu().numpy()
    return predictions


def _check_shortcut(
    model: ESTGNN,
    example: np.ndarray,
    baseline: np.ndarray,
    stand_in: torch.Tensor,
    masks: np.ndarray,
    device: torch.device,
) -> None:
    """Refuse a network whose full run disagrees with the stage-once route on a few groups."""
    n_nodes = masks.shape[1]
    every = (1 << n_nodes) - 1
    alternate = sum(1 << i for i in range(0, n_nodes, 2))
    first_half = (1 << (n_nodes // 2)) - 1
    sample = masks[[0, every, alternate, every ^ alternate, first_half, every ^ first_half]]
    whole = torch.from_numpy(_masked_batch(example, baseline, sample)).to(device)
    direct = model.predict_cycles(whole).cpu().numpy()
    shortcut = _group_predictions(model, example, stand_in, sample, len(sample), device)
    gap = float(np.max(np.abs(direct - shortcut)))
    if gap > 1e-3:
        raise ValueError(
            f"Running the whole network and running its temporal stage once per "
            f"window disagree by {gap:.2e} cycles on the same groups of sensors. "
            "Either the temporal stage reads more than each sensor's own window, "
            "or ESTGNN.forward has changed and _after_stage in this file no "
            "longer repeats it. Either way these would not be Shapley values of "
            "this network."
        )


@torch.no_grad()
def shap_values(
    model: ESTGNN,
    features: np.ndarray,
    baseline: np.ndarray,
    *,
    batch_size: int = 8192,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """How many cycles each sensor added to the prediction, for each window.

    Predicts 16,384 times per window, once for every combination of the 14
    sensors; the temporal stage runs once per window and once for the stand-in.
    """
    single = features.ndim == 2
    batch = features[None, ...] if single else features
    if batch.ndim != 3:
        raise ValueError(
            "features must be (nodes, window) or (examples, nodes, window), got "
            f"{features.shape}."
        )
    n_nodes = batch.shape[1]
    if baseline.shape != batch.shape[1:]:
        raise ValueError(
            f"baseline must be shaped {batch.shape[1:]} to match one example, "
            f"got {baseline.shape}."
        )
    if n_nodes > MAX_EXACT_NODES:
        raise ValueError(
            f"Exact Shapley values need 2**{n_nodes} model evaluations per "
            f"window, which is not reasonable. The limit is {MAX_EXACT_NODES} "
            "nodes; above it an approximation is required, and this module "
            "deliberately does not provide one."
        )

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device).eval()

    # Every subset of the 14 sensors, as an integer whose bit i says whether
    # sensor i is present.
    codes = np.arange(2**n_nodes, dtype=np.int64)
    bit = np.arange(n_nodes, dtype=np.int64)
    masks = ((codes[:, None] >> bit[None, :]) & 1).astype(bool)
    sizes = masks.sum(axis=1)
    weights = coalition_weights(n_nodes)
    baseline = baseline.astype(np.float32)

    # For each sensor, the subsets that lack it and the matching subsets that
    # have it. Setting bit i turns one into the other.
    lacking = [np.flatnonzero(~masks[:, i]) for i in range(n_nodes)]
    holding = [lacking[i] | (1 << i) for i in range(n_nodes)]

    # What the temporal stage hands the graph layers for the stand-in window,
    # once for the whole call.
    stand_in = model.temporal(torch.from_numpy(baseline[None]).to(device))[0]
    _check_shortcut(model, batch[0].astype(np.float32), baseline, stand_in, masks, device)

    results = np.empty((batch.shape[0], n_nodes), dtype=np.float64)
    for index, example in enumerate(batch):
        predictions = _group_predictions(
            model, example.astype(np.float32), stand_in, masks, batch_size, device
        )
        for i in range(n_nodes):
            without, with_i = lacking[i], holding[i]
            marginal = predictions[with_i] - predictions[without]
            results[index, i] = float(np.sum(weights[sizes[without]] * marginal))

    return results[0] if single else results


@torch.no_grad()
def efficiency_error(
    model: ESTGNN,
    features: np.ndarray,
    baseline: np.ndarray,
    values: np.ndarray,
    *,
    device: torch.device | str | None = None,
) -> float:
    """Check the numbers add up. Anything above rounding noise means a bug.

    ASSUMPTION A-24: the fourteen values are measured from the prediction on
    the average window, not from the average label. Equation 14 of the paper
    calls its constant "the average value of target variable across all
    samples"; the two differ by 10.7 cycles on a trained model. Written up in
    docs/ASSUMPTIONS.md.
    """
    single = features.ndim == 2
    batch = features[None, ...] if single else features
    contributions = np.atleast_2d(values)

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device).eval()

    actual = model.predict_cycles(
        torch.from_numpy(batch.astype(np.float32)).to(device)
    ).cpu().numpy()
    base = float(
        model.predict_cycles(
            torch.from_numpy(baseline.astype(np.float32)[None, ...]).to(device)
        ).cpu().numpy()[0]
    )
    return float(np.max(np.abs(contributions.sum(axis=1) - (actual - base))))


def dynamic_shap(
    model: ESTGNN,
    dataset: WindowedDataset,
    baseline: np.ndarray,
    *,
    unit: int,
    symbols: Sequence[str],
    device: torch.device | str | None = None,
) -> pd.DataFrame:
    """Repeat the calculation at every cycle of one engine's life.

    Returns one row per cycle. Do not put this table next to the paper's
    Table 12: README.md in this folder explains why they do not match up.
    """
    selected = np.flatnonzero(dataset.units == unit)
    if selected.size == 0:
        available = np.unique(dataset.units)[:10]
        raise ValueError(
            f"engine {unit} is not in this dataset. The first few that are: "
            f"{available.tolist()}."
        )
    order = selected[np.argsort(dataset.end_cycles[selected])]
    features = dataset.features[order]

    logger.info(
        "Explaining engine %d: %d window(s), %d coalitions each, %d model "
        "evaluations in total.",
        unit, len(order), 2 ** features.shape[1],
        len(order) * 2 ** features.shape[1],
    )

    values = shap_values(model, features, baseline, device=device)

    if device is None:
        device = next(model.parameters()).device
    with torch.no_grad():
        predicted = (
            model.predict_cycles(
                torch.from_numpy(features.astype(np.float32)).to(torch.device(device))
            )
            .cpu()
            .numpy()
        )
        base = float(
            model.predict_cycles(
                torch.from_numpy(baseline.astype(np.float32)[None, ...]).to(
                    torch.device(device)
                )
            )
            .cpu()
            .numpy()[0]
        )

    table = pd.DataFrame(values, columns=list(symbols))
    table.insert(0, "baseline", base)
    table.insert(0, "predicted", predicted)
    table.insert(0, "rul", dataset.targets[order])
    table.insert(0, "end_cycle", dataset.end_cycles[order])
    return table


def shap_summary(
    values: np.ndarray,
    symbols: Sequence[str],
) -> pd.DataFrame:
    """Rank the sensors by how much they mattered. README.md lists the columns."""
    values = np.atleast_2d(values)
    if values.shape[1] != len(symbols):
        raise ValueError(
            f"values describe {values.shape[1]} node(s) but {len(symbols)} "
            "symbol(s) were given."
        )
    table = pd.DataFrame(
        {
            "sensor": list(symbols),
            "mean_abs": np.abs(values).mean(axis=0),
            "mean": values.mean(axis=0),
            "max_abs": np.abs(values).max(axis=0),
        }
    )
    return table.sort_values("mean_abs", ascending=False).reset_index(drop=True)
