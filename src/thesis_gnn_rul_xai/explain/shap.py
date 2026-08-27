"""
Dynamic SHAP: which sensor drove this prediction, and when
==========================================================

What a Shapley value is
-----------------------
A model gives one number, and fourteen sensors were involved in producing it.
The question "how much did each sensor contribute?" has no obvious answer,
because the sensors are not independent: removing one changes what the others
are worth.

Cooperative game theory answers it. Imagine building the group of sensors one
at a time, in some order. Each time a sensor joins, the prediction changes by
some amount, and that change is what the sensor contributed *in that order*.
Do this for every possible order and average. That average is the sensor's
**Shapley value** (Shapley, 1953), and it is the only way of dividing the
credit that satisfies a short list of fairness conditions at once.

Written out, for sensor ``i``::

    phi_i = sum over every subset S not containing i of
                |S|! (n - |S| - 1)! / n!  *  ( f(S + {i}) - f(S) )

which is Equation 15 of Wen et al. (2026). The fraction is the share of orders
in which exactly the sensors of ``S`` come before ``i``.

Why this implementation is exact, and most are not
---------------------------------------------------
The sum runs over every subset of the other sensors. With 14 sensors that is
2^14 = 16,384 subsets, which a computer can simply enumerate. Most published
SHAP work cannot do this --- with 100 features there are more subsets than
atoms in the observable universe --- so it approximates, by sampling orders or
by fitting a weighted linear model (Lundberg & Lee, 2017).

Here no approximation is needed. Every number this module returns is the
Shapley value as Equation 15 defines it, not an estimate of it. That is worth
saying explicitly in the thesis, because it removes a whole class of doubt: a
disagreement with the paper cannot be blamed on sampling noise.

What "removing a sensor" means
------------------------------
Equation 15 needs ``f(S)``, "the ESTGNN prediction when only sensors in subset
S are considered". A trained network has no way to accept thirteen inputs; it
always needs fourteen. So a sensor is not removed, it is **replaced** by an
uninformative stand-in, and the choice of stand-in decides what the resulting
numbers mean.

The paper does not say what it uses. This module replaces an absent sensor's
whole 30-cycle window with the average window of that sensor across the
training set, so the model sees "this sensor is behaving typically" rather than
"this sensor reads zero" (which after rescaling would be its lowest value ever
recorded, a strong and misleading signal). See assumption A-20.

A Shapley value here therefore answers: *how much did knowing this sensor's
actual readings, rather than a typical engine's, change the prediction?*

Dynamic: the same calculation at every cycle
--------------------------------------------
Equation 16 of the paper is Equation 15 applied to the window ending at each
cycle in turn. An engine that ran for 220 cycles yields 191 windows at a window
size of 30, so it yields 191 sets of 14 Shapley values --- a table of how the
model's attention moved as the engine wore out. That is what the paper's
Table 12 reports and its Fig. 10 draws.

The one property to check
-------------------------
Shapley values must add up: the fourteen contributions plus the baseline
prediction must equal the actual prediction, exactly. This is called
**efficiency**, it is Equation 14 of the paper, and it is the property that
makes the numbers trustworthy rather than merely suggestive. If it fails, the
attribution is wrong no matter how plausible it looks.
:func:`efficiency_error` measures it, and the test suite asserts it.

References
----------
Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model
predictions. *Advances in Neural Information Processing Systems 30 (NeurIPS)*,
4765-4774. https://arxiv.org/abs/1705.07874

Shapley, L. S. (1953). A value for n-person games. In H. W. Kuhn & A. W. Tucker
(Eds.), *Contributions to the Theory of Games, Volume II* (pp. 307-317).
Princeton University Press. https://doi.org/10.1515/9781400881970-018

Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
168, 110875. https://doi.org/10.1016/j.ast.2025.110875
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
    "mean_window_baseline",
    "shap_values",
    "efficiency_error",
    "dynamic_shap",
    "shap_summary",
]

logger = get_logger(__name__)

#: Above this many nodes, enumerating every subset stops being reasonable.
#: 2^20 is about a million model evaluations per window; 2^14, the size this
#: project needs, is 16,384.
MAX_EXACT_NODES: int = 20


def coalition_weights(n_nodes: int) -> np.ndarray:
    """Return the weight Equation 15 gives to a subset of each size.

    The weight for a subset of size ``s`` is ``s! (n - s - 1)! / n!``: the
    fraction of all orderings of the ``n`` sensors in which exactly those ``s``
    sensors come before the one being valued.

    Returns
    -------
    numpy.ndarray
        Shaped ``(n_nodes,)``; element ``s`` is the weight for subsets of
        size ``s``.

    Examples
    --------
    With three sensors, the empty subset and the full subset of the other two
    each carry weight 1/3, and each single-sensor subset carries 1/6:

    >>> coalition_weights(3).round(6).tolist()
    [0.333333, 0.166667, 0.333333]

    The weights account for every ordering exactly once, so multiplying each by
    the number of subsets of that size gives 1:

    >>> import numpy as np
    >>> from math import comb
    >>> n = 14
    >>> sizes = np.arange(n)
    >>> counts = np.array([comb(n - 1, int(s)) for s in sizes])
    >>> float(np.sum(counts * coalition_weights(n))).__round__(9)
    1.0
    """
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be at least 1, got {n_nodes}.")
    total = factorial(n_nodes)
    return np.array(
        [factorial(s) * factorial(n_nodes - s - 1) / total for s in range(n_nodes)],
        dtype=np.float64,
    )


def mean_window_baseline(dataset: WindowedDataset) -> np.ndarray:
    """The average window of each sensor, used to stand in for an absent one.

    ASSUMPTION A-20. Averaging over the training windows gives, for each
    sensor, the shape a typical engine's reading traces out. Substituting it
    says "this sensor is behaving unremarkably", which is the closest thing to
    absence a network with a fixed input size can be shown.

    Parameters
    ----------
    dataset : WindowedDataset
        The **training** windows. Using validation or test windows here would
        leak information into the explanation, in the same way and for the same
        reason it would leak into the scaling.

    Returns
    -------
    numpy.ndarray
        Shaped ``(nodes, window)``, matching one example's features.
    """
    if len(dataset) == 0:
        raise ValueError("cannot build a baseline from an empty dataset.")
    return dataset.features.mean(axis=0).astype(np.float32)


def _masked_batch(
    features: np.ndarray, baseline: np.ndarray, masks: np.ndarray
) -> np.ndarray:
    """Build one input per coalition, keeping present sensors and replacing the rest."""
    # masks: (n_masks, nodes) of bool. Broadcasting over the window axis keeps
    # or replaces a sensor's whole history, never part of it.
    keep = masks[:, :, None]
    return np.where(keep, features[None, :, :], baseline[None, :, :])


@torch.no_grad()
def shap_values(
    model: ESTGNN,
    features: np.ndarray,
    baseline: np.ndarray,
    *,
    batch_size: int = 8192,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Exact Shapley values for one or more windows (Equation 15).

    Parameters
    ----------
    model : ESTGNN
        A trained network. Explaining an untrained one is meaningless, though
        nothing here prevents it.
    features : numpy.ndarray
        One window shaped ``(nodes, window)``, or several shaped
        ``(examples, nodes, window)``.
    baseline : numpy.ndarray
        The stand-in for an absent sensor, shaped ``(nodes, window)``. Usually
        from :func:`mean_window_baseline`.
    batch_size : int
        How many coalitions to evaluate at once. Affects speed and memory only.
    device : torch.device or str, optional
        Where to run. Defaults to wherever the model already is.

    Returns
    -------
    numpy.ndarray
        Shaped ``(nodes,)`` for a single window, or ``(examples, nodes)`` for
        several. Units are flight cycles: a value of ``+4`` means this sensor's
        actual readings pushed the predicted remaining life four cycles higher
        than a typical engine's readings would have.

    Notes
    -----
    Cost is ``2 ** nodes`` model evaluations per window, which is 16,384 for
    the 14 sensors of this project. That is exact, not sampled.
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

    results = np.empty((batch.shape[0], n_nodes), dtype=np.float64)
    for index, example in enumerate(batch):
        inputs = _masked_batch(example.astype(np.float32), baseline, masks)
        predictions = np.empty(len(inputs), dtype=np.float64)
        for start in range(0, len(inputs), batch_size):
            chunk = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            predictions[start : start + batch_size] = (
                model.predict_cycles(chunk).cpu().numpy()
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
    """How far the attribution is from adding up (Equation 14).

    Shapley values must satisfy ``sum(phi) == f(x) - f(baseline)`` exactly.
    This returns the largest absolute violation across the given examples, in
    flight cycles. Anything beyond floating-point noise means the attribution
    is wrong.
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
    """Shapley values at every cycle of one engine's life (Equation 16).

    This is what makes the analysis "dynamic": the same calculation repeated at
    each time step, so the result shows how the model's reliance on each sensor
    changed as the engine degraded.

    Parameters
    ----------
    model : ESTGNN
        A trained network.
    dataset : WindowedDataset
        Windows to draw from, usually the training or validation set, which
        keep every window of every engine. The test set holds only one window
        per engine, so it cannot show an engine's trajectory.
    baseline : numpy.ndarray
        From :func:`mean_window_baseline`.
    unit : int
        Which engine to explain.
    symbols : sequence of str
        Sensor names in node order, for labelling the columns.

    Returns
    -------
    pandas.DataFrame
        One row per window, in cycle order. Columns: ``end_cycle``, the true
        ``rul`` at that cycle, ``predicted``, ``baseline``, and one column per
        sensor holding its Shapley value in cycles.

        The paper's Table 12 is this table for one engine, with the columns
        labelled ``G1``, ``G2`` and so on instead of by cycle.
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
    """Rank the sensors by how much they mattered overall.

    Parameters
    ----------
    values : numpy.ndarray
        Shaped ``(examples, nodes)``, from :func:`shap_values` or the sensor
        columns of :func:`dynamic_shap`.
    symbols : sequence of str
        Sensor names in node order.

    Returns
    -------
    pandas.DataFrame
        One row per sensor, sorted by ``mean_abs`` descending. Columns:

        ``mean_abs``
            Average size of the sensor's contribution, ignoring direction.
            This is the usual measure of "importance", and it is what the
            paper's Table 12 reports and what orders the rows of its Fig. 9.
        ``mean``
            Average signed contribution. A large negative value means the
            sensor consistently pulled the predicted life *down*.
        ``max_abs``
            The single largest contribution the sensor ever made.
    """
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
