"""
The four evaluation measures
============================

Wen et al. (2026) report four numbers for every experiment, and their Section
3.3 defines each one. This module implements all four so that a result from
this code can be placed next to their tables.

Everything here works on plain NumPy arrays. Evaluation deliberately does not
depend on PyTorch: the metrics are useful for a baseline, a published number
copied out of a table, or a model written in any framework.

The sign convention, and why it is not the paper's
--------------------------------------------------
Every measure below is built from the difference between a prediction and the
truth, so the direction of that subtraction decides everything. This module
uses

    d = predicted - actual

so that **d > 0 means the model predicted more remaining life than the engine
had**. That is the dangerous direction: an engine kept flying past the point
where it should have been serviced.

The paper's Equation 20 defines the opposite, ``delta = actual - predicted``,
and then its Equation 21 assigns ``e^(delta/10) - 1`` to ``delta >= 0`` and
``e^(-delta/13) - 1`` to ``delta < 0``. Substituting their own definition into
their own formula gives the gentler divisor of 13 to over-prediction and the
harsher divisor of 10 to under-prediction, which is the reverse of what the
surrounding text says ("the penalty for over-prediction is relatively large ...
over-prediction introduces security risks") and the reverse of the scoring
function published for the 2008 NASA PHM Data Challenge that the paper cites.

The two printed equations are therefore inconsistent with each other. This
module implements the intent the paper states in words, which is also the
standard function every other work on this dataset uses, so that the numbers
are comparable. See assumption A-15.

What each measure is for
------------------------
``rmse``
    Root mean squared error, in flight cycles. Symmetric: it does not care
    whether an error is early or late. The headline number in most papers.

``nasa_score``
    The official challenge score. Asymmetric and exponential, so it punishes
    late predictions far more than early ones, and a single very wrong engine
    can dominate the total. Lower is better. Note that it is a **sum**, not an
    average, so it grows with the size of the test set and can only be compared
    across runs on the same sub-dataset.

``r_squared``
    The fraction of the variance in the true labels that the model explains. 1
    is perfect; 0 is no better than always predicting the mean.

``asue``
    Average Safe Underestimation Error. The mean size of the error, counted
    only over the engines the model was *pessimistic* about. Underestimating is
    the safe direction, but doing it by a wide margin means servicing engines
    that still had life in them, so this measures the cost of excess caution.

References
----------
Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation
modeling for aircraft engine run-to-failure simulation. *2008 International
Conference on Prognostics and Health Management*, Denver, CO, pp. 1-9. IEEE.
https://doi.org/10.1109/PHM.2008.4711414

Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
168, 110875. https://doi.org/10.1016/j.ast.2025.110875
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..utils.logging_setup import get_logger

__all__ = [
    "EARLY_DIVISOR",
    "LATE_DIVISOR",
    "asue",
    "evaluate",
    "nasa_score",
    "prediction_error",
    "r_squared",
    "rmse",
]

logger = get_logger(__name__)

#: Divisor for a late prediction (predicted > actual). The smaller of the two,
#: so late predictions are punished harder. From the 2008 NASA PHM challenge.
LATE_DIVISOR: float = 10.0

#: Divisor for an early prediction (predicted < actual).
EARLY_DIVISOR: float = 13.0


def _as_pair(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert a pair of label arrays."""
    truth = np.asarray(actual, dtype=float).ravel()
    guess = np.asarray(predicted, dtype=float).ravel()
    if truth.shape != guess.shape:
        raise ValueError(
            f"actual and predicted must have the same length, got {truth.size} "
            f"and {guess.size}."
        )
    if truth.size == 0:
        raise ValueError("cannot evaluate an empty set of predictions.")
    if not np.isfinite(truth).all() or not np.isfinite(guess).all():
        raise ValueError(
            "actual and predicted must be finite. A NaN here usually means the "
            "model diverged during training."
        )
    return truth, guess


def prediction_error(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return ``predicted - actual``, one value per engine.

    Positive means the model was optimistic: it claimed more life than the
    engine had. See the module docstring for why the subtraction runs this way
    and not the way the paper's Equation 20 prints it.

    Examples
    --------
    >>> prediction_error([100, 50], [90, 60]).tolist()
    [-10.0, 10.0]
    """
    truth, guess = _as_pair(actual, predicted)
    return guess - truth


def rmse(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> float:
    """Root mean squared error in flight cycles (the paper's Equation 22).

    Examples
    --------
    >>> round(rmse([100, 50, 20], [110, 40, 20]), 4)
    8.165
    """
    error = prediction_error(actual, predicted)
    return float(np.sqrt(np.mean(error**2)))


def nasa_score(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> float:
    """The 2008 NASA PHM challenge score (the paper's Equation 21, SF).

    ASSUMPTION A-15: the paper's Equations 20 and 21 contradict each other
    about which direction of error is the dangerous one. The convention
    implemented here is the one its own prose describes and the one the
    2008 challenge defines. See the module docstring for the argument.

    For each engine::

        d < 0  (early, safe)      ->  exp(-d / 13) - 1
        d >= 0 (late, dangerous)  ->  exp( d / 10) - 1

    and the results are **summed**, not averaged. Lower is better; a perfect
    prediction scores 0.

    Because the penalty is exponential, one badly wrong engine can outweigh
    every other. That is deliberate: in maintenance a single missed failure
    matters more than many small errors.

    Examples
    --------
    Ten cycles late costs far more than ten cycles early:

    >>> round(nasa_score([100], [110]), 4)
    1.7183
    >>> round(nasa_score([100], [90]), 4)
    1.1581
    >>> nasa_score([100, 50], [100, 50])
    0.0
    """
    error = prediction_error(actual, predicted)
    penalties = np.where(
        error < 0,
        np.expm1(-error / EARLY_DIVISOR),
        np.expm1(error / LATE_DIVISOR),
    )
    return float(np.sum(penalties))


def r_squared(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> float:
    """Coefficient of determination (the paper's Equation 23).

    Computed as ``1 - sum((predicted - actual)^2) / sum((actual - mean)^2)``.

    ASSUMPTION A-16: the paper's Equation 23 prints the denominator as
    ``sum((predicted_i - mean)^2)``, measuring how far the *predictions* spread
    around the mean of the observations rather than how far the observations
    spread around it. That is not the coefficient of determination, and it
    rewards the wrong thing: a model can raise its score simply by spreading
    its output more widely, without becoming any more accurate. Predicting
    ``[-5, 15, 35, 55]`` for truth ``[10, 20, 30, 40]`` is no better than
    predicting the mean --- the standard formula scores it 0.0 --- but the
    paper's version scores it 0.75.

    The standard definition is used here, which is also what the text beside
    the equation describes ("the proportion of variance in the dependent
    variable that is predictable from the independent variables").

    Examples
    --------
    >>> r_squared([1, 2, 3, 4], [1, 2, 3, 4])
    1.0
    >>> round(r_squared([1, 2, 3, 4], [2.5, 2.5, 2.5, 2.5]), 4)
    0.0
    """
    truth, guess = _as_pair(actual, predicted)
    total = float(np.sum((truth - truth.mean()) ** 2))
    if total <= 0.0:
        raise ValueError(
            "R-squared is undefined when every true label is the same: the "
            "denominator, the total variance of the labels, is zero."
        )
    residual = float(np.sum((guess - truth) ** 2))
    return 1.0 - residual / total


def asue(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> float:
    """Average Safe Underestimation Error (the paper's Equation 24).

    The mean of ``actual - predicted`` over the engines where the model
    predicted *less* life than there was. Engines the model was optimistic
    about do not contribute.

    Returns
    -------
    float
        The average size of the safe errors, in cycles. ``0.0`` when the model
        never underestimated, which is the best possible value of this measure
        and not an error; a message is logged so the empty case is visible.

    Examples
    --------
    Only the two pessimistic engines count, and their shortfalls are 10 and 20:

    >>> asue([100, 50, 30], [90, 60, 10])
    15.0
    >>> asue([100], [110])
    0.0
    """
    truth, guess = _as_pair(actual, predicted)
    underestimated = guess < truth
    if not underestimated.any():
        logger.info(
            "ASUE is 0.0 because the model never underestimated. That is the "
            "best value this measure can take, not a missing result."
        )
        return 0.0
    return float(np.mean(truth[underestimated] - guess[underestimated]))


def evaluate(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Compute all four measures at once.

    Returns
    -------
    dict
        Keys ``"rmse"``, ``"score"``, ``"r2"``, ``"asue"``, ``"n_engines"``,
        and two extras that make a result easier to read at a glance:
        ``"mean_error"`` (positive means the model is optimistic overall) and
        ``"n_late"`` (how many engines it was optimistic about, which is the
        count that matters for safety).

    Examples
    --------
    The third engine is predicted exactly right, and counts as late, because
    ``nasa_score`` puts the boundary at ``d >= 0``:

    >>> result = evaluate([100, 50, 30], [95, 55, 30])
    >>> round(result["rmse"], 4)
    4.0825
    >>> result["n_late"]
    2
    """
    truth, guess = _as_pair(actual, predicted)
    error = guess - truth
    return {
        "rmse": rmse(truth, guess),
        "score": nasa_score(truth, guess),
        "r2": r_squared(truth, guess),
        "asue": asue(truth, guess),
        "n_engines": int(truth.size),
        "mean_error": float(error.mean()),
        # The same boundary nasa_score uses: d >= 0 is the late, dangerous
        # side. Counting only d > 0 here would disagree with the score on
        # an exactly-zero error.
        "n_late": int((error >= 0).sum()),
    }
