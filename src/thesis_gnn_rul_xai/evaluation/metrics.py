"""The four measures used to score predictions.

Every error in this file is predicted minus actual, so a positive error means
the model promised more life than the engine had.

Read README.md in this folder. It explains everything this file does.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

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

# Divisor for a late prediction (predicted > actual). The smaller of the two,
# so late predictions are punished harder. From the 2008 NASA PHM challenge.
LATE_DIVISOR: float = 10.0

# Divisor for an early prediction (predicted < actual).
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
    """Return predicted minus actual, one value per engine."""
    truth, guess = _as_pair(actual, predicted)
    return guess - truth


def rmse(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> float:
    """Root mean squared error, in flight cycles. The paper's Equation 22."""
    error = prediction_error(actual, predicted)
    return float(np.sqrt(np.mean(error**2)))


def nasa_score(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> float:
    """The 2008 NASA challenge score. Lower is better; a perfect run scores 0.

    ASSUMPTION A-15: punish late predictions harder than early ones. The
    paper's Equations 20 and 21 disagree with each other about which direction
    is the dangerous one, so we follow the words next to them. Explained in
    README.md in this folder; written up in docs/ASSUMPTIONS.md.
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
    """How much of the variation in the labels the model explains. 1 is perfect.

    ASSUMPTION A-16: use the standard formula. The paper's Equation 23 divides
    by the spread of the predictions instead of the spread of the labels, which
    is not this measure. Explained in README.md in this folder; written up in
    docs/ASSUMPTIONS.md.
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
    """Average shortfall over the engines predicted early, in cycles. The paper's Equation 24."""
    truth, guess = _as_pair(actual, predicted)
    early = guess < truth
    if not early.any():
        # ASSUMPTION A-39: with no engine predicted early, Equation 24 divides
        # zero by zero, so the answer is "not a number", with a warning, rather
        # than 0.0, which would read as the best value the measure can take. The
        # paper never says what the measure gives then. Explained in README.md
        # in this folder; written up in docs/ASSUMPTIONS.md.
        warnings.warn(
            "ASUE is undefined here: no engine was predicted early (below its true "
            "remaining life), so Equation 24 divides 0 by 0. Returning nan.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    return float(np.mean(truth[early] - guess[early]))


def evaluate(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Compute all four measures at once. Returns a dictionary."""
    truth, guess = _as_pair(actual, predicted)
    error = guess - truth
    return {
        "rmse": rmse(truth, guess),
        "score": nasa_score(truth, guess),
        "r2": r_squared(truth, guess),
        "asue": asue(truth, guess),
        "n_engines": int(truth.size),
        "mean_error": float(error.mean()),
        # Same boundary nasa_score uses: d >= 0 is the late side. Counting
        # only d > 0 would disagree with the score on an exactly-zero error.
        "n_late": int((error >= 0).sum()),
        # The engines ASUE averages over, Equation 24's N_under: the other side
        # of the same boundary, so n_early + n_late = n_engines. When it is 0,
        # ASUE is not a number (assumption A-39) and this count says why.
        "n_early": int((error < 0).sum()),
    }
