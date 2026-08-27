"""
Signal smoothing
================

Why smoothing is applied
------------------------
The sensor readings contain measurement noise: small, fast fluctuations that
have nothing to do with engine wear. Wear itself is slow and gradual. Removing
the fast fluctuations makes the slow trend easier for a model to see.

The filter used: exponentially weighted moving average
------------------------------------------------------
An *exponentially weighted moving average* (EWMA) replaces each reading with a
blend of that reading and the smoothed value from the previous cycle::

    S(t) = alpha * X(t) + (1 - alpha) * S(t-1)

where ``X(t)`` is the raw reading at cycle ``t``, ``S(t)`` is the smoothed
value, and ``alpha`` is a number between 0 and 1 that controls the strength of
the filter:

* ``alpha`` close to 1 means the smoothed value follows the raw signal almost
  exactly, so almost no smoothing happens.
* ``alpha`` close to 0 means the smoothed value changes very slowly, so the
  filter is strong but reacts late to a genuine change.

The name comes from expanding the formula: every past reading contributes, and
its weight shrinks exponentially the further back it lies.

The filter only looks backwards
-------------------------------
``S(t)`` depends on ``X(t)`` and on earlier values, never on later ones. This
matters for the honesty of the experiment. A filter that averaged over future
readings would leak information about the future into the present, and the
model would appear to predict failure better than it really can. Such leakage
is a well-known way to produce results that cannot be reproduced in operation.
The filter is also applied to each engine separately, so no reading from one
engine can influence another.

A note on the implementation
----------------------------
The pandas method ``ewm`` has two modes. With ``adjust=True`` (its default) it
applies a correction to the first few values; with ``adjust=False`` it uses the
plain recursive formula written above. This module sets ``adjust=False``,
because that is the formula given as Equation 18 in Wen et al. (2026). The two
modes give visibly different values at the start of a series, so this is not a
cosmetic detail.

The value of alpha
------------------
Section 3.2.3 of Wen et al. (2026) is self-contradictory on this point. It
states that "S(t-1) ... is equal to 0.5 in this study", which attaches the
number 0.5 to the previous smoothed value rather than to the smoothing
parameter alpha. Since a fixed previous value would make no sense in a
recursive formula, the sentence is read here as alpha = 0.5. The choice is
recorded in ``docs/ASSUMPTIONS.md``.

References
----------
Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
168, 110875. https://doi.org/10.1016/j.ast.2025.110875
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..utils.logging_setup import get_logger
from .sensors import CYCLE_COL, UNIT_COL

__all__ = ["apply_ewma", "ewma_series"]

logger = get_logger(__name__)


# ASSUMPTION A-09: pandas offers two conventions for this filter, which differ
# in how they weight the first few values of a series. ``adjust=False`` is the
# plain recursive form, which is exactly Equation 18 of the paper.
#
# ASSUMPTION A-04: the value of alpha itself. See config.PreprocessingConfig.
def ewma_series(values: Sequence[float] | np.ndarray, alpha: float) -> np.ndarray:
    """Apply the exponentially weighted moving average to a single series.

    This is a direct, readable implementation of the recursive formula. It is
    used by the tests to confirm that the vectorised pandas version produces
    the same numbers.

    Parameters
    ----------
    values : sequence of float
        The raw readings, in time order.
    alpha : float
        Smoothing parameter in the interval (0, 1].

    Returns
    -------
    numpy.ndarray
        The smoothed series, the same length as the input. The first smoothed
        value equals the first raw value.

    Examples
    --------
    >>> ewma_series([0.0, 1.0, 1.0], alpha=0.5).tolist()
    [0.0, 0.5, 0.75]
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}.")

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array

    smoothed = np.empty_like(array)
    smoothed[0] = array[0]
    for index in range(1, array.size):
        smoothed[index] = alpha * array[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def apply_ewma(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    alpha: float | None,
    unit_column: str = UNIT_COL,
    cycle_column: str = CYCLE_COL,
) -> pd.DataFrame:
    """Smooth selected columns of a C-MAPSS frame, engine by engine.

    Parameters
    ----------
    frame : pandas.DataFrame
        A loaded C-MAPSS frame.
    columns : sequence of str
        The columns to smooth, normally the 14 retained sensor columns.
    alpha : float or None
        Smoothing parameter. ``None`` disables smoothing and returns an
        unchanged copy, which makes it easy to measure how much the filter
        contributes to the final result.
    unit_column, cycle_column : str
        Column names identifying the engine and the time step.

    Returns
    -------
    pandas.DataFrame
        A copy of the input with the selected columns replaced by their
        smoothed versions. All other columns, including the label, are
        untouched.
    """
    if alpha is None:
        logger.info("Smoothing disabled (alpha is None); data passed through unchanged.")
        return frame.copy()
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}.")

    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"Columns not present in the frame: {missing}.")

    frame = frame.sort_values([unit_column, cycle_column]).copy()
    frame[list(columns)] = (
        frame.groupby(unit_column, sort=False)[list(columns)]
        .transform(lambda column: column.ewm(alpha=alpha, adjust=False).mean())
    )

    logger.info(
        "Applied exponentially weighted moving average (alpha=%.3f) to %d columns "
        "across %d engines.",
        alpha, len(columns), frame[unit_column].nunique(),
    )
    return frame
