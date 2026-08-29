"""Smooths out the jitter in the sensor readings, one engine at a time.

Read README.md in this folder. It explains the filter, why it only ever looks
backwards in time, and where its one setting comes from.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..utils.logging_setup import get_logger
from .sensors import CYCLE_COL, UNIT_COL

__all__ = ["apply_ewma", "ewma_series"]

logger = get_logger(__name__)


# ASSUMPTION A-09: use pandas' adjust=False form of this filter. Pandas offers
# two, which disagree about the first few values of a series; adjust=False is
# the plain formula the paper prints as its Equation 18. Written up in
# docs/ASSUMPTIONS.md, together with A-04, the strength of the filter.
def ewma_series(values: Sequence[float] | np.ndarray, alpha: float) -> np.ndarray:
    """Smooth one series, written out the slow and obvious way.

    The tests use this to confirm the fast pandas version agrees with it.
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
    """Smooth the chosen columns, one engine at a time. alpha=None does nothing.

    Only those columns change. The label is left alone.
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
