"""
Remaining Useful Life labels
============================

What the label means
--------------------
*Remaining Useful Life* (RUL) is the number of flight cycles an engine will
still run before it fails. It is the quantity the model has to predict. The
label is measured in cycles, so a value of 47 means "this engine will fail 47
flight cycles from now".

Why the label is capped: the piecewise-linear target
----------------------------------------------------
If the label were simply "cycles until failure", a brand-new engine in a long
history would carry a label of 300 or more. That creates a problem, because a
new engine gives the model no way to tell 300 from 280: the sensors of a
healthy engine look the same either way, since almost no wear has happened yet.
Asking a model to distinguish them forces it to guess, and the guessing shows
up as noise in every prediction it makes.

The standard solution, introduced by Heimes (2008), is a **piecewise-linear**
target. The label is held at a constant ceiling while the engine is healthy,
and only starts to fall once real degradation begins::

    RUL(t) = min(cycles_until_failure(t), RUL_max)

Drawn against time, this is a flat line followed by a straight decline. The
point where the flat part meets the sloping part is often called the *change
point*. Heimes reported that any ceiling between roughly 120 and 130 cycles
works well on this dataset, and 125 has since become the most common choice.

The ceiling in the paper being reproduced
-----------------------------------------
Wen et al. (2026) give this formula as their Equation 19 but never print the
value they used for the ceiling. Section 4.2 of their paper describes predicted
life holding "around 125" while an engine is healthy, which indicates 125. The
same value is used by Jin et al. (2025) on the same dataset. This project
therefore uses 125 and records the choice as an assumption.

References
----------
Heimes, F. O. (2008). Recurrent neural networks for remaining useful life
estimation. *2008 International Conference on Prognostics and Health
Management*, pp. 1-6. IEEE. https://doi.org/10.1109/PHM.2008.4711422

Jin, Y., Meng, Z., Hu, Q., & Chen, D. (2025). Heterogeneous dynamic-aware GNN
for RUL prediction of aeroengine. *IEEE Transactions on Instrumentation and
Measurement*, 74, 3538014. https://doi.org/10.1109/TIM.2025.3569885
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["apply_piecewise_rul", "piecewise_rul_curve"]


def apply_piecewise_rul(
    frame: pd.DataFrame,
    *,
    rul_max: int,
    column: str = "rul",
    output_column: str | None = None,
) -> pd.DataFrame:
    """Cap a Remaining Useful Life column at a ceiling value.

    Parameters
    ----------
    frame : pandas.DataFrame
        A frame that already contains an uncapped RUL column, as produced by
        :func:`thesis_gnn_rul_xai.data.loading.attach_rul`.
    rul_max : int
        The ceiling, in flight cycles.
    column : str
        Name of the uncapped column to read.
    output_column : str, optional
        Name of the column to write. Defaults to ``column``, replacing it in
        place. Passing a different name keeps both versions, which is useful
        for plotting the effect of the cap.

    Returns
    -------
    pandas.DataFrame
        A copy of the input with the capped column added or replaced.

    Examples
    --------
    An engine with 300 cycles of life left is labelled 125 when the ceiling is
    125, while an engine with 47 cycles left keeps its true value:

    >>> import pandas as pd
    >>> frame = pd.DataFrame({"rul": [300, 130, 125, 47, 0]})
    >>> apply_piecewise_rul(frame, rul_max=125)["rul"].tolist()
    [125, 125, 125, 47, 0]
    """
    if rul_max < 1:
        raise ValueError(f"rul_max must be positive, got {rul_max}.")
    if column not in frame.columns:
        raise KeyError(
            f"Column {column!r} not found. Available columns: {list(frame.columns)}."
        )

    frame = frame.copy()
    frame[output_column or column] = np.minimum(frame[column].to_numpy(), rul_max)
    return frame


def piecewise_rul_curve(total_cycles: int, rul_max: int) -> np.ndarray:
    """Return the ideal label curve for one engine, for plotting or testing.

    Parameters
    ----------
    total_cycles : int
        Total lifetime of the engine, in cycles.
    rul_max : int
        The ceiling applied to the label.

    Returns
    -------
    numpy.ndarray
        An array of length ``total_cycles``. Element ``i`` is the label at
        cycle ``i + 1``. The first elements equal ``rul_max`` and the last
        element is 0.

    Examples
    --------
    >>> piecewise_rul_curve(6, rul_max=3).tolist()
    [3, 3, 3, 2, 1, 0]
    """
    cycles = np.arange(1, total_cycles + 1)
    return np.minimum(total_cycles - cycles, rul_max)
