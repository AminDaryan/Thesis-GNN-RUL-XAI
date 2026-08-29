"""Caps the label at a ceiling, so a healthy engine is just labelled "healthy".

Read README.md in this folder. It explains why the cap is there and where the
number comes from.
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
    """Cap the remaining-life column at a ceiling. Anything above it becomes it.

    ASSUMPTION A-01 fixes the ceiling at 125 cycles; see config.py.
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
    """The label curve one engine should have, for drawing or for testing."""
    cycles = np.arange(1, total_cycles + 1)
    return np.minimum(total_cycles - cycles, rul_max)
