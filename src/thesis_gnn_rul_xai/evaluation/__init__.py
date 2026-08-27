"""
Evaluation measures
===================

The four numbers Wen et al. (2026) report for every experiment: root mean
squared error, the NASA challenge score, the coefficient of determination, and
the Average Safe Underestimation Error.

This subpackage works on NumPy arrays and does not need PyTorch, so a published
figure copied out of a table can be scored the same way a model's output is.
"""

from .metrics import (
    EARLY_DIVISOR,
    LATE_DIVISOR,
    asue,
    evaluate,
    nasa_score,
    prediction_error,
    r_squared,
    rmse,
)

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
