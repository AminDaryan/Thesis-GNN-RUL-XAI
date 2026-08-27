"""
Rescaling the sensor values (normalisation)
===========================================

Why rescaling is needed
-----------------------
The 14 sensors measure completely different physical quantities on completely
different scales. A turbine outlet temperature is around 1400 degrees Rankine,
a bypass ratio is around 8.4, and a coolant bleed flow is around 39 pounds per
second. If these numbers are fed to a neural network unchanged, the large ones
dominate the calculation simply because they are large, not because they are
informative. Rescaling puts every sensor on a comparable footing.

The three methods provided
--------------------------
``minmax_global``
    Each sensor is mapped so its smallest observed value becomes 0 and its
    largest becomes 1::

        scaled = (value - minimum) / (maximum - minimum)

``minmax_per_condition``
    The same formula, but with a separate minimum and maximum for each
    *operating condition*. See the next section.

``zscore``
    Each sensor is shifted and stretched so that it has an average of 0 and a
    standard deviation of 1::

        scaled = (value - mean) / standard_deviation

    Unlike min-max scaling, this does not bound the values to a fixed range,
    which makes it more tolerant of outliers.

What an operating condition is, and why it matters
--------------------------------------------------
The three "operational setting" columns describe how the aircraft was being
flown: its altitude, its speed (as a Mach number), and the throttle position.
Flying higher or faster changes every sensor reading immediately and by a large
amount, and that change has nothing to do with engine wear.

In sub-datasets FD002 and FD004 the engines are flown under six different
conditions, and the jumps between conditions are far larger than the slow drift
caused by degradation. Scaling all six together leaves the wear signal buried.
Scaling each condition separately removes the flight-related jumps and leaves
the wear visible. FD001, which this project starts with, has only one operating
condition, so per-condition scaling and global scaling give identical results
there. Implementing the general version now means the code will already be
correct when FD002 and FD004 are added later.

Conditions are identified with **k-means clustering**, a standard algorithm
that groups points into a chosen number of clusters by repeatedly assigning
each point to the nearest cluster centre and then recomputing the centres
(MacQueen, 1967). The three setting columns are clustered, and the resulting
cluster number is used as the condition label. This follows Jin et al. (2025),
who use the same approach on the same dataset.

The rule that protects the experiment: fit on training data only
-----------------------------------------------------------------
All statistics -- minima, maxima, means, standard deviations, and the cluster
centres themselves -- are computed from the **training** engines only, and then
applied unchanged to the test engines. This is not a stylistic preference; it
is what makes the evaluation valid.

If the maximum were computed over training and test data together, information
about the test engines would enter the training process. The model would then
be evaluated on data it had partly seen, and the reported error would be
smaller than the error the same model would produce on genuinely new engines.
This mistake is called **data leakage** and is one of the most common causes of
results that cannot be reproduced (Kapoor & Narayanan, 2023).

What the source paper says about this
-------------------------------------
Nothing. Wen et al. (2026) never describe how they scale their inputs, and
never mention operating conditions in their preprocessing. The default used
here follows Jin et al. (2025). This is a documented substitution, not a claim
about what the original authors did; see ``docs/ASSUMPTIONS.md``.

References
----------
Jin, Y., Meng, Z., Hu, Q., & Chen, D. (2025). Heterogeneous dynamic-aware GNN
for RUL prediction of aeroengine. *IEEE Transactions on Instrumentation and
Measurement*, 74, 3538014. https://doi.org/10.1109/TIM.2025.3569885

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in
machine-learning-based science. *Patterns*, 4(9), 100804.
https://doi.org/10.1016/j.patter.2023.100804

MacQueen, J. (1967). Some methods for classification and analysis of
multivariate observations. *Proceedings of the Fifth Berkeley Symposium on
Mathematical Statistics and Probability*, 1, 281-297.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..utils.logging_setup import get_logger
from .sensors import OP_SETTING_COLS

__all__ = [
    "CONDITION_COL",
    "OperatingConditionEncoder",
    "SensorNormalizer",
]

logger = get_logger(__name__)

#: Name of the column that holds the operating-condition label.
CONDITION_COL = "operating_condition"

# Values whose range is smaller than this are treated as constant. Scaling a
# constant signal would mean dividing by zero.
_ZERO_RANGE_TOLERANCE = 1e-12


@dataclass
class OperatingConditionEncoder:
    """Assign an operating-condition label to every row.

    Parameters
    ----------
    n_conditions : int or None
        Number of conditions to look for. ``None`` means "detect it from the
        data" by counting how many distinct rounded setting combinations occur,
        which recovers 1 for FD001 and FD003 and 6 for FD002 and FD004.
    setting_columns : sequence of str
        Which columns describe the flight condition.
    round_decimals : int
        Number of decimal places used when counting distinct combinations
        during automatic detection. The raw settings carry small measurement
        noise, so they must be rounded before distinct values are counted.
    seed : int
        Seed for k-means, which starts from randomly chosen centres.

    Notes
    -----
    When only one condition is detected, k-means is skipped entirely and every
    row is labelled 0. This avoids a pointless computation on FD001 and makes
    the behaviour easy to verify.
    """

    n_conditions: int | None = None
    setting_columns: Sequence[str] = tuple(OP_SETTING_COLS)
    round_decimals: int = 0
    seed: int = 42

    _model: KMeans | None = field(default=None, init=False, repr=False)
    _n_detected: int | None = field(default=None, init=False, repr=False)

    @property
    def n_detected(self) -> int:
        """How many conditions this encoder ended up using."""
        if self._n_detected is None:
            raise RuntimeError("The encoder must be fitted before use.")
        return self._n_detected

    def fit(self, frame: pd.DataFrame) -> OperatingConditionEncoder:
        """Learn the conditions from the training data.

        Parameters
        ----------
        frame : pandas.DataFrame
            Training rows only. Never pass test data here.
        """
        settings = frame[list(self.setting_columns)].to_numpy(dtype=float)

        if self.n_conditions is None:
            rounded = np.round(settings, self.round_decimals)
            detected = len(np.unique(rounded, axis=0))
            # Guard against noise creating spurious groups.
            detected = int(np.clip(detected, 1, 6))
            logger.info(
                "Detected %d distinct operating condition(s) from the settings.",
                detected,
            )
        else:
            detected = int(self.n_conditions)

        self._n_detected = detected

        if detected <= 1:
            self._model = None
            logger.info("A single operating condition: clustering is not needed.")
            return self

        self._model = KMeans(n_clusters=detected, n_init=10, random_state=self.seed)
        self._model.fit(settings)
        logger.info("Fitted k-means with %d clusters on the operating settings.", detected)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add the operating-condition column to a frame.

        Returns
        -------
        pandas.DataFrame
            A copy of the input with one extra integer column, named by
            :data:`CONDITION_COL`.
        """
        if self._n_detected is None:
            raise RuntimeError("Call fit() before transform().")

        frame = frame.copy()
        if self._model is None:
            frame[CONDITION_COL] = 0
        else:
            settings = frame[list(self.setting_columns)].to_numpy(dtype=float)
            frame[CONDITION_COL] = self._model.predict(settings).astype(int)
        return frame

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Convenience method: :meth:`fit` followed by :meth:`transform`."""
        return self.fit(frame).transform(frame)


# ASSUMPTION A-03: the paper never describes how its inputs are scaled, so the
# whole of this class is a substitution from Jin et al. (2025), not a
# reconstruction of what Wen et al. (2026) did.
@dataclass
class SensorNormalizer:
    """Rescale sensor columns using statistics learned from training data only.

    Parameters
    ----------
    method : str
        One of ``"minmax_per_condition"``, ``"minmax_global"``, ``"zscore"``.
    columns : sequence of str
        The sensor columns to rescale.
    feature_range : tuple of float
        Target range for the two min-max methods. The default ``(0.0, 1.0)``
        matches Jin et al. (2025).

    Attributes
    ----------
    statistics_ : pandas.DataFrame
        The learned statistics, one row per (condition, sensor) pair. Exposed
        so that they can be inspected, plotted, and saved alongside results.
    """

    method: str = "minmax_per_condition"
    columns: Sequence[str] = ()
    feature_range: tuple[float, float] = (0.0, 1.0)

    statistics_: pd.DataFrame | None = field(default=None, init=False, repr=False)

    _VALID_METHODS = ("minmax_per_condition", "minmax_global", "zscore")

    def __post_init__(self) -> None:
        if self.method not in self._VALID_METHODS:
            raise ValueError(
                f"Unknown normalisation method {self.method!r}. "
                f"Valid options are: {list(self._VALID_METHODS)}."
            )
        if not self.columns:
            raise ValueError("At least one column must be given to normalise.")
        low, high = self.feature_range
        if not low < high:
            raise ValueError(
                f"feature_range must be increasing, got ({low}, {high})."
            )

    # -- fitting ----------------------------------------------------------

    def fit(self, frame: pd.DataFrame) -> SensorNormalizer:
        """Learn the scaling statistics from training data.

        Parameters
        ----------
        frame : pandas.DataFrame
            Training rows only. For ``"minmax_per_condition"`` the frame must
            already contain the operating-condition column produced by
            :class:`OperatingConditionEncoder`.
        """
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise KeyError(f"Columns not present in the frame: {missing}.")

        if self.method == "minmax_per_condition":
            if CONDITION_COL not in frame.columns:
                raise KeyError(
                    f"Column {CONDITION_COL!r} is required for per-condition "
                    "scaling. Run OperatingConditionEncoder first."
                )
            groups = frame.groupby(CONDITION_COL, sort=True)
        else:
            groups = [(0, frame)]  # type: ignore[assignment]

        records: list[dict[str, object]] = []
        for condition, group in groups:
            for column in self.columns:
                values = group[column].to_numpy(dtype=float)
                records.append(
                    {
                        "condition": int(condition),
                        "column": column,
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=0)),
                        "n_rows": int(values.size),
                    }
                )

        self.statistics_ = pd.DataFrame.from_records(records)
        self._warn_about_degenerate_columns()
        logger.info(
            "Fitted '%s' normalisation on %d column(s) across %d condition(s).",
            self.method,
            len(self.columns),
            self.statistics_["condition"].nunique(),
        )
        return self

    def _warn_about_degenerate_columns(self) -> None:
        """Report any sensor that is constant within a condition.

        A constant sensor cannot be min-max scaled, because the formula would
        divide by zero. Such columns are mapped to the bottom of the target
        range instead. This is worth a warning rather than silence, because a
        constant sensor in the retained set may mean the sensor selection needs
        to be revisited.
        """
        assert self.statistics_ is not None
        stats = self.statistics_
        if self.method == "zscore":
            degenerate = stats[stats["std"] <= _ZERO_RANGE_TOLERANCE]
        else:
            degenerate = stats[(stats["max"] - stats["min"]) <= _ZERO_RANGE_TOLERANCE]

        for _, row in degenerate.iterrows():
            logger.warning(
                "Sensor column %s is constant within operating condition %d. "
                "It will be mapped to a fixed value, and it carries no "
                "information for the model.",
                row["column"],
                row["condition"],
            )

    # -- applying ---------------------------------------------------------

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned scaling to any frame, training or test.

        Returns
        -------
        pandas.DataFrame
            A copy of the input with the selected columns rescaled. All other
            columns, including the label, are unchanged.
        """
        if self.statistics_ is None:
            raise RuntimeError("Call fit() before transform().")

        frame = frame.copy()
        low, high = self.feature_range

        if self.method == "minmax_per_condition":
            if CONDITION_COL not in frame.columns:
                raise KeyError(
                    f"Column {CONDITION_COL!r} is required for per-condition scaling."
                )
            condition_values = frame[CONDITION_COL].to_numpy()
        else:
            condition_values = np.zeros(len(frame), dtype=int)

        indexed = self.statistics_.set_index(["condition", "column"])

        for column in self.columns:
            raw = frame[column].to_numpy(dtype=float)
            scaled = np.empty_like(raw)

            for condition in np.unique(condition_values):
                mask = condition_values == condition
                try:
                    stats = indexed.loc[(int(condition), column)]
                except KeyError as exc:
                    raise KeyError(
                        f"No statistics were learned for column {column!r} under "
                        f"operating condition {condition}. This happens when the "
                        "test data contains a condition that never occurred in "
                        "training."
                    ) from exc

                if self.method == "zscore":
                    spread = float(stats["std"])
                    scaled[mask] = (
                        (raw[mask] - float(stats["mean"])) / spread
                        if spread > _ZERO_RANGE_TOLERANCE
                        else 0.0
                    )
                else:
                    minimum, maximum = float(stats["min"]), float(stats["max"])
                    span = maximum - minimum
                    if span > _ZERO_RANGE_TOLERANCE:
                        unit = (raw[mask] - minimum) / span
                        scaled[mask] = unit * (high - low) + low
                    else:
                        scaled[mask] = low

            frame[column] = scaled

        return frame

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Convenience method: :meth:`fit` followed by :meth:`transform`."""
        return self.fit(frame).transform(frame)
