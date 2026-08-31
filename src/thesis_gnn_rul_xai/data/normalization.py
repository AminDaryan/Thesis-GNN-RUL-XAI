"""Puts the 14 sensors on a comparable scale before the network sees them.

Read README.md in this folder. It explains the three methods on offer, what an
operating condition is, and the rule that keeps the test engines out of it.
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

# The name of the column holding the operating condition.
CONDITION_COL = "operating_condition"

# A sensor whose readings span less than this counts as constant. Rescaling a
# constant signal would mean dividing by zero.
_ZERO_RANGE_TOLERANCE = 1e-12


@dataclass
class OperatingConditionEncoder:
    """Works out how the aircraft was being flown at each row.

    README.md in this folder explains what an operating condition is and why
    it matters. Leaving n_conditions as None detects the number from the data.
    """

    n_conditions: int | None = None
    setting_columns: Sequence[str] = tuple(OP_SETTING_COLS)
    round_decimals: int = 0
    seed: int = 42

    _model: KMeans | None = field(default=None, init=False, repr=False)
    _n_detected: int | None = field(default=None, init=False, repr=False)

    @property
    def n_detected(self) -> int:
        """How many conditions this ended up using."""
        if self._n_detected is None:
            raise RuntimeError("The encoder must be fitted before use.")
        return self._n_detected

    def fit(self, frame: pd.DataFrame) -> OperatingConditionEncoder:
        """Learn the conditions. Pass training rows only, never test rows."""
        settings = frame[list(self.setting_columns)].to_numpy(dtype=float)

        if self.n_conditions is None:
            rounded = np.round(settings, self.round_decimals)
            detected = len(np.unique(rounded, axis=0))
            # Stop measurement noise from inventing extra groups.
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
        """Add the operating-condition column to a table."""
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
        """Do fit() and then transform(), in one call."""
        return self.fit(frame).transform(frame)


# ASSUMPTION A-03: everything in this class is our own choice. Wen et al.
# (2026) never describe how they put their sensors on a comparable scale, so
# nothing here reconstructs what they did. The default is the z-score, picked
# under the rule in A-22; the per-condition min-max option follows Jin et al.
# (2025) and was the default until 2026-08-29. Written up in
# docs/ASSUMPTIONS.md, where it is measured as the largest single effect in
# the project.
@dataclass
class SensorNormalizer:
    """Rescales the sensors, using numbers measured on the training engines only.

    README.md in this folder describes the three methods and the rule that
    keeps the test engines out of the measurement.
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
        """Measure the training engines. Pass training rows only.

        For "minmax_per_condition" the table must already have been through
        OperatingConditionEncoder above.
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
        """Warn about any sensor that never moves. It cannot be rescaled."""
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
        """Apply what was measured to any table. Only the sensors change."""
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
        """Do fit() and then transform(), in one call."""
        return self.fit(frame).transform(frame)
