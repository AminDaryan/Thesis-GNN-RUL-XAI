"""
Turning histories into training examples (windowing)
====================================================

The idea
--------
A model cannot be given a whole engine history at once, because histories have
different lengths. Instead the history is cut into short, fixed-length pieces
called **time windows**. One window is one training example.

With a window size of 30, the first example covers cycles 1 to 30, the second
covers cycles 2 to 31, and so on. The step between the start of one window and
the start of the next is the **stride**; a stride of 1 produces the largest
possible number of examples.

The label of a window is the Remaining Useful Life at its **last** cycle. This
matches how the model would be used in practice: an operator has the last 30
cycles of readings available and wants to know the remaining life right now,
not 30 cycles ago.

Wen et al. (2026) give a window size of 30 in their Table 6. They do not state
a stride; 1 is used here and recorded as an assumption.

Shape of the result
-------------------
The output is arranged for a Graph Neural Network, in which every sensor is a
**node** of a graph. The array has three axes::

    (number_of_examples, number_of_sensors, features_per_sensor)

With ``node_feature_mode="window"`` the third axis holds the whole window, so
each sensor node carries 30 numbers -- its own recent history. With
``node_feature_mode="last_step"`` the third axis has length 1, holding only the
most recent reading, which reproduces the input shape ``[batch, 14, 1]``
printed in Table 4 of Wen et al. (2026).

How the test set is treated
---------------------------
Test histories stop before failure, and the official label file gives one
number per engine: the remaining life at the last recorded cycle. The
evaluation therefore uses exactly **one window per test engine**, the final
one. This gives 100 predictions on FD001, which matches the "number of testing
samples" column in Table 5 of Wen et al. (2026).

Splitting off a validation set, and why it must be done by engine
-----------------------------------------------------------------
A validation set is a portion of the training data that is held back and never
used to update the model. It is used to decide when to stop training and to
compare settings, so that those decisions are not made by looking at the test
set.

The split must be made **by engine, not by window**. Consecutive windows from
the same engine overlap heavily: with a window size of 30 and a stride of 1,
two neighbouring windows share 29 of their 30 cycles and have labels that
differ by one cycle. If windows were split at random, nearly identical examples
would end up on both sides of the split, and the validation error would be far
too optimistic -- the model would effectively be tested on data it had already
seen. Splitting whole engines keeps the two sets genuinely separate.

Wen et al. (2026) describe no validation split at all; they report training for
a fixed 200 epochs. Introducing one here is a deliberate improvement over the
source, and is recorded as such in ``docs/ASSUMPTIONS.md``.

References
----------
Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
168, 110875. https://doi.org/10.1016/j.ast.2025.110875
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.logging_setup import get_logger
from ..utils.seeding import make_rng
from .sensors import CYCLE_COL, UNIT_COL

__all__ = ["WindowedDataset", "make_windows", "make_last_windows", "split_units_by_id"]

logger = get_logger(__name__)


@dataclass
class WindowedDataset:
    """A set of windowed examples, ready to be handed to a model.

    Attributes
    ----------
    features : numpy.ndarray
        Shape ``(n_examples, n_sensors, n_features_per_sensor)``, dtype
        ``float32``.
    targets : numpy.ndarray
        Shape ``(n_examples,)``, dtype ``float32``. The Remaining Useful Life
        at the last cycle of each window, in flight cycles.
    units : numpy.ndarray
        Shape ``(n_examples,)``, dtype ``int64``. Which engine each example
        came from. Needed for grouping predictions by engine at evaluation time
        and for checking that no engine appears in two splits.
    end_cycles : numpy.ndarray
        Shape ``(n_examples,)``, dtype ``int64``. The cycle number of the last
        row in each window.
    sensor_columns : tuple of str
        Names of the sensor columns, in the order of the second axis. Keeping
        these with the array prevents the sensor order from being lost, which
        would silently misalign the data with the sensor graph.
    """

    features: np.ndarray
    targets: np.ndarray
    units: np.ndarray
    end_cycles: np.ndarray
    sensor_columns: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __repr__(self) -> str:
        # An empty dataset is legitimate: it is what a validation fraction of
        # zero produces. Reporting the minimum and maximum of an empty array
        # would raise, so that case is described separately.
        if len(self) == 0:
            return (
                f"WindowedDataset(empty, "
                f"sensors={len(self.sensor_columns)})"
            )
        return (
            f"WindowedDataset(examples={len(self)}, "
            f"shape={tuple(self.features.shape)}, "
            f"engines={len(np.unique(self.units))}, "
            f"target_range=({self.targets.min():.0f}, {self.targets.max():.0f}))"
        )

    def subset_by_units(self, units: Sequence[int]) -> WindowedDataset:
        """Return a new dataset containing only the given engines."""
        mask = np.isin(self.units, np.asarray(list(units)))
        return WindowedDataset(
            features=self.features[mask],
            targets=self.targets[mask],
            units=self.units[mask],
            end_cycles=self.end_cycles[mask],
            sensor_columns=self.sensor_columns,
        )


def _extract_one_unit(
    history: pd.DataFrame,
    sensor_columns: Sequence[str],
    window_size: int,
    stride: int,
    last_only: bool,
    node_feature_mode: str,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut the history of a single engine into windows."""
    values = history[list(sensor_columns)].to_numpy(dtype=np.float32)
    targets = history[target_column].to_numpy(dtype=np.float32)
    cycles = history[CYCLE_COL].to_numpy(dtype=np.int64)
    n_rows = values.shape[0]

    if n_rows < window_size:
        # Too short for a full window. Rather than discard the engine, repeat
        # its first reading backwards until the window is filled. The
        # alternative -- dropping the engine -- would silently change the size
        # of the test set and make results incomparable with published ones.
        pad = window_size - n_rows
        values = np.vstack([np.repeat(values[:1], pad, axis=0), values])
        targets = np.concatenate([np.repeat(targets[:1], pad), targets])
        cycles = np.concatenate([np.repeat(cycles[:1], pad), cycles])
        n_rows = window_size
        logger.warning(
            "Engine %s has fewer rows than the window size (%d); its first "
            "reading was repeated %d time(s) to fill the window.",
            history[UNIT_COL].iloc[0], window_size, pad,
        )

    starts = (
        [n_rows - window_size]
        if last_only
        else list(range(0, n_rows - window_size + 1, stride))
    )

    windows = np.stack([values[s : s + window_size] for s in starts])   # (n, W, S)
    window_targets = np.array([targets[s + window_size - 1] for s in starts], dtype=np.float32)
    window_cycles = np.array([cycles[s + window_size - 1] for s in starts], dtype=np.int64)

    # Move the sensor axis in front of the time axis, so each sensor node owns
    # a contiguous block of features: (n, W, S) -> (n, S, W)
    windows = np.transpose(windows, (0, 2, 1))

    if node_feature_mode == "last_step":
        windows = windows[:, :, -1:]
    elif node_feature_mode != "window":
        raise ValueError(
            f"Unknown node_feature_mode {node_feature_mode!r}. "
            "Valid options are 'window' and 'last_step'."
        )

    return np.ascontiguousarray(windows), window_targets, window_cycles


def _build(
    frame: pd.DataFrame,
    sensor_columns: Sequence[str],
    *,
    window_size: int,
    stride: int,
    last_only: bool,
    node_feature_mode: str,
    target_column: str,
) -> WindowedDataset:
    missing = [c for c in sensor_columns if c not in frame.columns]
    if missing:
        raise KeyError(f"Sensor columns not present in the frame: {missing}.")
    if target_column not in frame.columns:
        raise KeyError(f"Target column {target_column!r} not present in the frame.")
    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, got {window_size}.")

    feature_blocks, target_blocks, unit_blocks, cycle_blocks = [], [], [], []

    for unit, history in frame.sort_values([UNIT_COL, CYCLE_COL]).groupby(
        UNIT_COL, sort=True
    ):
        windows, targets, cycles = _extract_one_unit(
            history,
            sensor_columns,
            window_size,
            stride,
            last_only,
            node_feature_mode,
            target_column,
        )
        feature_blocks.append(windows)
        target_blocks.append(targets)
        cycle_blocks.append(cycles)
        unit_blocks.append(np.full(windows.shape[0], int(unit), dtype=np.int64))

    dataset = WindowedDataset(
        features=np.concatenate(feature_blocks).astype(np.float32),
        targets=np.concatenate(target_blocks).astype(np.float32),
        units=np.concatenate(unit_blocks),
        end_cycles=np.concatenate(cycle_blocks),
        sensor_columns=tuple(sensor_columns),
    )
    logger.info(
        "Built %d window(s) of shape %s from %d engine(s).",
        len(dataset),
        tuple(dataset.features.shape[1:]),
        len(np.unique(dataset.units)),
    )
    return dataset


def make_windows(
    frame: pd.DataFrame,
    sensor_columns: Sequence[str],
    *,
    window_size: int = 30,
    stride: int = 1,
    node_feature_mode: str = "window",
    target_column: str = "rul",
) -> WindowedDataset:
    """Cut every engine history into overlapping windows.

    Use this for training data, where all windows are wanted.
    """
    return _build(
        frame,
        sensor_columns,
        window_size=window_size,
        stride=stride,
        last_only=False,
        node_feature_mode=node_feature_mode,
        target_column=target_column,
    )


def make_last_windows(
    frame: pd.DataFrame,
    sensor_columns: Sequence[str],
    *,
    window_size: int = 30,
    node_feature_mode: str = "window",
    target_column: str = "rul",
) -> WindowedDataset:
    """Take only the final window of each engine history.

    Use this for test data, where the official label describes exactly one
    moment per engine: the last recorded cycle.
    """
    return _build(
        frame,
        sensor_columns,
        window_size=window_size,
        stride=1,
        last_only=True,
        node_feature_mode=node_feature_mode,
        target_column=target_column,
    )


def split_units_by_id(
    units: Sequence[int],
    *,
    validation_fraction: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a list of engine numbers into a training part and a validation part.

    Parameters
    ----------
    units : sequence of int
        All engine numbers available for training.
    validation_fraction : float
        Fraction of engines to hold out, between 0 and 1. A value of 0 returns
        an empty validation set.
    seed : int
        Seed controlling the shuffle, so the split is repeatable.

    Returns
    -------
    tuple of numpy.ndarray
        ``(training_units, validation_units)``, both sorted. The two arrays
        never share an element.

    Examples
    --------
    >>> train, val = split_units_by_id(range(1, 11), validation_fraction=0.2, seed=0)
    >>> len(train), len(val)
    (8, 2)
    >>> set(train) & set(val)
    set()
    """
    unique = np.unique(np.asarray(list(units), dtype=np.int64))
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(
            f"validation_fraction must be in [0, 1), got {validation_fraction}."
        )

    n_validation = int(round(len(unique) * validation_fraction))
    if n_validation == 0:
        return unique, np.array([], dtype=np.int64)
    if n_validation >= len(unique):
        raise ValueError(
            "validation_fraction would leave no engines for training "
            f"({n_validation} of {len(unique)} selected)."
        )

    shuffled = make_rng(seed).permutation(unique)
    validation = np.sort(shuffled[:n_validation])
    training = np.sort(shuffled[n_validation:])

    logger.info(
        "Split %d engines into %d for training and %d for validation (seed=%d).",
        len(unique), len(training), len(validation), seed,
    )
    return training, validation
