"""Cuts each engine's history into short fixed-length pieces, one per example.

Read README.md in this folder. It explains the shape of the result, what the
test set gets, and why engines rather than windows are held back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...utils.logging_setup import get_logger
from ...utils.seeding import make_rng
from ..sensors import CYCLE_COL, UNIT_COL

__all__ = ["WindowedDataset", "make_last_windows", "make_windows", "split_units_by_id"]

logger = get_logger(__name__)


@dataclass
class WindowedDataset:
    """A set of examples ready for the network. README.md lists every field."""

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
        """Return a copy holding only the engines listed."""
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
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut one engine's history into windows."""
    values = history[list(sensor_columns)].to_numpy(dtype=np.float32)
    targets = history[target_column].to_numpy(dtype=np.float32)
    cycles = history[CYCLE_COL].to_numpy(dtype=np.int64)
    n_rows = values.shape[0]

    if n_rows < window_size:
        # ASSUMPTION A-23: an engine with fewer rows than the window is left
        # out entirely. The paper never says what it does with such an engine.
        # The alternative is to invent the missing rows -- repeat the first
        # reading backwards until the window is full -- which puts sensor
        # values into the record that the engine never produced and then asks
        # a model to predict from them. Dida, Cheriet and Belhadj do the same
        # as this: "Engines in sub-datasets FD002 and FD004 that contained
        # fewer than 30 samples were excluded from the study." No FD001 engine
        # is affected (its shortest test history is 31 cycles, see F-03), but
        # 6 FD002 and 11 FD004 test engines are, and dropping a test engine
        # changes the NASA score, which is a sum. Written up in
        # docs/ASSUMPTIONS.md and docs/LITERATURE_BASIS.md.
        logger.warning(
            "Engine %s has %d row(s), fewer than the window size of %d, so it "
            "is left out. Its readings are not padded: that would invent data. "
            "Any score computed from what remains covers fewer engines.",
            history[UNIT_COL].iloc[0], n_rows, window_size,
        )
        empty_windows = np.empty(
            (0, len(sensor_columns), window_size), dtype=np.float32
        )
        return (empty_windows,
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64))

    starts = (
        [n_rows - window_size]
        if last_only
        else list(range(0, n_rows - window_size + 1, stride))
    )

    windows = np.stack([values[s : s + window_size] for s in starts])   # (n, W, S)
    window_targets = np.array([targets[s + window_size - 1] for s in starts], dtype=np.float32)
    window_cycles = np.array([cycles[s + window_size - 1] for s in starts], dtype=np.int64)

    # Put the sensors in front of the time steps, so each sensor owns one
    # block of numbers: (n, W, S) becomes (n, S, W).
    windows = np.transpose(windows, (0, 2, 1))

    # Every cycle of the window is handed on. Which of them a sensor node ends
    # up carrying is a question about the MODEL, and it is answered there, by
    # the temporal stage in models/temporal/ -- see A-07. Answering it here
    # made the reduction look like a property of the data, and left a path from
    # a preprocessing setting to the width of the first graph weight matrix.
    return np.ascontiguousarray(windows), window_targets, window_cycles


def _build(
    frame: pd.DataFrame,
    sensor_columns: Sequence[str],
    *,
    window_size: int,
    stride: int,
    last_only: bool,
    target_column: str,
) -> WindowedDataset:
    """Cut every engine in the table into windows and stack them into one dataset."""
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
            target_column,
        )
        if windows.shape[0] == 0:      # left out by A-23
            continue
        feature_blocks.append(windows)
        target_blocks.append(targets)
        cycle_blocks.append(cycles)
        unit_blocks.append(np.full(windows.shape[0], int(unit), dtype=np.int64))

    if not feature_blocks:
        raise ValueError(
            f"No engine has at least {window_size} rows, so every one was left "
            "out under A-23 and there is nothing to return. The window is "
            "longer than the longest history in this frame."
        )

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
    target_column: str = "rul",
) -> WindowedDataset:
    """Cut every engine's history into overlapping windows. Use for training."""
    return _build(
        frame,
        sensor_columns,
        window_size=window_size,
        stride=stride,
        last_only=False,
        target_column=target_column,
    )


def make_last_windows(
    frame: pd.DataFrame,
    sensor_columns: Sequence[str],
    *,
    window_size: int = 30,
    target_column: str = "rul",
) -> WindowedDataset:
    """Take only the last window of each engine. Use for test data.

    The official test labels describe exactly one moment per engine.
    """
    return _build(
        frame,
        sensor_columns,
        window_size=window_size,
        stride=1,
        last_only=True,
        target_column=target_column,
    )


def split_units_by_id(
    units: Sequence[int],
    *,
    validation_fraction: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the engine numbers in two: the ones to train on, the ones to hold back.

    The two lists never share an engine. Splitting by engine, not by window,
    is what stops the held-back data from leaking; README.md explains why.
    """
    unique = np.unique(np.asarray(list(units), dtype=np.int64))
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(
            f"validation_fraction must be in [0, 1), got {validation_fraction}."
        )

    n_validation = round(len(unique) * validation_fraction)
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
