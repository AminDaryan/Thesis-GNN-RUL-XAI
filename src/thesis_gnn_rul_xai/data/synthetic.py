"""
Synthetic C-MAPSS-format data
=============================

Why generated data is part of a scientific project
--------------------------------------------------
The real NASA files cannot be included in this repository, and a reader who
wants to check the code should not have to download them first. This module
writes files that have exactly the same *format* as the real ones: the same 26
columns, the same three-file structure, the same meaning for every column.

That makes two things possible:

1. **Automated testing.** Every function in the pipeline can be tested on data
   whose correct answer is known by construction. For example, if a synthetic
   engine is built to live for exactly 200 cycles, then the label produced for
   cycle 150 must be 50. A test can check that exactly.
2. **A runnable demonstration.** The notebook executes from beginning to end on
   any machine, even without the NASA files, so the logic can be read and
   verified before the real data is available.

What generated data is *not*
----------------------------
It is not a substitute for the real dataset. The signals here follow a simple
mathematical model of wear that the author of this code wrote; they are not the
output of NASA's engine simulator. **No number computed from synthetic data may
ever be reported as a scientific result.** Every function in this module writes
files into a separate ``data/synthetic`` directory, and the notebook prints a
clear warning whenever it is running in this mode.

The degradation model used here
-------------------------------
Each engine is given a random total lifetime. A hidden *health index* runs from
0 when the engine is new to 1 when it fails. Health worsens slowly at first and
then faster, which is the behaviour normally seen in wear processes; this is
represented by raising the elapsed-life fraction to a power greater than one.

Each informative sensor is then built as::

    value = baseline + drift * health**exponent + noise

where ``drift`` is positive for sensors that rise as the engine wears (such as
turbine outlet temperature) and negative for those that fall. The seven sensors
that Wen et al. (2026) exclude are written as exact constants, so that the
constant-sensor detection in :mod:`thesis_gnn_rul_xai.data.sensors` has
something to find.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging_setup import get_logger
from ..utils.seeding import make_rng
from .sensors import (
    OP_SETTING_COLS,
    PAPER_EXCLUDED_SYMBOLS,
    RAW_COLUMNS,
    SENSORS,
    symbol_to_column,
)

__all__ = ["generate_synthetic_subset", "write_synthetic_subset"]

logger = get_logger(__name__)

#: Sensors that physically rise as the engine wears. Used only to give the
#: generated signals a realistic direction of change; it has no effect on the
#: correctness of the pipeline.
_RISING_SYMBOLS = frozenset({"T24", "T30", "T50", "Ps30", "phi", "htBleed", "Nc"})


def _make_engine(
    unit: int,
    n_cycles: int,
    rng: np.random.Generator,
    *,
    noise_scale: float,
) -> pd.DataFrame:
    """Build the full history of one synthetic engine."""
    cycles = np.arange(1, n_cycles + 1, dtype=int)

    # Health index: 0 when new, 1 at failure, worsening faster near the end.
    elapsed_fraction = cycles / n_cycles
    degradation_exponent = float(rng.uniform(1.6, 2.6))
    health = elapsed_fraction**degradation_exponent

    frame = pd.DataFrame({"unit": unit, "cycle": cycles})

    # FD001 has a single operating condition, so the three settings stay almost
    # constant and only carry a little measurement noise.
    frame[OP_SETTING_COLS[0]] = rng.normal(0.0, 0.002, n_cycles).round(4)
    frame[OP_SETTING_COLS[1]] = rng.normal(0.0, 0.0002, n_cycles).round(4)
    frame[OP_SETTING_COLS[2]] = 100.0

    excluded = set(PAPER_EXCLUDED_SYMBOLS)
    for sensor in SENSORS:
        if sensor.symbol in excluded:
            # A perfectly flat signal, as the excluded sensors are in FD001.
            frame[sensor.column] = _constant_value_for(sensor.symbol)
            continue

        # A per-engine baseline, because no two real engines are identical.
        baseline = _baseline_for(sensor.symbol) * float(rng.normal(1.0, 0.0015))
        span = abs(baseline) * 0.012 + 0.05
        direction = 1.0 if sensor.symbol in _RISING_SYMBOLS else -1.0
        drift = direction * span * float(rng.uniform(0.7, 1.3))
        noise = rng.normal(0.0, span * noise_scale, n_cycles)

        frame[sensor.column] = baseline + drift * health + noise

    return frame[RAW_COLUMNS]


def _baseline_for(symbol: str) -> float:
    """A plausible resting value for a sensor, used to set its magnitude.

    The exact numbers are unimportant. They exist so that the generated signals
    have realistically different scales, which is what makes normalisation a
    meaningful step to test.
    """
    baselines = {
        "T24": 642.0, "T30": 1590.0, "T50": 1400.0, "P15": 21.6, "P30": 554.0,
        "Nf": 2388.0, "Nc": 9050.0, "Ps30": 47.5, "phi": 521.7, "NRf": 2388.0,
        "BPR": 8.42, "htBleed": 393.0, "W31": 38.9, "W32": 23.4,
    }
    return baselines.get(symbol, 100.0)


def _constant_value_for(symbol: str) -> float:
    """The fixed value written for each excluded, non-informative sensor."""
    constants = {
        "T2": 518.67, "P2": 14.62, "epr": 1.3, "NRc": 8138.0,
        "farB": 0.03, "Nf_dmd": 2388.0, "PCNfR_dmd": 100.0,
    }
    return constants.get(symbol, 0.0)


def generate_synthetic_subset(
    *,
    n_train_units: int = 40,
    n_test_units: int = 20,
    min_cycles: int = 140,
    max_cycles: int = 320,
    min_test_remaining: int = 8,
    max_test_remaining: int = 140,
    noise_scale: float = 0.18,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Generate a training frame, a test frame and the matching labels.

    Parameters
    ----------
    n_train_units, n_test_units : int
        How many engines to generate for each split.
    min_cycles, max_cycles : int
        Range of total engine lifetimes, in flight cycles.
    min_test_remaining, max_test_remaining : int
        Range for how much life is left when recording stops for a test engine.
        This is what the ``RUL_FD00X.txt`` file records.
    noise_scale : float
        Amount of measurement noise, as a fraction of each sensor's drift span.
    seed : int
        Random seed, so that the generated dataset is always the same.

    Returns
    -------
    tuple
        ``(train_frame, test_frame, rul_series)``, in the same form that
        :mod:`thesis_gnn_rul_xai.data.loading` produces from real files, but
        without a ``rul`` column: labels are attached later by the same code
        path that handles real data, so that path is genuinely tested.
    """
    rng = make_rng(seed)

    train_parts = [
        _make_engine(
            unit,
            int(rng.integers(min_cycles, max_cycles + 1)),
            rng,
            noise_scale=noise_scale,
        )
        for unit in range(1, n_train_units + 1)
    ]
    train = pd.concat(train_parts, ignore_index=True)

    test_parts: list[pd.DataFrame] = []
    remaining_life: list[int] = []
    for unit in range(1, n_test_units + 1):
        total = int(rng.integers(min_cycles, max_cycles + 1))
        left = int(rng.integers(min_test_remaining, max_test_remaining + 1))
        # Never truncate an engine so early that it has too little history.
        observed = max(total - left, min_cycles // 2)
        left = total - observed

        history = _make_engine(unit, total, rng, noise_scale=noise_scale)
        test_parts.append(history.iloc[:observed].copy())
        remaining_life.append(left)

    test = pd.concat(test_parts, ignore_index=True)
    rul = pd.Series(
        remaining_life,
        index=pd.Index(range(1, n_test_units + 1), name="unit"),
        name="rul_at_last_cycle",
        dtype=int,
    )

    logger.info(
        "Generated synthetic data: %d training rows over %d engines, "
        "%d test rows over %d engines.",
        len(train), n_train_units, len(test), n_test_units,
    )
    return train, test, rul


def write_synthetic_subset(
    output_dir: str | Path,
    subset: str = "FD001",
    **kwargs: object,
) -> dict[str, Path]:
    """Generate a synthetic sub-dataset and write it in real C-MAPSS format.

    The files use the same names and the same layout as the NASA release, so
    the loading code cannot tell the difference. That is the point: the loading
    code is exercised exactly as it would be on real data.

    Parameters
    ----------
    output_dir : str or Path
        Directory to write into. Created if it does not exist.
    subset : str
        Name used in the file names, for example ``"FD001"``.
    **kwargs
        Passed through to :func:`generate_synthetic_subset`.

    Returns
    -------
    dict
        Mapping from ``"train"``, ``"test"`` and ``"rul"`` to the written paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test, rul = generate_synthetic_subset(**kwargs)  # type: ignore[arg-type]

    paths = {
        "train": output_dir / f"train_{subset}.txt",
        "test": output_dir / f"test_{subset}.txt",
        "rul": output_dir / f"RUL_{subset}.txt",
    }

    # Match the real files: space separated, no header, no index, 4 decimals.
    train.to_csv(paths["train"], sep=" ", header=False, index=False, float_format="%.4f")
    test.to_csv(paths["test"], sep=" ", header=False, index=False, float_format="%.4f")
    rul.to_csv(paths["rul"], header=False, index=False)

    logger.info("Wrote synthetic %s files to %s", subset, output_dir)
    return paths
