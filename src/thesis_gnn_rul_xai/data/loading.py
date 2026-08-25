"""
Reading the raw C-MAPSS files
=============================

File format
-----------
Each C-MAPSS sub-dataset consists of three plain text files:

``train_FD001.txt``
    Complete run-to-failure histories. Every engine is recorded from when it
    was new until it failed. The last row for an engine is therefore the moment
    of failure, and the Remaining Useful Life at that row is zero.

``test_FD001.txt``
    Partial histories. Recording stops at some point *before* failure, and the
    task is to predict how much life is left at that point.

``RUL_FD001.txt``
    One number per test engine: the true Remaining Useful Life, in flight
    cycles, at the last row that appears for that engine in the test file.

The two history files have 26 columns separated by spaces, with no header row
and a trailing space at the end of each line. The RUL file has a single column.

Turning the labels into a value for every row
---------------------------------------------
The training file needs no label file. Because each engine is recorded until it
fails, the Remaining Useful Life at any cycle is simply how many cycles remain
until the last recorded one::

    RUL(t) = last_cycle_of_this_engine - t

The test file needs the label file, because recording stopped early. If an
engine's last recorded cycle is 150 and the label file says 47 cycles of life
were left at that moment, then at cycle 130 the engine still had::

    RUL(130) = (150 - 130) + 47 = 67

In general::

    RUL(t) = (last_recorded_cycle - t) + RUL_at_last_recorded_cycle

Both cases are handled by :func:`attach_rul`.

References
----------
Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation
modeling for aircraft engine run-to-failure simulation. *2008 International
Conference on Prognostics and Health Management*, pp. 1-9. IEEE.
https://doi.org/10.1109/PHM.2008.4711414
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging_setup import get_logger
from .sensors import CYCLE_COL, RAW_COLUMNS, UNIT_COL

__all__ = [
    "CMAPSSData",
    "DOWNLOAD_URL",
    "read_cmapss_frame",
    "read_rul_file",
    "attach_rul",
    "load_subset",
    "EXPECTED_SHAPES",
]

logger = get_logger(__name__)

#: Where the official C-MAPSS files are distributed. It is quoted in the
#: error a first-time user is most likely to hit, so the fix travels with
#: the problem instead of living only in the README.
DOWNLOAD_URL = (
    "https://www.nasa.gov/intelligent-systems-division/"
    "discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
)

#: Row and engine counts reported in the literature for each sub-dataset.
#: These are used only as a sanity check when real data is loaded: a mismatch
#: means the files are truncated, modified, or not the official release. The
#: check produces a warning, never an error, because a legitimate variant of
#: the files should not stop the pipeline.
EXPECTED_SHAPES: dict[str, dict[str, int]] = {
    "FD001": {"train_rows": 20631, "train_units": 100, "test_units": 100},
    "FD002": {"train_rows": 53759, "train_units": 260, "test_units": 259},
    "FD003": {"train_rows": 24720, "train_units": 100, "test_units": 100},
    "FD004": {"train_rows": 61249, "train_units": 249, "test_units": 248},
}


class CMAPSSData:
    """A loaded C-MAPSS sub-dataset with labels already attached.

    Attributes
    ----------
    subset : str
        Which sub-dataset this is, for example ``"FD001"``.
    train : pandas.DataFrame
        Training histories, with an added ``rul`` column.
    test : pandas.DataFrame
        Test histories, with an added ``rul`` column.
    test_rul : pandas.Series
        The official Remaining Useful Life at the last recorded cycle of each
        test engine, indexed by engine number. Kept separately because it is
        the quantity the final evaluation compares against.
    """

    def __init__(
        self,
        subset: str,
        train: pd.DataFrame,
        test: pd.DataFrame,
        test_rul: pd.Series,
    ) -> None:
        self.subset = subset
        self.train = train
        self.test = test
        self.test_rul = test_rul

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"CMAPSSData(subset={self.subset!r}, "
            f"train_rows={len(self.train)}, train_units={self.train[UNIT_COL].nunique()}, "
            f"test_rows={len(self.test)}, test_units={self.test[UNIT_COL].nunique()})"
        )

    def summary(self) -> pd.DataFrame:
        """Return a small table describing the loaded data.

        Useful in a notebook as a first check that the files were read
        correctly.
        """
        def describe(frame: pd.DataFrame, name: str) -> dict[str, object]:
            lengths = frame.groupby(UNIT_COL)[CYCLE_COL].max()
            return {
                "split": name,
                "rows": len(frame),
                "engines": int(frame[UNIT_COL].nunique()),
                "shortest_history": int(lengths.min()),
                "longest_history": int(lengths.max()),
                "median_history": float(lengths.median()),
            }

        return pd.DataFrame(
            [describe(self.train, "train"), describe(self.test, "test")]
        ).set_index("split")


def read_cmapss_frame(path: str | Path) -> pd.DataFrame:
    """Read one C-MAPSS history file into a data frame.

    Parameters
    ----------
    path : str or Path
        Path to ``train_FD00X.txt`` or ``test_FD00X.txt``.

    Returns
    -------
    pandas.DataFrame
        A frame with the 26 named columns of :data:`RAW_COLUMNS`, sorted by
        engine and then by cycle.

    Raises
    ------
    FileNotFoundError
        If the file does not exist. The message explains how to obtain the
        dataset, because a missing file is the most common first-run problem.
    ValueError
        If the file does not have 26 columns.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"C-MAPSS file not found: {path}\n\n"
            "The NASA files are not included in this repository because they "
            "are redistributed under NASA's own terms. Download the "
            "'Turbofan Engine Degradation Simulation Data Set' from the NASA "
            "Prognostics Center of Excellence repository:\n\n"
            f"    {DOWNLOAD_URL}\n\n"
            "and put the three files for this sub-dataset into data/raw/. "
            "See README.md, section 'Getting the data'."
        )

    # The files use one or more spaces as the separator and have a trailing
    # space on each line, which would otherwise create an empty 27th column.
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        engine="python",
    )
    frame = frame.dropna(axis=1, how="all")

    if frame.shape[1] != len(RAW_COLUMNS):
        raise ValueError(
            f"Expected {len(RAW_COLUMNS)} columns in {path.name}, "
            f"found {frame.shape[1]}. The file may be corrupted or may not be "
            "an official C-MAPSS file."
        )

    frame.columns = RAW_COLUMNS
    frame[UNIT_COL] = frame[UNIT_COL].astype(int)
    frame[CYCLE_COL] = frame[CYCLE_COL].astype(int)
    return frame.sort_values([UNIT_COL, CYCLE_COL]).reset_index(drop=True)


def read_rul_file(path: str | Path) -> pd.Series:
    """Read a ``RUL_FD00X.txt`` label file.

    Parameters
    ----------
    path : str or Path
        Path to the label file.

    Returns
    -------
    pandas.Series
        Remaining Useful Life at the last recorded cycle, indexed by engine
        number starting at 1. Engine numbers are assigned by position: the
        first line of the file belongs to engine 1, the second to engine 2, and
        so on. This is how the dataset is defined.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"RUL label file not found: {path}\n"
            "See README.md, section 'Getting the data'."
        )

    values = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    values = values.dropna(axis=1, how="all")
    if values.shape[1] != 1:
        raise ValueError(
            f"Expected a single column in {path.name}, found {values.shape[1]}."
        )

    series = values.iloc[:, 0].astype(int)
    series.index = pd.Index(range(1, len(series) + 1), name=UNIT_COL)
    series.name = "rul_at_last_cycle"
    return series


def attach_rul(
    frame: pd.DataFrame,
    *,
    rul_at_last_cycle: pd.Series | None = None,
) -> pd.DataFrame:
    """Add a ``rul`` column giving the Remaining Useful Life at every row.

    Parameters
    ----------
    frame : pandas.DataFrame
        A loaded history frame.
    rul_at_last_cycle : pandas.Series, optional
        For test data, the official label for each engine, as returned by
        :func:`read_rul_file`. For training data leave this as ``None``: the
        histories run to failure, so the remaining life at the last row is zero
        by definition.

    Returns
    -------
    pandas.DataFrame
        A copy of ``frame`` with one extra integer column, ``rul``.

    Raises
    ------
    ValueError
        If a label is missing for an engine that appears in the frame.
    """
    frame = frame.copy()
    last_cycle = frame.groupby(UNIT_COL)[CYCLE_COL].transform("max")

    if rul_at_last_cycle is None:
        offset = 0
    else:
        missing = set(frame[UNIT_COL].unique()) - set(rul_at_last_cycle.index)
        if missing:
            raise ValueError(
                f"No Remaining Useful Life label for engine(s) {sorted(missing)}. "
                "The label file and the test file do not describe the same engines."
            )
        offset = frame[UNIT_COL].map(rul_at_last_cycle).to_numpy()

    frame["rul"] = (last_cycle - frame[CYCLE_COL] + offset).astype(int)
    return frame


def load_subset(
    subset: str = "FD001",
    *,
    data_dir: str | Path,
    verify_expected_shape: bool = True,
) -> CMAPSSData:
    """Load one complete C-MAPSS sub-dataset with labels attached.

    Parameters
    ----------
    subset : str
        One of ``"FD001"``, ``"FD002"``, ``"FD003"``, ``"FD004"``.
    data_dir : str or Path
        Directory containing the three text files for this sub-dataset.
    verify_expected_shape : bool
        When ``True``, compare the number of rows and engines against the
        values published for the official release and warn on a mismatch.

    Returns
    -------
    CMAPSSData
        Training and test frames with a ``rul`` column, plus the official test
        labels.
    """
    data_dir = Path(data_dir)
    logger.info("Loading %s from %s", subset, data_dir)

    train = read_cmapss_frame(data_dir / f"train_{subset}.txt")
    test = read_cmapss_frame(data_dir / f"test_{subset}.txt")
    test_rul = read_rul_file(data_dir / f"RUL_{subset}.txt")

    train = attach_rul(train)
    test = attach_rul(test, rul_at_last_cycle=test_rul)

    if verify_expected_shape and subset in EXPECTED_SHAPES:
        expected = EXPECTED_SHAPES[subset]
        actual = {
            "train_rows": len(train),
            "train_units": int(train[UNIT_COL].nunique()),
            "test_units": int(test[UNIT_COL].nunique()),
        }
        differences = {
            key: (expected[key], actual[key])
            for key in expected
            if expected[key] != actual[key]
        }
        if differences:
            logger.warning(
                "Loaded %s does not match the published shape of the official "
                "release. Differences (expected, actual): %s. The files may be "
                "a different version, truncated, or modified. Investigate this "
                "before reporting any result computed from them.",
                subset,
                differences,
            )
        else:
            logger.info("%s matches the published shape of the official release.", subset)

    logger.info(
        "Loaded %s: %d training rows over %d engines, %d test rows over %d engines.",
        subset,
        len(train),
        train[UNIT_COL].nunique(),
        len(test),
        test[UNIT_COL].nunique(),
    )
    return CMAPSSData(subset=subset, train=train, test=test, test_rul=test_rul)
