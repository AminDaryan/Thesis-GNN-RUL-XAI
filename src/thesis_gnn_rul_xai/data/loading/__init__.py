"""Reads the three raw C-MAPSS text files and works out a label for every row.

Read README.md in this folder. It explains the file format, how the labels are
worked out, and everything else this file does.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...utils.logging_setup import get_logger
from ..sensors import CYCLE_COL, RAW_COLUMNS, UNIT_COL

__all__ = [
    "DOWNLOAD_URL",
    "EXPECTED_SHAPES",
    "CMAPSSData",
    "attach_rul",
    "load_subset",
    "read_cmapss_frame",
    "read_rul_file",
]

logger = get_logger(__name__)

# Where the official C-MAPSS files come from. Quoted in the error a first-time
# user is most likely to hit, so the fix arrives with the problem.
DOWNLOAD_URL = (
    "https://www.nasa.gov/intelligent-systems-division/"
    "discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
)

# How many rows and engines each subset should have: the counts of the NASA
# files held in data/raw/, counted from the files themselves. Wen et al.'s
# Table 5 (page 7) prints the same engine counts; NASA's own readme
# (docs/references/nasa_2008_cmapss_readme.txt) swaps FD004's two (248
# training, 249 test) where the files and Table 5 have 249 and 248. No source
# held in docs/references/ prints the row counts.
# (Corrected 2026-10-02: this called every count "published"; found when the
# cited papers were collected into docs/references/ and checked against the
# places that cite them.) Checked when files are loaded; a mismatch means the
# files are cut short, changed, or not the ones NASA distributes. It warns
# rather than stopping, because a legitimate variant should not halt the work.
EXPECTED_SHAPES: dict[str, dict[str, int]] = {
    "FD001": {"train_rows": 20631, "train_units": 100, "test_units": 100},
    "FD002": {"train_rows": 53759, "train_units": 260, "test_units": 259},
    "FD003": {"train_rows": 24720, "train_units": 100, "test_units": 100},
    "FD004": {"train_rows": 61249, "train_units": 249, "test_units": 248},
}


class CMAPSSData:
    """One loaded subset, with a label already worked out for every row."""

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
        """A small table describing what was loaded, to check it looks right."""
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
    """Read one history file into a table, sorted by engine and then by cycle."""
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
    """Read a RUL_FD00X.txt label file. Line 1 is engine 1, line 2 engine 2."""
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
    """Add a rul column saying how much life is left at every row.

    For training data leave rul_at_last_cycle as None: those engines are
    recorded until they fail, so the last row is zero by definition. For test
    data pass what read_rul_file() returned.
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
    """Read all three files of one subset and attach the labels."""
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
                "Loaded %s does not match the expected counts, which are those "
                "of NASA's files (Wen et al.'s Table 5 prints the same engine "
                "counts; NASA's readme swaps FD004's two). Differences "
                "(expected, actual): %s. The files may be a different version, "
                "truncated, or modified. Investigate this before reporting any "
                "result computed from them.",
                subset,
                differences,
            )
        else:
            logger.info(
                "%s matches the expected counts, those of NASA's files "
                "(Wen et al.'s Table 5 prints the same engine counts).",
                subset,
            )

    logger.info(
        "Loaded %s: %d training rows over %d engines, %d test rows over %d engines.",
        subset,
        len(train),
        train[UNIT_COL].nunique(),
        len(test),
        test[UNIT_COL].nunique(),
    )
    return CMAPSSData(subset=subset, train=train, test=test, test_rul=test_rul)
