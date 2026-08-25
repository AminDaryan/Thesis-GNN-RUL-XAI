"""
Shared fixtures for the test suite
==================================

The tests run on the **real** NASA C-MAPSS files. A preprocessing pipeline can
only be trusted against the data it will actually be used on, and several of
the properties worth testing -- that ``P15`` takes two distinct values, that
the shortest test history is 31 cycles, that eleven test labels exceed the
ceiling -- exist only in the real measurements. Testing against generated data
would confirm that the code runs, not that it is right.

The files are not redistributed with this repository. See README.md, section
"Getting the data". If they are absent, the tests that need them are skipped
with an explanatory message rather than failing, so a fresh clone reports a
missing download rather than a broken pipeline.

A *fixture* is a piece of prepared input that several tests reuse. The dataset
is loaded once per session and shared, which keeps the suite fast while still
giving every test the same known input.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.loading import DOWNLOAD_URL, CMAPSSData, load_subset
from thesis_gnn_rul_xai.utils.paths import raw_data_dir

#: The sub-dataset the suite runs against. FD001 has a single operating
#: condition and a single fault mode, which makes it the case where a
#: disagreement between the code and the data has the fewest explanations.
SUBSET = "FD001"


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    """Directory holding the three FD001 text files, or skip the test."""
    directory = raw_data_dir()
    required = [
        f"train_{SUBSET}.txt",
        f"test_{SUBSET}.txt",
        f"RUL_{SUBSET}.txt",
    ]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        pytest.skip(
            f"The {SUBSET} files are not in {directory}: {missing}.\n"
            "Download the 'Turbofan Engine Degradation Simulation Data Set' "
            f"from {DOWNLOAD_URL} and put the three files there, or run "
            "`python scripts/prepare_data.py --check` for the same report."
        )
    return directory


@pytest.fixture(scope="session")
def loaded(raw_dir: Path) -> CMAPSSData:
    """The FD001 dataset with Remaining Useful Life attached to every row."""
    return load_subset(SUBSET, data_dir=raw_dir)


@pytest.fixture()
def config() -> PreprocessingConfig:
    """A configuration with a small window, to keep the suite quick."""
    return PreprocessingConfig(
        subset=SUBSET,
        window_size=20,
        validation_fraction=0.25,
        seed=7,
    )
