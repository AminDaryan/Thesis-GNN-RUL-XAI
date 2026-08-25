"""
Shared fixtures for the test suite
==================================

A *fixture* is a piece of prepared input that several tests can reuse. Building
the synthetic dataset once per test session and sharing it keeps the suite
fast, while still giving every test a clean, known input.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.loading import CMAPSSData, load_subset
from thesis_gnn_rul_xai.data.synthetic import write_synthetic_subset


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a small synthetic FD001 dataset to a temporary directory."""
    directory = tmp_path_factory.mktemp("synthetic_cmapss")
    write_synthetic_subset(
        directory,
        "FD001",
        n_train_units=12,
        n_test_units=6,
        min_cycles=90,
        max_cycles=180,
        seed=7,
    )
    return directory


@pytest.fixture(scope="session")
def loaded(synthetic_dir: Path) -> CMAPSSData:
    """A loaded synthetic dataset with Remaining Useful Life attached."""
    return load_subset("FD001", data_dir=synthetic_dir, verify_expected_shape=False)


@pytest.fixture()
def config(synthetic_dir: Path) -> PreprocessingConfig:
    """A configuration pointing at the synthetic data, with a small window."""
    return PreprocessingConfig(
        subset="FD001",
        window_size=20,
        validation_fraction=0.25,
        use_synthetic_data=True,
        seed=7,
    )
