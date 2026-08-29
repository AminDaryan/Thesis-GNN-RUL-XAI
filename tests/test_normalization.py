"""Tests for operating-condition detection and sensor rescaling.

The most important test in this file is
`test_test_data_never_influences_the_scaling`. It checks the rule that
keeps the whole evaluation valid: statistics are learned from training engines
only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_gnn_rul_xai.data.normalization import (
    CONDITION_COL,
    OperatingConditionEncoder,
    SensorNormalizer,
)
from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_COLUMNS

COLUMNS = list(PAPER_SENSOR_COLUMNS)


def test_fd001_has_a_single_operating_condition(loaded):
    encoder = OperatingConditionEncoder().fit(loaded.train)
    assert encoder.n_detected == 1
    assert (encoder.transform(loaded.train)[CONDITION_COL] == 0).all()


def test_several_conditions_are_separated():
    """Three well-separated flight conditions must be recovered as three groups."""
    rng = np.random.default_rng(0)
    centres = [(0.0, 0.0, 100.0), (10.0, 0.25, 100.0), (25.0, 0.62, 60.0)]
    rows = []
    for altitude, mach, throttle in centres:
        for _ in range(60):
            rows.append(
                {
                    "op_setting_1": altitude + rng.normal(0, 0.01),
                    "op_setting_2": mach + rng.normal(0, 0.001),
                    "op_setting_3": throttle,
                }
            )
    frame = pd.DataFrame(rows)

    encoder = OperatingConditionEncoder(n_conditions=3, seed=0).fit(frame)
    labels = encoder.transform(frame)[CONDITION_COL]
    assert labels.nunique() == 3
    # Each true group must map to exactly one cluster.
    for start in (0, 60, 120):
        assert labels.iloc[start : start + 60].nunique() == 1


def test_minmax_maps_training_data_onto_the_unit_interval(loaded):
    encoder = OperatingConditionEncoder().fit(loaded.train)
    train = encoder.transform(loaded.train)
    scaled = SensorNormalizer(method="minmax_global", columns=COLUMNS).fit_transform(train)
    values = scaled[COLUMNS].to_numpy()
    assert values.min() == pytest.approx(0.0)
    assert values.max() == pytest.approx(1.0)


def test_zscore_centres_and_scales(loaded):
    scaled = SensorNormalizer(method="zscore", columns=COLUMNS).fit_transform(loaded.train)
    values = scaled[COLUMNS].to_numpy()
    assert values.mean() == pytest.approx(0.0, abs=1e-9)
    for column in COLUMNS:
        assert scaled[column].std(ddof=0) == pytest.approx(1.0)


def test_test_data_never_influences_the_scaling(loaded):
    """Corrupting the test set must not change how training data is scaled.

    If this test ever fails, the pipeline has data leakage: information from
    the test engines is reaching the training process, and every reported
    error would be too optimistic.
    """
    encoder = OperatingConditionEncoder().fit(loaded.train)
    train = encoder.transform(loaded.train)

    normalizer = SensorNormalizer(method="minmax_global", columns=COLUMNS).fit(train)
    before = normalizer.transform(train)[COLUMNS].to_numpy()

    corrupted_test = encoder.transform(loaded.test).copy()
    corrupted_test[COLUMNS] = corrupted_test[COLUMNS] * 1e6
    normalizer.transform(corrupted_test)  # applying it must not refit anything

    after = normalizer.transform(train)[COLUMNS].to_numpy()
    np.testing.assert_allclose(before, after)


def test_test_values_may_fall_outside_the_unit_interval(loaded):
    """Going slightly outside [0, 1] on test data is correct, not a bug.

    The scale comes from the training engines. A test engine that runs a little
    hotter than anything seen in training will map above 1. Clipping it would
    hide a genuine difference between the two sets.
    """
    encoder = OperatingConditionEncoder().fit(loaded.train)
    normalizer = SensorNormalizer(method="minmax_global", columns=COLUMNS).fit(
        encoder.transform(loaded.train)
    )
    scaled_test = normalizer.transform(encoder.transform(loaded.test))
    assert np.isfinite(scaled_test[COLUMNS].to_numpy()).all()


def test_a_constant_column_does_not_divide_by_zero():
    frame = pd.DataFrame(
        {"sensor_2": [5.0] * 10, "sensor_3": np.linspace(0, 1, 10), CONDITION_COL: 0}
    )
    scaled = SensorNormalizer(
        method="minmax_per_condition", columns=["sensor_2", "sensor_3"]
    ).fit_transform(frame)
    assert np.isfinite(scaled["sensor_2"].to_numpy()).all()
    assert (scaled["sensor_2"] == 0.0).all()


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown normalisation method"):
        SensorNormalizer(method="quantile", columns=COLUMNS)


def test_transform_before_fit_is_rejected(loaded):
    normalizer = SensorNormalizer(method="minmax_global", columns=COLUMNS)
    with pytest.raises(RuntimeError, match="Call fit"):
        normalizer.transform(loaded.train)
