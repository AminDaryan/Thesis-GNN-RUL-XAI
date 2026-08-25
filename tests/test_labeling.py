"""Tests for the piecewise-linear Remaining Useful Life target."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_gnn_rul_xai.data.labeling import apply_piecewise_rul, piecewise_rul_curve


def test_values_above_the_ceiling_are_capped():
    frame = pd.DataFrame({"rul": [300, 126, 125, 124, 0]})
    result = apply_piecewise_rul(frame, rul_max=125)["rul"].tolist()
    assert result == [125, 125, 125, 124, 0]


def test_capping_never_increases_a_label():
    """The operation may only lower values, never raise them."""
    frame = pd.DataFrame({"rul": np.arange(0, 400)})
    capped = apply_piecewise_rul(frame, rul_max=125)["rul"].to_numpy()
    assert (capped <= frame["rul"].to_numpy()).all()


def test_the_original_column_can_be_preserved():
    frame = pd.DataFrame({"rul": [300, 50]})
    result = apply_piecewise_rul(frame, rul_max=125, output_column="rul_capped")
    assert result["rul"].tolist() == [300, 50]
    assert result["rul_capped"].tolist() == [125, 50]


def test_the_curve_is_flat_then_straight():
    """The label curve must consist of a flat part followed by a -1 slope."""
    curve = piecewise_rul_curve(total_cycles=200, rul_max=125)
    assert curve[0] == 125
    assert curve[-1] == 0
    differences = np.diff(curve)
    assert set(np.unique(differences).tolist()) <= {0, -1}
    # The flat part must come first and the sloping part second.
    first_decline = int(np.argmax(differences == -1))
    assert (differences[:first_decline] == 0).all()
    assert (differences[first_decline:] == -1).all()


def test_an_engine_shorter_than_the_ceiling_is_never_capped():
    curve = piecewise_rul_curve(total_cycles=40, rul_max=125)
    assert curve.max() == 39


def test_invalid_ceiling_is_rejected():
    with pytest.raises(ValueError, match="rul_max must be positive"):
        apply_piecewise_rul(pd.DataFrame({"rul": [1]}), rul_max=0)
