"""Tests for the four evaluation measures.

The one that matters most is
:func:`test_late_predictions_are_punished_harder_than_early_ones`. The paper's
Equations 20 and 21 contradict each other on the direction of the asymmetry
(see A-15), so the property has to be pinned down by a test rather than trusted
to a transcription.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from thesis_gnn_rul_xai.evaluation import (
    asue,
    evaluate,
    nasa_score,
    prediction_error,
    r_squared,
    rmse,
)

# --- the sign convention ------------------------------------------------


def test_prediction_error_is_positive_when_the_model_is_optimistic():
    """d = predicted - actual, so a positive error means "too much life left"."""
    assert prediction_error([100], [120]).tolist() == [20.0]
    assert prediction_error([100], [80]).tolist() == [-20.0]


def test_late_predictions_are_punished_harder_than_early_ones():
    """The whole point of the NASA score, and the thing A-15 had to resolve.

    Claiming an engine has 20 cycles more life than it does is dangerous;
    claiming it has 20 fewer is merely wasteful. The score must reflect that.
    """
    late = nasa_score([100], [120])
    early = nasa_score([100], [80])
    assert late > early
    assert late == pytest.approx(math.exp(20 / 10) - 1)
    assert early == pytest.approx(math.exp(20 / 13) - 1)


def test_a_perfect_prediction_scores_zero():
    assert nasa_score([100, 50, 7], [100, 50, 7]) == 0.0


def test_the_score_is_a_sum_not_a_mean():
    """It grows with the test set, so it only compares within one sub-dataset."""
    one = nasa_score([100], [110])
    three = nasa_score([100, 100, 100], [110, 110, 110])
    assert three == pytest.approx(3 * one)


def test_one_bad_engine_can_dominate_the_score():
    """The exponential is deliberate: a missed failure outweighs small errors."""
    many_small = nasa_score([100] * 20, [105] * 20)
    one_large = nasa_score([100], [180])
    assert one_large > many_small


# --- the other three ----------------------------------------------------


def test_rmse_is_symmetric():
    """Unlike the score, RMSE does not care which way the error points."""
    assert rmse([100], [110]) == pytest.approx(rmse([100], [90]))


def test_rmse_is_in_cycles():
    assert rmse([100, 100], [110, 90]) == pytest.approx(10.0)


def test_r_squared_is_one_for_a_perfect_fit_and_zero_for_the_mean():
    truth = [10.0, 20.0, 30.0, 40.0]
    assert r_squared(truth, truth) == pytest.approx(1.0)
    assert r_squared(truth, [25.0] * 4) == pytest.approx(0.0)


def test_r_squared_can_be_negative():
    """A model worse than the mean earns a negative score, not a floor of zero."""
    assert r_squared([10, 20, 30], [30, 20, 10]) < 0


def test_r_squared_uses_the_true_labels_in_its_denominator():
    """Assumption A-16, pinned down with the case that separates the two.

    The paper's Equation 23 puts the predicted values in the denominator, which
    lets a model raise its score just by spreading its output more widely. The
    prediction below is the truth stretched to twice its spread about the mean:
    it is exactly as accurate as predicting the mean, so the correct answer is
    0.0. The paper's formula would give 0.75.
    """
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    over_spread = np.array([-5.0, 15.0, 35.0, 55.0])

    assert r_squared(truth, over_spread) == pytest.approx(0.0)

    residual = float(np.sum((over_spread - truth) ** 2))
    as_printed = 1 - residual / float(np.sum((over_spread - truth.mean()) ** 2))
    assert as_printed == pytest.approx(0.75)


def test_r_squared_needs_labels_that_vary():
    with pytest.raises(ValueError, match="every true label is the same"):
        r_squared([50, 50, 50], [40, 50, 60])


def test_asue_counts_only_the_pessimistic_engines():
    # Shortfalls of 10 and 20 on two engines; the optimistic one is ignored.
    assert asue([100, 50, 30], [90, 60, 10]) == pytest.approx(15.0)


def test_asue_is_zero_when_the_model_never_underestimates():
    assert asue([100, 50], [110, 60]) == 0.0


# --- the bundle and its guards ------------------------------------------


def test_evaluate_returns_every_measure():
    result = evaluate([100, 50, 30], [95, 55, 30])
    assert set(result) == {
        "rmse", "score", "r2", "asue", "n_engines", "mean_error", "n_late"
    }
    assert result["n_engines"] == 3
    # The 50 -> 55 engine is late, and so is the 30 -> 30 one: nasa_score puts
    # the boundary at d >= 0, and n_late must agree with it.
    assert result["n_late"] == 2
    assert result["mean_error"] == pytest.approx(0.0)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        rmse([1, 2, 3], [1, 2])


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        rmse([], [])


def test_non_finite_values_are_rejected():
    """A NaN normally means the model diverged, so it must not score silently."""
    with pytest.raises(ValueError, match="finite"):
        rmse([1.0, 2.0], [1.0, float("nan")])


def test_the_measures_accept_numpy_arrays_and_lists_alike():
    truth, guess = np.array([100.0, 50.0]), np.array([90.0, 60.0])
    assert rmse(truth, guess) == pytest.approx(rmse([100, 50], [90, 60]))
