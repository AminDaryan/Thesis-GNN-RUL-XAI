"""Tests for the training loop and its settings.

These run on the real FD001 windows but for only a handful of epochs, so they
check that the machinery is correct rather than that the model is good.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.models import ESTGNN
from thesis_gnn_rul_xai.training import (
    TrainingConfig,
    evaluate_model,
    load_training_config,
    predict,
    resolve_device,
    train,
)


@pytest.fixture(scope="module")
def prepared(raw_dir):
    """A small FD001 preparation: window 20, so the run stays quick."""
    return run_preprocessing(
        PreprocessingConfig(window_size=20, validation_fraction=0.25, seed=7),
        data_dir=raw_dir,
    )


def _model(prepared) -> ESTGNN:
    return ESTGNN(
        prepared.adjacency_normalized,
        in_features=prepared.train.features.shape[2],
        rul_max=prepared.config.rul_max,
    )


# --- the settings object ------------------------------------------------


def test_defaults_are_the_values_of_table_6():
    config = TrainingConfig()
    assert config.epochs == 200
    assert config.batch_size == 32
    assert config.learning_rate == 1e-3
    assert config.final_learning_rate == 1e-4
    assert config.optimizer == "adam"
    assert config.loss == "mse"


def test_a_growing_learning_rate_is_rejected():
    with pytest.raises(ValueError, match="must not exceed"):
        TrainingConfig(learning_rate=1e-4, final_learning_rate=1e-3)


def test_unimplemented_optimiser_and_loss_are_rejected():
    with pytest.raises(ValueError, match="Only 'adam'"):
        TrainingConfig(optimizer="sgd")
    with pytest.raises(ValueError, match="Only 'mse'"):
        TrainingConfig(loss="mae")


def test_settings_round_trip_through_yaml(tmp_path):
    config = TrainingConfig(epochs=7, lr_schedule="cosine", seed=3)
    config.save(tmp_path / "training.yaml")
    assert load_training_config(tmp_path / "training.yaml") == config


def test_an_unknown_key_in_the_settings_file_is_rejected(tmp_path):
    path = tmp_path / "training.yaml"
    path.write_text("epochs: 5\nlearnign_rate: 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_training_config(path)


def test_device_resolution():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}


# --- the loop -----------------------------------------------------------


def test_training_reduces_the_error(prepared):
    model = _model(prepared)
    history = train(
        model,
        prepared.train,
        prepared.validation,
        TrainingConfig(epochs=3, device="cpu", early_stopping_patience=None),
    )
    assert len(history.epochs) == 3
    first, last = history.epochs["train_loss"].iloc[0], history.epochs["train_loss"].iloc[-1]
    assert last < first


def test_the_history_records_every_epoch(prepared):
    model = _model(prepared)
    history = train(
        model, prepared.train, prepared.validation,
        TrainingConfig(epochs=2, device="cpu", early_stopping_patience=None),
    )
    assert list(history.epochs.columns) == [
        "epoch", "train_loss", "learning_rate",
        "validation_loss", "validation_rmse_cycles",
    ]
    assert history.epochs["epoch"].tolist() == [1, 2]


def test_the_learning_rate_reaches_its_stated_final_value(prepared):
    """A-17: the exponential schedule must land exactly on final_learning_rate."""
    model = _model(prepared)
    history = train(
        model, prepared.train, prepared.validation,
        TrainingConfig(epochs=5, device="cpu", early_stopping_patience=None,
                       learning_rate=1e-3, final_learning_rate=1e-4),
    )
    rates = history.epochs["learning_rate"].tolist()
    assert rates[0] == pytest.approx(1e-3)
    assert rates[-1] == pytest.approx(1e-4, rel=1e-6)
    assert rates == sorted(rates, reverse=True)      # never increases


def test_a_constant_schedule_never_moves_the_rate(prepared):
    model = _model(prepared)
    history = train(
        model, prepared.train, prepared.validation,
        TrainingConfig(epochs=3, device="cpu", lr_schedule="constant",
                       early_stopping_patience=None),
    )
    assert history.epochs["learning_rate"].nunique() == 1


def test_early_stopping_restores_the_best_weights(prepared):
    """An impossible improvement threshold makes patience run out at once."""
    model = _model(prepared)
    history = train(
        model, prepared.train, prepared.validation,
        TrainingConfig(epochs=30, device="cpu", early_stopping_patience=2,
                       min_improvement=1e6),
    )
    assert history.stopped_early
    assert len(history.epochs) < 30
    # The weights in the model are the ones from the best epoch, not the last.
    best_row = history.epochs.loc[history.epochs["epoch"] == history.best_epoch]
    assert float(best_row["validation_rmse_cycles"].iloc[0]) == pytest.approx(
        history.best_validation_rmse
    )
    reported = evaluate_model(model, prepared.validation)["rmse"]
    assert reported == pytest.approx(history.best_validation_rmse, rel=1e-4)


def test_training_without_a_validation_set_runs_to_the_epoch_limit(prepared):
    """The paper's own procedure: fixed epochs, nothing held out."""
    model = _model(prepared)
    history = train(
        model, prepared.train, None,
        TrainingConfig(epochs=2, device="cpu"),
    )
    assert len(history.epochs) == 2
    assert not history.stopped_early
    assert "validation_rmse_cycles" not in history.epochs.columns


def test_the_run_is_repeatable(prepared):
    """Same seed, same data, same numbers."""
    results = []
    for _ in range(2):
        torch.manual_seed(11)
        model = _model(prepared)
        train(
            model, prepared.train, prepared.validation,
            TrainingConfig(epochs=2, device="cpu", seed=11,
                           early_stopping_patience=None),
        )
        results.append(predict(model, prepared.test))
    np.testing.assert_allclose(results[0], results[1], rtol=1e-6)


def test_history_can_be_saved(prepared, tmp_path):
    model = _model(prepared)
    history = train(
        model, prepared.train, prepared.validation,
        TrainingConfig(epochs=2, device="cpu", early_stopping_patience=None),
    )
    directory = history.save(tmp_path / "run")
    assert (directory / "training_history.csv").is_file()
    assert load_training_config(directory / "training_config.yaml") == history.config


# --- predictions --------------------------------------------------------


def test_predictions_are_in_cycles_and_respect_the_ceiling(prepared):
    model = _model(prepared)
    predicted = predict(model, prepared.test)
    assert predicted.shape == (len(prepared.test),)
    assert float(predicted.min()) >= 0.0
    assert float(predicted.max()) <= prepared.config.rul_max


def test_evaluate_model_returns_the_four_measures(prepared):
    model = _model(prepared)
    result = evaluate_model(model, prepared.test)
    assert {"rmse", "score", "r2", "asue"} <= set(result)
    assert result["n_engines"] == len(prepared.test)
