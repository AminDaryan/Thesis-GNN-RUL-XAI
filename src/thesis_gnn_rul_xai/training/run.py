"""Trains the network, predicts with it, and scores it.

Read README.md in this folder. It explains everything this file does, and
warns that the held-out error and the test error must never be swapped.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..data.windowing import WindowedDataset
from ..evaluation.metrics import evaluate
from ..models.estgnn import ESTGNN
from ..utils.logging_setup import get_logger
from .config import TrainingConfig

__all__ = [
    "TrainingHistory",
    "evaluate_model",
    "predict",
    "resolve_device",
    "train",
]

logger = get_logger(__name__)


def resolve_device(preference: str = "auto") -> torch.device:
    """Decide whether to run on the graphics card or the processor, and say so."""
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' was requested but PyTorch cannot see a CUDA device. "
            "Either install the CUDA build of PyTorch (see README.md, section "
            "4) or use device='auto'."
        )
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" or torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Training on GPU: %s", torch.cuda.get_device_name(device))
        return device
    logger.info("No CUDA device available; training on the CPU.")
    return torch.device("cpu")


@dataclass
class TrainingHistory:
    """The record of one finished training run. README.md lists every field."""

    epochs: pd.DataFrame
    best_epoch: int
    best_validation_rmse: float
    stopped_early: bool
    seconds: float
    config: TrainingConfig
    _best_state: dict[str, torch.Tensor] = field(repr=False, default_factory=dict)

    def summary(self) -> str:
        """One paragraph describing the run, to print when it finishes."""
        reason = "early stopping" if self.stopped_early else "the epoch limit"
        opening = (
            f"Trained for {len(self.epochs)} epoch(s) in {self.seconds:.1f}s, "
            f"ending on {reason}."
        )
        if math.isnan(self.best_validation_rmse):
            return (
                f"{opening} No engines were held back, so nothing was measured "
                f"between epochs and no epoch was chosen over any other. The "
                f"model holds the weights from the last one."
            )
        return (
            f"{opening} At its best, epoch {self.best_epoch}, the model was out "
            f"by {self.best_validation_rmse:.3f} flight cycles on the engines "
            f"held back from training. It holds the weights from that epoch."
        )

    def save(self, directory: str | Path) -> Path:
        """Write the per-epoch table and the settings into a folder."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.epochs.to_csv(directory / "training_history.csv", index=False)
        self.config.save(directory / "training_config.yaml")
        logger.info("Saved the training history to %s", directory)
        return directory


def _build_scheduler(
    optimiser: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build the rule that shrinks the learning rate over the run.

    Which of the five rules to use is ASSUMPTION A-17; see config.py.
    """
    span = config.final_learning_rate / config.learning_rate
    if config.lr_schedule == "constant" or config.epochs == 1:
        return None
    if config.lr_schedule == "step":
        # One drop, straight from the first rate to the last, so the run only
        # ever uses the two values Table 6 actually prints. Halfway unless the
        # caller says otherwise, which spends equal time at each.
        when = config.lr_step_epoch or max(1, config.epochs // 2)
        return torch.optim.lr_scheduler.MultiStepLR(
            optimiser, milestones=[when], gamma=span
        )
    if config.lr_schedule == "exponential":
        # Chosen so that the last epoch lands exactly on final_learning_rate.
        gamma = span ** (1.0 / (config.epochs - 1))
        return torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=gamma)
    if config.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=config.epochs - 1, eta_min=config.final_learning_rate
        )
    # "plateau": react to the validation loss rather than to the epoch number.
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=0.1,
        patience=max(1, (config.early_stopping_patience or 20) // 2),
        min_lr=config.final_learning_rate,
    )


def _as_loader(
    dataset: WindowedDataset,
    rul_max: int,
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Repackage the examples as something PyTorch can hand out in batches.

    Three separate jobs: convert the arrays to tensors, shrink the labels into
    the range the last layer can reach, and pair each window with its label.
    README.md in this folder walks through all three.
    """
    features = torch.from_numpy(np.ascontiguousarray(dataset.features)).float()
    targets = torch.from_numpy(np.ascontiguousarray(dataset.targets)).float()
    # ASSUMPTION A-13: divide the labels by rul_max so they fit the range the
    # last layer can produce, and multiply predictions back afterwards. The
    # paper never states this arithmetic. Written up in docs/ASSUMPTIONS.md.
    targets = targets / float(rul_max)
    return DataLoader(
        TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


@torch.no_grad()
def predict(
    model: ESTGNN,
    dataset: WindowedDataset,
    *,
    batch_size: int = 256,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Predict remaining life in flight cycles, one number per window."""
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device).eval()

    features = torch.from_numpy(np.ascontiguousarray(dataset.features)).float()
    outputs: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].to(device)
        outputs.append(model.predict_cycles(batch).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def train(
    model: ESTGNN,
    train_set: WindowedDataset,
    validation_set: WindowedDataset | None = None,
    config: TrainingConfig | None = None,
    *,
    on_epoch: Callable[[dict[str, float]], None] | None = None,
) -> TrainingHistory:
    """Train the network and return the record of the run.

    The model is changed in place. Holding no engines back is allowed and is
    what the paper does; README.md explains what changes when you do.
    """
    config = config or TrainingConfig()
    device = resolve_device(config.device)

    # One seed covers weight initialisation (already done when the model was
    # built) and the batch order below. Reported in the history so a run can be
    # repeated exactly.
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    has_validation = validation_set is not None and len(validation_set) > 0
    if not has_validation:
        logger.warning(
            "No validation set. Training will run for all %d epoch(s) with no "
            "early stopping, which is what the paper describes, but there is "
            "then no honest way to choose between settings.",
            config.epochs,
        )

    model.to(device)
    train_loader = _as_loader(
        train_set, model.rul_max, config.batch_size, shuffle=True, generator=generator
    )
    loss_function = nn.MSELoss()
    # ASSUMPTION A-40: nothing is added to keep the weights small or to stop
    # the network memorising its training engines. Adam keeps PyTorch's weight
    # decay of 0, the network has no dropout, and no step is limited unless
    # gradient_clip is set, which it is not by default. Section 3.4 and Table 6
    # of the paper name the loss, the optimiser, the two rates, the passes and
    # the batch size, and nothing of the kind. The only guard against
    # over-training is the held-back engines and stopping on them (assumptions
    # A-06 and A-18). Written up in docs/ASSUMPTIONS.md.
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = _build_scheduler(optimiser, config)
    is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    # The plateau schedule only moves when it is handed a held-out error, and
    # below there is nothing to hand it. Without this warning the run finishes
    # at its starting learning rate, never reaching the final rate Table 6
    # gives, and nothing in the output or the run directory name says so.
    if is_plateau and not has_validation:
        logger.warning(
            "lr_schedule='plateau' reacts to the held-out error, and nothing is "
            "held out. The learning rate will stay at %.1e for the whole run and "
            "never reach the %.1e of Table 6. Use 'exponential' or 'cosine' to "
            "decay without a held-out set, or hold engines back.",
            config.learning_rate,
            config.final_learning_rate,
        )

    rows: list[dict[str, float]] = []
    best_rmse = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    stopped_early = False
    started = time.perf_counter()

    logger.info(
        "Training for up to %d epoch(s) on %d window(s), batch size %d, "
        "learning rate %.1e -> %.1e (%s schedule).",
        config.epochs, len(train_set), config.batch_size,
        config.learning_rate, config.final_learning_rate, config.lr_schedule,
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_function(model(features), targets)
            loss.backward()
            if config.gradient_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimiser.step()
            running += loss.item() * features.size(0)
            seen += features.size(0)

        learning_rate = optimiser.param_groups[0]["lr"]
        row: dict[str, float] = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "learning_rate": learning_rate,
        }

        if has_validation:
            assert validation_set is not None
            predicted = predict(model, validation_set, device=device)
            truth = validation_set.targets.astype(float)
            validation_rmse = float(np.sqrt(np.mean((predicted - truth) ** 2)))
            row["validation_loss"] = float(
                np.mean(((predicted - truth) / model.rul_max) ** 2)
            )
            row["validation_rmse_cycles"] = validation_rmse

            if validation_rmse < best_rmse - config.min_improvement:
                best_rmse, best_epoch = validation_rmse, epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        rows.append(row)
        if on_epoch is not None:
            on_epoch(dict(row))

        if epoch % 10 == 0 or epoch == 1:
            message = f"epoch {epoch:>4}/{config.epochs}  train {row['train_loss']:.5f}"
            if has_validation:
                message += f"  validation {row['validation_rmse_cycles']:.3f} cycles"
            message += f"  lr {learning_rate:.2e}"
            logger.info(message)

        if scheduler is not None:
            if is_plateau:
                if has_validation:
                    scheduler.step(row["validation_rmse_cycles"])
            else:
                scheduler.step()

        if (
            config.early_stopping_patience is not None
            and has_validation
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            logger.info(
                "Stopping at epoch %d: the validation error has not improved "
                "for %d epoch(s). Restoring the weights from epoch %d.",
                epoch, epochs_without_improvement, best_epoch,
            )
            break

    model.load_state_dict(best_state)
    history = TrainingHistory(
        epochs=pd.DataFrame(rows),
        best_epoch=best_epoch,
        best_validation_rmse=best_rmse if has_validation else float("nan"),
        stopped_early=stopped_early,
        seconds=time.perf_counter() - started,
        config=config,
        _best_state=best_state,
    )
    logger.info(history.summary())
    return history


def evaluate_model(
    model: ESTGNN,
    dataset: WindowedDataset,
    *,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Predict, then score. Returns the same dictionary as evaluate()."""
    predicted = predict(model, dataset, device=device)
    return evaluate(dataset.targets.astype(float), predicted)
