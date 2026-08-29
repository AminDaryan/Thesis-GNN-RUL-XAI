"""
The training loop
=================

What happens here
-----------------
One pass of this module trains the network of Table 4 on the windows produced
by the preprocessing pipeline, using the hyper-parameters of Table 6, and
returns a record of what happened at every epoch.

The order of operations per epoch is the ordinary one: shuffle the training
windows, take gradient steps in batches of 32, then measure the error on the
validation engines without updating anything. The validation figure is what
decides when to stop.

Targets are scaled to [0, 1]; results are reported in cycles
------------------------------------------------------------
The network ends in a sigmoid, so its output can only lie in ``[0, 1]``. A
label measured in flight cycles is out of reach for such a layer: no setting of
the weights makes a sigmoid emit 40. Two conversions therefore bracket the
model (assumption A-13). On the way in, ``_as_loader`` divides every target by
``rul_max`` before the loss ever sees it; on the way out,
``ESTGNN.predict_cycles`` multiplies the sigmoid back by the same constant, so
``predict`` returns cycles.

Worked example, for a single window whose true remaining life is 40 cycles,
with the default ``rul_max = 125``::

    label from the preprocessor   40                        cycles
    target given to the loss      40 / 125 = 0.32           normalised
    what the sigmoid emits        0.36                      normalised
    what MSELoss sees             (0.36 - 0.32)^2 = 0.0016  normalised
    prediction handed back        0.36 * 125 = 45           cycles
    error a reader is shown       45 - 40 = 5               cycles

Because the constant is the same for every window, the change of unit cannot
change any comparison: an RMSE in cycles is exactly ``rul_max`` times the
normalised RMSE. Each epoch of the history accordingly records both
``validation_loss`` (mean squared error, normalised) and
``validation_rmse_cycles`` (root mean squared error, in cycles), and
``sqrt(validation_loss) * rul_max`` reproduces the second from the first. The
two are kept apart for readability rather than arithmetic: 0.0016 is not a
quantity anyone can judge, and "five cycles of remaining life" is.

The distinction that matters most in this file
----------------------------------------------
The validation set here is **not** a stand-in for the test set, and its error
must never be reported as though it were. The two weight their engines quite
differently:

* validation holds *every* window of the held-out engines, so on FD001 its 20
  engines contribute between 108 and 240 windows each and a single unusual
  engine can move the figure;
* the test set holds *one* window per engine, so each of the 100 engines is
  worth exactly one per cent.

Neither is reliably the harder of the two, and the gap between them says
nothing about the model. Validation error is the right signal for deciding when
to stop training and for comparing settings, because those are relative
judgements made against a fixed set. It is the wrong number to publish. See
assumption A-06.

What the paper does instead
---------------------------
Wen et al. (2026) train for a fixed 200 epochs with no validation set and no
early stopping described. Keeping early stopping here is a deliberate
improvement (A-18); setting ``early_stopping_patience=None`` reproduces their
procedure exactly.

References
----------
Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
168, 110875. https://doi.org/10.1016/j.ast.2025.110875
"""

from __future__ import annotations

import copy
import math
import time
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

__all__ = ["TrainingHistory", "predict", "resolve_device", "train"]

logger = get_logger(__name__)


def resolve_device(preference: str = "auto") -> torch.device:
    """Turn a device preference into a concrete :class:`torch.device`.

    ``"auto"`` uses a CUDA device when one is available and the CPU otherwise,
    and says which it chose, because a run that silently falls back to the CPU
    and takes forty times as long is a confusing way to spend an afternoon.
    """
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
    """What happened during a run, and the weights it ended on.

    Attributes
    ----------
    epochs : pandas.DataFrame
        One row per epoch: the training loss, the validation loss, the
        validation error in cycles, and the learning rate that was in force.
    best_epoch : int
        The epoch whose weights the model now holds (1-based).
    best_validation_rmse : float
        Validation error in cycles at that epoch.
    stopped_early : bool
        Whether patience ran out before ``epochs`` was reached.
    seconds : float
        Wall-clock duration of the run.
    config : TrainingConfig
        The settings used, saved so a result can be traced back to them.
    """

    epochs: pd.DataFrame
    best_epoch: int
    best_validation_rmse: float
    stopped_early: bool
    seconds: float
    config: TrainingConfig
    _best_state: dict[str, torch.Tensor] = field(repr=False, default_factory=dict)

    def summary(self) -> str:
        """A one-paragraph description, for printing at the end of a run."""
        reason = "early stopping" if self.stopped_early else "the epoch limit"
        return (
            f"Trained for {len(self.epochs)} epoch(s) in {self.seconds:.1f}s, "
            f"ending on {reason}. Best validation error {self.best_validation_rmse:.3f} "
            f"cycles at epoch {self.best_epoch}. The model holds the weights "
            f"from that epoch."
        )

    def save(self, directory: str | Path) -> Path:
        """Write the per-epoch table and the settings to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.epochs.to_csv(directory / "training_history.csv", index=False)
        self.config.save(directory / "training_config.yaml")
        logger.info("Saved the training history to %s", directory)
        return directory


def _build_scheduler(
    optimiser: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create the learning-rate schedule described by A-17."""
    span = config.final_learning_rate / config.learning_rate
    if config.lr_schedule == "constant" or config.epochs == 1:
        return None
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
    """Wrap a windowed dataset as a PyTorch loader with normalised targets."""
    features = torch.from_numpy(np.ascontiguousarray(dataset.features)).float()
    targets = torch.from_numpy(np.ascontiguousarray(dataset.targets)).float()
    targets = targets / float(rul_max)          # ASSUMPTION A-13
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
    """Predict remaining life **in flight cycles** for every window.

    Parameters
    ----------
    model : ESTGNN
        A trained (or untrained) network.
    dataset : WindowedDataset
        Windows to predict for, in their existing order.
    batch_size : int
        How many windows to push through at once. Larger is faster and uses
        more memory; it has no effect on the numbers.
    device : torch.device or str, optional
        Where to run. Defaults to wherever the model already is.

    Returns
    -------
    numpy.ndarray
        One prediction per window, shaped ``(len(dataset),)``, in cycles.
    """
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
) -> TrainingHistory:
    """Train the network and return a record of the run.

    Parameters
    ----------
    model : ESTGNN
        The network to train. It is modified in place, and on return it holds
        the weights from the best epoch rather than the last one.
    train_set, validation_set : WindowedDataset
        The windows to learn from, and the ones to watch. ``validation_set``
        may be ``None`` or empty, in which case there is nothing to stop early
        on and the run lasts the full ``config.epochs``, which is what the
        paper describes.
    config : TrainingConfig, optional
        Settings. Defaults to the values of Table 6.

    Returns
    -------
    TrainingHistory
        The per-epoch table, the best epoch, and the settings used.
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
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = _build_scheduler(optimiser, config)
    is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

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
    """Score a trained model on a windowed dataset, in cycles.

    A convenience wrapper that runs :func:`predict` and hands the result to
    :func:`thesis_gnn_rul_xai.evaluation.metrics.evaluate`.
    """
    predicted = predict(model, dataset, device=device)
    return evaluate(dataset.targets.astype(float), predicted)
