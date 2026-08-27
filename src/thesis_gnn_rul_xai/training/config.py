"""
Training settings
=================

Table 6 of Wen et al. (2026) lists six hyper-parameters, and every one of them
appears here with the paper's value as its default:

===================  =========================
Table 6              Field
===================  =========================
Learning rate        ``learning_rate`` to ``final_learning_rate``
Epochs               ``epochs``
Batch size           ``batch_size``
Time steps           (set in ``PreprocessingConfig.window_size``)
Optimizer            ``optimizer``
Loss function        ``loss``
===================  =========================

Two things Table 6 does not say are recorded as assumptions rather than hidden
in the code: **how** the learning rate travels from one value to the other
(A-17), and when to stop early (A-18). Both are marked below.

As with :mod:`thesis_gnn_rul_xai.config`, one file fully describes one
experiment, so the settings can be saved next to the results.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

__all__ = [
    "LearningRateSchedule",
    "TrainingConfig",
    "load_training_config",
]

#: How the learning rate moves from ``learning_rate`` to
#: ``final_learning_rate``.
#:
#: ``"exponential"``
#:     Multiply by a constant factor after every epoch, chosen so the final
#:     epoch lands exactly on ``final_learning_rate``. This is the reading of
#:     "beginning at 1e-3 and reducing to 1e-4" that arrives at the stated end
#:     value, and it is the default.
#: ``"cosine"``
#:     Follow a cosine curve between the two values: slow at first, fastest in
#:     the middle, slow again at the end.
#: ``"plateau"``
#:     Drop the rate by a factor of ten only when the validation loss stops
#:     improving, never going below ``final_learning_rate``. This is what
#:     "adaptive learning rate schedule" means in some of the literature, so it
#:     is offered as the main alternative reading of the paper's sentence.
#: ``"constant"``
#:     Never change it. Included so the effect of the schedule can be measured.
LearningRateSchedule = Literal["exponential", "cosine", "plateau", "constant"]


@dataclass
class TrainingConfig:
    """Everything that controls one training run.

    Attributes
    ----------
    epochs : int
        Number of passes over the training set. Table 6 gives 200.
    batch_size : int
        Examples per gradient step. Table 6 gives 32.
    learning_rate : float
        Starting learning rate. Table 6 gives 1e-3.
    final_learning_rate : float
        Learning rate at the last epoch. Table 6 gives 1e-4.
    lr_schedule : LearningRateSchedule
        How to travel between the two. ASSUMPTION A-17: Table 6 gives both
        endpoints and the words "adaptive learning rate schedule", but never
        the mechanism.
    optimizer : str
        Currently only ``"adam"``, which is what Table 6 specifies.
    loss : str
        Currently only ``"mse"``, which is what Table 6 specifies.
    early_stopping_patience : int or None
        Stop when the validation error has not improved for this many epochs,
        and restore the weights from the best epoch. ``None`` trains for the
        full ``epochs`` regardless, which is what the paper describes.
        ASSUMPTION A-18: the paper has no validation set and no early stopping;
        this is a deliberate improvement, of a piece with A-06.
    min_improvement : float
        How much the validation error must fall, in cycles, to count as an
        improvement worth resetting the patience counter for.
    seed : int
        Seed for weight initialisation and batch shuffling.
    device : str
        ``"auto"`` picks a CUDA device when one is present, otherwise the CPU.
        ``"cpu"`` and ``"cuda"`` force the choice.
    gradient_clip : float or None
        Maximum gradient norm, or ``None`` to leave gradients untouched. Not
        mentioned by the paper; ``None`` by default so the run matches it.
    notes : str
        Free text stored alongside the settings.
    """

    # --- from Table 6 ------------------------------------------------------
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    final_learning_rate: float = 1e-4
    optimizer: str = "adam"
    loss: str = "mse"

    # --- not stated by the paper -------------------------------------------
    lr_schedule: LearningRateSchedule = "exponential"  # ASSUMPTION A-17
    early_stopping_patience: int | None = 20           # ASSUMPTION A-18
    min_improvement: float = 1e-4
    gradient_clip: float | None = None

    # --- execution ---------------------------------------------------------
    seed: int = 42
    device: str = "auto"

    notes: str = field(
        default=(
            "Training configuration. Values under 'from Table 6' are taken from "
            "Wen et al. (2026); lr_schedule and early_stopping_patience are "
            "assumptions A-17 and A-18 in docs/ASSUMPTIONS.md."
        )
    )

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}.")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}."
            )
        if self.final_learning_rate <= 0:
            raise ValueError(
                "final_learning_rate must be positive, got "
                f"{self.final_learning_rate}."
            )
        if self.final_learning_rate > self.learning_rate:
            raise ValueError(
                "final_learning_rate must not exceed learning_rate: the paper "
                "describes a rate that reduces, not one that grows. Got "
                f"{self.learning_rate} -> {self.final_learning_rate}."
            )
        if self.optimizer.lower() != "adam":
            raise ValueError(
                f"Only 'adam' is implemented, which is what Table 6 of the "
                f"paper specifies. Got {self.optimizer!r}."
            )
        if self.loss.lower() != "mse":
            raise ValueError(
                f"Only 'mse' is implemented, which is what Table 6 of the paper "
                f"specifies. Got {self.loss!r}."
            )
        valid_schedules = ("exponential", "cosine", "plateau", "constant")
        if self.lr_schedule not in valid_schedules:
            raise ValueError(
                f"lr_schedule must be one of {list(valid_schedules)}, got "
                f"{self.lr_schedule!r}."
            )
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError(
                "early_stopping_patience must be at least 1, or None to disable "
                f"early stopping. Got {self.early_stopping_patience}."
            )
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError(
                f"device must be 'auto', 'cpu' or 'cuda', got {self.device!r}."
            )
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError(
                f"gradient_clip must be positive or None, got {self.gradient_clip}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a plain dictionary, ready to be saved."""
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> Path:
        """Write the configuration to a YAML file and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path


def load_training_config(path: str | Path) -> TrainingConfig:
    """Read a :class:`TrainingConfig` from a YAML file.

    Raises
    ------
    ValueError
        If the file contains a key that is not a field of the configuration.
        A silently ignored typo in a settings file is a common and
        hard-to-find source of wrong results, so this is deliberately strict.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    known = {f.name for f in dataclasses.fields(TrainingConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown configuration key(s) in {path}: {sorted(unknown)}. "
            f"Valid keys are: {sorted(known)}."
        )
    return TrainingConfig(**raw)
