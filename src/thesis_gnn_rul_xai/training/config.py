"""The settings for one training run, with the paper's values as the defaults.

Read README.md in this folder. It explains what every setting below means and
which of them the paper does not state.
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

# The five ways the learning rate can travel from its first value to its last.
# README.md in this folder describes each one.
LearningRateSchedule = Literal[
    "step", "exponential", "cosine", "plateau", "constant"
]


@dataclass
class TrainingConfig:
    """Everything that controls one training run. README.md explains each field."""

    # --- from Table 6 ------------------------------------------------------
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    final_learning_rate: float = 1e-4
    optimizer: str = "adam"
    loss: str = "mse"

    # --- not stated by the paper -------------------------------------------
    # ASSUMPTION A-17: drop the learning rate once, by a single factor, at the
    # halfway point. The paper gives the first and last rate but never how it
    # gets between them. Its two rates are exactly one factor of ten apart,
    # which is one step of the kind Li, Zhao, Sun, Yan and Chen (2021) use
    # ("multiplying 0.1 in epochs 120 and 160"); no C-MAPSS paper found decays
    # exponentially. Where the step falls is not stated either, so the halfway
    # point is used: it spends equal time at each of the two rates the paper
    # does state. Written up in docs/assumptions/A-17.md and docs/LITERATURE_BASIS.md.
    lr_schedule: LearningRateSchedule = "step"
    # Which epoch the step happens on. None means halfway through the run.
    lr_step_epoch: int | None = None
    # ASSUMPTION A-18: stop early once the held-out error stops improving. The
    # paper holds nothing out and never stops early, so this is a departure
    # from it, not a reconstruction. Written up in docs/assumptions/A-18.md.
    early_stopping_patience: int | None = 20
    min_improvement: float = 1e-4
    # ASSUMPTION A-40: no limit on the size of a step by default, because the
    # paper names none, and nothing else keeps the weights small either (the
    # comment where train() in run.py builds Adam). Written up in
    # docs/assumptions/A-40.md.
    gradient_clip: float | None = None

    # --- execution ---------------------------------------------------------
    seed: int = 42
    device: str = "auto"

    notes: str = field(
        default=(
            "Training settings. The values under 'from Table 6' come from "
            "Wen et al. (2026). lr_schedule and early_stopping_patience are "
            "our own choices, written up as A-17 and A-18 in "
            "docs/assumptions/."
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
        if self.lr_step_epoch is not None and not 1 <= self.lr_step_epoch <= self.epochs:
            raise ValueError(
                "lr_step_epoch must fall inside the run, between 1 and "
                f"{self.epochs}, or be None for halfway. Got {self.lr_step_epoch}."
            )
        valid_schedules = ("step", "exponential", "cosine", "plateau", "constant")
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
    """Read settings back from a YAML file. An unknown key is an error."""
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
