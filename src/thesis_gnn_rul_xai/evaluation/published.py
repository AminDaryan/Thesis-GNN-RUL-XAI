"""The figures Wen et al. (2026) print, transcribed once so every page quotes the same ones."""

from __future__ import annotations

from dataclasses import dataclass

# An ablation is a run with one part of the network removed, to see what that
# part was worth; ABLATION_ROWS holds the paper's Table 11 of them.
__all__ = [
    "ABLATION_ROWS",
    "DATASET_DESCRIPTION",
    "SUBSETS",
    "TRAINING_SETTINGS",
    "PublishedRow",
    "published_row",
]

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

# Table 5, "Description of C-MAPSS dataset". Transcribed from the PDF's text
# layer on 2026-09-14.
DATASET_DESCRIPTION: dict[str, dict[str, int]] = {
    "FD001": {"train_engines": 100, "test_engines": 100, "conditions": 1, "fault_modes": 1},
    "FD002": {"train_engines": 260, "test_engines": 259, "conditions": 6, "fault_modes": 1},
    "FD003": {"train_engines": 100, "test_engines": 100, "conditions": 1, "fault_modes": 2},
    "FD004": {"train_engines": 249, "test_engines": 248, "conditions": 6, "fault_modes": 2},
}

# Table 6. The paper prints the learning rate as "1e-3 -> 1e-4" and nothing
# about how it moves between the two, which is A-17.
TRAINING_SETTINGS: dict[str, object] = {
    "learning_rate": 1e-3,
    "final_learning_rate": 1e-4,
    "epochs": 200,
    "batch_size": 32,
    "time_steps": 30,
    "optimizer": "Adam",
    "loss": "MSE",
}


@dataclass(frozen=True)
class PublishedRow:
    """One row of the paper's results, for one sub-dataset."""

    key: str
    label: str
    source: str
    rmse: float
    score: float | None = None
    r2: float | None = None
    asue: float | None = None


# The ESTGNN row of Tables 7 (score), 8 (RMSE), 9 (R squared) and 10 (ASUE).
_FULL = {
    "FD001": (11.78, 278.6, 0.921, 8.312),
    "FD002": (16.21, 1438.0, 0.603, 21.623),
    "FD003": (12.34, 332.3, 0.932, 8.511),
    "FD004": (18.67, 2043.0, 0.510, 22.196),
}

# Table 11, RMSE only: the ablation table prints no other measure.
_ABLATED = {
    "without_gcn": ("ESTGNN w/o GCN", (16.42, 28.15, 17.67, 27.52)),
    "without_lstm": ("ESTGNN w/o LSTM", (14.93, 23.34, 16.25, 24.21)),
    "without_physics_prior": ("ESTGNN w/o Physics prior", (12.34, 18.12, 12.84, 20.97)),
}

# The averages and deltas Table 11 prints, kept only so a test can check the
# transcription above against the paper's own arithmetic.
ABLATION_ROWS: dict[str, dict[str, float]] = {
    "full_average": {"rmse": 14.75},
    "without_gcn": {"average": 22.44, "FD001_delta": 4.64},
    "without_lstm": {"average": 19.68, "FD001_delta": 3.15},
    "without_physics_prior": {"average": 16.07, "FD001_delta": 0.56},
}


def published_row(key: str, subset: str) -> PublishedRow:
    """The paper's figures for one model variant on one sub-dataset."""
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {list(SUBSETS)}, got {subset!r}.")
    column = SUBSETS.index(subset)
    if key == "full":
        rmse, score, r2, asue = _FULL[subset]
        return PublishedRow(
            key=key,
            label="ESTGNN (Full)",
            source="Tables 7, 8, 9 and 10",
            rmse=rmse,
            score=score,
            r2=r2,
            asue=asue,
        )
    if key in _ABLATED:
        label, values = _ABLATED[key]
        return PublishedRow(key=key, label=label, source="Table 11", rmse=values[column])
    raise ValueError(
        f"no published row called {key!r}; the paper prints 'full' and "
        f"{sorted(_ABLATED)}."
    )
