"""The figures Wen et al. (2026) print, transcribed once so every page quotes the same ones."""

from __future__ import annotations

from dataclasses import dataclass

# An ablation is a run with one part of the network removed, to see what that
# part was worth; ABLATION_ROWS holds the paper's Table 11 of them.
__all__ = [
    "ABLATION_ROWS",
    "DATASET_DESCRIPTION",
    "FIG_10_RING",
    "SHAP_TABLE_12",
    "SHAP_TABLE_12_GRAPHS",
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


# Table 12, "SHAP importance values for 14 sensors across degradation phases",
# for one engine the paper does not name. Its columns G1 to G191 are the numbers
# of Fig. 10's graphs. Transcribed from the PDF's text layer on 2026-09-14, with
# the paper's "Phi" spelt as this project spells it. Every cell is positive,
# which signed Shapley values cannot all be (F-08).
# ASSUMPTION A-35: graph g is read as the engine's g-th window, the one ending
# at cycle g + 29, so graph 1 ends at cycle 30 and graph 191 at cycle 220. The
# paper heads the columns "Time" and never says which cycle a graph is.
SHAP_TABLE_12_GRAPHS: tuple[int, ...] = (1, 2, 3, 4, 5, 78, 79, 80, 81, 82, 187, 188, 189, 190, 191)
SHAP_TABLE_12: dict[str, tuple[float, ...]] = {
    "T24": (3.39, 3.18, 2.94, 2.69, 2.44, 2.20, 1.99, 1.81, 1.81, 1.68, 1.47, 1.46, 1.36, 1.22, 0.76),
    "T30": (3.49, 2.78, 2.40, 2.27, 2.29, 2.41, 2.46, 2.42, 2.46, 2.21, 1.49, 1.49, 1.11, 0.82, 0.76),
    "T50": (4.18, 3.98, 3.79, 3.60, 3.41, 3.21, 3.00, 2.78, 2.76, 2.54, 2.06, 1.97, 1.84, 1.65, 1.42),
    "P15": (1.79, 1.63, 1.42, 1.37, 1.24, 1.32, 1.39, 1.40, 1.38, 1.34, 0.78, 0.79, 0.58, 0.46, 0.38),
    "P30": (3.33, 2.88, 2.68, 2.65, 2.71, 2.78, 2.79, 2.66, 2.66, 2.34, 1.44, 1.42, 1.05, 0.84, 1.31),
    "Nf": (3.15, 2.67, 2.35, 2.15, 2.04, 1.97, 1.91, 1.82, 1.78, 1.66, 1.25, 1.21, 1.05, 0.90, 0.86),
    "Nc": (4.20, 3.89, 3.57, 3.25, 2.95, 2.68, 2.46, 2.30, 2.30, 2.21, 2.21, 2.21, 2.28, 2.37, 2.60),
    "Ps30": (4.69, 4.16, 3.76, 3.46, 3.22, 3.01, 2.80, 2.56, 2.56, 2.28, 1.88, 1.88, 1.94, 2.29, 4.23),
    "phi": (3.91, 4.04, 4.01, 3.84, 3.58, 3.26, 2.91, 2.58, 2.58, 2.29, 1.85, 1.85, 1.70, 1.59, 1.51),
    "NRf": (2.44, 2.06, 1.87, 1.81, 1.83, 1.87, 1.85, 1.79, 1.74, 1.58, 0.99, 0.98, 0.73, 0.59, 0.89),
    "BPR": (3.77, 3.47, 3.17, 2.88, 2.62, 2.39, 2.21, 2.10, 2.10, 2.05, 2.03, 2.02, 1.98, 1.84, 1.18),
    "htBleed": (1.53, 1.29, 1.20, 1.21, 1.27, 1.34, 1.36, 1.30, 1.28, 1.12, 0.58, 0.54, 0.34, 0.20, 0.44),
    "W31": (2.30, 2.15, 2.01, 1.86, 1.72, 1.58, 1.44, 1.31, 1.32, 1.18, 0.95, 0.95, 0.84, 0.74, 0.56),
    "W32": (3.34, 3.15, 2.96, 2.78, 2.60, 2.42, 2.24, 2.07, 2.02, 1.90, 1.55, 1.53, 1.36, 1.16, 0.70),
}

# The order Fig. 10 places the sensors around its circle, clockwise from the
# node just left of the top. Read at scale 8 on 2026-09-14; Fig. 9 uses the same.
FIG_10_RING: tuple[str, ...] = (
    "P30", "P15", "T50", "T30", "T24", "W32", "W31",
    "htBleed", "BPR", "NRf", "phi", "Ps30", "Nc", "Nf",
)


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
