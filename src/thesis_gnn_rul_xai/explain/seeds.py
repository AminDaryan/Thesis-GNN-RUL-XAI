"""Exact Shapley values for every saved network of a stored run, one network at a time.

Read README.md in this folder. It explains everything this file does.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.pipeline import PreparedData
from ..data.windowing import WindowedDataset
from ..models.estgnn import ESTGNN
from ..training.run import evaluate_model
from ..training.stored import StoredRun, load_seed
from .shap import dynamic_shap, mean_window_baseline

__all__ = ["SeedExplanation", "longest_engine", "shap_for_each_seed"]

# The sets an engine can be explained from. The first two keep every window of
# an engine; the test set keeps only its last one.
SPLITS: tuple[str, ...] = ("validation", "train", "test")

# How far, in cycles, a reloaded network's test error may sit from the one its
# run recorded before it counts as a different network.
RECORDED_ERROR_TOLERANCE = 1e-4


@dataclass(frozen=True)
class SeedExplanation:
    """One saved network of a stored run, the data rebuilt for it, and its engines explained."""

    seed: int
    model: ESTGNN
    prepared: PreparedData
    baseline: np.ndarray
    reference: float
    tables: dict[int, pd.DataFrame]


def longest_engine(dataset: WindowedDataset) -> int:
    """The engine with the most windows in a set, the lowest-numbered one on a tie."""
    if len(dataset) == 0:
        raise ValueError(
            "this set holds no windows, so it has no longest engine. A run that "
            "held nothing back has an empty validation set: pass split='train' "
            "and name the engines."
        )
    units, counts = np.unique(dataset.units, return_counts=True)
    return int(units[np.argmax(counts)])


def shap_for_each_seed(
    run: StoredRun,
    *,
    units: Sequence[int] | None = None,
    split: str = "validation",
    seeds: Sequence[int] | None = None,
    data_dir: str | Path | None = None,
    device: torch.device | str | None = None,
) -> Iterator[SeedExplanation]:
    """Load each saved seed in turn, check it is the network the run recorded, and explain whole engines."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}.")
    if split == "test" and units is None:
        raise ValueError(
            "the test set keeps one window per engine, so it has no longest "
            "engine: name the engines with units=."
        )
    chosen = list(run.seeds if seeds is None else seeds)
    unknown = sorted(set(chosen) - set(run.seeds))
    if unknown:
        raise ValueError(f"{run.name} trained seeds {run.seeds}, not {unknown}.")
    recorded = run.results.set_index("seed")["rmse"]

    for seed in chosen:
        model, prepared = load_seed(run, seed, data_dir=data_dir)
        scored = evaluate_model(model, prepared.test, device=device)["rmse"]
        if abs(scored - float(recorded.loc[seed])) > RECORDED_ERROR_TOLERANCE:
            raise ValueError(
                f"seed {seed} of {run.name} scores {scored:.6f} cycles on the test "
                f"engines where the run recorded {float(recorded.loc[seed]):.6f}. The "
                "data rebuilt today, or the network, is not what the run trained, "
                "so its explanation would describe something else."
            )
        dataset = getattr(prepared, split)
        engines = [longest_engine(dataset)] if units is None else [int(u) for u in units]
        baseline = mean_window_baseline(prepared.train)
        on = torch.device(device) if device is not None else next(model.parameters()).device
        with torch.no_grad():
            reference = float(model.to(on).eval().predict_cycles(torch.from_numpy(baseline[None]).to(on))[0])
        symbols = list(prepared.sensor_symbols)
        tables = {
            unit: dynamic_shap(model, dataset, baseline, unit=unit, symbols=symbols, device=device)
            for unit in engines
        }
        yield SeedExplanation(
            seed=seed,
            model=model,
            prepared=prepared,
            baseline=baseline,
            reference=reference,
            tables=tables,
        )
