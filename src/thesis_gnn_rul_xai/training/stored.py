"""Finds a finished run in reports/ and loads one of its trained networks back.

Read README.md in this folder. It explains what a stored run holds and why the
data has to be rebuilt with the seed the weights were trained under.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ..config import PreprocessingConfig, load_config, settings_without_notes
from ..data.pipeline import PreparedData, run_preprocessing
from ..models import ESTGNN, ModelConfig, build_model, load_model_config
from ..utils.paths import reports_dir

__all__ = ["StoredRun", "find_runs", "load_seed", "read_run"]


@dataclass(frozen=True)
class StoredRun:
    """One finished run in reports/: where it is, and what it recorded about itself."""

    path: Path
    summary: dict[str, Any]
    preprocessing: PreprocessingConfig
    model_config: ModelConfig
    results: pd.DataFrame

    @property
    def name(self) -> str:
        """The run's directory name, which carries its settings, commit and time."""
        return self.path.name

    @property
    def seeds(self) -> list[int]:
        """Every seed the run trained, in the order it trained them."""
        return [int(seed) for seed in self.results["seed"]]

    def weights(self, seed: int) -> Path:
        """Where the trained weights of one seed are saved."""
        return self.path / f"seed_{seed}" / "model.pt"

    def has_weights(self) -> bool:
        """Whether every seed of the run saved its trained weights."""
        return all(self.weights(seed).is_file() for seed in self.seeds)


def read_run(path: str | Path) -> StoredRun:
    """Read one run directory. Refuses a directory that is not a finished run."""
    path = Path(path)
    if not path.is_dir():
        path = reports_dir() / str(path)
    needed = ("summary.json", "results.csv", "preprocessing_config.yaml")
    missing = [name for name in needed if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path} is not a finished run: it has no {missing}.")

    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    model_config = (
        load_model_config(path / "model_config.yaml")
        if (path / "model_config.yaml").is_file()
        # Runs from before model_config.yaml existed recorded only the stage.
        else ModelConfig(
            temporal=summary.get("temporal_stage", "last_cycle"),
            temporal_hidden=summary.get("temporal_hidden") or 32,
        )
    )
    return StoredRun(
        path=path,
        summary=summary,
        preprocessing=load_config(path / "preprocessing_config.yaml"),
        model_config=model_config,
        results=pd.read_csv(path / "results.csv"),
    )


def find_runs(
    *,
    preprocessing: PreprocessingConfig | None = None,
    seeds: int | None = None,
    with_weights: bool = False,
    root: str | Path | None = None,
    **summary_fields: Any,
) -> list[StoredRun]:
    """Every stored run matching the given settings, oldest first."""
    root = Path(root) if root is not None else reports_dir()
    wanted = (
        settings_without_notes(preprocessing.to_dict()) if preprocessing is not None else {}
    )
    found: list[StoredRun] = []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        try:
            run = read_run(path)
        except (FileNotFoundError, ValueError, TypeError):
            # Not a run, or a run written before a setting was renamed; such a
            # run is skipped rather than guessed at.
            continue
        stored = run.preprocessing.to_dict()
        if any(stored.get(key) != value for key, value in wanted.items()):
            continue
        if seeds is not None and run.summary.get("seeds") != seeds:
            continue
        if any(run.summary.get(key) != value for key, value in summary_fields.items()):
            continue
        if with_weights and not run.has_weights():
            continue
        found.append(run)
    return sorted(found, key=lambda run: run.summary.get("finished_utc", ""))


def load_seed(
    run: StoredRun,
    seed: int,
    *,
    data_dir: str | Path | None = None,
) -> tuple[ESTGNN, PreparedData]:
    """The trained network of one seed, and the data rebuilt exactly as it saw it."""
    if seed not in run.seeds:
        raise ValueError(f"{run.name} trained seeds {run.seeds}, not {seed}.")
    if not run.weights(seed).is_file():
        raise FileNotFoundError(
            f"{run.weights(seed)} does not exist. Runs made before 2026-09-14 saved "
            "no weights, so they cannot be loaded; train the configuration again."
        )
    # Unless the run held the split fixed, the seed decides which engines were
    # held back, so the data must be rebuilt under the same seed.
    preprocessing = (
        run.preprocessing
        if run.summary.get("split") == "fixed"
        else replace(run.preprocessing, seed=seed)
    )
    prepared = run_preprocessing(preprocessing, data_dir=data_dir)
    model = build_model(
        prepared.adjacency_normalized,
        run.model_config,
        window_size=preprocessing.window_size,
        rul_max=preprocessing.rul_max,
    )
    model.load_state_dict(torch.load(run.weights(seed), map_location="cpu"))
    model.eval()
    return model, prepared
