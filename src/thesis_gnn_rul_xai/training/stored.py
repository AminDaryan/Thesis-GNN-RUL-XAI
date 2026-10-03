"""Finds a finished run in reports/ and loads one of its trained networks back.

Read README.md in this folder. It explains what a stored run holds and why the
data has to be rebuilt with the seed the weights were trained under.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import PreprocessingConfig, load_config, settings_without_notes
from ..data.pipeline import PreparedData, run_preprocessing
from ..graph.adjacency import connections_of
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


# How far a rescaled graph saved with a network may differ from the one rebuilt
# for it. Both are the same float32 arithmetic on the same graph, so any real
# difference is far larger.
GRAPH_TOLERANCE = 1e-6


def _named_pairs(pairs: Any, symbols: list[str]) -> list[str]:
    """Connections as 'A-B', each pair and the list in the order of the sensors."""
    def place(symbol: str) -> tuple[int, str]:
        return (symbols.index(symbol) if symbol in symbols else len(symbols), symbol)

    ordered = [sorted(pair, key=place) for pair in pairs]
    ordered.sort(key=lambda pair: (place(pair[0]), place(pair[1])))
    return [f"{a}-{b}" for a, b in ordered]


def _differences(expected: set[frozenset[str]], rebuilt: set[frozenset[str]], symbols: list[str]) -> str:
    """How a rebuilt graph's connections differ from the expected ones, naming up to three of each."""
    def shown(pairs: set[frozenset[str]]) -> str:
        names = _named_pairs(pairs, symbols)
        return ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")

    missing, added = expected - rebuilt, rebuilt - expected
    if not missing:
        return f"the rebuild joins {len(added)} pair(s) that graph does not ({shown(added)})."
    words = f"{len(missing)} of its {len(expected)} connections are missing from the rebuild ({shown(missing)})"
    return words + (f", which joins {len(added)} other pair(s) instead ({shown(added)})." if added else ".")


def _check_graph(run: StoredRun, seed: int, prepared: PreparedData, saved: dict[str, Any]) -> None:
    """Refuse a rebuilt sensor graph that is not the one this seed's network was trained on."""
    symbols = [str(symbol) for symbol in prepared.sensor_symbols]
    rebuilt = {frozenset((symbols[i], symbols[j])) for i, j in connections_of(prepared.adjacency_raw)}
    where = f"{run.name}, seed {seed}"
    advice = (" Something that decides this graph has changed since the run was made: the code, the run's "
              "settings file or the NASA files. The data handed back would not be the data the network was "
              f"trained on, so rebuild it from a checkout of the run's commit, {run.summary.get('commit', 'unknown')}, "
              "or train the configuration again.")
    record_path = run.path / f"seed_{seed}" / "graph.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("graph") != prepared.config.graph:
            raise ValueError(f"{where}: seed_{seed}/graph.json records a {record.get('graph')!r} graph, but "
                             f"the run's settings rebuild a {prepared.config.graph!r} one." + advice)
        if record.get("drawn_from_seed") != prepared.diagnostics.get("graph_seed"):
            raise ValueError(f"{where}: seed_{seed}/graph.json records a graph drawn from seed "
                             f"{record.get('drawn_from_seed')}, but the rebuild drew it from seed "
                             f"{prepared.diagnostics.get('graph_seed')}." + advice)
        recorded = {frozenset(pair) for pair in record["connections"]}
        if recorded != rebuilt:
            raise ValueError(f"{where}: the sensor graph rebuilt for this seed is not the one recorded in "
                             f"seed_{seed}/graph.json: " + _differences(recorded, rebuilt, symbols) + advice)
    # The rescaled graph saved with the weights is the one the network was
    # trained on, and every stored run has it, so it is checked with or without
    # graph.json.
    held = saved.get("adjacency")
    if held is None:
        return
    held = held.detach().cpu().numpy().astype(np.float64)
    expected = np.asarray(prepared.adjacency_normalized, dtype=np.float64)
    if held.shape != expected.shape:
        raise ValueError(f"{where}: the graph saved inside seed_{seed}/model.pt joins {held.shape[0]} "
                         f"sensors, but the rebuilt one joins {expected.shape[0]}." + advice)
    joined = (held != 0) | (held.T != 0)
    np.fill_diagonal(joined, False)
    trained = {frozenset((symbols[i], symbols[j])) for i, j in connections_of(joined)}
    if trained != rebuilt:
        raise ValueError(f"{where}: the sensor graph rebuilt for this seed is not the one saved inside "
                         f"seed_{seed}/model.pt, the graph the network was trained on: "
                         + _differences(trained, rebuilt, symbols) + advice)
    gap = float(np.abs(held - expected).max())
    if gap > GRAPH_TOLERANCE:
        raise ValueError(f"{where}: the rebuilt graph joins the same sensors as the one saved inside "
                         f"seed_{seed}/model.pt, but its rescaled numbers differ from the saved ones by up "
                         f"to {gap:.3g}, so it was rescaled differently." + advice)


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
    # held back, so the data must be rebuilt under the same seed. A run that
    # held nothing back records "none" since 2026-10-03 and is rebuilt under
    # each seed, because scripts/train.py now prepares every such seed under
    # its own seed even when told --fixed-split. Before that day it recorded
    # "varies_with_seed", and was prepared under each seed, or "fixed" under
    # --fixed-split, and was prepared once; each is rebuilt the way it was
    # prepared. The seed matters even with nothing held back: it numbers the
    # operating conditions of the per-condition scaling and draws a random or
    # rewired graph (README.md, "The split part of a run folder's name").
    preprocessing = (
        run.preprocessing
        if run.summary.get("split") == "fixed"
        else replace(run.preprocessing, seed=seed)
    )
    prepared = run_preprocessing(preprocessing, data_dir=data_dir)
    saved = torch.load(run.weights(seed), map_location="cpu")
    # Checked before the weights go in: loading them puts the trained graph
    # back into the network whatever was rebuilt, so a rebuilt graph that
    # differs would otherwise pass unseen (README.md, "The graph a reload
    # rebuilds must be the one the network was trained on").
    _check_graph(run, seed, prepared, saved)
    model = build_model(
        prepared.adjacency_normalized,
        run.model_config,
        window_size=preprocessing.window_size,
        rul_max=preprocessing.rul_max,
    )
    model.load_state_dict(saved)
    model.eval()
    return model, prepared
