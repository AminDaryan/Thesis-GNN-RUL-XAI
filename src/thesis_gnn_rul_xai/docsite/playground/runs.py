"""The runs and explanations stored in reports/, read for the page, and two runs compared."""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ...config import settings_without_notes
from ...evaluation.published import SUBSETS
from .design import caveats
from .settings import (
    _METHOD_NAMES,
    GAMMA_SETTINGS,
    RESULT_SEEDS,
    SettingsError,
    paper_row_for,
)


def validate_explain(payload: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """The request to explain one stored run, checked."""
    if not isinstance(payload, Mapping):
        raise SettingsError("The request must be a JSON object.")
    run = _run_directory(root, str(payload.get("run", "")))
    method = payload.get("method", "shap")
    if method not in _METHOD_NAMES:
        raise SettingsError(
            f"method must be 'shap', 'occlusion' or 'walks', got {method!r}."
        )
    scope = payload.get("scope", "engine")
    if scope not in ("engine", "test"):
        raise SettingsError(f"scope must be 'engine' or 'test', got {scope!r}.")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SettingsError(f"seed must be a whole number, got {seed!r}.")
    if method == "walks" and _spatial_of(run) == "none":
        raise SettingsError("This run's network has no graph layers, so there are no walks through the "
                            "sensor graph to explain; Shapley values and occlusion still apply.")
    if not (run / f"seed_{seed}" / "model.pt").is_file():
        raise SettingsError(
            f"This run saved no weights for seed {seed}. Runs made before 2026-09-14 "
            "saved none; train the configuration again to explain it."
        )
    engine = payload.get("engine")
    if engine is not None and (isinstance(engine, bool) or not isinstance(engine, int)):
        raise SettingsError(f"engine must be a whole number or empty, got {engine!r}.")
    every = payload.get("every", 1)
    if isinstance(every, bool) or not isinstance(every, int) or not 1 <= every <= 100:
        raise SettingsError(f"every must be a whole number from 1 to 100, got {every!r}.")
    # Only a name from the list is taken, never numbers, so no request can
    # hand the first graph layer a gamma: each name gives it 0.
    gamma = payload.get("gamma")
    names = [setting["value"] for setting in GAMMA_SETTINGS]
    if method != "walks":
        if gamma is not None:
            raise SettingsError("gamma is a setting of walk explanations only.")
    elif gamma is None:
        gamma = names[0]
    elif gamma not in names:
        raise SettingsError(f"gamma must be one of {names}, got {gamma!r}.")
    return {"run": run.name, "method": method, "scope": scope, "seed": seed,
            "engine": engine, "every": every, "gamma": gamma}


# -- stored runs ------------------------------------------------------------


def _spatial(summary: Mapping[str, Any], model: Mapping[str, Any] | None) -> str:
    """Whether a stored run's network had graph layers: "gcn", or "none" when they were left out."""
    # A run stored before the choice existed recorded nothing, and had them.
    return str((model or {}).get("spatial") or summary.get("spatial") or "gcn")


def _spatial_of(run: Path) -> str:
    """The graph-layer choice of the run stored in one folder, read from its summary and model files."""
    try:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "gcn"
    model = summary.get("model") or (
        _read_yaml(run / "model_config.yaml") if (run / "model_config.yaml").is_file() else None)
    return _spatial(summary, model if isinstance(model, Mapping) else None)



def _clean(value: Any) -> Any:
    """Replace NaN and infinity with None, which JSON can carry and they cannot."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    return value


def _run_directory(root: Path, name: str) -> Path:
    """A run folder in reports/, refusing any name that is not one."""
    reports = (root / "reports").resolve()
    candidate = Path(name)
    path = (reports / candidate).resolve()
    # A sensitivity study groups runs below one study folder, so a valid name
    # may contain several parts.  Every part must still stay below reports/.
    # Looking only for separators is not enough on Windows: a name carrying a
    # drive, "C:x", can throw the base away.
    try:
        relative = path.relative_to(reports)
    except ValueError:
        relative = None
    if (
        not name
        or candidate.is_absolute()
        or candidate.drive
        or relative is None
        or relative == Path(".")
        or any(part in {"", ".", ".."} or part.startswith(".") for part in relative.parts)
    ):
        raise SettingsError(f"{name!r} is not the name of a run.")
    if not (path / "summary.json").is_file():
        raise SettingsError(f"There is no finished run called {name!r} in reports/.")
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


@functools.lru_cache(maxsize=4096)
def _yaml_in_state(path: str, changed: int, size: int) -> dict[str, Any]:
    """A YAML file read once for each state it has been in, named by its time of change and its size."""
    return _read_yaml(Path(path))


def _seed_settings(folder: Path) -> dict[str, Any]:
    """The training settings one seed's folder stored, empty when it stored none."""
    path = folder / "training_config.yaml"
    try:
        state = path.stat()
    except OSError:
        return {}
    return dict(_yaml_in_state(str(path), state.st_mtime_ns, state.st_size))


def _seed_number(folder: Path) -> tuple[int, str]:
    """A seed folder's place in the order of the seeds, by number where its name ends in one."""
    tail = folder.name.removeprefix("seed_")
    return (int(tail), "") if tail.isdigit() else (10**9, tail)


#: What stands for a seed that stored no value where the other seeds of its run stored one.
NOT_RECORDED = "not recorded"


def _recorded_parameters(path: Path, summary: Mapping[str, Any], config: Mapping[str, Any],
                         model: Mapping[str, Any] | None, described: Mapping[str, Any]) -> dict[str, Any]:
    """Every setting one stored run recorded, on one level, named as the comparison of two runs names them."""
    preprocessing = settings_without_notes(config)
    for key in ("subset", "graph", "adjacency_normalization", "clip_test_labels", "validation_fraction"):
        if key not in preprocessing and key in summary:
            preprocessing[key] = summary[key]
    network = dict(model or {})
    for key, written_as in (("temporal", "temporal_stage"), ("temporal_hidden", "temporal_hidden")):
        if key not in network and written_as in summary:
            network[key] = summary[written_as]
    # A run stored before the graph layers could be left out recorded no choice, and had them.
    network.setdefault("spatial", described["spatial"])

    whole_run = summary.get("training")
    whole_run = dict(whole_run) if isinstance(whole_run, Mapping) else {}
    folders = sorted((folder for folder in path.glob("seed_*") if folder.is_dir()), key=_seed_number)
    # The prose notes change no number, here as in the preprocessing settings.
    per_seed = [settings_without_notes(_seed_settings(folder) or whole_run) for folder in folders] \
        or [settings_without_notes(whole_run)]
    training: dict[str, Any] = {}
    for key in dict.fromkeys(key for saved in per_seed for key in saved):
        # The seed is what differs between the seeds of one run, by design.
        if key == "seed":
            continue
        values = [saved.get(key, NOT_RECORDED) for saved in per_seed]
        same = all(json.dumps(value, sort_keys=True) == json.dumps(values[0], sort_keys=True) for value in values)
        training[key] = values[0] if same else values
    if "lr_schedule" not in training and summary.get("lr_schedule") is not None:
        training["lr_schedule"] = summary["lr_schedule"]

    found: dict[str, Any] = {}
    for group, values in (("preprocessing", preprocessing), ("model", network), ("training", training)):
        for key, value in values.items():
            found[f"{group}.{key}"] = value
    found["seeds"] = described["seeds"]
    found["commit"] = described["commit"]
    for key in ("split", "reported_weights"):
        if described.get(key) is not None:
            found[key] = described[key]
    if "early_stopping" in summary:
        found["early_stopping"] = summary["early_stopping"]
    return found


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """A CSV file as a list of rows, with missing values as None."""
    import pandas as pd

    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    return _clean(frame.astype(object).where(frame.notna(), None).to_dict(orient="records"))


def _describe(path: Path, name: str | None = None) -> dict[str, Any] | None:
    """One stored run, as the results table lists it."""
    try:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if "mean" not in summary or not (path / "results.csv").is_file():
        return None
    config = _read_yaml(path / "preprocessing_config.yaml")
    model = summary.get("model") or (
        _read_yaml(path / "model_config.yaml") if (path / "model_config.yaml").is_file() else None
    )
    temporal = summary.get("temporal_stage")
    spatial = _spatial(summary, model if isinstance(model, Mapping) else None)
    subset = str(summary.get("subset", config.get("subset", "")))
    validation_fraction = config.get("validation_fraction", summary.get("validation_fraction"))
    row, reason = (paper_row_for(subset, temporal, summary.get("graph"), spatial)
                   if subset in SUBSETS else (None, "Unknown sub-dataset."))
    seeds = int(summary.get("seeds", 0))
    commit = str(summary.get("commit", "unknown"))
    settings_for_caveats = {
        "seeds": seeds, "temporal": temporal, "spatial": spatial, "subset": subset,
        "window_size": config.get("window_size"),
        "graph": summary.get("graph"), "clip_test_labels": summary.get("clip_test_labels"),
        "validation_fraction": validation_fraction or 0.0,
        "fixed_split": summary.get("split") == "fixed",
        "gcn_hidden": (model or {}).get("gcn_hidden"),
        "fc_hidden": (model or {}).get("fc_hidden"),
        "shared_across_sensors": (model or {}).get("shared_across_sensors"),
        "lr_schedule": summary.get("lr_schedule"),
    }
    described = {
        "name": name or path.name,
        "commit": commit,
        "dirty": commit.endswith("-dirty"),
        "finished": str(summary.get("finished_utc", "")),
        "subset": subset,
        "seeds": seeds,
        "exploratory": seeds < RESULT_SEEDS,
        "temporal": temporal,
        "temporal_hidden": summary.get("temporal_hidden"),
        "spatial": spatial,
        "graph": summary.get("graph"),
        "adjacency_normalization": (summary.get("adjacency_normalization")
                                    or config.get("adjacency_normalization")),
        # What happened, not only what was recorded: with nothing held back there
        # is no split, which runs since 2026-10-03 record as "none" and runs
        # before it recorded as "varies_with_seed" (audit item D-4).
        "split": "none" if validation_fraction == 0 else summary.get("split"),
        "normalization": config.get("normalization"),
        "smoothing_alpha": config.get("smoothing_alpha"),
        "window_size": config.get("window_size"),
        "rul_max": config.get("rul_max"),
        "clip_test_labels": summary.get("clip_test_labels"),
        "validation_fraction": validation_fraction,
        # What train() does, read from the settings rather than trusted from a
        # field: with nothing held back it can only keep the last pass.
        "reported_weights": summary.get("reported_weights") or (
            None if validation_fraction is None else
            "final_epoch" if not validation_fraction else "best_validation_epoch"
        ),
        "lr_schedule": summary.get("lr_schedule"),
        "model": model,
        "training": summary.get("training"),
        "has_weights": any(path.glob("seed_*/model.pt")),
        "mean": summary.get("mean"),
        "std": summary.get("std"),
        "paper_row": row,
        "no_row_reason": reason,
        "caveats": caveats(settings_for_caveats),
    }
    described["parameters"] = _recorded_parameters(
        path, summary, config, model if isinstance(model, Mapping) else None, described)
    return _clean(described)


def list_runs(root: Path) -> list[dict[str, Any]]:
    """Every finished run in reports/, newest first."""
    reports = root / "reports"
    if not reports.is_dir():
        return []
    runs = []
    for results_path in sorted(reports.rglob("results.csv")):
        path = results_path.parent
        relative = path.relative_to(reports)
        if any(part.startswith(".") for part in relative.parts):
            continue
        described = _describe(path, relative.as_posix())
        if described is not None:
            runs.append(described)
    runs.sort(key=lambda run: run["finished"], reverse=True)
    return runs


def run_detail(root: Path, name: str) -> dict[str, Any]:
    """One run in full: every seed's numbers, curves, predictions and explanations."""
    path = _run_directory(root, name)
    described = _describe(path, name)
    if described is None:
        raise SettingsError(f"{name!r} is not a training run.")
    seeds = {}
    for folder in sorted(path.glob("seed_*")):
        seeds[folder.name.removeprefix("seed_")] = {
            "history": _read_csv(folder / "training_history.csv"),
            "predictions": _read_csv(folder / "test_predictions.csv"),
            "has_weights": (folder / "model.pt").is_file(),
            "training": _read_yaml(folder / "training_config.yaml"),
        }
    explanations = []
    for summary_path in sorted((path / "explanations").glob("*/summary.json")):
        try:
            explanations.append(
                {"name": summary_path.parent.name,
                 **json.loads(summary_path.read_text(encoding="utf-8"))}
            )
        except (OSError, json.JSONDecodeError):
            continue
    explanations.sort(key=lambda item: str(item.get("finished_utc", "")), reverse=True)
    return _clean({
        **described,
        "results": _read_csv(path / "results.csv"),
        "preprocessing": _read_yaml(path / "preprocessing_config.yaml"),
        "recorded_summary": json.loads((path / "summary.json").read_text(encoding="utf-8")),
        "recorded_model": _read_yaml(path / "model_config.yaml"),
        "per_seed": seeds,
        "explanations": explanations,
    })


def list_explanations(root: Path) -> list[dict[str, Any]]:
    """Every stored explanation in reports/, with the run it explains, newest first."""
    found = []
    for run in list_runs(root):
        for summary_path in sorted((root / "reports" / run["name"] / "explanations").glob("*/summary.json")):
            try:
                stored = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            found.append({**stored, "name": summary_path.parent.name, "run": run["name"],
                          "run_subset": run.get("subset"), "run_temporal": run.get("temporal"),
                          "run_temporal_hidden": run.get("temporal_hidden"), "run_graph": run.get("graph"),
                          "run_spatial": run.get("spatial"),
                          "run_finished": run.get("finished")})
    found.sort(key=lambda item: str(item.get("finished_utc", "")), reverse=True)
    return _clean(found)


def _explanation_edges(run: Path, summary: Mapping[str, Any]) -> list[list[str]] | None:
    """Read the chosen seed's actual graph, or report that it is unavailable."""
    import pickle

    import numpy as np

    from ...data.sensors import PAPER_SENSOR_SYMBOLS

    try:
        import torch

        seed = int(summary["seed"])
        saved = torch.load(run / f"seed_{seed}" / "model.pt", map_location="cpu", weights_only=True)
        matrix = saved["adjacency"].detach().cpu().numpy()
        symbols = PAPER_SENSOR_SYMBOLS
        if matrix.shape != (len(symbols), len(symbols)) or not np.isfinite(matrix).all():
            return None
        return [[a, b] for i, a in enumerate(symbols) for j, b in enumerate(symbols)
                if i < j and (matrix[i, j] != 0 or matrix[j, i] != 0)]
    except (ImportError, OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError,
            pickle.UnpicklingError):
        return None


def explanation_detail(root: Path, run: str, name: str) -> dict[str, Any]:
    """One stored explanation: its summary and every value it computed."""
    path = (_run_directory(root, run) / "explanations").resolve()
    folder = (path / name).resolve()
    # The same hazard as a run name: a drive carried in the name escapes.
    if not name or name.startswith(".") or folder.parent != path:
        raise SettingsError(f"{name!r} is not the name of an explanation.")
    if not (folder / "summary.json").is_file():
        raise SettingsError(f"Run {run!r} has no explanation called {name!r}.")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    layers = _spatial_of(path.parent) != "none"
    return _clean({
        "summary": summary,
        "values": _read_csv(folder / "values.csv"),
        # Only a walk explanation writes this file; the others get an empty list.
        "top_walks": _read_csv(folder / "top_walks.csv"),
        # The GNN-LRP paper's own check, stored only with walk explanations.
        "flipping": _read_csv(folder / "flipping.csv"),
        # A network with no graph layers never used its sensor graph, so it
        # has no connections to draw, whatever graph its data were prepared with.
        "graph_layers": layers,
        "graph_edges": (None if summary.get("method") == "walks"
                        else _explanation_edges(path.parent, summary) if layers else []),
    })


_COMPARED = ("rmse", "score", "r2", "asue")


def _flat_settings(detail: Mapping[str, Any]) -> dict[str, Any]:
    """Every setting of a run on one level, for spotting what two runs change."""
    flat: dict[str, Any] = {}
    for key, value in settings_without_notes(detail.get("preprocessing") or {}).items():
        if key != "seed":
            flat[f"preprocessing.{key}"] = value
    for key, value in (detail.get("model") or {"temporal": detail.get("temporal")}).items():
        flat[f"model.{key}"] = value
    for key, value in (detail.get("training") or {}).items():
        flat[f"training.{key}"] = value
    flat["seeds"] = detail.get("seeds")
    flat["split"] = detail.get("split")
    flat["commit"] = detail.get("commit")
    return flat


def compare_runs(root: Path, first: str, second: str) -> dict[str, Any]:
    """Two runs paired seed by seed: what differs, by how much, and whether seeds explain it."""
    a, b = run_detail(root, first), run_detail(root, second)
    flat_a, flat_b = _flat_settings(a), _flat_settings(b)
    differences = [
        {"setting": key, "first": flat_a.get(key), "second": flat_b.get(key)}
        for key in sorted(set(flat_a) | set(flat_b))
        if flat_a.get(key) != flat_b.get(key)
    ]
    by_seed_a = {row["seed"]: row for row in a["results"]}
    by_seed_b = {row["seed"]: row for row in b["results"]}
    shared = sorted(set(by_seed_a) & set(by_seed_b))

    measures: dict[str, Any] = {}
    for measure in _COMPARED:
        # Each pair keeps its own seed. A seed missing the measure -- an ASUE
        # with no engine predicted early is not a number, read here as None --
        # leaves the mean difference and the test undefined, as a mean over
        # seeds is (assumption A-39), rather than taken over the other seeds.
        triples = [(s, by_seed_a[s].get(measure), by_seed_b[s].get(measure)) for s in shared]
        entry: dict[str, Any] = {
            "per_seed": [{"seed": s, "first": x, "second": y,
                          "difference": None if x is None or y is None else y - x}
                         for s, x, y in triples],
            "mean_difference": None, "t": None, "p": None,
        }
        defined = all(x is not None and y is not None for _, x, y in triples)
        pairs = [(x, y) for _, x, y in triples] if defined else []
        if pairs:
            differences_here = [y - x for x, y in pairs]
            entry["mean_difference"] = sum(differences_here) / len(differences_here)
        if len(pairs) >= 2:
            from scipy import stats

            test = stats.ttest_rel([y for _, y in pairs], [x for x, _ in pairs])
            entry["t"], entry["p"] = float(test.statistic), float(test.pvalue)
        measures[measure] = entry
    return _clean({
        "first": {"name": a["name"], "mean": a["mean"], "std": a["std"], "seeds": a["seeds"]},
        "second": {"name": b["name"], "mean": b["mean"], "std": b["std"], "seeds": b["seeds"]},
        "shared_seeds": shared,
        "differences": differences,
        "measures": measures,
    })
