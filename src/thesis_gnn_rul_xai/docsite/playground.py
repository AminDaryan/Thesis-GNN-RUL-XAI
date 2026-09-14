"""The playground behind the website: checks settings, runs the real scripts, and reads what they stored.

Read README.md in this folder. It explains why every run started here goes
through scripts/train.py and lands in reports/, and what the page shows.
"""

from __future__ import annotations

import json
import math
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ..evaluation.published import (
    DATASET_DESCRIPTION,
    SUBSETS,
    TRAINING_SETTINGS,
    published_row,
)
from ..utils.logging_setup import get_logger

__all__ = [
    "FIELDS",
    "JobManager",
    "SettingsError",
    "compare_runs",
    "explanation_detail",
    "list_runs",
    "options",
    "paper_row_for",
    "preview",
    "run_detail",
    "train_arguments",
    "validate_explain",
    "validate_settings",
]

logger = get_logger(__name__)

# Where the playground keeps the files of the jobs it starts: the settings it
# wrote, the command's printed output and its progress. Inside reports/, which
# is not committed, and hidden so the run listing never mistakes it for a run.
JOBS_FOLDER = Path("reports") / ".playground"

# Fewer seeds than this is marked as exploratory everywhere it appears.
# CLAUDE.md rule 3: one seed is not a result, and five is the floor.
RESULT_SEEDS = 5


class SettingsError(ValueError):
    """A setting the playground refuses, with the reason in words."""


def _source(kind: str, ref: str, text: str) -> dict[str, str]:
    """Where a setting's default comes from: the paper, a logged choice, or neither."""
    return {"kind": kind, "ref": ref, "text": text}


def _paper(ref: str, text: str) -> dict[str, str]:
    return _source("paper", ref, text)


def _choice(ref: str, text: str) -> dict[str, str]:
    return _source("assumption", ref, text)


def _project(text: str) -> dict[str, str]:
    return _source("project", "", text)


def _extension(text: str) -> dict[str, str]:
    return _source("extension", "", text)


# Every setting the page offers, in the order of the pipeline. Each carries the
# source of its default, so the page can say beside every control whether the
# paper states the value, whether a numbered assumption chose it, or neither.
FIELDS: tuple[dict[str, Any], ...] = (
    # -- 1. data -----------------------------------------------------------
    {"key": "subset", "group": "data", "label": "Sub-dataset", "kind": "choice",
     "default": "FD001",
     "choices": [{"value": s, "label": s,
                  "source": _paper("Table 5", "one of the four C-MAPSS sub-datasets")}
                 for s in SUBSETS],
     "source": _paper("Table 5", "the paper reports all four")},
    # -- 2. preprocessing --------------------------------------------------
    {"key": "window_size", "group": "preprocessing", "label": "Window length (cycles)",
     "kind": "int", "default": 30, "min": 1, "max": 100,
     "source": _paper("Table 6", "\"Time steps 30\"")},
    {"key": "window_stride", "group": "preprocessing", "label": "Cycles between windows",
     "kind": "int", "default": 1, "min": 1, "max": 30,
     "source": _choice("A-05", "the paper does not say how far apart windows start")},
    {"key": "rul_max", "group": "preprocessing", "label": "Label ceiling (cycles)",
     "kind": "int", "default": 125, "min": 10, "max": 400,
     "source": _choice("A-01", "read off Fig. 6 by calibrating its plateau against its own axis")},
    {"key": "clip_test_labels", "group": "preprocessing", "label": "Cap the test labels too",
     "kind": "bool", "default": True,
     "source": _choice("A-02", "can only move reported numbers favourably")},
    {"key": "normalization", "group": "preprocessing", "label": "Sensor scaling",
     "kind": "choice", "default": "zscore",
     "choices": [
         {"value": "zscore", "label": "Mean and spread of the training engines",
          "source": _choice("A-03", "the default; the paper never describes its scaling (F-10)")},
         {"value": "minmax_per_condition", "label": "Range 0 to 1, per operating condition",
          "source": _choice("A-03", "an alternative weighed against the default")},
         {"value": "minmax_global", "label": "Range 0 to 1, over all rows",
          "source": _choice("A-03", "an alternative weighed against the default")},
     ],
     "source": _choice("A-03", "the paper does not describe its scaling")},
    {"key": "smoothing", "group": "preprocessing", "label": "Smooth each sensor",
     "kind": "bool", "default": True,
     "source": _choice("A-04", "exponential smoothing before windowing")},
    {"key": "smoothing_alpha", "group": "preprocessing", "label": "Smoothing weight on the newest reading",
     "kind": "float", "default": 0.5, "min": 0.01, "max": 1.0, "step": 0.05,
     "source": _choice("A-04", "the paper gives no value")},
    {"key": "validation_fraction", "group": "preprocessing",
     "label": "Training engines held back",
     "kind": "float", "default": 0.2, "min": 0.0, "max": 0.5, "step": 0.05,
     "source": _choice("A-06", "the paper describes no held-back engines; 0 is its procedure")},
    {"key": "graph", "group": "preprocessing", "label": "Sensor graph", "kind": "choice",
     "default": "physics",
     "choices": [
         {"value": "physics", "label": "The paper's physics graph",
          "source": _paper("Table 3", "41 edges between 14 sensors, also drawn as Fig. 3")},
         {"value": "complete", "label": "Every sensor joined to every other",
          "source": _choice("A-19", "stands in for the \"w/o Physics prior\" row of Table 11")},
         {"value": "correlation", "label": "The 41 most correlated sensor pairs",
          "source": _choice("A-27", "built from the training engines; the paper never "
                                    "builds one")},
     ],
     "source": _paper("Table 3", "the paper's own adjacency matrix")},
    {"key": "adjacency_normalization", "group": "preprocessing",
     "label": "Graph rescaling", "kind": "choice", "default": "row",
     "choices": [
         {"value": "row", "label": "Each row sums to one",
          "source": _choice("A-21", "the default")},
         {"value": "symmetric", "label": "Symmetric, by both degrees",
          "source": _choice("A-21", "the alternative, measured against the default")},
     ],
     "source": _choice("A-21", "the paper describes the step but prints no formula")},
    {"key": "seed", "group": "repeats", "label": "First seed", "kind": "int",
     "default": 42, "min": 0, "max": 2**31 - 1,
     "source": _project("each further seed adds one")},
    # -- 3. temporal stage -------------------------------------------------
    {"key": "temporal", "group": "temporal", "label": "Temporal stage", "kind": "choice",
     "default": "lstm",
     "choices": [
         {"value": "lstm", "label": "LSTM, one per sensor",
          "source": _paper("Section 2.2", "a dedicated recurrent network for each of the 14 sensors")},
         {"value": "last_cycle", "label": "Removed: keep only the last cycle",
          "source": _choice("A-07", "the \"w/o LSTM\" row of Table 11")},
         {"value": "gru", "label": "GRU, one per sensor",
          "source": _extension("a smaller recurrent cell; not in the paper")},
         {"value": "mean_window", "label": "Average over the window",
          "source": _extension("no learned parameters; not in the paper")},
         {"value": "raw_window", "label": "All 30 cycles straight into the graph layer",
          "source": _extension("the arrangement the last-cycle stage replaced; not in the paper")},
     ],
     "source": _paper("Section 2.2", "the paper's own stage")},
    {"key": "temporal_hidden", "group": "temporal", "label": "Hidden size", "kind": "int",
     "default": 32, "min": 1, "max": 512,
     "source": _choice("A-26", "the paper never states the size")},
    {"key": "temporal_out_features", "group": "temporal",
     "label": "Numbers handed to the graph per sensor", "kind": "int",
     "default": 1, "min": 1, "max": 64,
     "source": _choice("A-26", "1 is the node width Table 4 prints")},
    {"key": "shared_across_sensors", "group": "temporal",
     "label": "One network shared by all sensors", "kind": "bool", "default": False,
     "source": _paper("Section 2.2", "the paper gives every sensor its own; sharing is not the paper")},
    # -- 4. graph layers ---------------------------------------------------
    {"key": "gcn_hidden", "group": "graph", "label": "Graph layer widths", "kind": "widths",
     "default": [32, 64, 32], "min": 1, "max": 512, "length": 3,
     "source": _paper("Table 4", "GCN-1, GCN-2 and GCN-3 output 32, 64 and 32")},
    {"key": "fc_hidden", "group": "graph", "label": "First fully connected width",
     "kind": "int", "default": 128, "min": 1, "max": 2048,
     "source": _paper("Table 4", "FC-1 outputs 128")},
    {"key": "gcn_bias", "group": "graph", "label": "Bias in the graph layers",
     "kind": "bool", "default": False,
     "source": _choice("A-12", "Table 4 lists only a weight matrix per layer")},
    # -- 5. training -------------------------------------------------------
    {"key": "epochs", "group": "training", "label": "Passes over the training data",
     "kind": "int", "default": TRAINING_SETTINGS["epochs"], "min": 1, "max": 1000,
     "source": _paper("Table 6", "\"Epochs 200\"")},
    {"key": "batch_size", "group": "training", "label": "Windows per update",
     "kind": "int", "default": TRAINING_SETTINGS["batch_size"], "min": 1, "max": 4096,
     "source": _paper("Table 6", "\"Batch size 32\"")},
    {"key": "learning_rate", "group": "training", "label": "First learning rate",
     "kind": "float", "default": TRAINING_SETTINGS["learning_rate"],
     "min": 1e-6, "max": 1.0, "step": 1e-4,
     "source": _paper("Table 6", "\"1e-3 -> 1e-4\"")},
    {"key": "final_learning_rate", "group": "training", "label": "Last learning rate",
     "kind": "float", "default": TRAINING_SETTINGS["final_learning_rate"],
     "min": 1e-7, "max": 1.0, "step": 1e-5,
     "source": _paper("Table 6", "\"1e-3 -> 1e-4\"")},
    {"key": "lr_schedule", "group": "training", "label": "How the rate gets there",
     "kind": "choice", "default": "step",
     "choices": [
         {"value": v, "label": label, "source": _choice("A-17", note)}
         for v, label, note in (
             ("step", "One drop, halfway", "the default"),
             ("exponential", "The same factor every pass", "an alternative"),
             ("cosine", "A half cosine", "an alternative"),
             ("plateau", "Drop when held-back error stalls", "needs held-back engines"),
             ("constant", "Never changes", "ignores the last rate"),
         )
     ],
     "source": _choice("A-17", "the paper gives the first and last rate, not the path")},
    {"key": "early_stopping", "group": "training", "label": "Stop when held-back error stalls",
     "kind": "bool", "default": True,
     "source": _choice("A-18", "the paper describes no early stopping")},
    {"key": "early_stopping_patience", "group": "training",
     "label": "Passes allowed without improvement", "kind": "int",
     "default": 20, "min": 1, "max": 1000,
     "source": _choice("A-18", "the paper gives no value")},
    # -- 6. repeats --------------------------------------------------------
    {"key": "seeds", "group": "repeats", "label": "Seeds", "kind": "int",
     "default": RESULT_SEEDS, "min": 1, "max": 20,
     "source": _project("rule 3: fewer than five is exploratory, never a result")},
    {"key": "fixed_split", "group": "repeats", "label": "Same held-back engines for every seed",
     "kind": "bool", "default": False,
     "source": _project("understates the spread; say so wherever the number appears")},
    {"key": "device", "group": "repeats", "label": "Device", "kind": "choice",
     "default": "auto",
     "choices": [{"value": v, "label": v, "source": _project("where the arithmetic runs")}
                 for v in ("auto", "cpu", "cuda")],
     "source": _project("where the arithmetic runs")},
)

_BY_KEY = {spec["key"]: spec for spec in FIELDS}

GROUPS = (
    ("data", "Data"),
    ("preprocessing", "Preprocessing"),
    ("temporal", "Temporal stage"),
    ("graph", "Graph layers"),
    ("training", "Training"),
    ("repeats", "Repeats"),
)

# The four model variants of Table 11, as settings. "w/o GCN" cannot be built:
# the graph layers are the one part this playground keeps.
ABLATIONS = (
    {"key": "full", "label": "Full ESTGNN", "settings": {"temporal": "lstm", "graph": "physics"}},
    {"key": "without_lstm", "label": "w/o LSTM",
     "settings": {"temporal": "last_cycle", "graph": "physics"}},
    {"key": "without_physics_prior", "label": "w/o Physics prior",
     "settings": {"temporal": "lstm", "graph": "complete"}},
    {"key": "without_gcn", "label": "w/o GCN", "settings": None,
     "reason": "Removing the graph layers is not buildable here: they are the part "
               "this playground holds fixed, and a model without them is not built yet."},
)

# The paper's training procedure, and this project's default one. F-16 is why
# the difference is offered as a single switch: it moves what the LSTM is worth.
PROCEDURES = (
    {"key": "paper", "label": "The paper's procedure",
     "text": "Nothing held back, all 200 passes, the final weights (Table 6).",
     "settings": {"validation_fraction": 0.0, "early_stopping": False, "epochs": 200}},
    {"key": "project", "label": "This project's default",
     "text": "A fifth of the engines held back; the best pass's weights are restored "
             "(A-06, A-18). The paper never describes this step.",
     "settings": {"validation_fraction": 0.2, "early_stopping": True, "epochs": 200}},
)


# -- settings ---------------------------------------------------------------


def _number(spec: Mapping[str, Any], value: Any, integer: bool) -> float | int:
    """One numeric setting, converted and checked against its range."""
    if isinstance(value, bool):
        raise SettingsError(f"{spec['label']} must be a number, got {value!r}.")
    try:
        number = int(value) if integer and float(value) == int(float(value)) else float(value)
    except (TypeError, ValueError):
        raise SettingsError(f"{spec['label']} must be a number, got {value!r}.") from None
    if integer and not isinstance(number, int):
        raise SettingsError(f"{spec['label']} must be a whole number, got {value!r}.")
    if not math.isfinite(number) or not spec["min"] <= number <= spec["max"]:
        raise SettingsError(
            f"{spec['label']} must lie between {spec['min']} and {spec['max']}, got {value!r}."
        )
    return number


def validate_settings(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Every setting, checked and filled from its default. Unknown keys are refused."""
    if not isinstance(payload, Mapping):
        raise SettingsError("The settings must be a JSON object.")
    unknown = set(payload) - set(_BY_KEY)
    if unknown:
        raise SettingsError(f"Unknown setting(s): {sorted(unknown)}.")

    settings: dict[str, Any] = {}
    for spec in FIELDS:
        value = payload.get(spec["key"], spec["default"])
        kind = spec["kind"]
        if kind == "choice":
            allowed = [choice["value"] for choice in spec["choices"]]
            if value not in allowed:
                raise SettingsError(f"{spec['label']} must be one of {allowed}, got {value!r}.")
        elif kind == "bool":
            if not isinstance(value, bool):
                raise SettingsError(f"{spec['label']} must be true or false, got {value!r}.")
        elif kind == "int":
            value = _number(spec, value, integer=True)
        elif kind == "float":
            value = float(_number(spec, value, integer=False))
        elif kind == "widths":
            if not isinstance(value, list | tuple) or len(value) != spec["length"]:
                raise SettingsError(
                    f"{spec['label']} must be {spec['length']} widths: the graph layers "
                    "are held at Table 4's three."
                )
            value = [_number(spec, width, integer=True) for width in value]
        settings[spec["key"]] = value
    return settings


def validate_explain(payload: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """The request to explain one stored run, checked."""
    if not isinstance(payload, Mapping):
        raise SettingsError("The request must be a JSON object.")
    run = _run_directory(root, str(payload.get("run", "")))
    method = payload.get("method", "shap")
    if method not in ("shap", "occlusion"):
        raise SettingsError(f"method must be 'shap' or 'occlusion', got {method!r}.")
    scope = payload.get("scope", "engine")
    if scope not in ("engine", "test"):
        raise SettingsError(f"scope must be 'engine' or 'test', got {scope!r}.")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SettingsError(f"seed must be a whole number, got {seed!r}.")
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
    return {"run": run.name, "method": method, "scope": scope, "seed": seed,
            "engine": engine, "every": every}


def paper_row_for(
    subset: str, temporal: str | None, graph: str | None
) -> tuple[dict[str, Any] | None, str]:
    """The row of the paper a configuration may be set beside, or why there is none."""
    if graph == "correlation":
        return None, "The paper never builds a graph from correlations, so no row describes one."
    if temporal == "lstm" and graph == "physics":
        key = "full"
    elif temporal == "last_cycle" and graph == "physics":
        key = "without_lstm"
    elif temporal == "lstm" and graph == "complete":
        key = "without_physics_prior"
    elif temporal is None:
        return None, "The run did not record its temporal stage, so no row can be chosen."
    elif temporal == "last_cycle" and graph == "complete":
        return None, "Two parts are removed at once; Table 11 removes one at a time."
    else:
        return None, "The paper has no row for this temporal stage."
    row = published_row(key, subset)
    return {"key": row.key, "label": row.label, "source": row.source, "rmse": row.rmse,
            "score": row.score, "r2": row.r2, "asue": row.asue}, ""


def caveats(settings: Mapping[str, Any]) -> list[dict[str, str]]:
    """What a reader must be told before setting this configuration beside the paper."""
    notes: list[dict[str, str]] = []

    def add(level: str, ref: str, text: str) -> None:
        notes.append({"level": level, "ref": ref, "text": text})

    seeds = int(settings.get("seeds") or 0)
    if seeds < RESULT_SEEDS:
        add("warn", "rule 3", f"{seeds} seed(s) is exploratory. Quote nothing from it: "
            "a result is a mean over at least five seeds, with its spread.")
    if settings.get("temporal") == "lstm":
        add("info", "A-26", "The LSTM's hidden size and projection are chosen here; "
            "the paper states neither.")
    if settings.get("temporal") == "last_cycle":
        add("info", "D-6", "Removing the stage this way also discards 29 of the 30 "
            "cycles, which the paper's \"w/o LSTM\" model may not have done.")
    if settings.get("temporal") in ("gru", "mean_window", "raw_window"):
        add("info", "", "This temporal stage is not in the paper, so no published row applies.")
    if settings.get("shared_across_sensors") and settings.get("temporal") in ("lstm", "gru"):
        add("info", "Section 2.2", "One shared network is not the paper's stage, which "
            "gives each sensor its own.")
    if list(settings.get("gcn_hidden") or [32, 64, 32]) != [32, 64, 32] or \
            int(settings.get("fc_hidden") or 128) != 128:
        add("info", "Table 4", "The layer widths differ from Table 4.")
    if float(settings.get("validation_fraction") or 0.0) > 0:
        add("warn", "F-16", "Held-back engines mean the best pass's weights are restored, "
            "a step the paper never describes. On FD001 it moved what the LSTM is worth "
            "from 0.77 to 3.19 cycles.")
    elif settings.get("lr_schedule") == "plateau":
        add("warn", "A-17", "The plateau schedule needs held-back engines; with none the "
            "rate never leaves its first value.")
    if settings.get("clip_test_labels", True):
        add("info", "A-02", "Test labels are capped too, which can only flatter the numbers.")
    if settings.get("graph") == "complete":
        add("info", "D-2", "Row-rescaled, the complete graph averages every sensor equally, "
            "so it measures the cost of averaging as much as the value of the physics.")
    if settings.get("graph") == "correlation":
        add("info", "A-27", "The graph is rebuilt from each seed's own training engines, "
            "so two seeds can be given slightly different graphs.")
    if settings.get("subset") in ("FD002", "FD004"):
        add("warn", "A-23", "Test engines shorter than the window are left out, so the "
            "score, which is a sum over engines, cannot be set beside Table 7.")
    if settings.get("fixed_split"):
        add("info", "rule 3", "The same held-back engines for every seed understates the spread.")
    return notes


@lru_cache(maxsize=8)
def _test_lengths(subset: str) -> tuple[int, ...]:
    """How many cycles each test engine of a sub-dataset recorded, from the real file."""
    from ..data.loading import read_cmapss_frame
    from ..data.sensors import CYCLE_COL, UNIT_COL
    from ..utils.paths import raw_data_dir

    path = raw_data_dir() / f"test_{subset}.txt"
    if not path.is_file():
        return ()
    frame = read_cmapss_frame(path)
    return tuple(int(n) for n in frame.groupby(UNIT_COL)[CYCLE_COL].max().tolist())


def dataset_facts(subset: str, window_size: int) -> dict[str, Any]:
    """What Table 5 says about a sub-dataset, and what the real files add for this window."""
    facts: dict[str, Any] = {"table5": DATASET_DESCRIPTION[subset], "counted": None}
    lengths = _test_lengths(subset)
    if lengths:
        facts["counted"] = {
            "test_engines": len(lengths),
            "shortest_test_history": min(lengths),
            "left_out": sum(1 for n in lengths if n < window_size),
        }
    return facts


def preview(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the page shows before a run: the network built for real, and the caveats."""
    row, reason = paper_row_for(settings["subset"], settings["temporal"], settings["graph"])
    answer: dict[str, Any] = {
        "paper_row": row,
        "no_row_reason": reason,
        "caveats": caveats(settings),
        "dataset": dataset_facts(settings["subset"], int(settings["window_size"])),
        "command": " ".join(train_arguments(settings, Path("<job>/preprocessing_config.yaml"),
                                            None, display=True)),
    }
    try:
        from ..graph.adjacency import (
            build_fully_connected_adjacency,
            build_physics_adjacency,
            normalize_adjacency,
        )
        from ..models import ModelConfig, build_model
    except ImportError:
        answer["network_error"] = ("Building the network needs PyTorch, which is not "
                                   "installed. Nothing can be trained without it either.")
        return answer

    # The correlation graph needs the training engines, which a preview does not
    # load. The layers and their parameter counts do not depend on which edges
    # the graph has, so the paper's graph stands in for it here.
    matrix, symbols = build_physics_adjacency()
    if settings["graph"] == "complete":
        matrix, _ = build_fully_connected_adjacency(len(symbols))
    try:
        model = build_model(
            normalize_adjacency(matrix, method=settings["adjacency_normalization"]),
            _model_config(settings, ModelConfig),
            window_size=int(settings["window_size"]),
            rul_max=int(settings["rul_max"]),
        )
    except ValueError as refused:
        answer["network_error"] = str(refused)
        return answer
    answer["layers"] = model.layer_summary()
    answer["parameters"] = model.count_parameters()
    return answer


def _model_config(settings: Mapping[str, Any], factory: Any) -> Any:
    """The ModelConfig the settings describe."""
    return factory(
        temporal=settings["temporal"],
        temporal_hidden=int(settings["temporal_hidden"]),
        temporal_out_features=int(settings["temporal_out_features"]),
        shared_across_sensors=bool(settings["shared_across_sensors"]),
        gcn_hidden=tuple(settings["gcn_hidden"]),
        fc_hidden=int(settings["fc_hidden"]),
        gcn_bias=bool(settings["gcn_bias"]),
    )


def preprocessing_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The part of the settings that becomes preprocessing_config.yaml."""
    return {
        "subset": settings["subset"],
        "window_size": int(settings["window_size"]),
        "window_stride": int(settings["window_stride"]),
        "rul_max": int(settings["rul_max"]),
        "clip_test_labels": bool(settings["clip_test_labels"]),
        "normalization": settings["normalization"],
        "smoothing_alpha": float(settings["smoothing_alpha"]) if settings["smoothing"] else None,
        "graph": settings["graph"],
        "adjacency_normalization": settings["adjacency_normalization"],
        "validation_fraction": float(settings["validation_fraction"]),
        "seed": int(settings["seed"]),
        "notes": "Written by the playground of the documentation website.",
    }


def train_arguments(
    settings: Mapping[str, Any],
    config_path: Path,
    progress_path: Path | None,
    *,
    display: bool = False,
) -> list[str]:
    """The command that trains these settings: a list for the process, or words to show."""
    program = ["python"] if display else [sys.executable, "-u"]
    arguments = [
        *program, "scripts/train.py",
        "--config", str(config_path).replace("\\", "/") if display else str(config_path),
        "--seeds", str(settings["seeds"]),
        "--temporal", settings["temporal"],
    ]
    if settings["temporal"] in ("lstm", "gru"):
        arguments += ["--temporal-hidden", str(settings["temporal_hidden"]),
                      "--temporal-out-features", str(settings["temporal_out_features"])]
        if settings["shared_across_sensors"]:
            arguments.append("--shared-across-sensors")
    arguments += ["--gcn-hidden", *(str(w) for w in settings["gcn_hidden"]),
                  "--fc-hidden", str(settings["fc_hidden"])]
    if settings["gcn_bias"]:
        arguments.append("--gcn-bias")
    arguments += [
        "--epochs", str(settings["epochs"]),
        "--batch-size", str(settings["batch_size"]),
        "--learning-rate", repr(float(settings["learning_rate"])),
        "--final-learning-rate", repr(float(settings["final_learning_rate"])),
        "--lr-schedule", settings["lr_schedule"],
    ]
    if settings["early_stopping"]:
        arguments += ["--early-stopping-patience", str(settings["early_stopping_patience"])]
    else:
        arguments.append("--no-early-stopping")
    if settings["fixed_split"]:
        arguments.append("--fixed-split")
    arguments += ["--device", settings["device"]]
    if progress_path is not None:
        arguments += ["--progress", str(progress_path)]
    return arguments


def explain_arguments(request: Mapping[str, Any], progress_path: Path | None,
                      *, display: bool = False) -> list[str]:
    """The command that explains a stored run."""
    program = ["python"] if display else [sys.executable, "-u"]
    arguments = [*program, "scripts/explain.py", "--run", request["run"],
                 "--seed", str(request["seed"]), "--method", request["method"],
                 "--scope", request["scope"]]
    if request["scope"] == "engine":
        if request.get("engine") is not None:
            arguments += ["--engine", str(request["engine"])]
        if int(request.get("every") or 1) > 1:
            arguments += ["--every", str(request["every"])]
    if progress_path is not None:
        arguments += ["--progress", str(progress_path)]
    return arguments


def options() -> dict[str, Any]:
    """Everything the page needs to draw its form, before anything is chosen."""
    from .figures import _graph_layout

    graph, positions, degrees = _graph_layout()
    return {
        "fields": list(FIELDS),
        "groups": [{"key": k, "label": label} for k, label in GROUPS],
        "ablations": list(ABLATIONS),
        "procedures": list(PROCEDURES),
        "defaults": {spec["key"]: spec["default"] for spec in FIELDS},
        "result_seeds": RESULT_SEEDS,
        "dataset": DATASET_DESCRIPTION,
        "paper": {
            subset: {key: _row_dict(key, subset) for key in
                     ("full", "without_lstm", "without_physics_prior", "without_gcn")}
            for subset in SUBSETS
        },
        "graph": {
            "positions": {node: [round(x, 1), round(y, 1)] for node, (x, y) in positions.items()},
            "edges": [list(edge) for edge in graph.edges()],
            "degrees": degrees,
            "width": 860,
            "height": 520,
        },
    }


def _row_dict(key: str, subset: str) -> dict[str, Any]:
    row = published_row(key, subset)
    return {"key": row.key, "label": row.label, "source": row.source, "rmse": row.rmse,
            "score": row.score, "r2": row.r2, "asue": row.asue}


# -- stored runs ------------------------------------------------------------


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
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise SettingsError(f"{name!r} is not the name of a run.")
    path = root / "reports" / name
    if not (path / "summary.json").is_file():
        raise SettingsError(f"There is no finished run called {name!r} in reports/.")
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """A CSV file as a list of rows, with missing values as None."""
    import pandas as pd

    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    return _clean(frame.astype(object).where(frame.notna(), None).to_dict(orient="records"))


def _describe(path: Path) -> dict[str, Any] | None:
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
    subset = str(summary.get("subset", config.get("subset", "")))
    validation_fraction = config.get("validation_fraction", summary.get("validation_fraction"))
    row, reason = (paper_row_for(subset, temporal, summary.get("graph"))
                   if subset in SUBSETS else (None, "Unknown sub-dataset."))
    seeds = int(summary.get("seeds", 0))
    commit = str(summary.get("commit", "unknown"))
    settings_for_caveats = {
        "seeds": seeds, "temporal": temporal, "subset": subset,
        "graph": summary.get("graph"), "clip_test_labels": summary.get("clip_test_labels"),
        "validation_fraction": validation_fraction or 0.0,
        "fixed_split": summary.get("split") == "fixed",
        "gcn_hidden": (model or {}).get("gcn_hidden"),
        "fc_hidden": (model or {}).get("fc_hidden"),
        "shared_across_sensors": (model or {}).get("shared_across_sensors"),
        "lr_schedule": summary.get("lr_schedule"),
    }
    return _clean({
        "name": path.name,
        "commit": commit,
        "dirty": commit.endswith("-dirty"),
        "finished": str(summary.get("finished_utc", "")),
        "subset": subset,
        "seeds": seeds,
        "exploratory": seeds < RESULT_SEEDS,
        "temporal": temporal,
        "temporal_hidden": summary.get("temporal_hidden"),
        "graph": summary.get("graph"),
        "adjacency_normalization": summary.get("adjacency_normalization"),
        "split": summary.get("split"),
        "normalization": config.get("normalization"),
        "smoothing_alpha": config.get("smoothing_alpha"),
        "window_size": config.get("window_size"),
        "rul_max": config.get("rul_max"),
        "clip_test_labels": summary.get("clip_test_labels"),
        "validation_fraction": validation_fraction,
        # What train() does, read from the settings rather than trusted from a
        # field: with nothing held back it can only keep the last pass.
        "reported_weights": ("final_epoch" if not validation_fraction
                             else "best_validation_epoch"),
        "lr_schedule": summary.get("lr_schedule"),
        "model": model,
        "training": summary.get("training"),
        "has_weights": any(path.glob("seed_*/model.pt")),
        "mean": summary.get("mean"),
        "std": summary.get("std"),
        "paper_row": row,
        "no_row_reason": reason,
        "caveats": caveats(settings_for_caveats),
    })


def list_runs(root: Path) -> list[dict[str, Any]]:
    """Every finished run in reports/, newest first."""
    reports = root / "reports"
    if not reports.is_dir():
        return []
    runs = [described for path in sorted(reports.iterdir())
            if path.is_dir() and not path.name.startswith(".")
            and (described := _describe(path)) is not None]
    runs.sort(key=lambda run: run["finished"], reverse=True)
    return runs


def run_detail(root: Path, name: str) -> dict[str, Any]:
    """One run in full: every seed's numbers, curves, predictions and explanations."""
    path = _run_directory(root, name)
    described = _describe(path)
    if described is None:
        raise SettingsError(f"{name!r} is not a training run.")
    seeds = {}
    for folder in sorted(path.glob("seed_*")):
        seeds[folder.name.removeprefix("seed_")] = {
            "history": _read_csv(folder / "training_history.csv"),
            "predictions": _read_csv(folder / "test_predictions.csv"),
            "has_weights": (folder / "model.pt").is_file(),
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
        "per_seed": seeds,
        "explanations": explanations,
    })


def explanation_detail(root: Path, run: str, name: str) -> dict[str, Any]:
    """One stored explanation: its summary and every value it computed."""
    path = _run_directory(root, run) / "explanations"
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise SettingsError(f"{name!r} is not the name of an explanation.")
    folder = path / name
    if not (folder / "summary.json").is_file():
        raise SettingsError(f"Run {run!r} has no explanation called {name!r}.")
    return _clean({
        "summary": json.loads((folder / "summary.json").read_text(encoding="utf-8")),
        "values": _read_csv(folder / "values.csv"),
    })


_COMPARED = ("rmse", "score", "r2", "asue")


def _flat_settings(detail: Mapping[str, Any]) -> dict[str, Any]:
    """Every setting of a run on one level, for spotting what two runs change."""
    flat: dict[str, Any] = {}
    for key, value in (detail.get("preprocessing") or {}).items():
        if key not in ("notes", "seed"):
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
        pairs = [(by_seed_a[s][measure], by_seed_b[s][measure]) for s in shared
                 if by_seed_a[s].get(measure) is not None
                 and by_seed_b[s].get(measure) is not None]
        entry: dict[str, Any] = {
            "per_seed": [{"seed": s, "first": x, "second": y, "difference": y - x}
                         for s, (x, y) in zip(shared, pairs, strict=False)],
            "mean_difference": None, "t": None, "p": None,
        }
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


# -- jobs -------------------------------------------------------------------


@dataclass
class Job:
    """One command the playground started, or will start when the one before it ends."""

    id: str
    kind: str
    label: str
    arguments: list[str]
    display: str
    folder: Path
    created: float = field(default_factory=time.time)
    status: str = "queued"
    started: float | None = None
    finished: float | None = None
    returncode: int | None = None
    process: subprocess.Popen[bytes] | None = field(default=None, repr=False)

    def progress(self) -> dict[str, Any]:
        """What the job's progress file says so far, summarised for the page."""
        path = self.folder / "progress.jsonl"
        if not path.is_file():
            return {}
        events = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # the last line may be half-written
        summary: dict[str, Any] = {"seeds_done": [], "curve": []}
        for event in events:
            kind = event.get("event")
            if kind == "start":
                summary.update({k: v for k, v in event.items() if k != "event"})
            elif kind == "seed":
                summary["seed"], summary["seed_index"], summary["curve"] = (
                    event.get("seed"), event.get("index"), [])
            elif kind == "epoch":
                summary["epoch"] = event.get("epoch")
                summary["curve"].append({
                    "epoch": event.get("epoch"),
                    "train_loss": event.get("train_loss"),
                    "validation_rmse": event.get("validation_rmse_cycles"),
                })
            elif kind == "seed_done":
                summary["seeds_done"].append({k: v for k, v in event.items() if k != "event"})
            elif kind == "window":
                summary.update(done=event.get("done"), total=event.get("total"),
                               seconds=event.get("seconds"))
            elif kind == "done":
                summary["result"] = {k: v for k, v in event.items() if k != "event"}
        return summary

    def log_tail(self, lines: int = 25) -> str:
        """The last lines the command printed, which is where a failure explains itself."""
        path = self.folder / "output.txt"
        if not path.is_file():
            return ""
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

    def to_dict(self) -> dict[str, Any]:
        """The job as the page shows it."""
        now = time.time()
        return _clean({
            "id": self.id, "kind": self.kind, "label": self.label, "status": self.status,
            "command": self.display, "created": self.created, "started": self.started,
            "finished": self.finished, "returncode": self.returncode,
            "elapsed": (self.finished or now) - self.started if self.started else None,
            "progress": self.progress(),
            "log": self.log_tail() if self.status in ("failed", "cancelled") else "",
        })


class JobManager:
    """Runs the playground's commands one at a time, in the order they were asked for."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.jobs: dict[str, Job] = {}
        self.queue: list[str] = []
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stopping = False
        self.worker: threading.Thread | None = None

    def _ensure_worker(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(target=self._work, name="playground-jobs", daemon=True)
            self.worker.start()

    def _new_folder(self, kind: str) -> tuple[str, Path]:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        identifier = f"{stamp}_{kind}_{secrets.token_hex(3)}"
        folder = self.root / JOBS_FOLDER / identifier
        folder.mkdir(parents=True, exist_ok=True)
        return identifier, folder

    def submit_training(self, settings: Mapping[str, Any], label: str) -> Job:
        """Queue one training run of these settings."""
        identifier, folder = self._new_folder("train")
        config_path = folder / "preprocessing_config.yaml"
        config_path.write_text(
            yaml.safe_dump(preprocessing_settings(settings), sort_keys=False), encoding="utf-8"
        )
        (folder / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        relative = config_path.relative_to(self.root)
        job = Job(
            id=identifier, kind="train", label=label, folder=folder,
            arguments=train_arguments(settings, config_path, folder / "progress.jsonl"),
            display=" ".join(train_arguments(settings, relative, None, display=True)),
        )
        return self._enqueue(job)

    def submit_explanation(self, request: Mapping[str, Any], label: str) -> Job:
        """Queue one explanation of a stored run."""
        identifier, folder = self._new_folder("explain")
        job = Job(
            id=identifier, kind="explain", label=label, folder=folder,
            arguments=explain_arguments(request, folder / "progress.jsonl"),
            display=" ".join(explain_arguments(request, None, display=True)),
        )
        return self._enqueue(job)

    def submit_command(self, kind: str, label: str, arguments: list[str]) -> Job:
        """Queue any command. Only the tests use this directly."""
        identifier, folder = self._new_folder(kind)
        job = Job(id=identifier, kind=kind, label=label, folder=folder,
                  arguments=list(arguments), display=" ".join(arguments))
        return self._enqueue(job)

    def _enqueue(self, job: Job) -> Job:
        with self.lock:
            self.jobs[job.id] = job
            self.queue.append(job.id)
        self._ensure_worker()
        self.wake.set()
        return job

    def cancel(self, identifier: str) -> Job:
        """Remove a queued job, or stop a running one. A stopped run writes nothing."""
        with self.lock:
            job = self.jobs.get(identifier)
            if job is None:
                raise SettingsError(f"There is no job called {identifier!r}.")
            if job.status == "queued":
                self.queue.remove(identifier)
                job.status, job.finished = "cancelled", time.time()
            elif job.status == "running":
                job.status = "cancelling"
                # A job is marked running a moment before its process exists.
                # Asked to stop in that moment, there is nothing to end yet, so
                # _run sees "cancelling" and never starts it.
                if job.process is not None:
                    job.process.terminate()
        return job

    def snapshot(self) -> list[dict[str, Any]]:
        """Every job this server has been asked for, newest first."""
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda job: job.created, reverse=True)
        return [job.to_dict() for job in jobs]

    def get(self, identifier: str) -> Job | None:
        with self.lock:
            return self.jobs.get(identifier)

    def stop(self) -> None:
        """Stop the running job and forget the queue. Called when the server stops."""
        self.stopping = True
        with self.lock:
            self.queue.clear()
            running = [job for job in self.jobs.values() if job.process is not None
                       and job.process.poll() is None]
        for job in running:
            job.process.terminate()  # type: ignore[union-attr]
        self.wake.set()

    def _work(self) -> None:
        while not self.stopping:
            with self.lock:
                identifier = self.queue.pop(0) if self.queue else None
                job = self.jobs.get(identifier) if identifier else None
                if job is not None:
                    job.status, job.started = "running", time.time()
            if job is None:
                self.wake.clear()
                self.wake.wait(timeout=1.0)
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        logger.info("Playground job %s: %s", job.id, job.display)
        with (job.folder / "output.txt").open("wb") as output:
            try:
                # Started and published under the lock, so a stop asked for
                # before this point is seen here, and one asked for after it
                # finds the process to end.
                with self.lock:
                    if job.status == "cancelling":
                        job.returncode = None
                    else:
                        job.process = subprocess.Popen(
                            job.arguments, cwd=self.root, stdout=output,
                            stderr=subprocess.STDOUT,
                            env={**_environment(), "PYTHONIOENCODING": "utf-8"},
                        )
                if job.process is not None:
                    job.returncode = job.process.wait()
            except OSError as failure:
                output.write(f"Could not start the command: {failure}\n".encode())
                job.returncode = -1
        with self.lock:
            job.finished = time.time()
            if job.status == "cancelling":
                job.status = "cancelled"
            else:
                job.status = "done" if job.returncode == 0 else "failed"
            job.process = None


def _environment() -> dict[str, str]:
    import os

    return dict(os.environ)
