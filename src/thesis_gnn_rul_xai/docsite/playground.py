"""The playground behind the website: checks settings, runs the real scripts, and reads what they stored.

Read README.md in this folder. It explains why every run started here goes
through scripts/train.py and lands in reports/, and what the page shows.
"""

from __future__ import annotations

import calendar
import contextlib
import json
import math
import os
import secrets
import shutil
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
    FIG_10_RING,
    SHAP_TABLE_12,
    SHAP_TABLE_12_GRAPHS,
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
    "explanation_label",
    "job_label",
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


# Every setting the page offers, in the order of the pipeline. Each fact about
# where a default comes from is said once: on the row when every option shares
# it, on each option when the options differ, and nowhere when nothing sets it.
# An empty text leaves the tag alone, where the value in the control says it.
FIELDS: tuple[dict[str, Any], ...] = (
    # -- data: drawn at the top of the form, with no section of its own ----
    {"key": "subset", "group": "data", "label": "Sub-dataset", "kind": "choice",
     "default": "FD001",
     "choices": [{"value": s, "label": s, "source": None} for s in SUBSETS],
     "source": _paper("Table 5", "")},
    # -- procedure: set by the choice of procedure at the top of Training ---
    {"key": "validation_fraction", "group": "procedure",
     "label": "Training engines held back",
     "kind": "float", "default": 0.2, "min": 0.0, "max": 0.5, "step": 0.05,
     "source": _choice("A-06", "")},
    {"key": "early_stopping", "group": "procedure", "label": "Stop when held-back error stalls",
     "kind": "bool", "default": True,
     "source": _choice("A-18", "")},
    {"key": "early_stopping_patience", "group": "procedure",
     "label": "Passes allowed without improvement", "kind": "int",
     "default": 20, "min": 1, "max": 1000,
     "source": _choice("A-18", "")},
    # -- preprocessing -----------------------------------------------------
    {"key": "window_size", "group": "preprocessing", "label": "Window length (cycles)",
     "kind": "int", "default": 30, "min": 1, "max": 100,
     "source": _paper("Table 6", "")},
    {"key": "window_stride", "group": "preprocessing", "label": "Cycles between windows",
     "kind": "int", "default": 1, "min": 1, "max": 30,
     "source": _choice("A-05", "")},
    {"key": "rul_max", "group": "preprocessing", "label": "Label ceiling (cycles)",
     "kind": "int", "default": 125, "min": 10, "max": 400,
     "source": _choice("A-01", "read off Fig. 7 by calibrating its plateau against its own axis")},
    {"key": "clip_test_labels", "group": "preprocessing", "label": "Cap the test labels too",
     "kind": "bool", "default": True,
     "source": _choice("A-02", "can only move reported numbers favourably")},
    {"key": "normalization", "group": "preprocessing", "label": "Sensor scaling",
     "kind": "choice", "default": "zscore",
     "choices": [
         {"value": "zscore", "label": "Mean and spread of the training engines", "source": None},
         {"value": "minmax_per_condition", "label": "Range 0 to 1, per operating condition",
          "source": None},
         {"value": "minmax_global", "label": "Range 0 to 1, over all rows", "source": None},
     ],
     "source": _choice("A-03", "the paper does not describe its scaling (F-10)")},
    {"key": "smoothing", "group": "preprocessing", "label": "Smooth each sensor",
     "kind": "bool", "default": True,
     "source": _choice("A-04", "exponential smoothing before windowing")},
    {"key": "smoothing_alpha", "group": "preprocessing", "label": "Smoothing weight on the newest reading",
     "kind": "float", "default": 0.5, "min": 0.01, "max": 1.0, "step": 0.05,
     "source": _choice("A-04", "")},
    # -- temporal stage ----------------------------------------------------
    {"key": "temporal", "group": "temporal", "label": "Temporal stage", "kind": "choice",
     "default": "lstm",
     "choices": [
         {"value": "lstm", "label": "LSTM, one per sensor",
          "source": _paper("Section 2.2", "")},
         {"value": "last_cycle", "label": "w/o LSTM: keep only the last cycle",
          "source": _choice("A-07", "")},
         {"value": "gru", "label": "GRU, one per sensor",
          "source": _extension("a smaller recurrent cell")},
         {"value": "mean_window", "label": "Average over the window",
          "source": _extension("no learned parameters")},
         {"value": "window_summary", "label": "Last reading, average and change over the window",
          "source": _extension("three numbers per sensor, no learned parameters (A-28)")},
         {"value": "raw_window", "label": "The whole window straight into the graph layer",
          "source": _extension("the arrangement the last-cycle stage replaced")},
     ],
     "source": None},
    {"key": "temporal_hidden", "group": "temporal", "label": "Hidden size", "kind": "int",
     "default": 32, "min": 1, "max": 512,
     "source": _choice("A-26", "")},
    {"key": "temporal_out_features", "group": "temporal",
     "label": "Numbers handed to the graph per sensor", "kind": "int",
     "default": 1, "min": 1, "max": 64,
     "source": _choice("A-26", "1 is the node width Table 4 prints")},
    {"key": "shared_across_sensors", "group": "temporal",
     "label": "One network shared by all sensors", "kind": "bool", "default": False,
     "source": _extension("")},
    # -- graph layers: the graph they run on, then their sizes -------------
    # The sensor graph is built while the data is prepared, but it is drawn
    # with the layers that use it, where the page's pipeline shows it too.
    {"key": "graph", "group": "graph", "label": "Sensor graph", "kind": "choice",
     "default": "physics",
     "choices": [
         {"value": "physics", "label": "The paper's physics graph",
          "source": _paper("Table 3", "")},
         {"value": "complete", "label": "Every sensor joined to every other",
          "source": _choice("D-2", "averages every sensor equally, so it measures the cost "
                                   "of averaging as much as the value of the physics")},
         {"value": "correlation", "label": "The 41 most correlated sensor pairs",
          "source": _choice("A-27", "built from each seed's own training engines, so seeds "
                                    "can get slightly different graphs")},
     ],
     "source": None},
    {"key": "adjacency_normalization", "group": "graph",
     "label": "Graph rescaling", "kind": "choice", "default": "row",
     "choices": [
         {"value": "row", "label": "Each row sums to one", "source": None},
         {"value": "symmetric", "label": "Symmetric, by both degrees", "source": None},
     ],
     "source": _choice("A-21", "the paper describes the step but prints no formula")},
    {"key": "gcn_hidden", "group": "graph", "label": "Graph layer widths", "kind": "widths",
     "default": [32, 64, 32], "min": 1, "max": 512, "length": 3,
     "source": _paper("Table 4", "")},
    {"key": "fc_hidden", "group": "graph", "label": "First fully connected width",
     "kind": "int", "default": 128, "min": 1, "max": 2048,
     "source": _paper("Table 4", "")},
    {"key": "gcn_bias", "group": "graph", "label": "Bias in the graph layers",
     "kind": "bool", "default": False,
     "source": _choice("A-12", "Table 4 lists only a weight matrix per layer")},
    # -- training ----------------------------------------------------------
    {"key": "epochs", "group": "training", "label": "Passes over the training data",
     "kind": "int", "default": TRAINING_SETTINGS["epochs"], "min": 1, "max": 1000,
     "source": _paper("Table 6", "")},
    {"key": "batch_size", "group": "training", "label": "Windows per update",
     "kind": "int", "default": TRAINING_SETTINGS["batch_size"], "min": 1, "max": 4096,
     "source": _paper("Table 6", "")},
    {"key": "learning_rate", "group": "training", "label": "First learning rate",
     "kind": "float", "default": TRAINING_SETTINGS["learning_rate"],
     "min": 1e-6, "max": 1.0, "step": 1e-4,
     "source": _paper("Table 6", "")},
    {"key": "final_learning_rate", "group": "training", "label": "Last learning rate",
     "kind": "float", "default": TRAINING_SETTINGS["final_learning_rate"],
     "min": 1e-7, "max": 1.0, "step": 1e-5,
     "source": _paper("Table 6", "")},
    {"key": "lr_schedule", "group": "training", "label": "How the rate gets there",
     "kind": "choice", "default": "step",
     "choices": [
         {"value": v, "label": label, "source": None}
         for v, label in (
             ("step", "One drop, halfway"),
             ("exponential", "The same factor every pass"),
             ("cosine", "A half cosine"),
             ("plateau", "Drop when held-back error stalls"),
             ("constant", "Never changes"),
         )
     ],
     "source": _choice("A-17", "")},
    # -- repeats -----------------------------------------------------------
    {"key": "seed", "group": "repeats", "label": "First seed", "kind": "int",
     "default": 42, "min": 0, "max": 2**31 - 1,
     "source": None},
    {"key": "seeds", "group": "repeats", "label": "Seeds", "kind": "int",
     "default": RESULT_SEEDS, "min": 1, "max": 20,
     "source": _project("")},
    {"key": "fixed_split", "group": "repeats", "label": "Same held-back engines for every seed",
     "kind": "bool", "default": False,
     "source": _project("understates the spread; say so wherever the number appears")},
    {"key": "device", "group": "repeats", "label": "Device", "kind": "choice",
     "default": "auto",
     "choices": [{"value": v, "label": v, "source": None} for v in ("auto", "cpu", "cuda")],
     "source": None},
)

_BY_KEY = {spec["key"]: spec for spec in FIELDS}

# The sections of the form, numbered on the page in this order. The "procedure"
# group is left out on purpose: its settings are drawn at the top of Training,
# under the choice of procedure, which is their only control.
GROUPS = (
    ("data", "Data"),
    ("preprocessing", "Preprocessing"),
    ("temporal", "Temporal stage"),
    ("graph", "Graph layers"),
    ("training", "Training"),
    ("repeats", "Repeats"),
)

# The four model variants of Table 11, as settings: a stored run is set beside
# the row whose settings it has. Taking the LSTM out of the page's pipeline
# gives "w/o LSTM", taking the physics prior out gives "w/o Physics prior".
# "w/o GCN" has no settings: the graph layers are the one part this playground
# keeps, so their box has nothing to take them out with.
ABLATIONS = (
    {"key": "full", "settings": {"temporal": "lstm", "graph": "physics"}},
    {"key": "without_lstm", "settings": {"temporal": "last_cycle", "graph": "physics"}},
    {"key": "without_physics_prior", "settings": {"temporal": "lstm", "graph": "complete"}},
    {"key": "without_gcn", "settings": None},
)

# The two training procedures, and the only control for how many engines are
# held back and whether the best pass's weights are restored. They are one
# choice because it moves what the LSTM is worth (F-16), and the first one in
# the Training group. Neither sets the number of passes, which is the next one.
PROCEDURES = (
    {"key": "paper", "label": "The paper's procedure",
     "text": "Nothing held back, every pass, the final weights (Table 6).",
     "settings": {"validation_fraction": 0.0, "early_stopping": False}},
    {"key": "project", "label": "Held back, best pass",
     "text": "Engines held back and the best pass's weights restored, a step the paper "
             "never describes (A-06, A-18). On FD001 it moved what the LSTM is worth "
             "from 0.77 to 3.19 cycles (F-16).",
     "settings": {"validation_fraction": 0.2, "early_stopping": True}},
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
    for variant in ABLATIONS:
        if variant["settings"] == {"temporal": temporal, "graph": graph}:
            row = published_row(variant["key"], subset)
            return {"key": row.key, "label": row.label, "source": row.source, "rmse": row.rmse,
                    "score": row.score, "r2": row.r2, "asue": row.asue}, ""
    if temporal is None:
        return None, "The run did not record its temporal stage, so no row can be chosen."
    if temporal == "last_cycle" and graph == "complete":
        return None, "Two parts are removed at once; Table 11 removes one at a time."
    return None, "The paper has no row for this temporal stage."


def caveats(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What a reader must be told before setting this configuration beside the paper."""
    notes: list[dict[str, Any]] = []

    # "run": it qualifies a stored run's numbers, so the run's page lists it.
    # "design": it flags a conflict between settings the reader may not have
    # open, so the Design tab shows it beside the Train button as well.
    def add(level: str, ref: str, text: str, where: tuple[str, ...] = ("run",)) -> None:
        notes.append({"level": level, "ref": ref, "text": text, "where": list(where)})

    seeds = int(settings.get("seeds") or 0)
    if seeds < RESULT_SEEDS:
        add("warn", "", f"{seeds} seed{'' if seeds == 1 else 's'} is exploratory: quote "
            "nothing from it. A result is a mean over at least five seeds, with its spread.",
            ("design", "run"))
    if settings.get("temporal") in ("lstm", "gru"):
        add("info", "A-26", "The hidden size and the numbers handed to the graph per "
            "sensor are chosen here; the paper states neither.")
    if settings.get("temporal") == "last_cycle":
        add("info", "D-6", "Only each window's last cycle reaches the graph layers, which "
            "the paper's w/o LSTM model may not have done.")
    if settings.get("shared_across_sensors") and settings.get("temporal") in ("lstm", "gru"):
        add("info", "Section 2.2", "One shared network is not the paper's stage, which "
            "gives each sensor its own.")
    if list(settings.get("gcn_hidden") or [32, 64, 32]) != [32, 64, 32] or \
            int(settings.get("fc_hidden") or 128) != 128:
        add("info", "Table 4", "The layer widths differ from Table 4.")
    if float(settings.get("validation_fraction") or 0.0) > 0:
        add("warn", "F-16", "The paper never describes restoring the best pass; on FD001 "
            "it moved what the LSTM is worth from 0.77 to 3.19 cycles.")
    elif settings.get("lr_schedule") == "plateau":
        add("warn", "A-17", "The plateau schedule needs held-back engines; with none the "
            "rate never leaves its first value.", ("design", "run"))
    if settings.get("clip_test_labels", True):
        add("info", "A-02", "Test labels are capped too, which can only flatter the numbers.")
    if settings.get("graph") == "complete":
        # Every sensor of the complete graph has the same number of neighbours,
        # so both rescalings give the same matrix, 1/14 everywhere.
        add("info", "D-2", "The complete graph averages every sensor equally, so it "
            "measures the cost of averaging as much as the value of the physics.")
    if settings.get("graph") == "correlation":
        add("info", "A-27", "The graph is rebuilt from each seed's own training engines, "
            "so two seeds can be given slightly different graphs.")
    subset, window = settings.get("subset"), settings.get("window_size")
    if subset in SUBSETS and isinstance(window, int) and not isinstance(window, bool):
        counted = dataset_facts(str(subset), window)["counted"]
        if counted and counted["left_out"] > 0:
            one = counted["left_out"] == 1
            add("warn", "A-23", f"{counted['left_out']} of {counted['test_engines']} test "
                f"engines {'is' if one else 'are'} shorter than the window and "
                f"{'was' if one else 'were'} left out, so the score cannot be set beside "
                "Table 7.")
    if settings.get("fixed_split"):
        add("info", "", "The same held-back engines for every seed understates the spread.")
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
    """What Table 5 says about a sub-dataset, and how many test engines this window leaves out."""
    facts: dict[str, Any] = {"table5": DATASET_DESCRIPTION[subset], "counted": None}
    lengths = _test_lengths(subset)
    if lengths:
        # An engine shorter than the window yields no window at all (A-23).
        facts["counted"] = {
            "test_engines": len(lengths),
            "left_out": sum(1 for n in lengths if n < window_size),
        }
    return facts


def preview(settings: Mapping[str, Any]) -> dict[str, Any]:
    """What the Design tab draws for these settings: the data, the cautions, the network built for real."""
    answer: dict[str, Any] = {
        "dataset": dataset_facts(settings["subset"], int(settings["window_size"])),
        "design_cautions": [note for note in caveats(settings) if "design" in note["where"]],
        "torch": True,
    }
    try:
        from ..graph.adjacency import (
            build_fully_connected_adjacency,
            build_physics_adjacency,
            normalize_adjacency,
        )
        from ..models import ModelConfig, build_model
    except ImportError as missing:
        # Only a missing PyTorch is expected here; any other failed import is a fault.
        if not (missing.name or "").startswith("torch"):
            raise
        answer["torch"] = False
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

    graph, positions = _graph_layout()[:2]
    return {
        "fields": list(FIELDS),
        "groups": [{"key": k, "label": label} for k, label in GROUPS],
        "procedures": list(PROCEDURES),
        "defaults": {spec["key"]: spec["default"] for spec in FIELDS},
        "result_seeds": RESULT_SEEDS,
        "dataset": DATASET_DESCRIPTION,
        "graph": {
            "positions": {node: [round(x, 1), round(y, 1)] for node, (x, y) in positions.items()},
            "edges": [list(edge) for edge in graph.edges()],
            "width": 860,
            "height": 520,
        },
        "table12": {
            "graphs": list(SHAP_TABLE_12_GRAPHS),
            "values": {sensor: list(values) for sensor, values in SHAP_TABLE_12.items()},
            "ring": list(FIG_10_RING),
        },
    }


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
    reports = (root / "reports").resolve()
    path = (root / "reports" / name).resolve()
    # Looking for separators is not enough on Windows: a name carrying a
    # drive, "C:x", holds none of them, and joining it throws the base away.
    if not name or name.startswith(".") or path.parent != reports:
        raise SettingsError(f"{name!r} is not the name of a run.")
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
        "window_size": config.get("window_size"),
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
        "adjacency_normalization": (summary.get("adjacency_normalization")
                                    or config.get("adjacency_normalization")),
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
    path = (_run_directory(root, run) / "explanations").resolve()
    folder = (path / name).resolve()
    # The same hazard as a run name: a drive carried in the name escapes.
    if not name or name.startswith(".") or folder.parent != path:
        raise SettingsError(f"{name!r} is not the name of an explanation.")
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


# The states a job can no longer leave.
_ENDED = ("done", "failed", "cancelled", "interrupted")

# The one vocabulary the page uses for temporal stages and explanation methods.
_STAGE_NAMES = {"lstm": "LSTM", "gru": "GRU", "last_cycle": "w/o LSTM",
                "mean_window": "window average", "window_summary": "window summary",
                "raw_window": "whole window"}
_METHOD_NAMES = {"shap": "Exact Shapley values", "occlusion": "Occlusion"}
# What a job is called when its folder holds nothing to name it by.
_UNNAMED = {"train": "Training run", "explain": "Explanation"}


def job_label(settings: Mapping[str, Any]) -> str:
    """The name a training job is listed under: what it trains, on what, how many times."""
    parts = [str(settings.get("subset") or "")]
    temporal = settings.get("temporal")
    if temporal is not None:
        stage = _STAGE_NAMES.get(str(temporal), str(temporal))
        if temporal in ("lstm", "gru") and settings.get("temporal_hidden") is not None:
            stage += f" {settings['temporal_hidden']}"
        parts.append(stage)
    graph = str(settings.get("graph") or "physics")
    rescaling = settings.get("adjacency_normalization")
    parts.append(f"{graph} graph" if rescaling in (None, "row") else f"{graph} · {rescaling} graph")
    parts.append("best pass" if float(settings.get("validation_fraction") or 0.0) > 0
                 else "final pass")
    seeds = int(settings.get("seeds") or 0)
    parts.append(f"{seeds} seed{'' if seeds == 1 else 's'}")
    return " · ".join(part for part in parts if part)


def explanation_label(request: Mapping[str, Any]) -> str:
    """The name an explanation job is listed under: which method, on what, with which seed's weights."""
    method = request.get("method")
    parts = [_METHOD_NAMES.get(str(method), str(method or "Explanation"))]
    if request.get("scope") == "test":
        parts.append("test engines")
    elif request.get("scope") == "engine":
        engine = request.get("engine")
        parts.append("longest engine" if engine is None else f"engine {engine}")
    if request.get("seed") is not None:
        parts.append(f"seed {request['seed']}")
    return " · ".join(parts)


def _plain_name(name: Any) -> bool:
    """Whether a name read from a file is one plain folder name, with no way out of its folder."""
    return (isinstance(name, str) and bool(name) and not name.startswith(".")
            and ":" not in name and Path(name).name == name)


def _explain_request(folder: Path, command: str, progress: Mapping[str, Any]) -> dict[str, Any]:
    """What an explanation job asked for, from its command, and the engine it chose once it finished."""
    request: dict[str, Any] = {"method": progress.get("method"), "scope": progress.get("scope")}
    words = command.split()
    for flag, key in (("--method", "method"), ("--scope", "scope"),
                      ("--seed", "seed"), ("--engine", "engine")):
        if flag in words[:-1]:
            value = words[words.index(flag) + 1]
            request[key] = int(value) if value.isdigit() else value
    # With no engine asked for, the script picks the longest, and only its
    # stored summary says which one that was. The jobs folder sits inside
    # reports/, beside the run the explanation was written into.
    result = progress.get("result") or {}
    if _plain_name(result.get("run")) and _plain_name(result.get("directory")):
        stored = (folder.parent.parent / result["run"] / "explanations"
                  / result["directory"] / "summary.json")
        try:
            summary = json.loads(stored.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
        if isinstance(summary, dict):
            request.update({key: summary[key] for key in ("method", "scope", "seed")
                            if summary.get(key) is not None})
            if summary.get("unit") is not None:
                request["engine"] = summary["unit"]
    return request


def _job_name(folder: Path, kind: str, command: str, progress: Mapping[str, Any],
              fallback: str) -> str:
    """The name a job is listed under, built the same way for this server's jobs and for folders read back."""
    if kind == "train":
        try:
            settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            settings = None
        if isinstance(settings, dict) and settings:
            return job_label(settings)
    elif kind == "explain":
        request = _explain_request(folder, command, progress)
        if request.get("method"):
            return explanation_label(request)
    return fallback


def read_progress(folder: Path) -> dict[str, Any]:
    """What a job's progress file says so far, summarised for the page."""
    path = folder / "progress.jsonl"
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


def _process_alive(pid: Any) -> bool:
    """Whether a process with this number is still running on this machine."""
    # The number is read back from a file, so it may be anything at all. A
    # damaged one means "nothing is running", never an error raised out of
    # the listing that every job on the page is drawn from.
    try:
        number = int(pid)
    except (TypeError, ValueError):
        return False
    if number <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        # 0x1000 asks only "does this exist and how did it end", which needs no
        # right over the process; 259 is what Windows reports while it runs.
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, number)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            asked = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(asked) and code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(number, 0)
    except OSError:
        return False
    return True


def _log_tail(folder: Path, lines: int = 25) -> str:
    """The last lines a command printed, which is where a failure explains itself."""
    path = folder / "output.txt"
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


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
        return read_progress(self.folder)

    def log_tail(self, lines: int = 25) -> str:
        """The last lines the command printed, which is where a failure explains itself."""
        return _log_tail(self.folder, lines)

    def record(self) -> None:
        """Write what the page shows about this job beside its files, so a later server can list it."""
        summary = {
            "id": self.id, "kind": self.kind, "label": self.label, "command": self.display,
            "created": self.created, "status": self.status, "started": self.started,
            "finished": self.finished, "returncode": self.returncode,
            # So that a later server can tell a job still running under the
            # server that started it from one whose server died with it.
            "pid": self.process.pid if self.process is not None else None,
        }
        # If the folder was removed meanwhile, the listing is the loss, not the run.
        with contextlib.suppress(OSError):
            (self.folder / "job.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        """The job as the page shows it."""
        now = time.time()
        progress = self.progress()
        return _clean({
            "id": self.id, "kind": self.kind,
            "label": _job_name(self.folder, self.kind, self.display, progress, self.label),
            "status": self.status,
            "command": self.display, "created": self.created, "started": self.started,
            "finished": self.finished, "returncode": self.returncode,
            "elapsed": (self.finished or now) - self.started if self.started else None,
            "progress": progress,
            "log": self.log_tail() if self.status in ("failed", "cancelled") else "",
            "from_disk": False,
        })


def _stamp_to_time(stamp: str) -> float | None:
    """The moment a job folder's name records, as seconds since the epoch."""
    try:
        return float(calendar.timegm(time.strptime(stamp, "%Y%m%dT%H%M%S")))
    except ValueError:
        return None


def _describe_job_folder(path: Path) -> dict[str, Any] | None:
    """One job an earlier server left behind, read from its folder."""
    parts = path.name.split("_")
    if len(parts) < 3:
        return None
    recorded: dict[str, Any] = {}
    if (path / "job.json").is_file():
        try:
            recorded = json.loads((path / "job.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            recorded = {}
    progress = read_progress(path)
    kind = str(recorded.get("kind") or parts[1])
    status = str(recorded.get("status") or "")
    if status not in _ENDED:
        if "result" in progress:
            status = "done"
        elif _process_alive(recorded.get("pid")):
            # The server that started it is still running it. Calling this
            # ended would offer it to "remove all ended", which would delete
            # the folder the command is at that moment writing into.
            status = "running"
        else:
            # A job still running when its server stopped never wrote its end.
            status = "interrupted"
    started, finished = recorded.get("started"), recorded.get("finished")
    command = str(recorded.get("command") or "")
    return _clean({
        "id": path.name,
        "kind": kind,
        # Rebuilt rather than read from job.json, so a folder an older server
        # wrote is named in the same words as a job this server is running.
        "label": _job_name(path, kind, command, progress, str(
            recorded.get("label") or _UNNAMED.get(kind, f"{kind} job"))),
        "status": status,
        "command": command,
        "created": recorded.get("created") or _stamp_to_time(parts[0]) or path.stat().st_mtime,
        "started": started,
        "finished": finished,
        "elapsed": (finished - started) if started and finished else None,
        "progress": progress,
        "log": _log_tail(path) if status == "failed" else "",
        "from_disk": True,
    })


def job_history(root: Path) -> list[dict[str, Any]]:
    """Every job any server left in reports/.playground/, read from the folders."""
    folder = root / JOBS_FOLDER
    if not folder.is_dir():
        return []
    return [described for path in sorted(folder.iterdir())
            if path.is_dir() and (described := _describe_job_folder(path)) is not None]


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

    def submit_training(self, settings: Mapping[str, Any]) -> Job:
        """Queue one training run of these settings, under the name job_label gives it."""
        identifier, folder = self._new_folder("train")
        config_path = folder / "preprocessing_config.yaml"
        config_path.write_text(
            yaml.safe_dump(preprocessing_settings(settings), sort_keys=False), encoding="utf-8"
        )
        (folder / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        relative = config_path.relative_to(self.root)
        job = Job(
            id=identifier, kind="train", label=job_label(settings), folder=folder,
            arguments=train_arguments(settings, config_path, folder / "progress.jsonl"),
            display=" ".join(train_arguments(settings, relative, None, display=True)),
        )
        return self._enqueue(job)

    def submit_explanation(self, request: Mapping[str, Any]) -> Job:
        """Queue one explanation of a stored run, under the name explanation_label gives it."""
        identifier, folder = self._new_folder("explain")
        job = Job(
            id=identifier, kind="explain", label=explanation_label(request), folder=folder,
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
            # Written under the lock: the worker can take this job the moment
            # the lock is released, and its own record must land after this one.
            job.record()
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
                job.record()
            elif job.status == "running":
                job.status = "cancelling"
                # A job is marked running a moment before its process exists.
                # Asked to stop in that moment, there is nothing to end yet, so
                # _run sees "cancelling" and never starts it.
                if job.process is not None:
                    job.process.terminate()
        return job

    def forget(self, identifier: str = "", *, all_finished: bool = False) -> list[str]:
        """Drop finished jobs from the list and delete their folders. A running or queued job stays."""
        if all_finished:
            targets = [item["id"] for item in self.snapshot() if item["status"] in _ENDED]
        else:
            targets = [identifier]
        forgotten = []
        jobs_folder = (self.root / JOBS_FOLDER).resolve()
        for target in targets:
            # The name has to be one plain folder name inside the jobs folder.
            # Looking for separators is not enough: a Windows name carrying a
            # drive, "C:x", has none, and joining it throws the base away.
            folder = (self.root / JOBS_FOLDER / target).resolve()
            if not target or target.startswith(".") or folder.parent != jobs_folder:
                raise SettingsError(f"{target!r} is not the name of a job.")
            with self.lock:
                job = self.jobs.get(target)
                if job is not None:
                    if job.status not in _ENDED:
                        raise SettingsError(
                            f"Job {target!r} is {job.status}; stop it before removing it."
                        )
                    del self.jobs[target]
            if job is None:
                # Not this server's job. It may still belong to a running one.
                described = _describe_job_folder(folder)
                if described is not None and described["status"] not in _ENDED:
                    raise SettingsError(
                        f"Job {target!r} is {described['status']}; stop it before removing it."
                    )
            if folder.is_dir():
                # The job's own folder only: what it printed and its progress.
                # The run it produced lives elsewhere in reports/ and stays.
                shutil.rmtree(folder, ignore_errors=True)
            elif job is None:
                raise SettingsError(f"There is no job called {target!r}.")
            forgotten.append(target)
        return forgotten

    def snapshot(self) -> list[dict[str, Any]]:
        """Every job this server was asked for, then those earlier servers left on disk, newest first."""
        with self.lock:
            jobs = list(self.jobs.values())
        listed = [job.to_dict() for job in jobs]
        known = {job.id for job in jobs}
        listed += [item for item in job_history(self.root) if item["id"] not in known]
        listed.sort(key=lambda item: item["created"], reverse=True)
        return listed

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
                    # Recorded now the process exists, so its number reaches
                    # the file while the command is still running.
                    job.record()
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
        job.record()


def _environment() -> dict[str, str]:
    import os

    return dict(os.environ)
