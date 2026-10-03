"""The commands a checked request becomes, and everything the page needs to draw its form."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...evaluation.published import (
    DATASET_DESCRIPTION,
    FIG_10_RING,
    SHAP_TABLE_12,
    SHAP_TABLE_12_GRAPHS,
)
from .settings import (
    _READ_MORE,
    _STEP_DOCUMENTS,
    FIELDS,
    GAMMA_ABOUT,
    GAMMA_SETTINGS,
    GROUPS,
    PROCEDURE_ABOUT,
    PROCEDURES,
    RESULT_SEEDS,
    _with_more,
    read_more,
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
        "--epochs", str(settings["epochs"]),  # epochs: complete passes over the training data
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
    # A named setting only, never numbers: the script turns the name into one
    # gamma per graph layer, and every name gives the first layer 0.
    if request["method"] == "walks" and request.get("gamma") not in (None, GAMMA_SETTINGS[0]["value"]):
        arguments += ["--gamma", str(request["gamma"])]
    if progress_path is not None:
        arguments += ["--progress", str(progress_path)]
    return arguments


def _dataset_schema() -> dict[str, Any]:
    """The columns shared by all four raw sub-datasets, including every sensor's meaning."""
    from ...data.sensors import (
        OP_SETTING_COLS,
        PAPER_SENSOR_SYMBOLS,
        RAW_COLUMNS,
        SENSORS,
    )

    used = set(PAPER_SENSOR_SYMBOLS)
    return {
        "raw_column_count": len(RAW_COLUMNS),
        "column_groups": [
            {"name": "Engine number", "count": 1, "columns": "unit"},
            {"name": "Cycle number", "count": 1, "columns": "cycle"},
            {"name": "Operating settings", "count": len(OP_SETTING_COLS),
             "columns": ", ".join(OP_SETTING_COLS)},
            {"name": "Sensor readings", "count": len(SENSORS),
             "columns": f"sensor_1 to sensor_{len(SENSORS)}"},
        ],
        "sensors": [
            {"used": sensor.symbol in used, "symbol": sensor.symbol, "column": sensor.column,
             "description": sensor.description, "unit": sensor.unit, "station": sensor.station}
            for sensor in SENSORS
        ],
    }


def options() -> dict[str, Any]:
    """Everything the page needs to draw its form, before anything is chosen."""
    # The positions are scaled to the figures' drawing area, so the page is
    # handed that area too rather than a copy of its two numbers.
    from ..figures import _HEIGHT, _WIDTH, _graph_layout

    graph, positions = _graph_layout()[:2]
    return {
        "fields": [{**_with_more(spec, spec["key"]),
                    "choices": [_with_more(c, f"{spec['key']}={c['value']}") for c in spec.get("choices", [])]}
                   if "choices" in spec else _with_more(spec, spec["key"]) for spec in FIELDS],
        "groups": [{"key": k, "label": label} for k, label in GROUPS],
        "procedures": [_with_more(procedure, "procedure") for procedure in PROCEDURES],
        "procedure_about": PROCEDURE_ABOUT,
        "procedure_more": read_more("procedure"),
        "operation_more": {name: read_more(f"operation={name}") for name in _STEP_DOCUMENTS},
        "explain_more": {key.split("=")[1]: read_more(key) for key in _READ_MORE if key.startswith("explain=")},
        "gamma_about": GAMMA_ABOUT,
        "gamma_settings": [_with_more(setting, "explain=gamma") for setting in GAMMA_SETTINGS],
        "defaults": {spec["key"]: spec["default"] for spec in FIELDS},
        "result_seeds": RESULT_SEEDS,
        "dataset": DATASET_DESCRIPTION,
        "dataset_schema": _dataset_schema(),
        "graph": {
            "positions": {node: [round(x, 1), round(y, 1)] for node, (x, y) in positions.items()},
            "edges": [list(edge) for edge in graph.edges()],
            "width": _WIDTH,
            "height": _HEIGHT,
        },
        "table12": {
            "graphs": list(SHAP_TABLE_12_GRAPHS),
            "values": {sensor: list(values) for sensor, values in SHAP_TABLE_12.items()},
            "ring": list(FIG_10_RING),
        },
    }
