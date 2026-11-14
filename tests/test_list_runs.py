"""Checks that scripts/list_runs.py finds every finished run, at any depth, and reads its settings files.

src/thesis_gnn_rul_xai/training/README.md, under "One table of every stored
run", explains the script. The run folders built here hold settings and
results tables only, written by the package's own configuration classes; no
sensor data is invented.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from thesis_gnn_rul_xai.config import PreprocessingConfig

ROOT = Path(__file__).resolve().parents[1]


def _script():
    """Import scripts/list_runs.py as a module."""
    spec = importlib.util.spec_from_file_location("list_runs", ROOT / "scripts" / "list_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_folder(folder: Path, normalization: str, commit: str, rmse: list[float], held_back: list[float]) -> None:
    """Write the files a finished run leaves behind, with the given scaling and errors."""
    folder.mkdir(parents=True)
    PreprocessingConfig(normalization=normalization).save(folder / "preprocessing_config.yaml")
    pd.DataFrame({"seed": [42, 43], "rmse": rmse, "validation_rmse": held_back}).to_csv(
        folder / "results.csv", index=False
    )
    (folder / "summary.json").write_text(json.dumps({"commit": commit, "seeds": 2}), encoding="utf-8")


def test_every_run_is_listed_with_its_scaling_and_sorted_by_held_back_error(tmp_path):
    """A nested run is found, its scaling comes from its settings file, and the lowest held-back error comes first."""
    _run_folder(tmp_path / "fd001_run_a", "zscore", "a7c23d2", [16.0, 17.0], [21.0, 22.0])
    _run_folder(tmp_path / "sensitivity_x" / "runs" / "FD001" / "minmax", "minmax_global",
                "fd65a3b-dirty", [12.0, 13.0], [13.0, 14.0])
    (tmp_path / "figures").mkdir()

    table = _script().build_index(tmp_path).sort_values("validation_rmse_mean")

    assert list(table["run"]) == ["sensitivity_x/runs/FD001/minmax", "fd001_run_a"]
    assert list(table["pre.normalization"]) == ["minmax_global", "zscore"]
    assert list(table["dirty"]) == [True, False]
    assert list(table["validation_rmse_mean"]) == [13.5, 21.5]
    assert list(table["rmse_mean"]) == [12.5, 16.5]


def test_a_folder_without_its_settings_file_is_not_a_run(tmp_path):
    """results.csv alone, as a baselines folder writes it, is not counted as a training run."""
    folder = tmp_path / "baselines_x"
    folder.mkdir()
    pd.DataFrame({"seed": [42], "rmse": [12.0]}).to_csv(folder / "results.csv", index=False)

    assert _script().find_run_folders(tmp_path) == []
