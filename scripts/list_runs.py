"""Write one table of every stored training run in reports/, with all its settings and both errors.

src/thesis_gnn_rul_xai/training/README.md, under "One table of every stored
run", explains the columns and why the table exists.

Usage:
    python scripts/list_runs.py                  # writes reports/runs_index.csv
    python scripts/list_runs.py --sort rmse_mean
    python scripts/list_runs.py --root <folder> --output <file.csv>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_gnn_rul_xai.utils.paths import reports_dir

# The files that make a folder a finished training run.
RUN_FILES = ("results.csv", "preprocessing_config.yaml")

# The columns put first, in this order; every other setting follows them.
LEADING = [
    "run", "commit", "dirty", "finished_utc", "seeds",
    "validation_rmse_mean", "validation_rmse_std", "rmse_mean", "rmse_std",
]


def find_run_folders(root: Path) -> list[Path]:
    """Every folder under root that holds a finished run, at any depth, in name order."""
    found = {
        path.parent
        for path in root.rglob(RUN_FILES[0])
        if all((path.parent / name).is_file() for name in RUN_FILES)
    }
    return sorted(found)


def _read_yaml(path: Path) -> dict[str, Any]:
    """A YAML settings file as a dictionary, without its free-text notes; empty if absent."""
    if not path.is_file():
        return {}
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {key: value for key, value in values.items() if key != "notes"}


def _flat(value: Any) -> Any:
    """A list written as text, so that one setting stays one cell of the table."""
    return json.dumps(value) if isinstance(value, (list, dict)) else value


def describe_run(folder: Path, root: Path) -> dict[str, Any]:
    """One row: where the run is, which code made it, every setting and both errors."""
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    results = pd.read_csv(folder / "results.csv")
    seed_folders = sorted(folder.glob("seed_*/training_config.yaml"))
    commit = str(summary.get("commit", ""))

    row: dict[str, Any] = {
        "run": folder.relative_to(root).as_posix(),
        "commit": commit,
        "dirty": commit.endswith("-dirty"),
        "finished_utc": summary.get("finished_utc"),
        "seeds": len(results),
    }
    for column in ("rmse", "validation_rmse", "score", "r2", "asue"):
        if column in results and results[column].notna().any():
            row[f"{column}_mean"] = float(results[column].mean())
            row[f"{column}_std"] = float(results[column].std()) if len(results) > 1 else None
    for key, value in _read_yaml(folder / "preprocessing_config.yaml").items():
        row[f"pre.{key}"] = _flat(value)
    for key, value in _read_yaml(folder / "model_config.yaml").items():
        row[f"model.{key}"] = _flat(value)
    # Every seed of a run shares one training configuration except its seed,
    # so the first seed's file stands for the run.
    training = _read_yaml(seed_folders[0]) if seed_folders else dict(summary.get("training", {}))
    for key, value in training.items():
        if key not in ("seed", "device"):
            row[f"train.{key}"] = _flat(value)
    for key in ("temporal_stage", "lr_schedule", "early_stopping", "reported_weights", "split"):
        if key in summary:
            row[key] = summary[key]
    return row


def build_index(root: Path) -> pd.DataFrame:
    """The table of every run under root, with the leading columns first."""
    rows = [describe_run(folder, root) for folder in find_run_folders(root)]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    first = [column for column in LEADING if column in table]
    return table[first + [column for column in table if column not in first]]


def main() -> int:
    """Read every run under reports/ and write the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="Folder to search (default: reports/)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write the table (default: <root>/runs_index.csv)")
    parser.add_argument("--sort", default="validation_rmse_mean",
                        help="Column to sort by, lowest first (default: validation_rmse_mean)")
    arguments = parser.parse_args()

    root = arguments.root or reports_dir()
    table = build_index(root)
    if table.empty:
        print(f"No finished runs under {root}.")
        return 1
    if arguments.sort in table:
        table = table.sort_values(arguments.sort, na_position="last")
    output = arguments.output or (root / "runs_index.csv")
    table.to_csv(output, index=False)
    shown = [c for c in ("run", "pre.normalization", "temporal_stage", "pre.graph",
                         "validation_rmse_mean", "rmse_mean", "rmse_std") if c in table]
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(table[shown].to_string(index=False))
    print(f"\n{len(table)} runs, {int(table['dirty'].sum())} of them made with uncommitted code.")
    print(f"Saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
