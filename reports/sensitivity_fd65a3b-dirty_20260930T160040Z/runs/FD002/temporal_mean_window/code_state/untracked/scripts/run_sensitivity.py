"""Run and summarize the predefined five-seed sensitivity study."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.evaluation import evaluate
from thesis_gnn_rul_xai.utils.paths import raw_data_dir

DATASETS = ("FD001", "FD002", "FD003", "FD004")
METRICS = ("rmse", "score", "r2", "asue")


@dataclass(frozen=True)
class Variant:
    """One change from the reference configuration."""

    name: str
    factor: str
    level: str
    preprocessing: dict[str, Any] | None = None
    arguments: tuple[str, ...] = ()


VARIANTS = (
    Variant("reference", "reference", "paper LSTM and physics graph"),
    Variant("temporal_last_cycle", "temporal stage", "last reading", arguments=("--temporal", "last_cycle")),
    Variant("temporal_mean_window", "temporal stage", "window average", arguments=("--temporal", "mean_window")),
    Variant("temporal_gru", "temporal stage", "GRU", arguments=("--temporal", "gru")),
    Variant(
        "temporal_shared_lstm",
        "temporal weight sharing",
        "one LSTM shared by all sensors",
        arguments=("--shared-across-sensors",),
    ),
    Variant("graph_complete", "sensor graph", "all sensor pairs", preprocessing={"graph": "complete"}),
    Variant(
        "graph_correlation",
        "sensor graph",
        "41 strongest training correlations",
        preprocessing={"graph": "correlation"},
    ),
    Variant(
        "adjacency_symmetric",
        "graph rescaling",
        "symmetric by both endpoint degrees",
        preprocessing={"adjacency_normalization": "symmetric"},
    ),
    Variant("graph_layer_bias", "graph-layer bias", "included", arguments=("--gcn-bias",)),
    Variant("smoothing_off", "smoothing", "off", preprocessing={"smoothing_alpha": None}),
    Variant(
        "scaling_minmax_per_condition",
        "sensor scaling",
        "training range per operating condition",
        preprocessing={"normalization": "minmax_per_condition"},
    ),
    Variant(
        "scaling_minmax_global",
        "sensor scaling",
        "global training range",
        preprocessing={"normalization": "minmax_global"},
    ),
    Variant(
        "schedule_exponential",
        "learning-rate path",
        "exponential",
        arguments=("--lr-schedule", "exponential"),
    ),
    Variant(
        "schedule_cosine",
        "learning-rate path",
        "cosine",
        arguments=("--lr-schedule", "cosine"),
    ),
    Variant(
        "schedule_plateau",
        "learning-rate path",
        "reduce after held-back error stalls",
        arguments=("--lr-schedule", "plateau"),
    ),
    Variant(
        "schedule_constant",
        "learning-rate path",
        "constant at the first rate",
        arguments=("--lr-schedule", "constant"),
    ),
    # ASSUMPTION A-32: test one decade below and above the paper's two rates,
    # preserving their ten-to-one ratio. The paper reports no rate sensitivity.
    Variant(
        "learning_rate_tenth",
        "learning-rate scale",
        "one tenth of the paper's rates",
        arguments=("--learning-rate", "0.0001", "--final-learning-rate", "0.00001"),
    ),
    Variant(
        "learning_rate_tenfold",
        "learning-rate scale",
        "ten times the paper's rates",
        arguments=("--learning-rate", "0.01", "--final-learning-rate", "0.001"),
    ),
)


FIXED = {
    "gcn_hidden": [32, 64, 32],
    "fc_hidden": 128,
    "batch_size": 32,
    "epochs": 200,
    "first_seed": 42,
    "seeds": 5,
    "temporal_hidden": 32,
    "temporal_out_features": 1,
    "rul_max": 125,
    "window_size": 30,
    "window_stride": 1,
    "smoothing_alpha_when_enabled": 0.5,
}


def _stamp() -> str:
    """Return a compact UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _commit() -> str:
    """Return the current commit with a dirty marker when needed."""
    head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    return f"{head}-dirty" if dirty else head


def _configuration(dataset: str, variant: Variant) -> PreprocessingConfig:
    """Build one preprocessing configuration while enforcing fixed values."""
    values = {
        "subset": dataset,
        "window_size": FIXED["window_size"],
        "window_stride": FIXED["window_stride"],
        "rul_max": FIXED["rul_max"],
        "smoothing_alpha": FIXED["smoothing_alpha_when_enabled"],
        "seed": FIXED["first_seed"],
        "notes": (
            f"Sensitivity study: {variant.name} on {dataset}. Only the factor "
            "named by the variant differs from the reference configuration."
        ),
    }
    values.update(variant.preprocessing or {})
    return replace(PreprocessingConfig(), **values)


def _command(
    python: Path,
    config_path: Path,
    output_path: Path,
    progress_path: Path,
    variant: Variant,
    device: str,
) -> list[str]:
    """Build the exact training command for one cell of the study."""
    command = [
        str(python),
        str(ROOT / "scripts" / "train.py"),
        "--config", str(config_path),
        "--seeds", str(FIXED["seeds"]),
        "--epochs", str(FIXED["epochs"]),
        "--batch-size", str(FIXED["batch_size"]),
        "--learning-rate", "0.001",
        "--final-learning-rate", "0.0001",
        "--lr-schedule", "step",
        "--temporal", "lstm",
        "--temporal-hidden", str(FIXED["temporal_hidden"]),
        "--temporal-out-features", str(FIXED["temporal_out_features"]),
        "--gcn-hidden", *(str(value) for value in FIXED["gcn_hidden"]),
        "--fc-hidden", str(FIXED["fc_hidden"]),
        "--early-stopping-patience", "20",
        "--device", device,
        "--progress", str(progress_path),
        "--output", str(output_path),
        "--quiet",
    ]

    overridden = {variant.arguments[index] for index in range(0, len(variant.arguments), 2)
                  if variant.arguments[index].startswith("--")}
    if "--gcn-bias" in variant.arguments or "--shared-across-sensors" in variant.arguments:
        overridden.update(flag for flag in variant.arguments if flag.startswith("--"))
    pairs = {
        "--learning-rate": 2,
        "--final-learning-rate": 2,
        "--lr-schedule": 2,
        "--temporal": 2,
    }
    for option, width in pairs.items():
        if option in overridden:
            index = command.index(option)
            del command[index:index + width]
    command.extend(variant.arguments)
    return command


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one five-seed cell, or reuse its completed output."""
    output = Path(job["output"])
    summary = output / "summary.json"
    if summary.exists():
        return {**job, "status": "reused"}
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(job["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            job["command"], cwd=ROOT, stdout=stream,
            stderr=subprocess.STDOUT, text=True,
        )
    status = "completed" if completed.returncode == 0 else "failed"
    return {**job, "status": status, "returncode": completed.returncode}


def _aggregate(root: Path, selected: tuple[Variant, ...]) -> None:
    """Write per-seed results and paired changes from the reference."""
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for variant in selected:
            result_path = root / "runs" / dataset / variant.name / "results.csv"
            if not result_path.exists():
                continue
            table = pd.read_csv(result_path)
            for row in table.to_dict(orient="records"):
                records.append({
                    "dataset": dataset,
                    "variant": variant.name,
                    "factor": variant.factor,
                    "level": variant.level,
                    **row,
                })

    if not records:
        return
    seeds = pd.DataFrame(records)
    seeds.to_csv(root / "seed_results.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        subset = seeds[seeds["dataset"] == dataset]
        reference = subset[subset["variant"] == "reference"].set_index("seed")
        if reference.empty:
            continue
        for variant in selected:
            current = subset[subset["variant"] == variant.name].set_index("seed")
            common = reference.index.intersection(current.index)
            if common.empty:
                continue
            row: dict[str, Any] = {
                "dataset": dataset,
                "variant": variant.name,
                "factor": variant.factor,
                "level": variant.level,
                "seeds": len(common),
            }
            for metric in METRICS:
                values = current.loc[common, metric].astype(float)
                baseline = reference.loc[common, metric].astype(float)
                differences = values - baseline
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1)
                row[f"{metric}_delta"] = differences.mean()
                row[f"{metric}_delta_std"] = differences.std(ddof=1)
                row[f"{metric}_paired_p"] = (
                    1.0 if np.allclose(differences, 0.0, rtol=0.0, atol=1e-12)
                    else float(ttest_rel(values, baseline).pvalue)
                )
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(root / "summary.csv", index=False)


def _uncapped_reference_scores(root: Path) -> None:
    """Rescore reference predictions against uncapped test labels without retraining."""
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        run = root / "runs" / dataset / "reference"
        if not (run / "summary.json").exists():
            continue
        config = replace(
            _configuration(dataset, VARIANTS[0]), clip_test_labels=False
        )
        prepared = run_preprocessing(config, data_dir=raw_data_dir())
        truth = pd.DataFrame({
            "unit": prepared.test.units,
            "rul_uncapped": prepared.test.targets,
        })
        for seed in range(FIXED["first_seed"], FIXED["first_seed"] + FIXED["seeds"]):
            predictions = pd.read_csv(run / f"seed_{seed}" / "test_predictions.csv")
            joined = predictions.merge(truth, on="unit", how="inner", validate="one_to_one")
            measured = evaluate(
                joined["rul_uncapped"].to_numpy(dtype=float),
                joined["predicted"].to_numpy(dtype=float),
            )
            rows.append({"dataset": dataset, "variant": "reference_uncapped_scoring", "seed": seed, **measured})
    if rows:
        pd.DataFrame(rows).to_csv(root / "reference_uncapped_scoring.csv", index=False)


def _parse() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", choices=[item.name for item in VARIANTS])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Create, run and summarize the sensitivity matrix."""
    arguments = _parse()
    if arguments.workers < 1:
        raise ValueError("workers must be at least one")
    selected = tuple(
        item for item in VARIANTS
        if arguments.variants is None or item.name in arguments.variants
    )
    commit = _commit()
    root = arguments.output or ROOT / "reports" / f"sensitivity_{commit}_{_stamp()}"
    root = root.resolve()
    configs = root / "configs"
    logs = root / "logs"
    progress = root / "progress"
    for folder in (configs, logs, progress):
        folder.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for variant in selected:
        for dataset in arguments.datasets:
            config_path = configs / f"{dataset}_{variant.name}.yaml"
            _configuration(dataset, variant).save(config_path)
            output_path = root / "runs" / dataset / variant.name
            progress_path = progress / f"{dataset}_{variant.name}.jsonl"
            command = _command(
                arguments.python.resolve(), config_path, output_path,
                progress_path, variant, arguments.device,
            )
            jobs.append({
                "dataset": dataset,
                "variant": variant.name,
                "factor": variant.factor,
                "level": variant.level,
                "output": str(output_path),
                "log": str(logs / f"{dataset}_{variant.name}.log"),
                "command": command,
            })

    manifest = {
        "study": "one-factor-at-a-time sensitivity study",
        "created_utc": _stamp(),
        "commit_at_launch": commit,
        "fixed": FIXED,
        "datasets": arguments.datasets,
        "jobs": jobs,
    }
    (root / "study.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    with (root / "commands.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "variant", "factor", "level", "command"))
        writer.writeheader()
        for job in jobs:
            writer.writerow({**{key: job[key] for key in ("dataset", "variant", "factor", "level")},
                             "command": subprocess.list2cmdline(job["command"])})

    print(f"Study: {root}")
    print(f"Cells: {len(jobs)}; trained models: {len(jobs) * FIXED['seeds']}")
    if arguments.dry_run:
        return

    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        futures = {pool.submit(_run_job, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            print(f"{result['status']:>9}  {result['dataset']}  {result['variant']}", flush=True)
            if result["status"] == "failed":
                failures.append(result)
            _aggregate(root, selected)

    _uncapped_reference_scores(root)
    _aggregate(root, selected)
    if failures:
        failed = ", ".join(f"{item['dataset']}/{item['variant']}" for item in failures)
        raise SystemExit(f"Failed cells: {failed}. See their log files.")
    print(f"Complete: {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
