#!/usr/bin/env python3
"""
Train the graph network and score it
====================================

Runs the preprocessing pipeline, builds the network of Table 4, trains it with
the hyper-parameters of Table 6, and reports the four measures of Section 3.3
on the test engines.

Because one seed is not evidence, ``--seeds`` repeats the whole run with
different seeds and reports the mean and standard deviation. The paper reports
single runs with no error bars; this is a deliberate improvement, and the
figure worth quoting from this code is the mean across seeds, never the best
one.

Examples
--------
A single run with the paper's settings::

    python scripts/train.py

Five seeds, which is what a reported result should be based on::

    python scripts/train.py --seeds 5

The "without physics prior" ablation, on a fully connected sensor graph::

    python scripts/train.py --graph complete --seeds 5

The paper's own procedure, with no validation set and a fixed 200 epochs::

    python scripts/train.py --no-early-stopping
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_gnn_rul_xai.config import PreprocessingConfig, load_config
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.graph.adjacency import (
    build_fully_connected_adjacency,
    build_physics_adjacency,
    normalize_adjacency,
)
from thesis_gnn_rul_xai.models import ESTGNN
from thesis_gnn_rul_xai.training import (
    TrainingConfig,
    evaluate_model,
    train,
)
from thesis_gnn_rul_xai.utils.logging_setup import configure_logging
from thesis_gnn_rul_xai.utils.paths import reports_dir

#: Published figures for FD001, for orientation while a run is in progress.
#: These are not targets to tune towards; they are there so a result that is
#: wildly out of range is noticed immediately.
PAPER_FD001 = {
    "ESTGNN (full, Table 8)": {"rmse": 11.78, "score": 278.6, "r2": 0.921, "asue": 8.312},
    "ESTGNN w/o LSTM (Table 11)": {"rmse": 14.93},
    "ESTGNN w/o GCN (Table 11)": {"rmse": 16.42},
    "ESTGNN w/o physics prior (Table 11)": {"rmse": 12.34},
}


def build_model(prepared, graph: str, adjacency_norm: str = "symmetric") -> ESTGNN:
    """Build the network on either the physics graph or a complete one."""
    if graph == "physics":
        raw, _ = build_physics_adjacency()
    else:
        raw, _ = build_fully_connected_adjacency(len(prepared.sensor_symbols))
    adjacency = normalize_adjacency(raw, method=adjacency_norm)
    return ESTGNN(
        adjacency,
        in_features=prepared.train.features.shape[2],
        rul_max=prepared.config.rul_max,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the ESTGNN graph stage on C-MAPSS and score it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Preprocessing settings (default: the FD001 defaults)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory holding the C-MAPSS files (default: data/raw)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="How many seeds to average over (default: 1)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the epoch count (Table 6 gives 200)")
    parser.add_argument("--graph", choices=("physics", "complete"), default="physics",
                        help="'complete' is the 'without physics prior' ablation")
    parser.add_argument("--adjacency-norm", choices=("symmetric", "row"),
                        default="symmetric",
                        help="How to normalise the sensor graph. See assumption A-21")
    parser.add_argument("--lr-schedule",
                        choices=("exponential", "cosine", "plateau", "constant"),
                        default="exponential", help="See assumption A-17")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Train the full epoch count, as the paper describes")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=None,
                        help="Directory for the results (default: reports/<name>)")
    parser.add_argument("--quiet", action="store_true", help="Warnings only")
    arguments = parser.parse_args()

    configure_logging(logging.WARNING if arguments.quiet else logging.INFO)

    preprocessing = (
        load_config(arguments.config) if arguments.config else PreprocessingConfig()
    )
    prepared = run_preprocessing(preprocessing, data_dir=arguments.data_dir)

    runs: list[dict[str, float]] = []
    for index in range(arguments.seeds):
        seed = preprocessing.seed + index
        training = TrainingConfig(
            seed=seed,
            device=arguments.device,
            lr_schedule=arguments.lr_schedule,
            early_stopping_patience=None if arguments.no_early_stopping else 20,
            **({"epochs": arguments.epochs} if arguments.epochs else {}),
        )
        print(f"\n{'=' * 72}\nseed {seed}  ({index + 1} of {arguments.seeds})\n{'=' * 72}")

        import torch

        torch.manual_seed(seed)
        model = build_model(prepared, arguments.graph, arguments.adjacency_norm)
        history = train(model, prepared.train, prepared.validation, training)

        result = evaluate_model(model, prepared.test)
        result["seed"] = seed
        result["best_epoch"] = history.best_epoch
        result["validation_rmse"] = history.best_validation_rmse
        runs.append(result)
        print(
            f"  test: RMSE {result['rmse']:.3f}  score {result['score']:.1f}  "
            f"R2 {result['r2']:.3f}  ASUE {result['asue']:.3f}"
        )

    table = pd.DataFrame(runs)
    columns = ["rmse", "score", "r2", "asue"]

    print(f"\n{'=' * 72}")
    print(f"{preprocessing.subset}, {arguments.graph} graph, {arguments.seeds} seed(s)")
    print("=" * 72)
    print(table[["seed", "best_epoch", "validation_rmse", *columns]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    if arguments.seeds > 1:
        print("\nmean +/- standard deviation across seeds:")
        for column in columns:
            values = table[column]
            print(f"  {column:>6}: {values.mean():9.3f} +/- {values.std(ddof=1):.3f}")
    else:
        print(
            "\nOne seed is not evidence. Re-run with --seeds 5 before quoting "
            "any of these numbers."
        )

    print("\nPublished figures for FD001, for orientation only:")
    for name, values in PAPER_FD001.items():
        printed = "  ".join(f"{k} {v}" for k, v in values.items())
        print(f"  {name:<38} {printed}")
    print(
        "\nThe LSTM stage is not implemented here (A-11), so the row to compare\n"
        "against is 'ESTGNN w/o LSTM'. See docs/ASSUMPTIONS.md for every way\n"
        "this pipeline differs from the paper."
    )

    # The node feature mode belongs in the name: two runs that differ only in
    # it are different experiments, and without it the second silently
    # overwrites the first.
    name = (
        f"{preprocessing.subset.lower()}"
        f"_{arguments.graph}"
        f"_{preprocessing.node_feature_mode}"
        f"_{arguments.adjacency_norm}"
        f"_{arguments.seeds}seed"
    )
    output = arguments.output or (reports_dir() / name)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "results.csv", index=False)
    preprocessing.save(output / "preprocessing_config.yaml")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "subset": preprocessing.subset,
                "graph": arguments.graph,
                "node_feature_mode": preprocessing.node_feature_mode,
                "adjacency_norm": arguments.adjacency_norm,
                "seeds": arguments.seeds,
                "early_stopping": not arguments.no_early_stopping,
                "lr_schedule": arguments.lr_schedule,
                "mean": {c: float(table[c].mean()) for c in columns},
                "std": {
                    c: (float(table[c].std(ddof=1)) if arguments.seeds > 1 else None)
                    for c in columns
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
