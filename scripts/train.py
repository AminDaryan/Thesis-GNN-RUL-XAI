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
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import replace

from thesis_gnn_rul_xai.config import PreprocessingConfig, load_config
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.models import ESTGNN
from thesis_gnn_rul_xai.training import (
    TrainingConfig,
    evaluate_model,
    train,
)
from thesis_gnn_rul_xai.utils.logging_setup import configure_logging
from thesis_gnn_rul_xai.utils.paths import reports_dir

# Published figures for FD001, for orientation while a run is in progress.
# These are not targets to tune towards; they are there so a result that is
# wildly out of range is noticed immediately.
PAPER_FD001 = {
    "ESTGNN (full, Table 8)": {"rmse": 11.78, "score": 278.6, "r2": 0.921, "asue": 8.312},
    "ESTGNN w/o LSTM (Table 11)": {"rmse": 14.93},
    "ESTGNN w/o GCN (Table 11)": {"rmse": 16.42},
    "ESTGNN w/o physics prior (Table 11)": {"rmse": 12.34},
}


def current_commit() -> str:
    """The commit this run was produced at, or "unknown" outside a clone.

    A results file whose provenance cannot be established is not evidence.
    This project once carried one that contradicted its own documentation
    and could not be reproduced from any committed state; stamping the
    commit into every run is what stops that happening again.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent,
        )
        if out.returncode != 0:
            return "unknown"
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent,
        )
        return commit + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:  # pragma: no cover - git absent or not a clone
        return "unknown"


def build_model(prepared) -> ESTGNN:
    """Build the network on the graph the pipeline produced.

    Which graph that is, and how it was normalised, are decided by
    ``PreprocessingConfig.graph`` and ``.adjacency_normalization`` and are
    recorded in the configuration saved beside the results. Rebuilding the
    matrix here instead would let the model train on one graph while the
    stored ``adjacency.npz`` held another.
    """
    return ESTGNN(
        prepared.adjacency_normalized,
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
    parser.add_argument("--graph", choices=("physics", "complete"), default=None,
                        help="Overrides the config. 'complete' is the "
                             "'without physics prior' ablation (A-19)")
    parser.add_argument("--adjacency-norm", choices=("symmetric", "row"),
                        default=None,
                        help="Overrides the config. See assumption A-21")
    parser.add_argument("--fixed-split", action="store_true",
                        help="Reuse one train/validation split for every seed. "
                             "The default varies it, so the reported spread "
                             "includes split variance as well as "
                             "initialisation and batch order")
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
    # A command-line choice is written into the configuration object, so the
    # YAML saved beside the results describes the run that was made.
    if arguments.graph is not None:
        preprocessing.graph = arguments.graph
    if arguments.adjacency_norm is not None:
        preprocessing.adjacency_normalization = arguments.adjacency_norm
    preprocessing.__post_init__()          # re-validate after the overrides

    commit = current_commit()
    print(f"commit {commit}   graph {preprocessing.graph}   "
          f"normalisation {preprocessing.adjacency_normalization}   "
          f"split {'fixed' if arguments.fixed_split else 'varies with seed'}")

    # With --fixed-split the split is prepared once and reused, which is what
    # this script used to do unconditionally. That understates the spread:
    # five seeds then vary only the weight initialisation and the batch
    # order, while a real repetition of the experiment would also draw a
    # different set of validation engines.
    shared = (
        run_preprocessing(preprocessing, data_dir=arguments.data_dir)
        if arguments.fixed_split
        else None
    )

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

        if shared is not None:
            prepared = shared
        else:
            per_seed = replace(preprocessing, seed=seed)
            prepared = run_preprocessing(per_seed, data_dir=arguments.data_dir)

        torch.manual_seed(seed)
        model = build_model(prepared)
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

    # With no engines held back -- the default -- nothing was measured between
    # epochs, so validation_rmse is NaN for every seed and best_epoch is just
    # the epoch limit repeated. Both columns stay in results.csv, where NaN
    # honestly means "not measured", but printing them invites the reader to
    # compare numbers that were never computed.
    held_back = table["validation_rmse"].notna().any()
    shown = ["seed"] + (["best_epoch", "validation_rmse"] if held_back else []) + columns

    print(f"\n{'=' * 72}")
    print(f"{preprocessing.subset}, {preprocessing.graph} graph, "
          f"{preprocessing.adjacency_normalization} normalisation, "
          f"{arguments.seeds} seed(s), commit {commit}")
    print("=" * 72)
    print(table[shown].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not held_back:
        print(
            "\nNo engines were held back, so no epoch was chosen over any other "
            "and\nthere is no validation error to report. Every run above is the "
            "state of\nthe model at the epoch limit."
        )

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
    # The commit and a timestamp go in the name, so a second run of the same
    # configuration never silently overwrites the first.
    #
    # Every setting that changes the reported number and can be switched from
    # the command line belongs here too. The timestamp already makes the name
    # unique; what the rest buys is being able to tell two runs apart by
    # reading them. Leaving `clip_test_labels` out once produced two
    # directories differing only by timestamp, one capped and one not, whose
    # RMSEs differ by 1.3 cycles -- see A-02. Non-default switches are marked;
    # defaults stay silent so ordinary names stay short.
    #
    # Naming the fields that happen to matter today is not enough, because a
    # setting reachable only from a YAML file -- the normalisation method, the
    # smoothing parameter, the label ceiling -- changes the number just as
    # much and would collide again. `cfg` is a digest of the whole
    # preprocessing configuration, so two runs whose settings differ anywhere
    # get different names whether or not anyone remembered to add the field.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_digest = hashlib.sha256(
        json.dumps(preprocessing.to_dict(), sort_keys=True, default=str).encode()
    ).hexdigest()[:6]
    name = (
        f"{preprocessing.subset.lower()}"
        f"_{preprocessing.graph}"
        f"_{preprocessing.adjacency_normalization}"
        f"_{'cap' if preprocessing.clip_test_labels else 'uncap'}"
        f"_{arguments.seeds}seed"
        f"_{'fixedsplit' if arguments.fixed_split else 'varsplit'}"
        + ("_noearlystop" if arguments.no_early_stopping else "")
        + ("" if arguments.lr_schedule == "exponential" else f"_{arguments.lr_schedule}")
        + f"_cfg{config_digest}_{commit}_{stamp}"
    )
    output = arguments.output or (reports_dir() / name)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "results.csv", index=False)
    preprocessing.save(output / "preprocessing_config.yaml")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "config_digest": config_digest,
                "finished_utc": stamp,
                "subset": preprocessing.subset,
                "graph": preprocessing.graph,
                "adjacency_normalization": preprocessing.adjacency_normalization,
                "seeds": arguments.seeds,
                "split": "fixed" if arguments.fixed_split else "varies_with_seed",
                "clip_test_labels": preprocessing.clip_test_labels,
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
