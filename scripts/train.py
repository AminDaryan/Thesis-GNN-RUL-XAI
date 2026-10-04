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

The paper's graph against one built from the data, with the same number of
edges::

    python scripts/train.py --graph correlation --seeds 5

The paper's graph against graphs with its 41 edges but none of its
engineering knowledge, one drawn at random for every seed: 41 sensor pairs
chosen at random, or the paper's own edges rewired so every sensor keeps its
number of connections::

    python scripts/train.py --graph random --seeds 5
    python scripts/train.py --graph rewired --seeds 5

The paper's own procedure, with no validation set and a fixed 200 epochs, comes
from a configuration that holds no engine back, alone: with nothing to stop on,
every pass runs and the last one's weights are kept, with or without
--no-early-stopping. The option matters only when engines are held back, and
then it runs all 200 passes but still restores the best held-back pass
(A-18)::

    python scripts/train.py --config <a copy of config/fd001.yaml with validation_fraction: 0.0>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_gnn_rul_xai.config import SENSOR_GRAPHS, PreprocessingConfig, load_config
from thesis_gnn_rul_xai.data.pipeline import PreparedData, run_preprocessing
from thesis_gnn_rul_xai.evaluation import evaluate
from thesis_gnn_rul_xai.evaluation.published import published_row
from thesis_gnn_rul_xai.graph.adjacency import connections_of
from thesis_gnn_rul_xai.models import ModelConfig, build_model
from thesis_gnn_rul_xai.training import TrainingConfig, predict, train
from thesis_gnn_rul_xai.utils.logging_setup import configure_logging
from thesis_gnn_rul_xai.utils.paths import raw_data_dir, reports_dir
from thesis_gnn_rul_xai.utils.provenance import (
    capture_code_state,
    current_commit,
    environment,
    file_fingerprints,
    utc_stamp,
    write_code_state,
)

# The rows of the paper's results a run can be set beside, for orientation
# while it is in progress. These are not targets to tune towards; they are
# there so a result that is wildly out of range is noticed immediately.
PUBLISHED_ROWS = ("full", "without_lstm", "without_gcn", "without_physics_prior")


class Progress:
    """Appends one line of JSON per event to a file another program can follow."""

    def __init__(self, path: Path | None) -> None:
        self.handle = path.open("a", encoding="utf-8") if path else None

    def __call__(self, event: str, **fields: object) -> None:
        """Write one event, and flush it so a reader sees it at once."""
        if self.handle is None:
            return
        self.handle.write(json.dumps({"event": event, **fields}, default=float) + "\n")
        self.handle.flush()


def split_record(validation_engines: list[list[int]], fixed_split: bool) -> tuple[str, str]:
    """What summary.json's `split` and the run's folder name say about the engines held back."""
    if not any(validation_engines):
        return "none", "nosplit"
    return ("fixed", "fixedsplit") if fixed_split else ("varies_with_seed", "varsplit")


def shared_preparation(
    preprocessing: PreprocessingConfig, fixed_split: bool, data_dir: Path | None
) -> PreparedData | None:
    """The data every seed reuses under --fixed-split, or None when each seed prepares its own."""
    if not fixed_split:
        return None
    prepared = run_preprocessing(preprocessing, data_dir=data_dir)
    # With no engine held back the run records its split as "none", which
    # training/stored.py rebuilds under each seed, so every seed is prepared
    # under its own seed here too. The seed still numbers the operating
    # conditions of the per-condition scaling, so one shared table would not be
    # the one a reload rebuilds (training/README.md, "The split part of a run
    # folder's name").
    return prepared if len(prepared.validation) else None


def graph_record(prepared: PreparedData) -> dict[str, object]:
    """The graph one seed was given: its kind, the seed it was drawn from, and every connection by name."""
    symbols = prepared.sensor_symbols
    return {
        "graph": prepared.config.graph,
        "drawn_from_seed": prepared.diagnostics.get("graph_seed"),
        "connections": [[symbols[i], symbols[j]] for i, j in connections_of(prepared.adjacency_raw)],
    }


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
    parser.add_argument("--graph", choices=SENSOR_GRAPHS, default=None,
                        help="Overrides the config. 'complete' joins every "
                             "sensor to every other, for the paper's 'without "
                             "physics prior' experiment (A-19); 'correlation' "
                             "joins the 41 most strongly correlated sensor "
                             "pairs on the training engines (A-27); 'random' "
                             "joins 41 pairs chosen at random and 'rewired' "
                             "rewires the paper's graph keeping every sensor's "
                             "number of connections, each drawn from the seed "
                             "the data is prepared under (A-41)")
    parser.add_argument("--adjacency-norm", choices=("symmetric", "row"),
                        default=None,
                        help="Overrides the config. See assumption A-21")
    parser.add_argument("--fixed-split", action="store_true",
                        help="Reuse one train/validation split for every seed. "
                             "The default varies it, so the reported spread "
                             "includes split variance as well as "
                             "initialisation and batch order")
    parser.add_argument("--lr-schedule",
                        choices=("step", "exponential", "cosine", "plateau",
                                 "constant"),
                        default="step", help="See assumption A-17")
    parser.add_argument("--temporal", default="last_cycle",
                        choices=["last_cycle", "mean_window", "window_summary",
                                 "raw_window", "lstm", "gru"],
                        help="Which temporal stage sits in front of the graph "
                             "layers. 'last_cycle' is the stand-in of A-07; "
                             "'lstm' is the paper's own Section 2.2 stage, one "
                             "network per sensor; 'window_summary' gives each "
                             "sensor its last reading, average and change over "
                             "the window, which the paper does not (A-28); 'gru' "
                             "puts a gated recurrent unit where each LSTM goes, "
                             "a stage the paper does not have (A-36)")
    parser.add_argument("--temporal-hidden", type=int, default=32,
                        help="Hidden size of the recurrent stage (A-26; the "
                             "paper states none)")
    parser.add_argument("--temporal-out-features", type=int, default=1,
                        help="Numbers per sensor the recurrent stage hands on "
                             "(A-26 chose 1, the width Table 4 prints)")
    parser.add_argument("--shared-across-sensors", action="store_true",
                        help="One recurrent network for all sensors instead of "
                             "the dedicated one per sensor of Section 2.2")
    parser.add_argument("--gcn-hidden", type=int, nargs="+", default=None,
                        help="Output width of each graph layer (Table 4: 32 64 32)")
    parser.add_argument("--fc-hidden", type=int, default=None,
                        help="Width of the first fully connected layer (Table 4: 128)")
    parser.add_argument("--gcn-bias", action="store_true",
                        help="Give the graph layers a bias; A-12 leaves it out")
    parser.add_argument("--subset", choices=("FD001", "FD002", "FD003", "FD004"),
                        default=None, help="Overrides the config")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Windows per update (Table 6: 32)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="First learning rate (Table 6: 1e-3)")
    parser.add_argument("--final-learning-rate", type=float, default=None,
                        help="Last learning rate (Table 6: 1e-4)")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Passes over the training data without improvement "
                             "before stopping (A-18: 20)")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Train the full epoch count with engines held back; "
                             "the best held-back pass is still restored (A-18). "
                             "With none held back every pass runs anyway")
    parser.add_argument("--progress", type=Path, default=None,
                        help="Append one JSON line per event to this file, for "
                             "the documentation site's playground to follow")
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
    if arguments.subset is not None:
        preprocessing.subset = arguments.subset
    preprocessing.__post_init__()          # re-validate after the overrides
    # ASSUMPTION A-41: a random or rewired graph is drawn from the seed the
    # data is prepared under, so --fixed-split holds one graph for every seed.
    # With no engine held back, though, every seed is prepared under its own
    # seed whatever --fixed-split says (shared_preparation), so the one graph
    # asked for could not be given. Refused rather than silently ignored; the
    # same check runs again below for a fraction too small to hold one engine.
    fixed_graph_refusal = (
        f"--fixed-split with --graph {preprocessing.graph} needs engines held "
        "back (validation_fraction above 0): with none, every seed is prepared "
        "under its own seed and would be given a graph of its own."
    )
    if (arguments.fixed_split and preprocessing.graph in ("random", "rewired")
            and not preprocessing.validation_fraction):
        parser.error(fixed_graph_refusal)

    model_config = ModelConfig(
        temporal=arguments.temporal,
        temporal_hidden=arguments.temporal_hidden,
        temporal_out_features=arguments.temporal_out_features,
        shared_across_sensors=arguments.shared_across_sensors,
        **({"gcn_hidden": tuple(arguments.gcn_hidden)} if arguments.gcn_hidden else {}),
        **({"fc_hidden": arguments.fc_hidden} if arguments.fc_hidden else {}),
        gcn_bias=arguments.gcn_bias,
    )
    # Everything a seed does not change. Built once, so the settings written
    # beside the results are the ones every seed used.
    training_template = TrainingConfig(
        device=arguments.device,
        lr_schedule=arguments.lr_schedule,
        **({"epochs": arguments.epochs} if arguments.epochs else {}),
        **({"batch_size": arguments.batch_size} if arguments.batch_size else {}),
        **({"learning_rate": arguments.learning_rate} if arguments.learning_rate else {}),
        **({"final_learning_rate": arguments.final_learning_rate}
           if arguments.final_learning_rate else {}),
        **({"early_stopping_patience": arguments.early_stopping_patience}
           if arguments.early_stopping_patience else {}),
    )
    if arguments.no_early_stopping:
        training_template = replace(training_template, early_stopping_patience=None)
    progress = Progress(arguments.progress)
    progress("start", seeds=arguments.seeds, epochs=training_template.epochs,
             subset=preprocessing.subset, temporal=arguments.temporal)

    commit = current_commit()
    # Read now, while the code on disk is the code this process imported.
    code_state = capture_code_state()
    print(f"commit {commit}   graph {preprocessing.graph}   "
          f"normalisation {preprocessing.adjacency_normalization}   "
          f"split {'none, nothing held back' if not preprocessing.validation_fraction else 'fixed' if arguments.fixed_split else 'varies with seed'}")

    # With --fixed-split the split is prepared once and reused, which is what
    # this script used to do unconditionally. That understates the spread:
    # five seeds then vary only the weight initialisation and the batch
    # order, while a real repetition of the experiment would also draw a
    # different set of validation engines. With no engine held back there is
    # no split to hold, and every seed is prepared under its own seed.
    shared = shared_preparation(preprocessing, arguments.fixed_split, arguments.data_dir)
    if arguments.fixed_split and shared is None:
        if preprocessing.graph in ("random", "rewired"):
            parser.error(fixed_graph_refusal)
        print("No engine is held back, so --fixed-split has no split to hold: every "
              "seed is prepared under its own seed, and the run records its split as none.")

    runs: list[dict[str, float]] = []
    # What each seed leaves behind besides its four numbers. Held until every
    # seed has finished, so a run that is stopped part-way writes nothing that
    # could be mistaken for a finished one.
    artefacts: list[dict[str, object]] = []
    import torch

    for index in range(arguments.seeds):
        seed = preprocessing.seed + index
        training = replace(training_template, seed=seed)
        print(f"\n{'=' * 72}\nseed {seed}  ({index + 1} of {arguments.seeds})\n{'=' * 72}")
        progress("seed", index=index, seed=seed)

        if shared is not None:
            prepared = shared
        else:
            per_seed = replace(preprocessing, seed=seed)
            prepared = run_preprocessing(per_seed, data_dir=arguments.data_dir)

        torch.manual_seed(seed)
        model = build_model(
            prepared.adjacency_normalized,
            model_config,
            window_size=prepared.config.window_size,
            rul_max=prepared.config.rul_max,
        )
        history = train(
            model, prepared.train, prepared.validation, training,
            on_epoch=lambda row, seed=seed: progress("epoch", seed=seed, **row),
        )

        predicted = predict(model, prepared.test)
        result = evaluate(prepared.test.targets.astype(float), predicted)
        result["seed"] = seed
        result["best_epoch"] = history.best_epoch
        result["validation_rmse"] = history.best_validation_rmse
        result["epochs_run"] = len(history.epochs)
        result["stopped_early"] = history.stopped_early
        result["seconds"] = round(history.seconds, 1)
        runs.append(result)
        artefacts.append({
            "seed": seed,
            "history": history,
            # Which engines this seed trained on, held back and was scored on,
            # and what the scaling measured on its training engines: both follow
            # from the seed, and are kept so a reload can check it rebuilt them.
            "split": {
                name: sorted(int(unit) for unit in set(dataset.units.tolist()))
                for name, dataset in (("train", prepared.train),
                                      ("validation", prepared.validation),
                                      ("test", prepared.test))
            },
            "scaling": prepared.normalizer.statistics_,
            # The graph this seed was given, written out by sensor name: a
            # random or rewired graph differs from seed to seed (A-41).
            "graph": graph_record(prepared),
            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "predictions": pd.DataFrame({
                "unit": prepared.test.units,
                "end_cycle": prepared.test.end_cycles,
                "rul": prepared.test.targets,
                "predicted": predicted,
            }),
        })
        progress("seed_done", seed=seed,
                 **{k: result[k] for k in ("rmse", "score", "r2", "asue", "n_early")})
        print(
            f"  test: RMSE {result['rmse']:.3f}  score {result['score']:.1f}  "
            f"R2 {result['r2']:.3f}  ASUE {result['asue']:.3f} "
            f"over {result['n_early']} engines predicted early"
        )

    table = pd.DataFrame(runs)
    columns = ["rmse", "score", "r2", "asue"]
    # The count ASUE divides by goes wherever ASUE is reported: without it an
    # ASUE cannot be read, and with none it is not a number (assumption A-39).
    reported = [*columns, "n_early"]

    # With no engines held back -- the default -- nothing was measured between
    # epochs, so validation_rmse is NaN for every seed and best_epoch is just
    # the epoch limit repeated. Both columns stay in results.csv, where NaN
    # honestly means "not measured", but printing them invites the reader to
    # compare numbers that were never computed.
    held_back = table["validation_rmse"].notna().any()
    shown = ["seed"] + (["best_epoch", "validation_rmse"] if held_back else []) + reported

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

    # A seed with no ASUE leaves the mean over seeds undefined too, rather than
    # quietly averaging the seeds that have one (assumption A-39).
    means = table[reported].mean(skipna=False)
    spreads = table[reported].std(ddof=1, skipna=False)
    undefined = table.loc[table["asue"].isna(), "seed"].tolist()
    if undefined:
        print(f"\nASUE is not a number on seed(s) {undefined}: no engine was predicted "
              "early, so Equation 24 divides 0 by 0 (assumption A-39).")
    if arguments.seeds > 1:
        print("\nmean +/- standard deviation across seeds:")
        for column in reported:
            print(f"  {column:>7}: {means[column]:9.3f} +/- {spreads[column]:.3f}")
    else:
        print(
            "\nOne seed is not evidence. Re-run with --seeds 5 before quoting "
            "any of these numbers."
        )

    print(f"\nPublished figures for {preprocessing.subset}, for orientation only:")
    for key in PUBLISHED_ROWS:
        row = published_row(key, preprocessing.subset)
        printed = "  ".join(
            f"{measure} {getattr(row, measure)}"
            for measure in ("rmse", "score", "r2", "asue")
            if getattr(row, measure) is not None
        )
        print(f"  {row.label + ' (' + row.source + ')':<52} {printed}")
    print(
        "\nWhich row above this run may be set beside depends on its temporal\n"
        "stage. Without one -- the default -- it is 'ESTGNN w/o LSTM'; with\n"
        "--temporal lstm it is the full ESTGNN, under the size A-26 chose.\n"
        "Any other stage is not in the paper, and no row above applies to it.\n"
        "See docs/assumptions/ for every way this pipeline differs from the\n"
        "paper."
    )
    # The paper never says what its "w/o Physics prior" row used in place of
    # its graph, so it cannot be said that it never trained on one of these;
    # what keeps them from that row is this project's choice (A-19, A-41).
    undescribed = {"correlation": "graph built from correlations",
                   "random": "graph drawn at random", "rewired": "rewiring of its graph"}
    if preprocessing.graph in undescribed:
        print(f"For its network the paper describes no {undescribed[preprocessing.graph]}, "
              "and its\n'w/o Physics prior' row never says what replaced its graph. Only the\n"
              "every-sensor graph is set beside that row (A-19), so no row above applies\n"
              "to this run.")

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
    # much and would collide again. `cfg` is a digest of every preprocessing
    # setting, so two runs whose settings differ anywhere get different names
    # whether or not anyone remembered to add the field. Since 2026-10-03 it
    # leaves out the prose `notes`, which once gave runs with identical
    # settings different names; stored runs keep the names they were given.
    #
    # The same goes for the network and the training settings. Their defaults
    # stay silent; anything else adds a digest of the whole block, `net` for
    # the shape of the network and `train` for how it was trained, and the
    # blocks themselves are written out in full beside the results.
    #
    # Whether the held-back engines change with the seed comes from the
    # engines each seed actually held back, not from --fixed-split alone: with
    # none held back there is no split to vary or to fix, so the name says
    # `nosplit` and summary.json says "none". Until 2026-10-03 such a run was
    # named `varsplit` and recorded "varies_with_seed" (audit item D-4); stored
    # runs keep what they recorded, and training/README.md says how to read it.
    split, split_name = split_record(
        [kept["split"]["validation"] for kept in artefacts], arguments.fixed_split
    )
    stamp = utc_stamp()
    default_model = ModelConfig(
        temporal=arguments.temporal, temporal_hidden=arguments.temporal_hidden
    )
    training_settings = {
        key: getattr(training_template, key)
        for key in ("epochs", "batch_size", "learning_rate", "final_learning_rate",
                    "early_stopping_patience")
    }
    default_training = {key: getattr(TrainingConfig(), key) for key in training_settings}
    if arguments.no_early_stopping:
        default_training["early_stopping_patience"] = None
    training_digest = hashlib.sha256(
        json.dumps(training_settings, sort_keys=True).encode()
    ).hexdigest()[:6]
    config_digest = preprocessing.digest()
    name = (
        f"{preprocessing.subset.lower()}"
        f"_{preprocessing.graph}"
        f"_{preprocessing.adjacency_normalization}"
        f"_{'cap' if preprocessing.clip_test_labels else 'uncap'}"
        f"_{arguments.seeds}seed"
        f"_{split_name}"
        + ("" if arguments.temporal == "last_cycle"
           else f"_{arguments.temporal}{arguments.temporal_hidden}"
           if arguments.temporal in ("lstm", "gru")
           else f"_{arguments.temporal}")
        + ("_noearlystop" if arguments.no_early_stopping else "")
        + ("" if arguments.lr_schedule == "step" else f"_{arguments.lr_schedule}")
        + ("" if model_config == default_model else f"_net{model_config.digest()}")
        + ("" if training_settings == default_training else f"_train{training_digest}")
        + f"_cfg{config_digest}_{commit}_{stamp}"
    )
    output = arguments.output or (reports_dir() / name)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "results.csv", index=False)
    preprocessing.save(output / "preprocessing_config.yaml")
    model_config.save(output / "model_config.yaml")
    # One folder per seed: the trained weights, the per-epoch record and the
    # prediction for every test engine. Without the weights nothing downstream
    # of a trained model -- an explanation, a new plot -- can be made from a
    # stored run; without the record, whether a model over-trained can only be
    # inferred (audit D-10).
    for kept in artefacts:
        folder = output / f"seed_{kept['seed']}"
        kept["history"].save(folder)
        torch.save(kept["state"], folder / "model.pt")
        kept["predictions"].to_csv(folder / "test_predictions.csv", index=False)
        (folder / "split.json").write_text(json.dumps(kept["split"]), encoding="utf-8")
        (folder / "graph.json").write_text(json.dumps(kept["graph"]), encoding="utf-8")
        if kept["scaling"] is not None:
            kept["scaling"].to_csv(folder / "scaling.csv", index=False)
    # A "-dirty" commit says the code differed from the commit, not how. The
    # difference itself is kept, so the run can be rebuilt from the commit.
    changed_code = write_code_state(output, code_state)
    data_dir = Path(arguments.data_dir) if arguments.data_dir else raw_data_dir()
    data_files = [data_dir / f"{kind}_{preprocessing.subset}.txt"
                  for kind in ("train", "test", "RUL")]
    (output / "summary.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "config_digest": config_digest,
                "finished_utc": stamp,
                "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
                "environment": environment(),
                "data_files": file_fingerprints(data_files),
                "uncommitted_code": changed_code,
                "subset": preprocessing.subset,
                "graph": preprocessing.graph,
                "adjacency_normalization": preprocessing.adjacency_normalization,
                "seeds": arguments.seeds,
                "split": split,
                "clip_test_labels": preprocessing.clip_test_labels,
                # Whether early stopping could fire, not only whether it was
                # asked for: with nothing held back there is nothing to stop on.
                "early_stopping": bool(
                    held_back and training_template.early_stopping_patience is not None
                ),
                "validation_fraction": preprocessing.validation_fraction,
                "reported_weights": (
                    "best_validation_epoch" if held_back else "final_epoch"
                ),
                "lr_schedule": arguments.lr_schedule,
                "temporal_stage": arguments.temporal,
                "temporal_hidden": (
                    arguments.temporal_hidden
                    if arguments.temporal in ("lstm", "gru")
                    else None
                ),
                "model": model_config.to_dict(),
                "training": {**training_settings, "lr_schedule": arguments.lr_schedule},
                "seed_folders": [f"seed_{kept['seed']}" for kept in artefacts],
                # The seed each seed folder's graph was drawn from, in the same
                # order; null for a graph that is not drawn at random (A-41).
                "graph_seeds": (
                    [kept["graph"]["drawn_from_seed"] for kept in artefacts]
                    if preprocessing.graph in ("random", "rewired") else None
                ),
                # An undefined value is written as null, which every JSON reader
                # accepts; NaN is not JSON (assumption A-39 for ASUE).
                "mean": {c: None if pd.isna(means[c]) else float(means[c]) for c in reported},
                "std": {
                    c: None if arguments.seeds < 2 or pd.isna(spreads[c]) else float(spreads[c])
                    for c in reported
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    progress("done", directory=output.name, path=str(output))
    print(f"\nSaved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
