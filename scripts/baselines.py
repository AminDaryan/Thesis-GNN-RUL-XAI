#!/usr/bin/env python3
"""Sets ordinary models beside a stored run's networks, seed by seed, on the same test engines.

Explained in src/thesis_gnn_rul_xai/evaluation/README.md, under "Ordinary models
beside the network".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from thesis_gnn_rul_xai.config import settings_without_notes
from thesis_gnn_rul_xai.data.windowing import WindowedDataset
from thesis_gnn_rul_xai.evaluation import evaluate, rmse
from thesis_gnn_rul_xai.utils.paths import raw_data_dir, reports_dir
from thesis_gnn_rul_xai.utils.provenance import (
    capture_code_state,
    current_commit,
    environment,
    file_fingerprints,
    utc_stamp,
    write_code_state_archive,
)

MEASURES = ("rmse", "score", "r2", "asue")

# A saved prediction is written with a handful of decimals, so a reloaded
# network can differ from it by rounding; anything this large is a different
# network. The same bound as tests/test_run_records.py.
PREDICTION_TOLERANCE = 1e-3

# Rescoring the saved predictions differs from results.csv only by that same
# rounding: at most 1.85e-7 of the stored value, measured on 2026-10-02 over
# every seed and measure of the two runs the comparison was first made on.
RESCORE_TOLERANCE = 1e-6

DRAWS = 20_000
RESAMPLING_SEED = 0


class NotReproduced(RuntimeError):
    """A stored run whose own records the rebuilt data does not reproduce."""


@dataclass(frozen=True)
class Baseline:
    """One ordinary model of the ladder: its key, its label, what it is given and how it is built."""

    key: str
    label: str
    inputs: str
    build: Callable[[int], Any]
    shuffled_labels: bool = False


def _boosting(seed: int) -> GradientBoostingRegressor:
    """scikit-learn's gradient boosting at its defaults, with the seed fixed."""
    return GradientBoostingRegressor(random_state=seed)


# Nothing tuned: scikit-learn's defaults, with the seed of the stored network
# as the random seed. The one departure is the width of the small network, 64
# rather than 100. The list was not drawn up blind: five of the first eight
# models (linear regression, the network of 64, the random forest, gradient
# boosting and a constant) and that width follow the audit of 2026-09-08, which
# had already measured them on the same fourteen numbers under an earlier
# configuration. Shuffled labels, engine age and the 42 summaries were added
# for the comparison of 2026-09-14, and the six that take the summaries apart
# the day after, once the 42-summary row had been seen. Corrected 2026-10-02:
# this comment first said the first eight were listed before any result was
# seen, which a review found contradicted by the audit's own table.
LADDER: tuple[Baseline, ...] = (
    Baseline("constant", "constant: mean training label", "none",
             lambda seed: DummyRegressor()),
    # Given the same fourteen numbers as the network, so that the only thing
    # taken away is the link between a window and its own label.
    Baseline("boosting_shuffled", "gradient boosting trained on shuffled labels, same 14 numbers",
             "last", _boosting, shuffled_labels=True),
    Baseline("boosting_age", "gradient boosting on engine age alone", "age", _boosting),
    Baseline("linear_last", "linear regression, same 14 numbers", "last",
             lambda seed: LinearRegression()),
    Baseline("network64_last", "one-hidden-layer network of 64, same 14 numbers", "last",
             lambda seed: MLPRegressor(hidden_layer_sizes=(64,), random_state=seed)),
    Baseline("forest_last", "random forest, same 14 numbers", "last",
             lambda seed: RandomForestRegressor(random_state=seed)),
    Baseline("boosting_last", "gradient boosting, same 14 numbers", "last", _boosting),
    Baseline("boosting_summary", "gradient boosting, 42 window summaries",
             "last+mean+slope", _boosting),
    Baseline("boosting_mean", "gradient boosting, window mean alone", "mean", _boosting),
    Baseline("boosting_slope", "gradient boosting, window slope alone", "slope", _boosting),
    Baseline("boosting_last_mean", "gradient boosting, last value and mean",
             "last+mean", _boosting),
    Baseline("boosting_last_slope", "gradient boosting, last value and slope",
             "last+slope", _boosting),
    Baseline("boosting_mean_slope", "gradient boosting, mean and slope",
             "mean+slope", _boosting),
    Baseline("linear_summary", "linear regression, 42 window summaries",
             "last+mean+slope", lambda seed: LinearRegression()),
)


def window_slopes(features: np.ndarray) -> np.ndarray:
    """The least-squares slope of each sensor's window, in reading units per cycle."""
    steps = np.arange(features.shape[2], dtype=np.float64)
    steps -= steps.mean()
    return (features.astype(np.float64) * steps).sum(axis=2) / float(steps @ steps)


def model_inputs(dataset: WindowedDataset, inputs: str) -> np.ndarray:
    """The table an ordinary model is given: one row per window, one column per number."""
    if inputs == "none":
        return np.zeros((len(dataset), 1))
    if inputs == "age":
        return dataset.end_cycles.astype(np.float64)[:, None]
    # Averaged in double precision, so each mean is the average of the thirty
    # single-precision readings to about sixteen digits whatever order they are
    # added in. The tree models round every input back to single precision
    # before building a tree, so this hands them no extra digits; it only fixes
    # which single-precision value they see (evaluation/README.md, "The window
    # summaries").
    features = dataset.features.astype(np.float64)
    # The last reading of each window is index -1. Index 0 is the first
    # reading, 29 cycles out of date, and a baseline given it measures
    # something else without any error being raised.
    parts = {
        "last": lambda: features[:, :, -1],
        "mean": lambda: features.mean(axis=2),
        "slope": lambda: window_slopes(features),
    }
    return np.concatenate([parts[name]() for name in inputs.split("+")], axis=1)


def fit_and_predict(
    baseline: Baseline, train: WindowedDataset, test: WindowedDataset, seed: int, rul_max: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit one ordinary model on the training windows; its clipped and raw test predictions."""
    targets = train.targets.astype(np.float64)
    if baseline.shuffled_labels:
        targets = np.random.default_rng(seed).permutation(targets)
    model = baseline.build(seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(model_inputs(train, baseline.inputs), targets)
    predicted = np.asarray(model.predict(model_inputs(test, baseline.inputs)), dtype=np.float64)
    # ASSUMPTION A-38: clip every prediction to [0, the label ceiling], so an
    # ordinary model is held to the range the network cannot leave: its output
    # is a sigmoid scaled by the ceiling. Every capped label lies in the same
    # range (A-02 caps the labels; this clip is a separate choice), so the clip
    # can never raise a baseline's error; per_seed.csv records the RMSE without
    # it, to show by how much it lowers it.
    clipped = np.clip(predicted, 0.0, rul_max)
    messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    return clipped, predicted, messages


def short_name(run_name: str) -> str:
    """A run's directory name without its commit and timestamp."""
    match = re.match(r"^(?P<short>.+?)_[0-9a-f]{7,40}(?:-dirty)?_\d{8}T\d{6}Z$", run_name)
    return match.group("short") if match else run_name


def folder_name(
    run_name: str, seeds: Sequence[int], run_seeds: Sequence[int], models: int, commit: str, stamp: str
) -> str:
    """The new folder's name: the source run's settings, what was left out, the commit and the time."""
    return (
        f"baselines_{short_name(run_name)}"
        + ("" if list(seeds) == list(run_seeds) else f"_seeds{'-'.join(str(s) for s in seeds)}")
        + ("" if models == len(LADDER) else f"_{models}of{len(LADDER)}models")
        + f"_{commit}_{stamp}"
    )


def network_label(summary: dict[str, Any]) -> str:
    """How a stored network is named in the table: its temporal stage and which weights it kept."""
    stage = str(summary.get("temporal_stage", "last_cycle")).replace("_", " ")
    kept = ("best held-back pass restored"
            if summary.get("reported_weights") == "best_validation_epoch"
            else "weights of the last pass")
    return f"stored network, {stage} stage, {kept}"


def rebuild_and_verify(run: Any, seed: int, data_dir: Path) -> tuple[Any, np.ndarray, float]:
    """Rebuild one seed's data under the run's settings, and refuse unless its records reproduce."""
    import torch

    from thesis_gnn_rul_xai.training import load_seed
    from thesis_gnn_rul_xai.training.run import predict

    folder = run.path / f"seed_{seed}"
    saved_path = folder / "test_predictions.csv"
    if not saved_path.is_file():
        raise NotReproduced(f"{run.name}, seed {seed}: no saved test predictions.")
    if not run.weights(seed).is_file():
        raise NotReproduced(
            f"{run.name}, seed {seed}: no saved network, so the engines it trained on "
            "cannot be confirmed by reloading it."
        )
    with torch.no_grad():
        model, prepared = load_seed(run, seed, data_dir=data_dir)
        reloaded = predict(model, prepared.test, device="cpu")
    saved = pd.read_csv(saved_path)
    problems: list[str] = []
    gap = float("nan")
    if not np.array_equal(saved["unit"].to_numpy(), prepared.test.units):
        problems.append("the saved test engines are not the rebuilt ones")
    elif not np.allclose(saved["rul"].to_numpy(), prepared.test.targets, rtol=0, atol=1e-6):
        problems.append("the saved test labels are not the rebuilt ones")
    else:
        gap = float(np.abs(reloaded - saved["predicted"].to_numpy()).max())
        if not gap < PREDICTION_TOLERANCE:
            problems.append(
                f"the reloaded network's predictions differ from the saved ones by {gap:.2e} cycles"
            )
    recorded = run.results.loc[run.results["seed"] == seed]
    if len(recorded) != 1:
        problems.append("results.csv does not hold exactly one row for this seed")
    else:
        rescored = evaluate(saved["rul"].to_numpy(float), saved["predicted"].to_numpy(float))
        for measure in MEASURES:
            stored = float(recorded.iloc[0][measure])
            # equal_nan: an ASUE with no engine predicted early is not a number
            # on both sides (assumption A-39), and that is agreement.
            if not np.isclose(rescored[measure], stored, rtol=RESCORE_TOLERANCE,
                              atol=RESCORE_TOLERANCE, equal_nan=True):
                problems.append(
                    f"{measure} of the saved predictions is {rescored[measure]!r}, "
                    f"results.csv says {stored!r}"
                )
    if (folder / "split.json").is_file():
        split = json.loads((folder / "split.json").read_text(encoding="utf-8"))
        for name in ("train", "validation", "test"):
            rebuilt = sorted(int(u) for u in set(getattr(prepared, name).units.tolist()))
            if rebuilt != split[name]:
                problems.append(f"the {name} engines differ from split.json")
    if (folder / "scaling.csv").is_file():
        stored_scaling = pd.read_csv(folder / "scaling.csv")
        rebuilt_scaling = prepared.normalizer.statistics_.reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(rebuilt_scaling, stored_scaling,
                                          check_dtype=False, rtol=1e-12)
        except AssertionError:
            problems.append("the rebuilt scaling differs from scaling.csv")
    if problems:
        raise NotReproduced(f"{run.name}, seed {seed}: " + "; ".join(problems) + ".")
    return prepared, saved["predicted"].to_numpy(np.float64), gap


def check_data_files(run: Any, data_dir: Path) -> None:
    """Refuse when the NASA files differ from the ones the run recorded reading."""
    recorded = run.summary.get("data_files")
    if not recorded:
        return
    today = file_fingerprints([data_dir / name for name in recorded])
    if today != recorded:
        raise NotReproduced(f"{run.name}: the NASA files differ from the ones this run read.")


def check_same_data(run: Any, other: Any, seeds: Sequence[int]) -> None:
    """Refuse a second run whose seeds or preprocessing differ from the first one's."""
    missing = sorted(set(seeds) - set(other.seeds))
    if missing:
        raise NotReproduced(f"{other.name} has no seed {missing}.")
    mine = settings_without_notes(run.preprocessing.to_dict())
    theirs = settings_without_notes(other.preprocessing.to_dict())
    # With nothing held back there is no split, and runs stored before
    # 2026-10-03 recorded "varies_with_seed" for it anyway while later ones
    # record "none" (audit item D-4); the field is compared only when engines
    # were held back.
    held_back = bool(mine.get("validation_fraction"))
    if mine != theirs or (held_back and run.summary.get("split") != other.summary.get("split")):
        raise NotReproduced(
            f"{other.name} was prepared differently from {run.name}, so its "
            "engines and inputs are not the ones the ladder is fitted on."
        )


def resampled_rmse(truth: np.ndarray, predicted: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """The RMSE averaged over seeds, once per resampling of the test engines."""
    squared = (predicted - truth) ** 2
    per_seed = [np.sqrt(squared[s][draws].mean(axis=1)) for s in range(squared.shape[0])]
    return np.mean(per_seed, axis=0)


def compare(
    network: str, rows: dict[str, dict[str, Any]], resampled: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Every model set against one stored network: the difference, the seeds and both tests."""
    reference = np.asarray(rows[network]["rmse_by_seed"])
    out: list[dict[str, Any]] = []
    for key, row in rows.items():
        values = np.asarray(row["rmse_by_seed"])
        entry: dict[str, Any] = {
            "network": network,
            "model": key,
            "label": row["label"],
            "inputs": row["inputs"],
            "seeds": len(values),
            "rmse_mean": float(values.mean()),
            "rmse_std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
            # n_early beside ASUE: the engines it averages over, and with none
            # it is not a number and so is this mean (assumption A-39).
            **{f"{m}_mean": float(np.mean(row[f"{m}_by_seed"])) for m in (*MEASURES[1:], "n_early")},
            "rmse_low": float(np.percentile(resampled[key], 2.5)),
            "rmse_high": float(np.percentile(resampled[key], 97.5)),
        }
        if key == network:
            entry.update(network_minus_model=float("nan"), seeds_network_better=float("nan"),
                         paired_t_p=float("nan"), difference_low=float("nan"),
                         difference_high=float("nan"), resamples_network_better=float("nan"))
        else:
            difference = resampled[network] - resampled[key]
            entry.update(
                network_minus_model=float((reference - values).mean()),
                seeds_network_better=int((reference < values).sum()),
                paired_t_p=(float(ttest_rel(reference, values).pvalue)
                            if len(values) > 1 else float("nan")),
                difference_low=float(np.percentile(difference, 2.5)),
                difference_high=float(np.percentile(difference, 97.5)),
                resamples_network_better=float((difference < 0).mean()),
            )
        entry.update({f"rmse_seed_{s}": float(v)
                      for s, v in zip(row["seeds"], values, strict=True)})
        out.append(entry)
    return out


def _signed(value: float, digits: int = 3) -> str:
    """A difference with its sign written out, the minus as a true minus sign."""
    minus = "\N{MINUS SIGN}"
    return f"+{value:.{digits}f}" if value >= 0 else f"{minus}{abs(value):.{digits}f}"


def _p(value: float) -> str:
    """A p-value as the table prints it."""
    return "< 0.0001" if value < 1e-4 else f"{value:.4f}"


def table(results: pd.DataFrame, labels: dict[str, str], runs: dict[str, str], draws: int) -> str:
    """The comparison as Markdown tables, one per stored network."""
    lines: list[str] = []
    for network, block in results.groupby("network", sort=False):
        n = int(block["seeds"].iloc[0])
        lines += [
            f"## Set against: {labels[network]}",
            "",
            f"Stored run `{runs[network]}`.",
            "",
            f"| Model | RMSE, {n} seeds | Network's RMSE minus this | Seeds network better "
            f"| Paired t over seeds | 95 % interval over {draws:,} resamplings of the test engines |",
            "|---|---|---|---|---|---|",
        ]
        for row in block.itertuples():
            spread = f"{row.rmse_mean:.3f} ± {row.rmse_std:.3f}"
            if row.model == network:
                lines.append(f"| {row.label} | {spread} | — | — | — "
                             f"| own RMSE [{row.rmse_low:.2f}, {row.rmse_high:.2f}] |")
            else:
                lines.append(
                    f"| {row.label} | {spread} | {_signed(row.network_minus_model)} "
                    f"| {int(row.seeds_network_better)}/{n} | {_p(row.paired_t_p)} "
                    f"| [{_signed(row.difference_low, 2)}, {_signed(row.difference_high, 2)}] |"
                )
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Read the stored runs, verify them, fit the ladder seed by seed, and write the folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True,
                        help="Stored run directory, or its name under reports/")
    parser.add_argument("--also", action="append", default=[],
                        help="Another stored run prepared the same way, set beside as a row")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Only these of the run's seeds (default: every one)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=[b.key for b in LADDER],
                        help="Only these models of the ladder (default: every one)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory holding the C-MAPSS files (default: data/raw)")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Where the new folder is made (default: reports/)")
    arguments = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from thesis_gnn_rul_xai.training import read_run

    commit = current_commit()
    # Read now, while the code on disk is the code this process imported.
    code_state = capture_code_state()
    started = time.perf_counter()

    run = read_run(arguments.run)
    others = [read_run(path) for path in arguments.also]
    seeds = list(arguments.seeds or run.seeds)
    unknown = sorted(set(seeds) - set(run.seeds))
    if unknown:
        parser.error(f"{run.name} trained seeds {run.seeds}, not {unknown}.")
    ladder = [b for b in LADDER if arguments.models is None or b.key in arguments.models]
    data_dir = Path(arguments.data_dir) if arguments.data_dir else raw_data_dir()
    rul_max = float(run.preprocessing.rul_max)

    networks = {"network": run}
    for index, other in enumerate(others, start=1):
        networks[f"network_{index}"] = other
    labels = {key: network_label(stored.summary) for key, stored in networks.items()}
    output_root = Path(arguments.output_root) if arguments.output_root else reports_dir()
    # Checked before the models are fitted rather than after: Windows refuses a
    # path longer than 259 characters, and the run takes minutes.
    longest = output_root.resolve() / folder_name(
        run.name, seeds, run.seeds, len(ladder), commit, utc_stamp()
    ) / "predictions.csv"
    if sys.platform == "win32" and len(str(longest)) > 259:
        parser.error(f"{longest} would be {len(str(longest))} characters, longer than "
                     "Windows allows; give a shorter --output-root.")
    labels.update({b.key: b.label for b in ladder})

    try:
        for stored in networks.values():
            check_data_files(stored, data_dir)
        for other in others:
            check_same_data(run, other, seeds)

        records: list[dict[str, Any]] = []
        predictions: list[pd.DataFrame] = []
        verified: dict[str, dict[str, float]] = {key: {} for key in networks}
        warned: dict[str, list[str]] = {}
        truth_by_seed: list[np.ndarray] = []
        units: np.ndarray | None = None
        for seed in seeds:
            print(f"seed {seed}: rebuilding the data and checking the stored networks", flush=True)
            prepared, saved, gap = rebuild_and_verify(run, seed, data_dir)
            verified["network"][str(seed)] = gap
            # A stored network was trained elsewhere, so it has no fitting time
            # here, and nothing clips its predictions, so its raw ones are the same.
            outputs: dict[str, tuple[np.ndarray, np.ndarray, float]] = {
                "network": (saved, saved, float("nan"))
            }
            for key, other in list(networks.items())[1:]:
                theirs, their_saved, their_gap = rebuild_and_verify(other, seed, data_dir)
                if not (np.array_equal(theirs.test.features, prepared.test.features)
                        and np.array_equal(theirs.train.units, prepared.train.units)):
                    raise NotReproduced(f"{other.name}, seed {seed}: rebuilt data differ from {run.name}'s.")
                verified[key][str(seed)] = their_gap
                outputs[key] = (their_saved, their_saved, float("nan"))
            truth = prepared.test.targets.astype(np.float64)
            if units is None:
                units = prepared.test.units
            elif not np.array_equal(units, prepared.test.units):
                raise NotReproduced(f"seed {seed}: the test engines changed between seeds.")
            truth_by_seed.append(truth)
            for baseline in ladder:
                begun = time.perf_counter()
                predicted, raw, messages = fit_and_predict(
                    baseline, prepared.train, prepared.test, seed, rul_max
                )
                outputs[baseline.key] = (predicted, raw, time.perf_counter() - begun)
                if messages:
                    warned.setdefault(baseline.key, []).append(f"seed {seed}: {messages[0]}")
            for key, (predicted, raw, seconds) in outputs.items():
                scored = evaluate(truth, predicted)
                records.append({"model": key, "seed": seed, "seconds": round(seconds, 1),
                                **{m: scored[m] for m in (*MEASURES, "mean_error", "n_late", "n_early")},
                                "n_clipped": int((predicted != raw).sum()),
                                "rmse_unclipped": rmse(truth, raw)})
                predictions.append(pd.DataFrame({"model": key, "seed": seed, "unit": units,
                                                 "rul": truth, "predicted": predicted}))
            print("  " + "  ".join(f"{key} {evaluate(truth, p)['rmse']:.3f}"
                                   for key, (p, _, _) in outputs.items()), flush=True)
    except NotReproduced as refusal:
        print(f"\nRefused: {refusal}\nNothing was written.", file=sys.stderr)
        return 2

    per_seed = pd.DataFrame(records)
    every_prediction = pd.concat(predictions, ignore_index=True)
    truth = np.vstack(truth_by_seed)
    keys = list(networks) + [b.key for b in ladder]
    inputs = dict.fromkeys(networks, "the network's own")
    inputs.update({b.key: b.inputs for b in ladder})
    rows = {
        key: {
            "label": labels[key],
            "inputs": inputs[key],
            "seeds": seeds,
            **{f"{m}_by_seed": [float(per_seed.loc[(per_seed.model == key) & (per_seed.seed == s), m].iloc[0])
                                for s in seeds] for m in (*MEASURES, "n_early")},
        }
        for key in keys
    }
    # The same resampled sets of engines for every model and every seed, so a
    # difference compares two models on one set of engines, and each seed's
    # RMSE on that set is averaged before two models are compared.
    draws = np.random.default_rng(RESAMPLING_SEED).integers(0, truth.shape[1], size=(DRAWS, truth.shape[1]))
    resampled = {
        key: resampled_rmse(
            truth,
            np.vstack([every_prediction.loc[(every_prediction.model == key)
                                            & (every_prediction.seed == s), "predicted"].to_numpy()
                       for s in seeds]),
            draws,
        )
        for key in keys
    }
    results = pd.DataFrame([entry for network in networks for entry in compare(network, rows, resampled)])

    stamp = utc_stamp()
    output = output_root / folder_name(run.name, seeds, run.seeds, len(ladder), commit, stamp)
    output.mkdir(parents=True, exist_ok=False)
    results.to_csv(output / "results.csv", index=False)
    per_seed.to_csv(output / "per_seed.csv", index=False)
    every_prediction.to_csv(output / "predictions.csv", index=False)
    runs = {key: stored.name for key, stored in networks.items()}
    (output / "table.md").write_text(table(results, labels, runs, DRAWS), encoding="utf-8")
    changed_code = write_code_state_archive(output, code_state)
    data_files = [data_dir / f"{kind}_{run.preprocessing.subset}.txt" for kind in ("train", "test", "RUL")]
    probe = {b.key: b.build(seeds[0]) for b in ladder}
    (output / "summary.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "finished_utc": stamp,
                "command": [Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])],
                "environment": environment(),
                "data_files": file_fingerprints(data_files),
                "uncommitted_code": changed_code,
                "source_run": {
                    "name": run.name,
                    "commit": run.summary.get("commit"),
                    "finished_utc": run.summary.get("finished_utc"),
                    "temporal_stage": run.summary.get("temporal_stage"),
                    "reported_weights": run.summary.get("reported_weights"),
                },
                "also_runs": [
                    {"key": key, "name": stored.name, "commit": stored.summary.get("commit"),
                     "temporal_stage": stored.summary.get("temporal_stage"),
                     "reported_weights": stored.summary.get("reported_weights")}
                    for key, stored in list(networks.items())[1:]
                ],
                "seeds": seeds,
                "subset": run.preprocessing.subset,
                "fitted_on": "each seed's training engines; the held-back engines unused",
                "scored_on": f"the {truth.shape[1]} test engines, last window of each",
                "predictions_clipped_to": [0.0, rul_max],
                "models": {
                    b.key: {"label": b.label, "inputs": b.inputs,
                            "estimator": repr(probe[b.key]).replace(f"random_state={seeds[0]}",
                                                                    "random_state=<seed>"),
                            "shuffled_labels": b.shuffled_labels}
                    for b in ladder
                },
                "resampling": {
                    "draws": DRAWS,
                    "generator_seed": RESAMPLING_SEED,
                    "method": "the test engines drawn with replacement, the same draw for "
                              "every seed and model; each seed's RMSE on the draw, averaged "
                              "over seeds; 2.5th and 97.5th percentiles",
                },
                "verification": {
                    "max_reload_gap_cycles": verified,
                    "prediction_tolerance": PREDICTION_TOLERANCE,
                    "rescore_tolerance": RESCORE_TOLERANCE,
                },
                "convergence_warnings": warned,
                "seconds": round(time.perf_counter() - started, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print((output / "table.md").read_text(encoding="utf-8"))
    print(f"Written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
