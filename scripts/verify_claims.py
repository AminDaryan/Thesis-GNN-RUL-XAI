#!/usr/bin/env python3
"""
Recompute the tables in docs/ASSUMPTIONS.md and check them against what is written
=================================================================================

Every numbered entry in the assumptions log quotes measurements. Those numbers
go stale silently: the architecture moves, or a default changes, and an entry
keeps asserting a figure nobody has recomputed since. That has happened here
more than once — A-03 kept recommending min-max after the default became
z-score, A-20 and F-05 described a scaling that had been replaced, and F-07
printed a count its own table contradicted.

This script recomputes what can be recomputed and prints each figure beside the
one recorded in the log, so drift is a failing check rather than something a
reader has to notice. It writes its results into ``reports/`` with the commit
that produced them, the same stamping ``scripts/train.py`` uses, so a
verification run is itself reproducible under working rule 3a.

Exit status is 1 if any claim disagrees with the log, so this can be a gate.

Examples
--------
Everything that needs no trained model — a few seconds::

    python scripts/verify_claims.py --fast

Everything, including the two checks that train a network — a few minutes::

    python scripts/verify_claims.py

One group only::

    python scripts/verify_claims.py --only graph sensors

What is deliberately not checked here
-------------------------------------
**A-01's calibration of Fig. 7.** The ceiling of 125 was established by
measuring the plateau of the paper's Fig. 7 against its own axis ticks, giving
125.2. That is a measurement on a rendered image, and automating it means
locating axis ticks by pixel search — fragile enough that a silent drift in the
answer is a real risk, which is the opposite of what this script is for. It was
done once, by hand, with the method written down in A-01. Redo it by hand.

A-08's ordering check measures the real FD002 engines, which genuinely have six
operating conditions. An earlier version of that entry used a constructed
signal with a known wear curve, which working rule 7 forbids here; the entry
now records the real-data figures this check reproduces, and keeps the
constructed ones only as history.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_gnn_rul_xai.config import PreprocessingConfig, load_config
from thesis_gnn_rul_xai.data.loading import load_subset
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.data.sensors import (
    CYCLE_COL,
    PAPER_SENSOR_COLUMNS,
    UNIT_COL,
    detect_uninformative_sensors,
    symbol_to_column,
)
from thesis_gnn_rul_xai.data.smoothing import apply_ewma
from thesis_gnn_rul_xai.graph.adjacency import (
    PHYSICS_ADJACENCY_ROWS,
    adjacency_report,
    build_physics_adjacency,
    pairs_beyond_reach,
)
from thesis_gnn_rul_xai.utils.logging_setup import configure_logging
from thesis_gnn_rul_xai.utils.paths import raw_data_dir, reports_dir


@dataclass
class Claim:
    """One recomputed figure, beside the value docs/ASSUMPTIONS.md records for it."""

    entry: str
    claim: str
    recorded: object
    measured: object
    tolerance: float = 0.0
    note: str = ""

    @property
    def ok(self) -> bool | None:
        """True, False, or None when the log records nothing to compare against."""
        if self.recorded is None:
            return None
        if isinstance(self.recorded, (int, float)) and isinstance(
            self.measured, (int, float)
        ):
            return abs(float(self.recorded) - float(self.measured)) <= self.tolerance
        return str(self.recorded) == str(self.measured)


@dataclass
class Check:
    """A group of claims recomputed together, because they share their inputs."""

    name: str
    entries: str
    needs_data: bool
    needs_model: bool
    run: object
    claims: list[Claim] = field(default_factory=list)


# --- the checks that need nothing but the package ------------------------


def check_graph() -> list[Claim]:
    """F-07's edge counts and the structural figures quoted beside them."""
    matrix, symbols = build_physics_adjacency()
    report = adjacency_report(matrix, symbols)

    def edge(a: str, b: str) -> bool:
        return PHYSICS_ADJACENCY_ROWS[a][symbols.index(b)] == 1

    required = (
        [("P30", "Ps30"), ("T30", "P30"), ("T30", "Ps30"), ("P15", "BPR"),
         ("W31", "W32")]
        + [("Nf", "NRf")]
        + [("T24", "T30"), ("T30", "T50")]
        + list(combinations(["P15", "Nf", "NRf", "BPR"], 2))
        + [("phi", x) for x in ("T30", "P30", "Ps30", "htBleed", "W31", "W32")]
        + [("W31", "W32"), ("W31", "htBleed"), ("W32", "htBleed")]
    )
    distinct = {frozenset(p) for p in required}
    missing = [f"{a}-{b}" for a, b in required if not edge(a, b)]

    return [
        Claim("F-07", "rule checks that hold",
              22, sum(edge(*p) for p in required)),
        Claim("F-07", "rule checks in total", 23, len(required)),
        Claim("F-07", "distinct edges that hold",
              19, sum(edge(*tuple(p)) for p in distinct)),
        Claim("F-07", "distinct edges in total", 20, len(distinct)),
        Claim("F-07", "the absent edge", "phi-W32", sorted(set(missing))[0]),
        Claim("F-07", "edges in the graph", 41, report["n_edges"]),
        Claim("F-07", "diameter", 4, report["diameter"]),
        Claim("F-07", "articulation points", "['T24']", str(report["articulation_points"])),
        Claim("F-07", "distance histogram",
              "{1: 41, 2: 25, 3: 21, 4: 4}", str(report["distance_histogram"])),
        Claim("F-07", "pairs beyond three layers",
              4, len(pairs_beyond_reach(matrix, symbols, n_layers=3))),
        Claim("F-07", "pairs beyond four layers",
              0, len(pairs_beyond_reach(matrix, symbols, n_layers=4))),
    ]


# --- the checks that need the NASA files ---------------------------------


def check_sensors() -> list[Claim]:
    """F-01 on P15, and A-10 on the seventh sensor the paper drops without naming."""
    data = load_subset("FD001", data_dir=raw_data_dir())
    train = data.train
    p15 = train[symbol_to_column("P15")]
    nf = train[symbol_to_column("Nf")]
    nrc = train[symbol_to_column("NRc")]
    detected = detect_uninformative_sensors(train)

    named_by_the_paper = ["T2", "P2", "epr", "farB", "Nf_dmd", "PCNfR_dmd"]
    return [
        Claim("F-01", "training rows", 20631, len(train)),
        Claim("F-01", "distinct P15 values", 2, int(p15.nunique())),
        Claim("F-01", "the two P15 levels", "[21.6, 21.61]",
              str(sorted(round(float(v), 2) for v in p15.unique()))),
        Claim("F-01", "distinct Nf values", 53, int(nf.nunique())),
        Claim("F-01", "P15 relative range", 4.6e-4,
              float((p15.max() - p15.min()) / abs(p15.mean())), tolerance=1e-5),
        Claim("F-01", "Nf relative range", 2.8e-4,
              float((nf.max() - nf.min()) / abs(nf.mean())), tolerance=1e-5),
        Claim("A-10", "sensors the paper names as removed", 6, len(named_by_the_paper)),
        Claim("A-10", "sensors measured perfectly constant",
              str(sorted(named_by_the_paper)), str(detected["constant"])),
        Claim("A-10", "NRc is constant", False, int(nrc.nunique()) <= 1),
        Claim("A-10", "distinct NRc values", 6078, int(nrc.nunique())),
    ]


def check_labels() -> list[Claim]:
    """F-02 on how many test labels the ceiling touches, and F-03 on the shortest."""
    data = load_subset("FD001", data_dir=raw_data_dir())
    labels = data.test_rul
    lengths = data.test.groupby(UNIT_COL)[CYCLE_COL].max()
    return [
        Claim("F-02", "test engines", 100, len(labels)),
        Claim("F-02", "test labels above 125", 11, int((labels > 125).sum())),
        Claim("F-02", "largest test label", 145, int(labels.max())),
        Claim("F-03", "shortest test history", 31, int(lengths.min())),
        Claim("F-03", "test engines shorter than a 30-cycle window",
              0, int((lengths < 30).sum())),
    ]


def check_ordering() -> list[Claim]:
    """A-08's claim, tested on the real six-condition FD002 rather than a built signal."""
    data = load_subset("FD002", data_dir=raw_data_dir())
    frame = data.train.copy()
    columns = list(PAPER_SENSOR_COLUMNS)

    from thesis_gnn_rul_xai.data.normalization import (
        OperatingConditionEncoder,
        SensorNormalizer,
    )

    encoder = OperatingConditionEncoder().fit(frame)
    conditions = encoder.n_detected

    def wear_agreement(processed: pd.DataFrame) -> float:
        """Mean |correlation| of each scaled sensor with the remaining-life label."""
        rul = processed["rul"].to_numpy(dtype=float)
        scores = []
        for column in columns:
            values = processed[column].to_numpy(dtype=float)
            if values.std() <= 0:
                continue
            scores.append(abs(float(np.corrcoef(values, rul)[0, 1])))
        return float(np.mean(scores))

    def scale(f: pd.DataFrame) -> pd.DataFrame:
        f = encoder.transform(f)
        return SensorNormalizer(
            method="minmax_per_condition", columns=columns
        ).fit(f).transform(f)

    smoothed_then_scaled = scale(apply_ewma(frame, columns, alpha=0.5))
    scaled_then_smoothed = apply_ewma(scale(frame), columns, alpha=0.5)
    scaled_only = scale(frame)

    return [
        Claim("A-08", "operating conditions detected in FD002", 6, conditions),
        Claim("A-08", "smoothing then scaling, the paper's order (Fig. 1)",
              0.054, round(wear_agreement(smoothed_then_scaled), 3), tolerance=0.01),
        Claim("A-08", "scaling then smoothing, condition-aware",
              0.591, round(wear_agreement(scaled_then_smoothed), 3), tolerance=0.01),
        Claim("A-08", "scaling only, no smoothing",
              0.485, round(wear_agreement(scaled_only), 3), tolerance=0.01),
    ]


# --- the checks that need a trained network ------------------------------


def _trained_model(config: PreprocessingConfig):
    """Train one network on the given settings, seed 42, and return it with its data."""
    import torch

    from thesis_gnn_rul_xai.models import ESTGNN
    from thesis_gnn_rul_xai.training import TrainingConfig, train

    prepared = run_preprocessing(config, data_dir=raw_data_dir(), keep_frames=True)
    torch.manual_seed(42)
    model = ESTGNN(
        prepared.adjacency_normalized,
        in_features=prepared.train.features.shape[2],
        rul_max=config.rul_max,
    )
    train(model, prepared.train, None, TrainingConfig(seed=42, device="cpu"))
    return model, prepared


def check_baselines(config: PreprocessingConfig) -> list[Claim]:
    """A-20's table: what the choice of stand-in does to the ranking."""
    from scipy.stats import spearmanr

    from thesis_gnn_rul_xai.explain import mean_window_baseline, shap_values

    model, prepared = _trained_model(config)
    symbols = list(prepared.sensor_symbols)
    windows = prepared.test.features[:20]

    mean_window = mean_window_baseline(prepared.train)
    baselines = {
        "mean window": mean_window,
        "median window": np.median(prepared.train.features, axis=0).astype(np.float32),
        "all zeros": np.zeros_like(mean_window),
    }
    importance = {
        name: np.abs(shap_values(model, windows, base)).mean(axis=0)
        for name, base in baselines.items()
    }
    top4 = {
        name: [symbols[i] for i in np.argsort(-imp)[:4]]
        for name, imp in importance.items()
    }
    reference = importance["mean window"]

    return [
        Claim("A-20", "top four, mean window",
              "['T50', 'Nc', 'W31', 'T24']", str(top4["mean window"])),
        Claim("A-20", "top four, median window",
              "['T50', 'T24', 'W31', 'Nc']", str(top4["median window"])),
        Claim("A-20", "top four, all zeros",
              "['T50', 'Nc', 'W31', 'T24']", str(top4["all zeros"])),
        Claim("A-20", "Spearman, median window vs mean window", 0.952,
              round(float(spearmanr(reference, importance["median window"]).statistic), 3),
              tolerance=0.02),
        Claim("A-20", "Spearman, all zeros vs mean window", 0.996,
              round(float(spearmanr(reference, importance["all zeros"]).statistic), 3),
              tolerance=0.02),
    ]


def check_p15(config: PreprocessingConfig) -> list[Claim]:
    """F-05: where P15 ranks, what it is worth, and the quantisation behind it."""
    from thesis_gnn_rul_xai.explain import dynamic_shap, mean_window_baseline

    model, prepared = _trained_model(config)
    symbols = list(prepared.sensor_symbols)
    baseline = mean_window_baseline(prepared.train)
    p15 = symbols.index("P15")

    claims: list[Claim] = []
    recorded_rank = {5: 14, 25: 14, 82: 14}
    recorded_max = {5: 9.449, 25: 1.769, 82: 9.526}
    for unit in (5, 25, 82):
        table = dynamic_shap(model, prepared.train, baseline, unit=unit, symbols=symbols)
        values = table[symbols].to_numpy()
        mean_abs = np.abs(values).mean(axis=0)
        rank = int(np.where(np.argsort(-mean_abs) == p15)[0][0]) + 1
        claims.append(Claim("F-05", f"engine {unit}: P15 rank of 14",
                            recorded_rank[unit], rank))
        claims.append(Claim("F-05", f"engine {unit}: largest |phi| for P15",
                            recorded_max[unit],
                            round(float(np.abs(values[:, p15]).max()), 3),
                            tolerance=0.2))

    raw = prepared.raw.train
    column = symbol_to_column("P15")
    recorded_changes = {5: 34, 25: 0, 82: 20}
    for unit in (5, 25, 82):
        series = raw.loc[raw[UNIT_COL] == unit, column].to_numpy()
        claims.append(Claim("F-05", f"engine {unit}: times P15 changes level",
                            recorded_changes[unit], int((np.diff(series) != 0).sum())))

    changes = np.array([
        int((np.diff(g[column].to_numpy()) != 0).sum())
        for _, g in raw.groupby(UNIT_COL)
    ])
    claims.append(Claim("F-05", "engines whose P15 never changes",
                        38, int((changes == 0).sum())))
    claims.append(Claim("F-05", "median level changes across engines",
                        2, int(np.median(changes))))

    spread = {s: float(prepared.train.features[:, symbols.index(s), :].std())
              for s in ("P15", "Nc", "T50", "T24")}
    claims.append(Claim("F-05", "scaled spread of P15", 0.862,
                        round(spread["P15"], 3), tolerance=0.02))
    claims.append(Claim("F-05", "scaled spread of Nc", 1.053,
                        round(spread["Nc"], 3), tolerance=0.02))
    return claims


CHECKS = [
    Check("graph", "F-07", False, False, check_graph),
    Check("sensors", "F-01, A-10", True, False, check_sensors),
    Check("labels", "F-02, F-03", True, False, check_labels),
    Check("ordering", "A-08", True, False, check_ordering),
    Check("baselines", "A-20", True, True, check_baselines),
    Check("p15", "F-05", True, True, check_p15),
]


def _commit_stamp() -> str:
    """The short commit, with -dirty when a tracked file has been modified."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    # Untracked files do not change what the code computes, so they do not
    # make a run unreproducible and are not counted as dirty here.
    modified = git("status", "--porcelain", "--untracked-files=no")
    return commit + ("-dirty" if modified else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        choices=[c.name for c in CHECKS],
                        help="run only these checks")
    parser.add_argument("--fast", action="store_true",
                        help="skip the checks that train a network")
    parser.add_argument("--config", type=Path, default=None,
                        help="settings to measure under; defaults to config/fd001.yaml")
    parser.add_argument("--output", type=Path, default=None,
                        help="where to write the result; defaults under reports/")
    parser.add_argument("--quiet", action="store_true", help="warnings only")
    arguments = parser.parse_args(argv)

    import logging
    configure_logging(logging.WARNING if arguments.quiet else logging.INFO)

    root = Path(__file__).resolve().parents[1]
    config_path = arguments.config or (root / "config" / "fd001.yaml")
    config = load_config(config_path)

    selected = [c for c in CHECKS if not arguments.only or c.name in arguments.only]
    if arguments.fast:
        selected = [c for c in selected if not c.needs_model]

    commit = _commit_stamp()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"Verifying against {config_path.name} "
          f"({config.normalization}, {config.adjacency_normalization}, "
          f"validation_fraction={config.validation_fraction})")
    print(f"commit {commit}\n")

    rows: list[dict[str, object]] = []
    for check in selected:
        print(f"--- {check.name}  ({check.entries})")
        try:
            claims = (check.run(config) if check.needs_model else check.run())
        except FileNotFoundError as exc:
            print(f"    skipped: {exc}\n")
            continue
        for claim in claims:
            mark = {True: "ok  ", False: "FAIL", None: "--  "}[claim.ok]
            print(f"    {mark} {claim.entry:5s} {claim.claim:48s} "
                  f"recorded {claim.recorded!s:28s} measured {claim.measured}")
            rows.append({
                "check": check.name, "entry": claim.entry, "claim": claim.claim,
                "recorded": claim.recorded, "measured": claim.measured,
                "ok": claim.ok, "note": claim.note,
            })
        print()

    table = pd.DataFrame(rows)
    failed = int((table["ok"] == False).sum()) if len(table) else 0  # noqa: E712
    checked = int(table["ok"].notna().sum()) if len(table) else 0
    print(f"{checked} claim(s) checked against docs/ASSUMPTIONS.md, {failed} disagree.")
    if len(table) - checked:
        print(f"{len(table) - checked} figure(s) reported with nothing recorded to "
              "compare against.")

    output = arguments.output or (reports_dir() / f"verification_{commit}_{stamp}")
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "claims.csv", index=False)
    (output / "summary.json").write_text(json.dumps({
        "commit": commit,
        "finished_utc": stamp,
        "config": str(config_path),
        "checks": [c.name for c in selected],
        "claims_checked": checked,
        "claims_failed": failed,
    }, indent=2), encoding="utf-8")
    print(f"\nWritten to {output}")

    if failed:
        print("\nA disagreement means either the code changed or the log is stale. "
              "Find out which before quoting either.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
