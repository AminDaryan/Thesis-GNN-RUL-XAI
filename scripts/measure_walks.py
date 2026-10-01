#!/usr/bin/env python3
"""
Measure walk explanations over every saved seed of a stored run
===============================================================

Loads each seed's trained network from a stored run, rebuilds the data it saw,
and explains the last window of every test engine with walks through the sensor
graph (``explain/walks.py``, assumption A-29). Beside each trained network it
explains an untrained one of the same shape, started from the same seed, because
much of what a walk share says is fixed by the graph's own weights before any
training: a sensor's connection to itself weighs one over its number of
connections plus one, so walks that move between sensors carry most of the
score whatever the network learned. A number is about training only where the
trained and the untrained network differ.

It also removes each connection of the raw graph in turn, rescales, and reruns
the same network (``explain/connections.py``, assumption A-30), so that what
flowed along a connection can be set beside what removing it does.

The untrained network is built exactly as ``scripts/train.py`` builds each
seed's network before training it -- the same seed, then the same builder -- so
it is that network's starting point, and comparing the two shows what training
changed.

What is written
---------------
Under ``reports/<run>/walks/<commit>_<time>/``, never under ``explanations/``,
which the playground reads as one number per sensor:

``per_seed.csv``
    One row per seed and network (trained or untrained): the median and mean of
    each measure over the test windows.
``connections.csv``
    One row per seed and connection: the average size of what flowed along it,
    summed over the graph layers and both directions, at each gamma, and the
    average size of what removing it does, in cycles.
``summary.json``
    The commit, the settings, and each measure's mean and spread over seeds.
``code_state.zip``
    When the code had uncommitted changes, the list of changed files, the
    changes as a patch and every new file whole. It is one zip file rather than
    a folder because the folder's paths would pass Windows's 260-character
    limit under the longest run names.

Each measure:

``stays_share``
    The walks that never leave their starting sensor, as a share of every
    walk's size, ignoring signs. Sensors the graph cannot tell apart count as
    one sensor here, and what flows into them at the last graph layer is split
    equally among them, because how it really splits follows the random start
    (finding F-18).
``constant_share``
    The size of the dense layers' constant against the size of the constant
    plus the walks' sum.
``cancellation``
    Every walk's size added up, divided by the size of the walks' sum.
``removal_agreement``
    Spearman's rank correlation, over the 41 connections, between the average
    size of what flowed along a connection and the average size of what
    removing it does.
``gamma_agreement_*``
    Spearman's rank correlation, over the 41 connections, between what flowed
    along each at the authors' setting of gamma and at the other setting.
``aufc_<task>_<ranking>``
    The GNN-LRP paper's own check of an explanation (its Section 4.1; A-33):
    the mean, over the test windows, of the area under the curve the score
    traces as sensors are added to an empty graph (``activation``; higher is
    better) or removed from the full one (``pruning``; lower is better), in the
    order each ranking gives.
``same_rows_drift``
    Trained networks only. For every pair of sensors with identical rows in the
    rescaled graph, how far the difference between their blocks of ``fc1``
    weights moved during training: the largest change over all such pairs.
``control_drift``
    The same for the first two sensors that are in no such group, for scale.
``same_rows_block_moved``
    The smallest distance any of those sensors' blocks of ``fc1`` weights moved
    during training, to show that the weights themselves did move.

Example
-------
::

    python scripts/measure_walks.py --run reports/<run directory>
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_gnn_rul_xai.utils.logging_setup import configure_logging
from thesis_gnn_rul_xai.utils.provenance import (
    capture_code_state,
    current_commit,
    utc_stamp,
    write_code_state_archive,
)

# The authors' own setting comes first and every measure is taken under it;
# None asks walk_relevance for their default schedule, 0 at the first graph
# layer. Gamma 0 everywhere is their simpler baseline, GNN-GI.
# ASSUMPTION A-29: the other two settings, tried only to see how far a ranking
# moves with gamma, are ours, one falling and one flat, chosen before any
# result was seen.
GAMMAS: dict[str, tuple[float, float, float] | None] = {
    "authors": None,
    "0": (0.0, 0.0, 0.0),
    "0_0.5_0.25": (0.0, 0.5, 0.25),
    "0_1_1": (0.0, 1.0, 1.0),
}


def _drift(trained, start, symbols) -> dict[str, float]:
    """How far training moved the fc1 weights of sensors the graph cannot tell apart."""
    from thesis_gnn_rul_xai.explain import indistinguishable_sensors

    names = list(symbols)
    shape = (trained.fc1.out_features, trained.n_nodes, -1)
    after = trained.fc1.weight.detach().double().reshape(shape)
    before = start.fc1.weight.detach().double().reshape(shape)

    def change(a: int, b: int) -> float:
        return float(((after[:, a] - after[:, b]) - (before[:, a] - before[:, b])).abs().max())

    groups = indistinguishable_sensors(trained, names)
    grouped = {names.index(s) for group in groups for s in group}
    pairs = [(names.index(g[0]), names.index(s)) for g in groups for s in g[1:]]
    free = [i for i in range(len(names)) if i not in grouped]
    return {
        "same_rows_drift": max((change(a, b) for a, b in pairs), default=float("nan")),
        "control_drift": change(free[0], free[1]) if len(free) >= 2 else float("nan"),
        "same_rows_block_moved": min(
            (float((after[:, i] - before[:, i]).abs().max()) for i in grouped),
            default=float("nan"),
        ),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, ties given their average rank."""
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def _connection_flow(result, raw: np.ndarray, symbols, groups) -> np.ndarray:
    """What flowed along each connection, both directions and every layer, per window."""
    from thesis_gnn_rul_xai.explain import connection_relevance, connections_of

    passed = connection_relevance(result, symbols, pool=groups)
    return np.stack([passed[:, i, j] + passed[:, j, i] for i, j in connections_of(raw)], axis=1)


def _measures(model, features, prepared) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Every measure for one network over the given windows."""
    from thesis_gnn_rul_xai.explain import (
        aufc,
        connection_removal,
        crossing_relevance,
        indistinguishable_sensors,
        node_flipping,
        walk_relevance,
    )

    raw = prepared.adjacency_raw
    symbols = prepared.sensor_symbols
    # Sensors the graph cannot tell apart count as one wherever a walk ends
    # (finding F-18): where it ends among them follows the random start.
    groups = indistinguishable_sensors(model, symbols)
    flows: dict[str, np.ndarray] = {}
    results = {}
    for name, gamma in GAMMAS.items():
        results[name] = walk_relevance(model, features, gamma=gamma)
        flows[name] = np.abs(_connection_flow(results[name], raw, symbols, groups)).mean(axis=0)
    result = results["authors"]
    table = crossing_relevance(result, symbols, pool=groups)
    stay_size = table["absolute_stays"]
    walks_sum = table["stays"] + table["moves"]
    removal = connection_removal(
        model, features, raw, prepared.sensor_symbols,
        method=prepared.config.adjacency_normalization,
    )
    flows["removal"] = removal.abs().mean(axis=0).to_numpy()
    per_window = {
        "stays_share": stay_size / table["absolute_total"],
        "constant_share": np.abs(table["constant"]) / (np.abs(table["constant"]) + np.abs(walks_sum)),
        "cancellation": table["absolute_total"] / np.abs(walks_sum),
    }
    row: dict[str, float] = {}
    for key, values in per_window.items():
        row[f"{key}_median"] = float(np.median(values))
        row[f"{key}_mean"] = float(np.mean(values))
    row["removal_agreement"] = _spearman(flows["authors"], flows["removal"])
    for name in GAMMAS:
        if name != "authors":
            row[f"gamma_agreement_{name}"] = _spearman(flows["authors"], flows[name])
    row["score_at_zero"] = result.score_at_zero
    # The authors' own check (their Section 4.1), against their GNN-GI baseline,
    # node-level rankings and random orders.
    checked = aufc(node_flipping(model, features, {"GNN-LRP": result, "GNN-GI": results["0"]}))
    for entry in checked.itertuples():
        row[f"aufc_{entry.task}_{entry.ranking}"] = float(entry.mean)
    return row, flows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure walk explanations over every saved seed of a stored run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run", required=True, help="The run directory, or its name inside reports/")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true", help="Warnings only")
    arguments = parser.parse_args()
    configure_logging(logging.WARNING if arguments.quiet else logging.INFO)

    code_state = capture_code_state()
    import torch

    from thesis_gnn_rul_xai.explain import connections_of
    from thesis_gnn_rul_xai.models import build_model
    from thesis_gnn_rul_xai.training import load_seed, read_run

    try:
        stored = read_run(arguments.run)
    except (FileNotFoundError, ValueError) as refused:
        raise SystemExit(str(refused)) from refused
    if not stored.has_weights():
        raise SystemExit(f"{stored.name} saved no weights, so its networks cannot be explained.")

    rows, connection_rows = [], []
    flows_by_seed: dict[int, pd.Series] = {}
    for seed in stored.seeds:
        trained, prepared = load_seed(stored, seed, data_dir=arguments.data_dir)
        features = prepared.test.features[np.argsort(prepared.test.units)]
        torch.manual_seed(seed)
        untrained = build_model(
            prepared.adjacency_normalized,
            stored.model_config,
            window_size=prepared.config.window_size,
            rul_max=prepared.config.rul_max,
        ).eval()
        names = [f"{prepared.sensor_symbols[i]}-{prepared.sensor_symbols[j]}"
                 for i, j in connections_of(prepared.adjacency_raw)]
        for label, network in (("trained", trained), ("untrained", untrained)):
            row, flows = _measures(network, features, prepared)
            if label == "trained":
                row.update(_drift(trained, untrained, prepared.sensor_symbols))
            rows.append({"seed": seed, "network": label, "windows": len(features), **row})
            for index, name in enumerate(names):
                connection_rows.append({
                    "seed": seed, "network": label, "connection": name,
                    **{f"flow_gamma_{g}": float(flows[g][index]) for g in GAMMAS},
                    "removal_cycles": float(flows["removal"][index]),
                })
            if label == "trained":
                flows_by_seed[seed] = pd.Series(flows["authors"], index=names)

    per_seed = pd.DataFrame(rows)
    connections = pd.DataFrame(connection_rows)
    seeds = sorted(flows_by_seed)
    # A graph built from each seed's own training engines can join different
    # pairs in different seeds, so two seeds are compared on the connections
    # they share, matched by name, never by position.
    stability, shared = [], []
    for a, b in itertools.combinations(seeds, 2):
        common = flows_by_seed[a].index.intersection(flows_by_seed[b].index)
        shared.append(len(common))
        stability.append(_spearman(
            flows_by_seed[a][common].to_numpy(), flows_by_seed[b][common].to_numpy()
        ))

    summary_measures = {}
    for label in ("trained", "untrained"):
        part = per_seed[per_seed["network"] == label]
        summary_measures[label] = {
            column: {
                "mean": float(part[column].mean()),
                "std": float(part[column].std(ddof=1)),
                "min": float(part[column].min()),
                "max": float(part[column].max()),
            }
            for column in part.columns
            if column not in ("seed", "network", "windows") and part[column].notna().any()
        }

    commit = current_commit()
    stamp = utc_stamp()
    output = stored.path / "walks" / f"{commit}_{stamp}"
    output.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output / "per_seed.csv", index=False)
    connections.to_csv(output / "connections.csv", index=False)
    changed = write_code_state_archive(output, code_state)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "finished_utc": stamp,
                "run": stored.name,
                "seeds": seeds,
                "windows": "the last window of every test engine",
                "gammas": {name: (None if value is None else list(value))
                           for name, value in GAMMAS.items()},
                "uncommitted_files": changed,
                "measures": summary_measures,
                "connection_ranking_agreement_between_seeds": {
                    "pairs": len(stability),
                    "connections_shared_min": int(min(shared)),
                    "min": float(min(stability)),
                    "median": float(np.median(stability)),
                    "max": float(max(stability)),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
