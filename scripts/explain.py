#!/usr/bin/env python3
"""
Explain a stored training run
=============================

Loads the weights one seed of a finished run saved, rebuilds the data it was
trained on from the settings stored beside it, and works out how much each of
the 14 sensors moved its predictions. The answer is written into the run's own
folder, under ``explanations/``, stamped with the commit that computed it.

Three methods are available, and only the first is the paper's:

``shap``
    Exact Shapley values, Equations 14 and 15 of Wen et al. Every one of the
    16,384 subsets of the sensors is evaluated for every window. A sensor left
    out of a subset is replaced by its average training window (A-20).
``occlusion``
    Not in the paper. Each sensor in turn is replaced by the same baseline
    while the other 13 stay, and the change in the prediction is recorded:
    15 evaluations per window instead of 16,384. It is the one term of the
    Shapley average in which the sensor arrives last; the Shapley value
    averages what the sensor adds over every point at which it could arrive.
    The two agree whenever that last amount equals the average: always for a
    sensor that adds the same amount wherever it arrives, and also for one
    whose amounts differ but happen to average to the last one. Agreement
    therefore does not show that a sensor acts independently of the others.
    (Until 2026-10-02 this said the two agree only when the sensors do not
    interact. That guarantees agreement, but is not the only way to get it.
    Found by the review of the playground's information cards on 2026-09-26.)
``walks``
    Not in the paper. GNN-LRP (Schnake et al., 2022; A-29): the network's
    weights are read and each window's score before the sigmoid -- the number
    the network's last step turns into a prediction -- is split exactly into
    one share per walk through the sensor graph, plus a part that belongs to no
    sensor, with the authors' own setting of its rule. Nothing is replaced. The
    shares are parts of that score, not cycles. The authors' own check of an
    explanation, node flipping (A-33), is run on every window beside it.
    ``--gamma`` names another setting of the rule: ``gnn-gi`` (gamma 0 at
    every graph layer, the authors' simpler baseline), ``falling`` (0, then
    0.5 halving at each later layer) or ``flat`` (0, then 1). Every setting
    gives the first graph layer 0, whose inputs can be negative. For three
    graph layers the last two are (0, 0.5, 0.25) and (0, 1, 1), the settings
    A-29 compares with the authors' when it measures how far a ranking of
    connections moves with gamma.

and two things can be explained:

``engine``
    Every window of one engine's life, in order: the "dynamic" explanation of
    Section 4.4. The test set keeps only the last window of each engine, so the
    engine comes from the held-back engines when there are any and from the
    training engines otherwise.
``test``
    The last window of every test engine, once each.

What is written
---------------
Every method first reads the commit and, when the code has uncommitted
changes, keeps a copy of them in ``code_state.zip``, so the result names the
code that computed it.

``values.csv`` holds one row per window: the engine, the cycle the window ends
at, the label (the remaining life capped at 125 cycles wherever the run caps
it; assumptions A-01 and A-02) and the predicted remaining life, the
prediction at the baseline,
one column per sensor with its value in cycles, and one column
``reading_<sensor>`` per sensor holding what that sensor was reading at the
window's last cycle, so that a chart can colour each value by whether the
sensor read high or low, as the paper's Fig. 9 does. That reading is the one
the model was given: after the noise filter and the rescaling, not the raw
number. ``summary.json`` holds the settings, the
time taken, the largest gap in Equation 14's sum, and every sensor's average
size.

A ``walks`` explanation writes different columns, on purpose, so that nothing
written for sensor values in cycles can misread it: the engine, the cycle, the
label and the predicted remaining life, then ``score`` (the score before the
sigmoid), ``constant`` (the part no sensor owns), ``stays`` and ``moves`` (the
walks that stay within their starting sensor's group and the rest, sensors the
graph cannot tell apart counting as one group), ``absolute_total`` (every
walk's share added up ignoring its sign), one ``start_<sensor>`` per sensor (the
walks starting there, added up), one ``link_<from>_<to>`` per connection the
graph has, in each direction and including each sensor's connection to itself
(what passed along it, over every graph layer), and the ``reading_<sensor>``
columns. ``top_walks.csv`` holds the ten walks with the largest average share.
``flipping.csv`` holds, for every window, both of the paper's flipping tasks
and every ranking checked -- the walks and their starting sensors, at the
authors' setting and at their simpler baseline, at the setting chosen when it
is neither, and random orders -- the order the sensors were flipped in and the
curve the paper plots, one column per step. Its ``summary.json`` holds no
``mean_abs`` and no ``baseline``; it holds the sensors, the setting chosen
(``gamma_setting``) and its gamma at each graph layer (``gamma``), the largest
miss of walks plus constant against the score, and the area under each
flipping curve averaged over the windows.

Examples
--------
The paper's dynamic explanation, for the engine with the longest history::

    python scripts/explain.py --run reports/<run directory> --method shap --scope engine

A quick first look across the test engines::

    python scripts/explain.py --run reports/<run directory> --method occlusion --scope test

The same walks without the rule's extra weight, to set beside the default::

    python scripts/explain.py --run reports/<run directory> --method walks --scope test --gamma gnn-gi
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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

METHODS = ("shap", "occlusion", "walks")
SCOPES = ("engine", "test")

# The settings of the walk rule's gamma, by the names --gamma and the
# playground take, the authors' own first; gamma_values turns one into numbers.
GAMMA_SETTINGS = ("authors", "gnn-gi", "falling", "flat")

# What each setting's ranking is called in flipping.csv: the GNN-LRP paper's
# own names for its method and its simpler baseline, then the method's name
# with the setting's for the other two.
FLIPPING_NAMES = {"authors": "GNN-LRP", "gnn-gi": "GNN-GI",
                  "falling": "GNN-LRP falling", "flat": "GNN-LRP flat"}


def gamma_values(setting: str, n_layers: int) -> list[float]:
    """One gamma per graph layer for a named setting, 0 at the first whatever the name."""
    from thesis_gnn_rul_xai.explain import paper_gammas

    if setting not in GAMMA_SETTINGS:
        raise ValueError(f"gamma must be one of {list(GAMMA_SETTINGS)}, got {setting!r}.")
    if n_layers < 1:
        raise ValueError(f"n_layers must be at least 1, got {n_layers}.")
    later = n_layers - 1
    if setting == "authors":
        values = paper_gammas(n_layers)
    elif setting == "gnn-gi":
        values = [0.0] * n_layers
    elif setting == "falling":
        # ASSUMPTION A-29: 0.5 at the second graph layer, halving at each later
        # one: the halving the GNN-LRP paper uses for VGG-16 (its Section 5.3),
        # started here at the second graph layer. Its numbers for this network
        # are ours, like "flat" below (1 at every layer after the first), and
        # both are offered only to see how far the walks move with gamma. For
        # three graph layers they are the settings scripts/measure_walks.py
        # compares with the authors'; for any other number of layers this
        # halving and that constant 1 are how we carry them over, not a
        # measured choice.
        values = [0.0] + [0.5**step for step in range(1, later + 1)]
    else:
        values = [0.0] + [1.0] * later
    # ASSUMPTION A-29: gamma is 0 at the first graph layer whatever the setting.
    # The rule's derivation needs every number entering a layer to be at least 0,
    # and this network's first graph layer receives rescaled readings, which are
    # negative below their training average.
    values[0] = 0.0
    return values


def walk_settings(model, features: np.ndarray, setting: str) -> dict:
    """The walks under the authors' setting, their baseline and the one chosen, each once."""
    from thesis_gnn_rul_xai.explain import walk_relevance

    n_layers = len(model.graph_layers)
    # The authors' own check sets their method beside their simpler baseline,
    # so both are always computed; a third setting is checked beside them.
    names = dict.fromkeys((GAMMA_SETTINGS[0], "gnn-gi", setting))
    return {name: walk_relevance(model, features, gamma=gamma_values(name, n_layers))
            for name in names}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explain the predictions of one seed of a stored run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run", required=True,
                        help="The run directory, or its name inside reports/")
    parser.add_argument("--seed", type=int, default=None,
                        help="Which seed's weights to explain (default: the first)")
    parser.add_argument("--method", choices=METHODS, default="shap")
    parser.add_argument("--scope", choices=SCOPES, default="engine")
    parser.add_argument("--engine", type=int, default=None,
                        help="With --scope engine: the engine number (default: "
                             "the one with the most windows)")
    parser.add_argument("--every", type=int, default=1,
                        help="With --scope engine: explain every n-th window")
    parser.add_argument("--gamma", choices=GAMMA_SETTINGS, default=None,
                        help="With --method walks: the setting of the rule's gamma "
                             "(default: authors, their own)")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--progress", type=Path, default=None,
                        help="Append one JSON line per event to this file")
    parser.add_argument("--quiet", action="store_true", help="Warnings only")
    arguments = parser.parse_args()
    if arguments.every < 1:
        raise SystemExit("--every must be at least 1.")
    if arguments.gamma is not None and arguments.method != "walks":
        raise SystemExit("--gamma is a setting of --method walks only.")
    if arguments.method == "walks" and arguments.gamma is None:
        arguments.gamma = GAMMA_SETTINGS[0]

    configure_logging(logging.WARNING if arguments.quiet else logging.INFO)

    # Read before anything runs, so the stamp names the code that computes the
    # result rather than whatever is on disk when it finishes.
    code_state = capture_code_state()
    commit = current_commit()

    import torch

    from thesis_gnn_rul_xai.explain import (
        mean_window_baseline,
        occlusion_values,
        shap_values,
    )
    from thesis_gnn_rul_xai.training import load_seed, read_run, resolve_device

    def progress(event: str, **fields: object) -> None:
        if arguments.progress is None:
            return
        with arguments.progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, **fields}, default=float) + "\n")

    try:
        stored = read_run(arguments.run)
        seed = arguments.seed if arguments.seed is not None else stored.preprocessing.seed
        model, prepared = load_seed(stored, seed, data_dir=arguments.data_dir)
    except (FileNotFoundError, ValueError) as refused:
        raise SystemExit(str(refused)) from refused
    run = stored.path

    device = resolve_device(arguments.device)
    model.to(device).eval()

    if arguments.scope == "test":
        source, source_name = prepared.test, "test"
        order = np.argsort(source.units)
        unit = None
    else:
        held_back = prepared.validation is not None and len(prepared.validation) > 0
        source = prepared.validation if held_back else prepared.train
        source_name = "validation" if held_back else "training"
        units, counts = np.unique(source.units, return_counts=True)
        unit = arguments.engine if arguments.engine is not None else int(units[np.argmax(counts)])
        chosen = np.flatnonzero(source.units == unit)
        if chosen.size == 0:
            raise SystemExit(
                f"Engine {unit} is not among the {source_name} engines of this "
                f"seed's split. Some that are: {units[:12].tolist()}."
            )
        order = chosen[np.argsort(source.end_cycles[chosen])][:: arguments.every]

    features = source.features[order]
    symbols = list(prepared.sensor_symbols)
    progress("start", windows=len(order), method=arguments.method, scope=arguments.scope)
    started = time.perf_counter()

    if arguments.method == "walks":
        return _explain_walks(
            arguments, model, source, source_name, order, unit, seed, run,
            features, symbols, started, progress, code_state, commit,
        )

    # ASSUMPTION A-20: a sensor left out is replaced by its average window over
    # the training engines. Equation 15 needs a prediction without a sensor and
    # the paper never says what stands in for it. Walks replace nothing, so they
    # never reach this line.
    baseline = mean_window_baseline(prepared.train)

    if arguments.method == "shap":
        values = np.empty((len(order), features.shape[1]))
        for index in range(len(order)):
            values[index] = shap_values(model, features[index], baseline, device=device)
            progress("window", done=index + 1, total=len(order),
                     seconds=time.perf_counter() - started)
    elif arguments.method == "occlusion":
        values = occlusion_values(model, features, baseline, device=device)
        progress("window", done=len(order), total=len(order),
                 seconds=time.perf_counter() - started)

    with torch.no_grad():
        predicted = model.predict_cycles(
            torch.from_numpy(features.astype(np.float32)).to(device)
        ).cpu().numpy().astype(np.float64)
        base = float(
            model.predict_cycles(
                torch.from_numpy(baseline.astype(np.float32)[None]).to(device)
            ).cpu().numpy()[0]
        )
    seconds = time.perf_counter() - started

    table = pd.DataFrame(values, columns=symbols)
    table.insert(0, "baseline", base)
    table.insert(0, "predicted", predicted)
    table.insert(0, "rul", source.targets[order])
    table.insert(0, "end_cycle", source.end_cycles[order])
    table.insert(0, "unit", source.units[order])
    # What each sensor was reading at the window's last cycle, as the model was
    # given it: smoothed, then rescaled. The paper's Fig. 9 colours every
    # Shapley value by whether the sensor read high or low, and a chart cannot
    # do that from the values alone.
    for index, symbol in enumerate(symbols):
        table[f"reading_{symbol}"] = features[:, index, -1]

    # Equation 14's promise, checked: the Shapley values of one window add up
    # to its prediction minus the prediction at the baseline. Occlusion makes
    # no such promise, and the gap says by how much it misses.
    gap = np.abs(values.sum(axis=1) - (predicted - base))

    output = _output_folder(arguments, run, unit, seed, commit)
    write_code_state_archive(output, code_state)
    table.to_csv(output / "values.csv", index=False)
    ranking = table[symbols].abs().mean().sort_values(ascending=False)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "finished_utc": output.name.rsplit("_", 1)[1],
                "run": run.name,
                "seed": seed,
                "method": arguments.method,
                "scope": arguments.scope,
                "source": source_name,
                "unit": unit,
                "every": arguments.every,
                "windows": len(order),
                "baseline": "average training window of each sensor (A-20)",
                "baseline_prediction": base,
                "sum_gap_max": float(gap.max()),
                "sum_gap_mean": float(gap.mean()),
                "seconds": round(seconds, 1),
                "mean_abs": {symbol: float(v) for symbol, v in ranking.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    progress("done", directory=output.name, run=run.name)
    print(f"{arguments.method} over {len(order)} window(s) in {seconds:.1f}s; "
          f"largest gap between the values' sum and the prediction's distance "
          f"from the baseline: {gap.max():.2e} cycles.")
    print(f"Saved to {output}")
    return 0


def _output_folder(arguments, run: Path, unit, seed: int, commit: str) -> Path:
    """Make the explanation's own folder inside the run, named by what and when."""
    name = (
        f"{arguments.method}_{arguments.scope}"
        + (f"_unit{unit}" if unit is not None else "")
        + (f"_every{arguments.every}" if arguments.every > 1 else "")
        + (f"_gamma-{arguments.gamma}"
           if arguments.gamma not in (None, GAMMA_SETTINGS[0]) else "")
        + f"_seed{seed}_{commit}_{utc_stamp()}"
    )
    output = run / "explanations" / name
    output.mkdir(parents=True, exist_ok=True)
    return output


def _explain_walks(
    arguments, model, source, source_name, order, unit, seed, run,
    features, symbols, started, progress, code_state, commit,
) -> int:
    """Split every window's score into walks and write what the playground draws."""
    from thesis_gnn_rul_xai.explain import (
        aufc,
        connection_relevance,
        conservation_error,
        crossing_relevance,
        indistinguishable_sensors,
        node_flipping,
        node_relevance,
        top_walks,
    )

    try:
        results = walk_settings(model, features, arguments.gamma)
    except (ValueError, RuntimeError, TypeError) as refused:
        raise SystemExit(str(refused)) from refused
    result = results[arguments.gamma]
    progress("window", done=len(order), total=len(order),
             seconds=time.perf_counter() - started)
    groups = indistinguishable_sensors(model, symbols)
    starts = node_relevance(result)
    links = connection_relevance(result, symbols, pool=groups)
    parts = crossing_relevance(result, symbols, pool=groups)
    top = top_walks(result, symbols, count=10, pool=groups)
    flipping = node_flipping(
        model, features, {FLIPPING_NAMES[name]: walks for name, walks in results.items()}
    )
    checked = aufc(flipping)
    seconds = time.perf_counter() - started

    columns = {
        "unit": source.units[order],
        "end_cycle": source.end_cycles[order],
        "rul": source.targets[order],
        "predicted": result.predicted,
        "score": result.score,
        "constant": result.constant,
        "stays": parts["stays"].to_numpy(),
        "moves": parts["moves"].to_numpy(),
        "absolute_total": parts["absolute_total"].to_numpy(),
    }
    for index, symbol in enumerate(symbols):
        columns[f"start_{symbol}"] = starts[:, index]
    # One column per connection the graph has, in each direction: axis 1 of
    # links sends and axis 2 receives, the transpose of the graph's own rows.
    allowed = model.adjacency.detach().cpu().numpy().T != 0
    for sender, receiver in zip(*np.nonzero(allowed), strict=True):
        columns[f"link_{symbols[sender]}_{symbols[receiver]}"] = links[:, sender, receiver]
    for index, symbol in enumerate(symbols):
        columns[f"reading_{symbol}"] = features[:, index, -1]
    table = pd.DataFrame(columns)

    walks_sum = np.abs(parts["stays"] + parts["moves"]).to_numpy()
    output = _output_folder(arguments, run, unit, seed, commit)
    write_code_state_archive(output, code_state)
    table.to_csv(output / "values.csv", index=False)
    top.to_csv(output / "top_walks.csv", index=False)
    flipping.to_csv(output / "flipping.csv", index=False)
    error = conservation_error(result)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "finished_utc": output.name.rsplit("_", 1)[1],
                "run": run.name,
                "seed": seed,
                "method": "walks",
                "scope": arguments.scope,
                "source": source_name,
                "unit": unit,
                "every": arguments.every,
                "windows": len(order),
                "value_unit": "score before the sigmoid",
                "rul_max": float(model.rul_max),
                "steps": result.n_steps,
                "gamma_setting": arguments.gamma,
                "gamma": list(result.gamma),
                "score_at_zero": result.score_at_zero,
                "conservation_error_max": error,
                "groups": [list(group) for group in groups],
                "walks_start_after_learned_stage": bool(
                    any(p.numel() for p in model.temporal.parameters())
                ),
                "cancellation_median": float(np.median(
                    parts["absolute_total"].to_numpy() / np.maximum(walks_sum, 1e-300)
                )),
                "seconds": round(seconds, 1),
                "sensors": symbols,
                "flipping_aufc": [
                    {"task": row.task, "ranking": row.ranking, "mean": float(row.mean),
                     "std": float(row.std) if row.count > 1 else None, "windows": int(row.count)}
                    for row in checked.itertuples()
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    progress("done", directory=output.name, run=run.name)
    print(f"walks over {len(order)} window(s) in {seconds:.1f}s; largest miss between "
          f"walks plus constant and the score before the sigmoid: {error:.2e}.")
    print(f"Saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
