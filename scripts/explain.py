#!/usr/bin/env python3
"""
Explain a stored training run
=============================

Loads the weights one seed of a finished run saved, rebuilds the data it was
trained on from the settings stored beside it, and works out how much each of
the 14 sensors moved its predictions. The answer is written into the run's own
folder, under ``explanations/``, stamped with the commit that computed it.

Two methods are available, and only the first is the paper's:

``shap``
    Exact Shapley values, Equations 14 and 15 of Wen et al. Every one of the
    16,384 subsets of the sensors is evaluated for every window. A sensor left
    out of a subset is replaced by its average training window (A-20).
``occlusion``
    Not in the paper. Each sensor in turn is replaced by the same baseline
    while the other 13 stay, and the change in the prediction is recorded:
    15 evaluations per window instead of 16,384. It is the one term of the
    Shapley average in which the sensor arrives last, so the two agree only
    when the sensors do not interact.

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
``values.csv`` holds one row per window: the engine, the cycle the window ends
at, the true and the predicted remaining life, the prediction at the baseline,
one column per sensor with its value in cycles, and one column
``reading_<sensor>`` per sensor holding what that sensor was reading at the
window's last cycle, so that a chart can colour each value by whether the
sensor read high or low, as the paper's Fig. 9 does. That reading is the one
the model was given: after the noise filter and the rescaling, not the raw
number. ``summary.json`` holds the settings, the
time taken, the largest gap in Equation 14's sum, and every sensor's average
size.

Examples
--------
The paper's dynamic explanation, for the engine with the longest history::

    python scripts/explain.py --run reports/<run directory> --method shap --scope engine

A quick first look across the test engines::

    python scripts/explain.py --run reports/<run directory> --method occlusion --scope test
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
from thesis_gnn_rul_xai.utils.provenance import current_commit, utc_stamp

METHODS = ("shap", "occlusion")
SCOPES = ("engine", "test")


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
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--progress", type=Path, default=None,
                        help="Append one JSON line per event to this file")
    parser.add_argument("--quiet", action="store_true", help="Warnings only")
    arguments = parser.parse_args()
    if arguments.every < 1:
        raise SystemExit("--every must be at least 1.")

    configure_logging(logging.WARNING if arguments.quiet else logging.INFO)

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

    # ASSUMPTION A-20: a sensor left out is replaced by its average window over
    # the training engines. Equation 15 needs a prediction without a sensor and
    # the paper never says what stands in for it.
    baseline = mean_window_baseline(prepared.train)

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
    progress("start", windows=len(order), method=arguments.method, scope=arguments.scope)
    started = time.perf_counter()

    if arguments.method == "shap":
        values = np.empty((len(order), features.shape[1]))
        for index in range(len(order)):
            values[index] = shap_values(model, features[index], baseline, device=device)
            progress("window", done=index + 1, total=len(order),
                     seconds=time.perf_counter() - started)
    else:
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

    symbols = list(prepared.sensor_symbols)
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

    commit = current_commit()
    stamp = utc_stamp()
    name = (
        f"{arguments.method}_{arguments.scope}"
        + (f"_unit{unit}" if unit is not None else "")
        + (f"_every{arguments.every}" if arguments.every > 1 else "")
        + f"_seed{seed}_{commit}_{stamp}"
    )
    output = run / "explanations" / name
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "values.csv", index=False)
    ranking = table[symbols].abs().mean().sort_values(ascending=False)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "finished_utc": stamp,
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
    progress("done", directory=name, run=run.name)
    print(f"{arguments.method} over {len(order)} window(s) in {seconds:.1f}s; "
          f"largest gap between the values' sum and the prediction's distance "
          f"from the baseline: {gap.max():.2e} cycles.")
    print(f"Saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
