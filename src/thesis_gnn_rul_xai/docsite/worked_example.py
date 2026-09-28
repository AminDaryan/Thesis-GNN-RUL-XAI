"""Trace one real window through a named saved network for the documentation."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from html import escape

RUN = "fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z"


@lru_cache(maxsize=1)
def trace() -> dict:
    """Recompute one saved seed's arithmetic, not a performance estimate."""
    import numpy as np
    import torch

    from ..data.loading import load_subset
    from ..data.pipeline import run_preprocessing
    from ..data.sensors import PAPER_SENSOR_COLUMNS, PAPER_SENSOR_SYMBOLS
    from ..training import load_seed, read_run
    from ..utils.paths import raw_data_dir

    run = read_run(RUN)
    seed = run.seeds[0]
    model, prepared = load_seed(run, seed)
    config = replace(run.preprocessing, seed=seed)
    frames = run_preprocessing(config, keep_frames=True)
    raw = load_subset("FD001", data_dir=raw_data_dir())
    group, data = next((name, data) for name, data in
                       (("training", prepared.train), ("validation", prepared.validation))
                       if 1 in data.units)
    index = int(np.flatnonzero(data.units == 1)[0])
    end = int(data.end_cycles[index])
    start = end - config.window_size + 1
    engine = raw.train[raw.train.unit == 1]
    rows = engine[engine.cycle.between(start, end)]
    scaled = frames.frames["train"]
    scaled = scaled[(scaled.unit == 1) & scaled.cycle.between(start, end)]
    features = torch.as_tensor(data.features[index:index + 1], dtype=torch.float32)
    steps = []

    def record(name, incoming, outgoing, operation, terms=None):
        steps.append({"name": name, "input_shape": list(incoming.shape),
                      "output_shape": list(outgoing.shape),
                      "input": incoming.detach().flatten()[:3].tolist(),
                      "output": outgoing.detach().flatten()[:3].tolist(),
                      "operation": operation, "terms": terms or []})

    with torch.no_grad():
        hidden = model.temporal(features)
        record("Temporal algorithm", features, hidden,
               f"Keep the final preprocessed reading at cycle {end} for each sensor. "
               f"T24: {features[0, 0, -1].item():.6f} becomes {hidden[0, 0, 0].item():.6f}. "
               "This saved model uses Last cycle, not LSTM.")
        for i, layer in enumerate(model.graph_layers, 1):
            mixed = model.adjacency @ hidden
            before = layer(hidden, model.adjacency)
            after = torch.relu(before)
            neighbours = torch.nonzero(model.adjacency[0]).flatten()
            terms = [{"from": PAPER_SENSOR_SYMBOLS[int(j)], "input": hidden[0, j, 0].item(),
                      "multiplier": model.adjacency[0, j].item(),
                      "product": (hidden[0, j, 0] * model.adjacency[0, j]).item()} for j in neighbours]
            record(f"Graph layer {i}", hidden, after,
                   f"For T24's first number, add the neighbour products below: {mixed[0, 0, 0].item():.6f}. "
                   f"Apply this layer's learned multipliers to all {hidden.shape[-1]} mixed numbers. "
                   f"Its first result is {before[0, 0, 0].item():.6f}; replacing negative results by zero "
                   f"gives {after[0, 0, 0].item():.6f}.", terms)
            hidden = after
        flat = hidden.flatten(start_dim=1)
        record("Flatten", hidden, flat, "Lay each sensor's numbers end to end, keeping sensor order. No arithmetic changes them.")
        before_dense = model.fc1(flat)
        dense = torch.relu(before_dense)
        terms = [{"from": str(i + 1), "input": value.item(), "multiplier": weight.item(),
                  "product": (value * weight).item()}
                 for i, (value, weight) in enumerate(zip(flat[0], model.fc1.weight[0], strict=True))]
        record("Fully connected", flat, dense,
               f"For output 1, add all {flat.shape[1]} products below and the learned offset "
               f"{model.fc1.bias[0].item():.6f}: {before_dense[0, 0].item():.6f}. "
               f"Keep the larger of that and zero: {dense[0, 0].item():.6f}. Repeat with different weights for each output.", terms)
        score = model.fc2(dense)
        fraction = torch.sigmoid(score)
        record("Output", dense, fraction,
               f"Multiply each input by its output-layer weight and add the offset: internal score "
               f"{score.item():.6f}. The sigmoid converts this to 1 / (1 + exp(-score)) = {fraction.item():.6f}. "
               "Here exp means raising the mathematical constant e to the given power; this is not a probability.")
        prediction = fraction * model.rul_max
        record("Scale to cycles", fraction, prediction,
               f"{fraction.item():.6f} x {model.rul_max} = {prediction.item():.6f} cycles.")
        reference = model.predict_cycles(features).item()
    return {"run": RUN, "seed": seed, "group": group, "start": start, "end": end,
            "ceiling": model.rul_max, "last_cycle": int(engine.cycle.max()),
            "raw": rows.sensor_2.tolist(), "scaled": scaled.sensor_2.tolist(),
            "columns": list(PAPER_SENSOR_COLUMNS), "steps": steps,
            "prediction": prediction.item(), "reference": reference, "target": float(data.targets[index])}


def draw() -> str:
    """Render the live trace with explicit inputs, operations and outputs."""
    try:
        example = trace()
    except (FileNotFoundError, ImportError, ValueError) as error:
        return '<p class="figure-notice">The worked example requires the real FD001 files and saved run ' + escape(RUN) + ': ' + escape(str(error)) + '</p>'
    fmt = lambda values: ", ".join(f"{x:.6f}" for x in values)  # noqa: E731
    out = '<div class="worked-trace"><p><strong>Live example, not an accuracy result.</strong> Training-file engine 1, cycles ' + str(example["start"]) + '-' + str(example["end"]) + '. This seed assigns it to ' + example["group"] + '.</p><p>Saved run: <code>' + RUN + '</code>; seed ' + str(example["seed"]) + '.</p>'
    out += '<p>T24 first three raw readings: ' + fmt(example["raw"][:3]) + '. After the saved smoothing and rescaling: ' + fmt(example["scaled"][:3]) + '.</p>'
    for step in example["steps"]:
        out += '<section><h4>' + escape(step["name"]) + '</h4><dl class="worked-io"><dt>Input</dt><dd>Shape ' + str(step["input_shape"]) + '; first three numbers: ' + fmt(step["input"]) + '</dd><dt>Operation</dt><dd>' + escape(step["operation"]) + '</dd><dt>Output</dt><dd>Shape ' + str(step["output_shape"]) + '; first three numbers: ' + fmt(step["output"]) + '</dd></dl>'
        if step["terms"]:
            out += '<details><summary>Show every product in the addition</summary><div class="pg-data-table-wrap"><table class="pg-data-table"><thead><tr><th>Input position</th><th>Input value</th><th>Multiplier</th><th>Product</th></tr></thead><tbody>'
            for term in step["terms"]:
                out += '<tr><th>' + escape(term["from"]) + '</th>' + ''.join('<td>' + f'{term[key]:.6f}' + '</td>' for key in ("input", "multiplier", "product")) + '</tr>'
            out += '</tbody></table></div></details>'
        out += '</section>'
    return out + '<p>Prediction: ' + f'{example["prediction"]:.6f}' + ' cycles. Separate capped answer: ' + str(example["target"]) + ' cycles. The answer did not enter any of the model calculations. Displayed numbers are rounded; calculations use full precision.</p></div>'
