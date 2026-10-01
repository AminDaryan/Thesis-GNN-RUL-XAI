# XAI

## Why a number is not enough

A remaining-life prediction tells us what the model expects, but not which
readings led to it. **Explainable artificial intelligence**, or **XAI**, makes
parts of that calculation visible. It helps us inspect what the model relies
on; it cannot establish that a sensor or engine component caused damage.

Training changes the model's multipliers and offsets using examples with
known answers. Explaining keeps that model fixed and examines one prediction.
A different saved model can give a different explanation for the same readings.

## Choose the question you want to answer

| Question | Method | What it examines | What comes out |
|---|---|---|---|
| Which sensor readings moved this prediction away from a reference-input prediction? | [Dynamic SHAP](10_dynamic_shap.md) | Predictions after selected sensors are replaced by training-derived reference readings | A signed contribution per sensor, in remaining-life cycles |
| Through which sensor connections did the internal score form? | [GNN-LRP](11_gnn_lrp.md) | The fixed model's multipliers, intermediate values and connections | A signed share per route through the graph layers, in internal-score units |

**Do not confuse a contribution with a physical cause.** Both methods explain
the model. Neither establishes which component is damaged, and neither proves
the model predicts accurately. A contribution is also different from a sensor
reading: one is calculated by an explanation method; the other was measured.

## Example setup

All three XAI pages use the same small demonstration: **FD001 test engine 5,
its last three recorded cycles, and sensors T24, T30 and T50**. These are real
readings after smoothing and rescaling using training engines only. The tables
show every value in this reduced window. The normal model receives fourteen
sensors and a longer window; this example deliberately keeps the arithmetic small.

The example network is **untrained**. Its multipliers are chosen to make its
calculation inspectable, not to predict engine life well. It keeps each sensor's
newest reading, combines neighbours on the chain T24 ↔ T30 ↔ T50, turns negative
mixtures into zero, and combines the results with multipliers 2, 1 and −1 plus
an offset of 0.5. A negative combined result becomes zero. The
[GNN-LRP example](11_gnn_lrp.md) shows every term. Its output is an arithmetic
example, not a result about prediction accuracy.

For an actual explanation, select the **saved run and seed** whose trained
model you want to inspect. Keep its preprocessing, graph, temporal algorithm
and other settings: changing them would change the question being answered.
The playground's Run history records those settings for the selected run.

Tables are rounded to six decimal places. To reproduce them, run the optional
setup and examples in order from the repository root.

<details class="process-example">
<summary>Set up the Python examples</summary>

```python
import numpy as np
import pandas as pd
import torch
from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.graph.adjacency import normalize_adjacency
from thesis_gnn_rul_xai.models.estgnn import ESTGNN
from thesis_gnn_rul_xai.explain import (
    mean_window_baseline, shap_values, walk_relevance, node_relevance,
)

prepared = run_preprocessing(PreprocessingConfig(seed=42))
row = int(np.flatnonzero(prepared.test.units == 5)[0])
names = ["T24", "T30", "T50"]
chosen = [list(prepared.sensor_symbols).index(name) for name in names]
window = prepared.test.features[row, chosen, -3:]
end_cycle = int(prepared.test.end_cycles[row])
readings = pd.DataFrame(window.T, columns=names)
readings.insert(0, "Cycle", range(end_cycle - 2, end_cycle + 1))
reference = mean_window_baseline(prepared.train)[chosen, -3:]

chain = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
with torch.random.fork_rng():
    torch.manual_seed(7)
    model = ESTGNN(normalize_adjacency(chain, method="row"),
                   window_size=3, gcn_hidden=(1,), fc_hidden=1)
with torch.no_grad():
    model.graph_layers[0].weight.fill_(1.0)
    model.fc1.weight.copy_(torch.tensor([[2.0, 1.0, -1.0]]))
    model.fc1.bias.fill_(0.5)
    model.fc2.weight.fill_(1.0)
    model.fc2.bias.fill_(0.0)
model.eval()
```

</details>

<div class="algorithm-intro">
<p class="algorithm-label">Before following an XAI algorithm</p>
<p>Follow these two steps to identify the prediction and understand the units
of its explanation. Then choose the Dynamic SHAP or GNN-LRP page for the
complete input → calculation → output tutorial.</p>
</div>

---

## 1 — Choose the prediction

An explanation belongs to one model and one input window. Record the engine
and ending cycle so that a contribution cannot be mistaken for a statement
about every engine or every moment of its life.

### Process

#### Input

This is the complete reduced example window. Each row is a cycle; each sensor
column contains rescaled readings, not temperatures or pressures in their
original units. The remaining-life answer is kept separately.

<!-- sample: readings -->
| Cycle | T24 | T30 | T50 |
|---|---|---|---|
| 96 | 0.285568 | 0.143344 | -0.050896 |
| 97 | 0.194873 | 0.328677 | 0.359872 |
| 98 | -0.373355 | 0.114623 | 0.817606 |

#### Function

Pass these readings through the fixed example model. It first forms an
**internal score**, a number before conversion to remaining life. A smooth
S-shaped conversion, called the sigmoid, changes that score into a fraction
between zero and one. Multiplying by the life ceiling gives cycles.

For this window the score is **0.220177**. Its conversion is
**125 ÷ (1 + exp(−0.220177)) = 69.352867 cycles**, where
`exp` is the exponential function. The ceiling is 125 cycles. The
[model's output tutorial](05_the_model.md#6--predicted-life-fraction) explains that conversion.
The known answer helps evaluate a prediction afterwards; it does not enter
this calculation or the explanation.

<details class="process-example">
<summary>Example in Python</summary>

```python
with torch.no_grad():
    batch = torch.from_numpy(window[None])
    nodes = model.temporal(batch)
    for layer in model.graph_layers:
        nodes = torch.relu(layer(nodes, model.adjacency))
    hidden = torch.relu(model.fc1(nodes.flatten(start_dim=1)))
    score = float(model.fc2(hidden).item())
    predicted = float(model.predict_cycles(batch).item())

prediction = pd.DataFrame([{
    "Engine": 5, "Ending cycle": end_cycle,
    "Internal score": score, "Prediction (cycles)": predicted,
    "Separate answer (cycles)": float(prepared.test.targets[row]),
}])
assert np.isclose(125 / (1 + np.exp(-score)), predicted, atol=1e-5)
```

</details>

#### Output

One prediction with its context. The answer is shown beside it for orientation;
this untrained example is not an accuracy comparison.

<!-- sample: prediction -->
| Engine | Ending cycle | Internal score | Prediction (cycles) | Separate answer (cycles) |
|---|---|---|---|---|
| 5 | 98 | 0.220177 | 69.352867 | 91.000000 |

---

## 2 — Read an explanation in context

Two methods can explain the same prediction while dividing different
quantities. Check both the **unit** and the **starting value** before reading
a positive or negative contribution.

### Process

#### Input

Here are both methods' shares for the same example. The GNN-LRP column sums
all routes starting at each sensor; its detailed page shows those routes
individually. The SHAP page shows the sensor combinations used to obtain its
column. These are explanation outputs, not new sensor readings.

<!-- sample: shares -->
| Sensor | SHAP (cycles) | Routes by starting sensor (score) |
|---|---|---|
| T24 | -10.524223 | -0.124452 |
| T30 | 0.023663 | -0.019104 |
| T50 | -4.159435 | -0.136268 |

#### Function

**SHAP starts from the prediction for the reference window.** That window
contains averages from training windows, a project choice (assumption A-20).
Add the three signed contributions to its prediction to recover the actual
prediction. Here that is **84.012863 + (-14.659996) =
69.352867 cycles**, apart from rounding. The reference prediction is
not the known answer, or the mean of the answers (assumption A-24).

**GNN-LRP splits the internal score.** Add all route shares and the separate
contribution from offsets to reconstruct that score. Here **0.500000
+ (-0.279823) = 0.220177**. Only then convert the total into
cycles. Applying the sigmoid to each share and adding would not give the same
prediction, because that conversion is not an addition-preserving operation.

A positive SHAP contribution raises predicted life relative to its reference;
a negative one lowers it. A positive route share contributes positively to the
internal score. Their sizes cannot be compared directly: cycles and internal
score are different units, and their starting values differ.

<details class="process-example">
<summary>Example in Python</summary>

```python
phi = shap_values(model, window, reference)
walks = walk_relevance(model, window)
routes = node_relevance(walks)[0]
constant = float(walks.constant[0])
with torch.no_grad():
    base = float(model.predict_cycles(torch.from_numpy(reference[None])).item())
shares = pd.DataFrame({"Sensor": names,
                       "SHAP (cycles)": phi,
                       "Routes by starting sensor (score)": routes})
shap_total = base + float(phi.sum())
walk_total = constant + float(routes.sum())
reconstruction = pd.DataFrame([
    {"Method": "SHAP", "Starting value": base,
     "Sum of shares": float(phi.sum()), "Total": shap_total, "Unit": "cycles"},
    {"Method": "GNN-LRP", "Starting value": constant,
     "Sum of shares": float(routes.sum()), "Total": walk_total, "Unit": "internal score"},
])
assert np.isclose(shap_total, predicted, atol=1e-5)
assert np.isclose(walk_total, score, atol=1e-6)
```

</details>

#### Output

The two checks recover the quantities they explain. A correct sum checks the
calculation, not the quality of the model or the physical truth of an
explanation. The small example uses the simpler gamma-zero rule for dividing
the score; the GNN-LRP tutorial explains this setting and distinguishes it
from the project's rule for a model with several graph layers.

<!-- sample: reconstruction -->
| Method | Starting value | Sum of shares | Total | Unit |
|---|---|---|---|---|
| SHAP | 84.012863 | -14.659996 | 69.352867 | cycles |
| GNN-LRP | 0.500000 | -0.279823 | 0.220177 | internal score |

Continue with [Dynamic SHAP](10_dynamic_shap.md) to work through input
replacement and averaging, or [GNN-LRP](11_gnn_lrp.md) to trace the routes and
test their ranking by changing which sensors remain connected.
