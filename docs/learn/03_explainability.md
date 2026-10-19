# XAI

## Why a number is not enough

A remaining-life prediction tells us what the model expects, but not which
readings led to it. **Explainable artificial intelligence**, or **XAI**, is the
research area that tries to extract understandable information from trained
models (Schnake et al., 2022, p. 1); here, it makes parts of that calculation
visible. It helps us inspect what the model relies on (Montavon et al., 2019,
Section 10.1). It cannot establish that a sensor or engine component caused
damage. A model can rely on a reading that only happens to vary with the
answer in the data it learned from (Montavon et al., 2019, Section 10.1), and
an explanation reports that reliance all the same; the conclusion about engine
damage is this project's reading of that point, not a sentence in the source.

Training changes the model's multipliers and offsets using examples with
known answers. Explaining keeps that model fixed and examines one prediction
(Lundberg & Lee, 2017, p. 2; Montavon et al., 2019, Section 10.1).
A different saved model can give a different explanation for the same
readings, because each method computes its explanation from the model as well
as the input (Lundberg & Lee, 2017, Eq. 8; Schnake et al., 2022, Eq. 9).

## Choose the question you want to answer

| Question | Method | What it examines | What comes out |
|---|---|---|---|
| Which sensor readings moved this prediction away from a reference-input prediction? | [Dynamic SHAP](10_dynamic_shap.md) (Wen et al., 2026, Section 2.4) | Predictions after selected sensors are replaced by training-derived reference readings (assumption A-20) | A signed contribution per sensor, in remaining-life cycles |
| Through which sensor connections did the internal score form? | [GNN-LRP](11_gnn_lrp.md) (Schnake et al., 2022, Section 3.2), added by this project (assumption A-29) | The fixed model's multipliers, intermediate values and connections (Schnake et al., 2022, Eq. 9) | A signed share per route through the graph layers, in internal-score units |

**Do not confuse a contribution with a physical cause.** Both methods explain
the model: each divides a number the model computed, not the engine's true
remaining life. SHAP divides the model's prediction (Lundberg & Lee, 2017,
Eq. 5), here in cycles. GNN-LRP divides a graph network's output
(Schnake et al., 2022, Fig. 1 and Proposition 1); here that is the internal
score the network forms before converting it to remaining life, a project
choice (assumption A-29). Neither establishes which component is damaged, and
neither proves the model predicts accurately; that follows from what they
divide, and is this project's reading rather than a claim either source makes. A contribution is also
different from a sensor reading: one is calculated by an explanation method;
the other was measured.

## Example setup

All three XAI pages use the same small demonstration: **FD001 test engine 5,
its last three recorded cycles, and sensors T24, T30 and T50**, from NASA's
data (NASA Prognostics Center of Excellence [NASA PCoE], 2008). T24, T30 and
T50 are the total temperatures at the outlets of the low-pressure compressor,
the high-pressure compressor and the low-pressure turbine (Wen et al., 2026,
Section 2.3.1 and Table 2). These are real readings after smoothing
(Wen et al., 2026, Section 3.2.3), with half the weight on the newest reading,
which is how this project reads that section's value of 0.5 (assumption A-04).
They were then rescaled by subtracting each sensor's mean and dividing by its
standard deviation, its spread around that mean (assumption A-03), both
learned from training engines only (assumption A-25). The tables show every
value in this reduced window. The normal model receives fourteen
sensors and a longer window (Wen et al., 2026, Tables 3 and 6); this example
deliberately keeps the arithmetic small.

The example network is **untrained**. Its multipliers are chosen to make its
calculation inspectable, not to predict engine life well. It keeps each sensor's
newest reading, as the project's default model does (assumption A-07). It
combines neighbours on the chain T24 ↔ T30 ↔ T50, turns negative
mixtures into zero, and combines the results with multipliers 2, 1 and −1 plus
an offset of 0.5. A negative combined result becomes zero. Wen et al. (2026)
call that chain "the temperature transfer chain T24→T30→T50" (p. 5), and its
two connections are exactly the ones among these three sensors in their graph
(Table 3). The [GNN-LRP example](11_gnn_lrp.md) shows every term. Its output is an arithmetic
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

This is the complete reduced example window. Each row is a cycle (NASA PCoE,
2008); each sensor column contains rescaled readings, not temperatures in
their original units. The remaining-life answer is kept
separately; NASA supplies it as a separate list of true remaining lives for
the test engines (NASA PCoE, 2008).

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
between zero and one (Wen et al., 2026, Eq. 13). Multiplying by the life
ceiling gives cycles (assumption A-13).

For this window the score is **0.220177**. Its conversion is
**125 ÷ (1 + exp(−0.220177)) = 69.352867 cycles**, where
`exp` is the exponential function. The ceiling is 125 cycles (assumption
A-01). The
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
all routes starting at each sensor (Schnake et al., 2022, pp. 6–7); its
detailed page shows those routes individually. The SHAP page shows the sensor
combinations used to obtain its column. These are explanation outputs, not new
sensor readings.

<!-- sample: shares -->
| Sensor | SHAP (cycles) | Routes by starting sensor (score) |
|---|---|---|
| T24 | -10.524223 | -0.124452 |
| T30 | 0.023663 | -0.019104 |
| T50 | -4.159435 | -0.136268 |

#### Function

**SHAP starts from the prediction for the reference window**, the model's
output when every sensor counts as missing (Lundberg & Lee, 2017, p. 4). That
window contains averages from training windows, a project choice (assumption A-20).
Add the three signed contributions to its prediction to recover the actual
prediction (Lundberg & Lee, 2017, Eq. 5). Here that is **84.012863 + (-14.659996) =
69.352867 cycles**, apart from rounding. The reference prediction is
not the known answer, or the mean of the answers; Wen et al. (2026, Eq. 14)
describe their starting value as that mean (assumption A-24).

**GNN-LRP splits the internal score.** Add all route shares and the separate
contribution from offsets to reconstruct that score (assumption A-29). Here **0.500000
+ (-0.279823) = 0.220177**. Only then convert the total into
cycles. Applying the sigmoid to each share and adding would not give the same
prediction, because that conversion is not an addition-preserving operation.

A positive SHAP contribution raises predicted life relative to its reference;
a negative one lowers it (Lundberg & Lee, 2017, Fig. 1). A positive route
share contributes positively to the internal score (Schnake et al., 2022,
Fig. 4). Their sizes cannot be compared directly: cycles and internal
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
the score (Montavon et al., 2019, Section 10.2.1; Schnake et al., 2022, p. 6);
the GNN-LRP tutorial explains this setting and distinguishes it from the
project's rule for a model with several graph layers (assumption A-29).

<!-- sample: reconstruction -->
| Method | Starting value | Sum of shares | Total | Unit |
|---|---|---|---|---|
| SHAP | 84.012863 | -14.659996 | 69.352867 | cycles |
| GNN-LRP | 0.500000 | -0.279823 | 0.220177 | internal score |

Continue with [Dynamic SHAP](10_dynamic_shap.md) to work through input
replacement and averaging, or [GNN-LRP](11_gnn_lrp.md) to trace the routes and
test their ranking by changing which sensors remain connected, a check adapted
from Schnake et al. (2022, Section 4.1) to this network (assumption A-33).

## References

- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. In I. Guyon, U. von Luxburg, S. Bengio, H. Wallach,
  R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), *Advances in Neural
  Information Processing Systems 30 (NIPS 2017)* (pp. 4765–4774).
  https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
  [Copy](../references/lundberg_2017_unified_shap.pdf). Page numbers on this
  page are those printed on the copy held, which numbers its pages 1 to 10.
- Montavon, G., Binder, A., Lapuschkin, S., Samek, W., & Müller, K.-R.
  (2019). Layer-wise relevance propagation: An overview. In W. Samek,
  G. Montavon, A. Vedaldi, L. K. Hansen, & K.-R. Müller (Eds.), *Explainable
  AI: Interpreting, explaining and visualizing deep learning* (Lecture Notes
  in Computer Science, Vol. 11700, pp. 193–209). Springer.
  https://doi.org/10.1007/978-3-030-28954-6_10
  [Copy](../references/montavon_2019_lrp_overview.pdf). The copy held is the
  authors' own copy, whose printed pages differ from the book's, so it is
  cited by section only.
- NASA Prognostics Center of Excellence. (2008). readme.txt, the description
  of the C-MAPSS data set distributed inside its zip file. NASA Prognostics
  Data Repository.
  [Copy](../references/nasa_2008_cmapss_readme.txt)
- Schnake, T., Eberle, O., Lederer, J., Nakajima, S., Schütt, K. T., Müller,
  K.-R., & Montavon, G. (2022). Higher-order explanations of graph neural
  networks via relevant walks. *IEEE Transactions on Pattern Analysis and
  Machine Intelligence*, *44*(11), 7581–7596.
  https://doi.org/10.1109/TPAMI.2021.3115452
  [Copy](../references/schnake_2022_gnn_lrp_relevant_walks.pdf). Page,
  section, figure and equation numbers on this page are those of arXiv
  version 3, the copy held.
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and
  Technology*, *168*, 110875. https://doi.org/10.1016/j.ast.2025.110875
  [Copy](../references/wen_2026_estgnn_dynamic_shap.pdf)
