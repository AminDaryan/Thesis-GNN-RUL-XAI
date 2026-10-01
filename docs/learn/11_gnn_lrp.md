# GNN-LRP

A sensor reading can affect a prediction through several neighbours. Naming
the sensor alone does not show which route the calculation used. GNN-LRP,
short for **graph neural network layer-wise relevance propagation**, follows
those routes and assigns each one a signed share of the network's score.

A **walk** records one sensor at the input and one after each graph layer.
**T50 → T30 → T30 → Ps30** means that information starts at T50, reaches T30
in the first layer, stays there in the second, and reaches Ps30 in the third.
The arrows are steps through the calculation, not later engine cycles.
[Schnake and colleagues' method](https://arxiv.org/html/2006.03589v3) is an
additional explanation method in this project (assumption A-29).

> Sensor readings → graph inputs → routes through the layers → shares of
> the internal score → check against the network.

The explanation reads the network's connections, multipliers and intermediate
numbers. Predictions alone are insufficient: it needs this internal access.
It does not retrain the network. Its target is the **score before conversion
to remaining-life cycles**, so its shares have score units, not cycles.

## Example setup

The example uses **FD001 test engine 5**, the final three available cycles for
T24, T30 and T50. The ordinary preprocessing uses seed 42, and only training
engines supply the rescaling averages and spreads. Afterwards, we select three
sensors and three cycles to keep every input cell visible.

This is the same **small, untrained demonstration network** used on the
[Dynamic SHAP page](10_dynamic_shap.md). Its fixed multipliers make the
arithmetic readable. It uses the newest reading per sensor, a three-sensor
path, and one graph layer. It is not a stored trained run, and its prediction
is not a result about accuracy. A normal run can have a different temporal
stage, graph and number of layers.

The graph joins **T24 ↔ T30 ↔ T50**, and each sensor can keep its own number.
The graph layer averages arriving numbers, multiplies by 1, and sets negative
totals to zero. This last operation is a **ReLU**: a switch that keeps positive
values and replaces negative ones with zero. The final calculation multiplies
the three results by 2, 1 and −1, adds 0.5, and applies another such switch.
The remaining multiplier is 1 and its added constant is 0.

Run the optional examples in page order after this setup. Tables show six
decimal places; the calculations retain full precision.

<details class="process-example">
<summary>Set up the Python examples</summary>

```python
import itertools
import numpy as np
import pandas as pd
import torch
from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.graph.adjacency import normalize_adjacency
from thesis_gnn_rul_xai.models import ESTGNN
from thesis_gnn_rul_xai.explain import (
    walk_relevance, node_relevance, conservation_error,
    paper_gammas, subgraph_relevance, subgraph_scores, node_flipping,
)

prepared = run_preprocessing(PreprocessingConfig(seed=42))
row = int(np.flatnonzero(prepared.test.units == 5)[0])
names = ["T24", "T30", "T50"]
chosen = [list(prepared.sensor_symbols).index(name) for name in names]
window = prepared.test.features[row, chosen, -3:]
end_cycle = int(prepared.test.end_cycles[row])
source = pd.DataFrame(window.T, columns=names)
source.insert(0, "Cycle", range(end_cycle - 2, end_cycle + 1))

path = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
model = ESTGNN(normalize_adjacency(path, method="row"),
               window_size=3, gcn_hidden=(1,), fc_hidden=1)
with torch.no_grad():
    model.graph_layers[0].weight.fill_(1.0)
    model.fc1.weight.copy_(torch.tensor([[2.0, 1.0, -1.0]]))
    model.fc1.bias.fill_(0.5)
    model.fc2.weight.fill_(1.0)
    model.fc2.bias.fill_(0.0)
model.eval()
adjacency = model.adjacency.detach().double().numpy()
graph_input = window[:, -1].astype(np.float64)
graph_weights = pd.DataFrame(adjacency, columns=["From " + n for n in names])
graph_weights.insert(0, "Arrives at", names)
```

</details>

<div class="algorithm-intro">
<p class="algorithm-label">Walk explanation algorithm</p>
<p>Start with one fixed network and one engine window. Follow steps 1–4:
trace the score, divide it among walks, check the totals, then test the
explanation by adding and removing sensors.</p>
</div>

---

## 1 — Follow the graph calculation

Before dividing a score, establish how the network made it. Each sensor is a
**node**, a place in the graph that holds numbers. A connection says which
other sensor's numbers may arrive there. Readings and connections are
different inputs: one supplies values; the other controls where they can go.

### Process

#### Input

This is the complete three-cycle input to the demonstration. Each row is a
cycle of the same engine; every column is a sensor. Values are distances from
each sensor's training average in units of its training spread, so a negative
reading means below average.

<!-- sample: source -->
| Cycle | T24 | T30 | T50 |
| --- | --- | --- | --- |
| 96 | 0.285568 | 0.143344 | -0.050896 |
| 97 | 0.194873 | 0.328677 | 0.359872 |
| 98 | -0.373355 | 0.114623 | 0.817606 |


The graph supplies these receiving weights. Read across a row to see what one
sensor receives. A zero blocks that route.

<!-- sample: graph_weights -->
| Arrives at | From T24 | From T30 | From T50 |
| --- | --- | --- | --- |
| T24 | 0.500000 | 0.500000 | 0.000000 |
| T30 | 0.333333 | 0.333333 | 0.333333 |
| T50 | 0.000000 | 0.500000 | 0.500000 |


#### Function

The temporal stage keeps the final reading of each column. The graph input
becomes **three sensors × one number**. The earlier rows remain visible above
so the decision to ignore them is explicit; an LSTM calculates different
inputs from the history.

The graph layer combines the final readings:

- T24: (-0.373355 + 0.114623) ÷ 2 =
  -0.129366. Its switch outputs zero.
- T30: (-0.373355 + 0.114623 +
  0.817606) ÷ 3 = 0.186291.
  Its switch keeps that positive result.
- T50: (0.114623 + 0.817606) ÷ 2 =
  0.466114. Its switch also stays on.

The final calculation gives the score:

> 2 × 0 + 1 × 0.186291 − 1 × 0.466114
> + 0.5 = 0.220177.

That total is positive, so the second switch keeps it. The network converts
the score into cycles using **125 ÷ (1 + exp(−score))**. Here 125 is the
prediction ceiling; `exp` raises the mathematical constant e to the given
power. This curved conversion bounds the prediction between zero and the
ceiling. The network reports **69.352867 cycles** for this input. Walks will
divide the score before the conversion.

<details class="process-example">
<summary>Example in Python</summary>

```python
layer_before = adjacency @ graph_input
layer_after = np.maximum(layer_before, 0.0)
final_weights = model.fc1.weight.detach().double().numpy()[0]
score = float(max(final_weights @ layer_after + 0.5, 0.0))
converted_score = float(model.rul_max / (1.0 + np.exp(-score)))
with torch.no_grad():
    prediction = float(model.predict_cycles(torch.from_numpy(window[None]))[0])
np.testing.assert_allclose(converted_score, prediction, atol=1e-5)
layer_table = pd.DataFrame({
    "Sensor": names, "Newest reading": graph_input,
    "Received total": layer_before, "After switch": layer_after,
})
score_table = pd.DataFrame({
    "Engine": [5], "End cycle": [end_cycle],
    "Score": [score], "Prediction (cycles)": [prediction],
})
```

</details>

#### Output

The shape is still **three sensors × one number**, but each output now
combines permitted neighbours:

<!-- sample: layer_table -->
| Sensor | Newest reading | Received total | After switch |
| --- | --- | --- | --- |
| T24 | -0.373355 | -0.129366 | 0.000000 |
| T30 | 0.114623 | 0.186291 | 0.186291 |
| T50 | 0.817606 | 0.466114 | 0.466114 |


The final calculation turns that table into one score and one prediction:

<!-- sample: score_table -->
| Engine | End cycle | Score | Prediction (cycles) |
| --- | --- | --- | --- |
| 5 | 98 | 0.220177 | 69.352867 |


**Do not confuse a graph output with an explanation share.** T50's graph
output is positive, yet the final calculation multiplies it by −1. Its
incoming routes can therefore carry negative shares of the score.

---

## 2 — Split the score into walks

Now ask which routes built the graph outputs. Each walk share records the
starting reading and the multipliers along its route to the score. Positive
and negative shares can cancel, like terms in an ordinary sum.

### The rule, and its setting γ

The method passes a share backwards through a layer, dividing it among the
quantities that produced it. **γ**, pronounced gamma, adjusts that division:
positive multipliers receive extra weight, then shares are rescaled to
preserve their total. With gamma 1.5, a positive multiplier of 1 is counted
as 1 + 1.5 × 1 = 2.5 before that rescaling. This changes the explanation rule,
not the prediction. [The method's Table 1](https://arxiv.org/html/2006.03589v3)
defines the rule.

The project's three graph layers use **0, 1.5, 0**. The
[authors' implementation](https://git.tu-berlin.de/thomas_schnake/paper_gnn_lrp)
starts at 3 and falls evenly to 0. Here the first value is replaced by 0
because rescaled readings can be negative, while the rule's derivation
assumes nonnegative layer inputs. The layers after the graph use 0 too
(assumption A-29).

This demonstration has only one graph layer, also its first, so its gamma is
**0**. That is the GNN-GI special case: no extra weighting. It makes every
multiplication visible; the larger network uses the schedule above. Older
saved explanations made entirely at gamma 0 are labelled GNN-GI, distinguishing
them from the current setting.

### Process

#### Input

Each arrival sensor has a number and a final multiplier. Its switch determines
whether information arriving there can reach the score.

<!-- sample: route_inputs -->
| Arrives at | Graph output | Switch (0 or 1) | Final multiplier | Score per arriving unit |
| --- | --- | --- | --- | --- |
| T24 | 0.000000 | 0.000000 | 2.000000 | 0.000000 |
| T30 | 0.186291 | 1.000000 | 1.000000 | 1.000000 |
| T50 | 0.466114 | 1.000000 | -1.000000 | -1.000000 |


Starting readings and receiving weights come from step 1.

#### Function

For this one-layer example, multiply:

> starting reading × receiving weight × graph multiplier
> × receiving switch × final multiplier.

The graph multiplier is 1. The receiving switch is 1 when its total is
positive and 0 otherwise. Final multipliers are 2, 1 and −1 for T24, T30 and
T50. The later switch is on and the remaining multiplier is 1.

For **T50 → T30**, this gives
0.817606 × 1/3 × 1 × 1 × 1 =
**0.272535**. For **T30 → T50**, it gives
0.114623 × 1/2 × 1 × 1 × (−1) =
**-0.057311**. For **T30 → T24**, the receiving
switch is off, so the share is zero even though the connection exists.

Repeat for every start-and-arrival pair. With more graph layers, continue
through one sensor per layer and apply each layer's rule. Keep the separately
added constants apart from routes starting at sensors.

<details class="process-example">
<summary>Example in Python</summary>

```python
switches = (layer_before > 0).astype(np.float64)
worth = switches * final_weights
route_inputs = pd.DataFrame({
    "Arrives at": names, "Graph output": layer_after,
    "Switch (0 or 1)": switches, "Final multiplier": final_weights,
    "Score per arriving unit": worth,
})
walks = walk_relevance(model, window)
manual_shares = graph_input[:, None] * adjacency.T * worth[None, :]
np.testing.assert_allclose(walks.relevance[0], manual_shares, atol=1e-12)
assert walks.gamma == (0.0,)
assert paper_gammas(3) == [0.0, 1.5, 0.0]
assert 1.0 + paper_gammas(3)[1] * 1.0 == 2.5
walk_table = pd.DataFrame([
    {"Starts at": names[start], "Arrives at": names[end],
     "Receiving weight": adjacency[end, start],
     "Share of score": walks.relevance[0, start, end]}
    for start in range(3) for end in range(3)
])
```

</details>

#### Output

Here are **all nine start-and-arrival pairs**. Four have zero share: two
connections do not exist and two reach T24's closed switch. Zero rows show
why an available connection need not carry a contribution.

<!-- sample: walk_table -->
| Starts at | Arrives at | Receiving weight | Share of score |
| --- | --- | --- | --- |
| T24 | T24 | 0.500000 | 0.000000 |
| T24 | T30 | 0.333333 | -0.124452 |
| T24 | T50 | 0.000000 | 0.000000 |
| T30 | T24 | 0.500000 | 0.000000 |
| T30 | T30 | 0.333333 | 0.038208 |
| T30 | T50 | 0.500000 | -0.057311 |
| T50 | T24 | 0.000000 | 0.000000 |
| T50 | T30 | 0.333333 | 0.272535 |
| T50 | T50 | 0.500000 | -0.408803 |


T24's own graph output was zero, yet **T24 → T30** carries a nonzero share.
The reading left T24 before the receiving switches acted; T30's stayed on.
A list of graph outputs alone would miss this distinction.

---

## 3 — Add the shares and check the total

First keep routes visible, then add those with the same starting sensor.
Finally add the part from the network's constants and compare with the
independently calculated score.

### Process

#### Input

Arrange the same shares as **three starting sensors × three arrival sensors**.
This table contains contributions, not the graph's receiving weights.

<!-- sample: share_grid -->
| Starts at | To T24 | To T30 | To T50 |
| --- | --- | --- | --- |
| T24 | 0.000000 | -0.124452 | 0.000000 |
| T30 | 0.000000 | 0.038208 | -0.057311 |
| T50 | 0.000000 | 0.272535 | -0.408803 |


The separately carried constant is **0.500000** here.
The target score from step 1 is **0.220177**.

#### Function

Add across each row. T30 starts two nonzero routes:
0.038208 + (-0.057311) =
**-0.019104**. One route raises the score; the
other lowers it by more.

Then add the three sensor totals and the constant:

> -0.124452 + (-0.019104)
> + (-0.136268) + 0.5 = 0.220177.

This agreement is called **conservation**: the division preserves the score.
It checks the accounting, not whether the explanation reveals an engine's
physical cause of failure.

<details class="process-example">
<summary>Example in Python</summary>

```python
share_grid = pd.DataFrame(walks.relevance[0], columns=["To " + n for n in names])
share_grid.insert(0, "Starts at", names)
sensor_totals = node_relevance(walks)[0]
sensor_shares = pd.DataFrame({"Starting sensor": names, "Share of score": sensor_totals})
reconstructed = float(sensor_totals.sum() + walks.constant[0])
balance = pd.DataFrame({
    "Quantity": ["All walks", "Separate constant", "Reconstructed score", "Original score"],
    "Value": [float(sensor_totals.sum()), float(walks.constant[0]), reconstructed, score],
})
assert conservation_error(walks) < 1e-12
np.testing.assert_allclose(reconstructed, score, atol=1e-12)
```

</details>

#### Output

Each starting sensor now has one total. This view no longer tells us which
arrival sensor carried each part.

<!-- sample: sensor_shares -->
| Starting sensor | Share of score |
| --- | --- |
| T24 | -0.124452 |
| T30 | -0.019104 |
| T50 | -0.136268 |


The accounting checks out:

<!-- sample: balance -->
| Quantity | Value |
| --- | --- |
| All walks | -0.279823 |
| Separate constant | 0.500000 |
| Reconstructed score | 0.220177 |
| Original score | 0.220177 |


**Do not confuse the constant with a SHAP reference prediction.** It is the
part carried from numbers added inside the network's layers. In a larger
network, different switches can change it between windows. It is not the
prediction of an average engine. [SHAP](10_dynamic_shap.md) instead divides
a difference from a chosen reference.

---

## 4 — Check by adding and removing sensors

Exact accounting alone does not tell us how a ranking behaves when the graph
changes. A second check adds sensors to an empty graph and removes them from
the full graph, measuring the network again after every change. The method's
authors call this **node flipping** and summarize each curve by its mean
([Section IV-A](https://arxiv.org/html/2006.03589v3)).

> Adding: no sensors → one sensor → two sensors → all sensors.
>
> Removing: all sensors → two sensors → one sensor → no sensors.

Here removing a sensor cuts all of its receiving and sending connections,
including its connection to itself. Surviving weights stay as they were:
they are not divided by new totals. Readings and learned multipliers remain
fixed (assumption A-33).

### Process

#### Input

Rate a sensor set by adding **only walks whose every sensor is inside that
set**. A route leaving the set does not count. The constant is omitted because
it belongs to no sensor. Here are all possible sets for this example:

<!-- sample: subset_table -->
| Sensors present | Walks wholly inside |
| --- | --- |
| None | 0.000000 |
| T24 | 0.000000 |
| T30 | 0.038208 |
| T50 | -0.408803 |
| T24, T30 | -0.086244 |
| T24, T50 | -0.408803 |
| T30, T50 | -0.155371 |
| T24, T30, T50 | -0.279823 |


These are estimates from the existing explanation, not predictions made by
running an altered graph.

#### Function

For **adding**, choose the next sensor giving the largest estimated total
for the enlarged set:

1. Start empty. T30 alone gives 0.038208, more than
   T24's 0.000000 or T50's
   -0.408803. Add T30.
2. Compare adding T24, giving -0.086244, with adding
   T50, giving -0.155371. Add T24.
3. T50 is the only remaining sensor. Add it last.

After each choice, run the network with only those sensors connected. T30
alone keeps its original self-weight of 1/3, making its score
0.114623 ÷ 3 + 0.5 = 0.538208.
Adding T24 makes T30's received total negative, closing its switch and
returning the score to 0.5. That changed switch explains why a subset's walk
estimate need not equal the actual change in score.

For **removing**, start full and choose the next removal whose remaining
walks stay closest to the original full walk sum. Run the altered network and
record its score's **absolute distance from the original score**. A high mean
score is better for adding; a low mean distance is better for removing.

<details class="process-example">
<summary>Example in Python</summary>

```python
subsets = [tuple(group) for size in range(4)
           for group in itertools.combinations(range(3), size)]
subset_values = {}
for group in subsets:
    present = np.array([index in group for index in range(3)])
    subset_values[group] = subgraph_relevance(walks.relevance[0], present)
subset_table = pd.DataFrame({
    "Sensors present": [", ".join(names[i] for i in group) or "None" for group in subsets],
    "Walks wholly inside": [subset_values[group] for group in subsets],
})
checks = node_flipping(model, window, {"GNN-LRP": walks}, random_orders=0)
adding = checks[(checks.task == "activation") & (checks.ranking == "GNN-LRP, walks")].iloc[0]
removing = checks[(checks.task == "pruning") & (checks.ranking == "GNN-LRP, walks")].iloc[0]

def present_after(order, adding_sensors):
    """Record every sensor mask before and after each change."""
    present = np.full(3, not adding_sensors)
    masks = [present.copy()]
    for sensor in order:
        present[sensor] = adding_sensors
        masks.append(present.copy())
    return np.stack(masks)

adding_order = [int(i) for i in adding["order"].split()]
removing_order = [int(i) for i in removing["order"].split()]
adding_masks = present_after(adding_order, True)
removing_masks = present_after(removing_order, False)
adding_scores = subgraph_scores(model, window, adding_masks)
removing_scores = subgraph_scores(model, window, removing_masks)
adding_curve = pd.DataFrame({
    "Step": range(4), "Sensor added": ["Start"] + [names[i] for i in adding_order],
    "Sensors present": [", ".join(np.array(names)[mask]) or "None" for mask in adding_masks],
    "Actual score": adding_scores,
})
removing_curve = pd.DataFrame({
    "Step": range(4), "Sensor removed": ["Start"] + [names[i] for i in removing_order],
    "Sensors present": [", ".join(np.array(names)[mask]) or "None" for mask in removing_masks],
    "Actual score": removing_scores,
    "Distance from original": np.abs(removing_scores - score),
})
curve_summary = pd.DataFrame({
    "Task": ["Adding", "Removing"],
    "Mean of": ["Score", "Absolute distance from original score"],
    "Value": [float(adding["aufc"]), float(removing["aufc"])],
})
assert adding_order == [1, 0, 2]
np.testing.assert_allclose(adding["aufc"], adding_scores.mean(), atol=1e-12)
np.testing.assert_allclose(removing["aufc"], np.abs(removing_scores - score).mean(), atol=1e-12)
```

</details>

#### Output

The adding check records the actual score at each step:

<!-- sample: adding_curve -->
| Step | Sensor added | Sensors present | Actual score |
| --- | --- | --- | --- |
| 0 | Start | None | 0.500000 |
| 1 | T30 | T30 | 0.538208 |
| 2 | T24 | T24, T30 | 0.500000 |
| 3 | T50 | T24, T30, T50 | 0.220177 |


The removing check follows its own order and records distance from the
original score:

<!-- sample: removing_curve -->
| Step | Sensor removed | Sensors present | Actual score | Distance from original |
| --- | --- | --- | --- | --- |
| 0 | Start | T24, T30, T50 | 0.220177 | 0.000000 |
| 1 | T24 | T30, T50 | 0.344629 | 0.124452 |
| 2 | T30 | T50 | 0.091197 | 0.128980 |
| 3 | T50 | None | 0.500000 | 0.279823 |


Each summary is the mean of four curve values. For adding:
(0.500000 + 0.538208 +
0.500000 + 0.220177) ÷ 4 =
**0.439596**.

<!-- sample: curve_summary -->
| Task | Mean of | Value |
| --- | --- | --- |
| Adding | Score | 0.439596 |
| Removing | Absolute distance from original score | 0.133314 |


This is an arithmetic check of one demonstration. To assess trained networks,
the project compares curves with random sensor orders and simpler rankings,
across windows and repeated training runs. The stored check uses 20 random
orders per window (assumption A-33). The two mean values above are not a
claim about prediction accuracy or superiority to other methods.

---

## What the explanation does and does not show

- **A model route, not a physical cause.** A negative share contributes a
  negative term to the score. It does not mean that sensor damages an engine
  or reduces its actual remaining life.
- **Score shares, not cycles.** The final curved conversion acts on the whole
  score. Applying it separately to each share would not preserve their sum.
- **Existing contributions, not removal effects.** Step 2 holds the original
  calculation fixed. Step 4 changes connections and recalculates switches.
  Removing one connection and rescaling the graph again is a different
  intervention still (assumption A-30).
- **Starting at graph inputs.** With the newest-reading stage, each start is
  a known cycle. With an LSTM, walks start at numbers computed from memory;
  this method does not divide them back among earlier cycles.
- **Groups the graph cannot distinguish.** Sensors with identical receiving
  rows can acquire identical graph outputs. The playground combines such
  route endings rather than treating their arbitrary split as evidence of
  a difference (finding F-18).
- **The selected network, not the value of its graph design.** Showing which
  routes it uses does not establish that its graph is better than another.
  That requires separate trained comparisons (assumption A-27).

Keep a saved run's network settings, selected engine and cycle, and gamma
schedule beside its explanation. The precise claim here is: **T50 → T30
carries 0.272535 of this demonstration's score at
engine 5's last available cycle.** The inputs and rule tell us exactly what
that statement refers to.
