# Dynamic SHAP

A prediction combines information from several sensors. To understand one
prediction, ask **how much each sensor's actual readings changed it from a
specified reference**. SHAP, short for SHapley Additive exPlanations, divides
that difference among the sensors (Lundberg & Lee, 2017, p. 1; p. 4, Eq. 5).

## Dividing credit fairly

Two sensors can matter together in a way neither does alone (Lundberg & Lee,
2017, p. 3). Giving each sensor all of the change caused by removing it can
therefore count the same interaction more than once.

Instead, imagine adding the sensors one at a time. Record how much the
prediction changes when a sensor joins. Repeat for every possible order, then
average that sensor's changes. That average is its **Shapley value**
(Lundberg & Lee, 2017, p. 4; p. 5, Fig. 1).

{{figure:shapley}}

The figure illustrates why order matters: B may add a different amount when
it arrives alone than when A is already present. The project uses this same
calculation for sensors, as set out in the paper's Equation 15, and repeats it
through time as in Equation 16 (Wen et al., 2026, Section 2.4, Eqs. 15–16).

A SHAP explanation uses the model's **inputs and predictions**. It need not
read its internal multipliers or follow its graph connections. It keeps the
trained model fixed and changes which sensor readings it presents to it
(Lundberg & Lee, 2017, pp. 4–5).

## Example setup

We follow **FD001 test engine 5**, using the T24, T30 and T50 readings at its
last three recorded cycles. They have already been smoothed (Wen et al., 2026,
Section 3.2.3, Eq. 18; assumption A-04) and rescaled with statistics from the
training engines selected using seed 42 (assumptions A-06 and A-25). The
numbers therefore describe readings relative to their training means and
spreads (assumption A-03); they are not temperatures in their original units.

This is a **small, untrained demonstration**, shared with the
[GNN-LRP tutorial](11_gnn_lrp.md). It has three sensors, one graph layer with
one value per sensor, and deliberately visible multipliers. Its temporal
route keeps each sensor's last reading (assumption A-07). The earlier two
readings are shown so the rule for replacing a sensor's entire history is
visible. These settings teach the arithmetic; they are not the full model, a
trained result, or a claim about engine health. The full explanation uses all
fourteen sensors (Wen et al., 2026, Table 3) and the window length saved with
the run.

All input and output columns of this small example are shown. Tables round
to six decimal places; calculations use full precision. The optional Python
examples run in page order from the repository root with the real FD001 files.

<details class="process-example">
<summary>Set up the Python examples</summary>

```python
from math import factorial
import numpy as np
import pandas as pd
import torch
from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.data.windowing import WindowedDataset
from thesis_gnn_rul_xai.explain.shap import (
    coalition_weights, dynamic_shap, efficiency_error,
    mean_window_baseline, shap_values,
)
from thesis_gnn_rul_xai.graph.adjacency import build_physics_adjacency, normalize_adjacency
from thesis_gnn_rul_xai.models import ESTGNN

prepared = run_preprocessing(PreprocessingConfig(seed=42))
names = ["T24", "T30", "T50"]
chosen = [list(prepared.sensor_symbols).index(name) for name in names]
index = int(np.flatnonzero(prepared.test.units == 5)[0])
features = prepared.test.features[index, chosen, -3:]
end_cycle = int(prepared.test.end_cycles[index])
training_windows = prepared.train.features[:, chosen, -3:]
raw_graph, _ = build_physics_adjacency()
small_graph = raw_graph[np.ix_(chosen, chosen)]
with torch.random.fork_rng():
    torch.manual_seed(0)
    model = ESTGNN(normalize_adjacency(small_graph, method="row"),
                   window_size=3, gcn_hidden=(1,), fc_hidden=1)
with torch.no_grad():
    model.graph_layers[0].weight.fill_(1)
    model.fc1.weight.copy_(torch.tensor([[2., 1., -1.]]))
    model.fc1.bias.fill_(0.5)
    model.fc2.weight.fill_(1)
    model.fc2.bias.zero_()
model.eval()

def history_table(values, cycles):
    """Show every reading in the three-sensor example."""
    table = pd.DataFrame(values.T, columns=names)
    table.insert(0, "Cycle", cycles)
    return table

actual_window = history_table(features, range(end_cycle - 2, end_cycle + 1))
training_sums = training_windows.sum(axis=0, dtype=np.float64)
reference_input = pd.DataFrame(training_sums,
    columns=["Sum at first position", "Sum at middle position", "Sum at last position"])
reference_input.insert(0, "Sensor", names)
reference_input["Training windows"] = len(training_windows)
```

</details>

<div class="algorithm-intro">
<p class="algorithm-label">Dynamic SHAP algorithm</p>
<p>Start with a fixed model, one window to explain, and training windows for
comparison. Follow steps 1–5: choose the reference, evaluate every group of
sensors, divide the prediction change, check the total, then repeat along an
engine's history.</p>
</div>

---

## 1 — Choose a reference window

The model always expects the same number of sensor inputs. A sensor that is
"absent" from an explanation still needs something in its place, since most
models "cannot handle arbitrary patterns of missing input values" (Lundberg &
Lee, 2017, p. 5). Here its actual history is replaced by its **average
training window**: for each position in the window, average that sensor's
reading at that position across the training windows.

This reference answers a particular question: what changes when the model sees
this engine's readings instead of the reference readings? The paper's
Equation 15 needs predictions from groups of sensors but does not specify
these replacement values (Wen et al., 2026, Section 2.4, Eq. 15). The
training-window average is this project's choice (assumption A-20).

### Process

#### Input

The complete small window to explain is:

<!-- sample: actual_window -->
| Cycle | T24 | T30 | T50 |
|---|---|---|---|
| 96 | 0.285568 | 0.143344 | -0.050896 |
| 97 | 0.194873 | 0.328677 | 0.359872 |
| 98 | -0.373355 | 0.114623 | 0.817606 |

The reference is built separately from the training windows. This table shows
the sums and counts needed to calculate **every** reference value in the
three-sensor example. Each sum uses the same relative position across all
training windows, not successive readings from the test engine.

<!-- sample: reference_input -->
| Sensor | Sum at first position | Sum at middle position | Sum at last position | Training windows |
|---|---|---|---|---|
| T24 | 1075.329490 | 1320.236446 | 1563.394170 | 14459 |
| T30 | 1137.202889 | 1372.523602 | 1611.045623 | 14459 |
| T50 | 1166.284838 | 1414.503699 | 1675.003476 | 14459 |

#### Function

Divide each sum by the number of training windows. For example, the first
position of T24 becomes 1075.329490 ÷
14,459 = 0.074371. The other positions
and sensors receive the same operation with their own sums.

Use only the training windows. Do not add validation or test windows when
building the reference for a particular explanation. A training engine can
be explained using this same fixed reference, as in step 5.
The reference need not equal zero: overlapping windows give readings a
different weighting from the individual rows used to rescale the data.

<details class="process-example">
<summary>Example in Python</summary>

```python
baseline = mean_window_baseline(prepared.train)[chosen, -3:]
reference_window = history_table(baseline, ["First position", "Middle position", "Last position"])
reference_window = reference_window.rename(columns={"Cycle": "Window position"})
np.testing.assert_allclose(baseline, training_sums / len(training_windows), atol=1e-6)
```

</details>

#### Output

One reference history for every sensor, with the same three positions as the
actual window. These are comparison values, not an observed engine history.

<!-- sample: reference_window -->
| Window position | T24 | T30 | T50 |
|---|---|---|---|
| First position | 0.074371 | 0.078650 | 0.080661 |
| Middle position | 0.091309 | 0.094925 | 0.097829 |
| Last position | 0.108126 | 0.111422 | 0.115845 |

---

## 2 — Predict with every group of sensors

For each group, keep the actual histories of the sensors in that group and
replace all other histories with the reference. The model and its graph stay
fixed. Only the sensor readings change.

With three sensors there are eight groups, including none and all three.
This allows us to see what a sensor adds both alone and in combination with
others.

### Process

#### Input

The actual histories from step 1:

<!-- sample: actual_window -->
| Cycle | T24 | T30 | T50 |
|---|---|---|---|
| 96 | 0.285568 | 0.143344 | -0.050896 |
| 97 | 0.194873 | 0.328677 | 0.359872 |
| 98 | -0.373355 | 0.114623 | 0.817606 |

The matching reference histories from step 1:

<!-- sample: reference_window -->
| Window position | T24 | T30 | T50 |
|---|---|---|---|
| First position | 0.074371 | 0.078650 | 0.080661 |
| Middle position | 0.091309 | 0.094925 | 0.097829 |
| Last position | 0.108126 | 0.111422 | 0.115845 |

#### Function

Build one complete input per group and ask the same model for a prediction.
For the group **T24 + T50**, keep their actual columns, but replace **all three**
T30 readings with T30's reference column. This does not delete a sensor node
or any of its connections.

The model in this example keeps only the last position before exchanging
information between sensors. It follows the connections the paper draws
among these three sensors (Wen et al., 2026, Table 3) and averages what each
sensor receives, its own reading included (Wen et al., 2026, Section 2.3.2,
Eq. 8; assumption A-21). Its first
sensor uses the average of T24 and T30; its second uses the average of all
three; its third uses the average of T30 and T50. Negative averages become
zero. It then doubles the first result, adds the second, subtracts the third,
and adds 0.5. Negative totals become zero, and the final conversion produces a
number between zero and the ceiling of 125 cycles (assumption A-01). The
[GNN-LRP page](11_gnn_lrp.md) traces these same calculations.
SHAP only needs the resulting predictions.

<details class="process-example">
<summary>Example in Python</summary>

```python
codes = np.arange(2 ** len(names))
masks = ((codes[:, None] >> np.arange(len(names))) & 1).astype(bool)
masked_windows = np.where(masks[:, :, None], features[None], baseline[None])
with torch.no_grad():
    predictions = model.predict_cycles(torch.from_numpy(masked_windows)).numpy().astype(np.float64)
groups = [" + ".join(name for name, keep in zip(names, mask) if keep) or "None"
          for mask in masks]
group_predictions = pd.DataFrame(masked_windows[:, :, -1], columns=names)
group_predictions.insert(0, "Actual sensors", groups)
group_predictions["Prediction (cycles)"] = predictions
```

</details>

#### Output

All eight predictions. The three sensor columns show the **last reading actually
received** in each case, because this example's temporal route uses only that
position. The complete three-position inputs were built using the same
keep-or-replace choice for each sensor.

<!-- sample: group_predictions -->
| Actual sensors | T24 | T30 | T50 | Prediction (cycles) |
|---|---|---|---|---|
| None | 0.108126 | 0.111422 | 0.115845 | 84.012863 |
| T24 | -0.373355 | 0.111422 | 0.115845 | 74.425964 |
| T30 | 0.108126 | 0.114623 | 0.115845 | 84.086311 |
| T24 + T30 | -0.373355 | 0.114623 | 0.115845 | 74.377762 |
| T50 | 0.108126 | 0.111422 | 0.817606 | 80.728539 |
| T24 + T50 | -0.373355 | 0.111422 | 0.817606 | 69.369339 |
| T30 + T50 | 0.108126 | 0.114623 | 0.817606 | 80.804764 |
| T24 + T30 + T50 | -0.373355 | 0.114623 | 0.817606 | 69.352867 |

"None" still makes a prediction: it means all sensors are at their reference
histories. That prediction is the starting point for the attribution sum.

---

## 3 — Average what each sensor adds

A sensor's contribution depends on which other sensors are already present.
Its Shapley value averages over **all arrival orders**, giving each order equal
weight. It is not just the effect of removing the sensor from the full input.

### Process

#### Input

The complete table of group predictions from step 2:

<!-- sample: group_predictions -->
| Actual sensors | T24 | T30 | T50 | Prediction (cycles) |
|---|---|---|---|---|
| None | 0.108126 | 0.111422 | 0.115845 | 84.012863 |
| T24 | -0.373355 | 0.111422 | 0.115845 | 74.425964 |
| T30 | 0.108126 | 0.114623 | 0.115845 | 84.086311 |
| T24 + T30 | -0.373355 | 0.114623 | 0.115845 | 74.377762 |
| T50 | 0.108126 | 0.111422 | 0.817606 | 80.728539 |
| T24 + T50 | -0.373355 | 0.111422 | 0.817606 | 69.369339 |
| T30 + T50 | 0.108126 | 0.114623 | 0.817606 | 80.804764 |
| T24 + T30 + T50 | -0.373355 | 0.114623 | 0.817606 | 69.352867 |

#### Function

To explain T24, pair every group without T24 with the matching group containing
T24. Subtract the prediction before it joins from the prediction afterwards.

Three sensors can arrive in six orders. T24 comes first in two orders,
second after T30 in one, second after T50 in one, and last in two. Therefore
the four changes receive weights 2/6, 1/6, 1/6 and 2/6. This counts each
arrival order once without recalculating the same group prediction.

<!-- sample: marginal_table -->
| Group before T24 | Before (cycles) | After (cycles) | Change (cycles) | Weight | Weighted change |
|---|---|---|---|---|---|
| None | 84.012863 | 74.425964 | -9.586899 | 0.333333 | -3.195633 |
| T30 | 84.086311 | 74.377762 | -9.708549 | 0.166667 | -1.618092 |
| T50 | 80.728539 | 69.369339 | -11.359200 | 0.166667 | -1.893200 |
| T30 + T50 | 80.804764 | 69.352867 | -11.451897 | 0.333333 | -3.817299 |

For example, when T24 joins T30, the prediction changes by
74.377762 − 84.086311 =
-9.708549 cycles. Multiply by 1/6 to get
-1.618092 cycles toward T24's final share.
Adding all four weighted changes gives -10.524223 cycles.
Repeat the same pairing and averaging for T30 and T50.

The general rule in the paper's Equation 15 (Wen et al., 2026, Section 2.4,
Eq. 15) gives each group a weight based on how many orders put exactly that
group before the sensor. The full model
has 87,178,291,200 arrival orders but only 16,384 distinct groups.
The project evaluates all those groups, so there is no sampling of arrival
orders. **Exact** describes this calculation for the chosen model and
reference; it does not make the reference uniquely correct or the model accurate.

<details class="process-example">
<summary>Example in Python</summary>

```python
without_t24 = np.flatnonzero(~masks[:, 0])
with_t24 = without_t24 | 1
weights = coalition_weights(len(names))[masks[without_t24].sum(axis=1)]
changes = predictions[with_t24] - predictions[without_t24]
marginal_table = pd.DataFrame({
    "Group before T24": [groups[i] for i in without_t24],
    "Before (cycles)": predictions[without_t24],
    "After (cycles)": predictions[with_t24],
    "Change (cycles)": changes,
    "Weight": weights,
    "Weighted change": weights * changes,
})
values = shap_values(model, features, baseline)
contributions = pd.DataFrame({"Sensor": names, "SHAP contribution (cycles)": values})
np.testing.assert_allclose(values[0], np.sum(weights * changes), atol=1e-5)
```

</details>

#### Output

One signed contribution per sensor:

<!-- sample: contributions -->
| Sensor | SHAP contribution (cycles) |
|---|---|
| T24 | -10.524223 |
| T30 | 0.023663 |
| T50 | -4.159435 |

A positive value raises the predicted life relative to the reference; a negative
one lowers it. The sign describes the model's prediction, not whether a sensor
is physically good or bad. T24's negative contribution here is an average of
its changes across different groups, not a claim that changing T24 alone
would produce that exact difference.

**Do not confuse SHAP with occlusion.** Occlusion, which Zeiler and Fergus
(2014, Section 4.2, p. 823) applied to images by covering one region with a
grey square and watching the output, here replaces one sensor while keeping
every other sensor actual. For T24 that uses only the last row of the
calculation table, which gives -11.451897 cycles here. SHAP combines
all four rows. Agreement in a particular example would not make the two
methods interchangeable.

---

## 4 — Check that the contributions add up

Each arrival order starts from the all-reference input and ends with the
all-actual input. The changes along that order therefore add up to the same
overall prediction difference. Averaging those changes over orders preserves
the total. Lundberg and Lee (2017, p. 4, Eq. 5) call this adding-up property
local accuracy.

### Process

#### Input

The sensor contributions from step 3:

<!-- sample: contributions -->
| Sensor | SHAP contribution (cycles) |
|---|---|
| T24 | -10.524223 |
| T30 | 0.023663 |
| T50 | -4.159435 |

Their starting point is the prediction for "None" in step 2,
84.012863 cycles.

#### Function

Add the sensor contributions to the reference prediction:

> 84.012863 + (-10.524223) + (0.023663) +
> (-4.159435) = 69.352867 cycles.

Compare this with the model's prediction for the all-actual window. The
difference should be zero apart from numerical rounding. Use the **prediction
at the reference window**, not the average remaining-life label, which is
what the paper's Equation 14 adds the contributions to (Wen et al., 2026,
Section 2.4, Eq. 14): only the former is the starting point of the group
calculations (assumption A-24).

<details class="process-example">
<summary>Example in Python</summary>

```python
reconstructed = predictions[0] + values.sum()
residual = efficiency_error(model, features, baseline, values)
check_table = pd.DataFrame({
    "Reference prediction": [predictions[0]],
    "Sum of contributions": [values.sum()],
    "Reconstructed prediction": [reconstructed],
    "Actual prediction": [predictions[-1]],
    "Absolute difference": [residual],
})
assert residual < 1e-4
```

</details>

#### Output

The complete check, in cycles:

<!-- sample: check_table -->
| Reference prediction | Sum of contributions | Reconstructed prediction | Actual prediction | Absolute difference |
|---|---|---|---|---|
| 84.012863 | -14.659996 | 69.352867 | 69.352867 | 0.000000 |

This checks the attribution arithmetic. It does not check prediction accuracy.
Our deliberately untrained example also passes it. Evaluating the model against
known remaining-life answers is a separate task.

---

## 5 — Repeat along an engine's history

**Dynamic SHAP** repeats the same explanation at successive windows of one
engine. Wen et al. (2026), the paper this project re-implements, compute it
"on each time step of the prediction sequence" (p. 7, Eq. 16) and report it
for "a representative engine unit" (p. 13); reading each time step as the
next window, one cycle later, is this project's choice (assumption A-35).
Keep the model and reference fixed
so each row answers the same question at a different time. No additional
model training happens here.

### Process

#### Input

To show three consecutive explanations, now use **training engine 1**, with
windows ending at cycles 30, 31 and 32. Every reading supplied to the small
three-cycle demonstrator is shown. Overlap is intentional: the window moves
forward one cycle at a time (assumption A-05).

<!-- sample: trajectory_input -->
| Window ends | Cycle | T24 | T30 | T50 |
|---|---|---|---|---|
| 30 | 28 | -0.692949 | -0.739689 | -1.142521 |
| 30 | 29 | -1.235571 | -0.918714 | -1.045265 |
| 30 | 30 | -1.169914 | -0.159372 | -1.293769 |
| 31 | 29 | -1.235571 | -0.918714 | -1.045265 |
| 31 | 30 | -1.169914 | -0.159372 | -1.293769 |
| 31 | 31 | -1.346239 | -0.692049 | -1.368398 |
| 32 | 30 | -1.169914 | -0.159372 | -1.293769 |
| 32 | 31 | -1.346239 | -0.692049 | -1.368398 |
| 32 | 32 | -1.074194 | -0.255078 | -1.196328 |

The reference remains the training-window average from step 1. The test data
prepared for evaluation contains only the last window of each engine, the
point at which the benchmark asks for "the number of operational cycles after
the last cycle that the engine will continue to operate" (NASA Prognostics
Center of Excellence [NASA PCoE], 2008). That single window alone cannot show
a time history.

#### Function

For the window ending at cycle 30, evaluate the eight groups and average each
sensor's added changes. Repeat for the window ending at cycle 31, then for the
window ending at cycle 32. Each input window stops at its own end cycle.
The rescaling statistics and reference were fitted beforehand from the full
training histories, which include later cycles of this training engine. This
is a retrospective demonstration, not an evaluation that trains on only the
history available at each of those cycles. Validation and test engines do not
enter those fitted training statistics or the reference (assumptions A-20 and
A-25).

For each row, preserve the reference prediction, the actual prediction and
all three contributions. This lets the sum check from step 4 be applied
separately at every time.

<details class="process-example">
<summary>Example in Python</summary>

```python
indices = np.flatnonzero(prepared.train.units == 1)[:3]
trajectory_features = prepared.train.features[indices][:, chosen, -3:]
trajectory_rows = []
for end, window in zip(prepared.train.end_cycles[indices], trajectory_features):
    for offset, readings in enumerate(window.T):
        trajectory_rows.append([int(end), int(end) - 2 + offset, *readings])
trajectory_input = pd.DataFrame(trajectory_rows, columns=["Window ends", "Cycle", *names])
trajectory_dataset = WindowedDataset(
    features=trajectory_features,
    targets=prepared.train.targets[indices],
    units=prepared.train.units[indices],
    end_cycles=prepared.train.end_cycles[indices],
    sensor_columns=tuple(prepared.train.sensor_columns[i] for i in chosen),
)
trajectory = dynamic_shap(model, trajectory_dataset, baseline, unit=1, symbols=names)
trajectory_output = trajectory[["end_cycle", "baseline", "predicted", *names]].rename(columns={
    "end_cycle": "Window ends", "baseline": "Reference prediction", "predicted": "Prediction",
})
np.testing.assert_allclose(trajectory[names].sum(axis=1) + trajectory["baseline"],
                           trajectory["predicted"], atol=1e-4)
```

</details>

#### Output

One explanation per window. All predictions and sensor contributions are in
cycles; "Window ends" identifies when the input history stops.

<!-- sample: trajectory_output -->
| Window ends | Reference prediction | Prediction | T24 | T30 | T50 |
|---|---|---|---|---|---|
| 30 | 84.012863 | 77.807419 | -4.343442 | -2.337285 | 0.475283 |
| 31 | 84.012863 | 77.807419 | -4.238299 | -2.547572 | 0.580427 |
| 32 | 84.012863 | 77.807419 | -4.238299 | -2.547572 | 0.580427 |

Here the prediction stays the same while some contributions change: the
changes in their shares cancel. An unchanged prediction does not require an
unchanged explanation.

A colour map can show this same table with time in one direction and sensors
in the other. Keep the sign visible when asking which sensors raised or lowered
a prediction. Taking the absolute value answers a different question: how large
a contribution was, regardless of direction.

---

## What an explanation does and does not establish

The full playground explanation uses the selected **saved run and saved
seed's model**, its matching preprocessing, and all fourteen sensors. Changing
the model, the input window or the reference can change the explanation.
A sensor ranking from one trained model is not a general result across training
runs.

An earlier version of this page quoted two things from a single trained
network: how the ranking of the sensors changes with the reference, and where
the two-value sensor P15 ranks. Both have since been measured on five
networks, trained with five different seeds (42 to 46), which set their random
starting values and the order in which they see the training windows. All five
share one configuration: the default settings, except that no training engine
is held back, so every network learns from the same 100 engines for all 200
passes over them, and the network as it stands after the last pass is the one
explained. Ranking the sensors by the average size of their contributions,
sign set aside:

- **The same four sensors lead on every network**, Nc, T50, W31 and T24,
  whichever of three references stands in for an absent sensor: the average
  training window used here, the median training window, or zeros.
- **P15 comes last of the fourteen in every case measured**: on 20 test
  windows under each of the three references, and over every window of three
  training engines' lives, on all five networks.
- **The sizes are not settled.** The largest contribution P15 makes over one
  engine's life varies by tens of per cent across the five networks: of the
  five, the smallest is between 36 and 50 per cent below the largest,
  depending on the engine. Two of them can still agree closely: on one engine,
  seeds 42 and 46 give the same 1.675 cycles to three decimal places.

So the top and the bottom of the ranking hold across seeds for this one
configuration; the exact sizes do not, and no other configuration has been
measured this way. The settings, the tables and the code that produces them
are in the records for
[the reference choice](../assumptions/A-20.md)
(assumption A-20) and
[the two-value sensor](../findings/F-05.md)
(finding F-05).

(Corrected 2026-10-03: this said both results came from one network and were
limited case studies, not established behaviour across seeds. The five-network
measurements were added to the two records that day, and a review found this
page had not followed them.)

Replacing sensor histories can also create combinations not observed together
in a real engine. Lundberg and Lee (2017, p. 5, Eqs. 11–12) arrive at this
kind of stand-in, the average put in place of the absent inputs, by assuming
two things: that the inputs are independent of one another, which sets aside
how sensors move together, and that the model is linear, its output a
weighted sum of its inputs plus a constant. SHAP describes
how the model responds to those combinations under this reference choice. It
does not prove physical causation or identify a failed component. Nor does it
provide a confidence interval for remaining life: the paper describes its own
SHAP analysis as working "without addressing prediction uncertainty or
confidence intervals" (Wen et al., 2026, p. 7). A large contribution is a
statement about a model's calculation, not a diagnosis.

To follow routes **inside** the network instead of replacing its inputs,
continue to [GNN-LRP](11_gnn_lrp.md). Its scores answer a different question and
use a different unit.

## References

- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. In I. Guyon, U. von Luxburg, S. Bengio, H. Wallach,
  R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), *Advances in Neural
  Information Processing Systems 30 (NIPS 2017)* (pp. 4765–4774).
  https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
  [Copy](../references/lundberg_2017_unified_shap.pdf). Page numbers on this
  page are those printed on the copy held, which numbers its pages 1 to 10.
- NASA Prognostics Center of Excellence. (2008). readme.txt, the description
  of the C-MAPSS data set distributed inside its zip file. NASA Prognostics
  Data Repository.
  [Copy](../references/nasa_2008_cmapss_readme.txt)
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and
  Technology*, *168*, 110875. https://doi.org/10.1016/j.ast.2025.110875
  [Copy](../references/wen_2026_estgnn_dynamic_shap.pdf)
- Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding
  convolutional networks. In D. Fleet, T. Pajdla, B. Schiele, &
  T. Tuytelaars (Eds.), *Computer Vision – ECCV 2014* (Lecture Notes in
  Computer Science, Vol. 8689, pp. 818–833). Springer.
  https://doi.org/10.1007/978-3-319-10590-1_53
  [Copy](../references/zeiler_2014_visualizing_convnets.pdf)
