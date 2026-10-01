# Architecture and training

The [GNN page](02_graphs.md) builds the table of permitted sensor connections
and combining shares. This page follows sensor values through the layers that
use it, until one window has one remaining-life prediction.

There are two separate operations in a graph layer: the **fixed graph shares**
combine allowed senders, then **learned multipliers** turn that mixture into
new values. Training adjusts the latter, not the graph table.

## The whole route

{{figure:architecture-overview}}

The figure shows the command-line default network, whose temporal stage keeps the last
reading. The paper's Table 4 specifies graph widths 32, 64 and 32, followed by
a fully connected width of 128 and one output. Here **width** means how many
calculated values a sensor or window carries, not how many cycles are read.
The paper describes separate LSTMs before these layers in Section 2.2; that
stage is omitted by the command-line default (assumption A-11).

## Example setup

To show **every input and output value**, this tutorial uses graph widths
**3, 2, 1**, and a fully connected width of **3**. It uses the real Physics
graph and FD001 training engine 1, cycles 1–30, with the last-reading temporal
route. These smaller widths are teaching settings, not the paper's settings
or a change to the playground defaults.

This is an **untrained demonstration**. Its parameters come from the normal
initializer with seed 0, chosen so nonzero values can be followed through all
three graph layers. The output illustrates arithmetic, not useful prediction
accuracy. [Follow one real example](08_worked_examples.md#through-the-saved-model)
retains the separate walkthrough of a named, saved trained network.

All fourteen sensor rows and all calculated columns are shown. Each Output
is the next step's Input. Numbers are rounded
to six decimal places for reading; calculations use full precision.

The optional setup and examples run in page order from the repository root,
with the real NASA FD001 files in `data/raw/`.

<details class="process-example">
<summary>Set up the Python examples</summary>

```python
import numpy as np
import pandas as pd
import torch
from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.graph.adjacency import build_physics_adjacency, normalize_adjacency
from thesis_gnn_rul_xai.models import ESTGNN
from thesis_gnn_rul_xai.models.temporal import LastCycle

prepared = run_preprocessing(PreprocessingConfig(seed=42))
index = int(np.flatnonzero(prepared.train.units == 1)[0])
assert prepared.train.end_cycles[index] == 30
window = torch.from_numpy(prepared.train.features[index:index + 1])
raw_graph, names = build_physics_adjacency()
names = list(names)
with torch.random.fork_rng():
    torch.manual_seed(0)
    model = ESTGNN(normalize_adjacency(raw_graph, method="row"),
                   temporal=LastCycle(30), gcn_hidden=(3, 2, 1), fc_hidden=3)
model.eval()

def sensors(values):
    """Show all sensor rows and all value columns for this window."""
    frame = pd.DataFrame(values[0].detach().numpy(),
                         columns=[f"Value {i + 1}" for i in range(values.shape[-1])])
    frame.insert(0, "Sensor", names)
    return frame

def learned_table(weight):
    """Show every input-to-output multiplier in a graph layer."""
    frame = pd.DataFrame(weight.detach().numpy(),
                         columns=[f"Output {i + 1}" for i in range(weight.shape[1])])
    frame.insert(0, "Input value", range(1, len(frame) + 1))
    return frame

inputs = model.temporal(window)
input_table = sensors(inputs)
```

</details>

<div class="algorithm-intro">
<p class="algorithm-label">Prediction algorithm</p>
<p>Start with one value per sensor from the temporal route and the rescaled
graph. Follow steps 1–7 in order: exchange sensor information three times,
place the results in one list, combine the list, obtain a life fraction,
then convert it to cycles.</p>
</div>

---

## 1 — Graph layer 1

Let each sensor use its neighbours’ information before calculating several new values.

### Process

#### Input

Every sensor carries 1 value in this example. A second input is the same fixed fourteen-by-fourteen combining table from [Rescale the graph](02_graphs.md#2--rescale-the-graph). Its rows receive from its columns.

<!-- sample: input_table -->
| Sensor | Value 1 |
|---|---|
| T24 | -1.169914 |
| T30 | -0.159372 |
| T50 | -1.293769 |
| P15 | 0.227969 |
| P30 | 1.176178 |
| Nf | -0.433867 |
| Nc | -0.833875 |
| Ps30 | -0.732719 |
| phi | 1.256691 |
| NRf | -0.900765 |
| BPR | -0.858223 |
| htBleed | -1.544337 |
| W31 | 1.158283 |
| W32 | 0.977928 |

#### Function

First multiply every permitted sender's values by its fixed graph share and add them at the receiver, separately for each input column. Do this for every sensor using the previous layer's values.


T24's complete first combination is below. Every allowed sender is present; there are no omitted nonzero terms. Add these products to get its mixed value `-0.356339`.

<!-- sample: neighbour_terms -->
| Sender | Reading | Graph share | Product |
|---|---|---|---|
| T24 | -1.169914 | 0.125000 | -0.146239 |
| T30 | -0.159372 | 0.125000 | -0.019922 |
| P15 | 0.227969 | 0.125000 | 0.028496 |
| P30 | 1.176178 | 0.125000 | 0.147022 |
| Nf | -0.433867 | 0.125000 | -0.054233 |
| Ps30 | -0.732719 | 0.125000 | -0.091590 |
| NRf | -0.900765 | 0.125000 | -0.112596 |
| BPR | -0.858223 | 0.125000 | -0.107278 |

The complete combined table is:

<!-- sample: layer1_mixtures -->
| Sensor | Value 1 |
|---|---|
| T24 | -0.356339 |
| T30 | -0.238093 |
| T50 | -0.030161 |
| P15 | -0.626960 |
| P30 | -0.106133 |
| Nf | -0.626960 |
| Nc | 0.000556 |
| Ps30 | -0.106133 |
| phi | 0.045836 |
| NRf | -0.626960 |
| BPR | -0.626960 |
| htBleed | 0.162347 |
| W31 | 0.000556 |
| W32 | -0.307154 |

Next multiply each combined row by the following table. One column supplies the multipliers for one new output. The same table is used at every sensor; each output is the sum of its products across all input values.

<!-- sample: layer1_weights -->
| Input value | Output 1 | Output 2 | Output 3 |
|---|---|---|---|
| 1 | -0.009169 | 0.657007 | -1.008020 |

For T24's first output, every product is:

<!-- sample: layer1_terms -->
| Input value | T24 mixed value | Multiplier for output 1 | Product |
|---|---|---|---|
| 1 | -0.356339 | -0.009169 | 0.003267 |

Their sum is `0.003267`. Finally apply **ReLU**, meaning keep a nonnegative total and replace a negative total with zero. T24's first output is therefore `0.003267`. This demonstration adds no graph-layer offset, matching the project's default choice (assumption A-12).

<details class="process-example">
<summary>Example in Python</summary>

```python
with torch.no_grad():
    layer1_mixed = model.adjacency @ inputs
    layer1_before_zero = model.graph_layers[0](inputs, model.adjacency)
    layer1 = torch.relu(layer1_before_zero)
layer1_mixtures = sensors(layer1_mixed)
layer1_weights = learned_table(model.graph_layers[0].weight)
layer1_terms = pd.DataFrame({"Input value": range(1, 2),
    "T24 mixed value": layer1_mixed[0, 0].numpy(),
    "Multiplier for output 1": model.graph_layers[0].weight[:, 0].detach().numpy(),
    "Product": (layer1_mixed[0, 0] * model.graph_layers[0].weight[:, 0]).detach().numpy()})
torch.testing.assert_close(torch.tensor(layer1_terms["Product"].sum()), layer1_before_zero[0, 0, 0])
layer1_output = sensors(layer1)

senders = torch.nonzero(model.adjacency[0]).flatten()
neighbour_terms = pd.DataFrame({"Sender": [names[i] for i in senders.tolist()],
    "Reading": inputs[0, senders, 0].numpy(), "Graph share": model.adjacency[0, senders].numpy(),
    "Product": (inputs[0, senders, 0] * model.adjacency[0, senders]).numpy()})
torch.testing.assert_close(torch.tensor(neighbour_terms["Product"].sum()), layer1_mixed[0, 0, 0])
```

</details>

#### Output

All fourteen sensors, each now carrying 3 calculated values. These are neither new measurements nor cycle positions. The complete table proceeds to the next step.

<!-- sample: layer1_output -->
| Sensor | Value 1 | Value 2 | Value 3 |
|---|---|---|---|
| T24 | 0.003267 | 0.000000 | 0.359197 |
| T30 | 0.002183 | 0.000000 | 0.240002 |
| T50 | 0.000277 | 0.000000 | 0.030403 |
| P15 | 0.005749 | 0.000000 | 0.631989 |
| P30 | 0.000973 | 0.000000 | 0.106984 |
| Nf | 0.005749 | 0.000000 | 0.631989 |
| Nc | 0.000000 | 0.000366 | 0.000000 |
| Ps30 | 0.000973 | 0.000000 | 0.106984 |
| phi | 0.000000 | 0.030114 | 0.000000 |
| NRf | 0.005749 | 0.000000 | 0.631989 |
| BPR | 0.005749 | 0.000000 | 0.631989 |
| htBleed | 0.000000 | 0.106663 | 0.000000 |
| W31 | 0.000000 | 0.000366 | 0.000000 |
| W32 | 0.002816 | 0.000000 | 0.309618 |

---

## 2 — Graph layer 2

Combine the preceding layer’s values. A neighbour now carries information it received in layer 1, so an output can depend on sensors up to two connections away.

### Process

#### Input

Every sensor carries 3 values in this example. A second input is the same fixed fourteen-by-fourteen combining table from [Rescale the graph](02_graphs.md#2--rescale-the-graph). Its rows receive from its columns.

<!-- sample: layer1_output -->
| Sensor | Value 1 | Value 2 | Value 3 |
|---|---|---|---|
| T24 | 0.003267 | 0.000000 | 0.359197 |
| T30 | 0.002183 | 0.000000 | 0.240002 |
| T50 | 0.000277 | 0.000000 | 0.030403 |
| P15 | 0.005749 | 0.000000 | 0.631989 |
| P30 | 0.000973 | 0.000000 | 0.106984 |
| Nf | 0.005749 | 0.000000 | 0.631989 |
| Nc | 0.000000 | 0.000366 | 0.000000 |
| Ps30 | 0.000973 | 0.000000 | 0.106984 |
| phi | 0.000000 | 0.030114 | 0.000000 |
| NRf | 0.005749 | 0.000000 | 0.631989 |
| BPR | 0.005749 | 0.000000 | 0.631989 |
| htBleed | 0.000000 | 0.106663 | 0.000000 |
| W31 | 0.000000 | 0.000366 | 0.000000 |
| W32 | 0.002816 | 0.000000 | 0.309618 |

#### Function

First multiply every permitted sender's values by its fixed graph share and add them at the receiver, separately for each input column. Do this for every sensor using the previous layer's values.


The complete combined table is:

<!-- sample: layer2_mixtures -->
| Sensor | Value 1 | Value 2 | Value 3 |
|---|---|---|---|
| T24 | 0.003799 | 0.000000 | 0.417640 |
| T30 | 0.000853 | 0.015279 | 0.093730 |
| T50 | 0.001055 | 0.000146 | 0.116005 |
| P15 | 0.005253 | 0.000000 | 0.577430 |
| P30 | 0.000925 | 0.017189 | 0.101646 |
| Nf | 0.005253 | 0.000000 | 0.577430 |
| Nc | 0.000803 | 0.015279 | 0.088221 |
| Ps30 | 0.000925 | 0.017189 | 0.101646 |
| phi | 0.000590 | 0.019644 | 0.064853 |
| NRf | 0.005253 | 0.000000 | 0.577430 |
| BPR | 0.005253 | 0.000000 | 0.577430 |
| htBleed | 0.000868 | 0.017189 | 0.095449 |
| W31 | 0.000803 | 0.015279 | 0.088221 |
| W32 | 0.000619 | 0.021479 | 0.068004 |

Next multiply each combined row by the following table. One column supplies the multipliers for one new output. The same table is used at every sensor; each output is the sum of its products across all input values.

<!-- sample: layer2_weights -->
| Input value | Output 1 | Output 2 |
|---|---|---|
| 1 | -0.806181 | -0.421915 |
| 2 | 0.293752 | -0.021704 |
| 3 | 0.868567 | -0.097214 |

For T24's first output, every product is:

<!-- sample: layer2_terms -->
| Input value | T24 mixed value | Multiplier for output 1 | Product |
|---|---|---|---|
| 1 | 0.003799 | -0.806181 | -0.003063 |
| 2 | 0.000000 | 0.293752 | 0.000000 |
| 3 | 0.417640 | 0.868567 | 0.362749 |

Their sum is `0.359686`. Finally apply **ReLU**, meaning keep a nonnegative total and replace a negative total with zero. T24's first output is therefore `0.359686`. This demonstration adds no graph-layer offset, matching the project's default choice (assumption A-12).

<details class="process-example">
<summary>Example in Python</summary>

```python
with torch.no_grad():
    layer2_mixed = model.adjacency @ layer1
    layer2_before_zero = model.graph_layers[1](layer1, model.adjacency)
    layer2 = torch.relu(layer2_before_zero)
layer2_mixtures = sensors(layer2_mixed)
layer2_weights = learned_table(model.graph_layers[1].weight)
layer2_terms = pd.DataFrame({"Input value": range(1, 4),
    "T24 mixed value": layer2_mixed[0, 0].numpy(),
    "Multiplier for output 1": model.graph_layers[1].weight[:, 0].detach().numpy(),
    "Product": (layer2_mixed[0, 0] * model.graph_layers[1].weight[:, 0]).detach().numpy()})
torch.testing.assert_close(torch.tensor(layer2_terms["Product"].sum()), layer2_before_zero[0, 0, 0])
layer2_output = sensors(layer2)
```

</details>

#### Output

All fourteen sensors, each now carrying 2 calculated values. These are neither new measurements nor cycle positions. The complete table proceeds to the next step.

<!-- sample: layer2_output -->
| Sensor | Value 1 | Value 2 |
|---|---|---|
| T24 | 0.359686 | 0.000000 |
| T30 | 0.085212 | 0.000000 |
| T50 | 0.099950 | 0.000000 |
| P15 | 0.497302 | 0.000000 |
| P30 | 0.092590 | 0.000000 |
| Nf | 0.497302 | 0.000000 |
| Nc | 0.080467 | 0.000000 |
| Ps30 | 0.092590 | 0.000000 |
| phi | 0.061624 | 0.000000 |
| NRf | 0.497302 | 0.000000 |
| BPR | 0.497302 | 0.000000 |
| htBleed | 0.087253 | 0.000000 |
| W31 | 0.080467 | 0.000000 |
| W32 | 0.064877 | 0.000000 |

---

## 3 — Graph layer 3

Repeat the exchange once more. An output can now depend on sensors up to three connections away. Reducing the width reduces columns of calculated values, not the number of sensors.

### Process

#### Input

Every sensor carries 2 values in this example. A second input is the same fixed fourteen-by-fourteen combining table from [Rescale the graph](02_graphs.md#2--rescale-the-graph). Its rows receive from its columns.

<!-- sample: layer2_output -->
| Sensor | Value 1 | Value 2 |
|---|---|---|
| T24 | 0.359686 | 0.000000 |
| T30 | 0.085212 | 0.000000 |
| T50 | 0.099950 | 0.000000 |
| P15 | 0.497302 | 0.000000 |
| P30 | 0.092590 | 0.000000 |
| Nf | 0.497302 | 0.000000 |
| Nc | 0.080467 | 0.000000 |
| Ps30 | 0.092590 | 0.000000 |
| phi | 0.061624 | 0.000000 |
| NRf | 0.497302 | 0.000000 |
| BPR | 0.497302 | 0.000000 |
| htBleed | 0.087253 | 0.000000 |
| W31 | 0.080467 | 0.000000 |
| W32 | 0.064877 | 0.000000 |

#### Function

First multiply every permitted sender's values by its fixed graph share and add them at the receiver, separately for each input column. Do this for every sensor using the previous layer's values.


The complete combined table is:

<!-- sample: layer3_mixtures -->
| Sensor | Value 1 | Value 2 |
|---|---|---|
| T24 | 0.327411 | 0.000000 |
| T30 | 0.115538 | 0.000000 |
| T50 | 0.082195 | 0.000000 |
| P15 | 0.469779 | 0.000000 |
| P30 | 0.117486 | 0.000000 |
| Nf | 0.469779 | 0.000000 |
| Nc | 0.082781 | 0.000000 |
| Ps30 | 0.117486 | 0.000000 |
| phi | 0.082886 | 0.000000 |
| NRf | 0.469779 | 0.000000 |
| BPR | 0.469779 | 0.000000 |
| htBleed | 0.080635 | 0.000000 |
| W31 | 0.082781 | 0.000000 |
| W32 | 0.082603 | 0.000000 |

Next multiply each combined row by the following table. One column supplies the multipliers for one new output. The same table is used at every sensor; each output is the sum of its products across all input values.

<!-- sample: layer3_weights -->
| Input value | Output 1 |
|---|---|
| 1 | 0.374219 |
| 2 | -0.427394 |

For T24's first output, every product is:

<!-- sample: layer3_terms -->
| Input value | T24 mixed value | Multiplier for output 1 | Product |
|---|---|---|---|
| 1 | 0.327411 | 0.374219 | 0.122523 |
| 2 | 0.000000 | -0.427394 | -0.000000 |

Their sum is `0.122523`. Finally apply **ReLU**, meaning keep a nonnegative total and replace a negative total with zero. T24's first output is therefore `0.122523`. This demonstration adds no graph-layer offset, matching the project's default choice (assumption A-12).

<details class="process-example">
<summary>Example in Python</summary>

```python
with torch.no_grad():
    layer3_mixed = model.adjacency @ layer2
    layer3_before_zero = model.graph_layers[2](layer2, model.adjacency)
    layer3 = torch.relu(layer3_before_zero)
layer3_mixtures = sensors(layer3_mixed)
layer3_weights = learned_table(model.graph_layers[2].weight)
layer3_terms = pd.DataFrame({"Input value": range(1, 3),
    "T24 mixed value": layer3_mixed[0, 0].numpy(),
    "Multiplier for output 1": model.graph_layers[2].weight[:, 0].detach().numpy(),
    "Product": (layer3_mixed[0, 0] * model.graph_layers[2].weight[:, 0]).detach().numpy()})
torch.testing.assert_close(torch.tensor(layer3_terms["Product"].sum()), layer3_before_zero[0, 0, 0])
layer3_output = sensors(layer3)
```

</details>

#### Output

All fourteen sensors, each now carrying 1 calculated value. These are neither new measurements nor cycle positions. The complete table proceeds to the next step.

<!-- sample: layer3_output -->
| Sensor | Value 1 |
|---|---|
| T24 | 0.122523 |
| T30 | 0.043236 |
| T50 | 0.030759 |
| P15 | 0.175800 |
| P30 | 0.043966 |
| Nf | 0.175800 |
| Nc | 0.030978 |
| Ps30 | 0.043966 |
| phi | 0.031018 |
| NRf | 0.175800 |
| BPR | 0.175800 |
| htBleed | 0.030175 |
| W31 | 0.030978 |
| W32 | 0.030912 |

---

## 4 — Flatten

The next layer needs one list for the whole engine window. Place the sensor rows end to end so it can use every position.

### Process

#### Input

The complete last graph-layer table:

<!-- sample: layer3_output -->
| Sensor | Value 1 |
|---|---|
| T24 | 0.122523 |
| T30 | 0.043236 |
| T50 | 0.030759 |
| P15 | 0.175800 |
| P30 | 0.043966 |
| Nf | 0.175800 |
| Nc | 0.030978 |
| Ps30 | 0.043966 |
| phi | 0.031018 |
| NRf | 0.175800 |
| BPR | 0.175800 |
| htBleed | 0.030175 |
| W31 | 0.030978 |
| W32 | 0.030912 |

#### Function

Keep sensor order: T24, T30, T50, and so on. Here each contributes one value. The first three positions become `0.122523`, `0.043236`, and `0.030759`. With several values per sensor, place all of T24's values first, then all of T30's, and continue.

This step averages nothing, changes no values and learns no multipliers. It changes the arrangement: **sensor rows × values per sensor → one row per window**.

<details class="process-example">
<summary>Example in Python</summary>

```python
flat = layer3.flatten(start_dim=1)
flat_output = pd.DataFrame(flat.numpy(), columns=[f"Position {i + 1}" for i in range(flat.shape[1])])
flat_output.insert(0, "Window", ["Engine 1, cycles 1-30"])
torch.testing.assert_close(flat.reshape_as(layer3), layer3)
```

</details>

#### Output

All fourteen positions in the window’s one list. At the default final graph width of 32, the same operation produces fourteen times 32 positions.

<!-- sample: flat_output -->
| Window | Position 1 | Position 2 | Position 3 | Position 4 | Position 5 | Position 6 | Position 7 | Position 8 | Position 9 | Position 10 | Position 11 | Position 12 | Position 13 | Position 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Engine 1, cycles 1-30 | 0.122523 | 0.043236 | 0.030759 | 0.175800 | 0.043966 | 0.175800 | 0.030978 | 0.043966 | 0.031018 | 0.175800 | 0.175800 | 0.030175 | 0.030978 | 0.030912 |

---

## 5 — Fully connected

Now combine information from the whole window, including sensor positions that cannot reach one another within three graph layers. Every output has a multiplier for every input position.

### Process

#### Input

The complete flattened list:

<!-- sample: flat_output -->
| Window | Position 1 | Position 2 | Position 3 | Position 4 | Position 5 | Position 6 | Position 7 | Position 8 | Position 9 | Position 10 | Position 11 | Position 12 | Position 13 | Position 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Engine 1, cycles 1-30 | 0.122523 | 0.043236 | 0.030759 | 0.175800 | 0.043966 | 0.175800 | 0.030978 | 0.043966 | 0.031018 | 0.175800 | 0.175800 | 0.030175 | 0.030978 | 0.030912 |

#### Function

For each output, multiply every input by its own multiplier and add all products plus an offset. Then replace a negative total with zero. The complete multiplier table is:

<!-- sample: dense_weights -->
| Input value | Output 1 | Output 2 | Output 3 |
|---|---|---|---|
| 1 | -0.052534 | 0.028280 | 0.129534 |
| 2 | -0.255328 | 0.241999 | 0.014056 |
| 3 | -0.177002 | -0.247930 | -0.137020 |
| 4 | -0.110171 | -0.168251 | 0.045217 |
| 5 | 0.009900 | -0.067661 | -0.249540 |
| 6 | 0.105658 | -0.104178 | -0.193114 |
| 7 | 0.160363 | 0.230914 | -0.137781 |
| 8 | -0.181187 | -0.173233 | 0.168625 |
| 9 | -0.116382 | -0.123029 | 0.156701 |
| 10 | 0.097074 | -0.186720 | -0.118529 |
| 11 | 0.221931 | -0.250306 | -0.009643 |
| 12 | -0.055002 | -0.156011 | 0.170930 |
| 13 | 0.199995 | 0.229737 | 0.265693 |
| 14 | -0.043078 | 0.119257 | 0.106071 |

Below is every contribution to output 1. The total of these products plus offset `0.036105` is `0.065503`. After replacing a negative total with zero, its output is `0.065503`. Outputs 2 and 3 use their own columns and offsets.

<!-- sample: dense_terms -->
| Input position | Input value | Multiplier for output 1 | Product |
|---|---|---|---|
| 1 | 0.122523 | -0.052534 | -0.006437 |
| 2 | 0.043236 | -0.255328 | -0.011039 |
| 3 | 0.030759 | -0.177002 | -0.005444 |
| 4 | 0.175800 | -0.110171 | -0.019368 |
| 5 | 0.043966 | 0.009900 | 0.000435 |
| 6 | 0.175800 | 0.105658 | 0.018575 |
| 7 | 0.030978 | 0.160363 | 0.004968 |
| 8 | 0.043966 | -0.181187 | -0.007966 |
| 9 | 0.031018 | -0.116382 | -0.003610 |
| 10 | 0.175800 | 0.097074 | 0.017066 |
| 11 | 0.175800 | 0.221931 | 0.039015 |
| 12 | 0.030175 | -0.055002 | -0.001660 |
| 13 | 0.030978 | 0.199995 | 0.006195 |
| 14 | 0.030912 | -0.043078 | -0.001332 |

<!-- sample: dense_offsets -->
| Output | Offset |
|---|---|
| 1 | 0.036105 |
| 2 | 0.179195 |
| 3 | -0.157364 |

<details class="process-example">
<summary>Example in Python</summary>

```python
with torch.no_grad():
    dense_before = model.fc1(flat)
    dense = torch.relu(dense_before)
dense_weights = learned_table(model.fc1.weight.T)
dense_offsets = pd.DataFrame({"Output": [1, 2, 3], "Offset": model.fc1.bias.detach().numpy()})
dense_terms = pd.DataFrame({"Input position": range(1, flat.shape[1] + 1),
    "Input value": flat[0].numpy(), "Multiplier for output 1": model.fc1.weight[0].detach().numpy(),
    "Product": (flat[0] * model.fc1.weight[0]).detach().numpy()})
torch.testing.assert_close((flat[0] * model.fc1.weight[0]).sum() + model.fc1.bias[0], dense_before[0, 0])
dense_output = pd.DataFrame(dense.numpy(), columns=["Value 1", "Value 2", "Value 3"])
```

</details>

#### Output

One row for the window, with three calculated values in this demonstration. These columns have no separate sensor axis.

<!-- sample: dense_output -->
| Value 1 | Value 2 | Value 3 |
|---|---|---|
| 0.065503 | 0.059617 | 0.000000 |

---

## 6 — Predicted life fraction

Combine the window’s calculated values into one score, then map it to a fraction between zero and one. A fraction of the life ceiling is easier to convert into cycles in the final step.

### Process

#### Input

All three values from the preceding step:

<!-- sample: dense_output -->
| Value 1 | Value 2 | Value 3 |
|---|---|---|
| 0.065503 | 0.059617 | 0.000000 |

#### Function

Multiply each input by its output multiplier and add the products:

<!-- sample: score_terms -->
| Input value | Multiplier | Product |
|---|---|---|
| 0.065503 | 0.107586 | 0.007047 |
| 0.059617 | -0.447623 | -0.026686 |
| 0.000000 | -0.400153 | -0.000000 |

Add the offset `-0.298250`, giving score `-0.317888`. The **sigmoid** maps this score using `1 / (1 + exp(−score))`; `exp` is the exponential function. It gives a larger fraction for a larger score and always stays between zero and one. Here it gives `0.421191`.

The fraction is **not a probability of failure**. It specifies a share of the chosen life ceiling.

<details class="process-example">
<summary>Example in Python</summary>

```python
with torch.no_grad():
    score = model.fc2(dense)
    fraction = torch.sigmoid(score).squeeze(-1)
score_terms = pd.DataFrame({"Input value": dense[0].numpy(),
    "Multiplier": model.fc2.weight[0].detach().numpy(),
    "Product": (dense[0] * model.fc2.weight[0]).detach().numpy()})
torch.testing.assert_close((dense[0] * model.fc2.weight[0]).sum() + model.fc2.bias[0], score[0, 0])
fraction_output = pd.DataFrame({"Window": ["Engine 1, cycles 1-30"], "Life fraction": fraction.numpy()})
```

</details>

#### Output

One fraction per window. In the diagram `(B,)` means a one-dimensional list of B fractions, where B counts the windows processed together. No prediction is removed by dropping an axis of length one.

<!-- sample: fraction_output -->
| Window | Life fraction |
|---|---|
| Engine 1, cycles 1-30 | 0.421191 |

---

## 7 — Scale to cycles

A life fraction needs a scale to become a remaining-life estimate. Multiply by the same ceiling used to scale the training targets (assumption A-13).

### Process

#### Input

The computed fraction:

<!-- sample: fraction_output -->
| Window | Life fraction |
|---|---|
| Engine 1, cycles 1-30 | 0.421191 |

#### Function

The ceiling here is 125 cycles. Multiplying gives `0.421191 × 125 = 52.648811` cycles, with rounding only for display. Because the fraction is bounded, the prediction cannot exceed that ceiling.

This is the result of the specified **untrained teaching network**, not a useful estimate of this engine's remaining life. It is shown to complete the path from measurements to output.

<details class="process-example">
<summary>Example in Python</summary>

```python
cycles = fraction * model.rul_max
with torch.no_grad():
    torch.testing.assert_close(fraction, model(window))
    torch.testing.assert_close(cycles, model.predict_cycles(window))
cycle_output = pd.DataFrame({"Window": ["Engine 1, cycles 1-30"], "Predicted remaining cycles": cycles.numpy()})
```

</details>

#### Output

One output for the whole window, rather than one per sensor:

<!-- sample: cycle_output -->
| Window | Predicted remaining cycles |
|---|---|
| Engine 1, cycles 1-30 | 52.648811 |

---

## Shapes and the default architecture

The teaching model changes widths only to make every number visible. A shape
lists the lengths of an array's dimensions, in order. B below is the number of
windows processed together; the sensor order is fixed throughout the graph layers.

| Stage | Teaching example | Default network | Meaning |
|---|---|---|---|
| Temporal output | B × 14 × 1 | B × 14 × 1 | Windows, sensors, values per sensor |
| Graph layer 1 | B × 14 × 3 | B × 14 × 32 | Windows, sensors, computed values |
| Graph layer 2 | B × 14 × 2 | B × 14 × 64 | Windows, sensors, computed values |
| Graph layer 3 | B × 14 × 1 | B × 14 × 32 | Windows, sensors, computed values |
| Flatten | B × 14 | B × 448 | Windows, positions in the combined list |
| Fully connected | B × 3 | B × 128 | Windows, computed values |
| Life fraction | B | B | One fraction per window |
| Scale to cycles | B | B | One prediction per window |

The graph fixes **who can send**. Its rescaling fixes **how their values are
combined**. The layer parameters decide **how those mixtures become new
values**. Finally, the flattened list allows separate parameters for separate
sensor positions. These are different sources of information and different
roles for the numbers in the calculation.

## Training

Training changes the parameters that the examples held fixed. It compares each
window's predicted life fraction with the known remaining life at that window's
final cycle, divided by the same ceiling. The **loss** is the average squared
gap between predictions and those targets. An individual error is one gap;
the loss combines the errors into the quantity training tries to reduce.

For one update, the windows and targets are held fixed. Differentiation measures
how changing each trainable multiplier or offset would change the loss; those
rates of change are called **gradients**. The calculation follows the chain
from parameter, through layer values and prediction, to loss. Adam, the update
rule used here, uses current and past gradients to choose parameter changes.
The [training guide](../../src/thesis_gnn_rul_xai/training/README.md) explains
the updates and stopping procedure in detail.

The paper's Table 6 specifies mean squared error, Adam, batches of 32 windows,
200 **epochs** — an epoch is one pass over all the training examples — a 30-cycle window,
and a learning rate going from 0.001 to 0.0001. The learning rate scales
parameter changes. The project's step schedule fills an unspecified detail
(assumption A-17). Its optional held-back validation engines select when to
stop and which saved weights to keep (assumption A-18).

Validation engines check training progress; test engines evaluate a finished
model. They have different window-sampling rules here, so their errors are not
directly comparable (assumption A-06). None of the arithmetic examples on this
page is a measured training result.
