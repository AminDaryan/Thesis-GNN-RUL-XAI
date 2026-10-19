# Temporal algorithm

A sensor's newest reading tells us its current level. Earlier readings can
also show whether it is rising, falling or staying steady. The temporal
algorithm decides which of that information reaches the graph layers.

Here **time** means successive cycles of the same sensor, one row of NASA's
data per operational cycle (NASA Prognostics Center of Excellence [NASA PCoE],
2008). **The graph** links
different sensors. This page follows time within each sensor; the
[GNN page](02_graphs.md) explains how sensors exchange information afterwards.

## Choose a route

Choose **one** temporal algorithm. These are alternatives, not operations to
apply one after another.

| Route | What each sensor passes onwards | What training changes here |
|---|---|---|
| LSTM | Numbers calculated from updated memory | The multipliers and offsets that update and read the memory |
| GRU | Numbers from a different memory update | Its multipliers and offsets |
| Keep the last reading | The newest reading | Nothing |
| Average the window | The average reading | Nothing |
| Three numbers per sensor | Last reading, average and fitted change | Nothing |
| Pass every cycle | All readings, in their original order | Nothing |

The **LSTM**, short for long short-term memory network (Hochreiter &
Schmidhuber, 1997, Section 4), and **GRU**, short for gated recurrent unit, the
unit introduced by Cho et al. (2014, Section 2.3), follow the four numbered
steps below. The fixed alternatives later on this page replace that entire
route.

## Example setup

Every example uses **FD001 training engine 1, cycles 1–4**. Four cycles keep
every reading visible; the normal window length is 30, as in the paper (Wen et
al., 2026, Section 3.4, Table 6). The readings have already been smoothed with
the paper's moving average (Wen et al., 2026, Section 3.2.3, Eq. 18), at the
strength this project reads from that section (assumption A-04), and
rescaled using statistics fitted only on the training engines selected with
seed 42 (assumptions A-03, A-06 and A-25). They are relative to each sensor's
training mean and spread, not temperatures or pressures in their original
units.

The memory examples use **one number per memory list** so every memory value
can be shown. These are **untrained demonstrations**, with starting parameters
obtained from the model's initializer, which keeps PyTorch's own starting
values (assumption A-34), using a fixed seed. They have not been
fitted to engine failures and demonstrate no prediction accuracy. The usual
memory width is 32, a project choice (assumption A-26).

The tables stand on their own. To reproduce them, run the optional setup and
examples in page order. Tables are rounded to six decimal places; the
calculations keep full precision.

<details class="process-example">
<summary>Set up the Python examples</summary>

```python
import numpy as np
import pandas as pd
import torch
from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.models.temporal import LastCycle, MeanWindow, WindowSummary, RawWindow, LSTMStage

prepared = run_preprocessing(PreprocessingConfig(seed=42, window_size=4))
index = int(np.flatnonzero(prepared.train.units == 1)[0])
assert prepared.train.end_cycles[index] == 4
windows = torch.from_numpy(prepared.train.features[index:index + 1])
names = list(prepared.sensor_symbols)
sensor = names.index("T50")

def cycle_table(values):
    """Show every sensor at every cycle of this window."""
    frame = pd.DataFrame(values.detach().numpy(), columns=names)
    frame.insert(0, "Cycle", range(1, len(frame) + 1))
    return frame

def sensor_table(values, columns):
    """Show all calculated values for all fourteen sensors."""
    frame = pd.DataFrame(values.detach().numpy(), columns=columns)
    frame.insert(0, "Sensor", names)
    return frame

def demo_stage(kind, shared=False):
    """Build a reproducible untrained stage to calculate on real readings."""
    with torch.random.fork_rng():
        torch.manual_seed(7)
        return LSTMStage(window_size=4, n_nodes=14, hidden_size=1,
                         cell=kind, shared_across_sensors=shared)

source = cycle_table(windows[0].T)
histories = sensor_table(windows[0], ["Cycle 1", "Cycle 2", "Cycle 3", "Cycle 4"])
```

</details>

<div class="algorithm-intro">
<p class="algorithm-label">Memory-based temporal algorithm</p>
<p>Start with a window of readings. Follow steps 1–4: separate the sensor
histories, update memory through time, keep its final output, then convert it
into graph inputs. At step 2 choose LSTM or GRU, with separate or shared rules.</p>
</div>

---

## 1 — Split by sensor

Each sensor needs its own history. Otherwise the next sensor could be mistaken for the next moment in time. Separating histories changes their grouping, not their readings.

### Process

#### Input

One window, with every sensor at every cycle:

<!-- sample: source -->
| Cycle | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -1.987346 | -0.146319 | -0.995210 | 0.227969 | 1.205599 | -0.550483 | -0.908708 | -0.268201 | 0.347164 | -1.140864 | -0.663762 | -0.896282 | 1.519113 | 1.337202 |
| 2 | -1.603900 | 0.060766 | -0.841500 | 0.227969 | 0.830894 | -0.703583 | -0.960871 | -0.228359 | 0.797742 | -0.763605 | -0.479070 | -0.896282 | 1.330503 | 1.361258 |
| 3 | -1.179785 | -0.209812 | -0.700499 | 0.227969 | 0.956820 | -0.473932 | -0.768707 | -0.646699 | 1.124774 | -0.876783 | -0.596943 | -1.650164 | 1.079023 | 0.958051 |
| 4 | -0.967728 | -0.853047 | -0.770999 | 0.227969 | 1.136493 | -0.129456 | -0.757757 | -1.134762 | 1.608055 | -0.556113 | -1.400656 | -1.273223 | 0.733238 | 0.911768 |

#### Function

Read down each sensor column to form a sequence. The first sequence is T24's four readings, the second is T30's, and the third is T50's. All fourteen preserve oldest-to-newest order.

A separate network receives each sensor's sequences independently. A shared network receives a list of independent window-and-sensor histories. Sharing the calculation rule does not combine their memories.

<details class="process-example">
<summary>Example in Python</summary>

```python
separate = [windows[:, node, :, None] for node in range(len(names))]
shared_input = windows.reshape(len(names), 4, 1)
split_output = sensor_table(torch.stack([s[0, :, 0] for s in separate]),
                            ["Cycle 1", "Cycle 2", "Cycle 3", "Cycle 4"])
torch.testing.assert_close(shared_input[:, :, 0], windows[0])
```

</details>

#### Output

The complete histories, now one sensor per row. With several windows there is one such history per sensor per window. Each cycle supplies one reading; that is different from the memory width.

<!-- sample: split_output -->
| Sensor | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|
| T24 | -1.987346 | -1.603900 | -1.179785 | -0.967728 |
| T30 | -0.146319 | 0.060766 | -0.209812 | -0.853047 |
| T50 | -0.995210 | -0.841500 | -0.700499 | -0.770999 |
| P15 | 0.227969 | 0.227969 | 0.227969 | 0.227969 |
| P30 | 1.205599 | 0.830894 | 0.956820 | 1.136493 |
| Nf | -0.550483 | -0.703583 | -0.473932 | -0.129456 |
| Nc | -0.908708 | -0.960871 | -0.768707 | -0.757757 |
| Ps30 | -0.268201 | -0.228359 | -0.646699 | -1.134762 |
| phi | 0.347164 | 0.797742 | 1.124774 | 1.608055 |
| NRf | -1.140864 | -0.763605 | -0.876783 | -0.556113 |
| BPR | -0.663762 | -0.479070 | -0.596943 | -1.400656 |
| htBleed | -0.896282 | -0.896282 | -1.650164 | -1.273223 |
| W31 | 1.519113 | 1.330503 | 1.079023 | 0.733238 |
| W32 | 1.337202 | 1.361258 | 0.958051 | 0.911768 |

---

## 2 — Read cycles with memory

An average loses order. A memory update can use it: each cycle's result becomes
part of the next cycle's input (Cho et al., 2014, Section 2.1, Eq. 1).

> Cycle 1 → memory → cycle 2 → updated memory → cycle 3 → updated memory → cycle 4.

**Parameters** are the multipliers and offsets adjusted during training.
**Memory** is recalculated as readings arrive. Parameters persist between
windows; the memory starts at zero for every new window in this implementation.

---

### One LSTM per sensor

An LSTM carries two lists: the **cell state**, the memory it updates, and the **hidden state**, the output it calculates from that memory. **Gates** are multipliers between zero and one. They control how much old memory to keep, how much proposed information to add, and how much updated memory to expose (Wen et al., 2026, Section 2.2, Eqs. 1–6). The gates that add and expose come from the original LSTM (Hochreiter & Schmidhuber, 1997, Section 4); the gate that keeps old memory, the forget gate, was added later (Gers et al., 2000, Section 3.1, Eq. 7).

Giving each sensor its own LSTM follows the paper (Wen et al., 2026, Section 2.2, Eq. 7). Memory width and conversion into graph inputs remain project choices (assumption A-26).

{{figure:lstm-cell}}

The figure redraws the cell of the paper's Fig. 2, and its bracketed numbers are the paper's equation numbers (Wen et al., 2026, Section 2.2, Fig. 2, Eqs. 1–6). In the figure, x(t) is this cycle's reading, C(t) is the cell state and h(t) is the hidden output. The index t means the current cycle; t−1 means the preceding cycle. The forget, input and output gates are called **keep**, **add** and **expose** in the calculation below. The candidate is the proposed addition to memory.

#### Process

##### Input

Every history starts with both memory values at zero. The complete readings are:

<!-- sample: histories -->
| Sensor | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|
| T24 | -1.987346 | -1.603900 | -1.179785 | -0.967728 |
| T30 | -0.146319 | 0.060766 | -0.209812 | -0.853047 |
| T50 | -0.995210 | -0.841500 | -0.700499 | -0.770999 |
| P15 | 0.227969 | 0.227969 | 0.227969 | 0.227969 |
| P30 | 1.205599 | 0.830894 | 0.956820 | 1.136493 |
| Nf | -0.550483 | -0.703583 | -0.473932 | -0.129456 |
| Nc | -0.908708 | -0.960871 | -0.768707 | -0.757757 |
| Ps30 | -0.268201 | -0.228359 | -0.646699 | -1.134762 |
| phi | 0.347164 | 0.797742 | 1.124774 | 1.608055 |
| NRf | -1.140864 | -0.763605 | -0.876783 | -0.556113 |
| BPR | -0.663762 | -0.479070 | -0.596943 | -1.400656 |
| htBleed | -0.896282 | -0.896282 | -1.650164 | -1.273223 |
| W31 | 1.519113 | 1.330503 | 1.079023 | 0.733238 |
| W32 | 1.337202 | 1.361258 | 0.958051 | 0.911768 |

##### Function

For each gate, add **reading × its multiplier**, **old hidden value × its multiplier**, and an offset. A **sigmoid**, a smooth function with values between zero and one, converts that sum to a gate. The proposal uses **tanh**, a smooth function with values between minus one and one (Wen et al., 2026, Section 2.2, Eqs. 1–3 and 5).

These are the demonstration's actual starting multipliers for T50:

<!-- sample: lstm_weights -->
| Calculation | Reading multiplier | Hidden multiplier | Offset |
|---|---|---|---|
| Add | 0.253301 | -0.222516 | 1.396124 |
| Keep | 0.138145 | -0.557107 | -0.379226 |
| Proposal | 0.487466 | -0.251596 | -0.706862 |
| Expose | 0.918433 | -0.609484 | 0.733743 |

Calculate **new cell = keep × old cell + add × proposal**, then **new hidden = expose × tanh(new cell)** (Wen et al., 2026, Section 2.2, Eqs. 4 and 6). All terms refer to this sensor and this cycle. The table traces every cycle:

<!-- sample: lstm_trace -->
| Cycle | Reading | Old hidden | Old cell | Keep | Add | Proposal | Expose | Cell | Hidden |
|---|---|---|---|---|---|---|---|---|---|
| 1 | -0.995210 | 0.000000 | 0.000000 | 0.373622 | 0.758420 | -0.831196 | 0.455049 | -0.630395 | -0.254065 |
| 2 | -0.841500 | -0.254065 | -0.630395 | 0.412429 | 0.775478 | -0.783025 | 0.528900 | -0.867212 | -0.370206 |
| 3 | -0.700499 | -0.370206 | -0.867212 | 0.432968 | 0.786014 | -0.742123 | 0.578353 | -0.958794 | -0.430143 |
| 4 | -0.770999 | -0.430143 | -0.958794 | 0.438784 | 0.785253 | -0.750664 | 0.571457 | -1.010164 | -0.437639 |

For cycle 1, the cell calculation is `0.373622 × 0 + 0.758420 × -0.831196 = -0.630395`. Cycle 2 uses that cell and the first hidden output as its previous values; cycle 3 uses cycle 2's results. Other sensors follow the same procedure with their own parameters.

<details class="process-example">
<summary>Example in Python</summary>

```python
lstm = demo_stage("lstm")
cell = lstm.cells[sensor]
lstm_weights = pd.DataFrame({
    "Calculation": ["Add", "Keep", "Proposal", "Expose"],
    "Reading multiplier": cell.weight_ih_l0[:, 0].detach().numpy(),
    "Hidden multiplier": cell.weight_hh_l0[:, 0].detach().numpy(),
    "Offset": (cell.bias_ih_l0 + cell.bias_hh_l0).detach().numpy(),
})
rows = []
h, c = torch.zeros(1), torch.zeros(1)
with torch.no_grad():
    for cycle, x in enumerate(windows[0, sensor], 1):
        z = cell.weight_ih_l0[:, 0] * x + cell.weight_hh_l0[:, 0] * h + cell.bias_ih_l0 + cell.bias_hh_l0
        add, keep, proposal, expose = z[0].sigmoid(), z[1].sigmoid(), z[2].tanh(), z[3].sigmoid()
        old_h, old_c = h, c
        c = keep * old_c + add * proposal
        h = expose * c.tanh()
        rows.append([cycle, x.item(), old_h.item(), old_c.item(), keep.item(), add.item(), proposal.item(), expose.item(), c.item(), h.item()])
    lstm_history = torch.stack([cell(separate[node])[0][0, :, 0] for node, cell in enumerate(lstm.cells)])
    torch.testing.assert_close(h, lstm_history[sensor, -1].reshape(1))
lstm_trace = pd.DataFrame(rows, columns=["Cycle", "Reading", "Old hidden", "Old cell", "Keep", "Add", "Proposal", "Expose", "Cell", "Hidden"])
lstm_output = cycle_table(lstm_history.T)
```

</details>

##### Output

One hidden output per cycle for every sensor. These are calculated memory outputs, not readings. The final row continues to step 3. The cell state helped calculate them but is not another graph input.

<!-- sample: lstm_output -->
| Cycle | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -0.003066 | 0.332822 | -0.254065 | 0.090482 | -0.160950 | -0.099609 | 0.008441 | -0.041137 | 0.073895 | 0.101170 | 0.109923 | -0.318259 | 0.034094 | -0.206204 |
| 2 | 0.006929 | 0.454560 | -0.370206 | 0.126821 | -0.188643 | -0.152120 | 0.005529 | -0.049208 | 0.106821 | 0.144465 | 0.104630 | -0.412619 | 0.037705 | -0.351446 |
| 3 | 0.030132 | 0.532239 | -0.430143 | 0.141155 | -0.205267 | -0.174579 | 0.021259 | -0.076958 | 0.117099 | 0.181816 | 0.139008 | -0.610520 | 0.028895 | -0.390220 |
| 4 | 0.052986 | 0.606997 | -0.437639 | 0.146840 | -0.225195 | -0.180037 | 0.025035 | -0.115920 | 0.095604 | 0.188476 | 0.368339 | -0.588520 | 0.011666 | -0.415119 |

---

### One LSTM for all sensors

Sharing means one set of parameters is used for every sensor. It does not mean all sensors share one memory. This is an alternative to the paper’s separate-sensor arrangement (Wen et al., 2026, Section 2.2, Eq. 7).

#### Process

##### Input

The same independent histories:

<!-- sample: histories -->
| Sensor | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|
| T24 | -1.987346 | -1.603900 | -1.179785 | -0.967728 |
| T30 | -0.146319 | 0.060766 | -0.209812 | -0.853047 |
| T50 | -0.995210 | -0.841500 | -0.700499 | -0.770999 |
| P15 | 0.227969 | 0.227969 | 0.227969 | 0.227969 |
| P30 | 1.205599 | 0.830894 | 0.956820 | 1.136493 |
| Nf | -0.550483 | -0.703583 | -0.473932 | -0.129456 |
| Nc | -0.908708 | -0.960871 | -0.768707 | -0.757757 |
| Ps30 | -0.268201 | -0.228359 | -0.646699 | -1.134762 |
| phi | 0.347164 | 0.797742 | 1.124774 | 1.608055 |
| NRf | -1.140864 | -0.763605 | -0.876783 | -0.556113 |
| BPR | -0.663762 | -0.479070 | -0.596943 | -1.400656 |
| htBleed | -0.896282 | -0.896282 | -1.650164 | -1.273223 |
| W31 | 1.519113 | 1.330503 | 1.079023 | 0.733238 |
| W32 | 1.337202 | 1.361258 | 0.958051 | 0.911768 |

##### Function

Apply the LSTM updates above to each history with a common rule. T24 starts from zero and reads its measurements. T30 starts from a separate zero memory and reads its own measurements, as does T50. No sensor reads another sensor's updated memory.

The shared demonstration's parameters differ from some dedicated networks, so their outputs can differ. These untrained numbers do not compare prediction accuracy.

<details class="process-example">
<summary>Example in Python</summary>

```python
shared_lstm = demo_stage("lstm", shared=True)
with torch.no_grad():
    shared_lstm_history = shared_lstm.cells[0](shared_input)[0][:, :, 0]
    alone = shared_lstm.cells[0](shared_input[sensor:sensor + 1])[0][0, :, 0]
torch.testing.assert_close(alone, shared_lstm_history[sensor])
shared_lstm_output = cycle_table(shared_lstm_history.T)
```

</details>

##### Output

Each sensor still has its own output at every cycle. The example verifies that T50 gives the same values when processed alone or alongside the other histories.

<!-- sample: shared_lstm_output -->
| Cycle | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -0.003066 | 0.073135 | 0.033150 | 0.092145 | 0.142548 | 0.053377 | 0.036928 | 0.067071 | 0.098280 | 0.026981 | 0.048048 | 0.037478 | 0.158409 | 0.149240 |
| 2 | 0.006929 | 0.116010 | 0.059502 | 0.128911 | 0.164315 | 0.074635 | 0.056210 | 0.100987 | 0.154916 | 0.059805 | 0.082409 | 0.059231 | 0.185434 | 0.185552 |
| 3 | 0.030132 | 0.121412 | 0.079891 | 0.143262 | 0.177878 | 0.096932 | 0.075876 | 0.098841 | 0.181792 | 0.071992 | 0.093865 | 0.042461 | 0.183797 | 0.179338 |
| 4 | 0.052986 | 0.100279 | 0.087050 | 0.148874 | 0.187008 | 0.121112 | 0.085996 | 0.082108 | 0.202690 | 0.093351 | 0.068691 | 0.050879 | 0.173146 | 0.179353 |

---

### One GRU per sensor

A GRU carries one memory list, its hidden state. A **reset gate** controls how much old memory enters a proposed replacement. An **update gate** blends the old memory with that proposal (Cho et al., 2014, Section 2.3, Eqs. 5–8). GRU is a project alternative, not the paper's LSTM stage, and it borrows the LSTM stage's sizes (assumption A-36).

#### Process

##### Input

Each history starts with zero memory:

<!-- sample: histories -->
| Sensor | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|
| T24 | -1.987346 | -1.603900 | -1.179785 | -0.967728 |
| T30 | -0.146319 | 0.060766 | -0.209812 | -0.853047 |
| T50 | -0.995210 | -0.841500 | -0.700499 | -0.770999 |
| P15 | 0.227969 | 0.227969 | 0.227969 | 0.227969 |
| P30 | 1.205599 | 0.830894 | 0.956820 | 1.136493 |
| Nf | -0.550483 | -0.703583 | -0.473932 | -0.129456 |
| Nc | -0.908708 | -0.960871 | -0.768707 | -0.757757 |
| Ps30 | -0.268201 | -0.228359 | -0.646699 | -1.134762 |
| phi | 0.347164 | 0.797742 | 1.124774 | 1.608055 |
| NRf | -1.140864 | -0.763605 | -0.876783 | -0.556113 |
| BPR | -0.663762 | -0.479070 | -0.596943 | -1.400656 |
| htBleed | -0.896282 | -0.896282 | -1.650164 | -1.273223 |
| W31 | 1.519113 | 1.330503 | 1.079023 | 0.733238 |
| W32 | 1.337202 | 1.361258 | 0.958051 | 0.911768 |

##### Function

Both gates combine the reading, old hidden state and offsets, then apply the sigmoid. These are T50's actual starting parameters. A reading contribution means **reading × reading multiplier + reading offset**; a memory contribution means **old hidden × hidden multiplier + hidden offset**.

<!-- sample: gru_weights -->
| Calculation | Reading multiplier | Hidden multiplier | Reading offset | Hidden offset |
|---|---|---|---|---|
| Reset | 0.041860 | 0.257178 | 0.711886 | 0.138145 |
| Update | 0.186462 | 0.530547 | 0.344169 | 0.487466 |
| Proposal | 0.759408 | -0.773559 | 0.253301 | 0.918433 |

Add the two contributions before applying the sigmoid for each gate. For the proposal, first multiply its memory contribution by the reset gate, then add its reading contribution and apply tanh. The reset acts on the memory offset as well as the multiplied hidden value. This order is PyTorch's, the version this project runs (assumption A-36); Cho et al. (2014, Section 2.3, Eqs. 5, 6 and 8) print the gates with no offsets and apply the reset gate to the old hidden value before its multiplier.

Finally, **new hidden = update × old hidden + (1 − update) × proposal** (Cho et al., 2014, Section 2.3, Eq. 7). A larger update gate keeps more old memory. Here is every T50 update:

<!-- sample: gru_trace -->
| Cycle | Reading | Old hidden | Reset | Update | Proposal | Hidden |
|---|---|---|---|---|---|---|
| 1 | -0.995210 | 0.000000 | 0.691762 | 0.656123 | 0.132091 | 0.045423 |
| 2 | -0.841500 | 0.045423 | 0.695612 | 0.667927 | 0.224785 | 0.104984 |
| 3 | -0.700499 | 0.104984 | 0.700086 | 0.680641 | 0.298128 | 0.166667 |
| 4 | -0.770999 | 0.166667 | 0.702790 | 0.684881 | 0.219047 | 0.183173 |

Cycle 1 gives `0.656123 × 0 + (1 − 0.656123) × 0.132091 = 0.045423`. That output becomes the old hidden state in cycle 2. Unlike an LSTM, there is no separate cell state.

<details class="process-example">
<summary>Example in Python</summary>

```python
gru = demo_stage("gru")
cell = gru.cells[sensor]
gru_weights = pd.DataFrame({
    "Calculation": ["Reset", "Update", "Proposal"],
    "Reading multiplier": cell.weight_ih_l0[:, 0].detach().numpy(),
    "Hidden multiplier": cell.weight_hh_l0[:, 0].detach().numpy(),
    "Reading offset": cell.bias_ih_l0.detach().numpy(),
    "Hidden offset": cell.bias_hh_l0.detach().numpy(),
})
h = torch.zeros(1)
rows = []
with torch.no_grad():
    for cycle, x in enumerate(windows[0, sensor], 1):
        incoming = cell.weight_ih_l0[:, 0] * x + cell.bias_ih_l0
        previous = cell.weight_hh_l0[:, 0] * h + cell.bias_hh_l0
        reset = (incoming[0] + previous[0]).sigmoid()
        update = (incoming[1] + previous[1]).sigmoid()
        proposal = (incoming[2] + reset * previous[2]).tanh()
        old_h = h
        h = update * old_h + (1 - update) * proposal
        rows.append([cycle, x.item(), old_h.item(), reset.item(), update.item(), proposal.item(), h.item()])
    gru_history = torch.stack([cell(separate[node])[0][0, :, 0] for node, cell in enumerate(gru.cells)])
    torch.testing.assert_close(h, gru_history[sensor, -1].reshape(1))
gru_trace = pd.DataFrame(rows, columns=["Cycle", "Reading", "Old hidden", "Reset", "Update", "Proposal", "Hidden"])
gru_output = cycle_table(gru_history.T)
```

</details>

##### Output

All hidden outputs for this window. The final row continues to step 3.

<!-- sample: gru_output -->
| Cycle | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -0.071137 | 0.427434 | 0.045423 | 0.242626 | -0.550074 | -0.009385 | 0.389259 | 0.038871 | -0.024649 | -0.104406 | -0.097510 | -0.535915 | -0.521294 | 0.604253 |
| 2 | -0.138051 | 0.545192 | 0.104984 | 0.378726 | -0.696320 | -0.040493 | 0.574102 | 0.075006 | 0.002500 | -0.212697 | -0.032140 | -0.738003 | -0.661309 | 0.818267 |
| 3 | -0.195238 | 0.584655 | 0.166667 | 0.461816 | -0.775571 | -0.014053 | 0.663546 | -0.135398 | 0.043073 | -0.206153 | -0.074416 | -0.887895 | -0.694454 | 0.827697 |
| 4 | -0.239541 | 0.610552 | 0.183173 | 0.514832 | -0.832841 | 0.056619 | 0.707624 | -0.437034 | 0.090676 | -0.278890 | -0.383979 | -0.928948 | -0.709106 | 0.820970 |

---

### One GRU for all sensors

The shared GRU uses a common update rule with an independent memory for each history.

#### Process

##### Input

All complete histories:

<!-- sample: histories -->
| Sensor | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|
| T24 | -1.987346 | -1.603900 | -1.179785 | -0.967728 |
| T30 | -0.146319 | 0.060766 | -0.209812 | -0.853047 |
| T50 | -0.995210 | -0.841500 | -0.700499 | -0.770999 |
| P15 | 0.227969 | 0.227969 | 0.227969 | 0.227969 |
| P30 | 1.205599 | 0.830894 | 0.956820 | 1.136493 |
| Nf | -0.550483 | -0.703583 | -0.473932 | -0.129456 |
| Nc | -0.908708 | -0.960871 | -0.768707 | -0.757757 |
| Ps30 | -0.268201 | -0.228359 | -0.646699 | -1.134762 |
| phi | 0.347164 | 0.797742 | 1.124774 | 1.608055 |
| NRf | -1.140864 | -0.763605 | -0.876783 | -0.556113 |
| BPR | -0.663762 | -0.479070 | -0.596943 | -1.400656 |
| htBleed | -0.896282 | -0.896282 | -1.650164 | -1.273223 |
| W31 | 1.519113 | 1.330503 | 1.079023 | 0.733238 |
| W32 | 1.337202 | 1.361258 | 0.958051 | 0.911768 |

##### Function

Apply the GRU update above independently to each row, starting from zero. Parameters are shared; readings and memories remain separate.

<details class="process-example">
<summary>Example in Python</summary>

```python
shared_gru = demo_stage("gru", shared=True)
with torch.no_grad():
    shared_gru_history = shared_gru.cells[0](shared_input)[0][:, :, 0]
    alone = shared_gru.cells[0](shared_input[sensor:sensor + 1])[0][0, :, 0]
torch.testing.assert_close(alone, shared_gru_history[sensor])
shared_gru_output = cycle_table(shared_gru_history.T)
```

</details>

##### Output

One separate output sequence per sensor. T50 again gives the same values when processed alone.

<!-- sample: shared_gru_output -->
| Cycle | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -0.071137 | -0.066112 | -0.084212 | -0.042909 | 0.074245 | -0.079790 | -0.084060 | -0.071377 | -0.033111 | -0.083830 | -0.081839 | -0.084013 | 0.128663 | 0.096209 |
| 2 | -0.138051 | -0.099946 | -0.148996 | -0.071435 | 0.062724 | -0.143201 | -0.150229 | -0.120826 | -0.004732 | -0.147464 | -0.138773 | -0.149538 | 0.161649 | 0.150279 |
| 3 | -0.195238 | -0.140606 | -0.196824 | -0.090652 | 0.072296 | -0.184932 | -0.199491 | -0.173486 | 0.058778 | -0.199509 | -0.185880 | -0.204151 | 0.141997 | 0.120474 |
| 4 | -0.239541 | -0.193628 | -0.236467 | -0.103711 | 0.102894 | -0.198080 | -0.238228 | -0.223993 | 0.174474 | -0.231519 | -0.235688 | -0.250504 | 0.090141 | 0.098575 |

---

## 3 — Keep the last hidden state

The last output has been calculated through the preceding updates, so it can summarise the history. It need not preserve every detail. Passing on the last cycle's hidden state is this project's reading of the paper's Eq. 7, which writes each sensor's LSTM result as the hidden state at the window's last cycle (Wen et al., 2026, Section 2.2, Eq. 7; assumption A-26).

### Process

#### Input

Continue the separate LSTM example with its complete output history. A GRU or shared-network route applies the same selection to its own output table:

<!-- sample: lstm_output -->
| Cycle | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -0.003066 | 0.332822 | -0.254065 | 0.090482 | -0.160950 | -0.099609 | 0.008441 | -0.041137 | 0.073895 | 0.101170 | 0.109923 | -0.318259 | 0.034094 | -0.206204 |
| 2 | 0.006929 | 0.454560 | -0.370206 | 0.126821 | -0.188643 | -0.152120 | 0.005529 | -0.049208 | 0.106821 | 0.144465 | 0.104630 | -0.412619 | 0.037705 | -0.351446 |
| 3 | 0.030132 | 0.532239 | -0.430143 | 0.141155 | -0.205267 | -0.174579 | 0.021259 | -0.076958 | 0.117099 | 0.181816 | 0.139008 | -0.610520 | 0.028895 | -0.390220 |
| 4 | 0.052986 | 0.606997 | -0.437639 | 0.146840 | -0.225195 | -0.180037 | 0.025035 | -0.115920 | 0.095604 | 0.188476 | 0.368339 | -0.588520 | 0.011666 | -0.415119 |

#### Function

Keep the last row, one hidden value per sensor. T50's final hidden value is `-0.437639`, whereas its last reading was `-0.770999`. One is a calculated memory output; the other is a preprocessed measurement. Selecting the final output does not rerun the memory updates.

<details class="process-example">
<summary>Example in Python</summary>

```python
last_hidden = lstm_history[:, -1:].unsqueeze(0)
last_hidden_output = sensor_table(last_hidden[0], ["Final hidden value"])
```

</details>

#### Output

One hidden value per sensor in this demonstration. A memory width of 32 would give 32 calculated values per sensor, not 32 cycles.

<!-- sample: last_hidden_output -->
| Sensor | Final hidden value |
|---|---|
| T24 | 0.052986 |
| T30 | 0.606997 |
| T50 | -0.437639 |
| P15 | 0.146840 |
| P30 | -0.225195 |
| Nf | -0.180037 |
| Nc | 0.025035 |
| Ps30 | -0.115920 |
| phi | 0.095604 |
| NRf | 0.188476 |
| BPR | 0.368339 |
| htBleed | -0.588520 |
| W31 | 0.011666 |
| W32 | -0.415119 |

---

## 4 — Turn into numbers per sensor

Memory width and graph-input width are separate choices. A weighted sum converts between them, using the same conversion for all sensors (assumption A-26).

### Process

#### Input

The final memory outputs:

<!-- sample: last_hidden_output -->
| Sensor | Final hidden value |
|---|---|
| T24 | 0.052986 |
| T30 | 0.606997 |
| T50 | -0.437639 |
| P15 | 0.146840 |
| P30 | -0.225195 |
| Nf | -0.180037 |
| Nc | 0.025035 |
| Ps30 | -0.115920 |
| phi | 0.095604 |
| NRf | 0.188476 |
| BPR | 0.368339 |
| htBleed | -0.588520 |
| W31 | 0.011666 |
| W32 | -0.415119 |

#### Function

For each desired output, multiply every hidden value by its multiplier, add the products, then add an offset. Here there is one hidden value and one output:

<!-- sample: projection_rule -->
| Hidden multiplier | Offset |
|---|---|
| 0.218370 | -0.592311 |

T50 gives `0.218370 × -0.437639 + -0.592311 = -0.687878`. With a wider memory, add one product for every hidden value. Multiple outputs have separate multipliers and offsets.

<details class="process-example">
<summary>Example in Python</summary>

```python
with torch.no_grad():
    projected = lstm.project(last_hidden)
    torch.testing.assert_close(projected, lstm(windows))
projection_rule = pd.DataFrame({"Hidden multiplier": [lstm.project.weight[0, 0].item()], "Offset": [lstm.project.bias[0].item()]})
projected_output = sensor_table(projected[0], ["Graph input 1"])
```

</details>

#### Output

These graph inputs are calculated descriptions of each sensor’s window, not remaining-life predictions. The graph will use them together with the sensor connections.

<!-- sample: projected_output -->
| Sensor | Graph input 1 |
|---|---|
| T24 | -0.580740 |
| T30 | -0.459760 |
| T50 | -0.687878 |
| P15 | -0.560245 |
| P30 | -0.641487 |
| Nf | -0.631625 |
| Nc | -0.586844 |
| Ps30 | -0.617624 |
| phi | -0.571433 |
| NRf | -0.551153 |
| BPR | -0.511876 |
| htBleed | -0.720826 |
| W31 | -0.589763 |
| W32 | -0.682960 |

---

## Fixed alternatives to the memory route

Each route below replaces steps 1–4. These summaries learn no parameters;
the graph layers that receive them still learn their own multipliers.

---

### Keep the last reading

Pass on the current level. This is the command-line default stand-in for the paper’s temporal stage (assumption A-07). Earlier cycles can still have influenced the reading through preprocessing’s smoothing, which mixes each reading with the previous smoothed value (Wen et al., 2026, Section 3.2.3, Eq. 18; assumption A-04).

#### Process

##### Input

The original window:

<!-- sample: source -->
| Cycle | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -1.987346 | -0.146319 | -0.995210 | 0.227969 | 1.205599 | -0.550483 | -0.908708 | -0.268201 | 0.347164 | -1.140864 | -0.663762 | -0.896282 | 1.519113 | 1.337202 |
| 2 | -1.603900 | 0.060766 | -0.841500 | 0.227969 | 0.830894 | -0.703583 | -0.960871 | -0.228359 | 0.797742 | -0.763605 | -0.479070 | -0.896282 | 1.330503 | 1.361258 |
| 3 | -1.179785 | -0.209812 | -0.700499 | 0.227969 | 0.956820 | -0.473932 | -0.768707 | -0.646699 | 1.124774 | -0.876783 | -0.596943 | -1.650164 | 1.079023 | 0.958051 |
| 4 | -0.967728 | -0.853047 | -0.770999 | 0.227969 | 1.136493 | -0.129456 | -0.757757 | -1.134762 | 1.608055 | -0.556113 | -1.400656 | -1.273223 | 0.733238 | 0.911768 |

##### Function

Select cycle 4 from every column. Retain `-0.770999` for T50. No averaging or memory update occurs. The **Last reading** branch in the three-number summary performs this same selection.

<details class="process-example">
<summary>Example in Python</summary>

```python
last_values = LastCycle(4)(windows)
last_output = sensor_table(last_values[0], ["Last reading"])
```

</details>

##### Output

One reading per sensor. Do not confuse this with the last hidden state: that is a calculated memory output.

<!-- sample: last_output -->
| Sensor | Last reading |
|---|---|
| T24 | -0.967728 |
| T30 | -0.853047 |
| T50 | -0.770999 |
| P15 | 0.227969 |
| P30 | 1.136493 |
| Nf | -0.129456 |
| Nc | -0.757757 |
| Ps30 | -1.134762 |
| phi | 1.608055 |
| NRf | -0.556113 |
| BPR | -1.400656 |
| htBleed | -1.273223 |
| W31 | 0.733238 |
| W32 | 0.911768 |

---

### Average the window

An average describes the overall level. It treats cycles equally and discards their order: reversing a history leaves its average unchanged.

#### Process

##### Input

The original window:

<!-- sample: source -->
| Cycle | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -1.987346 | -0.146319 | -0.995210 | 0.227969 | 1.205599 | -0.550483 | -0.908708 | -0.268201 | 0.347164 | -1.140864 | -0.663762 | -0.896282 | 1.519113 | 1.337202 |
| 2 | -1.603900 | 0.060766 | -0.841500 | 0.227969 | 0.830894 | -0.703583 | -0.960871 | -0.228359 | 0.797742 | -0.763605 | -0.479070 | -0.896282 | 1.330503 | 1.361258 |
| 3 | -1.179785 | -0.209812 | -0.700499 | 0.227969 | 0.956820 | -0.473932 | -0.768707 | -0.646699 | 1.124774 | -0.876783 | -0.596943 | -1.650164 | 1.079023 | 0.958051 |
| 4 | -0.967728 | -0.853047 | -0.770999 | 0.227969 | 1.136493 | -0.129456 | -0.757757 | -1.134762 | 1.608055 | -0.556113 | -1.400656 | -1.273223 | 0.733238 | 0.911768 |

##### Function

Add each sensor's readings and divide by four. For T50, `(-0.995210 + -0.841500 + -0.700499 + -0.770999) / 4 = -0.827052`. Repeat independently for every sensor. The **Average** branch of the three-number summary uses the same calculation.

<details class="process-example">
<summary>Example in Python</summary>

```python
mean_values = MeanWindow(4)(windows)
mean_output = sensor_table(mean_values[0], ["Average"])
```

</details>

##### Output

One average per sensor:

<!-- sample: mean_output -->
| Sensor | Average |
|---|---|
| T24 | -1.434690 |
| T30 | -0.287103 |
| T50 | -0.827052 |
| P15 | 0.227969 |
| P30 | 1.032452 |
| Nf | -0.464364 |
| Nc | -0.849011 |
| Ps30 | -0.569505 |
| phi | 0.969434 |
| NRf | -0.834341 |
| BPR | -0.785108 |
| htBleed | -1.178988 |
| W31 | 1.165469 |
| W32 | 1.142070 |

---

### Change over the window

An average can hide a rising or falling trend. Fit a straight line to all readings, then measure the fitted line’s change from the first cycle to the last. This is not generally the last reading minus the first.

#### Process

##### Input

The same window:

<!-- sample: source -->
| Cycle | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -1.987346 | -0.146319 | -0.995210 | 0.227969 | 1.205599 | -0.550483 | -0.908708 | -0.268201 | 0.347164 | -1.140864 | -0.663762 | -0.896282 | 1.519113 | 1.337202 |
| 2 | -1.603900 | 0.060766 | -0.841500 | 0.227969 | 0.830894 | -0.703583 | -0.960871 | -0.228359 | 0.797742 | -0.763605 | -0.479070 | -0.896282 | 1.330503 | 1.361258 |
| 3 | -1.179785 | -0.209812 | -0.700499 | 0.227969 | 0.956820 | -0.473932 | -0.768707 | -0.646699 | 1.124774 | -0.876783 | -0.596943 | -1.650164 | 1.079023 | 0.958051 |
| 4 | -0.967728 | -0.853047 | -0.770999 | 0.227969 | 1.136493 | -0.129456 | -0.757757 | -1.134762 | 1.608055 | -0.556113 | -1.400656 | -1.273223 | 0.733238 | 0.911768 |

##### Function

The middle of cycles 1–4 is 2.5. Subtract it from every cycle position. Multiply each centred position by its reading, add the products, then divide by the sum of squared centred positions. This gives the change per cycle of the straight line whose squared gaps from the readings are smallest. Multiply by the three cycle intervals to obtain the change over the window.

Every contribution for T50 is shown:

<!-- sample: slope_terms -->
| Cycle | Position minus 2.5 | T50 reading | Product | Squared position |
|---|---|---|---|---|
| 1.000000 | -1.500000 | -0.995210 | 1.492815 | 2.250000 |
| 2.000000 | -0.500000 | -0.841500 | 0.420750 | 0.250000 |
| 3.000000 | 0.500000 | -0.700499 | -0.350249 | 0.250000 |
| 4.000000 | 1.500000 | -0.770999 | -1.156499 | 2.250000 |

The product total is `0.406817`; divide by `5.000000` to get slope `0.081363`. Then `0.081363 × 3 = 0.244090`. Positive means rising and negative falling, in preprocessed reading units, not cycles of remaining life.

<details class="process-example">
<summary>Example in Python</summary>

```python
positions = torch.arange(1, 5, dtype=windows.dtype)
centered = positions - positions.mean()
products = centered * windows[0, sensor]
slope = products.sum() / centered.square().sum()
summary_values = WindowSummary(4)(windows)
torch.testing.assert_close(slope * 3, summary_values[0, sensor, 2])
slope_terms = pd.DataFrame({"Cycle": positions.numpy(), "Position minus 2.5": centered.numpy(), "T50 reading": windows[0, sensor].numpy(), "Product": products.numpy(), "Squared position": centered.square().numpy()})
change_output = sensor_table(summary_values[0, :, 2:3], ["Fitted change"])
```

</details>

##### Output

One fitted change per sensor:

<!-- sample: change_output -->
| Sensor | Fitted change |
|---|---|
| T24 | 1.044891 |
| T30 | -0.717229 |
| T50 | 0.244090 |
| P15 | 0.000000 |
| P30 | -0.024417 |
| Nf | 0.447819 |
| Nc | 0.193505 |
| Ps30 | -0.905406 |
| phi | 1.232911 |
| NRf | 0.492323 |
| BPR | -0.698566 |
| htBleed | -0.565412 |
| W31 | -0.782731 |
| W32 | -0.503852 |

---

### Three numbers per sensor

Keep level and trend together: last reading, average and fitted change. These are calculated side by side from the same window. This is a project extension (assumption A-28).

#### Process

##### Input

The three completed calculations, each shown across all sensors:

<!-- sample: summary_input -->
| Calculation | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Last reading | -0.967728 | -0.853047 | -0.770999 | 0.227969 | 1.136493 | -0.129456 | -0.757757 | -1.134762 | 1.608055 | -0.556113 | -1.400656 | -1.273223 | 0.733238 | 0.911768 |
| Average | -1.434690 | -0.287103 | -0.827052 | 0.227969 | 1.032452 | -0.464364 | -0.849011 | -0.569505 | 0.969434 | -0.834341 | -0.785108 | -1.178988 | 1.165469 | 1.142070 |
| Fitted change | 1.044891 | -0.717229 | 0.244090 | 0.000000 | -0.024417 | 0.447819 | 0.193505 | -0.905406 | 1.232911 | 0.492323 | -0.698566 | -0.565412 | -0.782731 | -0.503852 |

##### Function

Group each sensor's results in the order **last, average, change**. T50 becomes `[-0.770999, -0.827052, 0.244090]`. The three columns measure different properties; they are not three cycle positions.

<details class="process-example">
<summary>Example in Python</summary>

```python
summary_input = pd.DataFrame(summary_values[0].T.numpy(), columns=names)
summary_input.insert(0, "Calculation", ["Last reading", "Average", "Fitted change"])
summary_output = sensor_table(summary_values[0], ["Last reading", "Average", "Fitted change"])
torch.testing.assert_close(summary_values[:, :, :1], last_values)
torch.testing.assert_close(summary_values[:, :, 1:2], mean_values)
```

</details>

##### Output

Three complete input values for each graph node:

<!-- sample: summary_output -->
| Sensor | Last reading | Average | Fitted change |
|---|---|---|---|
| T24 | -0.967728 | -1.434690 | 1.044891 |
| T30 | -0.853047 | -0.287103 | -0.717229 |
| T50 | -0.770999 | -0.827052 | 0.244090 |
| P15 | 0.227969 | 0.227969 | 0.000000 |
| P30 | 1.136493 | 1.032452 | -0.024417 |
| Nf | -0.129456 | -0.464364 | 0.447819 |
| Nc | -0.757757 | -0.849011 | 0.193505 |
| Ps30 | -1.134762 | -0.569505 | -0.905406 |
| phi | 1.608055 | 0.969434 | 1.232911 |
| NRf | -0.556113 | -0.834341 | 0.492323 |
| BPR | -1.400656 | -0.785108 | -0.698566 |
| htBleed | -1.273223 | -1.178988 | -0.565412 |
| W31 | 0.733238 | 1.165469 | -0.782731 |
| W32 | 0.911768 | 1.142070 | -0.503852 |

---

### Pass every cycle

Keeping the entire history lets the first graph layer apply a multiplier to every cycle position, instead of receiving a summary. The oldest and newest readings occupy different columns.

#### Process

##### Input

The histories already in sensor-by-cycle order:

<!-- sample: histories -->
| Sensor | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|
| T24 | -1.987346 | -1.603900 | -1.179785 | -0.967728 |
| T30 | -0.146319 | 0.060766 | -0.209812 | -0.853047 |
| T50 | -0.995210 | -0.841500 | -0.700499 | -0.770999 |
| P15 | 0.227969 | 0.227969 | 0.227969 | 0.227969 |
| P30 | 1.205599 | 0.830894 | 0.956820 | 1.136493 |
| Nf | -0.550483 | -0.703583 | -0.473932 | -0.129456 |
| Nc | -0.908708 | -0.960871 | -0.768707 | -0.757757 |
| Ps30 | -0.268201 | -0.228359 | -0.646699 | -1.134762 |
| phi | 0.347164 | 0.797742 | 1.124774 | 1.608055 |
| NRf | -1.140864 | -0.763605 | -0.876783 | -0.556113 |
| BPR | -0.663762 | -0.479070 | -0.596943 | -1.400656 |
| htBleed | -0.896282 | -0.896282 | -1.650164 | -1.273223 |
| W31 | 1.519113 | 1.330503 | 1.079023 | 0.733238 |
| W32 | 1.337202 | 1.361258 | 0.958051 | 0.911768 |

##### Function

Pass every value on unchanged. No memory, averaging, reordering or learned calculation occurs in this temporal step. Here the graph receives four values per sensor; with the normal window it receives thirty.

<details class="process-example">
<summary>Example in Python</summary>

```python
all_cycles = RawWindow(4)(windows)
torch.testing.assert_close(all_cycles, windows)
all_cycles_output = sensor_table(all_cycles[0], ["Cycle 1", "Cycle 2", "Cycle 3", "Cycle 4"])
```

</details>

##### Output

Every reading is retained in its original position:

<!-- sample: all_cycles_output -->
| Sensor | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|
| T24 | -1.987346 | -1.603900 | -1.179785 | -0.967728 |
| T30 | -0.146319 | 0.060766 | -0.209812 | -0.853047 |
| T50 | -0.995210 | -0.841500 | -0.700499 | -0.770999 |
| P15 | 0.227969 | 0.227969 | 0.227969 | 0.227969 |
| P30 | 1.205599 | 0.830894 | 0.956820 | 1.136493 |
| Nf | -0.550483 | -0.703583 | -0.473932 | -0.129456 |
| Nc | -0.908708 | -0.960871 | -0.768707 | -0.757757 |
| Ps30 | -0.268201 | -0.228359 | -0.646699 | -1.134762 |
| phi | 0.347164 | 0.797742 | 1.124774 | 1.608055 |
| NRf | -1.140864 | -0.763605 | -0.876783 | -0.556113 |
| BPR | -0.663762 | -0.479070 | -0.596943 | -1.400656 |
| htBleed | -0.896282 | -0.896282 | -1.650164 | -1.273223 |
| W31 | 1.519113 | 1.330503 | 1.079023 | 0.733238 |
| W32 | 1.337202 | 1.361258 | 0.958051 | 0.911768 |

---

## The stage left out by default

The paper describes separate LSTM networks (Wen et al., 2026, Section 2.2). The
command-line default instead keeps the last reading, so its results belong
beside the paper's comparison without LSTM, its "ESTGNN w/o LSTM" row, not the
full-model headline (Wen et al., 2026, Section 4.3, Table 11). Using LSTM here
still requires unspecified choices about width and conversion to graph
inputs (assumptions A-11 and A-26).

These demonstrations teach calculations, not accuracy rankings.
[Follow one real example](08_worked_examples.md#through-the-saved-model)
traces a named saved network using the last-reading route. The
[implementation guide](../../src/thesis_gnn_rul_xai/models/temporal/README.md)
records the alternatives and their limits.

## References

- Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
  Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN
  encoder–decoder for statistical machine translation. In A. Moschitti, B.
  Pang, & W. Daelemans (Eds.), *Proceedings of the 2014 Conference on
  Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1724–1734).
  Association for Computational Linguistics. https://doi.org/10.3115/v1/D14-1179
  [Copy](../references/cho_2014_rnn_encoder_decoder.pdf)
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget:
  Continual prediction with LSTM. *Neural Computation*, *12*(10), 2451–2471.
  https://doi.org/10.1162/089976600300015015
  [Copy](../references/gers_2000_learning_to_forget.pdf). The copy held is the
  authors' accepted manuscript, cited here by section and equation.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural
  Computation*, *9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
  [Copy](../references/hochreiter_1997_long_short_term_memory.pdf). The copy
  held is the authors' 32-page manuscript, whose pages are not numbered as in
  the journal, so it is cited here by section.
- NASA Prognostics Center of Excellence. (2008). readme.txt, the description of
  the C-MAPSS data set distributed inside its zip file. NASA Prognostics Data
  Repository. [Copy](../references/nasa_2008_cmapss_readme.txt)
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and
  Technology*, *168*, 110875. https://doi.org/10.1016/j.ast.2025.110875
  [Copy](../references/wen_2026_estgnn_dynamic_shap.pdf)
