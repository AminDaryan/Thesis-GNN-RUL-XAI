# `models` — the neural network

This folder holds the PyTorch definition of the network this project
re-implements: **ESTGNN**, the model published by Wen et al. (2026) for
predicting how many flight cycles an aeroengine has left. There is no training
code here and no data handling; this is only the architecture.

> **The `.py` files here carry one line of description each, and nothing more.**
> All of the explanation lives in this file. The section
> [What each name means](#what-each-name-means) at the end lists every argument
> the two classes take.

PyTorch is an optional dependency of this project. The preprocessing half does
not need it. Install it with:

```
pip install -e ".[model]"
```

## The files

| File | What it is |
|------|------------|
| `layers.py` | `GraphConvolution`, one graph convolution layer. |
| `estgnn.py` | `ESTGNN`, the whole network: a temporal stage, three graph convolutions, a flatten, two dense layers. Built with `spatial="none"` it has no graph convolutions at all ([A network without graph layers](#a-network-without-graph-layers)). |
| `temporal/` | The temporal stages, each a different way of turning one sensor's 30-cycle window into what the graph layers read. That folder has its own README. |
| `config.py` | `ModelConfig`, the eight settings that decide the network's shape, kept in one record a run can save; and `build_model`, which builds the network such a record describes. Explained under [Writing down which network a run used](#writing-down-which-network-a-run-used). |
| `__init__.py` | Re-exports the names above, so `from thesis_gnn_rul_xai.models import ModelConfig` works. |

## What the network is

The paper prints the architecture as its Table 4, and the layer sizes below are
transcribed from it — none of them is a guess. `B` is the batch size, and 14 is
the number of sensors, which is fixed by the paper's sensor list.

| Layer | Shape in → shape out | Activation |
|-------|----------------------|------------|
| GCN-1 | `(B, 14, 1)` → `(B, 14, 32)` | ReLU |
| GCN-2 | `(B, 14, 32)` → `(B, 14, 64)` | ReLU |
| GCN-3 | `(B, 14, 64)` → `(B, 14, 32)` | ReLU |
| Flatten | `(B, 14, 32)` → `(B, 448)` | – |
| FC-1 | `(B, 448)` → `(B, 128)` | ReLU |
| FC-2 | `(B, 128)` → `(B, 1)` | Sigmoid |

Built with those defaults the model has 61,729 trainable parameters. Table 4
lists no total of its own: its five weight matrices come to 61,600, and the
remaining 129 are the two biases Equations 12 and 13 print for the dense
layers. (Corrected 2026-10-03: this said 61,729 is "the count Table 4's printed
shapes imply", and then corrected itself in the next sentence; the shapes alone
give 61,600. Found by the search for that wording in answering the audit of
2026-09-08, `docs/audits/AUDIT_2026-09-08_response.md`.) `ESTGNN.count_parameters()` returns that number and
`ESTGNN.layer_summary()` returns one dictionary per layer in the same form as
Table 4, so the transcription can be checked line by line: the row's name
(`layer`), its kind (`type`), the shape going in and coming out (`in`, `out`),
how many numbers it learns (`parameters`) and its activation.

### The Input row holds the temporal stage

Table 4 begins with a row the table above leaves out: `Input | Node features |
[batch size,14,1] | [batch size,14,1] | – | –`, one number per sensor and
nothing learned. In the paper that one number has already come out of a
recurrent stage: one small network per sensor, an LSTM (long short-term
memory, one kind of recurrent network), that reads the sensor's cycles one at
a time and carries numbers forward from each cycle to the next. Table 4 gives
that stage no row of its own. Here the
network receives each sensor's whole 30-cycle window, and a temporal stage from
`models/temporal/` makes the one number ([One number per sensor, and why 30
would be wrong](#one-number-per-sensor-and-why-30-would-be-wrong) says why it
lives in the model). For the paper's stage and for the default, what the stage
hands on is exactly the array Table 4's Input row lists, so `layer_summary()`
gives the two as one row, Table 4's Input row with the stage folded into it:

- `in` is what the stage receives, the window, `(B, 14, 30)`;
- `out` is what it hands to the first graph layer: Table 4's `(B, 14, 1)`,
  or three or thirty numbers per sensor for the two stages that hand on more
  (`window_summary` and `raw_window`, which `models/temporal/README.md`
  describes);
- `parameters` counts the numbers the stage learns;
- `from` names the stage, but only when it learns numbers of its own: `"LSTM"`
  for the paper's recurrent stage; `"GRU"` for a gated recurrent unit, a
  recurrent network with three parts where the LSTM has four and no separate
  memory, which this project offers in the LSTM's place and the paper's network
  does not have (assumption A-36); and an empty string for a fixed rule such as
  keeping the last cycle.

`from` says where the row's count comes from. A fixed rule leaves the count at
0, Table 4's "–", so there is nothing on the row to attribute; which rule a
run used is recorded in its `summary.json` as `temporal_stage`.

Built on the paper's sensor graph with a 30-cycle window, the first row reads,
for the two stages notebook 02 compares:

| Stage | `from` | `in` | `out` | `parameters` |
|---|---|---|---|---|
| `LastCycle(30)`, the default | `""` | `(B, 14, 30)` | `(B, 14, 1)` | 0 |
| `LSTMStage(window_size=30, n_nodes=14, hidden_size=32)` | `"LSTM"` | `(B, 14, 30)` | `(B, 14, 1)` | 62,753 |

The 62,753 is fourteen recurrent networks, one per sensor, of 4,480 numbers
each, which makes 62,720, plus the 33 of the layer that turns each network's
32 numbers into the one Table 4 wants: 32 multipliers and one offset.

The 4,480 needs one distinction first. Each of those networks carries two
lists of 32 numbers from one cycle to the next, and they are easy to confuse:
its memory (PyTorch calls it the cell state), and the 32 numbers it passed on
at the cycle before (the hidden state). The 32 it passes on at the last cycle
are the ones the layer after it turns into one number. Inside each network are
four parts. At every cycle each part looks at the new reading and at the 32
numbers passed on last time, and produces 32 numbers: one part proposes new
values for the memory, one decides how much of that proposal to write in, one
how much of the old memory to forget, and one how much of the memory to pass
on. So each part holds 32 multipliers for the one incoming reading,
32 × 32 = 1,024 for the 32 numbers passed on at the cycle before, and
2 × 32 = 64 offsets, because PyTorch keeps two sets of them:
4 × (32 + 1,024 + 64) = 4,480. No learned multiplier acts on the memory
itself: from one cycle to the next it is only scaled, number by number, by what
the forgetting part produced, and the written-in proposal is added to it. Set
all four parts' 1,024 multipliers to zero and what a network passes on no
longer changes,
at any cycle, when the 32 numbers it starts out passing on change; change the
memory it starts with instead, and what it passes on after the first cycle
does change (the forgetting part then shrinks that difference cycle by cycle).

The six rows below the Input row are the same for both stages, and the seven
rows add up to `count_parameters()`: 61,729 for the default, and
61,729 + 62,753 = 124,482 with the recurrent stage.
`tests/test_examples_torch.py` builds both networks, checks every one of these
numbers, and sets the four parts' 1,024 multipliers to zero on a real FD001
window to show which of the two lists they act on.

(Changed 2026-10-03: until then the summary listed the stage as a row of its
own, "Temporal", above Table 4's Input row, and both rows described the same
array. The playground and the site's figures already showed the two as one
Input row; the author decided on 2026-10-02 that the summary itself should, so
that everything showing it shows the same seven rows.)

## What a graph convolution does

Each of the 14 sensors is a node in a graph. An edge means two sensors are
related — here, related by the engine's physics. A graph convolution updates
every node by mixing its own values with the values of the nodes it is
connected to, and then applying a learned linear transformation. One layer lets
a sensor see its direct neighbours; three stacked layers let it see three edges
away.

The form used is the one from Kipf and Welling (2017), which is also what the
paper prints as its Equations 8 to 10:

```
H' = ReLU(A_hat @ H @ W)
```

- `H` is the current node features, shaped `(batch, nodes, features_in)`.
- `A_hat` is the normalised adjacency matrix, `D^(-1/2) (A + I) D^(-1/2)`. It
  describes the sensor graph. It is **not learned** — it is the engineering
  knowledge the paper builds into the model.
- `W` is the learned weight matrix, shaped `(features_in, features_out)`.

`A_hat @ H` is the mixing step; `@ W` is the learning step.

### Why plain matrix multiplication and not a graph library

Libraries such as PyTorch Geometric are built for graphs with thousands of
nodes and sparse connectivity that differs per example. This graph has 14
nodes, is dense enough that a sparse representation would be slower, and is
identical for every example in the batch. Two dense matrix multiplications are
exact, add no dependency, and are easy to check by hand against the paper's
equations.

## Traps and things you need to know

### The adjacency matrix must already be normalised

`ESTGNN` expects the output of `normalize_adjacency()` from
`thesis_gnn_rul_xai.graph.adjacency`, not the raw matrix. Normalisation is what
adds the self-loops that let a node see its own value. Handing over the raw
matrix would train quietly and badly, so the constructor checks the diagonal
and raises if it finds a zero there.

### Training never changes the adjacency matrix

A PyTorch model holds two kinds of numbers. **Parameters** are the ones
training is allowed to change — every weight in this network is one. **Buffers**
are numbers the model carries around but training never touches. Both are
written to disk when the model is saved, and both move together onto a graphics
card. The only difference is whether training may edit them.

The adjacency matrix is stored as a buffer. That is deliberate: the graph is
engineering knowledge transcribed from the paper's Table 3, not something to be
discovered from data. If training could edit it, then after 200 passes over the
data you would no longer have the paper's graph but whatever the fitting
procedure happened to prefer — and the comparison between "with the physics
graph" and "without" would be measuring two things that had both drifted.

To run the paper's "without physics prior" experiment you build the model with a
different matrix instead — `build_fully_connected_adjacency()` in
`thesis_gnn_rul_xai.graph.adjacency`.

One layer on its own, a `GraphConvolution`, does not store the matrix at all: it
is handed in as an argument every time the layer runs. That is why all three
layers can share a single copy, and why a different graph needs no new layers.

### The shape checks are deliberate

`GraphConvolution.forward()` checks the size of every input before doing
anything. Left to itself, PyTorch will sometimes stretch a mismatched array to
fit rather than complain — it is called *broadcasting*, and it is usually
convenient. Here it would be a disaster: the layer would quietly pair up the
wrong sensors and still return a number that looked entirely reasonable. A
wrong answer that looks plausible is the worst kind of bug this code could
have, so the sizes are checked and a mismatch is named rather than absorbed.

### One number per sensor, and why 30 would be wrong

Each node carries a single number into the graph layers. Table 4 prints the
input to the first graph layer as `[batch, 14, 1]` with a weight matrix of
shape `1 × 32`. In the paper, that number is the output of a per-sensor LSTM —
but the paper never states that LSTM's hidden size, its depth, or how it reduces
a 30-cycle window to one number. The pipeline hands the model each sensor's
whole window, and a **temporal stage** in front of the graph layers, from
`models/temporal/`, reduces it to that one number. The default stage keeps the
most recent reading of the window (assumption A-07), so the paper's LSTM stage
is not the default (assumption A-11); `LSTMStage` builds it when asked.

The first graph weight matrix takes its width from the stage, not from the
data. **A stage that hands the graph layers more than one number per sensor
without learning anything across the cycles is refused.** Handing each node its
whole 30-cycle window unchanged would make the first weight matrix `30 × 32`,
which is 32 learned linear filters over time, shared across sensors: a temporal
encoder hidden inside a layer that is not labelled as one. It is not the
configuration this project runs, and building it on purpose needs
`allow_time_in_the_graph_layer=True`, which records the choice.

Only the first graph layer depends on `in_features` anyway. GCN-2 and GCN-3
take their widths from the layer before them.

### The sigmoid, and the ceiling it creates

The last layer is a sigmoid, so the network's raw output is always between 0
and 1. The paper calls this a "normalized RUL value". A prediction in flight
cycles is that number multiplied by `rul_max`, the label ceiling (125 by
default here).

Two things follow, and both matter:

1. **Training targets must be divided by `rul_max` before they are used.**
   A sigmoid cannot leave `[0, 1]`, so a network trained against targets
   measured in cycles could never reach a label of 40, let alone 125. The
   division is not a preference; it is the only scaling that puts the targets
   where the output layer can reach them. The paper implies it without printing
   it (assumption A-13). The division happens in `training/run.py`; the
   multiplication back into cycles happens here, in `ESTGNN.predict_cycles`.

2. **The model can never predict a remaining life above `rul_max`, for any
   input at all.** This is a hard fact about the architecture, not a tendency.
   It is why the project also caps the official test labels at 125 before
   scoring (assumption A-02), because leaving them uncapped
   would charge the model for an error its own architecture forbids it from
   avoiding. Whenever a number from this project is quoted, the cap has to be
   quoted with it.

So: `ESTGNN.forward()` returns the normalised value in `[0, 1]`, and
`ESTGNN.predict_cycles()` returns flight cycles. Do not mix them up.

### What these results may be compared with

It depends on the temporal stage the run used and on whether it had graph
layers, and every run records both.

With **`last_cycle`**, the default, there is no recurrent network, so results
must be compared against the paper's own **"ESTGNN w/o LSTM"** ablation, which
its Table 11 reports as **14.93 RMSE on FD001** — and **not** against the full
model's headline 11.78.

With **`lstm`**, the stage of the paper's Section 2.2 is present, so the headline
row becomes the right comparison. Even then it is not a reproduction: the
paper states how many recurrent networks there are and what goes into each, and
never states their size, so this project had to choose one (assumption A-26).
What the stage is worth when trained depends on whether the best pass's weights
are kept, which the paper never describes (finding F-16).

With **`lstm` and `spatial="none"`**, the paper's **"ESTGNN w/o GCN"** row
(**16.42 RMSE on FD001**, Table 11) is the comparison, under the same two
warnings as the line above, and under this project's reading of what that row
removed (assumption A-42).

With **`last_cycle` and `spatial="none"`**, two parts of the paper's network are
gone, the recurrent stage and the graph layers. Table 11 has no row that
removes both, so no published number applies.

## A network without graph layers

### Why it exists

The paper reports a version of its network with the graph layers taken out,
"ESTGNN w/o GCN" (GCN, graph convolutional network, being the three graph
layers), and prints its error in Table 11: 16.42 on FD001. It is how the paper
argues that passing numbers between connected sensors is worth something. It
never says what the network looked like with those layers gone. This project
reads it literally: the layers are removed and nothing is put in their place
(assumption A-42). `ModelConfig(spatial="none")`, or `scripts/train.py --spatial
none`, builds that network.

### What it computes

The full network and this one differ only in the middle. Following one batch
of `B` windows, each holding 14 sensors and 30 cycles:

```text
full network (spatial="gcn"):
  window (B, 14, 30) -> temporal stage (B, 14, 1) -> GCN-1 (B, 14, 32)
    -> GCN-2 (B, 14, 64) -> GCN-3 (B, 14, 32) -> flatten (B, 448)
    -> FC-1 (B, 128) -> FC-2 (B, 1) -> sigmoid x 125 -> cycles

without graph layers (spatial="none"):
  window (B, 14, 30) -> temporal stage (B, 14, 1)  -> flatten (B, 14)
    -> FC-1 (B, 128) -> FC-2 (B, 1) -> sigmoid x 125 -> cycles
```

In the full network, each graph layer replaces a sensor's numbers with a
weighted average of its own and its neighbours' numbers, then multiplies by a
table of learned multipliers; that is the only place one sensor's reading can
reach another sensor's numbers before the dense layers. Without the graph
layers, the fourteen numbers the temporal stage hands on, one per sensor, are
written out as one row of 14 (the flatten) and go straight into FC-1, a dense
layer: each of its 128 outputs is its own weighted sum of all 14 numbers, plus
an offset, with negative results then set to zero (ReLU). Every sensor still
reaches the prediction, and FC-1 still combines all fourteen, but only once,
inside a layer that knows nothing of which sensors the paper's graph connects.

`layer_summary()` then has four rows, not seven: Input, Flatten, FC-1 and FC-2,
with the flatten going from `(B, 14, 1)` to `(B, 14)`.

### A worked example on a real window

`tests/test_examples_torch.py` builds this network (untrained, starting values
drawn after `torch.manual_seed(0)`) with the default stage and follows the
last window of test engine 1 through it. The default stage keeps each sensor's
latest reading, after smoothing and rescaling, so FC-1 receives these fourteen
numbers:

```text
T24 -0.4272   T30 -1.2246   T50 -1.1072   P15  0.2280   P30   1.0624
Nf  -0.2107   Nc  -0.6090   Ps30 -0.9797  phi  0.6584   NRf  -0.3687
BPR -0.9152   htBleed -0.7095   W31 0.5842   W32 0.8267
```

The first of FC-1's 128 outputs multiplies them by the first row of FC-1's
multipliers, `-0.0020, 0.1434, -0.2200, ...`, adds them up, and adds its offset
0.2127; the total is -0.3182, so ReLU sets it to 0. Over all 128 outputs, 63
come out at 0. FC-2 adds the 128 up, each times its own multiplier, plus an
offset, giving -0.2406; the sigmoid turns that into 1 / (1 + e^0.2406) = 0.4401,
and times the ceiling of 125 that is 55.017 cycles, the network's own
prediction to the third decimal (55.0172 before rounding the score). The test repeats the whole calculation with
NumPy and checks it against the network.

### How much it learns, and why that matters

| Temporal stage | With the graph layers | Without |
|---|---|---|
| `last_cycle`, the default | 61,729 | 2,049 |
| `lstm`, at the size this project chose (assumption A-26) | 124,482 | 64,802 |

FC-1's multipliers go from 448 × 128 to 14 × 128, and the three graph layers'
1 × 32 + 32 × 64 + 64 × 32 = 4,128 go with them: 14 × 128 + 128 + 128 + 1 =
2,049. The recurrent stage's own 62,753 stay, so 2,049 + 62,753 = 64,802.

So removing the graph layers removes two things at once: the passing of
numbers between connected sensors, and most of the numbers the network learns.
A difference in error between the two networks cannot say which of the two
caused it. Do not confuse this comparison with a test of the connections
alone: that test would keep all three graph layers and every learned number,
and run them on a graph in which each sensor is joined only to itself, so no
sensor ever receives a neighbour's numbers. That graph is not built here
(assumption A-42 says what was measured of it elsewhere).

### What stays the same, and what refuses it

- **The graph is still stored in the network.** The rescaled sensor graph is
  kept in `adjacency` as before, so the weights a run saved load back into a
  rebuilt network unchanged. Nothing in the prediction reads it; a test fills
  it with "not a number" values and gets the same prediction.
- **The guard on raw cycles still applies.** A stage that hands on more than
  one number per sensor without learning anything, `raw_window` or
  `window_summary`, still has to be asked for by name. Without graph layers its
  numbers land in FC-1 instead, as 14 × 30 = 420 inputs for `raw_window`, and
  FC-1's multipliers then do the temporal stage's work without being named one,
  so the same refusal is raised and `allow_time_in_the_graph_layer` lets it
  through. The flag's name predates this option; with no graph layers it
  covers FC-1.
- **The explanations of the graph refuse it.** The walks through the sensor
  graph, the removal of one connection and the taking out of sensors (node
  flipping) in `explain/` each describe what the graph layers did with the
  graph, and refuse a network whose `has_graph_layers` is `False`, with a
  message saying so. Shapley values and occlusion only swap sensors' readings
  and read the prediction, so they explain it as they explain any other
  network.

## Writing down which network a run used

### The problem

A stored result is worth quoting only if someone can say which network
produced it. The shape of the network is decided by eight settings, and they
arrive from different places: the training script's command line, the
documentation site's playground, and, when an explanation is computed later,
the folder of a run that finished days ago. If each of those places built the
network its own way, two of them could read the same settings and build two
different networks, and nothing would say so.

`config.py` answers that with two things. `ModelConfig` is one record holding
all eight settings. `build_model` is the one function that turns such a record
into a network. The training script, the playground and the reader of stored
runs in `training/stored.py` all go through both.

### The eight settings

| Setting | Default | Where the default comes from |
|---|---|---|
| `temporal` | `"last_cycle"` | Our choice: keep the most recent reading of each window (assumption A-07), so the paper's LSTM stage is off (assumption A-11). |
| `temporal_hidden` | `32` | Our choice, since the paper never gives the LSTM's size (assumption A-26). Read only by `lstm` and `gru`, the two stages that read a window one cycle at a time and carry a running summary of size `temporal_hidden` from cycle to cycle. |
| `temporal_out_features` | `1` | How many numbers each sensor hands the graph layers. Table 4 prints the input to the first graph layer as `[batch size,14,1]`. |
| `shared_across_sensors` | `False` | Section 2.2 of the paper gives each sensor "a dedicated LSTM network", so one network shared by all fourteen is not the paper. |
| `gcn_hidden` | `(32, 64, 32)` | Table 4: GCN-1, GCN-2 and GCN-3 output 32, 64 and 32 numbers per sensor. |
| `fc_hidden` | `128` | Table 4: FC-1 outputs 128. |
| `gcn_bias` | `False` | Our choice: Table 4 lists only a weight matrix for each graph layer (assumption A-12). |
| `spatial` | `"gcn"` | What sits between the temporal stage and the dense layers. `"gcn"` is Table 4's three graph layers; `"none"` removes them and puts nothing in their place, which is how this project reads the paper's "ESTGNN w/o GCN" (assumption A-42). Explained under [A network without graph layers](#a-network-without-graph-layers). |

A record checks its values the moment it is made, so a bad setting stops the
program before any training starts rather than after an hour of it:

```text
ModelConfig(temporal="transformer")
  -> ValueError: temporal must be one of ['last_cycle', 'mean_window', 'window_summary', 'raw_window', 'lstm', 'gru'], got 'transformer'.
ModelConfig(fc_hidden=0)
  -> ValueError: fc_hidden must be at least 1, got 0.
ModelConfig(spatial="gat")
  -> ValueError: spatial must be one of ['gcn', 'none'], got 'gat'.
ModelConfig(spatial="none", gcn_hidden=(16,))
  -> ValueError: gcn_hidden and gcn_bias describe the graph layers, and with spatial='none' there are none; leave both at their defaults.
```

The last refusal exists because the network's fingerprint, described below, is
made from every setting. A width for graph layers that are not built would
change the fingerprint and the folder name without changing the network, and
one network would be filed under two names.

`recurrent` is a yes-or-no answer read off the record: `True` for `lstm` and
`gru`, the two stages that have a hidden size at all, and `False` otherwise.
`has_graph_layers` is another: `False` only when `spatial` is `"none"`.

### Saving the record beside a run, and reading it back

`save(path)` writes the record as YAML, a plain-text format for settings. This
is the `model_config.yaml` that `scripts/train.py` leaves in every run folder.
For the default record the whole file is:

```yaml
temporal: last_cycle
temporal_hidden: 32
temporal_out_features: 1
shared_across_sensors: false
gcn_hidden:
- 32
- 64
- 32
fc_hidden: 128
gcn_bias: false
```

`spatial` is missing from it on purpose. `to_dict()`, which both the file and
the fingerprint below are made from, leaves `spatial` out while it is `"gcn"`,
so the record of a default network is the same, character for character, as
it was before the setting existed, and every stored run's name and fingerprint
still describe it. A network without graph layers gets one more line,
`spatial: none`. A file without the line loads as `"gcn"`, which is what every
run stored before 2026-11-13 trained.

`load_model_config(path)` reads such a file back into a `ModelConfig`, and
refuses a key it does not know. Python would refuse one anyway, with
`ModelConfig.__init__() got an unexpected keyword argument 'hidden'`; the check
exists so the message names the file and lists every key that is allowed:

```text
Unknown model setting(s) in .../bad.yaml: ['hidden']. Valid keys are: ['fc_hidden', 'gcn_bias', 'gcn_hidden', 'shared_across_sensors', 'temporal', 'temporal_hidden', 'temporal_out_features'].
```

A run from before `model_config.yaml` existed recorded only its temporal stage
and hidden size in `summary.json`, and `training/stored.py` rebuilds the record
from those two, with every other setting at its default.

### The six-character name of a network

A run folder's name has to change whenever the network does, or two different
networks would be filed under one name. Writing all eight settings into a
folder name would make it unreadable, so the record is reduced to a short
**fingerprint**: a few characters computed from the settings, which come out
the same every time for the same settings and completely different when any
one of them changes.

`digest()` makes it in three steps:

1. Write the record as JSON text, a standard way of writing nested data as
   characters, with the keys in alphabetical order. The order is fixed so that
   the same settings always become the same text.
2. Feed that text to SHA-256, a standard function that turns any text into 64
   hexadecimal characters (the digits 0 to 9 and the letters a to f). Change a
   single character of the text and the output changes beyond recognition.
3. Keep the first six characters.

For the default record, step 1 gives

```text
{"fc_hidden": 128, "gcn_bias": false, "gcn_hidden": [32, 64, 32], "shared_across_sensors": false, "temporal": "last_cycle", "temporal_hidden": 32, "temporal_out_features": 1}
```

step 2 gives `4e49f2beee0f32bea65cfa3f41da5a3a...`, and the digest is `4e49f2`.
Change only `fc_hidden` from 128 to 64 and the digest becomes `f98ee3`, with
nothing in common with the first.

`scripts/train.py` uses it like this. The temporal stage is already written
into a run's folder name in words — nothing for `last_cycle`, the default, and
`_lstm32` for an LSTM of hidden size 32 — so the stage and its size do not
count as a difference. Nor does `spatial`, which is written in words too:
nothing for `"gcn"`, and `_nogcn` after the temporal stage's mark for
`"none"`, so `fd001_physics_row_cap_5seed_varsplit_lstm32_nogcn_cfg818b40_...`. Any other setting that differs from the default adds
`_net<digest>`, so a run with `fc_hidden` at 64 carries `_netf98ee3`. Six
characters allow 16 million different names, which makes two different
networks sharing one unlikely but not impossible, and that is why the full
record is always saved beside the results rather than trusted to the name.

Every character fed to SHA-256 is a setting: `ModelConfig` has no field of free text,
so rewording a comment cannot rename a run.

### Building the network

`build_model(adjacency, config, window_size=..., rul_max=...)` builds the
temporal stage the record names, then the `ESTGNN` around it. What it takes:

- `adjacency`, the sensor graph, already rescaled by `normalize_adjacency()`;
- `config`, the `ModelConfig`;
- `window_size`, how many cycles each sensor's window holds (30 by default);
- `rul_max`, the label ceiling (125 by default).

It returns the network, not yet trained. Built on the paper's graph with a
window of 30, the stages give these sizes, from `count_parameters()`:

| `temporal` | Numbers the network learns |
|---|---|
| `last_cycle` | 61,729 |
| `mean_window` | 61,729 |
| `window_summary` | 61,793 |
| `lstm` | 124,482 |
| `raw_window` | 62,657 |

With `spatial="none"` the same stages give 2,049 (`last_cycle` and
`mean_window`), 5,633 (`window_summary`), 64,802 (`lstm`) and 54,017
(`raw_window`); [A network without graph layers](#a-network-without-graph-layers)
works the first and the fourth out.

The first two match because neither stage learns anything: one keeps the last
reading and the other averages the thirty, and both hand over one number per
sensor. `raw_window` hands over all 30, so the first graph weight matrix grows
from 1 × 32 = 32 numbers to 30 × 32 = 960, and 61,729 + 960 − 32 = 62,657.
`window_summary` hands over three numbers made from the thirty — the last
reading, the average and the change across the window — so the matrix is
3 × 32 = 96 numbers and the total 61,729 + 96 − 32 = 61,793. It is not in the
paper (assumption A-28).

That last stage is the one `ESTGNN` refuses to build unless told
`allow_time_in_the_graph_layer=True`, for the reason given in
[One number per sensor, and why 30 would be wrong](#one-number-per-sensor-and-why-30-would-be-wrong).
`window_summary` trips the same check, because it hands over three numbers and
learns nothing. `build_model` passes `True` only when `config.temporal` is
`"raw_window"` or `"window_summary"`: naming one of those stages in a record
counts as asking for it on purpose, and every other stage is still checked.

## Judgement calls made in this folder

Each of these is a place where the paper does not say what to do and we had to
choose. The full reasoning for every one is in `docs/assumptions/`, in the file
named after the code given in parentheses.

- **The graph layers carry no bias.** The paper's Equations 8 to 10 show only
  the weight matrix `W`, while Equations 12 and 13, describing the dense
  layers, print an explicit bias. Within the same short section it prints
  biases where it means them, so the omission reads as deliberate. We follow
  the equations as printed. `GraphConvolution` takes `bias=True` so the
  difference can be measured rather than argued about (assumption A-12).
- **The starting weights are drawn using the Glorot rule.** Training has to
  begin from somewhere, and the weights start as random numbers. The Glorot
  rule sets how wide a range to draw them from, based on how many numbers enter
  and leave each layer. Its point is to keep the numbers flowing through the
  network at a steady size: start them too large and they grow layer after
  layer until they overflow, too small and they shrink towards nothing and the
  network learns nothing. That is the rule's aim; for these layers it is not a
  guarantee. Each layer adds up its inputs, each multiplied by a weight, and
  passes the total through a function. Glorot and Bengio (2010, printed page
  253, the fifth page of the copy in `docs/references/`) work out the range by
  assuming that this function, at the start, passes numbers near zero on
  unchanged: its slope at zero is 1, as it is for the hyperbolic tangent
  (`math.tanh` in Python), one of the S-shaped functions they test. These graph
  layers instead apply ReLU, which keeps a positive number and replaces a
  negative one with zero, so its slope is 1 for positive numbers and 0 for
  negative ones. He et al. (2015, printed page 1029, the copy's fourth page)
  state that Glorot's assumption does not hold for ReLU.
  The [GCN resource page](../../../docs/resources/gcn.md#starting-values-glorot-initialisation-and-the-relu-caveat)
  works both through on these three layers; whether the difference matters to
  training here has not been measured. Wen et al. never say how their weights
  start. Glorot is what Kipf and Welling (2017) use for this exact layer, so it
  is what is used here. PyTorch calls the same
  rule `xavier_uniform_`, after Xavier Glorot who proposed it — the two names
  refer to one method, not two. The dense layers keep PyTorch's own default
  instead (assumption A-14).

  **Correction, 2026-10-02.** Until this date the paragraph presented the
  steady size as something the rule delivers here, without the caveat about
  ReLU. Found when the cited papers were collected into `docs/references/` and
  checked against the places that cite them.
- **The LSTM stage is off by default** (assumption A-11), and when it is off
  **each node carries the last cycle of its window** (assumption A-07) — both
  explained above. Its size, when it is on, is a choice too (assumption A-26).
- **Targets are divided by the label ceiling** (assumption A-13) and **test
  labels are capped at the same ceiling** (assumption A-02) — both explained
  above. The code for
  those two lives mostly outside this folder, but the reason for them is the
  sigmoid defined here.

## References

Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep
feedforward neural networks. In Y. W. Teh & M. Titterington (Eds.),
*Proceedings of the Thirteenth International Conference on Artificial
Intelligence and Statistics* (Vol. 9, pp. 249–256). PMLR.
https://proceedings.mlr.press/v9/glorot10a.html

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers:
Surpassing human-level performance on ImageNet classification. In
*Proceedings of the IEEE International Conference on Computer Vision (ICCV)*
(pp. 1026–1034). https://doi.org/10.1109/ICCV.2015.123

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph
convolutional networks. *International Conference on Learning Representations
(ICLR)*. https://arxiv.org/abs/1609.02907

Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
168, 110875. https://doi.org/10.1016/j.ast.2025.110875

## What each name means

### `ESTGNN(adjacency, ...)`

| Argument | What it is |
|---|---|
| `adjacency` | The sensor graph, **already through `normalize_adjacency()`**. A grid with a zero anywhere on its diagonal is rejected, because that means it has not been through it. |
| `temporal` | The temporal stage from `models/temporal/` that turns each sensor's window into what the graph layers read. When it is left out, the model builds `LastCycle(window_size)`. The first graph layer takes its width from the stage. |
| `window_size` | How many cycles each sensor's window holds, used only when `temporal` is left out. Defaults to 1, which makes the default stage pass the single reading straight through, so `ESTGNN(adjacency)` alone is exactly Table 4. |
| `allow_time_in_the_graph_layer` | Defaults to `False`, which refuses a stage that puts more than one number per sensor on the graph layers without learning anything across the cycles. Read the section on one number per sensor above before setting it. |
| `rul_max` | The label ceiling, used to turn the network's output back into flight cycles. Defaults to 125. |
| `gcn_hidden` | How wide each graph convolution is. Defaults to Table 4's `(32, 64, 32)`. |
| `fc_hidden` | How wide the first fully connected layer is. Table 4 gives 128. |
| `gcn_bias` | Whether the graph convolutions carry a bias. Defaults to `False`, as explained above (assumption A-12). |
| `spatial` | `"gcn"`, the default, builds the graph convolutions; `"none"` builds none and flattens the temporal stage's output straight into the first dense layer (assumption A-42). `gcn_hidden` is then not read, and `gcn_bias=True` is refused. |

| Method | What it does |
|---|---|
| `forward(features)` | The prediction as a fraction between 0 and 1, one per example. This is what training uses. |
| `predict_cycles(features)` | The same thing multiplied by `rul_max`, so it is in flight cycles. This is what scoring uses. |
| `count_parameters()` | How many numbers the network learns. Compare it with Table 4. |
| `has_graph_layers` | `True` when the network has graph convolutions, `False` when it was built with `spatial="none"`. Read as a value, without brackets. The explanations of the graph in `explain/` refuse a network for which it is `False`. |
| `layer_summary()` | One dictionary per row of Table 4 — its name, the shape going in and out, how many numbers it learns, and its activation. The temporal stage is folded into the Input row, which counts what the stage learns and names it when it learns anything; see [The Input row holds the temporal stage](#the-input-row-holds-the-temporal-stage). |

`adjacency` is stored as a **buffer** — a number the model carries but training
never changes, explained under
[Training never changes the adjacency matrix](#training-never-changes-the-adjacency-matrix)
above. It is saved alongside the weights and moves with them onto a graphics
card, but no amount of training will alter it.

### `GraphConvolution(in_features, out_features, bias=False)`

| Argument | What it is |
|---|---|
| `in_features` | How many numbers each sensor carries coming in. |
| `out_features` | How many it carries going out. |
| `bias` | Whether to add a learned offset after the multiplication. Defaults to `False`, as explained above (assumption A-12). |

`forward(features, adjacency)` takes the readings shaped
`(batch, sensors, in_features)` and the rescaled graph shaped
`(sensors, sensors)`, and returns `(batch, sensors, out_features)`. It checks
every dimension first and names the mismatch if one is wrong.
`reset_parameters()` re-draws the starting weights.

The layer holds `weight`, the matrix `W`, and `bias`, which is `None` unless
you asked for one. It does **not** hold the graph: that is passed in on every
call, which is why all three layers can share one and it can be swapped without
rebuilding the network.
