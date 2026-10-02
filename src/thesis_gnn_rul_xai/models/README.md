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
| `estgnn.py` | `ESTGNN`, the whole network: a temporal stage, three graph convolutions, a flatten, two dense layers. |
| `temporal/` | The temporal stages, each a different way of turning one sensor's 30-cycle window into what the graph layers read. That folder has its own README. |
| `config.py` | `ModelConfig`, the seven settings that decide the network's shape, kept in one record a run can save; and `build_model`, which builds the network such a record describes. Explained under [Writing down which network a run used](#writing-down-which-network-a-run-used). |
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

Built with those defaults the model has 61,729 trainable parameters, which is
the count Table 4's printed shapes imply. Table 4 lists no total of its own:
its five weight matrices come to 61,600, and the remaining 129 are the two
biases Equations 12 and 13 print for the dense layers. `ESTGNN.count_parameters()` returns that number and
`ESTGNN.layer_summary()` returns one dictionary per layer in the same form as
Table 4, so the transcription can be checked line by line.

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

It depends on the temporal stage the run used, and every run records which.

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

## Writing down which network a run used

### The problem

A stored result is worth quoting only if someone can say which network
produced it. The shape of the network is decided by seven settings, and they
arrive from different places: the training script's command line, the
documentation site's playground, and, when an explanation is computed later,
the folder of a run that finished days ago. If each of those places built the
network its own way, two of them could read the same settings and build two
different networks, and nothing would say so.

`config.py` answers that with two things. `ModelConfig` is one record holding
all seven settings. `build_model` is the one function that turns such a record
into a network. The training script, the playground and the reader of stored
runs in `training/stored.py` all go through both.

### The seven settings

| Setting | Default | Where the default comes from |
|---|---|---|
| `temporal` | `"last_cycle"` | Our choice: keep the most recent reading of each window (assumption A-07), so the paper's LSTM stage is off (assumption A-11). |
| `temporal_hidden` | `32` | Our choice, since the paper never gives the LSTM's size (assumption A-26). Read only by `lstm` and `gru`, the two stages that read a window one cycle at a time and carry a running summary of size `temporal_hidden` from cycle to cycle. |
| `temporal_out_features` | `1` | How many numbers each sensor hands the graph layers. Table 4 prints the input to the first graph layer as `[batch size,14,1]`. |
| `shared_across_sensors` | `False` | Section 2.2 of the paper gives each sensor "a dedicated LSTM network", so one network shared by all fourteen is not the paper. |
| `gcn_hidden` | `(32, 64, 32)` | Table 4: GCN-1, GCN-2 and GCN-3 output 32, 64 and 32 numbers per sensor. |
| `fc_hidden` | `128` | Table 4: FC-1 outputs 128. |
| `gcn_bias` | `False` | Our choice: Table 4 lists only a weight matrix for each graph layer (assumption A-12). |

A record checks its values the moment it is made, so a bad setting stops the
program before any training starts rather than after an hour of it:

```text
ModelConfig(temporal="transformer")
  -> ValueError: temporal must be one of ['last_cycle', 'mean_window', 'window_summary', 'raw_window', 'lstm', 'gru'], got 'transformer'.
ModelConfig(fc_hidden=0)
  -> ValueError: fc_hidden must be at least 1, got 0.
```

`recurrent` is a yes-or-no answer read off the record: `True` for `lstm` and
`gru`, the two stages that have a hidden size at all, and `False` otherwise.

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
networks would be filed under one name. Writing all seven settings into a
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
count as a difference. Any other setting that differs from the default adds
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
choose. The full reasoning for every one is in `docs/ASSUMPTIONS.md`, under the
code given in parentheses.

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

| Method | What it does |
|---|---|
| `forward(features)` | The prediction as a fraction between 0 and 1, one per example. This is what training uses. |
| `predict_cycles(features)` | The same thing multiplied by `rul_max`, so it is in flight cycles. This is what scoring uses. |
| `count_parameters()` | How many numbers the network learns. Compare it with Table 4. |
| `layer_summary()` | One dictionary per layer — its name, the shape going in and out, how many weights it holds, and its activation — in the same form as Table 4. |

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
