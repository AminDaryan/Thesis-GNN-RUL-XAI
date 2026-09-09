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
| `estgnn.py` | `ESTGNN`, the whole network: three graph convolutions, a flatten, two dense layers. |
| `__init__.py` | Re-exports `ESTGNN` and `GraphConvolution`. |

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

Each node carries a single number. Table 4 prints the input to the first graph
layer as `[batch, 14, 1]` with a weight matrix of shape `1 × 32`. In the paper,
that number is the output of a per-sensor LSTM — but the paper never states
that LSTM's hidden size, its depth, or how it reduces a 30-cycle window to one
number. This project leaves that stage out rather than invent it, and feeds
each node the most recent reading of its window instead. Those two decisions
are recorded as **A-11** (omit the LSTM stage) and **A-07** (use the last cycle
of the window) in `docs/ASSUMPTIONS.md`.

The `ESTGNN` constructor still accepts any `in_features`, because the model
should not hard-code a number the data pipeline decides. **Do not read that as
an invitation to pass 30.** Handing each node its whole 30-cycle window makes
the first weight matrix `30 × 32`, which is 32 learned linear filters over
time, shared across sensors. That is a temporal encoder — exactly the thing
A-11 declines to invent — hidden inside a layer that is not labelled as one. It
is not the configuration this project runs.

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
   where the output layer can reach them. That decision is recorded as **A-13**
   in `docs/ASSUMPTIONS.md`. The division happens in `training/run.py`; the
   multiplication back into cycles happens here, in `ESTGNN.predict_cycles`.

2. **The model can never predict a remaining life above `rul_max`, for any
   input at all.** This is a hard fact about the architecture, not a tendency.
   It is why the project also caps the official test labels at 125 before
   scoring — a decision recorded as **A-02** — because leaving them uncapped
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
never states their size, so A-26 had to choose one. F-16 records what the stage
is worth when it is trained.

## Judgement calls made in this folder

Each of these is a place where the paper does not say what to do and we had to
choose. The full reasoning for every one is in `docs/ASSUMPTIONS.md` under the
code given.

- **The graph layers carry no bias.** The paper's Equations 8 to 10 show only
  the weight matrix `W`, while Equations 12 and 13, describing the dense
  layers, print an explicit bias. Within the same short section it prints
  biases where it means them, so the omission reads as deliberate. We follow
  the equations as printed. `GraphConvolution` takes `bias=True` so the
  difference can be measured rather than argued about. This is **A-12**.
- **The starting weights are drawn using the Glorot rule.** Training has to
  begin from somewhere, and the weights start as random numbers. The Glorot
  rule sets how wide a range to draw them from, based on how many numbers enter
  and leave each layer. Its point is to keep the numbers flowing through the
  network at a steady size: start them too large and they grow layer after
  layer until they overflow, too small and they shrink towards nothing and the
  network learns nothing. The paper never says how it starts. Glorot is what
  Kipf and Welling (2017) use for this exact layer, so it is what is used here.
  PyTorch calls the same rule `xavier_uniform_`, after Xavier Glorot who
  proposed it — the two names refer to one method, not two. The dense layers
  keep PyTorch's own default instead. This is **A-14**.
- **The LSTM stage is off by default** (**A-11**), and when it is off **each node
  carries the last cycle of its window** (**A-07**) — both explained above. Its
  size is **A-26**.
- **Targets are divided by the label ceiling** (**A-13**) and **test labels are
  capped at the same ceiling** (**A-02**) — both explained above. The code for
  those two lives mostly outside this folder, but the reason for them is the
  sigmoid defined here.

## References

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
| `in_features` | How many numbers each sensor carries. Defaults to 1, which is what Table 4 prints and what the preprocessing produces. Read the warning above before passing anything else. |
| `rul_max` | The label ceiling, used to turn the network's output back into flight cycles. Defaults to 125. |
| `gcn_hidden` | How wide each graph convolution is. Defaults to Table 4's `(32, 64, 32)`. |
| `fc_hidden` | How wide the first fully connected layer is. Table 4 gives 128. |
| `gcn_bias` | Whether the graph convolutions carry a bias. Defaults to `False`; see **A-12** above. |

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
| `bias` | Whether to add a learned offset after the multiplication. Defaults to `False`; see **A-12** above. |

`forward(features, adjacency)` takes the readings shaped
`(batch, sensors, in_features)` and the rescaled graph shaped
`(sensors, sensors)`, and returns `(batch, sensors, out_features)`. It checks
every dimension first and names the mismatch if one is wrong.
`reset_parameters()` re-draws the starting weights.

The layer holds `weight`, the matrix `W`, and `bias`, which is `None` unless
you asked for one. It does **not** hold the graph: that is passed in on every
call, which is why all three layers can share one and it can be swapped without
rebuilding the network.
