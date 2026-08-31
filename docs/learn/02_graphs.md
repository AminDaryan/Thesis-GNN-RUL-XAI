# Graphs, and what a graph network computes

## A graph is points and connections

A **graph** is a set of points, called **nodes**, with connections between
them, called **edges**. That is the whole definition. A social network is a
graph — people are nodes, friendships are edges. A road map is a graph —
junctions are nodes, roads are edges.

Here, **every sensor is a node**, and an edge between two sensors means "these
two measurements are physically related". There are 14 sensors, so 14 nodes.

## The adjacency matrix is the graph written as a table

An **adjacency matrix** is a graph written as a grid of numbers. For 14 sensors
it is a 14-by-14 table where the entry in row *i*, column *j* is 1 if sensors
*i* and *j* are connected and 0 if they are not. The graph and the matrix are
the same object in two notations; the matrix is the one a computer can multiply.

{{figure:adjacency}}

The left panel is the graph as the paper prints it, in its Table 3. It is
**symmetric** — if `T24` is a neighbour of `P30` then `P30` is a neighbour of
`T24` — and its diagonal is zero, because a sensor is not listed as its own
neighbour. Both of those are checked the moment this project's code is
imported, because a silent transcription error in the table would corrupt every
downstream result while leaving the code apparently working.

The right panel is the same matrix after **normalisation**, which is explained
further down.

---

## The graph this project uses

{{figure:sensor-graph}}

This is the paper's graph, drawn. Measured from the transcription in this
repository: **14 nodes, 41 edges**, connected, with a **diameter of 4** —
meaning the two most distant sensors are four edges apart.

Two features of it are worth pausing on, because both come back later.

**`T24` is the only articulation point.** An articulation point is a node whose
removal would split the graph into disconnected pieces. Here, the four
bypass-side sensors on the left — `P15`, `Nf`, `NRf`, `BPR` — reach the engine
core through `T24` and through nothing else. That is not an artefact of the
drawing; it is a property of the matrix, computed by the code.

**The graph is a physical claim, not a statistical one.** The edges were
hand-designed by the paper's authors from engine physics, not learned from
correlations in the data. That makes it a **prior**: knowledge put into the
model from outside, before it sees any measurements. Whether that prior earns
its place is a question the paper answers with an ablation, and one this
project has measured for itself — see page 6.

---

## What a graph neural network does: message passing

A **graph neural network** (GNN) is a neural network that operates on a graph.
The mechanism is called **message passing**, and one round of it is three
steps:

1. **Collect.** Every node looks at its neighbours and gathers their current
   values.
2. **Combine.** It mixes those values together with its own.
3. **Update.** It transforms the mixture — the same transformation everywhere
   in the graph — to produce its new value.

Every node does this at the same time. One round is one **layer**. Stack three
layers and information travels up to three edges.

{{figure:message-passing}}

Watch a single value leave `P15` and spread. After one layer it has reached
`P15`'s immediate neighbours. After two, their neighbours. After three, almost
everything.

**Almost.** `W32` is four edges from `P15`, and three layers cannot cross four
edges. No amount of training fixes this — it is a property of the architecture,
not of the weights. `P15` and `W32` are never mixed, and neither are three
other pairs.

{{figure:reach}}

Of the 91 pairs of sensors, 41 are directly connected, 25 are two edges apart,
21 are three apart, and **four are four apart** — `BPR`–`W32`, `NRf`–`W32`,
`Nf`–`W32` and `P15`–`W32`. This project has a test whose whole job is to check
that the network really does respect this limit, called
`test_three_layers_cannot_reach_four_hops`. If message passing ever leaked
further than the graph allows, that test would catch it.

This is the kind of thing worth knowing about a model before trusting it: not
what it can do, but what it structurally cannot.

---

## The formula, and the one word doing the work

One graph convolution layer is this:

$$H' = \mathrm{ReLU}\left(\hat{A} H W\right)$$

Read it right to left. `H` is the current table of node values — one row per
sensor. `W` is a matrix of numbers the network learns; multiplying by it is the
"update" step, and it is the *same* `W` for every node, which is why the layer
works on a graph of any size. `Â` is the adjacency matrix, and multiplying by
it is the "collect and combine" step: row *i* of `ÂH` is a weighted sum of the
rows belonging to sensor *i*'s neighbours. `ReLU` is the non-linearity, which
simply replaces every negative number with zero.

The hat on `Â` is the normalisation:

$$\hat{A} = D^{-1/2} (A + I) D^{-1/2}$$

Two things happen there. Adding the identity matrix `I` puts a **self-loop** on
every node, so that a sensor's own value survives the mixing instead of being
replaced entirely by its neighbours'. Dividing by the degrees `D` stops
well-connected nodes from dominating: without it, a sensor with eight
neighbours would contribute twice as much as one with four simply by being
popular.

> **A choice hides here.** The paper says its matrix is "normalized ...
> incorporating self-loops" and never prints the formula. Two standard formulas
> fit that description — the symmetric one above, and the simpler row form
> $D^{-1}(A+I)$, which divides each row by its own degree so the row sums to 1.
> This project measured both. Row normalisation came out 0.35 cycles closer to
> the paper's figure with nothing held back (paired *p* = 0.047), and is the
> default. The full reasoning, including the fact that the advantage does *not*
> survive holding engines out, is assumption A-21.

---

## Why a graph at all

The honest answer is that it is a hypothesis, not a certainty. The claim is
that telling the model which sensors are physically related helps it more than
letting it work that out from data. The paper reports an ablation supporting
that claim, at a cost of 0.56 cycles.

This project has now measured the same ablation **three times, and got three
answers**:

| Configuration | With the physics graph | Fully connected | Cost of removing it |
|---|---|---|---|
| Current default — z-score, row normalisation, nothing held back | 16.694 ± 0.331 | 19.213 ± 0.056 | **+2.520** (*p* = 0.000057) |
| Min-max scaling, symmetric normalisation, 20 % held back | 20.353 ± 0.752 | 20.397 ± 0.395 | +0.044 (*p* = 0.913) |
| A since-removed node-feature mode | 13.45 ± 0.20 | 19.17 ± 0.45 | +5.7 |

Decisive, then nothing, then decisive again. **Nothing about the graph changed
between those three measurements.** What changed was the preprocessing and the
split — choices the paper does not specify at all.

That instability is finding F-04, and the instability is the finding. An
ablation is normally reported as a property of an architecture: *the physics
prior is worth X*. These measurements say it is not that. It is a property of
the architecture **and** the preprocessing **and** the split, and the paper
specifies only the first.

The consequence is sharper than any of the individual numbers: it makes the
paper's own ablation table unreproducible **in principle**, not merely in
practice. A reader cannot tell whether its 0.56 cycles would survive a
different input scaling, because the paper never says what its input scaling
was.

So the rule this project follows is: **a comparison is a fact about one whole
configuration, not about the two options being compared.** Name the
configuration whenever you quote one, and re-run it whenever anything moves.

---

## What you now know

- A **graph** is nodes and edges; the **adjacency matrix** is the same thing as
  a table of numbers.
- The sensor graph has **14 nodes and 41 edges**, is connected, has diameter 4,
  and `T24` is its only articulation point.
- **Message passing** is collect–combine–update, one round per layer, and three
  layers reach exactly three edges.
- **Four sensor pairs are beyond reach** of this architecture, permanently.
- The `Â` in the formula carries **self-loops** and a **degree normalisation**,
  and which normalisation to use was a choice (A-21).
- What the physics graph is worth has been measured **three times with three
  answers** — 5.7, 0.044 and 2.520 cycles — under settings the paper leaves
  open. That instability is finding F-04.

Next: how to make the model explain itself.
