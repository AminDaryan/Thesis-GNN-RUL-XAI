# GNN-LRP (2022)

A graph neural network computes along a graph. At every graph layer, each
node — in this project, one sensor — replaces its numbers with a combination
of the numbers held by the nodes it is joined to. Two different questions can
then be asked of one prediction: *which readings mattered?*, and *along which
connections, and in which layer, did they travel?* Schnake et al. (2022,
Abstract; [p. 3](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=3))
wrote their paper to answer the second. Their method, **GNN-LRP**
(graph neural network layer-wise relevance propagation), splits one output of
a fixed network into signed shares, one for each **walk**: an ordered list of
nodes, one at the input and one after each graph layer, that part of the
output passed through (Section 3).

This page gives the papers' own account: the problem the method was built
for, how it is derived, how it is computed, how its authors tested it, and
what it does not show. It cites version 3 of the paper on arXiv
([Schnake et al., 2022](../references/schnake_2022_gnn_lrp_relevant_walks.pdf))
by that version's own section, equation, table and page numbers; the journal
version may number them differently. It also uses the two papers the method
is built on, [Bach et al. (2015)](../references/bach_2015_layerwise_relevance_propagation.pdf)
and [Montavon et al. (2019)](../references/montavon_2019_lrp_overview.pdf),
and the node-level method it was compared against,
[Pope et al. (2019)](../references/pope_2019_explainability_gcn.pdf)
(Schnake et al., 2022, Sections 3.2 and 4).

How this project runs the method on its own network, step by step, is the
subject of the [GNN-LRP tutorial](../learn/11_gnn_lrp.md), and the
[XAI page](../learn/03_explainability.md) says how it differs from SHAP. This
page does not repeat them; it is the papers' account.

## The problem the paper set out to solve

In an ordinary network the input is one list of numbers entering the first
layer, and an explanation can divide the output among those numbers. A graph
network has two inputs. One is the table of **node features** — here, each
sensor's reading. The other is the graph itself, written as a table **Λ**
(capital lambda) with one row and one column per node, whose entries say how
strongly each node takes in each other node's numbers (Schnake et al., 2022,
[Section 2, Eqs. 1–3](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=2)).
The paper's abstract and its Section 2 (p. 2) stress that the graph does not
enter once, at the start: every graph layer uses it again, so the graph is
woven through the whole calculation.

The paper by Schnake et al. (2022) describes the explanation methods for
graph networks that came before it as limited to individual nodes or edges
([p. 1](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=1)), and
in [Section 1.1.2](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=2)
sets itself apart from them by scoring sequences of edges instead; the methods
it lists there also include some that pick out a subgraph, such as
GNNExplainer (Ying et al., 2019, p. 1). It argues that a score per node
"does not tell us whether a node is important by itself"
([Section 2.1, p. 3](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=3)),
or only because of what it is connected to. Its answer is to make the graph
the thing that is explained: work out how the output depends on the entries
of Λ, and split the output into terms that each multiply as many entries of
Λ as the network has graph layers (Section 2.2, Eq. 6); Section 3 then
assigns each of those entries to one layer, which gives the walks.

**Do not confuse the graph table with the node feature table.** Λ says which
nodes may combine and with what weight; in this project it is the same for
every engine window. The node features are what was measured, and they change
with every window. A walk's share involves both: it starts at one node's
features and follows entries of Λ.

## What is being explained

The object explained is **one output number of one fixed network for one
input**. Where the paper says which output it explains, that number is taken
before any final step that maps a score into a fixed range: for its sentiment
classifier, the difference between the positive and the negative class scores
before a softmax (the step that turns class scores into probabilities) is
applied (Schnake et al., 2022, [Fig. 7](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=9);
Appendix F.2);
for molecules, the predicted energy, for which the paper names no such step,
and the dipole
moment (a measure of how a molecule's electric charge is separated) after the
paper replaces that output by a linear one
([Section 5.2](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=10)).
For its synthetic graphs and for its image classifier the paper does not say
which output it explains.

The explanation is a set of **relevances**: signed numbers, one per input
element (or, for GNN-LRP, one per walk), which add up to the output. A
positive relevance pushes the output up; a negative one pulls it down
(Bach et al., 2015, [p. 3, Eq. 1](../references/bach_2015_layerwise_relevance_propagation.pdf#page=3);
Schnake et al., 2022, Proposition 1 and Fig. 4). To
compute them, GNN-LRP reads every weight of the network, the graph, and the
state of every switch (defined below) for this one input; it changes no input
and never runs the network on any other input (Schnake et al., 2022, Table 1
and Section 3.2.1).

**Do not confuse activation with relevance.** An *activation* is the number a
node or a neuron — one of the numbers a layer computes — outputs during the
prediction. A *relevance* is the part of the final output credited to it
afterwards (Bach et al., 2015, p. 5). A large positive activation can carry
a small or negative relevance when the layers above multiply it by a small or
negative weight.
The example below has one: T30's below-average part is 0.418705 as an
activation and −0.069784 as a relevance.

## The worked example used on this page

Every number on this page comes from one small network fed with real
readings: FD001 **test engine 75 at its last available cycle, 88**, sensors
T30, T50 and W32, after the project's own preparation with seed 42 —
smoothing (Wen et al., 2026, Section 3.2.3; assumption A-04), then
rescaling with averages and spreads taken from training engines only
(assumptions A-03 and A-25), as the
[preparation tutorial](../learn/04_the_pipeline.md) shows. A rescaled
reading is the distance from the sensor's training average in units of its
training spread, so a negative value means below average.

The three sensors form a path in the project's sensor graph, the table of
connections printed by Wen et al. (2026, Table 3): T30 is joined to
T50, and T50 to W32, but T30 is not joined to W32. The page cuts these three
rows and columns out of the project's graph, joins each sensor to itself, and
rescales each row of the small table to add up to 1, the way the project
rescales its full graph (assumption A-21). So T30 takes half its input from
itself and half from T50; T50 takes a third from each of the three; W32 takes
half from T50 and half from itself; the entries between T30 and W32 are zero.
In the project's full 14-sensor graph the same rows are spread wider: T30
takes 1/9 from each of nine sensors, and T50 and W32 take 1/5 from each of
five.

The network is an **untrained demonstration**. Its few multipliers are set by
hand so that each step of the papers' mechanism shows up in the arithmetic.
It is not a stored training run, and its prediction says nothing about
engines or about accuracy.

- **Graph layer 1** averages each sensor's permitted inputs, multiplies the
  average by 1 and by −1 to give two numbers per sensor, and sets negative
  results to zero. That last operation is a **ReLU** (rectified linear unit):
  a switch that passes a positive number unchanged and outputs 0 otherwise,
  the rectification function $\max(0, \cdot)$ (Schnake et al., 2022, Table 1).
  The first number is therefore the average where it is above zero (the
  *above-average part*) and the second its size where it is below zero (the
  *below-average part*).
- **Graph layer 2** averages the permitted inputs' two-number rows,
  multiplies the first number by 1 and the second by −0.5, adds them, and
  applies the switch.
- **The last step** adds the three sensors' outputs (weights 1, 1 and 1),
  applies a switch, and multiplies by 1. The result is the **score**. The
  project's network turns a score into cycles with the sigmoid curve
  (Wen et al., 2026, Eq. 13), here 125 ÷ (1 + e^(−score)), which maps any
  number into the range 0 to 125 (assumptions A-01 and A-13); this
  page, like the project, divides the score, not the cycles (assumption A-29).

No layer adds a constant of its own. That matters below: it is the condition
under which the paper's derivation is exact (Schnake et al., 2022,
Proposition 1; p. 4). Tables show six decimal places;
the calculations keep full precision, so adding printed values can differ
from a printed total in the last digit. Every Python block on this page needs
only NumPy and pandas, except the one in "The masked procedure, and the
project's code", which needs PyTorch.

<details class="process-example">
<summary>Set up the Python examples</summary>

```python
import itertools
import numpy as np
import pandas as pd
from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.data.sensors import SENSORS
from thesis_gnn_rul_xai.graph.adjacency import build_physics_adjacency, normalize_adjacency

prepared = run_preprocessing(PreprocessingConfig(seed=42))
names = ["T30", "T50", "W32"]
chosen = [list(prepared.sensor_symbols).index(name) for name in names]
row = int(np.flatnonzero(prepared.test.units == 75)[0])
end_cycle = int(prepared.test.end_cycles[row])
window = prepared.test.features[row, chosen, -1:]    # 3 sensors x 1 cycle
x = window[:, 0].astype(np.float64)                  # the three readings
physics, _ = build_physics_adjacency()
path = physics[np.ix_(chosen, chosen)]               # 1 where two sensors are joined
lam = normalize_adjacency(path, method="row").astype(np.float64)
full = normalize_adjacency(physics, method="row")    # the project's 14-sensor graph

W1 = np.array([[1.0, -1.0]])     # graph layer 1: one number in, two out
W2 = np.array([[1.0], [-0.5]])   # graph layer 2: two numbers in, one out
u = np.array([1.0, 1.0, 1.0])    # last step: add the three sensors' numbers

described = {sensor.symbol: sensor.description for sensor in SENSORS}
readings = pd.DataFrame({
    "Sensor": names,
    "What it measures": [described[name] for name in names],
    "Rescaled reading": x,
})
assert end_cycle == 88
np.testing.assert_allclose(lam, [[1/2, 1/2, 0], [1/3, 1/3, 1/3], [0, 1/2, 1/2]], atol=1e-7)
assert [int((full[i] > 0).sum()) for i in chosen] == [9, 5, 5]
np.testing.assert_allclose(full[chosen].max(axis=1), [1/9, 1/5, 1/5], atol=1e-7)
```

</details>

The three readings the whole page divides:

<!-- sample: readings -->
| Sensor | What it measures | Rescaled reading |
| --- | --- | --- |
| T30 | Total temperature at HPC outlet | -0.046632 |
| T50 | Total temperature at LPT outlet | -0.790778 |
| W32 | LPT coolant bleed | 1.364520 |

### The prediction, layer by layer

Each graph layer works on all three sensors at once. For T50, layer 1
averages all three readings: (−0.046632 − 0.790778 + 1.364520) ÷ 3 =
0.175703. That is positive, so T50's above-average part is 0.175703 and its
below-average part 0. For T30, (−0.046632 − 0.790778) ÷ 2 = −0.418705, so
its below-average part is 0.418705.

Layer 2 at T50 averages what the three sensors now hold, each row weighted
by 1 and −0.5: (−0.5 × 0.418705 + 0.175703 + 0.286871) ÷ 3 = 0.084407,
positive, so the switch passes it. At T30 the same step gives
(−0.5 × 0.418705 + 0.175703) ÷ 2 = −0.016825, and T30's switch outputs 0.

<details class="process-example">
<summary>Example in Python</summary>

```python
def forward(first, second, readings=x):
    """Run the demonstration network, with its own graph table for each layer."""
    z1 = (first @ readings[:, None]) @ W1    # layer-1 totals: 3 sensors x 2 numbers
    h1 = np.maximum(z1, 0.0)                 # the switch: a negative total becomes 0
    z2 = ((second @ h1) @ W2)[:, 0]          # layer-2 totals: one per sensor
    h2 = np.maximum(z2, 0.0)
    return z1, h1, z2, h2, max(float(u @ h2), 0.0)

z1, h1, z2, h2, score = forward(lam, lam)
prediction = 125 / (1 + np.exp(-score))
forward_table = pd.DataFrame({
    "Sensor": names, "Reading": x, "Layer-1 total": z1[:, 0],
    "Above-average part": h1[:, 0], "Below-average part": h1[:, 1],
    "Layer-2 total": z2, "Layer-2 output": h2,
})
score_table = pd.DataFrame({
    "Engine": [75], "Cycle": [end_cycle],
    "Score": [score], "Prediction (cycles)": [prediction],
})
```

</details>

<!-- sample: forward_table -->
| Sensor | Reading | Layer-1 total | Above-average part | Below-average part | Layer-2 total | Layer-2 output |
| --- | --- | --- | --- | --- | --- | --- |
| T30 | -0.046632 | -0.418705 | 0.000000 | 0.418705 | -0.016825 | 0.000000 |
| T50 | -0.790778 | 0.175703 | 0.175703 | 0.000000 | 0.084407 | 0.084407 |
| W32 | 1.364520 | 0.286871 | 0.286871 | 0.000000 | 0.231287 | 0.231287 |

The last step adds the three layer-2 outputs, and the sigmoid turns the
score into cycles:

<!-- sample: score_table -->
| Engine | Cycle | Score | Prediction (cycles) |
| --- | --- | --- | --- |
| 75 | 88 | 0.315695 | 72.284332 |

The question for the rest of the page: how should the score 0.315695 be
divided among the three readings, and among the routes they took?

## Where the method starts: layer-wise relevance propagation (2015)

### The idea: hand the output back, keeping the total

Bach et al. (2015, [p. 3](../references/bach_2015_layerwise_relevance_propagation.pdf#page=3))
introduced **layer-wise relevance propagation**, LRP, to
explain image classifiers pixel by pixel. They want one relevance $R_d$ per
input number $d$ such that $f(x) \approx \sum_d R_d$
([Eq. 1](../references/bach_2015_layerwise_relevance_propagation.pdf#page=3)).
Instead of computing the input shares directly, LRP starts with the whole
output at the top and hands it down one layer at a time, requiring that the
shares at every layer still add up to the output
([Eq. 2](../references/bach_2015_layerwise_relevance_propagation.pdf#page=4)):

$$f(x) = \cdots = \sum_{d \in l+1} R^{(l+1)}_d = \sum_{d \in l} R^{(l)}_d = \cdots = \sum_d R^{(1)}_d$$

- **Symbols.** $f(x)$ is the output for input $x$. $l$ numbers the layers
  from the input (layer 1) upwards. $d$ runs over the numbers computed at one
  layer (pixels, at layer 1). $R^{(l)}_d$ is the relevance of number $d$ at
  layer $l$.
- **What it says.** The sum of the shares is the same at every layer.
  Bach et al. (2015, p. 5) call this a conservation law.
- **Why it is needed.** Once the input layer is reached, the input shares
  must add up to the output, which is what Eq. 1 asks for.
- **Example.** In the table at the end of this section, the shares at the
  layer-2 outputs, at the layer-1 outputs and at the readings each add up to
  0.315695.
- **What uses it.** It constrains the rule that divides each number's
  relevance among the numbers that fed it: the rule may lose nothing.

Bach et al. (2015) also show why conservation alone is not
enough. Their Eq. 4
divides a linear classifier's output in proportion to the size of each
input's term: the total is kept, but every input receives the sign of the
output, whichever way it actually pushed
([p. 4](../references/bach_2015_layerwise_relevance_propagation.pdf#page=4)).
Conservation is a property a split must have, not evidence that a split is
right.

### The proportional rule

In the notation of Bach et al. (2015), for a layer in which lower number $i$
sends $z_{ij} = x_i w_{ij}$ to upper
number $j$ ([Eq. 50](../references/bach_2015_layerwise_relevance_propagation.pdf#page=19)),
the upper number totals $z_j = \sum_i z_{ij} + b_j$
(Eq. 51) and then passes $z_j$ through a function such as the switch
(Eq. 52), the first rule they give is
([Eq. 56](../references/bach_2015_layerwise_relevance_propagation.pdf#page=20)):

$$R^{(l,l+1)}_{i \leftarrow j} = \frac{z_{ij}}{z_j}\, R^{(l+1)}_j$$

- **Symbols.** $x_i$ is the lower number's value, $w_{ij}$ the weight joining
  it to $j$, $b_j$ a constant the upper number adds, $z_{ij}$ what $i$
  contributed to $j$, $z_j$ the total before the switch. $R^{(l+1)}_j$ is the
  relevance $j$ already holds, and $R_{i \leftarrow j}$ the part of it handed
  to $i$, which they call a message.
- **What it computes.** Each lower number receives the fraction of $j$'s
  total that it contributed.
- **Why.** Added over $i$, the messages give back $R_j$ times
  $1 - b_j / z_j$ (Eq. 57): the constant keeps the fraction $b_j / z_j$ for
  itself. With no constant, the relevance is passed on in full.
- **Example.** T50's layer-2 total, 0.084407, is made of −0.069784 from
  T30's below-average part, 0.058568 from T50's above-average part and
  0.095624 from W32's (the contributions are printed in the GNN-LRP section
  below). Its relevance is also 0.084407, because the last step adds the
  layer-2 outputs with weight 1. Each fraction times 0.084407 therefore
  returns the contribution itself.
- **What uses it.** Each lower number adds up the messages it receives from
  every upper number (Eq. 62); that sum is its relevance, and it is handed
  down again, layer after layer, to the input.

### The gamma rule (2019)

The overview by Montavon et al. (2019, [Section 10.2.1](../references/montavon_2019_lrp_overview.pdf#page=4))
calls the rule above **LRP-0** and states that
applying it at every layer gives the same shares as **gradient × input**:
each input's value times the rate at which the output changes when that
input alone changes (its partial derivative; the list of these rates over all
inputs is the gradient). Because the gradient of a deep network is noisy
(Section 10.1), Section 10.2.1 introduces a variant it names LRP-γ, the
gamma rule:

$$R_j = \sum_k \frac{a_j\,(w_{jk} + \gamma\, w^+_{jk})}{\sum_{0,j} a_j\,(w_{jk} + \gamma\, w^+_{jk})}\, R_k$$

- **Symbols.** In the overview's naming $j$ is the lower number and $k$ the
  upper one. $a_j$ is the lower activation, $w_{jk}$ the weight,
  $w^+_{jk} = \max(0, w_{jk})$ the weight if it is positive and 0 otherwise.
  The sum written $\sum_{0,j}$ includes a constant input $a_0 = 1$ whose
  weight $w_{0k}$ is the layer's added constant. **γ** (gamma), at least 0,
  is a setting of the explanation rule.
- **What it computes.** Before the fractions are taken, every contribution
  that comes through a positive weight is counted $(1 + \gamma)$ times. With
  γ = 0 it is LRP-0.
- **Why.** Weighting the positive contributions up leaves less of the total
  to be claimed by negative ones. In the overview's account, this keeps the
  positive and negative relevances from growing large as they are handed
  down, and the explanations come out steadier. Letting γ grow without limit
  turns the rule into the one it calls LRP-α1β0, which by the formula in its
  Appendix 10.A keeps positive contributions only.
- **Example.** The GNN-LRP section below applies γ = 1 at T50's layer-2 node.
- **What uses it.** GNN-LRP applies this rule at each graph layer
  (Schnake et al., 2022, Table 1), as set out below.

Three further points of the overview by Montavon et al. (2019) matter later.
Its Section 10.2.3
justifies these rules as Taylor expansions — approximations of a function by
its value and its rates of change at a reference point — made one neuron at a
time, at reference points that are never negative, which fits layers whose inputs are
switch outputs. Its Section 10.3.2 says an input layer is different, because
it receives raw values rather than switch outputs, and needs rules of its
own; the table in its Appendix 10.A lists the w²-rule and the zB-rule for
first layers. And for the image classifier VGG-16 its Section 10.3 uses
LRP-0 in the upper layers, LRP-ε (a variant that adds a small positive number
to each total, Section 10.2.1) in the middle layers and LRP-γ in the lower
layers, and it says that using LRP on a new task means inspecting that
network's own layers first.

**Do not confuse gamma with a learned weight.** The network's weights are set
by training (here, by hand), before anything is explained. γ is chosen by
whoever explains; it changes how the explanation divides the score and never
changes the score.

### Worked example: handing the score back node by node

Here LRP-0 is applied at every step: the last step hands each layer-2 output
its share of the score, layer 2 hands those shares to the six numbers
layer 1 produced, and layer 1 hands them to the three readings.

<details class="process-example">
<summary>Example in Python</summary>

```python
def node_level(gamma):
    """Hand the score back one layer at a time; gamma is used at graph layer 2 only."""
    at_layer2 = u * h2 / (u @ h2) * score
    counted = W2[:, 0] + gamma * np.maximum(W2[:, 0], 0.0)
    at_layer1 = np.zeros_like(h1)
    for end in range(3):
        if z2[end] > 0:
            parts = lam[end, :, None] * h1 * counted
            at_layer1 += parts / parts.sum() * at_layer2[end]
    at_readings = np.zeros(3)
    for middle in range(3):
        for j in range(2):
            if h1[middle, j] > 0:
                parts = lam[middle] * x * W1[0, j]
                at_readings += parts / parts.sum() * at_layer1[middle, j]
    return at_layer2, at_layer1, at_readings

at_layer2, at_layer1, at_readings = node_level(0.0)
step = 1e-6
slope = np.array([
    (forward(lam, lam, x + step * e)[4] - forward(lam, lam, x - step * e)[4]) / (2 * step)
    for e in np.eye(3)
])
layer_totals = pd.DataFrame(
    [at_layer2, at_layer1.sum(axis=1), at_readings, x * slope], columns=names,
)
layer_totals.insert(0, "Where the shares sit", [
    "Layer-2 outputs", "Layer-1 outputs", "Readings", "Reading × slope of the score",
])
layer_totals["Total"] = layer_totals[names].sum(axis=1)
np.testing.assert_allclose(at_readings, x * slope, atol=1e-8)
```

</details>

<!-- sample: layer_totals -->
| Where the shares sit | T30 | T50 | W32 | Total |
| --- | --- | --- | --- | --- |
| Layer-2 outputs | 0.000000 | 0.084407 | 0.231287 | 0.315695 |
| Layer-1 outputs | -0.069784 | 0.146420 | 0.239059 | 0.315695 |
| Readings | -0.016839 | -0.615049 | 0.947583 | 0.315695 |
| Reading × slope of the score | -0.016839 | -0.615049 | 0.947583 | 0.315695 |

Read down the Total column: 0.315695 at every layer, which is Eq. 2 of
Bach et al. (2015). Along the rows:

- **Layer-2 outputs.** The last step adds the outputs with weight 1, so each
  one's relevance is its own value. T30's is 0 because its switch is off.
- **Layer-1 outputs.** T30's −0.069784 is the activation-versus-relevance
  case from above: its below-average part is +0.418705, but it reaches T50
  through the weight −0.5, and T50 averages it with a third. Its other route,
  into T30's own layer-2 node, ends at a switch that is off.
- **Readings.** T50's reading is far below average (−0.790778) and enters
  the averages with positive entries, so it pulls the score down: −0.615049.
- **The last row** is each reading times the slope of the score with respect
  to it, the slope measured by nudging that reading by 0.000001 either way.
  It matches the readings row, as Montavon et al. (2019, Section 10.2.1) say
  LRP-0 throughout must.

This is a **node-level explanation**: one share per sensor reading, the
first-order explanation on the initial node features that Schnake et al.
(2022, Section 2.1) describe. It is also what their benchmark used under
the name of Pope et al. (2019): an attribution onto the initial node features,
computed by gradient × input, as here, or by LRP
([Section 4, item (a)](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=6)).
Pope et al.'s own maps are a different kind of score, described in the next
section.

**The key idea is** that LRP cuts each number's relevance into pieces in
proportion to what each lower number contributed to it, so nothing is lost on
the way down and the input shares add up to the output.

## Why one share per sensor, or per connection, is not enough

Schnake et al. (2022, [Section 2.1](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=2))
write the output as a first-order Taylor expansion in
the node features (their Eq. 4): the output near a reference point is that
point's output plus one term per node, the node's change times the rate at
which the output changes with it. Each term is one node's share, like the
readings row above. The paper raises two limits
([p. 3](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=3)): such
a share reflects what a node does at the first layer and not in the layers
above, and it does not say whether a node mattered by itself or through its
connections. It traces both limits to having expanded the output as a
function of the node features rather than of the entries of Λ.

**Pope et al. (2019)** is the node-level method the paper compares against
(Schnake et al., 2022, Section 4).
Pope et al. (2019, Section 3.3) carry image-explanation methods over to graph
convolutions: a gradient map, which for each node takes the size of the
positive part of the derivative of a class score with respect to that node's
features ([Eq. 12](../references/pope_2019_explainability_gcn.pdf#page=4));
class activation mapping (Eq. 13); its gradient-weighted variant
(Eqs. 14–16); and excitation backpropagation
([Eqs. 17–20](../references/pope_2019_explainability_gcn.pdf#page=5)). Each
gives one number per node. They judge the methods partly by fidelity: the
drop in accuracy when every node whose score exceeds 0.01 is hidden
([Section 4.4](../references/pope_2019_explainability_gcn.pdf#page=6)).

In the example, T50's reading has the node-level share −0.615049. The walk
table further down splits it into five nonzero routes, among them
T50 → W32 → W32 (−0.197694) and T50 → T50 → W32 (−0.131796). A node-level
share holds their sum and cannot separate them.

**One score per connection fails too.** Schnake et al. (2022,
[Section 2.2](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=3))
give the reason: the graph enters every layer and the layers multiply what
they receive, so the output depends on *products* of graph entries, and an
analysis with one first-order term per entry of Λ cannot pick those products
apart. The demonstration shows what goes wrong. For each connection, take the
slope of the score with respect to its entry — changed in both layers at
once, since the two layers share one graph — times the entry: gradient ×
input, one connection at a time.

<details class="process-example">
<summary>Example in Python</summary>

```python
def nudged_everywhere(entry, by):
    """The score after changing one entry of the graph both layers share."""
    changed = lam.copy()
    changed[entry] += by
    return forward(changed, changed)[4]

connections = [(K, J) for K in range(3) for J in range(3) if lam[K, J] > 0]
edge_scores = pd.DataFrame({
    "Connection": [f"{names[J]} sends to {names[K]}" for K, J in connections],
    "Entry": [lam[K, J] for K, J in connections],
    "Slope × entry": [
        (nudged_everywhere((K, J), step) - nudged_everywhere((K, J), -step)) / (2 * step) * lam[K, J]
        for K, J in connections
    ],
})
np.testing.assert_allclose(edge_scores["Slope × entry"].sum(), 2 * score, atol=1e-8)
```

</details>

<!-- sample: edge_scores -->
| Connection | Entry | Slope × entry |
| --- | --- | --- |
| T30 sends to T30 | 0.500000 | -0.003886 |
| T50 sends to T30 | 0.500000 | -0.065898 |
| T30 sends to T50 | 0.333333 | -0.082738 |
| T50 sends to T50 | 0.333333 | -0.161093 |
| W32 sends to T50 | 0.333333 | 0.474657 |
| T50 sends to W32 | 0.500000 | -0.241639 |
| W32 sends to W32 | 0.500000 | 0.711986 |

The seven scores add up to 0.631389, exactly twice the score 0.315695.
Each route's product holds two graph entries, one per layer, and a
connection's score collects every route that uses it, once per use: W32's
connection to itself collects 0.227420 and 0.341130 from the two routes that
use it in layer 1, and −0.197694 and 0.341130 from the two that use it in
layer 2 (the GNN-GI column of the walk table below), so W32 → W32 → W32 is
counted twice. With T graph layers and no added constants, this total is T
times the output: the total is the rate at which the output changes when
every entry is multiplied by the same $s$, taken at $s = 1$, and Step 3 below
shows that the output then changes as $s^T$, whose rate at $s = 1$ is T. A
first-order expansion made around the empty graph, where
Proposition 1 below makes the expansion exact, fares no better: every
first-order term there is 0 (Schnake et al., 2022, Appendix A). Hence the
paper's turn to terms that multiply T entries.

## From the graph to walks: the paper's derivation

### The network as a function of its graph

Schnake et al. (2022, [Section 2](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=2))
write one graph layer $t$ as an *aggregate* step and a *combine* step
(Eqs. 1 and 2). For the graph convolution of Kipf and Welling (2017, Eq. 2)
— the layer the [GCN page](gcn.md) derives, and the kind this project's
graph layers are — the GNN-LRP paper's
[Table 1](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=5)
gives the combine step explicitly:

$$Z_t = \Lambda H_{t-1}, \qquad H_t = \rho(Z_t W_t)$$

- **Symbols.** $H_{t-1}$ is an $n \times d_{t-1}$ table: one row per node
  ($n$ nodes), $d_{t-1}$ numbers per row. $\Lambda$ is the $n \times n$ graph
  with each node joined to itself. $Z_t$ is the aggregated table. $W_t$ is
  the layer's $d_{t-1} \times d_t$ table of learned weights. $\rho(\cdot) =
  \max(0, \cdot)$ is the switch, applied to every number.
- **What it computes.** Aggregate: each node's new row is a weighted sum of
  the rows of the nodes it is joined to, with entry $\Lambda[K, J]$ weighting
  node J's row in node K's sum. Combine: each row is multiplied by $W_t$ and
  switched.
- **Why.** It lets a node's numbers depend on its neighbours', and, after $T$
  layers, on nodes up to $T$ connections away.
- **Example.** T50 at layer 1: its row of Λ is (1/3, 1/3, 1/3), so
  $Z$ = 0.175703; times $W_1$ = (1, −1) gives (0.175703, −0.175703); the
  switch gives (0.175703, 0), the two parts in the forward table.
- **What uses it.** After $T$ graph layers a **readout** $g$, which is the
  final step that turns all nodes' numbers into the output, gives the output
  $f(\Lambda; H_0) = g(H_T(\Lambda, H_{T-1}(\ldots H_1(\Lambda, H_0))))$
  (Eq. 3). In the example, $T = 2$ and the readout is the last step.

**Do not confuse the paper's aggregate step with its combine step.** The
numbers arriving from the neighbours are added together in the step the paper
calls *aggregate*, $Z_t = \Lambda H_{t-1}$ (Schnake et al., 2022, p. 2). The
step it calls *combine*
involves no neighbour: it multiplies each node's own aggregated row by $W_t$
and applies the switch.

Schnake et al. (2022, p. 3) write $\lambda^\star_{JK}$, and $\lambda_{JK}$ in
their Table 1, for the element of the graph for the connection
between node J and node K; in a walk (…, J, K, …) it is the connection
through which J's numbers reach K at the next layer. The graphs whose
construction the paper writes out are built from undirected graphs and so
are symmetric ([Appendix F.1 and F.2](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=19)),
and it never has to say which index of Λ sends. This page's graph is not
symmetric once its rows are rescaled, and Eq. 1 settles the direction: node
K's aggregated row is $\sum_J \Lambda[K, J]\, H_{t-1}[J]$, so $\lambda_{JK}$
is the entry in row K, column J. In the example, T50 receives a third of its
input from T30, so $\lambda_{T30,T50}$ = 1/3, while T30 receives half of its
input from T50, so $\lambda_{T50,T30}$ = 1/2.

The derivation then takes five steps.

**Step 1 — hold the readings and the weights fixed, and vary the graph.**
With the node features $H_0$ and every weight fixed, the output depends only
on the entries of Λ: $f(\Lambda)$ (Schnake et al., 2022,
[Section 2.2](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=3)).

**Step 2 — between switch changes, the output is a polynomial of degree at
most T.** A polynomial is a sum of terms, each a number times a product of
variables; its degree is the largest number of variables multiplied together
in any one term. Each aggregate step multiplies by Λ once; with every switch
held in its current state, each combine step multiplies by fixed weights, and
may add a constant. After $T$ graph layers, and through a readout that is
also linear while its switches hold, no term contains more than $T$ entries
of Λ; a layer that adds a constant also creates terms with fewer. Schnake
et al. (2022, [Appendix A](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=15))
call this piecewise polynomial of order $T$:
one formula holds as long as no switch changes state, and an input on which
some switch flips lies on another piece, with another formula.

**Step 3 — with no added constants, scaling the graph by $s$ scales the
output by $s^T$.** In Proposition 1, Schnake et al. (2022,
[p. 3](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=3))
ask each combine step and the readout to be piecewise linear and
**positively homogeneous**: multiplying the input by a positive $s$
multiplies the output by $s$. The switch has this property, $\rho(s\,a) =
s\,\rho(a)$ for $s > 0$; a layer that adds a constant does not. Stacked over
$T$ graph layers, each of which multiplies by Λ once more, it makes $H_1$
scale by $s$, $H_2$ by $s^2$, and the output by $s^T$ (Appendix A, Eq. 2).
Appendix A draws two consequences (Eq. 3 and the sentence after it): every
term of the polynomial now holds exactly $T$ entries of Λ, and no switch
changes state anywhere along the line $s\Lambda$, $s > 0$, so one formula
covers that whole line.

**Step 4 — expand around a nearly empty graph.** A Taylor expansion writes a
function as a sum of terms built from its derivatives at a reference point
$\tilde\Lambda$; a term of order $k$ uses $k$-th derivatives and multiplies
$k$ entries' distances from the reference. Schnake et al. (2022,
[Eq. 5](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=3)) write
their terms of order $T$ in bags of connections and
choose the reference $\tilde\Lambda = s\Lambda$ with $s \to 0$: the graph
shrunk towards nothing, through which no information is exchanged. For
$k < T$, a $k$-th derivative of the degree-$T$ polynomial of Step 3 is itself
homogeneous of degree $T - k \ge 1$: every one of its terms still multiplies
at least one graph entry, so it falls to 0 as the graph shrinks to nothing.
For $k > T$ the derivatives of a degree-$T$ polynomial are all zero. And the
order-$T$ derivatives are constant on the piece (Appendix A, pp. 15–16).
What remains is exact (Eq. 6):

$$f(\Lambda) = \sum_{\mathcal{B}} R_{\mathcal{B}}, \qquad R_{\mathcal{B}} = \frac{1}{\alpha_{\mathcal{B}}!}\,\frac{\partial^T f}{\partial\lambda_{\mathcal{E}_1}\cdots\partial\lambda_{\mathcal{E}_T}}\;\lambda_{\mathcal{E}_1}\cdots\lambda_{\mathcal{E}_T}$$

- **Symbols.** A **bag** $\mathcal{B}$ is a group of $T$ connections
  $\mathcal{E}_1, \ldots, \mathcal{E}_T$ in which order does not matter and a
  connection may appear more than once. $\lambda_{\mathcal{E}}$ is the entry
  of Λ for connection $\mathcal{E}$. $\alpha_{\mathcal{B}}!$ multiplies the
  factorials of how often each connection appears in the bag (the factorial
  of 2 is 2 × 1 = 2). The fraction with $\partial^T f$ is the mixed
  derivative of order $T$; for $T = 2$ it says how much the slope of the
  output with respect to one entry changes when the other entry changes. In
  the example below, raising the entry "T50 receives from T30" by 0.001
  changes the slope of the score with respect to "W32 receives from T50" by
  0.001 × (−0.046632).
- **What it computes.** Each bag of connections receives its mixed
  derivative times its entries, and the bags' terms add up to the output with
  nothing left over.
- **Why.** Every Taylor expansion needs a reference point, and a poorly
  chosen one leaves an unexplained remainder. With the graph shrunk to
  nothing, the reference output is 0 for a network without constants, and
  the expansion is exact.
- **Example.** For $T = 2$, the bag holding "T50 receives from T30" and
  "W32 receives from T50" is worked out below: −0.007772. When a bag holds
  the same connection twice — W32's connection to itself, used in both
  layers — $\alpha_{\mathcal{B}}! = 2$, and the 1/2 cancels the factor 2 that
  comes from differentiating a squared entry twice.
- **What uses it.** Differentiating a whole network to order $T$ is
  impractical, so Section 3 rewrites the terms walk by walk.

**Step 5 — from bags to walks.**
Schnake et al. (2022, [Section 3](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=3))
define a **walk** as an ordered sequence of connections linking nodes in
consecutive layers, written by the nodes it visits, $(\ldots, J, K, L,
\ldots)$, and give each layer its own copy of the graph, $\Lambda^\star =
(\Lambda, \ldots, \Lambda)$. [Appendix B](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=16)
then argues in three moves (its Eqs. 7–9). A derivative with respect to a
shared entry is the sum of the derivatives with respect to that entry's copy
in each layer. A mixed derivative that takes two entries from the same
layer's copy is zero, because the output is piecewise linear in each copy, so
only terms that take exactly one entry from each layer survive. And of those,
only sequences of connections that form a walk on Λ are nonzero, which the
appendix states briefly, from the entries having to be nonzero.

Why consecutive connections must share a node — this page's own explanation,
not the appendix's: with every switch held, a layer-$t$ entry $\lambda_{JK}$
changes only node K's numbers at layer $t$, and a layer-$(t+1)$ entry
$\lambda_{K'L}$ multiplies only node K′'s numbers from layer $t$. Unless K and
K′ are the same node, neither entry can change the other's effect, and their
mixed derivative is zero.

Proposition 2 of Schnake et al. (2022,
[p. 4](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=4))
then gives each walk a score (Eq. 7), and a bag's term is the sum of its
walks' scores:

$$R_{\mathcal{W}} = \frac{\partial}{\partial \ldots}\left(\frac{\partial}{\partial\lambda^\star_{JK}}\left(\frac{\partial \ldots}{\partial\lambda^\star_{KL}}\,\lambda^\star_{KL}\right)\lambda^\star_{JK}\right)\cdots, \qquad R_{\mathcal{B}} = \sum_{\mathcal{W}\in\mathcal{B}} R_{\mathcal{W}}$$

- **Symbols.** $\mathcal{W}$ is a walk. $\lambda^\star_{JK}$ is the entry of
  the copy of Λ belonging to the layer in which the walk steps from J to K.
  The dots stand for the walk's other steps.
- **What it computes.** Differentiate the output with respect to the walk's
  last connection and multiply by it; differentiate that result with respect
  to the connection before, and multiply; continue to the first layer. For
  two layers and the walk (I, J, K) this is the mixed second derivative with
  respect to $\lambda^\star_{IJ}$ and $\lambda^\star_{JK}$, times both
  entries.
- **Why.** Each step is a first derivative, which one backward pass through
  the network computes — the sweep from the output back to the inputs that
  obtains every first derivative at once; no derivative of order $T$ of the
  whole network is needed (Section 3.1, p. 3).
- **Example.** The walk T30 → T50 → W32, below: −0.007772.
- **What uses it.** GNN-GI computes exactly this; GNN-LRP replaces each
  differentiation step by an LRP step.

A walk carries more than its bag. Fig. 2 of Schnake et al. (2022,
[p. 3](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=3))
shows one bag holding two walks that use the same connections in a different
order, and the paper notes that a walk therefore says in which layer two
nodes exchanged information, which the bag does not (p. 3). The worked
example below shows such a bag.

**Do not confuse a walk with a path.** A walk may visit a node more than once
(T50 → W32 → T50 is a walk) or stay where it is (W32 → W32 → W32). With $n$
nodes and $T$ layers there are $n^{T+1}$ possible walks, 27 here. A walk that
crosses a missing connection contains a zero entry and carries nothing. (For
why a walk's arrows are not later engine cycles, see the
[GNN-LRP tutorial](../learn/11_gnn_lrp.md).)

### Worked example: scaling the graph, one Taylor term, and one bag with two walks

First, Step 3: the network adds no constants, so multiplying every entry of
Λ by $s$ must multiply the score by $s^2$.

<details class="process-example">
<summary>Example in Python</summary>

```python
scales = [2.0, 1.0, 0.5, 0.1]
homogeneity = pd.DataFrame({
    "Graph entries multiplied by": scales,
    "Score": [forward(s * lam, s * lam)[4] for s in scales],
    "Multiplier squared × original score": [s ** 2 * score for s in scales],
})
```

</details>

<!-- sample: homogeneity -->
| Graph entries multiplied by | Score | Multiplier squared × original score |
| --- | --- | --- |
| 2.000000 | 1.262779 | 1.262779 |
| 1.000000 | 0.315695 | 0.315695 |
| 0.500000 | 0.078924 | 0.078924 |
| 0.100000 | 0.003157 | 0.003157 |

As $s$ falls towards 0 the score falls towards 0 — the reference point of
Step 4.

Second, Steps 4 and 5 for one walk, T30 → T50 → W32. Its layer-1 connection
is "T50 receives from T30" and its layer-2 connection "W32 receives from
T50". Giving each layer its own copy of the graph, nudge the first entry in
the layer-1 copy only and the second in the layer-2 copy only, by $h$ = 0.001
each way, and combine the four scores into the mixed second derivative with
the four-point difference
(f(+h, +h) − f(+h, −h) − f(−h, +h) + f(−h, −h)) ÷ (4h²). The nudges are too
small to flip any switch, so on this piece the score is linear in each entry
and the difference is exact up to rounding.

<details class="process-example">
<summary>Example in Python</summary>

```python
start, middle, end = 0, 1, 2      # the walk T30 → T50 → W32
nudge = 1e-3

def nudged(a, b):
    """The score after nudging the walk's layer-1 entry by a and its layer-2 entry by b."""
    first, second = lam.copy(), lam.copy()
    first[middle, start] += a     # T50 receives from T30, in layer 1 only
    second[end, middle] += b      # W32 receives from T50, in layer 2 only
    return forward(first, second)[4]

def product_along(a, b, c):
    """The walk a → b → c as one product: reading, entries, weights and switches."""
    return sum(x[a] * lam[b, a] * W1[0, j] * (h1[b, j] > 0)
               * lam[c, b] * W2[j, 0] * (z2[c] > 0) * u[c] * (score > 0)
               for j in range(2))

mixed = (nudged(nudge, nudge) - nudged(nudge, -nudge)
         - nudged(-nudge, nudge) + nudged(-nudge, -nudge)) / (4 * nudge ** 2)
taylor_table = pd.DataFrame({
    "Quantity": [
        "Second derivative, one entry from each layer",
        "Layer-1 entry: T50 receives from T30",
        "Layer-2 entry: W32 receives from T50",
        "Taylor term: the three multiplied",
        "Product along the walk",
    ],
    "Value": [mixed, lam[middle, start], lam[end, middle],
              mixed * lam[middle, start] * lam[end, middle], product_along(start, middle, end)],
})
```

</details>

<!-- sample: taylor_table -->
| Quantity | Value |
| --- | --- |
| Second derivative, one entry from each layer | -0.046632 |
| Layer-1 entry: T50 receives from T30 | 0.333333 |
| Layer-2 entry: W32 receives from T50 | 0.500000 |
| Taylor term: the three multiplied | -0.007772 |
| Product along the walk | -0.007772 |

The mixed derivative is −0.046632, which is T30's own reading: along this
walk every other multiplier is 1 — the weight 1 into T50's above-average
part, whose switch is on; the weight 1 out of that part at layer 2; W32's
switch, which is on; and the last step's weight 1. The Taylor term,
−0.046632 × 1/3 × 1/2 = −0.007772, is therefore the product of everything
along the walk: the starting reading, each graph entry, each weight and each
switch. No other walk uses this bag: in the opposite order the node reached
at layer 1, W32, would have to be the node sending at layer 2, T30. So the
bag's term and the walk's share are the same number.

Third, a bag that holds two walks: "W32 receives from T50" and "T50 receives
from W32". Used in that order it is the walk T50 → W32 → T50; used in the
other order, W32 → T50 → W32. Eq. 6 works on the one graph both layers share,
so this time both entries are nudged in that shared graph, in both layers at
once:

<details class="process-example">
<summary>Example in Python</summary>

```python
def nudged_shared(a, b):
    """The score after changing two entries of the graph both layers share."""
    changed = lam.copy()
    changed[2, 1] += a    # W32 receives from T50
    changed[1, 2] += b    # T50 receives from W32
    return forward(changed, changed)[4]

bag_mixed = (nudged_shared(nudge, nudge) - nudged_shared(nudge, -nudge)
             - nudged_shared(-nudge, nudge) + nudged_shared(-nudge, -nudge)) / (4 * nudge ** 2)
two_walks = [product_along(1, 2, 1), product_along(2, 1, 2)]
bag_table = pd.DataFrame({
    "Quantity": [
        "Second derivative, both entries in the shared graph",
        "Entry: W32 receives from T50",
        "Entry: T50 receives from W32",
        "Bag term (Eq. 6)",
        "Walk T50 → W32 → T50",
        "Walk W32 → T50 → W32",
        "The two walks added (Proposition 2)",
    ],
    "Value": [bag_mixed, lam[2, 1], lam[1, 2], bag_mixed * lam[2, 1] * lam[1, 2],
              *two_walks, sum(two_walks)],
})
```

</details>

<!-- sample: bag_table -->
| Quantity | Value |
| --- | --- |
| Second derivative, both entries in the shared graph | 0.573742 |
| Entry: W32 receives from T50 | 0.500000 |
| Entry: T50 receives from W32 | 0.333333 |
| Bag term (Eq. 6) | 0.095624 |
| Walk T50 → W32 → T50 | -0.131796 |
| Walk W32 → T50 → W32 | 0.227420 |
| The two walks added (Proposition 2) | 0.095624 |

The bag's term, 0.095624, is the sum of a negative walk that starts at T50
and a positive walk that starts at W32. A bag-level explanation would report
the one positive number; the walks show that T50's below-average reading,
sent to W32 in the first layer and back in the second, pulled the score down,
while W32's reading, sent to T50 in the first layer and back in the second,
pushed it up.

**The key idea is** that a network with no added constants computes, for a
fixed state of its switches, a sum of products, each taking one graph entry
per layer along a chain of nodes; a walk's share is one of those products,
and the shares of all walks add up to the output with nothing left over.

## GNN-GI: the gamma = 0 special case

The first algorithm of Schnake et al. (2022,
[Section 3.1](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=3))
computes Eq. 7 as written. For a walk ending (…, K, L, M) it starts at the
last connection with $R_{LM} = \frac{\partial f}{\partial\lambda^\star_{LM}}
\lambda^\star_{LM}$, then takes $R_{KLM} = \frac{\partial
R_{LM}}{\partial\lambda^\star_{KL}} \lambda^\star_{KL}$, and continues down to
the first layer ([p. 4](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=4)).
Each step is a gradient times an input, so the paper names it GNN-GI, a
nested gradient × input. In a switch network it is the product along the
walk that the worked example computed.

Schnake et al. (2022, p. 4) state two limits of GNN-GI. When the network is not
positively homogeneous — for instance when its neurons add constants — the
walk shares no longer add up to the output, and part of the output is
credited to nothing. And in deep networks the gradient GNN-GI relies on
becomes noisy and less reliable, an effect the paper calls shattering,
citing its references [34] and [35]. In Section 4 the paper describes GNN-GI
as the special case of GNN-LRP with γ = 0
([item (b), p. 6](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=6)).

**Do not confuse GNN-GI with the gradient map of Pope et al.** Both use
derivatives. GNN-GI multiplies derivatives by graph entries along a walk and
gives one signed share per walk, and the shares add up to the output. Pope et
al.'s (2019, Eq. 12) map takes the size of the positive part of a gradient
with respect to a node's features and gives one score per node, never negative and with no
requirement to add up to anything.

## GNN-LRP: one LRP step for each connection of the walk

### The nested procedure

Schnake et al. (2022, [Section 3.2](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=4))
replace each gradient × input step of Eq. 7 by an LRP step (Eq. 8): the
relevance arriving at the walk's last node is handed by LRP to the
connection before it, the result is handed to the connection before that,
and so on down to the first layer. LRP works on neurons, not nodes, so the
relevance arriving at a node is a collection, one per neuron of that node,
and each connection between two nodes is made of many smaller ones, one for
every pair of their neurons (Fig. 3).

### The rule for a graph convolution (Table 1, Eq. 9)

$$R_{jKL\ldots} = \sum_{k\in K} \frac{\lambda_{JK}\, h_j\, w^{\wedge}_{jk}}{\sum_{J'}\sum_{j'\in J'} \lambda_{J'K}\, h_{j'}\, w^{\wedge}_{j'k}}\; R_{kL\ldots}, \qquad w^{\wedge} = w + \gamma\,\rho(w)$$

(Schnake et al., 2022, [Table 1, Eq. 9, p. 5](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=5);
the paper reuses J as the running index of the lower sum, written J′ here to
keep the two apart.)

- **Symbols.** J is a node at layer $t - 1$, K the walk's next node at layer
  $t$, and L … the rest of the walk. $j$ is a neuron of node J and $k$ a
  neuron of node K. $h_j$ is $j$'s activation, $w_{jk}$ the entry of $W_t$
  joining $j$ to $k$, $\rho(w) = \max(0, w)$, and γ the rule's setting.
  $R_{kL\ldots}$ is the relevance held by neuron $k$ for the walks that
  continue through L …, and $R_{jKL\ldots}$ the part handed to neuron $j$ for
  the walks that go J → K → L ….
- **What it computes.** The overview's gamma rule (Montavon et al., 2019,
  Section 10.2.1) applied at a graph layer, with three changes: each
  contribution carries the graph entry $\lambda_{JK}$ as a factor; the denominator has no constant input, because
  Table 1's combine step adds none; and the result is added only over the
  neurons $k$ of the walk's next node K, not over every upper neuron.
  [Appendix C](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=17)
  gives the reason for the last change: the connections $\lambda_{JK}$ reach
  only node K, so they receive relevance from no other node.
- **Why.** Keeping K apart is what keeps the walk's identity. Adding over K
  would give back node-level LRP.
- **Example.** The table below.
- **What uses it.** At the first layer the walk's share is the sum over the
  starting node's neurons; with γ = 0 at every layer the result is GNN-GI
  (Schnake et al., 2022, Section 4, item (b)).

In this rule for a graph convolution, Table 1 of Schnake et al. (2022)
applies γ to the weights $w_{jk}$ only. The same table gives rules
for graph isomorphism networks (Eq. 10), whose combine step is a small
multi-layer network applied to each node's aggregated row, and whose rule
writes no γ of its own; and for spectral networks (Eq. 11), whose layers use
several graph tables at once, and whose rule applies γ to the product
$\lambda^s_{JK} w^s_{jk}$ of a graph entry and a weight.

Here is the rule at T50's layer-2 node with γ = 1. Three parts arrive there:
T30's below-average part through the weight −0.5, and the above-average parts
of T50 and W32 through the weight 1.

<details class="process-example">
<summary>Example in Python</summary>

```python
gamma = 1.0
counted = W2[:, 0] + gamma * np.maximum(W2[:, 0], 0.0)    # w + gamma * max(0, w)
active = [(0, 1), (1, 0), (2, 0)]                         # (sensor, part) feeding T50
contribution = np.array([lam[1, J] * h1[J, j] * W2[j, 0] for J, j in active])
weighted = np.array([lam[1, J] * h1[J, j] * counted[j] for J, j in active])
gamma_split = pd.DataFrame({
    "From": ["T30, below-average part", "T50, above-average part",
             "W32, above-average part", "Total"],
    "Contribution": [*contribution, contribution.sum()],
    "Counted with gamma 1": [*weighted, weighted.sum()],
    "Share with gamma 1": [*(weighted / weighted.sum() * h2[1]), h2[1]],
})
```

</details>

<!-- sample: gamma_split -->
| From | Contribution | Counted with gamma 1 | Share with gamma 1 |
| --- | --- | --- | --- |
| T30, below-average part | -0.069784 | -0.069784 | -0.024687 |
| T50, above-average part | 0.058568 | 0.117136 | 0.041438 |
| W32, above-average part | 0.095624 | 0.191247 | 0.067656 |
| Total | 0.084407 | 0.238599 | 0.084407 |

With γ = 0 the shares are the contributions themselves, because T50's
relevance equals its total, 0.084407. With γ = 1 the two contributions that
came through the positive weight are counted twice and the one through −0.5
is not, giving the rule-weighted total 0.238599; each share is its counted
value divided by 0.238599 and multiplied by 0.084407. The shares still add
up to 0.084407. The negative share shrinks from −0.069784 to −0.024687, and
the positive ones shrink too (0.058568 to 0.041438): less of the total is
now made of large shares cancelling each other, which is the behaviour the
overview describes (Montavon et al., 2019, Section 10.2.1). At W32's layer-2
node both arriving parts come through the weight 1, so both are doubled and their fractions do not move: γ has no
effect there.

### Why each step may be used: the induction of Appendix C

GNN-LRP has no closed formula like Eq. 6 (Schnake et al., 2022,
[p. 4](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=4)). The
paper justifies it with deep Taylor decomposition — the view, from
Montavon et al. (2019, Section 10.2.3), that each LRP step is a first-order
Taylor expansion made at one neuron — through an **induction**: an argument
that, once it holds at one layer, holds at the next one down. Section 3.2 states
its three steps, and
[Appendix C](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=17)
works them out for the graph convolution:

1. **Assume a product form.** Neuron $k$'s relevance can be written
   $R_{kL\ldots} = h_k\, c_{kL\ldots}$, with $c$ approximately constant
   nearby — it hardly moves when the activations of lower layers move.
2. **Treat one neuron as a small layer whose inputs are the graph entries.**
   Appendix C copies each graph entry to every pair of neurons:
   $\lambda_{jk} = \lambda_{JK}$ for every neuron $j$ of J and $k$ of K.
   Holding $c$ constant, neuron $k$ is $h_k = \rho\big(\sum_J \sum_{j \in J}
   \lambda_{jk} v_{jk}\big)$ with $v_{jk} = h_j w_{jk}$: a weighted sum and a
   switch whose inputs are the entries $\lambda_{jk}$, never negative, and
   whose weights are the $v_{jk}$. LRP-γ on that small layer, added over the
   neurons of K, gives
   $$R_{jKL\ldots} = \sum_{k\in K} \frac{\lambda_{jk}\,\big(v_{jk} + \gamma \max(0, v_{jk})\big)}{\sum_{J'}\sum_{j'\in J'} \lambda_{j'k}\,\big(v_{j'k} + \gamma \max(0, v_{j'k})\big)}\; R_{kL\ldots}$$
3. **Check the product form one layer down.** Appendix C's Eq. 15 writes the
   result as $R_{jKL\ldots} = h_j\, c_{jKL\ldots}$, where $c_{jKL\ldots}$
   depends on lower activations only through nested sums, or through
   $c_{kL\ldots}$, already assumed nearly constant. So the assumption of
   step 1 holds one layer down, and the argument repeats to the first layer.

The copies of step 2 are needed. Schnake et al. (2022, Appendix C) note
that written per node instead, with $\lambda_{JK}$ multiplying a whole node's
sum, LRP-γ gives a rule whose result cannot be put back in the form $h_j c_j$, and the argument
stops at that layer
([p. 18](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=18)).

Going from step 2 to Table 1's rule uses one more fact: $v_{jk} + \gamma
\max(0, v_{jk}) = h_j\,(w_{jk} + \gamma \max(0, w_{jk}))$ **because $h_j \ge
0$** (Schnake et al., 2022, Appendix C, p. 17). Step 2's form favours a
contribution by its own sign; Table 1's form favours it by the sign of its weight. The two agree
whenever the activation feeding the layer is not negative, which inside a
switch network it never is. A first graph layer that receives rescaled
readings is different: a reading below its training average is negative.

How much that matters depends on how many numbers each node hands the first
layer. In the demonstration network each sensor hands it one reading, so
every contribution into a first-layer neuron passes through that neuron's
single weight. Here is T50's above-average part in layer 1, dividing the
relevance it holds under LRP-0 (0.146420, from the node-by-node table) as
each form would with γ = 1:

<details class="process-example">
<summary>Example in Python</summary>

```python
def divide(counted_parts, relevance):
    """Shares of one neuron's relevance, in proportion to the counted contributions."""
    return counted_parts / counted_parts.sum() * relevance

held = at_layer1[1, 0]                                     # T50's above-average part
first_contribution = lam[1] * x * W1[0, 0]                 # what each reading sends to it
by_weight = lam[1] * x * (W1[0, 0] + gamma * max(W1[0, 0], 0.0))      # Table 1's form
by_sign = first_contribution + gamma * np.maximum(first_contribution, 0.0)   # step 2's form
first_layer_gamma = pd.DataFrame({
    "From": names,
    "Contribution": first_contribution,
    "Share at γ = 0": divide(first_contribution, held),
    "Share, Table 1's form": divide(by_weight, held),
    "Share, step 2's form": divide(by_sign, held),
})
np.testing.assert_allclose(divide(by_weight, held), divide(first_contribution, held), atol=1e-15)
```

</details>

<!-- sample: first_layer_gamma -->
| From | Contribution | Share at γ = 0 | Share, Table 1's form | Share, step 2's form |
| --- | --- | --- | --- | --- |
| T30 | -0.015544 | -0.012953 | -0.012953 | -0.003610 |
| T50 | -0.263593 | -0.219660 | -0.219660 | -0.061209 |
| W32 | 0.454840 | 0.379033 | 0.379033 | 0.211238 |

Table 1's form doubles all three contributions, because all three come
through the same weight +1, and the doubling cancels in the division: its
shares are the γ = 0 shares. Step 2's form doubles only W32's positive
contribution and shrinks the two negative shares. With one number per node,
then, Table 1's γ changes nothing at a first layer.

The difference appears when a node hands the first layer several numbers
through weights of both signs. To see it, give each sensor its two newest
readings, cycles 87 and 88, and give T50 a demonstration first-layer neuron
with weight −1 on the older reading and +1 on the newer one, so that it
measures how much the averaged readings rose over the last cycle. Its total,
0.563239, is positive, so its switch is on; take its relevance to be that
total, so the γ = 0 shares are the contributions themselves:

<details class="process-example">
<summary>Example in Python</summary>

```python
recent = prepared.test.features[row, chosen, -2:].astype(np.float64)   # cycles 87 and 88
rise = np.array([-1.0, 1.0])                   # weights: newer reading minus older one
sent = lam[1][:, None] * recent * rise         # what each (sensor, cycle) sends to T50
rise_total = sent.sum()
by_weight_2 = lam[1][:, None] * recent * (rise + gamma * np.maximum(rise, 0.0))
by_sign_2 = sent + gamma * np.maximum(sent, 0.0)
two_numbers = pd.DataFrame({
    "From": [name for name in names for _ in range(2)],
    "Cycle": [end_cycle - 1, end_cycle] * 3,
    "Reading": recent.ravel(),
    "Weight": np.tile(rise, 3),
    "Contribution": sent.ravel(),
    "Share, Table 1's form": divide(by_weight_2, rise_total).ravel(),
    "Share, step 2's form": divide(by_sign_2, rise_total).ravel(),
})
assert rise_total > 0
```

</details>

<!-- sample: two_numbers -->
| From | Cycle | Reading | Weight | Contribution | Share, Table 1's form | Share, step 2's form |
| --- | --- | --- | --- | --- | --- | --- |
| T30 | 87 | -0.878830 | -1.000000 | 0.292943 | 0.223288 | 0.204280 |
| T30 | 88 | -0.046632 | 1.000000 | -0.015544 | -0.023696 | -0.005420 |
| T50 | 87 | -0.913130 | -1.000000 | 0.304377 | 0.232003 | 0.212253 |
| T50 | 88 | -0.790778 | 1.000000 | -0.263593 | -0.401833 | -0.091906 |
| W32 | 87 | 0.629353 | -1.000000 | -0.209784 | -0.159902 | -0.073145 |
| W32 | 88 | 1.364520 | 1.000000 | 0.454840 | 0.693379 | 0.317177 |

Under Table 1's form, T50's newer reading — below average, arriving through
the positive weight — is counted twice although its contribution is
negative, and its share grows from −0.263593 to −0.401833: the rule enlarges
a negative share, the opposite of what the overview says γ is for
(Montavon et al., 2019, Section 10.2.1). The two
older readings, also below average but arriving through the negative weight,
make positive contributions that Table 1's form does not count up. Step 2's
form, whose inputs are the graph entries and so never negative whatever the
readings, shrinks every negative share instead.

This project sets γ = 0 at its first graph layer (assumption A-29), where
both forms give the γ = 0 shares. With one number per sensor, as
in the network on this page, Table 1's γ would change nothing at that layer
anyway; the choice matters only where a sensor hands the first graph layer
several numbers. The authors themselves fell back to γ = 0 for SchNet, their
network for molecules, whose layers lack the structure of a switch, then a
weighted sum, then a switch that LRP-γ needs (Schnake et al., 2022,
[Section 4.1, p. 8](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=8)).

**The key idea is** that if a neuron's relevance is its activation times a
factor that hardly moves, the gamma rule hands down relevances of the same
form one layer lower, so the same rule may be applied all the way to the
first layer; Table 1's compact form of it additionally needs the activations
entering each layer to be never negative.

### Worked example: every walk, at gamma 0 and gamma 1

Now the whole split, by hand. At layer 2, each walk's last node hands its
relevance to the (node, part) pairs that fed it, by Eq. 9; at layer 1, with
γ = 0, each part hands its share to the readings in proportion to their
contributions. The code also checks that, at γ = 0, every one of the 27
walks equals the product along it, and that each connection's score in the
table of the previous section is the sum of the walks that use it, once per
use.

<details class="process-example">
<summary>Example in Python</summary>

```python
def split_into_walks(gamma):
    """Divide the score among all 27 walks: gamma at graph layer 2, 0 at layer 1."""
    shares = np.zeros((3, 3, 3))                      # [start, middle, end]
    counted = W2[:, 0] + gamma * np.maximum(W2[:, 0], 0.0)
    for end in range(3):
        if z2[end] <= 0:
            continue                                  # switch off: nothing passes on
        parts = lam[end, :, None] * h1 * counted      # what each (middle, part) sends
        to_middle = parts / parts.sum() * at_layer2[end]
        for middle in range(3):
            for j in range(2):
                if h1[middle, j] > 0:
                    sent = lam[middle] * x * W1[0, j]  # gamma 0 at layer 1
                    shares[:, middle, end] += sent / sent.sum() * to_middle[middle, j]
    return shares

gi = split_into_walks(0.0)
lrp = split_into_walks(1.0)
every_walk = list(itertools.product(range(3), repeat=3))
for a, b, c in every_walk:
    assert abs(gi[a, b, c] - product_along(a, b, c)) < 1e-12
uses = [sum(gi[a, b, c] * (((b, a) == (K, J)) + ((c, b) == (K, J))) for a, b, c in every_walk)
        for K, J in connections]
np.testing.assert_allclose(edge_scores["Slope × entry"], uses, atol=1e-8)
walk_table = pd.DataFrame([
    {"Walk": f"{names[a]} → {names[b]} → {names[c]}",
     "Layer-1 entry": lam[b, a], "Layer-2 entry": lam[c, b],
     "GNN-GI share": gi[a, b, c], "GNN-LRP share": lrp[a, b, c]}
    for a, b, c in every_walk
    if lam[b, a] > 0 and lam[c, b] > 0
])
start_totals = pd.DataFrame({
    "Starting sensor": names + ["All"],
    "Node-level LRP": [*at_readings, at_readings.sum()],
    "GNN-GI walks from it": [*gi.sum(axis=(1, 2)), gi.sum()],
    "GNN-LRP walks from it": [*lrp.sum(axis=(1, 2)), lrp.sum()],
})
assert len(walk_table) == 17
np.testing.assert_allclose(lrp.sum(axis=(1, 2)), node_level(1.0)[2], atol=1e-12)
```

</details>

Of the 27 possible walks, these 17 use only existing connections; the other
10 cross the missing T30–W32 connection and are zero. The GNN-LRP column
uses γ = 1 at layer 2 and γ = 0 at layer 1:

<!-- sample: walk_table -->
| Walk | Layer-1 entry | Layer-2 entry | GNN-GI share | GNN-LRP share |
| --- | --- | --- | --- | --- |
| T30 → T30 → T30 | 0.500000 | 0.500000 | 0.000000 | 0.000000 |
| T30 → T30 → T50 | 0.500000 | 0.333333 | -0.003886 | -0.001375 |
| T30 → T50 → T30 | 0.333333 | 0.500000 | 0.000000 | 0.000000 |
| T30 → T50 → T50 | 0.333333 | 0.333333 | -0.005181 | -0.003666 |
| T30 → T50 → W32 | 0.333333 | 0.500000 | -0.007772 | -0.007772 |
| T50 → T30 → T30 | 0.500000 | 0.500000 | 0.000000 | 0.000000 |
| T50 → T30 → T50 | 0.500000 | 0.333333 | -0.065898 | -0.023312 |
| T50 → T50 → T30 | 0.333333 | 0.500000 | 0.000000 | 0.000000 |
| T50 → T50 → T50 | 0.333333 | 0.333333 | -0.087864 | -0.062166 |
| T50 → T50 → W32 | 0.333333 | 0.500000 | -0.131796 | -0.131796 |
| T50 → W32 → T50 | 0.500000 | 0.333333 | -0.131796 | -0.093249 |
| T50 → W32 → W32 | 0.500000 | 0.500000 | -0.197694 | -0.197694 |
| W32 → T50 → T30 | 0.333333 | 0.500000 | 0.000000 | 0.000000 |
| W32 → T50 → T50 | 0.333333 | 0.333333 | 0.151613 | 0.107270 |
| W32 → T50 → W32 | 0.333333 | 0.500000 | 0.227420 | 0.227420 |
| W32 → W32 → T50 | 0.500000 | 0.333333 | 0.227420 | 0.160905 |
| W32 → W32 → W32 | 0.500000 | 0.500000 | 0.341130 | 0.341130 |

What the table shows:

- **The five zero rows among them all end at T30**, whose layer-2 switch is
  off.
- **Walks ending at W32 are the same in both columns**, because γ has no
  effect at W32's layer-2 node. **Walks ending at T50 change**: T50 → T30 →
  T50 goes from −0.065898 to −0.023312, W32 → W32 → T50 from 0.227420 to
  0.160905 — the effect of the gamma table above, carried down to the
  readings.
- **The two-step walk T30 → T50 → W32** is the only way T30's reading reaches
  W32: two layers reach two connections away. Its share is the Taylor term
  computed above.
- **W32 → T50 → W32 and W32 → W32 → T50 have the same GNN-GI share,**
  0.227420, because both multiply W32's reading by 1/3 and 1/2 and by the
  same weights. Under γ = 1 they differ, 0.227420 against 0.160905, because
  they end at different nodes.

Added up by the node each walk starts from:

<!-- sample: start_totals -->
| Starting sensor | Node-level LRP | GNN-GI walks from it | GNN-LRP walks from it |
| --- | --- | --- | --- |
| T30 | -0.016839 | -0.016839 | -0.012813 |
| T50 | -0.615049 | -0.615049 | -0.508218 |
| W32 | 0.947583 | 0.947583 | 0.836726 |
| All | 0.315695 | 0.315695 | 0.315695 |

Both columns of walks add up to the score, 0.315695. The GNN-GI walks,
pooled by their first node, give back the node-level LRP of the earlier
table exactly; the code also checks that the GNN-LRP walks pooled the same
way equal node-level LRP with the same γ. That is the statement of
Schnake et al. (2022, [pp. 6–7](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=7))
that pooling walk explanations by their first node gives node explanations
equivalent to those of Pope et al. (2019),
in the form their benchmark used them: Pope et al.'s attribution onto the
initial node features, computed by gradient × input or by LRP. Walks lose
nothing that node shares hold, and add the route.

**The key idea is** that GNN-LRP runs the same backward division as
node-level LRP, but never adds together what arrives from different nodes,
so every share stays labelled with the full list of nodes it passed through.

## Computing the walks in practice

Table 1 defines the shares; Schnake et al. (2022,
[Section 3.2.1](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=5))
show how to obtain them with **automatic differentiation**, the library
feature that computes derivatives of any calculation it has recorded. A
**forward pass** computes the output from the input, layer by layer, and
records each operation; a **backward pass** then sweeps from the output back
to the inputs and, by the chain rule (the derivative through a chain of steps
is the product of the steps' own derivatives), computes the derivative of the
output with respect to every recorded number at once.

Writing each relevance as an activation times a factor, $R_{jKL\ldots} = h_j
c_{jKL\ldots}$ and $R_{kL\ldots} = h_k c_{kL\ldots}$, Eq. 9 becomes
(Schnake et al., 2022, Eq. 12):

$$c_{jKL\ldots} = \sum_{k \in K} \lambda_{JK}\, w^{\wedge}_{jk}\, \frac{h_k}{p_k}\, c_{kL\ldots}, \qquad p_k = \sum_{J'}\sum_{j' \in J'} \lambda_{J'K}\, h_{j'}\, w^{\wedge}_{j'k}$$

- **Symbols.** $c$ is the factor that multiplies an activation to give its
  relevance; $p_k$ is neuron $k$'s rule-weighted total; the other symbols are
  those of Eq. 9.
- **What it computes.** Eq. 9 divided through by $h_j$.
- **Why.** $\lambda_{JK} w^{\wedge}_{jk}$ is the derivative of $p_k$ with
  respect to $h_j$, so the equation has the shape of a backward pass
  (Eq. 13). It differs from ordinary differentiation in three ways (p. 5):
  it differentiates $p_k$ instead of $h_k$, multiplies by the extra factor
  $h_k / p_k$, and does not add over the receiving nodes K.
- **Example.** At T50's layer-2 neuron with γ = 1, $c_k = R_k / h_k$ = 1,
  because T50's layer-2 relevance equals its output; $h_k / p_k$ = 0.353763;
  and for T50's above-average part $\lambda_{JK} w^{\wedge}_{jk}$ = 1/3 × 2.
  So $c_j$ = 1/3 × 2 × 0.353763 × 1 = 0.235842, and $h_j c_j$ = 0.175703 ×
  0.235842 = 0.041438, the share in the gamma table. The table below
  computes each of these.
- **What uses it.** The forward rewrite below makes automatic
  differentiation compute exactly this.

The **forward rewrite** changes how each graph layer is written, not what it
outputs (Schnake et al., 2022, p. 5): $P_t \leftarrow Z_t W^{\wedge}_t$,
then $Q_t \leftarrow P_t
\odot [\rho(Z_t W_t) \oslash P_t]_{\text{cst.}}$, then $H_t \leftarrow Q_t
\odot M_K + [Q_t]_{\text{cst.}} \odot (1 - M_K)$. Here ⊙ and ⊘ multiply and
divide number by number, $[\cdot]_{\text{cst.}}$ marks a quantity that
automatic differentiation treats as a constant, and $M_K$ is a mask, 1 for
the neurons of the walk's node K at that layer and 0 elsewhere.

- $Q_t$ equals $\rho(Z_t W_t)$ in value, but because the factor in brackets
  is held fixed, its derivative with respect to an incoming activation is
  $\lambda\, w^{\wedge}$ times that factor, $h / p$ — the rule's fraction of
  Eq. 13, not the ordinary derivative.
- $H_t$ equals $Q_t$ in value, but every node other than K is replaced by a
  copy held fixed, so no derivative passes through it. With one mask per
  layer, chosen along the walk, the only route by which the output depends
  on the starting numbers is the walk itself.
- One backward pass then gives the walk's share as
  $R_\mathcal{W} = \langle \text{AUTOGRAD}(f, H_{0,I}), H_{0,I} \rangle$
  (p. 5): the derivative of the output with respect to the starting node I's
  numbers, multiplied by those numbers number by number and added
  ($\langle a, b \rangle = \sum_i a_i b_i$).

<details class="process-example">
<summary>Example in Python</summary>

```python
w_hat = W2[0, 0] + gamma * max(W2[0, 0], 0.0)    # the weight 1, counted with gamma 1
p = weighted.sum()
c_k = at_layer2[1] / h2[1]                        # T50's layer-2 relevance ÷ its output
c_j = lam[1, 1] * w_hat * (h2[1] / p) * c_k       # Eq. 12, for T50's above-average part
rewrite_table = pd.DataFrame({
    "Quantity": ["Output of T50's layer-2 switch, h", "Rule-weighted total, p",
                 "Held factor, h ÷ p", "Rewritten output, p × (h ÷ p)",
                 "Factor above, c = relevance ÷ output",
                 "Factor for T50's above-average part (Eq. 12)",
                 "Its share: activation × factor"],
    "Value": [h2[1], p, h2[1] / p, p * (h2[1] / p), c_k, c_j, h1[1, 0] * c_j],
})
along_connections = int(np.linalg.matrix_power((full > 0).astype(int), 3).sum())
assert along_connections == 5178
```

</details>

<!-- sample: rewrite_table -->
| Quantity | Value |
| --- | --- |
| Output of T50's layer-2 switch, h | 0.084407 |
| Rule-weighted total, p | 0.238599 |
| Held factor, h ÷ p | 0.353763 |
| Rewritten output, p × (h ÷ p) | 0.084407 |
| Factor above, c = relevance ÷ output | 1.000000 |
| Factor for T50's above-average part (Eq. 12) | 0.235842 |
| Its share: activation × factor | 0.041438 |

### The masked procedure, and the project's code

This block needs PyTorch. It runs the forward rewrite above on the
demonstration network, with one backward pass for each of the 27 walks, and
then explains the same network with the project's `walk_relevance`, which
explains the project's own network class, so the demonstration is rebuilt as
one, with the same multipliers. γ = 1 at layer 2 is a demonstration setting,
chosen to make its effect visible; the project's own setting (assumption
A-29) is described on the [GNN-LRP tutorial](../learn/11_gnn_lrp.md).

<details class="process-example">
<summary>Example in Python</summary>

```python
import torch
from thesis_gnn_rul_xai.models import ESTGNN
from thesis_gnn_rul_xai.explain import walk_relevance, conservation_error, subgraph_scores

def masked_walk(walk, gammas):
    """One walk's share from the forward rewrite of Section 3.2.1 and one backward pass."""
    start = torch.tensor(x[:, None], requires_grad=True)      # H0: 3 nodes x 1 number
    hidden = start
    for t, (weight, layer_gamma) in enumerate(zip([W1, W2], gammas)):
        weight = torch.tensor(weight)
        mixed = torch.tensor(lam) @ hidden                     # Z: the aggregate step
        p_t = mixed @ (weight + layer_gamma * weight.clamp(min=0))
        h_t = torch.relu(mixed @ weight)                       # what the layer outputs
        safe = torch.where(h_t > 0, p_t, torch.ones_like(p_t))
        held_factor = torch.where(h_t > 0, h_t / safe, torch.zeros_like(h_t)).detach()
        q_t = p_t * held_factor                                # Q: equal to h_t in value
        mask = torch.zeros_like(q_t)
        mask[walk[t + 1]] = 1.0                                # M_K: the walk's node here
        hidden = q_t * mask + q_t.detach() * (1 - mask)
    torch.relu(torch.tensor(u) @ hidden[:, 0]).backward()     # last step, then backward pass
    return float(start.grad[walk[0], 0]) * x[walk[0]]         # derivative times reading

model = ESTGNN(normalize_adjacency(path, method="row"), window_size=1,
               gcn_hidden=(2, 1), fc_hidden=1)
with torch.no_grad():
    model.graph_layers[0].weight.copy_(torch.tensor(W1))
    model.graph_layers[1].weight.copy_(torch.tensor(W2))
    model.fc1.weight.copy_(torch.tensor(u[None]))
    model.fc1.bias.zero_()
    model.fc2.weight.fill_(1.0)
    model.fc2.bias.zero_()
model.eval()

by_hand = {(0.0, 0.0): gi, (0.0, 1.0): lrp}
rows = []
for gammas, mine in by_hand.items():
    masked = np.array([masked_walk(walk, gammas) for walk in every_walk]).reshape(3, 3, 3)
    project = walk_relevance(model, window, gamma=list(gammas))
    assert conservation_error(project) < 1e-12
    for method, walks in [("Masked backward passes", masked),
                          ("Project's walk_relevance", project.relevance[0])]:
        rows.append({"Computed by": method,
                     "Gamma per graph layer": ", ".join(f"{g:g}" for g in gammas),
                     "Largest gap from the hand split": float(np.abs(walks - mine).max()),
                     "Walks added up": float(walks.sum())})
project_check = pd.DataFrame(rows)

masks = np.array([[index in group for index in range(3)]
                  for size in range(4) for group in itertools.combinations(range(3), size)])
np.testing.assert_allclose(
    subgraph_scores(model, window, masks),
    [forward(lam * m[:, None] * m[None, :], lam * m[:, None] * m[None, :])[4] for m in masks],
    atol=1e-6,
)
```

</details>

<!-- sample: project_check -->
| Computed by | Gamma per graph layer | Largest gap from the hand split | Walks added up |
| --- | --- | --- | --- |
| Masked backward passes | 0, 0 | 0.000000 | 0.315695 |
| Project's walk_relevance | 0, 0 | 0.000000 | 0.315695 |
| Masked backward passes | 0, 1 | 0.000000 | 0.315695 |
| Project's walk_relevance | 0, 1 | 0.000000 | 0.315695 |

The paper's masked procedure, the project's code and the hand split agree on
every one of the 27 walks, at both settings. The block also checks that the
project's `subgraph_scores`, which reruns the network with sensors taken out,
gives the same scores as the NumPy network used in the node-flipping example
below.

The masked procedure costs one forward and one backward pass for each walk
in the input graph (Schnake et al., 2022, p. 5): those along existing
connections, 17 of the 27 here, and 5,178 of the 14 × 14 × 14 × 14 = 38,416
possible on the project's 14-sensor graph with three graph layers (Wen et al.,
2026, Table 4). It also
allows a variation: removing the mask at one layer adds together all walks
through that layer, ignoring which node they pass there (p. 5). To save time,
it suggests defining walks first between groups of nodes and expanding only
the most relevant, stopping a walk early once its relevance falls below a
threshold, and, for regular grids such as an image's pixels, computing many
walks in one pass (pp. 5–6; Appendix D). The project's `walk_relevance` does not run one pass
per walk: it multiplies arrays so that all walks come out at once, and a
test, `test_every_walk_matches_the_authors_masked_gradient`, checks 30 of the
5,178 against the masked procedure.

**The key idea is** that multiplying each layer's output by a factor that
differentiation treats as fixed makes an ordinary backward pass compute the
rule's fractions instead of plain derivatives, and the masks leave open only
the one route that is the walk.

## How the authors tested the method

### What it was compared with

Schnake et al. (2022, [Section 4](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=6))
first train a graph isomorphism network — whose combine step is a small
two-layer network, 32 neurons wide at each node — with two graph layers, every node's
initial features set to 1, and the graph Λ = Ã/2, where Ã is the adjacency
table (1 where two nodes are joined, 0 elsewhere) with each node also joined
to itself. The network learns to tell two kinds of synthetic random graph
apart, reaching an accuracy above 95 %, and is explained with γ = 2 at the first
layer and γ = 1 at the second. Three methods are set beside GNN-LRP (p. 6):

- **(a) Pope et al.**, described as attributing the output to the nodes'
  initial features, with the attribution computed by gradient × input or by
  LRP.
- **(b) GNN-GI**, described as GNN-LRP with γ = 0.
- **(c) GNNExplainer**, which searches by optimisation for a mask $M =
  \sigma(R)$ over the connections that maximises the output $f(M \odot
  \Lambda)$, σ being the logistic sigmoid $1/(1 + e^{-r})$, which maps any
  number into the range 0 to 1; the explanation is $R$.

Comparing the four explanations of one graph
([Fig. 5](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=6)),
Schnake et al. (2022) report that Pope et al.'s method marks nodes without
showing whether they matter alone or through their neighbours; that GNNExplainer marks
roughly the regions GNN-LRP marks, but fewer of them and with less detail;
and that the methods built on gradient × input spread positive and negative
relevance over more of the graph than those built on LRP (pp. 6–7).

### Node flipping: adding and removing nodes

Schnake et al. (2022, [Section 4.1](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=7))
adapt pixel flipping, a check for image explanations, to graphs: nodes are
added or removed in the order an explanation suggests, and the network's
output is followed. If removing what the explanation calls relevant moves the
output a lot, and removing what it calls irrelevant moves it little, the
explanation describes the network faithfully.

To use an explanation for that, each method needs an estimate $R_\mathcal{G}$
of what a set of nodes $\mathcal{G}$ (a subgraph) is worth (Schnake et al.,
2022, p. 7):

- for a node-level explanation, $R_\mathcal{G} = \sum_{V \in \mathcal{G}}
  R_V$, the shares of the nodes in the set;
- for an edge-level explanation, the shares of the edges in the set;
- for bags or walks, $R_\mathcal{G} = \sum_{\mathcal{B} \in \mathcal{G}}
  R_\mathcal{B}$, where a bag counts only if all its connections lie in the
  set — for walks, only walks whose every node is in the set.

Schnake et al. (2022, p. 7) note that every estimate is 0 for the empty set
and equals the output for the full graph; that last part holds when the
shares add up to the output. Then two tasks:

$$\mathcal{V}^\star = \arg\max_{\mathcal{V} \notin \mathcal{G}} R_{\mathcal{G} \cup \{\mathcal{V}\}} \qquad\qquad \mathcal{V}^\star = \arg\min_{\mathcal{V} \in \mathcal{G}} \big|R_{\mathcal{G}_0} - R_{\mathcal{G} \setminus \{\mathcal{V}\}}\big|$$

- **Symbols.** $\mathcal{G}$ is the current set of nodes, $\mathcal{G}_0$
  the full original graph, $\mathcal{V}$ a candidate node, $\mathcal{V}^\star$
  the node chosen. $\mathcal{V} \notin \mathcal{G}$ means a node not yet in
  the set, $\mathcal{G} \cup \{\mathcal{V}\}$ the set with $\mathcal{V}$
  added, and $\mathcal{G} \setminus \{\mathcal{V}\}$ the set with
  $\mathcal{V}$ taken out. arg max picks the candidate that makes the
  expression largest, arg min the one that makes it smallest.
- **What they compute.** Eq. 14, the *activation task*: starting from no
  nodes, add the node whose addition the explanation rates highest. Eq. 15,
  the *pruning task*: starting from the full graph, remove the node whose
  removal the explanation says changes the output least.
- **Why.** Finding the best order exactly would mean trying every order. The
  paper's substitute is greedy: at each step it takes the single best next
  node by the explanation's estimate, without looking further ahead (p. 7).
  It notes that this can underrate walk-based methods, since nodes that
  matter only together are not found one at a time, and allows replacing a
  step at regular intervals by adding the node with the largest share on its
  own.
- **Example.** The table below.
- **What uses it.** After every step the network is run on the changed
  graph. The activation task plots the output $f(\Lambda)$; the pruning task
  plots its distance from the original, $|f(\Lambda) - f(\Lambda_0)|$
  ([Fig. 6](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=7)).
  The **area under the flipping curve**, AUFC, is the mean of the plotted
  values over the sequence, averaged over many graphs: higher is better for
  activation, lower for pruning.

Section 4.1 of Schnake et al. (2022) does not say how a network is run on a
graph with nodes taken out: whether a removed node's row and column of Λ are
set to zero, whether the remaining rows are rescaled, or what the removed node's own features
become. Fig. 6 only writes the output as $f(\Lambda)$ of the changed graph.
The expression $f(M \odot \Lambda)$ appears in Section 4 in the description
of GNNExplainer, item (c), not in Section 4.1. The project's procedure is its
own choice (assumption A-33), set out on the
[GNN-LRP tutorial](../learn/11_gnn_lrp.md#4--check-by-adding-and-removing-sensors);
the example below removes a sensor the same way, setting its row and column
of the small graph to zero and rescaling nothing.

**The key idea is** that an explanation which rates a set of nodes well
should predict what the network does when only those nodes remain, so the
network itself, rerun on each smaller graph, is the judge.

### Worked example: three ways to value a set of sensors

For every set of the three sensors: the node-level estimate (the GNN-LRP
shares of walks *starting* at a sensor in the set), the walk estimate (only
walks lying *wholly* inside the set), at γ = 0 and at γ = 1, and the
network's actual score with only those sensors connected.

<details class="process-example">
<summary>Example in Python</summary>

```python
starts = lrp.sum(axis=(1, 2))
subset_rows = []
for size in range(4):
    for group in itertools.combinations(range(3), size):
        present = np.array([index in group for index in range(3)])
        kept = lam * present[:, None] * present[None, :]   # absent rows and columns set to 0
        inside = np.ix_(present, present, present)
        subset_rows.append({
            "Sensors connected": ", ".join(names[i] for i in group) or "None",
            "Node-level estimate": float(starts[present].sum()),
            "Walk estimate, GNN-GI": float(gi[inside].sum()),
            "Walk estimate, GNN-LRP": float(lrp[inside].sum()),
            "Network's score": forward(kept, kept)[4],
        })
subset_table = pd.DataFrame(subset_rows)
```

</details>

<!-- sample: subset_table -->
| Sensors connected | Node-level estimate | Walk estimate, GNN-GI | Walk estimate, GNN-LRP | Network's score |
| --- | --- | --- | --- | --- |
| None | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| T30 | -0.012813 | 0.000000 | 0.000000 | 0.000000 |
| T50 | -0.508218 | -0.087864 | -0.062166 | 0.000000 |
| W32 | 0.836726 | 0.341130 | 0.341130 | 0.341130 |
| T30, T50 | -0.521031 | -0.162830 | -0.090519 | 0.000000 |
| T30, W32 | 0.823913 | 0.341130 | 0.341130 | 0.341130 |
| T50, W32 | 0.328507 | 0.398432 | 0.351820 | 0.398432 |
| T30, T50, W32 | 0.315695 | 0.315695 | 0.315695 | 0.315695 |

- **First and last rows.** Every estimate is 0 with no sensor and equals the
  score with all three, as Schnake et al. (2022, Section 4.1, p. 7) say. So
  is the network here, because
  it adds no constants.
- **Adding (Eq. 14).** Every estimate adds W32 first. For the second step
  the node-level estimate prefers T30 (0.823913 against 0.328507), because it
  counts W32's walks through T50 although T50 is not yet connected; both walk
  estimates prefer T50. The network scores 0.341130 with T30 added and
  0.398432 with T50 added.
- **Removing (Eq. 15).** From the full graph's 0.315695, the node-level
  estimate says removing T30 changes least (it leaves 0.328507); both walk
  estimates say removing T50 does (it leaves 0.341130). The network moves to
  0.398432 without T30 and to 0.341130 without T50.
- **At γ = 0 the walk estimate is the network wherever no switch changes
  state** — six of the eight rows. With every switch held, the network is the
  sum of one product per walk, and taking a sensor out deletes exactly the
  walks through it; every remaining walk is the same product as before.
  Without T30, for instance, T50's layer-1 total moves from 0.175703 to
  0.191247 but stays positive, and the GNN-GI walks inside {T50, W32} add up
  to the network's 0.398432.
- **At γ = 1 it is not, even where no switch changes.** The GNN-LRP estimate
  for T50 and W32 is 0.351820: the rule divided T50's layer-2 relevance in
  proportion to the rule-weighted total 0.238599, which included T30's
  −0.069784, so the walks ending at T50 carry shares fixed by the full graph.
- **Where a switch changes, neither estimate is the network.** With T50
  alone, T50's layer-1 total is 1/3 × (−0.790778) = −0.263593: its
  above-average part switches off and its layer-2 total turns negative, so
  the network gives 0, against −0.087864 and −0.062166 from the walks. The
  row with T30 and T50 fails the same way.

This is arithmetic on one demonstration, not evidence for either estimate.
The [GNN-LRP tutorial](../learn/11_gnn_lrp.md) runs the full adding and
removing procedure and its mean on the project's network.

### What the paper's tables report

Tables 2 and 3 of Schnake et al. (2022,
[p. 8](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=8))
give one AUFC per network and method — synthetic graphs with three kinds of
graph network, a sentiment classifier, SchNet for molecular energy and dipole
moment, and VGG-16 for images. For the synthetic graph convolution,
activation (higher is better): GNN-LRP 2.72, GNNExplainer 2.65, Pope et al.
with LRP 2.29, GNN-GI 1.86, random order 0.75; pruning (lower is better):
GNN-LRP 1.93, GNNExplainer 2.10, GNN-GI 2.20, random 3.39. GNN-LRP is marked
best in every row except the sentiment classifier's activation task, where
GNNExplainer is ahead (20.23 against 20.05); on SchNet, LRP was limited to
γ = 0, so gradient × input and LRP give one number each and share the best
mark (p. 8). The tables give no spread across graphs or training runs, so
how large a difference has to be to matter cannot be read from them. The
authors read the two tables as a consistent win for GNN-LRP across their
benchmark (p. 8).

## What the paper found with it

Schnake et al. (2022, [Section 5](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=8))
use GNN-LRP to examine three trained networks:

- **Sentiment** (Section 5.1, γ = 3). The graph of each sentence is its
  dependency tree, which joins each word to the word it grammatically
  depends on (Appendix F.2). Walks pick out word combinations, find a
  sarcastic sentence misread through local positive pairs (Fig. 8), and
  reveal a bias towards particular actors' names; over the whole corpus, the
  most positive and most negative walks are grouped by how many different
  words they contain, one to three
  ([Fig. 9](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=9)).
- **Quantum chemistry**
  ([Section 5.2](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=10)).
  SchNet's graph joins atoms and carries their distances. For the predicted
  energy of one molecule, paracetamol, walks that stay on one atom or cross
  one bond dominate (Fig. 10, left); over 1,000 molecules the average
  contribution per bond grows with bond order — single, double, triple —
  which the paper notes agrees with chemical intuition. For the dipole
  moment, which the network outputs as the length of a vector and so not
  linearly, the paper first replaces the output by a linear one, its
  projection on the predicted direction.
- **Images** ([Section 5.3](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=11)).
  VGG-16, an image classifier, is read as a graph network whose nodes are
  positions on a grid of pixels, with γ = 0.5 in block 3, halved in each
  later block, and γ = 0 in the top classifier. Walks show the parts of a
  teapot being assembled, and a prediction of the class *dumbbell* built
  partly from the arm holding it — a right answer reached through a cue that
  should not matter.

## What the method does not show

The general limits of a walk explanation — that a share is a route the
trained network used and not a cause in the engine, and that it says nothing
directly about what removing a sensor would do — are set out on the
[GNN-LRP tutorial](../learn/11_gnn_lrp.md#what-the-explanation-does-and-does-not-show)
and in the [explanation guide](../../src/thesis_gnn_rul_xai/explain/README.md).
Four more follow from the papers and from the example on this page:

- **A unique answer.** The split depends on the rule. In the example, every
  walk ending at T50 changes between γ = 0 and γ = 1 while the score they add
  up to, 0.315695, does not; even the totals by starting sensor move, T50's
  from −0.615049 to −0.508218. Eq. 4 of Bach et al. (2015, p. 4) shows that
  conservation can be met by a meaningless split.
- **Removal effects.** At γ > 0 a share is not what taking a node out would
  do, even where no switch changes (0.351820 against 0.398432 above); where a
  switch changes, no estimate is (−0.062166 against 0 for T50 alone). Node
  flipping is the separate check of that.
- **Exactness with added constants.** Proposition 1 needs a network without
  them; Schnake et al. (2022, Section 3.1) note that GNN-GI then stops adding
  up, and for their synthetic networks they forced every constant to be zero
  or negative during training, to avoid parts they could not explain
  ([Appendix F.1](../references/schnake_2022_gnn_lrp_relevant_walks.pdf#page=19)).
- **Anything beyond the explained input.** The switch states belong to that
  input; a neighbouring input can flip one and change many shares. The
  paper's patterns over a data set (Schnake et al., 2022, Figs. 9–11) are
  averages of explanations
  of single inputs — *local* explanations, each about one input — not an
  explanation of the network as a whole.

## Version 1 of the paper

The first arXiv version ([Schnake et al., 2020](../references/schnake_2020_gnn_lrp_arxiv_v1.pdf)),
titled *XAI for Graphs: Explaining Graph Neural Network Predictions by
Identifying Relevant Walks*, differs from version 3 where it matters here:

- It restricts the network to layers $H_t = \rho(\Lambda_t H_{t-1} W_t)$
  ([Eq. 3, p. 4](../references/schnake_2020_gnn_lrp_arxiv_v1.pdf#page=4)),
  and its Proposition 1 is the chain of gradients that version 3 calls
  Proposition 2.
- Its evaluation is an edge-removal test on the sentiment model, scored by
  the area under a perturbation curve ([p. 7](../references/schnake_2020_gnn_lrp_arxiv_v1.pdf#page=7));
  node flipping and Tables 2–3 do not exist yet.
- Its settings of γ differ: 5 for the sentiment model (p. 7), and 0.25 at
  every layer for its scale-free graphs, chosen after showing explanations at
  γ from 0 to 1: near 0 many positive and negative contributions mix; as γ
  grows the negative ones shrink but the positive ones spread to regions not
  typical of the class ([supplement, Appendix D](../references/schnake_2020_gnn_lrp_arxiv_v1.pdf#page=16)).

## What this project takes from the papers

The project's `walk_relevance` applies the graph-convolution rule of
Schnake et al. (2022, Table 1, Eq. 9) at its graph layers and LRP-0 at the two ordinary layers after them,
and divides the score before the sigmoid (assumption A-29); its
`node_flipping` runs Eqs. 14 and 15 (assumption A-33). Its networks add
constants in their last two layers (Wen et al., 2026, Eqs. 12 and 13) and
none in their graph layers (assumption A-12); Proposition 1 excludes such
constants, and their part of the score is reported beside the walks. How all of this is done on
the project's network, with which setting of γ and how a sensor is taken
out, is on the [GNN-LRP tutorial](../learn/11_gnn_lrp.md) and in the
[explanation guide](../../src/thesis_gnn_rul_xai/explain/README.md).

## References

- Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., & Samek,
  W. (2015). On pixel-wise explanations for non-linear classifier decisions by
  layer-wise relevance propagation. *PLOS ONE*, *10*(7), Article e0130140.
  https://doi.org/10.1371/journal.pone.0130140
  [Copy](../references/bach_2015_layerwise_relevance_propagation.pdf)
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with
  graph convolutional networks. *International Conference on Learning
  Representations (ICLR)*. https://arxiv.org/abs/1609.02907v4. The copy is
  arXiv version 4, the ICLR 2017 paper.
  [Copy](../references/kipf_2017_graph_convolutional_networks.pdf)
- Montavon, G., Binder, A., Lapuschkin, S., Samek, W., & Müller, K.-R.
  (2019). Layer-wise relevance propagation: An overview. In W. Samek,
  G. Montavon, A. Vedaldi, L. K. Hansen, & K.-R. Müller (Eds.), *Explainable
  AI: Interpreting, explaining and visualizing deep learning* (Lecture Notes
  in Computer Science, Vol. 11700, pp. 193–209). Springer.
  https://doi.org/10.1007/978-3-030-28954-6_10. The copy is the authors'
  manuscript, whose page numbers differ from the book's, so it is cited by
  section only.
  [Copy](../references/montavon_2019_lrp_overview.pdf)
- Pope, P. E., Kolouri, S., Rostami, M., Martin, C. E., & Hoffmann, H.
  (2019). Explainability methods for graph convolutional neural networks. In
  *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
  Recognition (CVPR)* (pp. 10772–10781).
  https://openaccess.thecvf.com/content_CVPR_2019/html/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.html.
  The copy is the Computer Vision Foundation's open-access version, cited
  here by equation and section.
  [Copy](../references/pope_2019_explainability_gcn.pdf)
- Schnake, T., Eberle, O., Lederer, J., Nakajima, S., Schütt, K. T., Müller,
  K.-R., & Montavon, G. (2022). Higher-order explanations of graph neural
  networks via relevant walks. *IEEE Transactions on Pattern Analysis and
  Machine Intelligence*, *44*(11), 7581–7596.
  https://doi.org/10.1109/TPAMI.2021.3115452. The copy is
  arXiv:2006.03589v3 (27 November 2020), whose section, equation, table and
  figure numbering this page follows. Page numbers on this page are those of
  the PDF file: they are the printed ones for the main paper (pp. 1–14), but
  the supplementary material, PDF pages 15–20, numbers its own pages from 1.
  [Copy](../references/schnake_2022_gnn_lrp_relevant_walks.pdf). Its first
  version, of 5 June 2020, is cited on this page as Schnake et al. (2020), by
  its own equation and PDF page numbers: Schnake, T., Eberle, O., Lederer, J.,
  Nakajima, S., Schütt, K. T., Müller, K.-R., & Montavon, G. (2020). *XAI for
  graphs: Explaining graph neural network predictions by identifying relevant
  walks* (Version 1). arXiv:2006.03589v1. https://arxiv.org/abs/2006.03589v1
  [Copy](../references/schnake_2020_gnn_lrp_arxiv_v1.pdf)
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and
  Technology*, *168*, 110875. https://doi.org/10.1016/j.ast.2025.110875
  [Copy](../references/wen_2026_estgnn_dynamic_shap.pdf)
- Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019).
  GNNExplainer: Generating explanations for graph neural networks. In
  *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*.
  https://arxiv.org/abs/1903.03894v4. The copy is arXiv version 4.
  [Copy](../references/ying_2019_gnnexplainer.pdf)
