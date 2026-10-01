# `explain/` — which sensor drove this prediction, and when

This folder is the "X" of the project's XAI. It takes a trained ESTGNN (the
graph neural network this repository re-implements, which predicts an engine's
remaining useful life from 14 sensors) and works out how much each sensor
contributed to a given prediction.

> **The `.py` files here carry one line of description each, and nothing more.**
> All of the explanation lives in this file. The section
> [What each name means](#what-each-name-means) at the end lists every function
> and what it gives you back.

It needs PyTorch, which is optional in this project. Install it with the
`model` extra:

```bash
pip install -e ".[model]"
```

## Files

- `shap.py` — the paper's method: the coalition weights, the baseline, the
  Shapley values themselves, the per-cycle version, and the check that the
  numbers add up.
- `occlusion.py` — a cheaper method that is not in the paper: replace one
  sensor at a time with the baseline and record how far the prediction moves.
- `plots.py` — the figure of every sensor's contribution at every cycle of one
  engine's life, for either method or both side by side.
- `walks.py` — a method that is not in the paper and works differently from the
  other two: instead of changing inputs and watching the output, it reads the
  network's own weights and splits one prediction into shares, one per route
  through the sensor graph (GNN-LRP, explained in
  [Walks through the sensor graph](#walks-through-the-sensor-graph-gnn-lrp)).
- `flipping.py` — the GNN-LRP paper's own check of an explanation: add or
  remove sensors in the order the walks rank them and follow the score
  ([The authors' check](#the-authors-check-node-flipping)).
- `connections.py` — its companion, also not in the paper: remove one
  connection of the sensor graph at a time, rescale, and record how far the
  prediction moves, in cycles
  ([Removing a connection](#removing-a-connection-a-different-question)).
- `__init__.py` — re-exports the public names from all six.

The notebooks in `notebooks/xai/` use these on a stored trained network, one
method per notebook; `03_walks_fd001.ipynb` covers walks and removing a
connection. The playground's Explain tab offers Shapley values, occlusion and
walks, and `scripts/measure_walks.py` measures walks over every saved seed of a
stored run.

## What a Shapley value is

The model gives one number, and fourteen sensors were involved in producing it.
"How much did each sensor contribute?" has no obvious answer, because the
sensors are not independent: removing one changes what the others are worth.

Cooperative game theory gives an answer. Imagine building the group of sensors
one at a time, in some order. Each time a sensor joins, the prediction changes
by some amount, and that change is what the sensor contributed *in that order*.
Do that for every possible order and average. That average is the sensor's
**Shapley value** (Shapley, 1953). It is the only way of splitting the credit
that satisfies a short list of fairness conditions at once.

Written out, for sensor `i`:

```
phi_i = sum over every subset S not containing i of
            |S|! (n - |S| - 1)! / n!  *  ( f(S + {i}) - f(S) )
```

That is Equation 15 of the paper this project re-implements (Wen et al., 2026).
The fraction is the share of orderings in which exactly the sensors of `S` come
before `i`. `coalition_weights(n)` returns those fractions, one per subset size.

## This implementation is exact, and most are not

The sum runs over every subset of the other sensors. With 14 sensors that is
2^14 = 16,384 subsets, which a computer can simply enumerate. Most published
SHAP work cannot: with 100 features there are more subsets than atoms in the
observable universe, so it approximates, either by sampling orders or by fitting
a weighted linear model (Lundberg & Lee, 2017).

Here no approximation is needed. Every number this folder returns is the
Shapley value as Equation 15 defines it, not an estimate of it. That is worth
stating in the thesis, because it removes a whole class of doubt: a
disagreement with the paper cannot be blamed on sampling noise.

The cost is `2 ** nodes` model evaluations per window — 16,384 for this
project's 14 sensors. `MAX_EXACT_NODES` is set to 20, above which enumeration
stops being reasonable (2^20 is about a million evaluations per window).
Ask for more nodes than that and `shap_values` raises instead of quietly
switching to an approximation. This module deliberately does not provide one.

## What "removing a sensor" means here

Equation 15 needs `f(S)`, "the ESTGNN prediction when only sensors in subset S
are considered". A trained network has no way to accept thirteen inputs; it
always needs fourteen. So a sensor is never actually removed. It is **replaced**
by an uninformative stand-in, and the choice of stand-in decides what the
resulting numbers mean.

The paper does not say what stand-in it uses. We replace an absent sensor's
reading with that sensor's average across the training windows, so the model
sees "this sensor is behaving typically" rather than some arbitrary value
(assumption A-20 in `docs/ASSUMPTIONS.md`).

**Why zero would be the obvious alternative, and why that is a trap.** The
temptation is to feed a sensor 0 when it is left out. Whether that is safe
depends entirely on the rescaling, not on the baseline:

- Under the **min-max** scaling that was the default until 2026-08-29, zero was
  a sensor's lowest ever reading. Substituting it produced a ranking unrelated
  to the mean-window one (Spearman −0.16) and promoted `P15`, the least
  informative sensor, to first place.
- Under the **z-score** default now in force — every sensor rescaled to say how
  many standard deviations it sits from its own training average, so that zero
  means "exactly average" — the number zero is the training mean, and the zero
  baseline agrees with the mean window almost exactly (Spearman +0.996).

The log records both measurements (assumption A-20). The point it draws from
them is that a SHAP
baseline cannot be justified on its own — only against the scaling in force —
and the mean window is kept because it is the one choice defensible under
either. Wen et al. state neither their baseline nor their scaling.

So a Shapley value here answers a specific question: *how much did knowing this
sensor's actual readings, rather than a typical engine's, change the
prediction?* The unit is flight cycles. A value of `+4` means this sensor's
actual readings pushed the predicted remaining life four cycles higher than a
typical engine's readings would have.

**Build the baseline from the training windows only.** `mean_window_baseline`
takes whatever dataset you hand it. Handing it validation or test windows leaks
information into the explanation, in the same way and for the same reason it
would leak into the scaling.

## Dynamic: the same calculation at every cycle

Equation 16 of the paper is Equation 15 applied to the window ending at each
cycle in turn. An engine that ran for 220 cycles yields 191 windows at a window
size of 30, so it yields 191 sets of 14 Shapley values — a table of how the
model's attention moved as the engine wore out. `dynamic_shap` produces that
table for one engine.

Give it the training or validation set. Those keep every window of every
engine. The test set holds only one window per engine, so it cannot show a
trajectory.

## Trap: our table and the paper's Table 12 are not comparable

The paper's Table 12 has the same *shape* as what `dynamic_shap` returns — one
engine, sensors down the side, time steps across, though its columns are
labelled `G1`, `G2` and so on instead of by cycle.

The *contents* are not comparable. Our values are signed and measured in flight
cycles. The paper's cannot be: Table 12's contents cannot be signed Shapley
values satisfying the paper's own Equation 14 (finding F-08 in
`docs/ASSUMPTIONS.md`). Do not place the two side by side without saying so.

The same caution applies to `shap_summary`. Its `mean_abs` column — the average
size of a sensor's contribution, ignoring direction — is the usual measure of
"importance", and it is the order in which the rows of the paper's Fig. 9
appear, which is the convention for a SHAP summary plot although the paper does
not say so. It is **not** established that `mean_abs` is what Table 12 reports.
By the same finding, that table holds magnitudes of something, computed in a way
the paper never states.

`shap_summary` also returns `mean`, the average signed contribution (a large
negative value means the sensor consistently pulled the predicted life *down*),
and `max_abs`, the single largest contribution the sensor ever made.

## The one property to check

Shapley values must add up: the fourteen contributions plus the baseline
prediction must equal the actual prediction, exactly. This is called
**efficiency**, it is Equation 14 of the paper, and it is what makes the
numbers trustworthy rather than merely suggestive. If it fails, the attribution
is wrong no matter how plausible it looks.

`efficiency_error` measures it. It returns the largest absolute violation of
`sum(phi) == f(x) - f(baseline)` across the given examples, in flight cycles.
Anything beyond floating-point noise means the attribution is wrong. The test
suite asserts this.

One more caveat: explaining an untrained network produces numbers that mean
nothing, and nothing in the code stops you doing it. The values will add up
correctly, because efficiency is a property of the arithmetic rather than of
the network, so a passing efficiency check is not evidence that the model
learned anything.

## Occlusion: one sensor at a time

`occlusion_values` answers a simpler question than the Shapley value: how far
does the prediction move when this one sensor's readings are replaced by the
baseline, with every other sensor left as it is? For one window that is

```
occlusion(sensor) = f(every sensor real) − f(every sensor real but this one)
```

and it needs 15 runs of the network — the window itself and one per sensor —
where the Shapley values need 16,384. It is not in the paper.

**It is one term of the Shapley sum.** In Equation 15 the group `S` holding all
thirteen other sensors gives `f(S + {i}) − f(S)`, which is exactly the
occlusion value, and the sum weights that term by `13! 0! / 14! = 1/14`.
Occlusion keeps that one term at full weight and drops the rest. The two agree
only for a sensor that adds the same amount whichever others are present.

Two consequences follow, and neither is about this project's network in
particular:

- **The fourteen occlusion values do not add up** to the prediction minus the
  prediction at the baseline. With `f(a, b) = 2a + 3b + 4ab`, input `(1, 1)` and
  baseline `(0, 0)`, occlusion gives `a` 6 and `b` 7, which sum to 13 where the
  whole change is 9: each is measured with the other present, so each is
  credited with the whole `4ab` term.
- **Two sensors carrying the same information can both look worthless.** With
  `g(a, b) = 10 × max(a, b)`, replacing either input alone leaves the other to
  keep the output at 10, so both get 0, where the Shapley values are 5 each.

`notebooks/xai/02_occlusion_fd001.ipynb` measures how far the two methods part
on the stored network, and
`tests/test_examples_torch.py::test_example_occlusion_is_the_shapley_term_where_the_sensor_arrives_last`
checks the one-term relationship on a three-sensor network.

## The figure: contributions over an engine's life

`plot_contributions_over_time` draws what `dynamic_shap` returns. The top panel
is the prediction being explained: the true remaining life, the predicted one,
the prediction at the baseline, and a line at the cycle where the true remaining
life first drops below its ceiling. Below it is one small panel per sensor,
holding that sensor's contribution at every window, ordered by average size with
the largest first, and all on **one vertical scale**, so that a tall panel is a
large contribution and not just a large axis.

With one method, the area between the line and zero is blue where the sensor
pushed the predicted life up and red where it pushed it down. Given a second
table through `compare` — occlusion values for the same windows, say — each
panel draws the two methods as two lines instead, and the panel order still
follows the first. It returns the matplotlib figure and draws nothing itself,
so a notebook calls `plt.show()` and a script can save it.

It needs the columns `dynamic_shap` produces, and refuses a table without them
rather than drawing an empty chart. `tests/test_shap.py` checks the panel order
and the shared scale.

## Walks through the sensor graph (GNN-LRP)

### What the method is for

A graph network passes numbers between connected sensors, so a prediction is
not built from each sensor on its own: part of what a sensor holds arrives from
its neighbours, and from theirs. **GNN-LRP** (Schnake et al., 2022) is an
explanation method made for exactly that kind of network. It reads every
weight and divides the network's output into shares, one per **walk** through
the sensor graph, so it can say which connections carried a prediction, not
only which sensors.

`walks.py` implements it as its authors define it, with their rule, their
setting of it and their own check of the result. It is not in the paper this
project re-implements, which explains sensors only (assumption A-29), and
nothing in it is taken from that paper's explanation method or its results.
The method's authors call a share a *relevance*, which is why the functions
here are named `walk_relevance`, `node_relevance` and so on; this file says
"share" for the same number.

### First, what one graph layer does

Each of the network's three graph layers gives every sensor a new number made
from its neighbours' current numbers. For sensor `i`:

```
new_i = ReLU( sum over sensors j of  A_hat[i, j] * old_j * W )
```

- `old_j` is what sensor `j` holds before the layer.
- `A_hat[i, j]` is the entry of the rescaled sensor graph for the pair: zero
  unless `j` is joined to `i` in the paper's Table 3, or `j` is `i` itself
  (every sensor is joined to itself before rescaling, so it keeps its own
  number). Rescaled by rows, the default, the entries of one row add up to 1.
- `W` is the layer's learned weight. The real layers hold many numbers per
  sensor, so `old_j` is a list and `W` a table, but one number is enough to see
  the idea.
- `ReLU` keeps a positive total and turns a negative one into zero.

After one layer, a sensor's number is a mix of its neighbours'. After two, it
is a mix of numbers that were already mixes, so it carries its neighbours'
neighbours too. After three, three steps' worth.

### What a walk is

A **walk** is the list of sensors one piece of a reading passes through, one
per layer. `T50 -> T30 -> T30 -> Ps30` is T50's reading, carried into T30 by the
first layer, kept at T30 by the second (through T30's connection to itself),
and carried into Ps30 by the third. A walk may stay put or come back, which is
what separates it from a path.

On the paper's graph there are 96 entries of `A_hat` that are not zero — the 41
connections, once in each direction, plus the fourteen connections of a sensor
to itself — and they allow **5,178 walks of three steps**
(`tests/test_walks.py::test_the_papers_graph_has_5178_walks_of_three_steps`).
A walk that would use a pair with no connection carries exactly zero.

### Why one prediction splits exactly into walks

Take one window. Every ReLU in the network is, for that window, either **on**
(it passes its input unchanged) or **off** (it passes zero). Hold that on/off
pattern fixed. What is left between the numbers entering the first graph layer
and the output is nothing but multiplication and addition: multiply by an entry
of `A_hat`, multiply by a weight, add. A sum of products can be multiplied out
term by term, and every term follows exactly one route: which sensor it started
at, which sensor it reached after layer one, after layer two, after layer three.
Grouping the terms by that route gives one number per walk, and the numbers add
up to the whole.

Two things make that exact for this network, and both are properties of the
network as it already is, not changes made for the explanation:

1. **The graph layers add no constant of their own** (assumption A-12, taken
   from the paper's Equations 8 to 10). Every number leaving a graph layer is
   therefore made entirely of numbers that entered it, and so traces back to
   some sensor. If the constants are switched on with `gcn_bias=True`, their
   part is carried forward separately and reported with the constant below.
2. **The layer after the graph layers keeps each sensor separate.** The network
   lays the fourteen sensors' numbers side by side (448 numbers) before its two
   ordinary layers, rather than averaging them, so the end of every walk is a
   known sensor.

What does not trace back to any sensor is the constant that the two ordinary
layers add, `fc1` and `fc2`. It is reported as `constant`, one number per
window. It changes from window to window, because it passes through `fc1`'s
ReLUs and which of those are on depends on the window. So it is not a fixed
point of reference: it is one number per window, not one for the whole
network.

**Do not confuse the constant with a reference engine.** The walks add up to
*score minus constant*. They do not add up to *score minus the score of some
typical engine*. The nearest thing to a
typical engine here is `score_at_zero`, the score when every number entering
the graph layers is zero — for the default stage, a window of training averages.
It is one number for the network, and it differs from the constant of a given
window by whatever `fc1`'s ReLUs switch between that zero input and the window.

### A worked example on three real sensors

Test engine 5's last readings of T24, T30 and T50, after smoothing and z-score
scaling (how many standard deviations each sits from its training average), are
−0.3734, 0.1146 and 0.8176. In Table 3 the three form a path: T24 is joined to
T30 and T30 to T50, but T24 is not joined to T50. Rescaled by rows, with each
sensor joined to itself:

```
            from T24   from T30   from T50
to T24        1/2        1/2         0
to T30        1/3        1/3        1/3
to T50         0         1/2        1/2
```

Take a network with one graph layer whose weight is 1, then `fc1` with weights
`[2, 1, -1]` and constant 0.5, then `fc2` with weight 1 and constant 0.

```
graph layer   T24: (-0.3734 + 0.1146) / 2          = -0.1294  -> ReLU off -> 0
              T30: (-0.3734 + 0.1146 + 0.8176) / 3 =  0.1863  -> on
              T50: ( 0.1146 + 0.8176) / 2          =  0.4661  -> on
fc1           2 * 0 + 1 * 0.1863 - 1 * 0.4661 + 0.5 = 0.2202  -> on
score         0.2202
prediction    125 / (1 + e^-0.2202)                = 69.35 cycles
```

Each walk here is one step. Its share is the reading it starts from, times the
graph's entry for the step, times what one unit arriving at the end sensor is
worth to the score: 0 at T24 (its ReLU is off), `1 * 1 = 1` at T30, and
`-1 * 1 = -1` at T50.

```
T24 -> T30   -0.3734 * 1/3 *  1 = -0.1245
T30 -> T30    0.1146 * 1/3 *  1 =  0.0382
T50 -> T30    0.8176 * 1/3 *  1 =  0.2725
T30 -> T50    0.1146 * 1/2 * -1 = -0.0573
T50 -> T50    0.8176 * 1/2 * -1 = -0.4088
constant      0.5 * 1           =  0.5
                           total   0.2202   = the score
```

The other four walks carry zero: T24 -> T24 and T30 -> T24 because T24's ReLU
was off, and T24 -> T50 and T50 -> T24 because the graph has no such
connection. T24's own node passed nothing on, yet T24's reading still has a
share, −0.1245, through the walk that left it for T30. That is exactly the
thing a list of sensors cannot show.
`tests/test_examples_torch.py::test_example_walks_on_three_real_sensors_worked_by_hand`
builds this network and checks every number above.

### The rule behind the shares, and what "the score" is

Written generally for the setting gamma = 0 (the next section says what a
positive gamma changes), with the ReLU pattern of this window held fixed, the
share of the walk `i0 -> i1 -> i2 -> i3` is

```
R(i0, i1, i2, i3) = A_hat[i1, i0] * A_hat[i2, i1] * A_hat[i3, i2]
                    * ( x_i0  W1  g1(i1)  W2  g2(i2)  W3  g3(i3)  worth(i3) )
```

- `x_i0` is the list of numbers the temporal stage hands the first graph layer
  for the starting sensor (one number for the default stage).
- `W1`, `W2`, `W3` are the three graph layers' weight tables.
- `g1(i1)` is the on/off pattern of the first layer's ReLUs at sensor `i1`: a
  1 for each of its 32 numbers whose total came out positive, a 0 otherwise.
  `g2`, `g3` likewise.
- `worth(i3)` is what one unit of each of sensor `i3`'s last-layer numbers adds
  to the score, through `fc1`'s ReLUs as they were and `fc2`.

Summed over every walk, and with the constant added, it gives back the score
exactly. Each walk's share is also what the authors of GNN-LRP get by running
the network backwards once per walk with every other route blocked; the test
`test_every_walk_matches_the_authors_masked_gradient` does it their way for 30
walks and requires the same numbers.

### The setting gamma: the authors' own

GNN-LRP has one setting, **gamma**, applied layer by layer. What it does is
easiest to see at one node that receives from two sources. Say one source
brings `2 x 1 = 2` (a number of 2 through a weight of 1) and the other brings
`1 x -0.5 = -0.5`, so the node holds 1.5 and its ReLU is on. The node's 1.5 has
to be divided between the two sources.

```
gamma = 0:  shares 2 and -0.5                         (they add up to 1.5)
gamma = 1:  count each positive weight twice: 2 x (1 + 1) = 4, and -0.5 stays
            rescale to keep the total: 4 x 1.5 / 3.5  =  1.714
                                     -0.5 x 1.5 / 3.5  = -0.214   (add up to 1.5)
```

A positive gamma shrinks the negative shares and keeps the total. In the
formula above it replaces each weight table `W` by `W + gamma * max(0, W)` and
each on/off pattern `g` by the factor that rescales the layer back to what it
computed. With gamma = 0 the method is what its authors call GNN-GI, a simpler
baseline they report as noisier; their method proper uses gamma > 0.

**The setting used here is the authors' own** (assumption A-29), `paper_gammas`
in `walks.py`:

- **Their default schedule.** Their published code, when given no setting,
  uses gamma = 3 at the first graph layer falling in equal steps to 0 at the
  last (`linspace(3, 0, depth)`), and their paper keeps gamma = 0 in the layers
  after the graph, which here are `fc1` and `fc2`.
- **Gamma 0 at the first graph layer.** The rule's derivation assumes the
  numbers entering a layer are never negative (their appendix C), and where a
  network lacks that structure — a ReLU, then a weighted sum, then a ReLU — the
  authors use gamma = 0 themselves, as for their SchNet model. This network's
  first graph layer receives rescaled readings, which are negative whenever a
  sensor is below its training average; there, a positive gamma pushes a
  negative contribution further the wrong way and can divide by a total of the
  wrong sign. `walk_relevance` refuses a positive gamma at the first layer.

For three graph layers the setting is therefore 0, 1.5 and 0. It matters: on
the stored default network, seed 42, the walks of the held-back engine 5 move
by a median of 34 % of their total size between this setting and gamma 0 on
184 of its 240 windows (`notebooks/xai/03_walks_fd001.ipynb`, step 4). On the
other 56 they do not move at all, which is the one case where gamma cannot
matter. With one number per sensor, every
sensor leaves the first graph layer holding the positive or the negative part
of the same weight row, scaled by its own total; when all the sensors feeding
a node at layer 2 come from the same side, the extra weight from gamma scales
the top and bottom of every share alike and cancels.

Any other setting can be passed, one number per graph layer with 0 first, to
see how far a conclusion depends on gamma.

One small difference from the authors' code: it adds `1e-6` to every layer's
rule-weighted total before dividing by it, so that a total of zero cannot
break the division. Here a layer's numbers are divided only where its ReLU is
on, and there the rule-weighted total is at least the layer's own total, which
is above zero, so no such guard is needed and the layer passes on exactly what
it computed.

**The score is not the prediction.** The network's last step turns a score
`z` into a prediction with `125 / (1 + e^-z)`, the sigmoid. That step bends:
equal changes of `z` move the prediction by different numbers of cycles
depending on where `z` starts, so shares of `z` cannot be turned into shares of
cycles that still add up. The shares here are therefore **shares of the score
before the sigmoid**, and have no unit of cycles. A score of 0 is a prediction
of 62.5 cycles; positive scores predict more, negative less.

### The one property to check: conservation

Walks plus constant must equal the score. `conservation_error` returns the
largest miss across the windows. It is a real check rather than a restatement,
because the three quantities are computed by separate routes: the walks by the
multiplication above, the constant by carrying the layers' added numbers
forward on their own, and the score by running the network's layers.

More checks sit in `tests/test_walks.py`. Three of them compare the walks at
gamma 0, the authors' GNN-GI, with a quantity PyTorch computes by its own
differentiation, sharing no code with `walks.py`; at gamma 0 every share is a
number times its derivative, so these must agree exactly:

- added up by **starting** sensor, the walks give that sensor's number times
  how fast the score changes when that number changes;
- added up by **ending** sensor, they give the last graph layer's numbers times
  how fast the score changes with each;
- added up over the three layers for **one connection**, they give the graph's
  entry for that pair times how fast the score changes when that one entry
  changes. That describes a small change to one entry. Removing the connection
  is a large change that also rescales the rest of its row, so what removal
  does is a different measurement, which this file does not make.

At any setting, each walk's share must equal what the authors' own procedure
gives — rewriting each layer, masking every route but the walk, differentiating
once (their Section 3.2.1) — which
`test_every_walk_matches_the_authors_masked_gradient` checks for 30 walks at
gamma 0 and at gamma (0, 0.5, 0.25). And the score, put through the sigmoid,
must reproduce the network's own prediction; `walk_relevance` itself refuses to
return anything if it does not, so a network that one day computes something
else is refused rather than explained.

### The authors' check: node flipping

The GNN-LRP paper checks every explanation against the network it describes,
by a procedure it calls node flipping (its Section 4.1). `flipping.py` runs it.

**What a ranking says a set of sensors is worth.** For walks, every walk lying
wholly inside the set, added up (`subgraph_relevance`); for a ranking of single
sensors, their shares added up. With no sensor the worth is 0; with all of them
it is the walks' whole sum.

**Two tasks.** *Adding*: start from a graph with no sensor and add, one at a
time, the sensor whose set the ranking rates highest (their Equation 14).
*Removing*: start from the full graph and remove the sensor whose loss the
ranking rates smallest (their Equation 15). After each step the network is run
on the graph that is left (`subgraph_scores`). The paper plots the score while
adding, and its distance from the full graph's score while removing; the mean
of each curve is its **area under the flipping curve**. A ranking that
describes the network well raises the score fastest when adding (higher is
better) and moves it least when removing (lower is better).

**What is set beside the walks.** As in the paper: the same check run with the
walks' shares added up by starting sensor, which the paper says matches a
node-level method (Pope et al.); with the simpler GNN-GI; and with random
orders, averaged (`RANDOM_ORDERS`, 20).

**How a sensor is taken out** (assumption A-33): its row and its column of the
rescaled graph are set to zero, its link to itself included, and nothing is
rescaled again — the paper feeds its networks the masked connectivity. Its
readings stay in the window, but with no connection left nothing of them
reaches the score through the graph layers. With no sensor at all, every graph
layer holds zeros and the score is `score_at_zero`
(`test_the_full_graph_scores_what_the_walks_split_and_the_empty_graph_its_zero`).

The worked example: on the three-sensor network of the walk example, the walks
rank T30 first, then T24, then T50. With no sensor the score is the constant,
0.5; with T30 it is 0.5382; adding T24 brings it back to 0.5; with all three
it is 0.2202; the area under that curve is their mean, 0.4396
(`tests/test_examples_torch.py::test_example_node_flipping_by_hand_on_the_same_three_sensors`).

The paper also allows replacing a greedy step, at regular intervals, by adding
the sensor with the largest share on its own; it calls this optional and gives
no interval, so `flipping.py` does not do it.

### What this method is, in seven answers

1. **What is explained:** one window's score before the sigmoid.
2. **What it can see:** every weight, the sensor graph, and the on/off pattern
   of every ReLU at that window. It is specific to this kind of network, and
   refuses a graph layer of any other kind.
3. **What it changes:** nothing. No input is replaced and no connection is
   removed.
4. **What it measures:** how the network's own arithmetic, at this window,
   divides the score among routes through the graph.
5. **What comes out:** one number per walk, 14 × 14 × 14 × 14 = 38,416 per
   window with most of them zero on the paper's graph. They can be added up by
   starting sensor (`node_relevance`), by the connection used at one layer
   (`step_relevance`), or into walks that never leave their starting sensor
   against walks that move (`crossing_relevance`).
6. **What it assumes:** that holding the ReLU pattern fixed describes the
   network near this window, and that each layer's numbers are divided among
   their sources by the authors' rule — at gamma 0, a number's share is the
   number times its effect; with the authors' setting, positive contributions
   count more heavily at the second graph layer. The first is less local than it sounds: the graph layers add no
   constant, so scaling the numbers entering them by any positive factor
   scales every graph layer's numbers by the same factor and switches none of
   their ReLUs
   (`test_scaling_the_graph_input_leaves_every_graph_relu_as_it_was`); only
   `fc1`'s ReLUs can switch on the way from zero to the window. The implicit point of comparison is zero at the graph layers'
   input. For the stages that learn nothing, which pass on readings or
   averages of readings, zero under z-score scaling means every sensor at its
   training average; for the recurrent stages it means nothing so simple.
7. **What it does not show:**
   - **What would happen if a connection were removed.** Removing one would
     change the row rescaling of the whole row and could switch ReLUs on or
     off — a different network, needing a different measurement.
   - **Anything causal about engines.** A large share for `T30 -> Ps30` says the
     trained network routed that much of its score that way, not that T30
     affects Ps30 in the engine.
   - **That the walks are stable.** A reading times its effect can swing
     between neighbouring windows where a ReLU flips, so single walks are best
     read in aggregate, over many windows.
   - **That a large share matters on its own.** Walk shares are often large
     and of both signs, and mostly cancel. `crossing_relevance` reports
     `absolute_total`, the sum of every walk's share ignoring its sign; when it
     is many times the size of the walks' actual sum, a list of the largest
     walks shows mostly cancellation, and should be read with that number
     beside it.

### Where the walks start, for each temporal stage

A walk starts at the numbers the temporal stage hands the first graph layer.

- `last_cycle`, the default: the sensor's last reading, so a walk starts at a
  reading.
- `mean_window`: the window's average.
- `window_summary`: three numbers per sensor — last reading, average, change —
  whose shares are added together for the sensor.
- `raw_window`: all 30 readings, added together for the sensor.
- `lstm` and `gru`: the output of that sensor's own recurrent network. Each
  one reads only its own sensor's window, so the starting sensor is still the
  right one, but the walks say nothing about which cycles of the window
  mattered, and zero at the recurrent network's output is not any real window.
  Going further back into the recurrent network needs rules for its gates that
  this file does not have. `walk_relevance` logs a warning saying so.

### Why this method, and not GNNExplainer or PGExplainer

The two other widely used graph explanation methods, GNNExplainer (Ying et al.,
2019) and PGExplainer (Luo et al., 2020), look for a small set of connections
that keeps the prediction nearly the same. Each learns, by gradient descent, a
number between 0 and 1 for every connection — a mask — which weakens that
connection by that factor. GNNExplainer learns one mask per prediction;
PGExplainer trains a second, small network that produces the mask from the
explained network's own internal numbers. For this network they bring five
costs that GNN-LRP does not:

- **Built around classification.** Both papers state their objective for
  predictions of a class, as a cross-entropy over the classes, and neither
  mentions regression. PyTorch Geometric, the common library, runs both on a
  number by swapping in a squared error — for PGExplainer against the true
  label rather than the network's own prediction — which is the library's
  choice, not the papers'.
- **Settings no source fixes for this network.** How strongly a large mask is
  penalised, how strongly an in-between one is, the learning rate, the number
  of steps and, for PGExplainer, a temperature schedule. The papers give values
  tuned on their own benchmarks. A replication of PGExplainer (Holdijk et al.,
  2021) could not reproduce its results from the paper alone and found them to
  depend strongly on exactly these settings.
- **A random start.** Both begin from a random mask in their usual code, and
  GNNExplainer's paper says its optimisation problem is not convex and that it
  reaches a good local minimum often, not always, so nothing guarantees that
  two runs on the same window agree.
- **A change to the network.** `ESTGNN` applies one fixed sensor graph, stored
  as a table, to every window. A mask needs the network to accept a weakened
  graph per window, and the library versions work only with its own layer type
  and call signature, which this network does not use.
- **Before or after the rescaling.** Weakening an entry of the rescaled graph
  leaves a row that no longer adds up to 1, a graph the network never saw in
  training; weakening the raw graph and rescaling again changes every other
  entry of that row too. The library masks after the rescaling and never masks
  a sensor's connection to itself; GNNExplainer's own code masks the raw graph
  and removes the connections to self. Either way it is a further choice.

The mask it returns is also a different kind of answer: a number between 0 and
1 per connection, with no unit, that does not add up to anything.

The evidence on which method explains graph networks best is thin and does not
cover this task. Of the independent comparisons, only a survey by Yuan et al.
(2022) ran GNN-LRP, where walk-based methods scored higher on its main measure
of faithfulness than GNNExplainer and PGExplainer; others that left GNN-LRP out
found GNNExplainer the strongest on classifying single nodes and PGExplainer
weak. All of them test classification. None tests a regression on a sensor
graph.

What GNN-LRP gives up in exchange: it is specific to networks it has rules for,
its cost grows as the number of sensors raised to the number of layers plus
one (38,416 per window here, which is small only because there are 14 sensors
and three layers), it explains the score before the sigmoid rather than the
prediction in cycles, and at the default setting it is the variant its authors
found noisier than their recommended one.

### Trap: sensors the graph cannot tell apart

On the paper's graph, three groups of sensors have **identical rows** in the
rescaled graph, under row and symmetric rescaling alike: P15, Nf, NRf and BPR;
P30 and Ps30; Nc and W31. Each member of a group is joined to exactly the same
sensors as the others, itself and the others in the group included.
`indistinguishable_sensors` finds such groups for whatever graph a network was
built on; the correlation-based graph has different ones.

A graph layer gives sensor `i` a mix of its row of neighbours, so two sensors
with the same row get the same mix. From the first graph layer on, every member
of a group holds exactly the same numbers. (The audit of 2026-09-08 records the
consequence for the network itself: the rescaled graph's fourteen rows can all
be written as sums of multiples of just eight of them.) Three things follow for
walks:

- **Walks that differ only by which member they pass through in the middle
  carry exactly the same share.** `P30 -> P30 -> T30 -> T50` and
  `P30 -> Ps30 -> T30 -> T50` are one fact, and a list of the largest walks will
  show it twice (`test_walks_differing_only_by_an_indistinguishable_sensor_in_the_middle_are_equal`).
- **Where a walk ends within a group is decided by the random starting
  weights, not by training.** `fc1` receives the same numbers from every member,
  so the blocks of `fc1` weights that read P30 and Ps30 always get the same
  correction during training, and the difference between them stays whatever
  the random start made it
  (`test_training_never_changes_the_difference_between_their_fc1_weights`).
  How a walk's share splits between ending at P30 and ending at Ps30 follows
  that difference. Add the shares of walks ending in one group together before
  reading them.
- **Where a walk starts within a group is meaningful.** Each member's own
  reading enters there, and the readings differ.

`top_walks(result, symbols, pool=indistinguishable_sensors(model, symbols))`
does the adding for you: every position after the first is named by its group,
`P30/Ps30`, and the shares of walks that then have the same name are added.
The starting sensor keeps its own name, for the reason just given. The same
`pool` does two more things:

- `crossing_relevance(result, symbols, pool=...)` counts a walk as staying when
  it never leaves its starting sensor's group, so P30 -> P30 -> P30 -> Ps30
  stays. Without the pool it would count as moving, and whether it moved would
  follow the random start.
- `connection_relevance(result, symbols, pool=...)` adds up what passed along
  each connection over every graph layer and splits what arrives at a group at
  the last layer equally among its members, keeping the group's total. The
  equal split is a way of drawing, not a measurement: the real split is the
  random start (finding F-18 records it on the stored networks).

On the graph where every sensor is joined to every other, all fourteen rows are
identical, so all fourteen sensors form one group. There, walks differ only by
where they start, and a walk list is no explanation of the graph at all.

The network's predictions are not affected, because they depend only on the
group's total, which training does shape; nor is any explanation that replaces
a sensor's readings whole, for the same reason.

### Traps

- **Every walk is stored.** `relevance` has one axis per sensor visited:
  `(windows, 14, 14, 14, 14)`. A network with more graph layers multiplies the
  size by 14 per layer; `walk_relevance` refuses above `MAX_WALKS_PER_EXAMPLE`
  (one million) walks per window rather than sampling some of them.
- **The order of the axes is the order of the walk.** Axis 1 is the starting
  sensor, the last axis is where the walk ends. `step_relevance` returns
  `(windows, from, to)` — the transpose of `A_hat`, whose rows are the sensor
  that receives.
- **An untrained network conserves perfectly.** Conservation is a property of
  the arithmetic, so passing it says nothing about whether the network
  learned anything.
- **Much of what a walk share says is fixed before training.** A sensor's
  connection to itself weighs one over its number of connections plus one, so
  walks that move between sensors carry most of the score in any network built
  on this graph, trained or not. `scripts/measure_walks.py` therefore explains
  each trained network beside its own untrained starting point, and a figure is
  about training only where the two differ. The measured table is in the log
  (assumption A-29).

### Shares over the cycles of a window

`cycle_relevance(model, features)` splits each sensor's walk total, at gamma 0,
over the cycles of its window: each reading times how fast the score changes
with it, the ReLUs held as the window set them. Added over the cycles, it gives
back `node_relevance` exactly
(`test_cycle_shares_add_up_to_each_sensors_walk_total`). It works only for the
stages that hand the graph layers a fixed weighted sum of the readings with
nothing added — `last_cycle`, `mean_window`, `window_summary`, `raw_window` —
and refuses a recurrent stage. With the default stage every share lands on the
last cycle, because that is the only cycle the network reads.

## Removing a connection: a different question

A walk's share says how much of the score *flowed* along a connection. It does
not say what the prediction would be *without* the connection. `connection_removal`
answers that second question: it deletes one connection from the raw graph, in
both directions, rescales the whole graph as the network's graph was made, runs
the same trained network again without retraining, and records the prediction
with the graph as trained minus the prediction without the connection, in
cycles (assumption A-30).

The worked example: the three-sensor network of the walk example, with T24 -
T30 removed. T24's row becomes `[1, 0, 0]` and T30's `[0, 1/2, 1/2]`, so T30
now weighs T50 by 1/2 instead of 1/3 — an entry nobody removed changes, which is
why removal is not the same thing as flow. The prediction goes from 69.35 to
77.81 cycles, so that connection is worth −8.45 cycles
(`tests/test_examples_torch.py::test_example_removing_a_connection_by_hand_on_the_same_three_sensors`).

On the default network's five stored seeds, ranking the connections by flow and
by removal gives weakly opposite orders (the table is in assumption A-30). Removing a
connection that touches one of the sensors the graph cannot tell apart makes
that sensor's row differ from its group's, so what that removal does is partly
the random start too.

`connection_removal` refuses a raw graph and rescaling that do not give back the
network's own graph, since every removal would then be measured against the
wrong one.

## Judgement calls made in this folder

- **What "a sensor not considered" means.** Equation 15 needs a prediction with
  a sensor absent, but the network always needs all fourteen inputs, and the
  paper never says what it substitutes. We substitute that sensor's average
  window over the training set (assumption A-20 in `docs/ASSUMPTIONS.md`).
  Occlusion uses the same substitute, so the two methods differ only in the
  arithmetic.
- **Offering occlusion at all.** It is not in the paper. It is here as the
  cheapest comparison, not as a replacement.
- **Offering walks at all, and how** (assumption A-29). Not in the paper, which
  explains sensors only. GNN-LRP rather than a mask method, for the reasons in
  [Why this method](#why-this-method-and-not-gnnexplainer-or-pgexplainer);
  the score before the sigmoid rather than the prediction in cycles, because
  the sigmoid cannot be split into shares that add up; the authors' own
  setting of gamma, with 0 at the first graph layer; the dense layers' constants reported as
  one number per window rather than spread over the walks; walks that start at
  the temporal stage's output rather than at the raw readings; and sensors the
  graph cannot tell apart counted as one wherever a walk ends.
- **What removing a connection means** (assumption A-30): deleted from the raw
  graph in both directions, the graph rescaled, the network not retrained.
- **Running the authors' check on this network** (assumption A-33): a sensor
  is taken out by zeroing its row and column of the rescaled graph without
  rescaling again, the random ranking averages 20 orders, and the optional
  coarse steps are left out.

That the paper's Table 12 cannot hold signed Shapley values satisfying its own
Equation 14 is not a choice of ours but an observation about the paper (finding
F-08, also in `docs/ASSUMPTIONS.md`).

## References

- Holdijk, L., Boon, M., Henckels, S., & de Jong, L. (2021). [Re] Parameterized
  explainer for graph neural network. *ReScience C*, 7(2), #7.
  https://doi.org/10.5281/zenodo.4834242
- Luo, D., Cheng, W., Xu, D., Yu, W., Zong, B., Chen, H., & Zhang, X. (2020).
  Parameterized explainer for graph neural network. *Advances in Neural
  Information Processing Systems 33 (NeurIPS)*. https://arxiv.org/abs/2011.04573
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. *Advances in Neural Information Processing Systems 30
  (NeurIPS)*, 4765-4774. https://arxiv.org/abs/1705.07874
- Shapley, L. S. (1953). A value for n-person games. In H. W. Kuhn & A. W.
  Tucker (Eds.), *Contributions to the Theory of Games, Volume II*
  (pp. 307-317). Princeton University Press.
  https://doi.org/10.1515/9781400881970-018
- Schnake, T., Eberle, O., Lederer, J., Nakajima, S., Schütt, K. T., Müller,
  K.-R., & Montavon, G. (2022). Higher-order explanations of graph neural
  networks via relevant walks. *IEEE Transactions on Pattern Analysis and
  Machine Intelligence*, 44(11), 7581-7596.
  https://doi.org/10.1109/TPAMI.2021.3115452 — section, table and appendix
  numbers in this file are those of the arXiv version,
  https://arxiv.org/abs/2006.03589 (version 3), since the journal's own text
  could not be retrieved to check them.
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and
  Technology*, 168, 110875. https://doi.org/10.1016/j.ast.2025.110875
- Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019).
  GNNExplainer: Generating explanations for graph neural networks. *Advances in
  Neural Information Processing Systems 32 (NeurIPS)*.
  https://arxiv.org/abs/1903.03894
- Yuan, H., Yu, H., Gui, S., & Ji, S. (2022). Explainability in graph neural
  networks: A taxonomic survey. *IEEE Transactions on Pattern Analysis and
  Machine Intelligence*. https://doi.org/10.1109/TPAMI.2022.3204236 — the
  benchmark described above was read in the arXiv version,
  https://arxiv.org/abs/2012.15445 (version 3).

## What each name means

Everything `shap.py`, `occlusion.py`, `plots.py` and `walks.py` offer.

| Name | What it is |
|---|---|
| `mean_window_baseline(dataset)` | The average reading of each sensor, which stands in for a sensor left out of the calculation. **Training windows only.** |
| `shap_values(model, features, baseline)` | How many cycles each sensor added to the prediction. Takes one window or many, and returns one number per sensor, or one row per window. Raises above `MAX_EXACT_NODES` sensors rather than silently switching to an estimate. |
| `dynamic_shap(model, dataset, baseline, unit=..., symbols=...)` | The same calculation at every cycle of one engine's life. Returns one row per cycle; the columns are listed below. |
| `shap_summary(values, symbols)` | Ranks the sensors by how much they mattered. One row per sensor; the columns are listed below. |
| `efficiency_error(model, features, baseline, values)` | The largest amount by which the numbers fail to add up, in cycles. Anything above rounding noise means a bug. |
| `coalition_weights(n_nodes)` | How much a group of each size counts, from Equation 15. Element `s` is the weight for groups of `s` sensors. |
| `MAX_EXACT_NODES` | 20. Above this, trying every combination stops being reasonable. |
| `occlusion_values(model, features, baseline, batch_size=512, device=None)` | From `occlusion.py`. The prediction with every sensor real minus the prediction with one sensor replaced by the baseline, one number per sensor. Takes one window or many, like `shap_values`; `batch_size` changes speed and memory only. |
| `plot_contributions_over_time(table, sensors, compare=None, method="Shapley value", compare_method="occlusion", title=None)` | From `plots.py`. The figure described above, from a table shaped like `dynamic_shap`'s. `compare` is a second table of values for the same windows, one column per sensor. |
| `walk_relevance(model, features, gamma=None, batch_size=16, device=None)` | From `walks.py`. Splits each window's score before the sigmoid into one share per walk through the sensor graph. Takes one window (`(sensors, cycles)`) or many (`(windows, sensors, cycles)`) and returns a `WalkRelevance`, described below. `gamma` left out is the authors' setting (`paper_gammas`); otherwise one number, used from the second graph layer on, or one number per graph layer with 0 first. Refuses a network whose own prediction it cannot rebuild, and more than `MAX_WALKS_PER_EXAMPLE` walks per window. `batch_size` changes speed and memory only. |
| `WalkRelevance` | The record `walk_relevance` returns, one entry per window in every array. Its fields are listed below. |
| `conservation_error(result)` | The largest amount by which walks plus constant miss the score, in units of the score. Anything above rounding noise means a bug. |
| `node_relevance(result)` | Each sensor's share: every walk that starts at it, added up. Shaped `(windows, sensors)`. |
| `step_relevance(result, step)` | What passed along each connection at one graph layer, `step` counted from 1. Shaped `(windows, from, to)`. At every step the entries add up to the same total. |
| `crossing_relevance(result, symbols=(), pool=())` | One row per window: the walks that never leave their starting sensor (or its group, with `pool`), the walks that move, the constant, the score and the prediction. The columns are listed below. |
| `connection_relevance(result, symbols, pool=())` | What passed along each connection, added over every graph layer, shaped `(windows, from, to)`. With `pool`, what arrives at a group at the last layer is split equally among its members. |
| `cycle_relevance(model, features, gamma=None)` | Each sensor's walk total, under the same setting of the rule as `walk_relevance`, split over the cycles of its window, shaped like the windows. Refuses a stage that learns its own numbers. |
| `paper_gammas(n_layers)` | The authors' default setting for that many graph layers: `linspace(3, 0, n_layers)` with 0 at the first layer. `[0, 1.5, 0]` for three. |
| `PAPER_GAMMA_START` | 3. The gamma the authors' published code gives the first graph layer when given none. |
| `node_flipping(model, features, explanations, random_orders=20)` | From `flipping.py`. The authors' check: for every window, both tasks and every ranking — each explanation's walks and its starting sensors, and random orders — the order sensors were flipped in (`order`), the curve the paper plots (`step_0` to `step_14`) and its mean (`aufc`). `explanations` maps a name to a `WalkRelevance` of the same windows. |
| `aufc(table)` | From `flipping.py`. The area under each flipping curve averaged over the windows: one row per task and ranking, with its spread and count. |
| `subgraph_relevance(relevance, present)` | From `flipping.py`. What one window's walks say a set of sensors is worth: every walk lying wholly inside it, added up. |
| `subgraph_scores(model, window, masks)` | From `flipping.py`. The network's score before the sigmoid with only each mask's sensors left in its graph, one per mask. |
| `FLIPPING_TASKS`, `RANDOM_ORDERS` | From `flipping.py`. The two tasks, adding (`activation`) and removing (`pruning`), and the 20 random orders averaged into the random ranking. |
| `connection_removal(model, features, raw_adjacency, symbols, method=...)` | From `connections.py`. One column per connection of the raw graph, named `T24-T30`: the prediction with the graph as trained minus the prediction with that connection removed and the graph rescaled, in cycles. `method` is the run's own rescaling. |
| `connections_of(raw_adjacency)` | Defined beside the graph inspection tools in `graph/adjacency.py` and re-exported here for existing explanation code. Every connection of the raw graph once, as pairs of sensor positions. |
| `top_walks(result, symbols, count=20, pool=())` | The walks that carried the most across the windows given, named by sensor. `pool` takes groups of sensors, usually from `indistinguishable_sensors`, and adds together walks that differ only by which member of a group they pass through or end at. The columns are listed below. |
| `indistinguishable_sensors(model, symbols)` | Groups of sensors whose rows of the network's rescaled graph are identical, so the graph layers give every member the same numbers. A list of tuples of names; on the paper's graph, three groups. Read the trap above before reading walks that end in one. |
| `MAX_WALKS_PER_EXAMPLE` | One million. Every walk is stored, so a network with more graph layers than this allows is refused rather than sampled. |

`dynamic_shap` returns these columns, in cycle order:

| Column | What it holds |
|---|---|
| `end_cycle` | The cycle the window ends at. |
| `rul` | The true remaining life at that cycle. |
| `predicted` | What the model predicted there, in cycles. |
| `baseline` | What the model predicts for a wholly typical engine. Constant down the column. |
| one per sensor | That sensor's contribution at that cycle, in cycles. |

`shap_summary` returns one row per sensor, sorted by `mean_abs` downwards:

| Column | What it holds |
|---|---|
| `mean_abs` | The average size of the sensor's contribution, ignoring direction. This is the usual measure of importance — but read the trap above before comparing it with the paper. |
| `mean` | The average signed contribution. A large negative value means the sensor consistently pulled the predicted life down. |
| `max_abs` | The single largest contribution the sensor ever made. |

`shap_values` and `dynamic_shap` both take an optional `device`, and default to
wherever the model already is. `shap_values` also takes `batch_size`, which
changes speed and memory only, never the numbers.

A `WalkRelevance` holds:

| Field | What it holds |
|---|---|
| `relevance` | Every walk's share, shaped `(windows, 14, 14, 14, 14)` for three graph layers: axis 1 is the sensor the walk starts at, the last axis the sensor it ends at. |
| `constant` | The part of each score from the layers' added constants, which belongs to no sensor. One number per window. |
| `score` | The score before the sigmoid. `relevance` summed over every walk, plus `constant`, gives it back. |
| `predicted` | The network's own prediction, in cycles. |
| `gamma` | The gamma used at each graph layer, first layer first. |
| `score_at_zero` | The score when every number entering the graph layers is zero: one number for the network, not a baseline the walks add up to. |
| `n_nodes`, `n_steps` | How many sensors, and how many steps every walk takes. |

`crossing_relevance` returns one row per window:

| Column | What it holds |
|---|---|
| `stays` | Added shares of the walks that stay on one sensor, reaching the end through its connection to itself at every layer; with `pool`, that never leave its group. |
| `moves` | Added shares of every other walk. |
| `absolute_stays` | The staying walks' shares added up ignoring their sign. |
| `constant`, `score`, `predicted` | As in `WalkRelevance`. `stays + moves + constant` equals `score`. |
| `absolute_total` | Every walk's share added up ignoring its sign. Many times the size of `stays + moves` means the walks mostly cancel. |

`top_walks` returns one row per walk, sorted by `mean_abs` downwards:

| Column | What it holds |
|---|---|
| `walk` | The sensors visited, in order, joined by `->`. |
| `mean` | The walk's average share over the windows given. |
| `mean_abs` | The average size of its share, ignoring direction. |
