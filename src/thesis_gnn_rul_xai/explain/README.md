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
- `seeds.py` — the Shapley values of `shap.py` for every saved network of a
  stored run, one network at a time, each checked against the test error its
  run recorded before it is explained
  ([Every saved network of a stored run](#every-saved-network-of-a-stored-run)).
- `__init__.py` — re-exports the public names from all seven.

The notebooks in `notebooks/xai/` use these on a stored trained network, one
method per notebook; `03_walks_fd001.ipynb` covers walks and removing a
connection, and step 8 of `01_shap_fd001.ipynb` uses `seeds.py` on the five
seeds of two runs. The playground's Explain tab offers Shapley values,
occlusion and walks, and `scripts/measure_walks.py` measures walks over every
saved seed of a stored run.

## What a Shapley value is

The model gives one number, and fourteen sensors were involved in producing it.
"How much did each sensor contribute?" has no obvious answer, because the
sensors are not independent: removing one changes what the others are worth.

Cooperative game theory gives an answer. Imagine building the group of sensors
one at a time, in some order. Each time a sensor joins, the prediction changes
by some amount, and that change is what the sensor contributed *in that order*.
Do that for every possible order and average. That average is the sensor's
**Shapley value** (Shapley, 1953). Lundberg and Lee (2017, page 4) state that
it is the only way of splitting the credit that meets three conditions at once,
two of them close to these: the shares add up exactly, and a sensor whose
contribution grows never receives a smaller share. They cite Shapley (1953) for
the value itself, but credit that proof of uniqueness to Young (1985), their
reference [10]. Neither original is held in `docs/references/`, so this is
Lundberg and Lee's account, not one checked against those originals.

**Correction, 2026-10-02.** This paragraph stated that uniqueness with
Shapley (1953) as its only source, without naming the conditions. Found when
the cited papers were collected into `docs/references/` and checked against
the places that cite them.

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

### Why the temporal stage runs only once per window

Each of the 16,384 evaluations is the network run on a mixed window: the real
30 readings of the sensors in the group, and for the rest the 30 readings of
the **stand-in**, the average training window that takes the place of a sensor
left out (explained under
[What "removing a sensor" means here](#what-removing-a-sensor-means-here)).
The network's first part, the **temporal stage**, turns each sensor's 30
readings into the number (or numbers) that sensor's graph node carries. Every
stage this project has reads one sensor's readings and nothing else: the last
one, their average, a three-number summary of them, all thirty unchanged, or
that sensor's own recurrent network. So in a mixed window, a kept sensor's
stage output is what it was on the real window, and a replaced sensor's is what
it was on the stand-in. `shap_values` therefore runs the stage on the real
window once, and on the stand-in once per call, and builds the 16,384 mixtures
out of those two results; only the graph layers and the two fully connected
layers after them run 16,384 times. The default stage only picks out the last
reading, so for it this saves little. For the recurrent stage it stops fourteen
recurrent networks from being run 16,384 times over windows of which each sees
only two different versions.

A worked example on real data: the recurrent network of seed 42 in the stored
run `fd001_physics_row_cap_5seed_varsplit_lstm32_cfg72f506_f386291-dirty_20260914T134955Z`,
on held-back engine 5's window ending at cycle 30. What the stage hands two of
the graph nodes:

```
                                              T50 hands over   Nc hands over
real window                                       0.8140          0.2757
stand-in window                                  -0.0085         -0.0234
T50 real, every other sensor a stand-in           0.8140         -0.0234
T50 a stand-in, every other sensor real          -0.0085          0.2757
```

The last two rows are the stage run on the mixed windows themselves, and they
repeat the first two exactly, to the last bit, for all fourteen sensors
(`tests/test_examples_torch.py::test_example_the_temporal_stage_runs_once_per_window_on_the_stored_recurrent_network`).

**It is checked, not assumed.** A stage that read other sensors' readings as
well would make the shortcut wrong. So `shap_values` first runs the whole
network on six of the mixed windows — no sensor kept, every sensor kept, every
other sensor, the rest, the first seven, the last seven — and refuses to return
anything if those six predictions and the shortcut's differ by more than a
thousandth of a cycle. The same check catches `ESTGNN.forward` changing while
the shortcut's copy of its last steps (`_after_stage` in `shap.py`) does not.
`tests/test_shap.py` requires the values to equal those from running the whole
network on all 16,384 mixed windows, for all six temporal stages, and requires
a stage that adds the average of every sensor's reading to each sensor's own to
be refused.

**Do not confuse the method with this shortcut.** The Shapley value is still a
method that needs only the network's inputs and outputs: the numbers are those
of Equation 15 whatever is inside the network. The shortcut belongs to this
implementation, which reads how the network is built (its temporal stage
first, its graph layers after) to reach the same numbers with less work, and
the check above refuses any network for which it would not.

(Added 2026-10-02. Until then every mixed window went through the whole
network, the recurrent networks included. On held-back engine 5 of the stored
recurrent run above, seed 42, the values computed the new way differ by at most
3.6e-15 cycles, which is rounding, from those the old way stored in that run's
`explanations/shap_engine_unit5_seed42_f386291-dirty_20260914T135414Z`, and
equally from the two later copies beside it.)

## What "removing a sensor" means here

Equation 15 needs `f(S)`, "the ESTGNN prediction when only sensors in subset S
are considered". A trained network has no way to accept thirteen inputs; it
always needs fourteen. So a sensor is never actually removed. It is **replaced**
by an uninformative stand-in, and the choice of stand-in decides what the
resulting numbers mean.

The paper does not say what stand-in it uses. We replace an absent sensor's
reading with that sensor's average across the training windows, so the model
sees "this sensor is behaving typically" rather than some arbitrary value
(assumption A-20, in `docs/assumptions/A-20.md`).

**Why zero would be the obvious alternative, and why that is a trap.** The
temptation is to feed a sensor 0 when it is left out. Whether that is safe
depends entirely on the rescaling, not on the baseline:

- Under the **min-max** scaling that was the default until 2026-08-29, zero was
  a sensor's lowest ever reading. Substituting it produced a ranking of the
  fourteen sensors unrelated to the mean-window one — a Spearman rank
  correlation of −0.16, where +1 would mean the two orderings are identical
  and 0 that they are unrelated — and promoted `P15`, the least informative
  sensor, to first place.
- Under the **z-score** default now in force — every sensor rescaled to say how
  many standard deviations it sits from its own training average, so that zero
  means "exactly average" — the number zero is the training mean, and the zero
  baseline agrees closely with the mean window, though not exactly. Over five
  networks trained with seeds 42 to 46 under the default settings, except that
  no training engine was held back, the rank correlation between the two
  orderings is +0.968 ± 0.015, from 0.952 to 0.991, and the same four sensors
  lead under both on every one.

(Corrected 2026-10-03: the second item said the two agree "almost exactly
(Spearman +0.996)". That figure came from one network, trained under the
learning-rate schedule of 2026-08-31, which shrank the rate after every pass
over the training data; the log's five-network measurement under today's
schedule, added that day, replaced it. Found by a review of the five-seed
results.)

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
cycle in turn. An engine that ran for 220 cycles yields 220 − 30 + 1 = 191
windows at a window size of 30, so it yields 191 sets of 14 Shapley values.
`dynamic_shap` returns them for one engine as a table with one row per window,
in the order of the cycle each window ends at. A row holds, in this order:

- that cycle;
- the window's label, the kind of number the network is trained to predict:
  the engine's remaining life there, capped at 125 cycles (assumption A-01),
  so early in a long engine's life it is 125 and not the true remaining life;
- the network's prediction, in cycles;
- the prediction for the stand-in window, the same in every row;
- then one number per sensor, its Shapley value in cycles.

That last number is the sensor's share of the gap between the prediction and
the stand-in's prediction. It is the change the sensor's actual readings make
when they join, averaged over every order in which the fourteen sensors could
join, as "What a Shapley value is" above describes, and the fourteen shares add
up to the whole gap. Reading one sensor's column from top to bottom shows how
its share changed as the engine wore out. The columns are listed by name under
"What each name means", at the end of this file.

**Do not confuse a Shapley value with a single swap.** Putting one sensor's
actual readings into the stand-in window, with every other sensor left at its
stand-in, changes the prediction by some number of cycles; so does taking them
out of the real window, with every other sensor left real. Those are the change
the sensor makes when it joins first and when it joins last, two of the orders
the Shapley value averages over, and they can be far apart, because what one
sensor's readings do depends on which others are already real. Seed 42 of the
stored default run
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z`
(commit `f386291-dirty`, described under "Every saved network of a stored run"
below), held-back engine 5, the window ending at cycle 30:

```
prediction 121.628, stand-in's prediction 91.885, gap +29.743
sum of the fourteen Shapley values            +29.743

T30, joining first: its readings alone into the stand-in window   +19.543
T30, joining last: the other thirteen already real                  +2.132
T30's Shapley value, the average over every order                 +14.061

label in the row: 125. Engine 5 ran 269 cycles, so 239 truly remain.
In 114 of engine 5's 240 windows the label is 125 where more remain.
```

`tests/test_examples_torch.py::test_example_a_dynamic_shap_row_on_the_stored_default_run`
computes every number above.

(Corrected 2026-10-03: this called the table one "of how the model's attention
moved". That was a figure of speech, and a misleading one: the network has no
part that computes weights for where to look, and the table holds predictions
and each sensor's contribution in cycles, nothing else. It was found on
2026-10-02, in a reading of this file while the component notebooks were being
built. The first rewrite, the same day, called the label column the true
remaining life, as the column table at the end of this file had done since it
was written, and described each sensor's number as a single swap rather than an
average over orders; an independent review recomputed both on the window
above.)

Give it the training or validation set. Those keep every window of every
engine. The test set holds only one window per engine, so it cannot show a
trajectory.

## Every saved network of a stored run

**Why one network is not enough.** A trained network is one draw. Training
starts from random numbers and visits the training windows in a random order,
and in this project the seed also decides which 20 engines are held back from
training. A second seed trains a different network, and its Shapley values can
differ, so a figure computed on one network says nothing yet about how far
another seed would move it. Working rule 3 asks for five seeds and their spread
before anything is called a result. `scripts/train.py --seeds 5` saves each
seed's trained network (its `model.pt`) beside the run's results in
`reports/`, so the five networks can be explained afterwards without training
again.

**What `shap_for_each_seed` does, for each seed in turn.**

1. It loads the saved network and rebuilds the data with the settings and the
   seed stored beside it (`load_seed` in `training/stored.py`). The seed
   matters for the data too: it decides which engines were held back, so
   rebuilding with another seed would hand the network engines it learned from
   and call them unseen.
2. It scores the 100 test engines again and compares the error with the one
   the run recorded for that seed. The error is the root-mean-square error in
   cycles: the square root of the average squared miss. If the two differ by
   more than 0.0001 cycles it raises, because either the network or the rebuilt
   data is not what the run trained, and the explanation would describe
   something else.
3. It builds the stand-in from that seed's own training windows
   (`mean_window_baseline`) and predicts on it once. That prediction is the
   **reference** that every window's fourteen values add up from.
4. It explains whole engines with `dynamic_shap`: the engines named in
   `units`, or, when none are named, the held-back engine with the most windows
   (`longest_engine`), which the network never learned from and which shows the
   most of an engine's life.
5. It hands back a `SeedExplanation`, the record of that one seed, and only
   then loads the next network.

It is a **generator**: a function that hands back one result each time a loop
asks for the next, and builds the next one only when asked. So one network at a
time is built and handed over, and at most two are in memory at once, not all
five: while the next seed's network is being loaded, the loop variable still
holds the record of the seed before it. (A loop that stores the records it is
handed keeps those as well.) A loop can also stop after the first seed.

```python
from thesis_gnn_rul_xai.explain import shap_for_each_seed
from thesis_gnn_rul_xai.training import read_run

run = read_run("fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z")
for explained in shap_for_each_seed(run):
    for engine, table in explained.tables.items():
        print(explained.seed, engine, len(table))   # one row per window
```

`split` says which set the engines come from:

- `"validation"`, the default: the held-back engines, every window of each.
- `"train"`: the training engines, every window of each. A run trained with
  nothing held back has an empty validation set, so there pass
  `split="train"` and name the engines. The three engines on which the log
  follows what `P15` contributes over a whole life (finding F-05) would be
  `units=[5, 25, 82]`.
- `"test"`: the test engines, which keep only their last window, so each named
  engine gives a table of one row, and `units` is required. The twenty test
  windows on which the log compares stand-ins for a left-out sensor
  (assumption A-20) are `units=range(1, 21)`.

`units=[]` explains nothing: each network is only loaded and checked, for a
caller that explains other windows, or uses another stand-in, with the `model`
and `prepared` data the record hands back.

**A worked example on real data**: seed 42 of the stored run
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z`
(commit `f386291-dirty`; the default network, whose temporal stage passes on
each sensor's last reading; 20 % of engines held back; at most 200 passes over
the training data, with early stopping after 20 passes without a lower
validation error. Seed 42 stopped after pass 71 and keeps the weights of pass
51, the one with the lowest validation error, so the drop of the learning rate
that the step schedule makes after pass 100 never happened).

```
held-back engines       20; the three longest: 5 (240 windows), 25 (201), 82 (185)
engine explained        5, the longest
reference prediction    91.885 cycles, for the stand-in window

engine 5, first window  ends at cycle 30, label 125 (239 truly left), predicted 121.628
four largest by size    T30 +14.061   P15 +10.703   phi -9.475   P30 -9.235
sum of all fourteen     +29.743  =  121.628 - 91.885
```

The last line is efficiency: the fourteen values add up to the prediction
minus the reference prediction. With `split="test"` and `units=[1, 2, 3]` the
same network gives one row per engine: test engine 1's last window ends at
cycle 31 with 112 cycles truly left, the network predicts 108.619, and the
fourteen values add up to +16.733, the prediction minus the reference
prediction. Raising every seed's recorded error by 0.01 cycles makes the
function refuse seed 42 before it explains anything
(`tests/test_examples_torch.py::test_example_shap_for_each_seed_on_the_stored_default_run`).

(Corrected 2026-10-03: the block above said "true remaining life 125" for
engine 5's first window. 125 is the window's label, the remaining life capped
at 125 cycles (assumption A-01); engine 5 ran 269 cycles, so 239 truly remain.
Found by an independent review that set the label column beside
`train_FD001.txt`.)

**Do not confuse the two checks.** The recorded error asks whether this is the
network the run trained. Efficiency asks whether the arithmetic of the
explanation is right. Passing the second says nothing about the first: an
untrained network's values add up just as exactly.

## Trap: our table and the paper's Table 12 are not comparable

The paper's Table 12 has the same *shape* as what `dynamic_shap` returns — one
engine, sensors down the side, time steps across, though its columns are
labelled `G1`, `G2` and so on instead of by cycle.

The *contents* are not comparable. Our values are signed and measured in flight
cycles. The paper's cannot be: Table 12's contents cannot be signed Shapley
values satisfying the paper's own Equation 14 (finding F-08, in
`docs/findings/F-08.md`). Do not place the two side by side without saying so.

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

## Trap: on the default network, the four largest values change almost every window

**What a reader sees.** The paper's Fig. 10 draws one small graph of the
sensors per time step of one engine, fifteen in all: graphs 1 to 5, 78 to 82
and 187 to 191. In each, the four sensors with the largest values, by size,
are red; call them the **red four**. Table 12 prints the numbers behind the
graphs, and inside each stretch of five the red four hardly changes. The
playground draws the same picture for a stored engine, and step 8 of
`notebooks/xai/01_shap_fd001.ipynb` draws it for this project's networks. On
the default network, whose temporal stage passes on each sensor's most recent
reading (assumption A-07), the red four is different at almost every window. A
sensor that turns red at one cycle and blue at the next has not, on that
evidence, told you anything about the engine at that cycle: for this network it
is the normal case.

**How it is counted.** The paper does not say which window of the engine a
time step is. Table 12 heads its columns "Time", and the paper computes its
dynamic explanation "on each time step" (its Section 2.4, in the paragraph
introducing Equation 16). We read graph *g* as the engine's *g*-th window, the
window ending at its (*g* + 29)-th cycle; that is our choice, not the paper's
statement (assumption A-35), and the playground reads it the same way. Moving
every graph to a later window by the same number of cycles, as far as putting
graph 191 at the engine's last window, still leaves the default network with
more joins than the recurrent one and than Table 12 at every such placement;
that assumption's entry counts them all.

A **join** is a sensor that is in the red four of one window and was not in the
red four of the window before, so one step between neighbouring windows can
have from zero to four. A made-up example: if one window's red four is {Nc,
T24, W31, T30} and the next window's is {Nc, T24, W31, T50}, T50 has joined and
T30 has left, which is one join. The **step-to-size ratio** is taken on
**sizes**, each value with its sign dropped: for one sensor, the average change
in its size from one window to the next, divided by its average size; the table
gives the median over the 14 sensors. Sizes, because Table 12 holds positive
numbers only (finding F-08), so it cannot show a value changing sign, and the
red four is chosen by size anyway; a value going from +2 to −2 adds a step of
0 on sizes and would add 4 on the signed values. And the **stage jitter** is
the same idea one stage earlier, for the number the temporal stage hands the
graph layers: per sensor, its average change between neighbouring windows
divided by its standard deviation over the engine's life, then the median over
the sensors.

**What was measured.** Two stored runs that differ only in the temporal stage:

- `fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z`,
  the default network (`last_cycle`);
- `fd001_physics_row_cap_5seed_varsplit_lstm32_cfg72f506_f386291-dirty_20260914T134955Z`,
  the recurrent network (`lstm`): one recurrent network of hidden size 32 per
  sensor, the stage of the paper's Section 2.2 at a size the paper does not
  give (assumption A-26).

Both were trained at commit `f386291-dirty` on FD001, with the paper's sensor
graph rescaled by rows, test labels capped at 125, a fifth of the training
engines held back (which ones depends on the seed), three graph layers of 32,
64 and 32 numbers per sensor and a fully connected layer of 128, batches of 32
windows, at most 200 passes over the training data with early stopping after 20
passes without a lower validation error, and the weights kept from the pass
with the lowest validation error. Every seed of both runs stopped between
passes 28 and 75, so the step learning-rate schedule they name never took
effect: its one drop of the rate comes after pass 100, and every pass used the
first rate, 0.001.

Their folder names carry two different configuration digests, `ba82e3` and
`72f506`. A configuration digest is a six-character code computed from the
recorded preprocessing settings alone; the temporal stage and the training
settings are not part of it, which is why a change of stage leaves it alone.
When these runs were named, the code also counted a one-sentence description
stored with the settings, and the two differ because the recurrent run,
started from the playground, stored a different sentence: with the default
run's sentence in its place, the recurrent run's preprocessing settings give
`ba82e3`. Since 2026-10-03 the description is left out, and both runs' settings
give `818b40`; the folder names keep the codes they were given
(`training/README.md` lists every old code beside its new one).

For each of the five seeds, the held-back engine with the most windows — the
same engine in both runs, because the seed decides which engines are held back
— was explained at every window with `shap_for_each_seed`. The explaining code
is at commit `6733d77`: step 8 was re-executed on 2026-10-03 from a clean tree
at that commit, and every figure below came out the same as in its first
execution at `19dceb5-dirty`, before `explain/seeds.py` and the change to
`explain/shap.py` that runs the temporal stage once per window were committed.
The notebook was executed twice more that day with uncommitted changes in the
working tree, none of them to those two files: once after the legend of its
step 5 figure was renamed, and once with other work in progress. Step 8 prints
the commit of the latest execution, so it now shows `6733d77-dirty`; every
figure below came out the same each time.
Mean ± standard deviation over the five seeds:

| | Joins over the 12 steps inside graphs 1–5, 78–82, 187–191 | Step-to-size ratio over those 12 steps, on sizes | Share of steps with at least one join, every window | Share of steps with two or more joins, every window | Joins per step, every window | Stage jitter |
|---|---|---|---|---|---|---|
| The paper's Table 12 | 1 | 0.084 | — | — | — | — |
| Recurrent network (`lstm`) | 4.6 ± 2.1 | 0.152 ± 0.059 | 0.364 ± 0.093 | 0.021 ± 0.017 | 0.384 ± 0.110 | 0.119 ± 0.005 |
| Default network (`last_cycle`) | 20.2 ± 4.8 | 0.394 ± 0.107 | 0.917 ± 0.027 | 0.453 ± 0.060 | 1.469 ± 0.119 | 0.281 ± 0.021 |

Table 12 holds only its fifteen graphs, so only the first two measures can be
counted on it. Its one join is `phi` at graph 191, and every seed of both
networks has more than that one. Over every window the step-to-size ratio is
0.365 ± 0.081 on the default network and 0.134 ± 0.035 on the recurrent one.

Read the shares together. On the default network at least one sensor joins at
0.917 ± 0.027 of the steps, but two or more join at 0.453 ± 0.060, and the
median step has one join on seeds 42, 43 and 44 and two on seeds 45 and 46. So
the red four is different at almost every window, but on three of the five
seeds the median step brings only one new sensor into it.

The step-to-size ratio separates the two networks, and the test below compares
only them. Whether the recurrent network's sizes move more than Table 12's is
not tested, and the seeds do not all say so: on seed 42 its ratio over the 12
steps is 0.078, below Table 12's 0.084, and on the other four seeds it is
above.

**It is far larger than the seed-to-seed spread.** The same seed holds back
the same engines in both runs, so the two can be compared seed by seed, as five
differences, default minus recurrent. A paired t-test asks whether those five
differences sit far enough from zero, given how much they vary among
themselves, that seed-to-seed chance is an unlikely explanation; its p is the
probability of differences at least that far from zero if the two stages did
not differ on average. The default network has 15.6 more joins in the 12 steps
(p = 4.9e-04), a join at 0.553 more of its steps (p = 5.3e-05), two or more
joins at 0.433 more of its steps (p = 2.9e-05), 1.085 more joins per step
(p = 2.0e-07) and a step-to-size ratio over the 12 steps higher by 0.242
(p = 1.7e-03), and it is the higher of the two on every seed, on all seven
measures the notebook compares.

**A worked example**, the first four windows of seed 42's default network on
held-back engine 5, each with its red four and its value in cycles:

```
cycle 30   T30 +14.06   P15 +10.70       phi -9.48    P30 -9.24
cycle 31   T30 +13.24   htBleed +11.28   W31 -11.11   Ps30 +10.20   joins: Ps30, W31, htBleed (3)
cycle 32   htBleed +13.89   T30 +11.44   W31 -10.97   W32 -8.69     joins: W32 (1)
cycle 33   T24 +12.95   htBleed +12.90   W31 -11.84   Ps30 +10.00   joins: Ps30, T24 (2)
```

Six joins in three steps, where Table 12 has one in twelve. These windows, the
T50 numbers below and Table 12's one join are checked by
`tests/test_examples_torch.py::test_example_the_red_four_changes_almost_every_window_on_the_stored_default_network`.

**Why, as far as it has been measured.** The graph layers receive one number
per sensor from the temporal stage, and a Shapley value measures how far the
prediction moves when that sensor's readings are swapped for the stand-in's.
The default stage hands over the window's last reading. Neighbouring windows
share 29 of their 30 cycles, but the last cycle is a new one every time, and a
single reading still carries whatever noise the smoothing left in it. The
recurrent stage computes its number from all 30 readings, 29 of which the
previous window also had. For T50 on seed 42's engine 5, the windows ending at
cycles 30 to 34:

```
default network     -1.444  -1.115  -1.266  -1.020  -1.515
recurrent network   +0.814  +0.786  +0.782  +0.756  +0.787
```

This goes with the joins, as the stage jitter in the table shows, but it is
not shown to cause them: the two runs also differ in the whole network trained
on each stage's numbers.

**What it does not settle.** Even the recurrent network's red four changes
more often than Table 12's: 4.6 ± 2.1 joins in the 12 steps against 1, and more
than 1 on every seed. That much depends on the reading of the graph numbers:
of the 447 placements of the graphs counted for the five seeds, ours and
every later one, it has one join or none at 27 (assumption A-35). Table 12
holds positive magnitudes computed in a way the paper never states (finding
F-08), so that remaining gap cannot be put down to the network. Neither network
is the paper's, so none of this is a reproduction of Fig. 10. And seed 42, the
seed the playground's stored explanations and the notebooks follow, gave the
fewest recurrent joins of the five, 2: its picture is the calmest of the five,
not a typical one.

**What to do with it.** Before reading a change of the red four on the default
network as a change in the engine, count the joins on neighbouring windows of
the same engine, and read the values over stretches of windows, as step 6 of
the Shapley-value notebook does by thirds of the engine's life, rather than
from one window to the next.

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
Occlusion keeps that one term at full weight and drops the rest. So the two
agree exactly when that one term equals the weighted average of all of them.
A sensor that adds the same amount whichever others are present always
satisfies this, but it is not the only way. With three inputs,
`h(a, b, c) = a + b + c + 2ab + 2ac + 2bc − 3abc`, input `(1, 1, 1)` and
baseline `(0, 0, 0)`, input `a` adds 1 when it arrives first, 3 when it
arrives after `b` alone or after `c` alone, and 2 when it arrives last. Those
amounts differ, so `a` does interact with the others. Equation 15 weights the
four groups `a` can find on arrival — nobody, `b` alone, `c` alone, both — by
1/3, 1/6, 1/6 and 1/3, so its Shapley value is
`1/3 × 1 + 1/6 × 3 + 1/6 × 3 + 1/3 × 2 = 2`, and its occlusion value, the
last amount alone, is also 2. By symmetry the same holds for `b` and `c`.
Agreement between the two methods therefore does not show that the sensors
act independently. (Corrected 2026-10-02: this said the two agree only for a
sensor that adds the same amount whichever others are present. The review of
the playground's information cards on 2026-09-26 found the overstatement;
`tests/test_examples.py` checks the example.)

Two consequences follow, and neither is about this project's network in
particular:

- **The fourteen occlusion values need not add up** to the prediction minus
  the prediction at the baseline. With `f(a, b) = 2a + 3b + 4ab`, input
  `(1, 1)` and baseline `(0, 0)`, occlusion gives `a` 6 and `b` 7, which sum to
  13 where the whole change is 9: each is measured with the other present, so
  each is credited with the whole `4ab` term. In `h` above they happen to add
  up, 2 + 2 + 2 = 6 = `h(1, 1, 1) − h(0, 0, 0)`. (Corrected 2026-10-02: this
  said the values do not add up, and the `h` example above is a case where
  they do; found by the review of this section after that example was added.)
- **Two sensors carrying the same information can both look worthless.** With
  `g(a, b) = 10 × max(a, b)`, replacing either input alone leaves the other to
  keep the output at 10, so both get 0, where the Shapley values are 5 each.

`notebooks/xai/02_occlusion_fd001.ipynb` measures how far the two methods part
on the stored network, and
`tests/test_examples_torch.py::test_example_occlusion_is_the_shapley_term_where_the_sensor_arrives_last`
checks the one-term relationship on a three-sensor network.

## The figure: contributions over an engine's life

`plot_contributions_over_time` draws what `dynamic_shap` returns. The top panel
is the prediction being explained: each window's label, the remaining life
capped at 125 cycles (assumption A-01), which the legend calls "label:
remaining life, capped"; the predicted remaining life; the prediction at the
baseline; and a line at the cycle where the label first drops below its
ceiling. Below it is one small panel per sensor,
holding that sensor's contribution at every window, ordered by average size with
the largest first, and all on **one vertical scale**, so that a tall panel is a
large contribution and not just a large axis.

(Corrected 2026-10-03: the legend itself said "true remaining life" after this
section and the column table below had been corrected. Early in a long life the
line sits below the life the engine truly had. Engine 2 of the training file
ran 287 cycles; with windows of 20 cycles its first window ends with 267 cycles
truly left and a label of 125, and the line is below the true remaining life in
142 of its 268 windows. The legend was renamed, and
`tests/test_shap.py::test_the_figure_names_its_grey_line_the_capped_label`
checks the name and those numbers. Found by a review of the wording around the
capped label.)

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
  last (`linspace(3, 0, depth)`).
- **Gamma 0 in the layers after the graph**, `fc1` and `fc2`. Their synthetic
  network has no such layers, and their paper says what such layers get only
  once, for its image model VGG-16: gamma = 0 in the top-level classifier (its
  Section 5.3). That one case is followed here; it is a choice, not a rule the
  paper states for every network.
- **Gamma 0 at the first graph layer.** The rule's derivation assumes the
  numbers entering a layer are never negative (their appendix C), and where a
  network lacks that structure — a ReLU, then a weighted sum, then a ReLU — the
  authors use gamma = 0 themselves, as for their SchNet model. This network's
  first graph layer receives rescaled readings, which are negative whenever a
  sensor is below its training average; there, a positive gamma pushes a
  negative contribution further the wrong way and can divide by a total of the
  wrong sign. `walk_relevance` refuses a positive gamma at the first layer.
  **On most networks here this changes nothing.** When the temporal stage hands
  the graph one number per sensor — the last cycle (the default), the window
  average, and the recurrent stage at its default size — every unit of the
  first graph layer receives all its inputs through one and the same weight, so
  the extra (1 + gamma) multiplies the top and the bottom of each share alike
  and cancels. On the five stored runs, every seed, the last window of every
  test engine, giving the first layer the authors' 3 instead of 0 changes no
  walk at all on the default, correlation-graph, complete-graph and recurrent
  networks; on the three-number window summary it moves the walks by a median
  of 106 ± 42 % of their total size (mean and spread over the five seeds of
  each seed's median, stored with the run's walk measurements on 2026-10-02;
  this said 103 ± 40 % until then, a figure measured before the measure was
  stored, as assumption A-29 records). The choice matters only there, and for
  the raw window (`first_layer_gamma_change` in `scripts/measure_walks.py`; the
  cancellation is `test_gamma_at_the_first_layer_cancels_with_one_number_per_sensor`).

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
see how far a conclusion depends on gamma. `scripts/explain.py` and the
playground's Explain tab offer three by name beside the authors' (the
script's `--gamma`): `gnn-gi`, gamma 0 at every layer; `falling`, 0 and then
0.5 halving at each later layer, (0, 0.5, 0.25) for three; and `flat`, 0 and
then 1, (0, 1, 1) for three. The last two, for three graph layers, are the
settings the log's walk measurements compare with the authors' (assumption
A-29).
Every name gives the first graph layer 0, and the script always checks the
authors' setting and their baseline beside the one chosen. The playground's
README, under "Explanations", says how the page offers them.

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
rescaled again. The paper runs its networks on the smaller graph (its Fig. 6
compares the output before and after pruning) but never says how a graph that
was rescaled loses a node, and the authors' published code holds no step that
runs a network with nodes removed. The one place that code masks a graph, for
GNNExplainer, it multiplies the mask into the graph it has already rescaled,
without rescaling again, which is what is done here. Its readings stay in the window, but with no connection left nothing of them
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
cover this task. The only comparison this project found that includes GNN-LRP
and was not run by the authors of GNN-LRP, GNNExplainer or PGExplainer is a
survey by Yuan et al. (2023). It is not free of an interest of its own: three
of its four authors wrote SubgraphX (the survey's reference [48], page 18),
another method it compares, which comes first in ten of the twelve columns of
the two tables described below. It tests classification only: of whole graphs
(sentences turned into graphs, molecules, and synthetic graphs built around
known shapes) and of single nodes in a synthetic graph (pages 10 and 11).

Its two measures are defined on pages 11 and 12; it calls them Fidelity+ and
Fidelity−. The first removes the parts of each graph that an explanation marks
as most important (connections, or nodes) and records how far the network's
probability for the category it originally predicted falls; a larger fall is
better. The reverse measure keeps only the marked parts, and there a smaller
fall is better. Its main comparison (Section 8.1) plots both against the share
of each graph marked important, from about a fifth to a half (its Figs. 6 and
7, page 13). Its text on page 14 says the two walk-based methods — it calls
them flow-based: GNN-LRP and its simpler form GNN-GI — "generally" score higher
on the first measure than GNNExplainer and PGExplainer, and, after SubgraphX,
do better than the rest on the reverse measure too.

The survey's own numbers bear that out for GNN-GI and not for GNN-LRP. Its
Tables 3 and 4 (page 14) belong to a separate check against connections given
random scores (Section 8.2, page 15), at one fixed share marked important:
30 %, or 40 % on BA-Shapes. On the four datasets that are not synthetic —
three sets of sentences and the molecule set BBBP — GNN-GI is ahead of both
GNNExplainer and PGExplainer on both measures. GNN-LRP is behind PGExplainer
on the first measure on Graph-SST2 (0.1363 against 0.1889), Graph-SST5 (0.1813
against 0.1854) and BBBP (0.0860 against 0.1464), and behind GNNExplainer on
BBBP (0.1057); its BBBP score is also below the random scoring's 0.1212, which
the table marks with a star. On the reverse measure it is behind PGExplainer on
five of the six datasets, by 0.0563 to 0.4087, all four that are not synthetic
among them, and behind GNNExplainer on two; Fig. 7 shows GNN-LRP behind
PGExplainer at every plotted share on the same five. It is ahead of both on
both measures only on BA-Shapes, the one task of classifying single nodes. The
survey's advice (pages 14 and 15) is SubgraphX in every case, GNNExplainer and
PGExplainer only for classifying single nodes, and "GNN-GI and GNN-LRP" only
for real-world datasets, although on the reverse measure its numbers put
GNN-LRP behind PGExplainer on every one of those. The copy in
`docs/references/` is the arXiv preprint, whose page numbers match the PDF's
but not the journal's; the journal version is not held. None of it tests a
regression on a sensor graph.

Until 2026-10-02 this paragraph called the survey independent, which it is not
where SubgraphX is concerned; dropped its "generally" and let the walk-based
methods' result stand for GNN-LRP, which the survey's own tables do not bear
out; dated it 2022; and added that other comparisons, which it did not name,
found GNNExplainer the strongest on classifying single nodes and PGExplainer
weak. None of those comparisons is held here to check, so that sentence is
withdrawn. The errors were found when the cited papers were collected into
`docs/references/` and checked against the places that cite them.

What GNN-LRP gives up in exchange: it is specific to networks it has rules for,
its cost grows as the number of sensors raised to the number of layers plus
one (38,416 per window here, which is small only because there are 14 sensors
and three layers), it explains the score before the sigmoid rather than the
prediction in cycles, and its shares depend on a setting of the rule, gamma:
the default here follows the authors' own (assumption A-29), and at gamma 0
everywhere the method becomes the simpler GNN-GI, which they report as
noisier.

(Corrected 2026-10-02: the last clause said that at the default setting the
method is the noisier variant. That was true while the default was gamma 0,
and stopped being true when the default became the authors' own setting of 0,
1.5 and 0. Found on 2026-10-02 by the review of the survey paragraph above,
while reading the text around it.)

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
  window over the training set (assumption A-20, in `docs/assumptions/A-20.md`).
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
F-08, in `docs/findings/F-08.md`).

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
- Yuan, H., Yu, H., Gui, S., & Ji, S. (2023). Explainability in graph neural
  networks: A taxonomic survey. *IEEE Transactions on Pattern Analysis and
  Machine Intelligence*, 45(5), 5782-5799.
  https://doi.org/10.1109/TPAMI.2022.3204236 — the benchmark described above
  was read in the arXiv version, https://arxiv.org/abs/2012.15445 (version 3),
  the copy held in `docs/references/`. This entry gave the year as 2022, with
  no volume or pages, until 2026-10-02: that was the early-access record. It
  was found when the cited papers were collected into `docs/references/` and
  checked against the places that cite them.

## What each name means

Everything the files listed at the top offer.

| Name | What it is |
|---|---|
| `mean_window_baseline(dataset)` | The average reading of each sensor, which stands in for a sensor left out of the calculation. **Training windows only.** |
| `shap_values(model, features, baseline)` | How many cycles each sensor added to the prediction. Takes one window or many, and returns one number per sensor, or one row per window. Raises above `MAX_EXACT_NODES` sensors rather than silently switching to an estimate. |
| `dynamic_shap(model, dataset, baseline, unit=..., symbols=...)` | The same calculation at every cycle of one engine's life. Returns one row per cycle; the columns are listed below. |
| `shap_for_each_seed(run, units=None, split="validation", seeds=None, data_dir=None, device=None)` | From `seeds.py`. For each saved seed of a stored run (a `StoredRun`, the record `read_run` makes of a finished training run in `reports/`), one at a time: loads the network, refuses it unless it scores the test error its run recorded, and explains whole engines with `dynamic_shap`. Hands back one `SeedExplanation` per seed. `split` is `"validation"`, `"train"` or `"test"`; `units` left out means the set's longest engine, and must be given for `"test"`; `units=[]` only loads and checks. |
| `SeedExplanation` | From `seeds.py`. One seed's record: `seed`; `model`, the loaded network; `prepared`, the data rebuilt for it; `baseline`, the stand-in window, shaped `(14, 30)`; `reference`, the prediction for that window, in cycles; `tables`, each explained engine's number mapped to its `dynamic_shap` table. |
| `longest_engine(dataset)` | From `seeds.py`. The engine with the most windows in a set, the lowest-numbered on a tie. Refuses an empty set. |
| `SPLITS` | From `seeds.py`, and only there: `__init__.py` does not re-export it. The three sets `shap_for_each_seed` explains engines from. |
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
| `rul` | The window's label: the remaining life at that cycle, capped at 125 cycles (assumption A-01). Early in a long engine's life it is 125 while more truly remain. |
| `predicted` | What the model predicted there, in cycles. |
| `baseline` | What the model predicts for a wholly typical engine. Constant down the column. |
| one per sensor | That sensor's contribution at that cycle, in cycles. |

(Corrected 2026-10-03: the `rul` row said "the true remaining life at that
cycle", and the figure section above called the same line the true remaining
life. The column is `dataset.targets`, the capped label; for held-back engine 5
of seed 42 of the stored default run it differs from the true remaining life in
114 of 240 windows. Found by an independent review of the dynamic-explanation
section.)

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
