# Response to the audit of 2026-09-08

Written on 2026-10-02.

## What this file is

`docs/AUDIT_2026-09-08_scientific_validity.md` is an independent audit of this
repository, run from outside it. It asked whether the code does what Wen et al.
(2026) describe, and whether the numbers this project reports can carry the
meaning they are given. It reported eleven defects, ranked by what each does
to a conclusion and numbered in that order (D-1 to D-11). The audit is a dated
record and is not rewritten.

This file is the project's answer, one section per item: what the audit said,
in plain words; what was done about it and where to read it; or why it does not
apply; and what is still open. The log of choices and findings,
`docs/ASSUMPTIONS.md`, numbers its entries: an assumption is a choice this
project made where the paper is silent (A-01 onwards), a finding is a fact
about the paper or the data established by measurement (F-01 onwards). Each
entry is named below by what it is about, with its number after it.

**Where the figures come from.** Every number below was recomputed on
2026-10-02 from a training run stored in `reports/`, named beside it with the
commit that produced it, or from the code and data named beside it. No network
was trained for this file. The stored runs are not committed, so a clone
without them cannot recompute these figures; the code under
[Reproduce with](#reproduce-with) recomputes each one, or names the log entry
whose own code does.

**Words used throughout.**

- *Test engines*: the 100 FD001 engines whose true remaining life NASA
  publishes in `RUL_FD001.txt`. Every reported error is measured on them.
- *Error*, or *RMSE*: the root mean square error in cycles, the square root of
  the average squared difference between predicted and true remaining life.
  Lower is better.
- *Held-back engines*: since 2026-09-08 the default sets a fifth of the 100
  training engines aside before training (assumption A-06). The network never
  learns from them; their error decides when training stops and which pass over
  the training data is kept (assumption A-18). That error is the one a choice
  between two settings should rest on, because the test engines are then still
  unseen when the choice is made. The two kinds of error weight engines
  differently and are never compared with each other.
- *Seed*: the number that fixes the network's random starting values, the order
  of the training examples and which engines are held back. Every result is a
  mean over five seeds with its standard deviation.
- *Paired test across seeds*: two settings trained with the same five seeds,
  the difference taken seed by seed, and a t-test asking whether those five
  differences could have come from nothing. It cannot say how a difference would
  come out on engines outside this data (item D-8 below).
- *Min-max and z-score scaling*: two ways of putting the fourteen sensors on
  one scale before training. Min-max maps each sensor's training range onto 0
  to 1; z-score subtracts each sensor's training mean and divides by its
  standard deviation. The paper describes neither (assumption A-03).
- *Row and symmetric rescaling*: two ways of rescaling the paper's table of
  sensor connections before the graph layers use it. Both first join each
  sensor to itself. Row rescaling then divides each sensor's row by its number
  of connections, itself included, so each sensor takes a plain average of
  itself and its neighbours; symmetric rescaling divides each entry by the
  square root of both sensors' numbers of connections. The paper prints neither
  formula (assumption A-21).
- *Learning-rate schedule*: how the learning rate, the setting that scales
  each change to the weights, moves from the paper's 0.001 to its 0.0001: one
  drop by ten at the halfway pass, the default, or a smooth exponential decay,
  the superseded option (assumption A-17).
- *The recurrent-stage reference*: the configuration at the centre of the
  sensitivity study stored at `reports/sensitivity_fd65a3b-dirty_20260929T101059Z`
  (finding F-19, at `fd65a3b-dirty`), which has the paper's per-sensor
  recurrent stage and otherwise the default settings. The study changes one
  setting at a time around it.

---

## The eleven items at a glance

| Item | What the audit said | State on 2026-10-02 |
|---|---|---|
| D-1 | The paper's graph has rank 8, so the network sees eight numbers, not fourteen | Recorded as a finding (F-18) on 2026-09-26; its consequence for Shapley values added to it today |
| D-2 | The "every sensor joined" control averages all sensors into one number | Measured and added to the entry on that control (A-19) today; random-graph controls never run here |
| D-3 | Choices between settings were decided on the test engines | Held-back error set beside test error in every entry that made such a choice; the choice of row rescaling does not survive it |
| D-4 | Two fields of each run's record describe an experiment that did not happen | One fixed on 2026-09-14; one still open, in a file that cannot change today |
| D-5 | Nothing compares the network with an ordinary model | A comparison script is in progress; no figure quoted until it runs from a clean commit |
| D-6 | The default network is barely temporal | Confirmed and added to the entry on what each sensor node carries (A-07) today, with the measurement of the window average it asked for |
| D-7 | The graph layers can only scale with their input | Recorded today as a finding (F-21) |
| D-8 | Seed spread is not uncertainty over engines | Interval over engines added to the comparison with the paper's figures (F-06) and to `README.md` today; caution added beside the paired tests |
| D-9 | Stale numbers presented as current | All three fixed; six more of the same kind found and fixed; two files still to correct (both corrected on 2026-10-03) |
| D-10 | Per-pass training curves were thrown away | Saved by every run since 2026-09-14; the curves of the recurrent stage show the over-training the log had only inferred |
| D-11 | An ASUE of 0 is called the best value; it is undefined | Agreed; not yet changed in the code |

---

## D-1. The paper's graph cannot tell three groups of sensors apart

**What the audit said.** After adding each sensor's link to itself and
rescaling, three groups of sensors have identical rows in the paper's graph:
{P15, Nf, NRf, BPR}, {P30, Ps30} and {Nc, W31}. A graph layer gives each sensor
a mix of its row of neighbours, so the members of a group carry identical
numbers from the first layer on, and the network reads their readings only as
a group total. The rescaled graph has rank 8: only eight of its fourteen rows
cannot be built from the others. The audit drew three consequences. First,
inside a group, Shapley values (the share of one prediction the explanation
method gives each sensor, found by comparing its reading with a stand-in, the
average training window) measure how far each reading sits from its stand-in,
not how the network uses that sensor. Second, the paper's Table 12 gives
different importance to sensors its model cannot separate. Third, the same
holds for the paper's full model.

**What was done.** The groups and the rank were written up on 2026-09-26 as
the finding that the paper's graph cannot tell three groups of sensors apart
(F-18), with a consequence the audit did not state: training never separates
the weights the first dense layer (the ordinary layer after the graph layers,
in which every input number has its own weight) gives to the members of a
group, so whatever is read off the split inside a group at that layer is partly
the random start.
`scripts/verify_claims.py --only same_rows` recomputes the groups and the 8,
and `tests/test_walks.py` checks them.

The audit's first consequence was not in that finding and was added to it
today, measured on seed 42 of the default stored run
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z`.
Swapping how far P15's and Nf's readings sit from their stand-ins leaves engine
1's prediction exactly where it was, 108.618858 cycles, and swaps their Shapley
values exactly (−0.427663 and +0.844460). Over the 100 test engines the four
members of the large group rank the same by the mean size of their Shapley
value as by the mean distance of their reading from its stand-in, and on 97 of
the 100 engines the order is the same engine by engine. So the audit is right
that the split inside a group follows the readings; its word "forced" fits the
mechanism, not the exact order, which held on 97 engines and not 100.
`tests/test_examples_torch.py::test_example_swapping_two_group_members_swaps_their_shapley_values`
shows the swap on a real test window, and the finding's own code recomputes
the figures.

The audit's sixth blind direction was checked with the code under
[Reproduce with](#reproduce-with): under row rescaling, the direction left over
once the five differences between group members are taken out of the graph's
six blind directions is `T30 + htBleed − 0.5 (P30 + Ps30 + Nc + W31)`, as the
audit says. Under symmetric rescaling the coefficients differ (0.943 on htBleed
and 0.471 on P30 and Ps30). The finding records the count of eight and not this
direction.

**Where it does not apply as stated.** The second and third consequences
concern the paper's full model, which has a recurrent stage: a small network
per sensor that turns each sensor's 30 readings into the one number the graph
receives. Each sensor has its own, so two members of a group can be treated
differently before the graph, and the group total the graph sees is a total of
learned numbers, not of readings. The paper's model is still a function of
eight numbers at the graph, but Table 12's different values for, say, Nc and
W31 are not contradicted by the rank argument alone. This is an argument; it
has not been measured on the recurrent stage. The finding's addendum says so.

**Not done.** The audit also suggested computing the rank and the groups when
`graph/adjacency.py` is imported. They are computed on demand instead, by
`indistinguishable_sensors` in `explain/walks.py` and by the check above, which
is enough to keep the fact visible without slowing every import.

---

## D-2. The "without physics prior" control averages every sensor into one number

**What the audit said.** Joining every sensor to every other and rescaling
gives a table in which every entry is 1/14, so every sensor receives the plain
average of all fourteen. That removes thirteen of the fourteen numbers, not
just the engineering knowledge, so the cost it measures is the cost of
averaging. The right control keeps the number of edges: random graphs with 41
edges, and rewirings of the paper's graph that keep each sensor's number of
connections.

**What was done.** On 2026-09-14 the entry on this control (A-19) gained a
correction saying this, and the entry on the data-driven graph (A-27) measured
a graph built from the data with the same 41 edges, the 41 most strongly
correlated sensor pairs: it costs 0.096 cycles of test error (p = 0.655) where
the every-sensor graph costs 2.784, under the same default settings.

Today the averaging itself was measured and added to the entry on the control.
Under both rescalings the every-sensor table has rank 1 and one distinct row,
against rank 8 and nine distinct rows for the paper's graph; the readings
1, 2, …, 14 come out as 7.5 for every sensor. In seed 42 of the stored run
`fd001_complete_row_cap_5seed_varsplit_cfgf21547_f386291-dirty_20260914T150527Z`,
every sensor holds exactly the same numbers after each of the three graph
layers, and shuffling which sensor's window goes where, or moving 1.0 of
rescaled reading from one sensor to another, changes no prediction by more
than 1.53e-05 cycles, the rounding of 32-bit arithmetic.
`tests/test_examples.py::test_example_the_every_sensor_graph_hands_every_sensor_the_same_average`
checks the table on engine 1's real readings.

Two things came out that the audit did not have. Around the recurrent-stage
reference, the every-sensor graph costs nothing measurable: +0.402 on test
error (p = 0.372), +0.112 on held-back error (p = 0.221). And on the held-back
engines, under the old min-max scaling where its test cost was zero, the
every-sensor graph came out worse, by 0.539 cycles on four seeds of five
(p = 0.040). That figure is one of several tests made in the same note, and
the held-back error it rests on flatters by an amount those runs cannot show,
so the entry on the every-sensor graph (F-04) records it as a direction that
did not reverse, not as a result of its own (item D-3 below).

**Not done.** The random and rewired graphs have never been run in this
repository. `scripts/train.py --graph` offers the paper's graph, the
every-sensor graph and the correlation graph. The audit's three-seed figures
for random graphs were measured outside the repository and name no commit
here, so they are not quoted as this project's results. Running them needs a
new graph option, in `config.py` and `scripts/train.py`, and **new training**.

---

## D-3. Choices between settings were decided on the test engines

**What the audit said.** From 2026-08-29 the default held no engines back, so
every comparison between two settings was decided by looking at the test
engines, which the entry recording the selection rule (A-22) itself forbids.
The results are not wrong, but a figure from a setting chosen that way is
flattered by an unmeasured amount.

**What was done.** The default has held a fifth of the engines back since
2026-09-08 (assumption A-06), so choices made since then can rest on held-back
error, and the entries on the data-driven graph and the window-summary stage
(A-27 and A-28) already set it beside test error. Today every earlier entry
that chose between settings, or compared two, was given the held-back error of
its stored runs where they had one. A run that held nothing back has none, and
those comparisons cannot be judged this way.

| Choice | How it was made | What held-back error says | Same choice? |
|---|---|---|---|
| Z-score scaling over min-max (assumption A-03) | Test error, under the selection rule (assumption A-22) | Z-score lower by 2.038 (p = 0.017), all five seeds, in the configuration it was chosen in: the last reading, symmetric rescaling, the exponential schedule, at `09628fa` | **Yes** for that network. Around the recurrent-stage reference min-max is lower by 1.234 (p = 0.00286, all five seeds), so there it would reverse |
| Row rescaling over symmetric (assumption A-21) | Test error with nothing held back | Under z-score with engines held back, no difference in any of the five configurations stored (four at `f4f40ad`, and the recurrent-stage reference), and no consistent direction: symmetric nominally lower in four, row in one. On the default's settings +0.045 (p = 0.311) | **No**: a tie. Only under the old min-max scaling does held-back error favour row (−0.226, p = 0.0103, all five seeds) |
| One step in the learning rate over a smooth decay (assumption A-17) | An argument, not a difference | No difference: −0.014 (p = 0.758) with the last reading, both runs at `f4f40ad`; +0.019 (p = 0.931) around the recurrent-stage reference | Yes, because it never rested on a difference |
| Keeping the smoothing (assumption A-04) | The paper describes the smoothing (its Equation 18); its 0.5 is the entry's reading of the text | With the last reading, per-condition min-max scaling, symmetric rescaling and the exponential schedule, at `09628fa`: turning it off is worse by 1.866 (p = 0.0002), as on test error (+1.219, p = 0.007). Around the recurrent-stage reference neither error tells on from off: −0.044 held back (p = 0.749), −0.070 on test (p = 0.804) | Yes; with the recurrent stage there is nothing measured to choose between |
| Every-sensor graph as the control (finding F-04) | Not a choice of default | Worse by 1.447 (p = 0.0041, all five seeds) on the default; under min-max worse by 0.539 (p = 0.040, four seeds of five) where the test engines showed nothing; around the recurrent-stage reference +0.112 (p = 0.221) | Same direction; the "zero" in that entry's history is a test-engine result, and the min-max figure is one of several tests |

The figures for the input scaling, the graph rescaling and the smoothing come
from each entry's own stored runs and the input-scaling finding's sweep
(F-10); the recurrent-stage figures are the sensitivity study's. Each entry
names its runs and adds two cautions: the held-back error a run records is the
lowest reached over all passes and also chose which pass to keep, so it
flatters, by an amount that can differ between the two settings; and the paired
tests say nothing about engines outside this data (item D-8). The notes are
dated 2026-10-02 in the entries on input scaling, the learning-rate schedule
and graph rescaling (A-03, A-17 and A-21), and in the findings on the
every-sensor graph and the input scaling (F-04 and F-10). The selection rule's
own entry (A-22) now points to them.

Setting the held-back errors beside the input-scaling sweep (F-10) found an
error in it: its row for a label ceiling of 130 also caps the test and
held-back answers at 130 instead of 125, so it is scored against a different
answer key, and its difference from the baseline mixes a change in the network
with a change in the answers. That entry now says so; the stored run saved no
predictions, so it cannot be rescored.

The entry on graph rescaling (A-21) also turned out to have set, in its table
row for engines held back with early stopping off, an early-stopping run of
row rescaling beside a symmetric run without it, because no run of row
rescaling under those settings had been stored. All eight runs, both
rescalings under both schedules with and without early stopping, were trained
again on 2026-10-02 from a clean checkout of `f4f40ad`, and that entry's table
and its held-back note now use them. Held-back error, row minus symmetric:

| Schedule | With early stopping | Without early stopping |
|---|---|---|
| one step (the default's) | +0.045, p = 0.311 | −0.014, p = 0.565 |
| exponential (superseded) | +0.120, p = 0.376 | +0.050, p = 0.178 |

No difference, and no consistent direction. The conclusion, a tie, stands.
With engines held back, "without early stopping" still keeps the pass with the
lowest held-back error (assumption A-18), so these four differ in how many
passes were run, not in how the kept pass was chosen.

**What the audit asked for that is no longer needed.** It asked for the
comparisons to be re-run under `config/fd001_validated.yaml`. That file held
engines back while the default did not; since 2026-09-08 the default does, the
two files became identical, and the separate one was deleted on 2026-09-14
(assumption A-06).

---

## D-4. Two fields of each run's record describe an experiment that did not happen

**What the audit said.** Every run's `summary.json` said `"split":
"varies_with_seed"` and `"early_stopping": true`, and every directory name said
`varsplit`, including runs that held nothing back, where there is no split to
vary and nothing for early stopping to watch. `scripts/train.py` took both from
command-line options rather than from the settings actually used.

**What was done.** Commit `c8ba1f1` of 2026-09-14 derives `early_stopping` from
whether engines were held back, and adds `validation_fraction` and
`reported_weights` (the best held-back pass, or the last pass) to every
`summary.json`.

**Still open.** `split` and the `varsplit` in the directory name still come
from the `--fixed-split` option alone (`scripts/train.py`, where the name is
built and where `summary.json` is written), so a run that holds nothing back
would still be recorded as `varies_with_seed`. No such run has been stored since
the fix: the last one, from 2026-09-09, also predates it. Runs stored before
`c8ba1f1` keep their old fields, because results files are never edited by
hand (working rule 3a); for them `preprocessing_config.yaml`'s
`validation_fraction` is the field to trust. The fix belongs in
`scripts/train.py`, which cannot change while the training study that launches
runs from this working tree is running. It must keep stored runs loadable:
`training/stored.py` reads `split` to decide whether to rebuild each seed's
held-back engines.

---

## D-5. Nothing compares the network with an ordinary model

**What the audit said.** The first question an examiner will ask is whether the
graph buys anything over an ordinary regression model given the same fourteen
numbers. Its own scikit-learn models, run outside the repository, came within
0.12 cycles of the network on test error.

**What was done.** A scratch measurement on 2026-09-14 found the same picture
and more: an ordinary model given each sensor's recent trend beat the default
network. Its numbers came from a script outside the repository, so they are not
quoted here (working rule 3a). A script, `scripts/baselines.py`, is being
written to fit that ladder of models from a stored run's own split; on
2026-10-02 it exists in the working tree and is not yet committed.

**Still open.** Its measurement has to be made from a clean commit before any
figure from it is quoted or added to `README.md` section 8. It fits ordinary
models and trains no network. One trap for anyone re-running the audit's own
procedure: it used `features[:, :, 0]`, which was the last cycle when the
pipeline handed over one cycle per sensor and has been the first cycle of the
window, 29 cycles out of date, since 2026-09-09. The last cycle is
`features[:, :, -1]`.

---

## D-6. The default network is barely temporal

**What the audit said.** With a smoothing weight of 0.5, almost all the weight
of the smoothed value sits on the last six readings, and the number each sensor
node receives is nearly the raw reading at that cycle. So the default is a
per-cycle regression on lightly smoothed readings, and "spatiotemporal" and "a
30-cycle window" should not describe it without saying so. It also said the
average of the window, the other one-number option, had never been measured.

**What was done.** Both are confirmed and added to the entry on what each
sensor node carries (A-07). The weights are 0.5, 0.25, 0.125 and so on, 0.984
on the last six readings together; on the 17,731 rows of the 100 training
engines that can end a 30-cycle window, the raw reading alone accounts for
between 0.750 (P15) and 0.990 (Nc) of the smoothed value's spread, the audit's
figures exactly. The average of the window had in fact been trained, in the
sensitivity study of 2026-09-29, though that entry still said it had not. With
every other setting at the default and the last reading in place of the
recurrent stage, the average is worse than the last reading by 5.744 cycles of
test error (22.434 ± 0.516 against 16.690 ± 0.229, p = 0.000019, all five
seeds), and the two cannot be told apart on held-back error (+0.583,
p = 0.399). The entry now says both, and that "spatiotemporal" belongs to the
paper's model. The window-summary stage (assumption A-28), which hands each
sensor its last reading, its average and its change, is the measured way of
giving the network the window's trend.

The audit's further point, that the paper's "without LSTM" row is not a clean
comparison for this configuration, is cited in the finding on what the
recurrent stage is worth (F-16).

---

## D-7. The graph layers can only scale with their input

**What the audit said.** With no offset in the graph layers, that is no
learned constant added after the multiplication by the weights (assumption
A-12), and one number per sensor, the three graph layers can only compute a
function that scales with its input: multiplying the input by a positive
number multiplies the output by the same number. After the first layer each
sensor's 32 numbers lie on one of two fixed directions. It is a faithful
reading of the paper, so not an error, but worth an entry, and adding an offset
had never been tried.

**What was done.** Recorded today as a finding about the paper's printed
architecture (F-21), measured on seed 42 of the default stored run named in
the section on the indistinguishable sensors (item D-1): scaling the input by
0.5 or 2 scales the graph stage's output exactly, by 10 to within 1.4e-06, and
by −1 not at all (0.91 apart); after the first layer all 1,400 pairs of an
engine and a sensor lie exactly on one of the two directions.
`tests/test_examples_torch.py::test_example_the_graph_layers_scale_with_their_input_and_keep_two_directions`
checks both properties on real test windows.

The two properties have different causes, and the finding now keeps them
apart. Having no offset is what makes the layers scale with their input and
puts the 32 numbers on two fixed directions; an offset would end both. That
the first layer works on one number per sensor comes from Table 4's input
shape, and an offset would not change it: the 32 numbers would still all be
worked out from that one number, only no longer along a fixed direction.

The entry on the missing offset (A-12) now points to the finding, and records
the one time an offset was trained: around the recurrent-stage reference, with
no difference five seeds can resolve (+0.282 on test error, p = 0.648; −0.253
on held-back error, p = 0.105).

**Done 2026-10-03.** Until this date this said an offset had never been
trained with the default's last-reading stage. It has now, five seeds from a
clean commit: +0.239 cycles on test error (p = 0.185) and +0.297 on held-back
error, higher on all five seeds (p = 0.025). The figures, runs and commits are
under finding F-21.

---

## D-8. Seed spread is not uncertainty over engines

**What the audit said.** The five seeds are all scored on the same 100 test
engines, so their spread measures how much the result depends on the random
start, not how much it would move on other engines. A paired test across seeds
answers only whether a difference could be an accident of the start. The gap to
the paper's 14.93 lies inside the uncertainty over engines, so it cannot be
called an established shortfall.

**What was done.** The stored test predictions of the current default,
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z`,
were resampled: 100 engines drawn from the 100 with repeats, 20,000 times, the
five-seed mean recomputed each time. The 95 % interval is 13.98 to 18.91
cycles, with a standard deviation of 1.262, three times the seed spread of
0.415 (the audit's "nine times" was for one network of the configuration that
held nothing back, whose seed spread was smaller). The paper's 14.93 lies inside
it, and 11.7 % of the resampled means fall at or below it. The entry comparing
this code with the paper's figures (F-06) and `README.md` section 8 now say that
the 1.57-cycle gap is not a difference this test set can establish. The same
caution about paired tests now stands beside the comparisons added for the
objection about choices made on the test engines (item D-3), in the entries on
the every-sensor graph, input scaling and graph rescaling (F-04, A-03 and
A-21); the entry on the recurrent stage (F-16) already carried it, and the
playground's comparison page says it under every comparison.

**Not done.** It was not added beside every p-value in the log, among them the
older tables of the entries on capping the test labels and on the learning-rate
schedule (A-02 and A-17). Uncertainty over engines can only be computed for
runs that saved their predictions, which runs have done since 2026-09-14; the
entry on the window-summary stage (A-28) is the other one that reports it.

---

## D-9. Stale numbers presented as current

**What the audit said.** Three places: `README.md` section 8 gave the
shortfall to the paper as 5.4 cycles, from a superseded configuration; the
cost of capping the test labels at 125 (assumption A-02) had been measured only
on a superseded configuration; and the comparison with the paper's figures
(finding F-06) still said "61,729 parameters — what Table 4's count exactly", a
wording an earlier audit meant to remove.

**What was done.** All three are fixed. `README.md` section 8 has given the
current default's 16.50 and a gap of 1.57 since 2026-09-14. The cost of capping
was re-measured on today's default on 2026-10-02: capping lowers test error by
1.508 cycles (p = 5.0e-06), from 18.005 to 16.496, at `9856794-dirty`. The
wording about 61,729 was corrected: the network's 61,729 numbers are Table 4's
weight shapes, 61,600, plus the two dense-layer offsets of Equations 12 and 13,
129, with none in the graph layers.

Six more of the same kind were found and corrected on 2026-10-02, each with a
dated note:

- the comparison with the paper's figures said the reproduction "now reports"
  16.818, the figure while nothing was held back;
- its opening table still called the configuration of 2026-08-28, at 20.353,
  "current" and "the default";
- `README.md` section 8 said 61,729 is "exactly what Table 4's printed shapes
  imply", when the shapes alone give 61,600;
- the opening summary of `README.md` said the same, "the 61,729 parameters its
  printed shapes imply";
- `README.md` and `docs/learn/06_reading_results.md` gave only the old
  1.31-cycle cost of capping;
- the input-scaling finding's "roughly 2.9" (F-10) adds that old cost to a
  figure from a different configuration.

**Corrected on 2026-10-03.** Two files still credited all 61,729 to Table 4's
shapes, found by searching the repository for "61,729" on 2026-10-02. Both now
say that the five weight matrices Table 4 prints hold 61,600 numbers and that
the other 129 are the offsets of Equations 12 and 13, each with a dated note:

- `src/thesis_gnn_rul_xai/models/README.md` said 61,729 is "the count Table
  4's printed shapes imply", and corrected itself in the next sentence. It now
  says only where each of the two parts comes from.
- `notebooks/02_model_fd001.ipynb` printed "Table 4 implies exactly 61,729
  parameters" in its step on the network, and named a check "Parameter count
  matches Table 4's shapes" in its closing checklist. The step now prints the
  two parts and their sum, and the check is named "Parameter count: Table 4's
  weights plus the dense offsets". A third copy of the claim, in the
  notebook's closing summary ("the 61 729 parameters implied by the weight
  shapes the paper's Table 4 prints"), was found on 2026-10-03 by searching for
  that wording rather than for the number, and corrected with them. A fourth,
  found by a review of these corrections on 2026-10-03, was the notebook's
  comparison of its two temporal stages, which printed that every row below
  the stage "is Table 4, unchanged"; the counts of the two dense rows include
  the offsets Table 4 does not list, and it now says those rows have Table 4's
  shapes and activations. The
  notebook was then run again, so that what it stores is what its code
  prints: the check passes with "61,600 + 129 = 61,729", and its seed 42
  matches seed 42 of the stored run
  `fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z`
  (commit `f4f40ad`) number for number, RMSE 16.694915. The first attempt to
  run it failed that check: the notebook's search for the stored run did not
  compare the network's own settings, and took a five-seed run stored on
  2026-10-02 whose graph layers have offsets. The search now requires Table
  4's network and the notebook's training settings, with a dated note in the
  notebook's section 6.

(This paragraph said "Still open" until 2026-10-03.)

Other places say "Table 4's 61,729" or "Table 4's network", meaning the network
Table 4 describes, whose total with the offsets of Equations 12 and 13 is
61,729; those were left. The search covered the Markdown files and notebooks of
the working tree; it is not a guarantee that no differently worded copy
survives.

---

## D-10. Per-pass training curves were thrown away

**What the audit said.** `scripts/train.py` recorded the training and held-back
error after every pass over the data and never saved it, so whether a stored
network was still improving or had over-trained could not be answered.

**What was done.** Since commit `c8ba1f1` of 2026-09-14 every run saves, for
each seed, `training_history.csv` with the error after every pass, beside the
trained weights and the test predictions.

The finding on what the recurrent stage is worth (F-16) inferred that the stage
over-trains, from where its best passes fall and from how much worse its final
weights are, and cited this item for the missing curves. The curves exist.
`fd001_physics_row_cap_5seed_varsplit_lstm32_cfg72f506_f386291-dirty_20260914T134955Z`,
at `f386291-dirty`, is that finding's early-stopping arm trained again after the
fix: the same five test RMSEs to every stored digit and the same kept passes
(10, 13, 9, 14, 21). From the kept pass to the last pass run, twenty passes
later, the training loss falls on every seed and the held-back RMSE rises on
every seed, for seed 42 from 0.01014 to 0.00665 and from 13.449 to 14.911
cycles. The network fits the training engines ever more closely while getting
worse on engines it never learns from: the over-training is now seen, not
inferred, for the first 41 passes. The finding sets out all five seeds and the
code. (Corrected the same day: this section and the finding first said the
curves exist only if the arm is trained again.)

**Still open.** Those runs stop twenty passes after the best, at most 41
passes, all at the first learning rate. What the held-back error does out to
pass 200, the length the paper states, needs the finding's full-length arm,
which held engines back and ran 200 passes before curves were saved, trained
again: **new training**. The paper's own procedure holds no engines back, so it
can never have a held-back curve.

---

## D-11. An ASUE of 0 is called the best value; it is undefined

**What the audit said.** ASUE, the average size of the errors on engines the
model predicted too early, is returned as 0.0 when no engine was predicted too
early, and the code calls that "the best value this measure can take". A model
that is never early is late on every engine, which is the behaviour the NASA
score exists to punish.

**Checked against the paper.** Equation 24 (page 9) divides the sum of the
early errors by `N_under`, the number of engines predicted too early, and the
text says the measure "is mathematically defined for instances where the
predicted RUL is less than the actual RUL". With no such engine it is 0 / 0:
not defined. So the audit is right that 0.0 is not the best value; "the worst"
overstates it, since the measure says nothing at all in that case.

The case has never occurred here. On 2026-10-02, 179 files of FD001 test
predictions were stored, one per seed: 89 in the runs under `reports/fd001_*`
and 90 in the FD001 part of the sensitivity study. The fewest engines predicted
too early in any of them is 25 of 100, in seed 44 of the study's arm with
min-max scaling over all conditions at once
(`reports/sensitivity_fd65a3b-dirty_20260929T101059Z/runs/FD001/scaling_minmax_global`,
at `fd65a3b-dirty`); in the runs under `reports/fd001_*` alone it is 30. A
second sensitivity study, still running, has written files for FD002 to FD004;
none written by that evening has zero either.

**Still open.** `evaluation/metrics.py` still returns 0.0, and its one-line
description and `src/thesis_gnn_rul_xai/evaluation/README.md` still call that
the best value. The fix is to return "not a number" with a warning and to report
the count of early engines beside it, and to record the choice as an assumption,
because the paper does not say what Equation 24 gives in that case. It was not
made today, because no stored figure depends on it and the change belongs to a
module outside this response.

---

## Beyond the eleven items

- **Reproduction across machines (the audit's Part 1).** The audit retrained
  the default elsewhere and matched it within the seed spread, not to the last
  digit. `src/thesis_gnn_rul_xai/utils/README.md` records that, and the entry
  on what each sensor node carries (A-07) now gives a case from this project's
  own runs. The same settings give 16.690 ± 0.229 under PyTorch 2.5.1 on a
  graphics card (the sensitivity study's last-reading arm, at `fd65a3b-dirty`)
  and 16.496 ± 0.415 under PyTorch 2.13.0 on the processor
  (`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z`,
  at `5602773-dirty`). The two also come from different commits. The default
  trained again on the processor from a clean checkout of `f4f40ad`, a commit
  after the study's
  (`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z`),
  gives the stored default's five results exactly, so the code is an unlikely
  cause. What remains is the device and the PyTorch version together, which no
  stored run separates.
- **No entry covers regularisation (the audit's Part 3).** The network has no
  weight decay, no dropout and no other penalty on its weights. The paper's text
  never mentions dropout, regularisation, weight decay or overfitting (searched
  in the copy in `docs/references/` on 2026-10-02), so adding nothing is
  faithful; but it is still a choice the paper leaves open, and it has no entry.
  Writing one needs a matching comment where the optimiser is built, in
  `training/`, which cannot change while the training study runs.
- **Per-condition z-score (the audit's Part 3).** The code still offers
  per-condition min-max and global min-max and z-score, and no per-condition
  z-score. It matters only for FD002 and FD004, where several operating
  conditions occur.

---

## What is still open, in one place

Needs new training:

- random graphs with 41 edges, and rewirings of the paper's graph (item D-2),
  after a new graph option is added;
- the recurrent stage's full-length arm with its curves saved (item D-10).

Needs a change in a file that waits for the running training study:

- the `split` field and the `varsplit` name in `scripts/train.py` (item D-4);
- an entry on regularisation, with its comment in `training/`.

Needs neither, but is not done:

- the comparison with ordinary models, from a clean commit (item D-5);
- ASUE's undefined case in `evaluation/metrics.py` (item D-11).

---

## Reproduce with

Every paired comparison above, from the stored runs named in it. Each line
prints the mean difference (first run minus second, seed by seed) and the
two-sided p of the paired t-test, on test RMSE and on held-back RMSE:

```python
import pandas as pd
from scipy import stats


def paired(a, b, column):
    """Run a minus run b on one column of their results.csv, paired by seed."""
    x = pd.read_csv(f"reports/{a}/results.csv")
    y = pd.read_csv(f"reports/{b}/results.csv")
    m = x.merge(y, on="seed", suffixes=("_a", "_b"))
    difference = float((m[column + "_a"] - m[column + "_b"]).mean())
    p = float(stats.ttest_rel(m[column + "_a"], m[column + "_b"]).pvalue)
    return round(difference, 3), f"{p:.3g}"


study = "sensitivity_fd65a3b-dirty_20260929T101059Z/runs/FD001/"
sweep = "fd001_physics_symmetric_cap_5seed_varsplit_{}09628fa_20260828T{}"   # the input-scaling sweep, 09628fa
clean = "fd001_physics_{}_cap_5seed_varsplit{}_cfg{}_f4f40ad_20261002T{}"    # trained at f4f40ad
default = "fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z"
comparisons = {
    # item D-3, input scaling
    "z-score minus min-max": (sweep.format("", "153058Z"), sweep.format("", "155320Z")),
    "min-max minus z-score, recurrent stage": (study + "scaling_minmax_per_condition", study + "reference"),
    # item D-3, graph rescaling
    "row minus symmetric, step, early stopping": (clean.format("row", "", "ba82e3", "152416Z"),
                                                  clean.format("symmetric", "", "5d7c10", "180802Z")),
    "row minus symmetric, step, no early stopping": (clean.format("row", "_noearlystop", "ba82e3", "170226Z"),
                                                     clean.format("symmetric", "_noearlystop", "5d7c10", "183412Z")),
    "row minus symmetric, exponential, early stopping": (
        clean.format("row", "_exponential", "ba82e3", "152750Z"),
        clean.format("symmetric", "_exponential", "5d7c10", "181841Z")),
    "row minus symmetric, exponential, no early stopping": (
        clean.format("row", "_noearlystop_exponential", "ba82e3", "170506Z"),
        clean.format("symmetric", "_noearlystop_exponential", "5d7c10", "183413Z")),
    "row minus symmetric, min-max": ("fd001_physics_row_5seed_varsplit_cdcc827-dirty_20260828T145616Z",
                                     "fd001_physics_symmetric_5seed_varsplit_cdcc827-dirty_20260828T143735Z"),
    "row minus symmetric, recurrent stage": (study + "reference", study + "adjacency_symmetric"),
    # item D-3, learning-rate schedule
    "exponential minus step": (clean.format("row", "_exponential", "ba82e3", "152750Z"),
                               clean.format("row", "", "ba82e3", "152416Z")),
    "exponential minus step, recurrent stage": (study + "schedule_exponential", study + "reference"),
    # item D-3, smoothing; the other rows of the input-scaling sweep
    "smoothing off minus on": (sweep.format("", "153446Z"), sweep.format("", "155320Z")),
    "smoothing off minus on, recurrent stage": (study + "smoothing_off", study + "reference"),
    "no early stopping minus baseline": (sweep.format("noearlystop_", "152617Z"), sweep.format("", "155320Z")),
    "constant rate minus baseline": (sweep.format("constant_", "154334Z"), sweep.format("", "155320Z")),
    "ceiling 130 minus baseline": (sweep.format("", "154003Z"), sweep.format("", "155320Z")),
    # items D-2 and D-3, the every-sensor graph
    "every sensor joined minus the paper's graph, default":
        ("fd001_complete_row_cap_5seed_varsplit_cfgf21547_f386291-dirty_20260914T150527Z", default),
    "every sensor joined minus the paper's graph, min-max":
        ("fd001_complete_symmetric_5seed_varsplit_cdcc827-dirty_20260828T144045Z",
         "fd001_physics_symmetric_5seed_varsplit_cdcc827-dirty_20260828T143735Z"),
    "every sensor joined minus the paper's graph, recurrent stage": (study + "graph_complete", study + "reference"),
    # item D-6, the window average; item D-7, the offset
    "window average minus last reading": (study + "temporal_mean_window", study + "temporal_last_cycle"),
    "offset minus none, recurrent stage": (study + "graph_layer_bias", study + "reference"),
}
for name, (a, b) in comparisons.items():
    print(name, "| test", paired(a, b, "rmse"), "| held back", paired(a, b, "validation_rmse"))

# item D-9: the cost of capping, the same networks scored two ways at 9856794-dirty
print(paired("fd001_physics_row_uncap_5seed_varsplit_cfge49671_9856794-dirty_20260908T141351Z",
             "fd001_physics_row_cap_5seed_varsplit_cfgba82e3_9856794-dirty_20260908T135935Z", "rmse"))
# item D-6 and Beyond: the means and spreads quoted
for arm in ("temporal_mean_window", "temporal_last_cycle"):
    r = pd.read_csv(f"reports/{study}{arm}/results.csv")
    print(arm, round(r.rmse.mean(), 3), round(r.rmse.std(), 3))
for run in ("fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z",
            clean.format("row", "", "ba82e3", "152416Z")):
    print(run, pd.read_csv(f"reports/{run}/results.csv").rmse.round(6).tolist())
```

The sixth blind direction (item D-1) needs only the code:

```python
import numpy as np
from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_SYMBOLS as S
from thesis_gnn_rul_xai.graph.adjacency import build_physics_adjacency, normalize_adjacency

table = normalize_adjacency(build_physics_adjacency()[0], method="row").astype(np.float64)
_, sizes, rows = np.linalg.svd(table)
blind = rows[sizes < 1e-10]                                   # six directions the graph cannot see
pairs = [("P15", "Nf"), ("P15", "NRf"), ("P15", "BPR"), ("P30", "Ps30"), ("Nc", "W31")]
within = np.zeros((5, 14))
for k, (a, b) in enumerate(pairs):
    within[k, S.index(a)], within[k, S.index(b)] = 1, -1
basis, _ = np.linalg.qr(within.T)
left = np.linalg.svd(blind - blind @ basis @ basis.T)[2][0]
left /= left[S.index("T30")]
print({S[i]: round(float(v), 6) for i, v in enumerate(left) if abs(v) > 1e-9})
```

The ranking of the large group over the 100 test engines (item D-1), about
40 seconds on a processor; it prints the mean sizes of the Shapley values and
of the displacements, then 97:

```python
import numpy as np
from thesis_gnn_rul_xai.training.stored import read_run, load_seed
from thesis_gnn_rul_xai.explain.shap import shap_values, mean_window_baseline

model, prepared = load_seed(read_run(
    "fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z"), 42)
stand_in = mean_window_baseline(prepared.train)                         # (14, 30), the average training window
group = [list(prepared.sensor_symbols).index(s) for s in ("P15", "Nf", "NRf", "BPR")]
windows = prepared.test.features.astype(np.float32)                     # (100, 14, 30)
phi = np.array([shap_values(model, w, stand_in)[group] for w in windows])  # (100, 4)
shift = windows[:, group, -1] - stand_in[group, -1]                     # (100, 4), last reading minus stand-in
print(np.abs(phi).mean(0).round(3), np.abs(shift).mean(0).round(3))
same = (np.argsort(-np.abs(phi), 1) == np.argsort(-np.abs(shift), 1)).all(1)
print(int(same.sum()))
```

The every-sensor network (item D-2), on seed 42 of its stored run:

```python
import numpy as np, torch
from thesis_gnn_rul_xai.training.stored import read_run, load_seed

model, prepared = load_seed(read_run(
    "fd001_complete_row_cap_5seed_varsplit_cfgf21547_f386291-dirty_20260914T150527Z"), 42)
x = torch.from_numpy(prepared.test.features.astype(np.float32))      # (100 engines, 14 sensors, 30 cycles)
with torch.no_grad():
    hidden = model.temporal(x)
    for layer in model.graph_layers:                                  # 0.0: every sensor holds the same numbers
        hidden = torch.relu(layer(hidden, model.adjacency))
        print(float((hidden - hidden[:, :1]).abs().max()))
    before = model.predict_cycles(x)
    shuffled = x[:, torch.randperm(14, generator=torch.Generator().manual_seed(0))]
    moved = x.clone(); moved[:, 0, -1] += 1.0; moved[:, 1, -1] -= 1.0  # 1.0 from T30's last reading to T24's
    print(float((model.predict_cycles(shuffled) - before).abs().max()),
          float((model.predict_cycles(moved) - before).abs().max()))
```

The fewest engines predicted too early (item D-11):

```python
from glob import glob
import pandas as pd

files = glob("reports/fd001_*/seed_*/test_predictions.csv") + \
        glob("reports/sensitivity_*/runs/FD001/*/seed_*/test_predictions.csv")
early = {f: int((t.predicted < t.rul).sum()) for f in files for t in [pd.read_csv(f)]}
print(len(files), min(early.values()), min(early, key=early.get))
```

Each of the other figures has its own code in the entry that records it: the
interval over engines in the comparison with the paper's figures (F-06), the
smoothing weights and shares in the entry on what each sensor node carries
(A-07), the averaging table in the entry on the every-sensor control (A-19),
the Shapley swap in the finding on indistinguishable sensors (F-18), the
scaling of the graph layers in the finding on what they can compute (F-21), and
the recurrent stage's per-pass curves in the finding on what that stage is
worth (F-16).
