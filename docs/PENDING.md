# Pending work

Things known to be wrong, unmeasured or unfinished, written down so they are not
rediscovered from scratch. Every entry says **what** is wrong, **how to check**
it, and **what finishing it looks like**, so a session with the budget for it can
pick one up without re-deriving the context.

Add to this file whenever you find something you are not fixing now. Delete an
entry when it is done — a stale entry here is worse than no entry, because the
next reader will spend budget confirming it is already fixed.

Ordered by consequence, not by effort. Identifiers are never reused, so a
gap in the numbering means an entry was finished and removed.

---

## P-17 — Nothing in the repository shows the network beats an ordinary model

The thesis has to answer "would any reasonable method have done as well?", and
no committed code, log entry or README row answers it. Every result is set
beside the paper and beside other configurations of the same network, never
beside a model with no graph in it. The audit of 2026-09-08 raised this (its
D-5) and it was never carried out.

**A scratch measurement on 2026-09-14 says the answer matters.** It set the two
stored five-seed runs — the current default,
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z`,
and the same settings with the recurrent stage,
`…_lstm32_cfg72f506_f386291-dirty_20260914T134955Z` — beside scikit-learn
models trained, for each seed, on exactly that seed's training engines (the
held-back engines unused) and scored on the same 100 test engines. The inputs
were `features[:, :, -1]`, the fourteen numbers the default network receives,
or for the last row the last value, mean and least-squares slope of each
sensor's 30-cycle window. Library defaults throughout, chosen before any result
was seen; predictions clipped to [0, 125]. Before comparing, the stored
predictions were checked to reproduce `results.csv` exactly, and all ten stored
networks were reloaded at `8b487ab` and reproduced their saved predictions to
3.8e-06 cycles.

| Model | RMSE, 5 seeds | Network's RMSE minus this | Seeds network better | Paired t over seeds | 95 % interval over 20,000 resamplings of the test engines |
|---|---|---|---|---|---|
| network, last cycle (stored) | 16.496 ± 0.415 | — | — | — | own RMSE [14.00, 18.90] |
| network + recurrent stage, best pass restored (stored) | 13.307 ± 0.532 | +3.189 | 0/5 | 0.0009 | [+1.03, +5.35] |
| constant: mean training label | 40.568 ± 0.089 | −24.072 | 5/5 | < 0.0001 | [−29.00, −19.12] |
| gradient boosting trained on shuffled labels | 40.846 ± 0.292 | −24.349 | 5/5 | < 0.0001 | [−29.33, −19.35] |
| gradient boosting on engine age alone | 31.239 ± 0.100 | −14.742 | 5/5 | < 0.0001 | [−18.92, −10.56] |
| linear regression, same 14 numbers | 20.457 ± 0.219 | −3.961 | 5/5 | < 0.0001 | [−6.30, −1.70] |
| one-hidden-layer network of 64, same 14 numbers | 17.702 ± 0.469 | −1.206 | 5/5 | 0.0023 | [−2.89, +0.64] |
| random forest, same 14 numbers | 17.870 ± 0.308 | −1.373 | 5/5 | 0.0057 | [−3.20, +0.56] |
| gradient boosting, same 14 numbers | 17.253 ± 0.329 | −0.756 | 5/5 | 0.0472 | [−2.56, +1.17] |
| gradient boosting, 42 window summaries | 12.134 ± 0.364 | +4.363 | 0/5 | 0.0001 | [+2.04, +6.69] |

The recurrent-stage network against the last row: +1.174 cycles, the booster
ahead on all five seeds (paired p = 0.0099), interval over engines
[−0.13, +2.45].

**Which summary does it**, measured the same way the next day (five seeds, no
paired tests run): boosting on the last value alone 17.253 ± 0.329, the mean
alone 21.207 ± 0.447, **the slope alone 13.693 ± 0.488**, last and mean
15.086 ± 0.595, **last and slope 11.835 ± 0.391**, mean and slope
12.827 ± 0.320; and **linear regression** on all 42, 14.520 ± 0.120, against
20.457 ± 0.219 on the last value. The slope carries the gain, and a linear
model given it beats the network, so it is the information the default network
never receives, not the boosting algorithm.

**The graph network given the same three numbers per sensor has since been
measured properly** (assumption A-28): 13.586 ± 0.675, 2.910 cycles below the
last-cycle default on the same code and seeds. So the default's gap to the
booster above is mostly the information it is not given. How it compares with
the booster itself stays open until the booster is committed and paired with it.

Read with care, and not quoted anywhere until finished:

- **The scores are not chance.** A model trained on shuffled labels lands on the
  constant: what the fitted models achieve needs the link between readings and
  remaining life. That is all the shuffled-label row shows.

  **Correction, 2026-09-15.** This bullet first said the shuffled row showed
  "no path other than the sensor-to-life relationship produces a good score",
  and the reply that proposed it said a leak from the test engines would let a
  model trained on shuffled labels score well. Both overstate it. If the answer
  leaked into the inputs, a model trained on shuffled labels would not learn to
  read it, and would still land on the constant. The row therefore rules out a
  scoring bug that rewards any prediction, not a leak. The author questioned it
  on 2026-09-15, rightly. A check that does catch a leak from the test answers
  is to change `RUL_FD001.txt` and require every test input, and every
  prediction of a trained network, to stay identical; only the scores may move.
- **The graph network's edge over an ordinary model on the same fourteen
  numbers is not established.** It wins on every seed against the three
  flexible models, but every one of those intervals over engines contains zero.
- **An untuned booster given the window's trend beats both networks** — the
  default clearly, the recurrent one on seeds but not resolvably over engines.
  That bears on assumption A-07 and on the audit's D-6.

**The numbers above came from a script outside the repository**, so under
working rule 3a they name no committed code and must be re-produced before any
of them is used.

**What finishing looks like:** a committed `scripts/baselines.py` that reads a
stored run, rebuilds each seed's split, fits the ladder above and writes its
own run directory with commit and timestamp; the answer-file test in the
correction above; the table written into
`docs/ASSUMPTIONS.md` as a finding with both kinds of uncertainty; the rows
added to `README.md` section 8; and the same ladder run for the recurrent stage
under the epoch-200 procedure (F-16) and for any configuration the thesis
quotes.

---

## P-12 — What the data-driven graph shows is measured for one architecture only

Assumption A-27 (added 2026-09-14) sets the paper's Table 3 beside a graph of
the 41 most strongly correlated sensor pairs. Under the current default — the
last-cycle temporal stage, so no recurrent stage — the two cannot be told apart
by test RMSE (+0.096, p = 0.655), the correlation graph is 2.0 cycles better on
the held-back engines (p = 0.012), and its test predictions run later. That
reads against the paper's claim that its graph goes beyond correlations, and it
is the most consequential measurement this project has for the thesis's
question about the graph. Three things keep it from being settled.

1. **It is one architecture.** Two of this project's comparisons reversed when
   the architecture moved (working rule 3). The paper's claim is about the full
   model. **Finishing:** run both graphs with the recurrent stage, at one commit,
   and add the table to A-27:

   ```bash
   python scripts/train.py --config config/fd001.yaml --temporal lstm --seeds 5
   python scripts/train.py --config config/fd001.yaml --temporal lstm --graph correlation --seeds 5
   ```

2. **The held-back comparison is flattered by selection.** The recorded
   held-back error is the lowest reached over all passes, and it also chose
   which pass to keep, so it favours whichever network's error wanders more.
   Runs since 2026-09-14 save every pass's error in each seed's
   `training_history.csv`. **Finishing:** compare the two graphs on a statistic
   no pass was chosen by — the held-back error averaged over the last 20 passes
   run, say — and replace the caution in A-27 with the result.

3. **FD002 and FD004 are untouched.** There the correlations are measured after
   per-condition scaling, and no run exists.

---

## P-20 — The Shapley tables of A-20 and F-05 come from one trained network

Both entries, including the tables added to them on 2026-09-14, are computed
from a single network: seed 42, nothing held back, all 200 passes. That day's
correction showed how loose that is. Changing only the learning-rate schedule
moved `P15`'s largest contribution by 27.9 % on engine 5 and 34.0 % on engine
82, and swapped the first two sensors of A-20's ranking under the mean window.
Every rank F-05 relies on held.

A different seed may move these figures as much, or more. It has not been
measured, so every size in both entries is one draw, and the rank claims have
been tested against two trainings rather than against the spread of five
(working rule 3).

**Finishing:**

1. Train five seeds under the entries' settings. Runs since 2026-09-14 save each
   seed's network.

   ```bash
   python scripts/train.py --config <a copy of config/fd001.yaml with validation_fraction: 0.0> --no-early-stopping --seeds 5
   ```

2. Recompute both tables on each saved network; `training/stored.py` loads them.
3. Write into each entry the spread of every figure, and whether `P15` ranks
   last on every seed.

---

## P-08 — The swappable-stages design: the temporal seam is built, the rest is not

`docs/DESIGN_swappable_stages.md` sets out nine steps. **Steps 1, 2, 3, 4 and 7
are done**: `models/temporal/` exists with five stages, `ESTGNN` takes one and
derives its width from it, `data/windowing/` hands over the whole window,
`LSTMStage` builds the paper's Section 2.2 stage, and since 2026-09-14
`scripts/train.py` writes `model_config.yaml` beside every run, names a
non-default network by its digest, and saves each seed's weights, per-pass
record and test predictions.

**What is not done:** step 5 (`models/spatial/`), step 6 (`raw_window`
re-measured at today's configuration, and recorded as the next free finding —
the design's "F-17" was provisional and that number now belongs to Fig. 10),
step 8 (`GATStage` and
`NoSpatialStage`, which would make the paper's "w/o GCN" row of 16.42
addressable for the first time), and step 9 (edge and subgraph attributions).
The design's assumption numbers from A-27 on were provisional and have been
overtaken; a note in the design says so.

**A defect it turned up, still open:** `PreprocessingConfig.notes` is inside
the run digest. `to_dict()` is `dataclasses.asdict`, so a prose field is hashed.
`PreprocessingConfig()` and `load_config("config/fd001.yaml")` agree on every
setting and still digest differently — `e3a331` against `ba82e3` — because of
one sentence of English. Editing a comment renames a run. `training/stored.py`
already ignores `notes` when matching runs, which is the right rule; the digest
does not.

**What finishing looks like:** a decision on the remaining steps, and the digest
fixed. Fixing it gives future runs a different `cfg` value than stored runs
with identical settings, so it is a one-time discontinuity that needs a line in
the log.

---

## P-10 — The audit of 2026-09-08 is written but not triaged

`docs/AUDIT_2026-09-08_scientific_validity.md` is an independent audit run from
outside the repository. It reports eleven defects, D-1 to D-11, ranked by what
they do to a conclusion. Its opening three are the heaviest:

- **D-1.** The paper's own adjacency matrix normalises to rank 8, so part of the
  fourteen-number input is structurally invisible to any graph convolution
  built on it. It says this applies to the per-sensor LSTM model as well, since
  that stage also hands the graph layers one number per sensor.
  `scripts/verify_claims.py --only correlation` now checks the 8.
- **D-2.** The "without physics prior" control is degenerate: the fully
  connected graph normalises to a rank-1 averaging operator. **Evidence now
  exists:** A-27's correlation graph keeps the edge count and costs 0.096 cycles
  where the fully connected graph costs 2.784 under the same configuration, and
  A-19 carries a correction saying so. What is left is to write the audit's
  point up as the next finding, or fold it into A-27.
- **D-3.** Every configuration comparison since 2026-08-29 was decided on the
  test set. Notebook 02 section 5 and A-27 now show held-back error beside test
  error for the graph comparison; the earlier comparisons are unchanged.

**A numbering collision to avoid.** The audit's recommendations say to *"write
the rank finding up as F-16"*. It was written against a tree in which F-16 did
not yet exist. **F-16 is now taken** by the recurrent-stage finding, so the rank
finding must take the next free identifier. `tests/test_docsite.py` requires
findings to be numbered consecutively from F-01, so skipping a number fails the
suite.

**A trap in re-running it, created after it was written.** The audit's
non-graph baselines (D-5) are specified as scikit-learn on
`prepared.train.features[:, :, 0]`. When the audit ran, the pipeline handed over
`(N, 14, 1)`, so index 0 was the one cycle there was — the last. Since
2026-09-09 the pipeline hands over `(N, 14, 30)` and the temporal stage reduces
it, so **index 0 is now the first cycle of the window, 29 cycles out of date**,
and the same command runs without complaint and measures something else. Use
`features[:, :, -1]` to reproduce what the audit measured. D-5 now has its own
entry, P-17, with a measurement at today's configuration.

**What finishing looks like:** each D-item either written into
`docs/ASSUMPTIONS.md` as a finding or an amendment, or answered in writing with
the reason it does not apply. Two of them already bear on F-16 and are cited
there: D-8, that seed spread is not uncertainty over engines, and D-10, that the
per-epoch training curves were not saved (they are for runs from 2026-09-14 on).

---

## P-11 — F-16's pairing crosses a refactor, and its central inference is unobserved

Two loose ends in F-16, both cheap to close.

**The pairing crosses the 2026-09-09 refactor.** The `last_cycle` run with
nothing held back is at commit `ae59bc1-dirty`; the recurrent run it is paired
against is at `c703693-dirty`, after the temporal stage became a component. The
refactor was gated on reproducing the previous five-seed run bit-identically and
did, so the training path is the same, but a reader should not have to take
that on trust. **Finishing:** re-run

```bash
python scripts/train.py --config <a copy of config/fd001.yaml with validation_fraction: 0.0> --no-early-stopping --seeds 5
```

at one commit, confirm it reproduces 16.818 ± 0.157, and replace the pairing.

**That the recurrent model over-trains is inferred, not seen.** F-16 infers it
from where the best epochs fall (9 to 21) and from how much worse the epoch-200
weights are. Since 2026-09-14 every run saves each seed's per-pass curves in
`training_history.csv`. **Finishing:** re-run the recurrent arm with
`--temporal lstm`, and put the held-back curve of one seed into F-16 in place of
the inference.

---

## P-18 — Without the recurrent stage, the red four of a dynamic explanation changes almost every window; measured only in a scratch script

On 2026-09-14 the author saw that in the playground's sensor graph, sensors
change colour between neighbouring windows far more than in the paper's Fig. 10.
It was measured that day on the longest held-back engine of each of the five
seeds of two stored runs, both at `f386291-dirty` with the step learning-rate
schedule — `fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z`
(`last_cycle`) and
`fd001_physics_row_cap_5seed_varsplit_lstm32_cfg72f506_f386291-dirty_20260914T134955Z`
(`lstm`, hidden 32) — with `dynamic_shap` run at `563c390-dirty`. For each seed
both runs explain the same engine. "Joins" counts sensors entering the four
largest by size from one window to the next:

| | Joins over the 12 steps inside graphs 1–5, 78–82, 187–191 | Joins per step, every window | Mean step ÷ mean size, median over sensors |
|---|---|---|---|
| The paper's Table 12 | 1 | — | 0.084 |
| `lstm` | 4.6 ± 2.1 | 0.38 ± 0.11 | 0.142 ± 0.041 |
| `last_cycle` | 20.2 ± 4.8 | 1.47 ± 0.12 | 0.390 ± 0.094 |

Paired by seed, `last_cycle` has 15.6 more joins in the 12 steps (p = 0.0005)
and 1.09 more per step (p = 2e-7). What each stage hands the graph layers was
measured alongside: its change between neighbouring windows, as a share of its
spread over the engine's life, is 0.281 ± 0.021 for `last_cycle`, which passes on
each window's last smoothed reading, and 0.119 ± 0.005 for `lstm`. For T50 on
seed 42's engine 5, windows ending at cycles 30 to 34, `last_cycle` hands over
−1.444, −1.115, −1.266, −1.020, −1.515 and `lstm` 0.814, 0.786, 0.782, 0.756,
0.787.

**What it does not settle.** Even the `lstm` network changes its red four more
often than Table 12 does, and Table 12 holds positive magnitudes computed in a
way the paper never states (F-08), so that remaining gap cannot be put down to
the network. Seed 42, the one the playground's stored explanation uses, gave the
fewest `lstm` joins of the five (2).

**Why it is here and not in the log.** The numbers came from a script in a
session scratchpad, not from committed code (working rule 3a).

**What finishing looks like:** the calculation in the repository — a section of
`notebooks/xai/01_shap_fd001.ipynb` that reads both stored runs, draws the
paper's Fig. 10 from `evaluation/published.py` beside the engine the way the
playground now does, and counts the joins — re-run at one commit, and the result
written into `explain/README.md` as a trap for anyone reading a dynamic
explanation of the default network. P-20 needs the same loop — Shapley values
for every saved seed of a stored run — on networks trained with nothing held
back, so one piece of code can serve both.

---

## P-19 — Outside the playground, the layer table still shows the temporal stage as a row before Table 4's

`ESTGNN.layer_summary()` returns a "Temporal" row and then Table 4's "Input" row,
and both describe the same `(B, 14, 1)` tensor: Table 4's Input row, which lists
no parameters, is what the recurrent stage hands to the first graph layer. On
2026-09-14 the author pointed this out on the playground, which now shows the two
as one Input row naming the stage. Three places still show both:

- notebook 02, cells 16 and 20, whose text says "the first row is the stage.
  Every row below it is Table 4, unchanged";
- the site's "The network" page, whose table `server.py` builds from
  `layer_summary()` and introduces as "the same table the paper prints", with a
  Temporal row reading `LastCycle`, `(B, 14, 1)` in and out, 0 parameters;
- `figures.py`'s `draw_architecture`, which draws a box for each row.

`tests/test_models.py::test_activations_and_shapes_match_table_4` asserts the
separate row, and `docs/DESIGN_swappable_stages.md` chose it on purpose.

**What finishing looks like:** a decision on whether `layer_summary()` itself
should fold the stage into the Input row. If yes: change it and that test,
re-execute notebook 02 (it trains, so it takes minutes) and add a dated line to
the design. If no: fold the rows on the "The network" page as the playground
does, and leave the notebook's framing, which says in words what it is doing.

---

## P-13 — Notebooks by component: `temporal/` and `graph/`, and the matrix explorer

On 2026-09-14 the author proposed organising the notebooks the way `xai/` now
is: a folder per swappable part of the model, one notebook per alternative.
`notebooks/xai/` is built that way, and `notebooks/README.md` names the two
folders still to come.

- **`notebooks/temporal/`** — one notebook per temporal stage (`last_cycle`,
  `mean_window`, `window_summary`, `lstm`, `gru`, and the gated `raw_window`), each showing on
  the real windows what the stage does to one sensor's 30 readings and what it
  costs in parameters.
- **`notebooks/graph/`** — one per sensor graph: the paper's Table 3, every
  sensor joined, and the 41 strongest correlations. Notebook 02 section 5 is
  the seed of it and would move there.

**Two rules for them, and both matter.** First, a component cannot produce a
result on its own — a temporal stage needs graph layers and training — so these
notebooks **read five-seed results from `reports/` rather than training
inline**; a number trained in a notebook is one seed and records no commit
(working rules 3 and 3a). Second, every comparison in them names the whole
configuration it was measured under, because this project's comparisons have
reversed when the rest of the configuration moved. Each folder gets a README
holding what its notebooks share, as `xai/README.md` does, and a comparison
notebook that lays the stored runs side by side with their spread and a paired
test.

**The documentation site's matrix explorer** belongs with this: a grid of
temporal stage x sensor graph x graph rescaling where every cell shows the
stored five-seed result with its commit, and every unmeasured cell is visibly
empty and shows the command that would fill it. The playground must not train
for this — the reasons were argued on 2026-09-09: a number shown in a browser
and never written to `reports/` with its commit cannot be quoted or reproduced.

**What finishing looks like:** the two folders built on that pattern, notebook
02's section 5 moved into `graph/`, and the explorer beside the existing
`#/explore/` pages.

---

## P-14 — Codes used as words outside the documents the check covers

`scripts/check_prose.py` now fails when a code such as `A-07` is used as a word
instead of as a pointer, in the READMEs, `docs/learn/` and the notebooks. It does
not look at:

- **`docs/ASSUMPTIONS.md`** itself, where entries refer to one another by code
  hundreds of times ("under A-22's rule", "see F-10"). It is the log, so a code
  there has its entry one click away, but the author's instruction covers it.
- **`docs/LITERATURE_BASIS.md`** and the dated audits and reviews. The dated
  documents are records of what was written then and should not be rewritten.
- **The documentation site's own interface.** The playground's warnings show
  bare codes as labels (`D-2`, `F-16`, `A-26`) in
  `src/thesis_gnn_rul_xai/docsite/playground.py`, `caveats()`.
- **`scripts/train.py --help`**, whose option texts end in "(A-19)" and the like.

**What finishing looks like:** the log's cross-references rewritten with the
thing said in words; the playground's labels showing the entry's title with its
code; the check extended to `docs/ASSUMPTIONS.md` with entry headings and table
identifier columns exempt.

---

## P-03 — Two of A-21's quoted figures do not agree with each other

The "current default" labels this entry used to track were corrected on
2026-09-14 in `README.md`, `docs/learn/02_graphs.md`, F-04 and A-21's table, and
F-04 gained the row for the current default. What remains is in A-21:

- The **summary table** at the top of the log says row rescaling came 0.25
  cycles closer with nothing held back, p = 0.068. `config.py`'s comment,
  `config/fd001.yaml`, the package README and `docs/learn/02_graphs.md` all
  quote 0.35 and p = 0.047. A-21's own table holds both: 0.253 under the step
  schedule, 0.352 under the superseded exponential one. So the documents quote
  the superseded schedule's figure as though it were current.
- A-21's row "z-score, 20 % held back, early stopping" gives row rescaling
  16.616 ± 0.289, while the stored run of the current default — the same
  settings — gives 16.496 ± 0.415 at `f386291-dirty`. The row names no commit,
  so it cannot be told which schedule or code produced it.

**How to check:** `grep -rn "0.35 cycles\|0.047" config src docs/learn README.md`.
**What finishing looks like:** every quotation names the schedule it was
measured under, and the 20 %-held-back rows of A-21 re-run at one commit.

---

## P-04 — Nothing links to the figures gallery

`docsite/assets/app.js` serves a gallery of all thirteen figures at the address
`#/figures`, and `grep -rn "#/figures"` finds no link to it anywhere — not in
the sidebar, not in `shell.html`, not in any document. It is reachable only by
typing the address.

**What finishing looks like:** a link in the sidebar built from
`figure_names()`, so the list cannot go stale.

---

## P-07 — nbconvert silently drops SVG outputs

Guarded by a test now
(`test_the_map_survives_into_the_page_the_site_builds_from_the_notebook`), but
the underlying trap is general: any notebook cell that displays an
`IPython.display.SVG` will show in JupyterLab and vanish from the documentation
site. Measured on nbconvert 7.17.1 against the `basic`, `lab` and `classic`
templates — all three drop `image/svg+xml` and keep `text/html`.

**What finishing looks like:** a line in `docsite/README.md` warning anyone
adding a notebook figure, so the next person does not spend an hour on it.
