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

## P-16 — The trained checks of `verify_claims.py` no longer agree with the log

`python scripts/verify_claims.py --fast` passes all 62 of its claims. Without
`--fast` it adds two checks that train a network and recompute the Shapley
tables of assumption A-20 and finding F-05, and on 2026-09-14 neither passes:

- **Under `config/fd001.yaml`** the P15 check stops with
  `ValueError: engine 5 is not in this dataset`. Since 2026-09-08 the default
  holds 20 % of the engines back, and engine 5 is one of seed 42's held-back
  engines, so it is no longer among the training engines the check reads.
- **Under a copy of that file with `validation_fraction: 0.0`** — the setting
  both entries say they were measured with, on 2026-08-31 at `f150570` — 18
  claims were checked and 6 disagreed. Two of those were the check's own fault
  and were fixed on 2026-09-14: it took the spread of
  `prepared.train.features[:, sensor, :]`, which held one cycle per window until
  2026-09-09 and holds thirty since. Taken over the last cycle, the spreads are
  0.862 for P15 and 1.053 for Nc, exactly as recorded. Four still disagree, all
  computed from a trained network:

  | Claim | Recorded | Measured |
  |---|---|---|
  | A-20, top four with the mean window | T50, Nc, W31, T24 | Nc, T50, W31, T24 |
  | A-20, Spearman of all zeros against the mean window | 0.996 | 0.965 |
  | F-05, largest contribution of P15, engine 5 | 9.449 | 6.814 |
  | F-05, largest contribution of P15, engine 82 | 9.526 | 6.287 |

**Not caused by the changes of 2026-09-14.** That day's five-seed run of the
default reproduces the `ae59bc1-dirty` run of 2026-09-08 on every seed and every
measure, so the training path is unchanged since then. What changed the trained
network between 2026-08-31 and 2026-09-08 is not established here; the
learning-rate schedule's default moved to a single step on 2026-09-01, which is
the first thing to test.

**How to check:** `python scripts/verify_claims.py --only baselines p15`, and
again with `--config` pointing at a copy of `config/fd001.yaml` holding nothing
back.
**What finishing looks like:** each check pins the settings its entry was
measured under instead of reading the default; the cause of the drift is found;
and A-20 and F-05 are re-measured and corrected in place, saying what was wrong
and how it was found (working rule 6).

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
re-measured at today's configuration, F-17), step 8 (`GATStage` and
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
`features[:, :, -1]` to reproduce what the audit measured.

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

## P-13 — Notebooks by component: `temporal/` and `graph/`, and the matrix explorer

On 2026-09-14 the author proposed organising the notebooks the way `xai/` now
is: a folder per swappable part of the model, one notebook per alternative.
`notebooks/xai/` is built that way, and `notebooks/README.md` names the two
folders still to come.

- **`notebooks/temporal/`** — one notebook per temporal stage (`last_cycle`,
  `mean_window`, `lstm`, `gru`, and the gated `raw_window`), each showing on
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
