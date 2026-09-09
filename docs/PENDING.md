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

## P-08 — The swappable-stages design: the temporal seam is built, the rest is not

`docs/DESIGN_swappable_stages.md` sets out nine steps. **Steps 1, 2, 3 and 7 are
done** as of 2026-09-09: `models/temporal/` exists with five stages, `ESTGNN`
takes one and derives its width from it, `data/windowing/` hands over the whole
window, and `LSTMStage` builds the paper's Section 2.2 stage. The refactor was
gated on reproducing the previous five-seed run and did so bit-identically,
including `best_epoch`. F-16 records what the stage is worth.

**What is not done:** step 4 (a `ModelConfig` digested into the run name, and
`model.pt` saved per seed), step 5 (`models/spatial/`), step 6 (`raw_window`
re-measured at today's configuration, F-17), step 8 (`GATStage` and
`NoSpatialStage`, which would make the paper's "w/o GCN" row of 16.42 addressable
for the first time), and step 9 (edge and subgraph attributions).

Two defects it turned up along the way, both real and both independent of
whether the design is ever built:

- **`PreprocessingConfig.notes` is inside the run digest.** `to_dict()` is
  `dataclasses.asdict`, so a prose field is hashed. `PreprocessingConfig()` and
  `load_config("config/fd001.yaml")` agree on every setting and still digest
  differently — `e3a331` against `ba82e3` — because of one sentence of English.
  Editing a comment renames a run.
- **`TrainingHistory.save()` is never called**, so `epochs`, `batch_size` and
  `learning_rate` appear in no run directory at all.

**What finishing looks like:** a decision on the design, and those two defects
fixed or written up whatever that decision is. Fixing the digest gives future
runs a different `cfg` value than stored runs with identical settings, so it is
a one-time discontinuity that needs a line in the log.

---

## P-09 — The notebooks and the playground, as agreed 2026-09-09

`notebooks/02_model_fd001.ipynb` is one document doing two jobs in 45 cells:
it builds and trains the network, and it explains the network with SHAP. It now
also carries the temporal stage. The agreed shape is a split at the
**model / explanation** seam rather than by component, because a temporal stage
without graph layers is not a runnable model and a component notebook could not
produce a result on its own.

The sequence, each step useful alone:

1. **`03_explaining_fd001.ipynb`** — move section 7 of notebook 02 out and give
   it room: what a Shapley value is, why this implementation is exact where
   most are not, what "removing a sensor" means under A-20, dynamic SHAP over
   cycles, and what an attribution does **not** establish. That last part is
   the thesis's research gap and currently gets a paragraph.
2. **Save `model.pt` per seed** in each run directory. Nothing downstream of a
   trained model can be built without it, and it is two lines in
   `scripts/train.py`. `TrainingHistory.save()` is also never called, so
   `epochs`, `batch_size` and `learning_rate` reach no run directory at all.
3. **`04_what_each_component_is_worth.ipynb`** — reads `reports/`, lays out the
   matrix of temporal x graph x normalisation, and shows each component's
   contribution with the seed spread and a paired test. This is the notebook
   that answers "how does each component affect the result", and it can only be
   honest if it reads recorded runs rather than training inline.
4. **A matrix explorer in the documentation site**, beside the four that exist
   (`#/explore/graph`, `architecture`, `results`, `assumptions`).

**The playground must not train.** This was argued and agreed rather than
assumed. A number that appears in a browser and is never written to `reports/`
with its commit cannot be quoted or reproduced (rule 3a, and F-09 is what
happens when that slips). A UI that answers in ten seconds shows single-seed
numbers, which rule 3 forbids as a result. And five seeds of `last_cycle` take
about thirteen minutes, the LSTM far longer, so it is not an interaction loop
under any design.

The line is **what needs training**:

| Instant, computed on request | Read from `reports/` |
|---|---|
| the graph under any choice, with degrees, diameter, reach | RMSE, score, R², ASUE |
| the architecture for any temporal x graph pair, with real parameter counts | any comparison, with its spread and paired test |
| preprocessing effects: smoothing, windows, the label ceiling | every number carrying its commit |

**The feature to build it around is the empty cells.** A matrix where the
unmeasured combinations are visibly blank, each showing the command that would
fill it. Not "here is a number" but "here is what has been measured, under
which commit, with what spread, and here is what nobody has run". For a thesis
that is the panel that answers the examiner's question, and it makes the gaps
in the experiment a first-class object instead of something a reader has to
reconstruct.

---

## P-03 — Results labelled "current default" were measured under the old default

The A-06 reversal changed what "the current default" means, and several tables
still attach that label to 16.818 ± 0.157, which was measured with nothing held
back.

| Where | Line |
|---|---|
| `README.md` | the results table row "**Current default** — z-score, row, no validation split (A-22)" |
| `docs/learn/02_graphs.md` | "Current default — z-score, row normalisation, nothing held back" |
| ~~`docs/learn/04_the_pipeline.md`~~ | fixed 2026-09-09: it said the project holds nothing back |
| `docs/ASSUMPTIONS.md` | F-04's table, A-21's table, A-22's result line, and the A-21 summary row |

None of these numbers is wrong as history. The label is what is wrong: they
describe a configuration that is no longer the default.

**What finishing looks like:** each row names the configuration it was measured
under instead of calling it current, and the figure the current default really
produces — 16.496 ± 0.415, in the table under A-06 — is added beside it.

---

## P-04 — Nothing links to the figures gallery

`docsite/assets/app.js` serves a gallery of all thirteen figures at the address
`#/figures`, and `grep -rn "#/figures"` finds no link to it anywhere — not in
the sidebar, not in `shell.html`, not in any document. It is reachable only by
typing the address.

**What finishing looks like:** a link in the sidebar built from
`figure_names()`, so the list cannot go stale.

---

## P-05 — `summary.json` reports `early_stopping: true` for runs that had no validation set

Every stored run writes `early_stopping`, and it records the *setting*
(`early_stopping_patience` is not None) rather than whether early stopping could
fire. A run with `validation_fraction: 0.0` has nothing to stop on, and
`training/run.py` guards the whole mechanism on `has_validation` — yet its
summary still says `true`.

This is how the headline 16.818 ± 0.157 came to look, at a glance, like a run
with a validation split. It is not: its `preprocessing_config.yaml` says
`validation_fraction: 0.0`.

**How to check:** compare `early_stopping` in `summary.json` against
`validation_fraction` in `preprocessing_config.yaml` for any run in `reports/`.

**What finishing looks like:** the field records what actually happened, and
older run directories are left alone — they are the record of what was run, and
rule 3a forbids editing them.

---

## P-06 — Two notebooks, one convention

`notebooks/01_preprocessing_fd001.ipynb` now shows its pipeline map through
`docsite.figures.render_figure`, which keeps the picture and the code in step.
`notebooks/02_model_fd001.ipynb` draws everything with matplotlib inside its own
cells.

Not a defect, but worth a decision: notebook 02 has an architecture diagram that
`draw_architecture()` already computes from a real `ESTGNN`.

**What finishing looks like:** a decision recorded either way, and if the answer
is to share, notebook 02 uses the figure rather than a hand-drawn copy.

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
