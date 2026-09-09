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

## P-08 — The swappable-stages design is proposed but not started

`docs/DESIGN_swappable_stages.md` sets out how to add the paper's LSTM stage
while keeping the current model, and how to make the temporal, graph and
explanation stages each swappable. Nine commits, each leaving the suite green;
the riskiest is step 3, where `data/windowing/` stops throwing away 29 of the 30
cycles and every documented shape moves with it.

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

## P-03 — Results labelled "current default" were measured under the old default

The A-06 reversal changed what "the current default" means, and several tables
still attach that label to 16.818 ± 0.157, which was measured with nothing held
back.

| Where | Line |
|---|---|
| `README.md` | the results table row "**Current default** — z-score, row, no validation split (A-22)" |
| `docs/learn/02_graphs.md` | "Current default — z-score, row normalisation, nothing held back" |
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
