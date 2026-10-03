# `thesis_gnn_rul_xai` — the package

This is all the computation for the thesis. It predicts **Remaining Useful
Life** (RUL) — how many more flight cycles an engine has left — for the NASA
C-MAPSS turbofan dataset, using a graph neural network, and then explains the
predictions.

The package is a **re-implementation study** of one published paper:

> Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
> (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
> enhancement for aeroengine RUL prediction. *Aerospace Science and
> Technology*, 168, 110875. https://doi.org/10.1016/j.ast.2025.110875

That paper does not publish source code and does not specify every
implementation detail. Wherever a value or a procedure had to be chosen by us
rather than read from the paper, the choice is marked in the code with a
comment starting `ASSUMPTION` and is written up, with its reasoning, in
`docs/ASSUMPTIONS.md`. Nothing in this package is an undocumented guess.

> **Where the explanation lives.** The `.py` files in this package carry one
> line of description per function, and nothing more. Every folder has a
> `README.md` like this one, and that is where the reasoning, the traps and
> the meaning of every argument are written down. If a file leaves you with a
> question, the answer is in the README next to it.

---

## Files directly in this folder

| File | What it is |
|------|------------|
| `__init__.py` | Marks the folder as an importable package. Holds the version number and the author. |
| `config.py` | Every setting that controls the preprocessing pipeline, in one object that can be saved to and loaded from a YAML file. |

## The subfolders

Each subfolder does one job. Each has its own README with the detail.

| Folder | What it is for |
|--------|----------------|
| `data/` | Turns the raw C-MAPSS text files into model-ready arrays: loading, sensor selection, labels, smoothing, rescaling, sliding windows, and the pipeline that runs those in order. |
| `graph/` | Builds the sensor graph — the adjacency matrix that tells the network which sensors are considered connected — and normalises it. |
| `models/` | The network itself: the graph convolution layer and the full model. |
| `models/temporal/` | What each sensor node carries: the last cycle, the window's average, or what a per-sensor recurrent network read out of it. |
| `training/` | The training configuration and the training loop. |
| `evaluation/` | The scoring measures used to report results. |
| `explain/` | The Shapley-value explanations of the model's predictions, and a split of each prediction into shares along routes through the sensor graph, which the paper does not have. |
| `utils/` | Housekeeping shared by everything else: project paths, reproducible random seeds, logging. |

---

## `config.py` — why the settings live in one object

Every number that controls the pipeline is collected in a single dataclass,
`PreprocessingConfig`, instead of being written into the code where it is used.
Two reasons, both about being able to defend the results:

1. **Repeatability.** One configuration file describes an experiment
   completely. Saving that file next to the results means anyone can see
   exactly which settings produced them.
2. **Visible assumptions.** When the paper is silent and a value had to be
   chosen, that choice shows up as a named field with a comment, rather than
   being buried inside some function.

The configuration is a plain Python dataclass. `save` writes it to YAML;
`load_config` reads it back. That means an experiment can be changed without
editing any code — see the files in the repository's `config/` folder.

### The three settings that are word-choices, not numbers

`config.py` defines three sets of allowed strings.

**`normalization`** — how sensor values are rescaled before the model sees
them:

- `"minmax_per_condition"` rescales each sensor to the range 0 to 1 separately
  for each operating condition. On the sub-datasets that have several operating
  conditions (FD002 and FD004) this removes the large jumps caused by the
  aircraft flying at different altitudes and speeds, which would otherwise hide
  the much smaller changes caused by engine wear.
- `"minmax_global"` rescales each sensor to 0 to 1 using one minimum and one
  maximum for the whole dataset. FD001 has only one operating condition, so on
  FD001 this gives the same result as the per-condition version.
- `"zscore"` subtracts the mean and divides by the standard deviation, so each
  sensor ends up with mean 0 and standard deviation 1.

**`graph`** — which sensor graph the model is given:

- `"physics"` is the hand-designed matrix printed in Table 3 of the paper.
- `"complete"` connects every sensor to every other one. It stands in for the
  paper's "without physics prior" experiment, which removes the engineering
  knowledge to see what it was worth.
- `"correlation"` joins the 41 pairs of sensors whose readings move together
  most strongly on the training engines — the same number of edges as the
  paper's graph, chosen by the data instead of by an engineer. The paper never
  builds this graph; it exists to test the paper's claim that its own graph
  captures more than correlations do. `graph/README.md` explains how it is
  built and why a fixed number of edges rather than a cut-off.

**`adjacency_normalization`** — how the adjacency matrix is rescaled before the
graph layers use it. Either `"symmetric"` or `"row"`. The function that applies
it is `normalize_adjacency` in `graph/adjacency.py`.

### Things that will catch you out

**`validation_fraction` defaults to `0.2`, which is not what the paper does.**
Twenty percent of the training engines are held back, and training keeps the
network's parameters from the pass over the data that did best on them and
stops early once they stop improving. The paper describes no held-back engines
and trains for a fixed 200 passes (its Table 6). Setting the field to `0.0`
gives the paper's arrangement. Holding engines back is kept as the default
because it is the only honest basis for choosing between configurations: the
alternative is choosing on the test engines, which the test score is meant to
be independent of. (This setting was `0.0` for a while and has been changed
back; the log records both decisions under assumption A-06.)

**When there *is* a split, it is split by engine unit, not by window.** Windows
from the same engine overlap heavily, so splitting by window would leak
training data into validation. `data/windowing/README.md` explains this.

**`clip_test_labels` should stay `True`.** The paper's output layer is a
sigmoid, so the model can never predict a value above the label ceiling
`rul_max`. Leaving the test labels uncapped would charge the model for an error
it is structurally unable to avoid. This is a real effect on the reported
numbers. It can never raise two of the four measures: the root mean squared
error (RMSE, the square root of the average squared miss, in cycles) and the
NASA score (a sum of penalties that grow exponentially with each miss, faster
for a late prediction than an early one). The other two can move either way:
R-squared, the share of the labels' spread around their average that the
predictions account for, and ASUE, the average shortfall over the engines
predicted too low. `data/labeling/README.md` works both through with the real
test labels. So it must be stated
whenever those numbers are compared with published ones. (Corrected
2026-10-02: this said capping can only move all the reported numbers in the
model's favour; found by the search for that claim after the review of the
playground's information cards on 2026-09-26.)

**`graph` and `adjacency_normalization` describe the model, not the data**, and
still live in the preprocessing configuration. That is on purpose: this is the
object that builds the matrix and the object that gets saved beside a result. A
setting that decides what the model sees and is not written down is a setting
that cannot be reproduced.

**`load_config` rejects unknown keys.** If a YAML file contains a key that is
not a field of `PreprocessingConfig`, loading raises an error rather than
ignoring it. A silently ignored typo in a settings file is a common and
hard-to-find cause of wrong results.

**The settings are checked as soon as the object is built.** `__post_init__`
validates the subset name, the window size and stride, the label ceiling, the
smoothing strength, the graph options and the validation fraction. Catching
an impossible configuration there, before any data is touched, gives a clear
error message instead of a confusing failure deep inside the pipeline.

---

## The judgement calls made in this folder

Each of these is a place where the paper does not say what to do. The code in
parentheses after each one (such as "assumption A-05") points at the entry in
`docs/ASSUMPTIONS.md` that gives the full reasoning; nothing here needs it to
be understood.

Several of these settings were not chosen independently. Where the paper is
silent, the author's rule was to pick the option that lands nearest the
paper's own reported error (assumption A-22), so those settings are not
independent of the figure they are compared with.

- **The label ceiling is 125 cycles.** RUL labels are capped, so an engine that
  is still healthy is simply labelled "125 cycles left". The paper defines this
  ceiling symbolically in its Equation 19 but never prints the value. 125 was
  read off the True RUL curve of the paper's Fig. 7; it also matches Section
  4.2, which describes predictions holding "around 125" while an engine is
  healthy, and it is the value used by the closely related work of Jin et al.
  (2025) on the same dataset (assumption A-01).

- **The window stride is 1 cycle.** A training example is 30 consecutive flight
  cycles (the paper's Table 6). How far the window moves between one example
  and the next is never stated. A stride of 1 produces the largest number of
  training examples and is the usual choice on C-MAPSS (assumption A-05).

- **Sensor values are rescaled with z-scores.** The paper never describes its
  input scaling at all. The default here was picked by the nearest-to-the-paper
  rule above, and it is worth 3.83 cycles against the per-condition min-max
  scaling of Jin et al. (2025) — the largest measured effect of any single
  choice in the project (assumption A-03, measured in finding F-10).

- **The smoothing strength is 0.5.** Sensor traces are passed through an
  exponentially weighted moving average filter, whose one parameter runs from
  just above 0 (heavy smoothing) to 1 (no smoothing). Section 3.2.3 of the
  paper is self-contradictory here: it attaches the number 0.5 to the previous
  *smoothed value* rather than to the smoothing parameter. Reading it as the
  smoothing parameter is the only interpretation that makes the sentence mean
  anything, so 0.5 is used (assumption A-04). Setting the field to `None`
  turns smoothing off entirely.

- **The "without physics prior" experiment uses a fully connected graph.** The
  paper names that experiment but never says what replaces the hand-designed
  matrix (assumption A-19).

- **The data-driven graph keeps the 41 most strongly correlated sensor pairs.**
  The paper never builds such a graph, so both the rule for joining sensors and
  the number of edges are choices (assumption A-27).

- **The adjacency matrix is row-normalised.** The paper describes normalising
  the matrix with self-loops added but never prints the formula, and two forms
  fit its description. With no engines held back, row normalisation measured
  0.25 cycles closer than the symmetric form under the learning-rate schedule
  the project uses now, a single drop at the halfway point (p = 0.068: if the
  two forms were equally good, five seeds, each trained once with each form,
  would show a gap at least this large against its seed-to-seed spread, in one
  direction or the other, 6.8 times in 100, more often than the usual 1-in-20
  threshold), and 0.35
  closer (p = 0.047) under the schedule used until 2026-09-01, which shrank the
  rate a little after every pass over the training data. It was selected by
  the nearest-to-the-paper rule; with engines held back the two cannot be told
  apart (assumption A-21).

  **Correction, 2026-10-02.** This paragraph gave 0.35 cycles at p = 0.047 as
  the current figure; that figure was measured under the superseded schedule.
  Found by setting it beside the table in the record of this choice, which
  holds both figures with their schedules (assumption A-21).

- **Twenty percent of the training engines are held back, and training stops
  early on them.** As above: the paper's Table 6 states 200 passes over the
  data and describes no held-back engines (assumptions A-06 and A-18).

---

## What each setting is

Every field of `PreprocessingConfig`, in the order it appears in `config.py`.
The section above explains the ones that need more than a line.

| Setting | Default | What it is |
|---|---|---|
| `subset` | `"FD001"` | Which of the four sub-datasets to use. |
| `window_size` | `30` | How many consecutive flight cycles go into one training example. From the paper's Table 6. |
| `window_stride` | `1` | How far the window moves between one example and the next, in cycles (assumption A-05). |
| `rul_max` | `125` | The ceiling put on the label, in cycles (assumption A-01). |
| `clip_test_labels` | `True` | Whether the same ceiling is put on the test labels. Leave it on; see above (assumption A-02). |
| `normalization` | `"zscore"` | How the sensors are put on a comparable scale (assumption A-03). |
| `smoothing_alpha` | `0.5` | How hard to smooth, between 0 and 1. Larger follows the raw signal more closely; smaller smooths harder. `None` turns smoothing off (assumption A-04). |
| `graph` | `"physics"` | Which sensor graph the network gets: `"physics"`, `"complete"` (assumption A-19) or `"correlation"` (assumption A-27). |
| `adjacency_normalization` | `"row"` | How that graph is rescaled before use (assumption A-21). |
| `validation_fraction` | `0.2` | What share of the **engines** to hold back. `0.0` holds nothing back, which is the paper's arrangement; 0.2 is this project's addition. Never split by window (assumption A-06). |
| `seed` | `42` | Fixes which engines get held back. |
| `notes` | — | Free text saved with the settings. It changes no number, so the digest below leaves it out. |

`save(path)` writes them to a YAML file, `to_dict()` returns them as a plain
dictionary, and `load_config(path)` reads a file back.

### The six characters that name a configuration: `digest()`

`scripts/train.py` saves every run in a folder whose name carries `cfg`
followed by six characters, such as `cfg818b40`. Those six characters are the
configuration's **digest**: a short code computed from the settings, the same
every time for the same settings and different when any one of them changes.
It is there so that two runs whose settings differ anywhere get different
folder names, even when nobody thought of writing the setting that differs into
the name in words.

`digest()` makes it in three steps. For `config/fd001.yaml`:

1. `settings_without_notes(config.to_dict())` keeps the eleven settings and
   leaves out `notes`, the one field of free prose. A sentence changes no
   number, so it must not change the name.
2. Those eleven are written as JSON text, a standard way of writing data as
   characters, with the names in alphabetical order so that the same settings
   always become the same text:

   ```text
   {"adjacency_normalization": "row", "clip_test_labels": true, "graph": "physics", "normalization": "zscore", "rul_max": 125, "seed": 42, "smoothing_alpha": 0.5, "subset": "FD001", "validation_fraction": 0.2, "window_size": 30, "window_stride": 1}
   ```

3. SHA-256, a standard function that turns any text into 64 hexadecimal
   characters (the digits 0 to 9 and the letters a to f), turns that line into
   `818b40380ec2c6a8...`; change a single character of the text and the output
   changes beyond recognition. The first six characters, `818b40`, are the
   digest.

`PreprocessingConfig()`, built without a file, carries a different sentence in
`notes` and the same eleven settings, so it gives `818b40` as well.
`config/fd001_uncapped.yaml` differs in one setting, `clip_test_labels`, and
gives `7c2c29`.

`settings_without_notes(values)` is the one place in the package and its
scripts that decides which saved fields count as settings. The digest uses it,
and so do the four places that ask whether two sets of settings are alike:
`find_runs` in `training/stored.py`, the playground's comparison of two runs,
`scripts/baselines.py`'s check of a second run, and the store in
`scripts/verify_claims.py` that trains one network per set of settings and
hands it to every check that asks for the same settings again.

Two settings can also be written differently and still be equal. A YAML file
may hold `validation_fraction: 0` where another holds `0.0`; Python counts the
two as equal, so `find_runs` would match them, but as JSON text they are `0`
and `0.0`, and the digest would differ. So `PreprocessingConfig` stores every
number as the type its field declares the moment it is built:
`PreprocessingConfig(validation_fraction=0)` holds `0.0` and gives `2c9b5b`,
the same as `validation_fraction=0.0`; until that was added on 2026-10-03 it
kept the `0` and gave `e950d9`. Whole-number fields go the other way, so
`rul_max=125.0` is held as `125`. With both in place, the saved settings of a
run that `find_runs` returns give the same digest as the settings it was asked
for, whatever code that run's folder name carries. All 141 saved settings
files under `reports/` on 2026-10-03 already held every number as its declared
type, so this changed none of their digests.

**Until 2026-10-03 the notes were hashed too**, so editing a sentence renamed
every run made afterwards: `PreprocessingConfig()` gave `e3a331` and
`config/fd001.yaml` gave `ba82e3` for the same eleven settings. Stored runs
keep the names they were given, so a run made since then can carry a
different `cfg` part from a stored run with identical settings.
`training/README.md` lists every old code found in a stored folder name beside
its new one.

## Reading order

`config.py` is the right first file in the package: it has no internal
dependencies, and this README explains where each of its values comes from.
After that, the repository's top-level `README.md` (section 9) suggests an
order for the rest.
