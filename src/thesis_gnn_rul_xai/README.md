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
| `training/` | The training configuration and the training loop. |
| `evaluation/` | The scoring measures used to report results. |
| `explain/` | The Shapley-value explanations of the model's predictions. |
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
- `"complete"` connects every sensor to every other one. This is the paper's
  "without physics prior" ablation.

**`adjacency_normalization`** — how the adjacency matrix is rescaled before the
graph layers use it. Either `"symmetric"` or `"row"`. The function that applies
it is `normalize_adjacency` in `graph/adjacency.py`.

### Things that will catch you out

**`validation_fraction` defaults to `0.0`, which means there is no validation
split at all.** Unless a configuration file asks for one, every engine is used
for training. This is deliberate: the paper describes no held-out set and
trains for a fixed 200 epochs (its Table 6). Holding engines out and then
restoring the best-validation checkpoint would be validation-based model
selection, which the paper never describes, so the reproduction does not do it.
The file `config/fd001_validated.yaml` puts the split back, and exists for
comparing one configuration against another.

**When there *is* a split, it is split by engine unit, not by window.** Windows
from the same engine overlap heavily, so splitting by window would leak
training data into validation. `data/windowing/README.md` explains this.

**`clip_test_labels` should stay `True`.** The paper's output layer is a
sigmoid, so the model can never predict a value above the label ceiling
`rul_max`. Leaving the test labels uncapped would charge the model for an error
it is structurally unable to avoid. This is a real effect on the reported
numbers and can only move them in the model's favour, so it must be stated
whenever those numbers are compared with published ones.

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
smoothing strength, the two graph options and the validation fraction. Catching
an impossible configuration there, before any data is touched, gives a clear
error message instead of a confusing failure deep inside the pipeline.

---

## The judgement calls made in this folder

Each of these is a place where the paper does not say what to do. The short
code (`A-05`, `A-19`, and so on) is the entry in `docs/ASSUMPTIONS.md` that
gives the full reasoning. Two of the entries mentioned below are not choices
about this project but observations about the paper: `A-22` records the rule
used to pick among settings the paper leaves open — choose the option that
lands nearest the paper's own reported figure — and `F-10` is a measurement
showing that the undescribed input scaling accounts for most of the remaining
gap to that figure.

- **The label ceiling is 125 cycles.** RUL labels are capped, so an engine that
  is still healthy is simply labelled "125 cycles left". The paper defines this
  ceiling symbolically in its Equation 19 but never prints the value. 125 was
  read off the True RUL curve of the paper's Fig. 7; it also matches Section
  4.2, which describes predictions holding "around 125" while an engine is
  healthy, and it is the value used by the closely related work of Jin et al.
  (2025) on the same dataset. Recorded as **A-01**.

- **The window stride is 1 cycle.** A training example is 30 consecutive flight
  cycles (the paper's Table 6). How far the window moves between one example
  and the next is never stated. A stride of 1 produces the largest number of
  training examples and is the usual choice on C-MAPSS. Recorded as **A-05**.

- **Sensor values are rescaled with z-scores.** The paper never describes its
  input scaling at all. The default here was picked under the rule in A-22, and
  it is worth 3.83 cycles against the per-condition min-max scaling of Jin et
  al. (2025) — the largest measured effect of any single choice in the project.
  Recorded as **A-03**, with the measurement in **F-10**.

- **The smoothing strength is 0.5.** Sensor traces are passed through an
  exponentially weighted moving average filter, whose one parameter runs from
  just above 0 (heavy smoothing) to 1 (no smoothing). Section 3.2.3 of the
  paper is self-contradictory here: it attaches the number 0.5 to the previous
  *smoothed value* rather than to the smoothing parameter. Reading it as the
  smoothing parameter is the only interpretation that makes the sentence mean
  anything, so 0.5 is used. Recorded as **A-04**. Setting the field to `None`
  turns smoothing off entirely.

- **The "without physics prior" ablation uses a fully connected graph.** The
  paper names that ablation but never says what replaces the hand-designed
  matrix. Recorded as **A-19**.

- **The adjacency matrix is row-normalised.** The paper describes normalising
  the matrix with self-loops added but never prints the formula, and two forms
  fit its description. Row normalisation measured 0.35 cycles closer than the
  symmetric form (p = 0.047), and was selected under A-22. Recorded as
  **A-21**.

- **No validation split and no early stopping in the reproduction.** As above:
  the paper's Table 6 states 200 epochs and describes no held-out set.
  Recorded as **A-06** and **A-18**.

---

## What each setting is

Every field of `PreprocessingConfig`, in the order it appears in `config.py`.
The section above explains the ones that need more than a line.

| Setting | Default | What it is |
|---|---|---|
| `subset` | `"FD001"` | Which of the four sub-datasets to use. |
| `window_size` | `30` | How many consecutive flight cycles go into one training example. From the paper's Table 6. |
| `window_stride` | `1` | How far the window moves between one example and the next, in cycles. **A-05.** |
| `rul_max` | `125` | The ceiling put on the label, in cycles. **A-01.** |
| `clip_test_labels` | `True` | Whether the same ceiling is put on the test labels. Leave it on; see above. **A-02.** |
| `normalization` | `"zscore"` | How the sensors are put on a comparable scale. **A-03.** |
| `smoothing_alpha` | `0.5` | How hard to smooth, between 0 and 1. Larger follows the raw signal more closely; smaller smooths harder. `None` turns smoothing off. **A-04.** |
| `graph` | `"physics"` | Which sensor graph the network gets. **A-19** covers the `"complete"` option. |
| `adjacency_normalization` | `"row"` | How that graph is rescaled before use. **A-21.** |
| `validation_fraction` | `0.0` | What share of the **engines** to hold back. `0.0` holds nothing back, which is the paper's arrangement. Never split by window. **A-06.** |
| `seed` | `42` | Fixes which engines get held back. |
| `notes` | — | Free text saved with the settings. |

`save(path)` writes them to a YAML file, `to_dict()` returns them as a plain
dictionary, and `load_config(path)` reads a file back.

## Reading order

`config.py` is the right first file in the package: it has no internal
dependencies, and this README explains where each of its values comes from.
After that, the repository's top-level `README.md` (section 9) suggests an
order for the rest.
