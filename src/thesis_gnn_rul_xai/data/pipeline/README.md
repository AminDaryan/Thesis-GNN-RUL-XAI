# `pipeline` — run all nine steps in order and hand back arrays ready for training

This file contains no preprocessing of its own. Every step it runs lives in one
of the six sibling files; this one decides **which steps, in what order, on
which rows**, and packages the result so that everything needed to check a run
travels with it.

If you only want the end result, this is the one function to call:

```python
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
prep = run_preprocessing()
```

For what C-MAPSS, a flight cycle and remaining useful life are, see
[the data folder overview](../README.md).

---

## What this file is given, and what it hands back

| | |
|---|---|
| **Given** | A settings object (`PreprocessingConfig`), and where the raw files are |
| **Hands back** | A `PreparedData` — three sets of examples, the sensor graph, the fitted scaling, and a record of everything observed on the way |

Where it sits in the chain: it *is* the chain.

```
raw .txt  ->  loading  ->  sensors  ->  labeling  ->  smoothing
          ->  split  ->  normalization  ->  windowing  ->  PreparedData
```

---

## The settings that drive it

Every one of these is a knob rather than a fixed value, and the reason is
recorded: the settings object is saved beside every run, so the numbers a result
was produced with cannot drift away from the result.

```python
PreprocessingConfig()
```

```
    subset                     'FD001'
    window_size                30
    window_stride              1
    rul_max                    125
    clip_test_labels           True
    normalization              'zscore'
    smoothing_alpha            0.5
    graph                      'physics'
    adjacency_normalization    'row'
    validation_fraction        0.2
    seed                       42
```

Several of those are this project's choices rather than values the paper
printed: how far the window moves (`window_stride`, assumption A-05), the label
ceiling (`rul_max`, assumption A-01), capping the test labels too
(`clip_test_labels`, assumption A-02), the input scaling (`normalization`,
assumption A-03), the smoothing strength (`smoothing_alpha`, assumption A-04),
the rescaling of the graph (`adjacency_normalization`, assumption A-21), and
holding any engines back at all (`validation_fraction`, assumption A-06).
`graph` can also name four graphs the paper never uses (assumptions A-19,
A-27 and A-41).

---

## The public names

| Name | What it is | What it gives you back |
|---|---|---|
| `run_preprocessing(config=None, data_dir=None, keep_frames=False)` | Runs all nine steps. | A `PreparedData` |
| `PreparedData` | Everything training needs, plus everything needed to check it. | An object with the fields listed below, a `summary()` and a `save()` |

---

## `run_preprocessing` — the nine steps, on a real run

### The input

Nothing is required — every argument has a default:

```python
prep = run_preprocessing()                       # the settings above, data/raw/
prep = run_preprocessing(cfg, data_dir=..., keep_frames=True)
```

`config` defaults to a fresh `PreprocessingConfig`, `data_dir` to the project's
`data/raw/`, and `keep_frames` to `False`. So the real input is three text
files and eleven settings.

### The output first, so the steps have something to refer to

This is the complete, unedited log of a real FD001 run with the default
settings:

```
INFO    | ...data.loading       | Loading FD001 from D:\...\data\raw
INFO    | ...data.loading       | FD001 matches the expected counts, those of NASA's files (Wen et al.'s Table 5 prints the same engine counts).
INFO    | ...data.loading       | Loaded FD001: 20631 training rows over 100 engines, 13096 test rows over 100 engines.
WARNING | ...data.pipeline      | The paper keeps sensor(s) ['P15'], but in this data they carry no
                                  usable variation. They are kept anyway, because the adjacency matrix
                                  is defined over the paper's fixed 14-sensor set. Report this
                                  observation rather than silently changing the selection.
INFO    | ...data.smoothing     | Applied exponentially weighted moving average (alpha=0.500) to 14 columns across 100 engines.
INFO    | ...data.smoothing     | Applied exponentially weighted moving average (alpha=0.500) to 14 columns across 100 engines.
INFO    | ...data.normalization | Detected 1 distinct operating condition(s) from the settings.
INFO    | ...data.normalization | A single operating condition: clustering is not needed.
INFO    | ...data.normalization | Fitted 'zscore' normalisation on 14 column(s) across 1 condition(s).
INFO    | ...data.windowing     | Built 17731 window(s) of shape (14, 30) from 100 engine(s).
INFO    | ...data.windowing     | Built 100 window(s) of shape (14, 30) from 100 engine(s).
INFO    | ...data.pipeline      | Preprocessing complete: 17731 training, 0 validation, 100 test example(s).
```

Note the smoothing line appears **twice** — once for the training table and
once for the test table. So does the windowing line, because training and test
are windowed differently.

### The steps, in order, with what actually flows through

| # | Step | Which file does it | Goes in | Comes out |
|---|---|---|---|---|
| 1 | Load the three files, attach labels | [`loading`](../loading/README.md) | 3 text files | train `(20631, 27)`, test `(13096, 27)` |
| 2 | Diagnose the sensor selection | [`sensors`](../sensors/README.md) | train table | a three-way report; **nothing is dropped** |
| 3–4 | Pick the 14 columns, cap the labels | [`labeling`](../labeling/README.md) | train + test | same shape, `rul` now capped at 125 |
| 5 | Smooth the 14 sensor columns | [`smoothing`](../smoothing/README.md) | train + test | same shape, sensor columns smoothed |
| 6 | Decide which engines are held back | [`windowing`](../windowing/README.md) | 100 engine numbers | two lists of engine numbers |
| 7 | Operating conditions, then rescaling | [`normalization`](../normalization/README.md) | train + test | `operating_condition` column added; sensors rescaled |
| 8 | Cut into windows | [`windowing`](../windowing/README.md) | train + test tables | train `(17731, 14, 30)`, test `(100, 14, 30)` |
| 9 | Apply the split from step 6 | [`windowing`](../windowing/README.md) | 17731 windows | training and validation sets |

### Step 2 — the diagnosis that changes nothing, and the warning it produces

```python
detected = detect_uninformative_sensors(data.train)
uninformative = set(detected["constant"]) | set(detected["low_resolution"])
kept_but_uninformative = sorted(uninformative & set(PAPER_SENSOR_SYMBOLS))
```

The set intersection asks: *which sensors does the paper keep that this data
says carry no information?* On FD001 the answer is `['P15']`, the bypass-duct
pressure, which takes exactly two distinct values across the whole training set.

**It is kept anyway.** The adjacency matrix is defined over the paper's fixed 14
nodes, and removing one would change the graph into something the paper did not
describe. So the run reports it loudly instead:

```
WARNING | The paper keeps sensor(s) ['P15'], but in this data they carry no usable
variation. They are kept anyway, because the adjacency matrix is defined over the
paper's fixed 14-sensor set. Report this observation rather than silently changing
the selection.
```

That warning fires on every FD001 run and is expected. It is a finding about
the paper, not a fault in the code.

### Steps 3 and 4 — an order that differs from the paper's, and why it is safe

The comment in the code is explicit about it:

```python
# Fig. 1 of the paper caps the labels after smoothing; here it happens
# before. That cannot change anything: capping touches only the label
# column and smoothing touches only the sensor columns, so neither reads
# what the other writes.
```

That is a real argument, not a hand-wave: `apply_piecewise_rul` writes only
`rul`, and `apply_ewma` writes only the 14 sensor columns and reads only those.
Two operations that touch disjoint columns commute.

Note also that the test labels are capped only `if config.clip_test_labels` —
capping them at all is this project's choice (assumption A-02), and it can
never raise two of the four reported measures: the root mean squared error
(RMSE, the square root of the average squared miss, in cycles) and the NASA
score (a sum of penalties that grow exponentially with each miss). The other
two can move either way: R-squared, the share of the labels' spread around
their average that the predictions account for, and ASUE, the average
shortfall over the engines predicted too low.
[`labeling`](../labeling/README.md) works both through with the real test
labels. (Corrected 2026-10-02: this said
capping can only move reported error favourably; found by the search for that
claim after the review of the playground's information cards on 2026-09-26.)

### Step 6 — the split happens *before* anything is measured

This is the step whose placement matters most, and it sits where it does on
purpose:

```python
train_units, validation_units = split_units_by_id(
    np.unique(train[UNIT_COL].to_numpy()),
    validation_fraction=config.validation_fraction,
    seed=config.seed,
)
```

Figure 1 of the paper prints the reverse order — normalisation, *then* the train
and test datasets — and never says which engines the scaling is fitted on.
Deciding the split first is this project's choice (assumption A-25), and the
reason is the one from
[`normalization`](../normalization/README.md): if the split happened after
fitting, the held-back engines would have helped decide the mean and the spread,
and their error would no longer measure how the model does on engines the
preparation never saw. Nothing guarantees which way it would move; the usual
finding is that it looks better than it really is. (Corrected 2026-10-02: this
said it would look better, which is the usual direction but not a certain one;
found by the search for that claim after the review of the playground's
information cards on 2026-09-26. Assumption A-25 in
[the assumptions log](../../../../docs/ASSUMPTIONS.md) has the argument.)

### Step 7 — fit on training engines only, apply to everything

```python
training_rows = train[UNIT_COL].isin(train_units)
encoder = OperatingConditionEncoder(seed=config.seed).fit(train[training_rows])
train = encoder.transform(train)
test  = encoder.transform(test)
```

`fit` sees `train[training_rows]` — the training engines alone. `transform` is
then applied to the whole training table (held-back engines included) and to the
test table.

One line here is easy to overlook and the code calls it out:

```python
# Worked out again rather than reused: `train` now points at a different
# table than it did above, so the old mask would line up with nothing.
training_rows = train[UNIT_COL].isin(train_units)
```

`encoder.transform` returns a **new** table, so the variable `train` no longer
refers to the object the first mask was computed from. Reusing the stale mask
would be a subtle alignment bug of exactly the kind that produces plausible
wrong numbers.

### Step 9 — the split applied, and checked

```python
validation_windows = train_windows.subset_by_units(validation_units)
train_windows      = train_windows.subset_by_units(train_units)

overlap = set(train_windows.units.tolist()) & set(validation_windows.units.tolist())
if overlap:
    raise AssertionError(
        f"Engines {sorted(overlap)} appear in both the training and the "
        "validation split. This would invalidate the experiment."
    )
```

The check is defensive — `split_units_by_id` cannot produce an overlap — but the
consequence of one would be an invalid experiment that looked entirely normal,
so it is verified rather than trusted.

### The sensor graph

Built last, from settings rather than fixed values:

```python
if config.graph == "physics":
    adjacency_raw, symbols = build_physics_adjacency()
elif config.graph == "complete":
    adjacency_raw, symbols = build_fully_connected_adjacency(len(PAPER_SENSOR_SYMBOLS))
elif config.graph == "random":
    adjacency_raw, symbols = build_random_adjacency(config.seed)
elif config.graph == "rewired":
    adjacency_raw, symbols = build_rewired_adjacency(config.seed)
elif config.graph == "correlation":
    physics_edges = int(build_physics_adjacency()[0].sum() // 2)
    adjacency_raw, symbols = build_correlation_adjacency(
        train[train[UNIT_COL].isin(train_units)], columns, n_edges=physics_edges,
    )
adjacency_normalized = normalize_adjacency(adjacency_raw, method=config.adjacency_normalization)
```

The code comment explains why these are settings: *"so that the graph saved next
to a result is the one the network was actually given. Before that was true, a
run could store one graph and train on another, and nothing would say so."*

**The correlation graph is the one graph built from data, so it is the one that
could leak.** It is measured only on the rows of the training engines, after
smoothing and rescaling, for the reason step 6 fits the rescaling on them: had
the held-back engines helped decide which sensors are joined, their error would
no longer be a fair measure of engines the model never saw, usually looking
better than it is, though nothing guarantees that direction. (Corrected
2026-10-02: this said their error would look better than it is; found by the
search for that claim after the review of the playground's information cards
on 2026-09-26.) Because each seed holds back different engines, each
seed can get a slightly different graph. `graph/README.md` explains why it
keeps exactly as many edges as the paper's graph (assumption A-27), and
`tests/test_pipeline.py` checks it is built from the training rows alone.

**The random and the rewired graph read no data at all.** They are drawn from
`config.seed` alone, so they cannot leak anything, and the same seed always
gives the same graph; `graph/README.md` explains how they are drawn
(assumption A-41). The record of what was observed names the seed as
`graph_seed`, which is `None` for the other three graphs. (The trace below was
printed before that entry existed, so it does not show it.)

```
  adjacency_raw       shape (14, 14) dtype float32
  row 0:                     [0. 1. 0. 1. 1. 1. 0. 1. 0. 1. 1. 0. 0. 0.]
  adjacency_normalized row 0: [0.125 0.125 0. 0.125 0.125 0.125 0. 0.125 0. 0.125 0.125 0. 0. 0.]

  raw row 0 sums to 7.0  |  normalized row 0 sums to 1.0
  sensor_symbols: ('T24', 'T30', 'T50', 'P15', 'P30', 'Nf', 'Nc', 'Ps30',
                   'phi', 'NRf', 'BPR', 'htBleed', 'W31', 'W32')
```

Two things to read there. The rescaled row sums to exactly 1, which is what
"row" normalisation means. And position 0 of the rescaled row is `0.125` where
the raw row had `0` — the rescaling adds a connection from each sensor to
itself, so there are 8 entries of 0.125 rather than 7. Which rescaling formula is
used is this project's choice (assumption A-21).

---

## `PreparedData` — every field

### What it is given, and what it hands back

A container rather than a function. `run_preprocessing` fills it with twelve
fields, and it offers two methods: `summary()` for a table to look at, and
`save()` for writing five files to disk.

### The twelve fields

| Field | What it holds |
|---|---|
| `config` | The exact settings that produced this object |
| `train`, `validation`, `test` | The three sets of examples. `validation` is empty when nothing was held back, which is the default |
| `adjacency_raw`, `adjacency_normalized` | The sensor graph before and after rescaling |
| `sensor_symbols` | The sensor names, in the order of the graph's rows |
| `normalizer`, `condition_encoder` | The two fitted objects, kept so what they measured can be inspected and saved |
| `diagnostics` | A free-form record of what the run observed |
| `raw`, `frames` | The intermediate tables, or `None` unless `keep_frames=True` |

### `summary()`

With nothing held back (`validation_fraction=0.0`):

```
            examples  engines feature_shape  target_min  target_max
split
train          17731      100      (14, 30)         0.0       125.0
validation         0        0      (14, 30)         NaN         NaN
test             100      100      (14, 30)         7.0       125.0
```

The default run, with 20 % of the engines held back (A-06):

```
            examples  engines feature_shape  target_min  target_max
split
train          14459       80      (14, 30)         0.0       125.0
validation      3272       20      (14, 30)         0.0       125.0
test             100      100      (14, 30)         7.0       125.0

  train engines: 80   validation engines: 20   overlap: set()
```

14459 + 3272 = 17731 — the same windows, divided by engine.

The test `target_max` is 125 and not higher because `clip_test_labels` is on
(**A-02**). Its `target_min` of 7 is a real test engine that was 7 flights from
failing when recording stopped.

### `diagnostics`, key by key

```
  subset: FD001
  data_dir: D:\...\data\raw
  sensor_diagnostics:
      constant        ['Nf_dmd', 'P2', 'PCNfR_dmd', 'T2', 'epr', 'farB']
      low_resolution  ['P15']
      informative     ['BPR', 'NRc', 'NRf', 'Nc', 'Nf', 'P30', 'Ps30', 'T24',
                       'T30', 'T50', 'W31', 'W32', 'htBleed', 'phi']
  kept_but_uninformative: ['P15']
  n_operating_conditions: 1
  graph:
      n_nodes                14
      n_edges                41
      density                0.450549453496933
      is_symmetric           True
      has_self_loops         False
      is_connected           True
      degrees                {'T24': 7, 'T30': 8, 'T50': 4, 'P15': 4, 'P30': 7,
                              'Nf': 4, 'Nc': 8, 'Ps30': 7, 'phi': 6, 'NRf': 4,
                              'BPR': 4, 'htBleed': 7, 'W31': 8, 'W32': 4}
      isolated_nodes         []
      articulation_points    ['T24']
      diameter               4
      distance_histogram     {1: 41, 2: 25, 3: 21, 4: 4}
      algebraic_connectivity 0.5564121622781517
  graph_kind: physics
  adjacency_normalization: row
  splits:
      train_examples         17731
      validation_examples    0
      test_examples          100
      train_engines          [1, 2, 3, ... 100]
      validation_engines     []
```

What each graph entry means: **n_edges** is how many sensor pairs are connected;
**density** is that count as a fraction of all possible pairs; **is_symmetric**
means every connection runs both ways; **is_connected** means every sensor can be
reached from every other; **degrees** is how many neighbours each sensor has;
**isolated_nodes** would be sensors connected to nothing; **articulation_points**
are sensors whose removal would split the graph in two — here just `T24`;
**diameter** is the largest number of steps between any two sensors;
**distance_histogram** counts pairs at each distance; **algebraic_connectivity**
is a single number summarising how hard the graph is to cut apart, larger
meaning harder.

### `keep_frames=True`

```
  raw   : CMAPSSData -> CMAPSSData(subset='FD001', train_rows=20631, train_units=100,
                                   test_rows=13096, test_units=100)
  frames: ['train', 'test'] -> train (20631, 28)  test (13096, 28)
```

The intermediate tables, after every step. 28 columns rather than the original
26, because `rul` and `operating_condition` were added on the way. It costs
memory — the full tables stay alive alongside the windowed arrays — and is what
a notebook wants when it needs to draw what happened.

### `save()`

```python
prep.save(directory)
```

```
  adjacency.npz                          913 bytes
  config.yaml                            400 bytes
  diagnostics.json                     2,639 bytes
  normalization_statistics.csv         1,209 bytes
  windows.npz                        902,095 bytes
```

| File | What it holds |
|---|---|
| `windows.npz` | The three splits: `train_features`, `train_targets`, `train_units`, `train_end_cycles`, the same four for `val` and `test`, and `sensor_columns` |
| `adjacency.npz` | `adjacency_raw`, `adjacency_normalized`, `sensor_symbols` |
| `normalization_statistics.csv` | 14 rows × 7 columns: `condition, column, min, max, mean, std, n_rows` |
| `config.yaml` | The exact settings used |
| `diagnostics.json` | The record above |

Storing `sensor_columns` inside `windows.npz` and `sensor_symbols` inside
`adjacency.npz` is the same safeguard twice: the order the data was in travels
with the data, so it cannot be re-derived wrongly later.

---

## The judgement calls recorded in this file

### The order of the steps is the paper's, not ours

This one turned out **not** to be a choice. Figure 1 of the paper prints the
chain explicitly, so the pipeline follows it. The log keeps the entry under the
number it was first given (assumption A-08, withdrawn), and it now records why
it is not an assumption — plus a
measured caution about what that order does on the multi-condition sub-datasets
FD002 and FD004. Read it before running this pipeline on those two.

The one departure — capping labels before smoothing rather than after — is
argued above and cannot change any value, because the two steps touch disjoint
columns.

### A-25 — the split is decided before the scaling is fitted

Figure 1 puts normalisation before the train and test datasets and never names
the fitting set. This pipeline splits first and fits on the training engines
only, because the alternative lets held-back engines influence the numbers used
to rescale them.

Both entries are in [the assumptions log](../../../../docs/ASSUMPTIONS.md).

---

## What can go wrong

This function raises little of its own — the guards live in the steps it calls,
so a missing file surfaces as [`loading`](../loading/README.md)'s
`FileNotFoundError` with the NASA download link, a bad setting as the relevant
step's `ValueError`, and so on.

The two things it does check itself:

**Engines in both splits**, which would invalidate the experiment:

```
AssertionError: Engines [...] appear in both the training and the validation
split. This would invalidate the experiment.
```

This is defensive and should be unreachable.

**A sensor the paper keeps that the data says is useless** — a warning, not an
error, and expected on every FD001 run.

The failure this file cannot catch is a **settings mismatch**: running with one
configuration and reporting against another. That is what saving `config.yaml`
beside every run is for, and why the graph is a setting rather than a constant.

---

The key idea is: every step here belongs to another file, and what this one
contributes is the order, the decision of which rows each measurement is allowed
to see, and a record complete enough that a stored result can be traced back to
the settings that produced it.
