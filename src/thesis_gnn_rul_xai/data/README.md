# `data` — from raw NASA text files to model input

This folder does all of the preprocessing. It reads the C-MAPSS text files,
picks the sensors, builds the labels, smooths and rescales the signals, cuts
each engine history into fixed-length windows, and hands back plain NumPy
arrays.

If you only want the end result, call `run_preprocessing()` from `pipeline.py`.
Everything else in this folder is one step of what that function does.

> **The `.py` files here carry one line of description each, and nothing more.**
> All of the explanation lives in this file. The section
> [What each name means](#what-each-name-means) at the end lists every function
> and value the folder offers, and what it gives you back.

## Background you need before reading the code

**C-MAPSS** is a NASA simulator of a large turbofan aircraft engine. NASA used
it to generate a public benchmark dataset in which simulated engines run until
they fail while 21 virtual sensors record the engine state once per flight
cycle. A **flight cycle** is one take-off, flight and landing. It is the unit of
time everywhere in this project — there are no seconds or hours anywhere.

**RUL** stands for Remaining Useful Life: the number of flight cycles an engine
will still run before it fails. It is the number the model has to predict. A
label of 47 means "this engine fails 47 flight cycles from now".

Each sub-dataset (`FD001` … `FD004`) comes as three plain text files:

| File | What it holds |
|---|---|
| `train_FD001.txt` | Complete run-to-failure histories. Every engine is recorded from new until it fails. |
| `test_FD001.txt` | Partial histories. Recording stops at some point *before* failure. |
| `RUL_FD001.txt` | One number per test engine: the true remaining life at the last row that engine has in the test file. |

The two history files have 26 columns separated by spaces, no header row, and a
trailing space at the end of every line (which would otherwise be read as an
empty 27th column). The RUL file has a single column. Column 1 is the engine
number, column 2 is the cycle count, columns 3–5 are three operational settings
(altitude, Mach number, throttle resolver angle), and columns 6–26 are the 21
sensor measurements.

The dataset is described in Saxena, Goebel, Simon & Eklund (2008), *Damage
propagation modeling for aircraft engine run-to-failure simulation*, IEEE PHM
2008, pp. 1–9. The paper this project re-implements is Wen, Liang, Choy, Fei,
Li & Keshtegar (2026), *Explainable spatiotemporal graph neutral network with
dynamic SHAP enhancement for aeroengine RUL prediction*, Aerospace Science and
Technology 168, 110875.

## The files

| File | What it does |
|---|---|
| `loading.py` | Reads the three text files, checks their shape, and adds a `rul` column to every row. |
| `sensors.py` | Names all 21 sensors, fixes the 14 the paper keeps and their order, and offers a diagnostic that finds sensors carrying no information. |
| `labeling.py` | Caps the RUL label at a ceiling, producing the piecewise-linear target. |
| `smoothing.py` | Exponentially weighted moving average over each engine's sensor signals. |
| `normalization.py` | Groups rows into operating conditions with k-means, then rescales the sensor columns. |
| `windowing.py` | Cuts histories into fixed-length windows, and splits engines into a training and a validation set. |
| `pipeline.py` | Runs all of the above in order and returns arrays plus a diagnostics record. |
| `__init__.py` | Package marker; imports nothing. |

## How the labels are built

The training file needs no label file. Each engine is recorded until it fails,
so the remaining life at any cycle is just how many cycles are left:

```
RUL(t) = last_cycle_of_this_engine - t
```

The test file does need the label file, because recording stopped early. If an
engine's last recorded cycle is 150 and the label file says 47 cycles of life
remained at that moment, then at cycle 130 the engine still had
`(150 - 130) + 47 = 67` cycles left. In general:

```
RUL(t) = (last_recorded_cycle - t) + RUL_at_last_recorded_cycle
```

Both cases are handled by `attach_rul` in `loading.py`. Engine numbers in the
label file are assigned by position: the first line belongs to engine 1, the
second to engine 2, and so on. That is how the dataset is defined.

### The label is then capped

Raw "cycles until failure" makes a bad target. A brand-new engine in a long
history would carry a label of 300 or more, and a new engine gives the model no
way to tell 300 from 280 — the sensors of a healthy engine look the same either
way, because almost no wear has happened. Asking a model to tell them apart
forces it to guess, and the guessing shows up as noise in every prediction.

The standard fix, from Heimes (2008), is a **piecewise-linear** target: hold the
label at a ceiling while the engine is healthy, and let it fall only once real
degradation starts.

```
RUL(t) = min(cycles_until_failure(t), RUL_max)
```

Plotted against time this is a flat line followed by a straight decline. The
corner between them is often called the *change point*. Heimes reported that
any ceiling between roughly 120 and 130 cycles works well on this dataset, and
125 has become the common choice.

## Sensors: only 14 of the 21, and the order matters

Some of the 21 sensors never change over an engine's whole life, or move
between only a few fixed levels. A flat signal says nothing about degradation.
Dropping such signals is standard practice on this dataset.

Wen et al. keep 14 sensors. Their Table 3 prints the adjacency matrix of the
sensor graph over exactly those 14 sensors, in a specific order.

**That order is load-bearing.** The adjacency matrix is indexed by position, so
reordering the sensor columns would silently wire the graph to the wrong
sensors — no error, just wrong results. The order is therefore frozen in
`PAPER_SENSOR_SYMBOLS` in `sensors.py` and checked by assertions that run when
the module is imported. Do not reorder that tuple.

`WindowedDataset` in `windowing.py` carries the sensor column names alongside
the array for the same reason: it stops the order from being lost between the
data and the graph.

### Two things we found out about the paper's selection

Both came from running `detect_uninformative_sensors` on the real FD001
training data. Both are recorded in `docs/ASSUMPTIONS.md`.

**The unnamed seventh dropped sensor is NRc.** Section 3.2.2 of the paper says
"seven sensors are removed" and then names only six: T2, P2, epr, farB, Nfdmd
and PCNRdmd. Measured on the data, exactly those six are perfectly constant.
The seventh sensor missing from Table 3 is NRc, the corrected core speed — and
it is *not* constant: it takes 6078 distinct values across 20 631 training
rows. So it must have been dropped for some other reason, most plausibly
redundancy, since corrected core speed is derived from the physical core speed
Nc, which the paper keeps.

**P15 is kept but carries almost no information.** The bypass-duct pressure P15
is one of the 14 sensors in the paper's graph, yet on FD001 it takes only *two*
distinct values in the entire training set (21.60 and 21.61). It cannot
represent gradual degradation. We keep it anyway, because the adjacency matrix
is defined over the paper's fixed 14-sensor set and removing a node would change
the graph. The observation is reported rather than quietly corrected.

### `detect_uninformative_sensors` is a diagnostic, not a filter

The pipeline always uses the paper's fixed 14-sensor list. The function exists
so we can check, from the data itself, whether the paper's choice agrees with
what the measurements show. A disagreement is a finding worth reporting.

It applies two criteria to each of the 21 sensor columns:

1. **Constant** — the difference between the largest and smallest value is
   essentially zero. Because sensors have wildly different magnitudes, the test
   is relative: a range of 0.01 is negligible for a temperature near 1400 degR
   but large for a ratio near 0.03.
2. **Few distinct values** — the sensor only ever takes a handful of discrete
   levels, usually because the quantity is rounded coarsely.

Counting *distinct values* is the right second criterion, and range is not. On
the real FD001 training data the fan speed `Nf` has a relative range of 2.8e-4
across 53 distinct levels and is genuinely informative, while the bypass-duct
pressure `P15` has a slightly larger relative range of 4.6e-4 spread over only
2 levels and is not. A range test would keep the wrong one of the two.

## Smoothing

Sensor readings carry measurement noise: small, fast fluctuations unrelated to
wear. Wear itself is slow. Removing the fast wobble makes the slow trend easier
to see.

The filter is an **exponentially weighted moving average**, which replaces each
reading with a blend of that reading and the previous smoothed value:

```
S(t) = alpha * X(t) + (1 - alpha) * S(t-1)
```

`X(t)` is the raw reading at cycle `t`, `S(t)` is the smoothed value, and
`alpha` is between 0 and 1. An `alpha` near 1 barely smooths at all; an `alpha`
near 0 smooths hard but reacts late to a real change. The name comes from
expanding the formula: every past reading contributes, with a weight that
shrinks exponentially the further back it lies.

**The filter only looks backwards.** `S(t)` depends on `X(t)` and earlier
values, never on later ones. This matters for the honesty of the experiment: a
filter that averaged over future readings would leak the future into the
present, and the model would appear to predict failure better than it really
can. The filter is also applied to each engine separately, so no reading from
one engine can influence another.

One implementation detail that is not cosmetic: pandas `ewm` has two modes.
With `adjust=True` (its default) it applies a correction to the first few
values; with `adjust=False` it uses the plain recursive formula above. This
folder uses `adjust=False`, because that is the formula printed as Equation 18
in the paper. The two modes give visibly different values at the start of a
series.

## Rescaling, and what an operating condition is

The 14 sensors measure completely different physical quantities on completely
different scales — a turbine outlet temperature around 1400 degrees Rankine, a
bypass ratio around 8.4, a coolant bleed flow around 39 pounds per second. Fed
to a neural network unchanged, the large numbers dominate simply because they
are large. Rescaling puts every sensor on comparable footing.

`SensorNormalizer` offers three methods:

- `minmax_global` — map each sensor's smallest observed value to 0 and its
  largest to 1: `scaled = (value - minimum) / (maximum - minimum)`.
- `minmax_per_condition` — the same formula, but with a separate minimum and
  maximum per operating condition.
- `zscore` — shift and stretch each sensor to mean 0 and standard deviation 1:
  `scaled = (value - mean) / standard_deviation`. Unlike min-max this does not
  bound the values, which makes it more tolerant of outliers.

An **operating condition** is a flight regime. The three "operational setting"
columns describe how the aircraft was being flown: altitude, speed as a Mach
number, and throttle position. Flying higher or faster changes every sensor
reading immediately and by a lot, and that change has nothing to do with wear.

In FD002 and FD004 the engines are flown under six different conditions, and
the jumps between conditions are far larger than the slow drift caused by
degradation, so scaling all six together buries the wear signal. Scaling each
condition separately removes the flight-related jumps and leaves the wear
visible. FD001, which this project starts with, has only one condition, so
per-condition and global scaling give identical results there. The general
version is implemented now so the code is already correct when FD002 and FD004
are added.

Conditions are found with **k-means clustering** (MacQueen, 1967), which groups
points into a chosen number of clusters by repeatedly assigning each point to
the nearest cluster centre and recomputing the centres. `OperatingConditionEncoder`
clusters the three setting columns and uses the cluster number as the condition
label. When it detects only one condition it skips k-means entirely and labels
every row 0. This follows Jin, Meng, Hu & Chen (2025), *Heterogeneous
dynamic-aware GNN for RUL prediction of aeroengine*, IEEE TIM 74, 3538014, who
use the same approach on the same dataset.

A sensor that is constant within a condition cannot be min-max scaled — the
formula would divide by zero. Such columns are mapped to the bottom of the
target range instead, and a warning is logged rather than passed over in
silence, because a constant sensor inside the retained set may mean the sensor
selection needs revisiting.

### The rule that protects the experiment: fit on training data only

Every statistic — minima, maxima, means, standard deviations, and the k-means
cluster centres themselves — is computed from the **training** engines only and
then applied unchanged to the validation and test engines.

This is not a style preference. If the maximum were computed over training and
test data together, information about the test engines would enter the training
process. The model would then be evaluated on data it had partly seen, and the
reported error would be smaller than the error the same model would produce on
genuinely new engines. This mistake is called **data leakage**, and it is one of
the most common causes of results that cannot be reproduced (Kapoor &
Narayanan, 2023, *Leakage and the reproducibility crisis in machine-learning-based
science*, Patterns 4(9), 100804).

## Windowing

A model cannot take a whole engine history at once, because histories have
different lengths. Instead each history is cut into short, fixed-length pieces
called **time windows**. One window is one training example.

With a window size of 30, the first example covers cycles 1–30, the second
covers 2–31, and so on. The gap between the start of one window and the start of
the next is the **stride**; a stride of 1 gives the largest possible number of
examples.

The label of a window is the RUL at its **last** cycle. That matches how the
model would really be used: an operator has the last 30 cycles of readings and
wants to know the remaining life *now*, not 30 cycles ago.

### The shape of the output, and why the third axis is 1

The output is arranged for a graph neural network in which every sensor is a
**node** of a graph. The array has three axes:

```
(number_of_examples, number_of_sensors, features_per_sensor)
```

The third axis always has length 1: each sensor node carries the most recent
reading of its window. That is the input shape `[batch, 14, 1]` printed in
Table 4 of the paper, whose first graph layer has `W(1)` in `R^(1 x 32)`.

The window still decides *which* cycles are eligible — the first example of an
engine ends at cycle `window_size`, not at cycle 1 — and it is what a temporal
encoder would consume if one were added. It is simply not handed to the graph
layer directly.

### Engines shorter than the window are padded, not dropped

If an engine has fewer rows than the window size, its first reading is repeated
backwards until the window is full. Dropping the engine instead would silently
change the size of the test set and make the results incomparable with
published ones.

### The test set gets exactly one window per engine

Test histories stop before failure, and the official label file gives one number
per engine: the remaining life at the last recorded cycle. Evaluation therefore
uses only the final window of each test engine. On FD001 that is 100
predictions, which matches the "number of testing samples" column in Table 5 of
the paper.

### The validation split must be made by engine, not by window

This is the single most important trap in this folder.

A validation set is a slice of the training data that is held back and never
used to update the model. It exists so that decisions — when to stop training,
which settings to prefer — are not made by looking at the test set.

Consecutive windows from the same engine overlap heavily. With a window size of
30 and a stride of 1, two neighbouring windows share 29 of their 30 cycles and
their labels differ by one cycle. If windows were split at random, nearly
identical examples would land on both sides of the split, and the validation
error would come out far too optimistic — the model would effectively be tested
on data it had already trained on. Splitting whole engines keeps the two sets
genuinely separate. That is what `split_units_by_id` does, and the pipeline
asserts afterwards that no engine appears in both splits.

The paper describes no validation split at all; it reports training for a fixed
200 epochs. Introducing one here is a deliberate improvement over the source.

## The pipeline, step by step

`run_preprocessing()` in `pipeline.py` runs the whole sequence. Having one
function do it, rather than a notebook calling the steps by hand, means the
experiment can be repeated exactly, from a script or a test.

1. **Load** the three raw text files and attach a RUL value to every row.
2. **Diagnose** which sensors carry no information and compare that with the
   paper's selection. This step changes nothing; it only reports.
3. **Select** the paper's 14 sensors, in the exact order of its Table 3, so the
   sensor axis lines up with the sensor graph.
4. **Cap** the labels at the ceiling, producing the piecewise-linear target.
5. **Smooth** the sensor signals with the exponentially weighted moving average.
6. **Split** the training engines into a training part and a validation part.
7. **Identify operating conditions** and **rescale** the sensors, learning every
   statistic from the training engines only.
8. **Cut** the histories into fixed-length windows.
9. **Apply** the split decided in step 6 to those windows.

Steps 3 and 4 are grouped together in the code, which puts the label cap before
the smoothing while the paper's Figure 1 puts it after. The two commute exactly:
capping touches only the label column and smoothing touches only the sensor
columns, so no value is read by both.

**Why smoothing comes before rescaling.** Smoothing removes measurement noise
and is most naturally expressed in the sensor's own physical units, where the
noise actually occurs. Rescaling is the last preparation before the model and
should therefore see the signal the model will use. This order is *taken from
the paper*, not chosen here: Figure 1 shows the chain as feature selection →
data smoothing → piece-wise RUL → data normalisation → train and test datasets.

**Why the split comes before the scaling.** The validation engines are chosen in
step 6, before any statistic is computed in step 7. Fitting the scaler on all
training engines and only afterwards holding some of them out would let a
held-out engine define a minimum or a maximum — the same kind of leak that
keeping the test set separate is meant to prevent. The effect is smaller than
test leakage, but it biases exactly the quantity a validation set exists to
measure.

**Where the training and test paths differ.** Both go through the same code.
There are exactly two deliberate differences:

- All statistics — cluster centres, minima, maxima, means, standard deviations
  — are learned from the training engines and applied to the validation and test
  engines unchanged.
- Training engines produce every window; test engines produce only their final
  window, because that is the single moment the official labels describe.

### Sanity checks and outputs

`load_subset` compares the number of rows and engines against the counts
published for the official release (`EXPECTED_SHAPES` in `loading.py`) and logs
a **warning**, never an error, on a mismatch. A mismatch means the files are
truncated, modified, or not the official release — investigate it before
reporting any result computed from them. A legitimate variant of the files
should not stop the pipeline, which is why it is only a warning.

The NASA files are not in this repository, because they are redistributed under
NASA's own terms. When a file is missing, the error message quotes the download
URL, so the fix travels with the problem instead of living only in the README.

`PreparedData.save()` writes five files, so a later stage never has to re-run
preprocessing and the exact inputs of any experiment can be recovered:

- `windows.npz` — the three splits as arrays.
- `adjacency.npz` — the raw and normalised sensor graphs, plus the node labels.
- `normalization_statistics.csv` — the learned scaling statistics, one row per
  sensor and condition.
- `config.yaml` — the settings used.
- `diagnostics.json` — what the pipeline observed while running.

Which sensor graph is used, and how it is normalised, are configuration rather
than constants, so the matrix saved beside a result is the one the model was
actually given. Before that was so, a run using row normalisation still stored
the symmetric matrix, and anything reading it afterwards saw a graph the model
had never used.

## The judgement calls made in this folder

Each of these is a place where the paper does not say what it did, so we had to
choose. Every one has a numbered entry in `docs/ASSUMPTIONS.md` with the full
reasoning.

- **The label ceiling is 125 cycles.** The paper gives the piecewise-linear
  formula as its Equation 19 but never prints the ceiling it used. Section 4.2
  describes predicted life holding "around 125" while an engine is healthy, and
  Jin et al. (2025) use 125 on the same dataset. Recorded as **A-01**.
- **The ceiling is applied to the test labels too.** The paper does not say
  whether the official test labels are capped before the error is computed. We
  cap them, because the model's sigmoid output cannot exceed the ceiling in the
  first place. This is controlled by `clip_test_labels`. It can only move the
  reported error favourably, so it must be stated whenever a number from this
  project is compared with a published one. Recorded as **A-02**.
- **How the sensors are rescaled.** The paper never describes how it scales its
  inputs and never mentions operating conditions in its preprocessing. The whole
  of `SensorNormalizer` is a substitution taken from Jin et al. (2025), not a
  reconstruction of what Wen et al. did. Recorded as **A-03**.
- **The smoothing parameter is 0.5.** Section 3.2.3 of the paper contradicts
  itself here. It states that "S(t-1) … is equal to 0.5 in this study", which
  attaches 0.5 to the previous smoothed value rather than to `alpha`. A fixed
  previous value makes no sense in a recursive formula, so we read the sentence
  as `alpha = 0.5`. Recorded as **A-04**.
- **Windows are taken one cycle apart.** The paper gives a window size of 30 in
  its Table 6 but never states a stride. We use 1. Recorded as **A-05**.
- **A validation set is held out at all.** The paper describes no validation
  split and trains for a fixed 200 epochs; holding engines out is our addition.
  Recorded as **A-06**.
- **Each sensor node carries the last cycle of its window.** Table 4 gives the
  input to the first graph layer as `[batch, 14, 1]`, one number per sensor. In
  the paper that number is the output of a per-sensor LSTM; with that stage
  omitted, it is the most recent reading of the window. Recorded as **A-07**,
  and the omission of the LSTM stage itself as **A-11**.
- **The order of the preprocessing steps.** This one turned out *not* to be a
  choice: Figure 1 of the paper prints the chain explicitly, so we follow it.
  The entry is kept under its original number, **A-08**, and now records why it
  is not an assumption — plus a measured caution about what that order does on
  the multi-condition subsets FD002 and FD004. Read it before running this
  pipeline on those two.
- **The recursive form of the smoothing filter.** pandas offers two conventions
  that differ in how they weight the first few values of a series;
  `adjust=False` is the plain recursive form, which is exactly Equation 18 of
  the paper. Recorded as **A-09**.
- **The seventh dropped sensor is NRc.** Section 3.2.2 says seven sensors are
  removed and then names only six. NRc is identified as the seventh: it is the
  one sensor of Table 2 absent from Table 3 without being named in that
  sentence. Recorded as **A-10**.

## What each name means

Everything the folder offers, file by file. Anything not listed is internal
and starts with an underscore.

### `loading.py`

| Name | What it is |
|---|---|
| `load_subset(subset, data_dir=...)` | Reads all three files of one subset, attaches the labels, and returns a `CMAPSSData`. This is the only one of these you normally call. |
| `read_cmapss_frame(path)` | Reads one history file into a table with the 26 named columns, sorted by engine and then by cycle. Raises if the file is missing (the message quotes the download link) or does not have 26 columns. |
| `read_rul_file(path)` | Reads a `RUL_FD00X.txt` file. Returns one number per engine, numbered from 1 by line position. |
| `attach_rul(frame, rul_at_last_cycle=None)` | Adds the `rul` column. Leave the second argument out for training data; pass what `read_rul_file` returned for test data. |
| `CMAPSSData` | Holds one loaded subset: `subset` (which one), `train` and `test` (the tables, each with a `rul` column), and `test_rul` (the official label at each test engine's last cycle). Its `summary()` returns a small table to eyeball. |
| `EXPECTED_SHAPES` | How many rows and engines each subset should have. Used for the warning described above. |
| `DOWNLOAD_URL` | Where the NASA files come from. Quoted in the missing-file error. |

### `sensors.py`

| Name | What it is |
|---|---|
| `SENSORS` | All 21 sensors, each a `SensorInfo`. |
| `SensorInfo` | One sensor: its `index` (1 to 21), its `symbol` (the short engineering name such as `T24`), a plain `description`, its `unit`, and the `station` — the part of the engine it sits in, used only in figures. Its `column` property gives the column name, `sensor_2`. |
| `PAPER_SENSOR_SYMBOLS` | The 14 sensors the paper keeps, in the order of its Table 3. **Do not reorder.** |
| `PAPER_SENSOR_COLUMNS` | The same 14, written as column names. |
| `PAPER_EXCLUDED_SYMBOLS` | The 7 the paper drops. |
| `symbol_to_column(symbol)` | `"T24"` to `"sensor_2"`. |
| `column_to_symbol(column)` | `"sensor_2"` to `"T24"`. |
| `detect_uninformative_sensors(frame)` | The diagnostic described above. Returns a dictionary with three keys — `"constant"`, `"low_resolution"` and `"informative"` — each a sorted list of sensor names. |
| `UNIT_COL`, `CYCLE_COL` | The names of the engine and cycle columns: `"unit"` and `"cycle"`. |
| `OP_SETTING_COLS`, `SENSOR_COLS`, `RAW_COLUMNS` | The three setting columns, the 21 sensor columns, and all 26 in file order. |

The units in `SensorInfo` are the ones American aerospace engineering uses:
`degR` is degrees Rankine, an absolute temperature scale; `psia` is pounds per
square inch, absolute; `rpm` is revolutions per minute; `lbm/s` is pounds of
mass per second; `-` means the quantity is a plain ratio with no unit.

### `labeling.py`

| Name | What it is |
|---|---|
| `apply_piecewise_rul(frame, rul_max=...)` | Caps the `rul` column at `rul_max`. Pass `output_column` to write to a new column instead of replacing the old one, which is useful for drawing the effect of the cap. |
| `piecewise_rul_curve(total_cycles, rul_max)` | The label curve one engine should have, for drawing or for testing. `piecewise_rul_curve(6, rul_max=3)` gives `[3, 3, 3, 2, 1, 0]`. |

### `smoothing.py`

| Name | What it is |
|---|---|
| `apply_ewma(frame, columns, alpha=...)` | Smooths the chosen columns, one engine at a time. `alpha=None` turns it off and returns an unchanged copy, which is how the filter's contribution is measured. Only those columns change; the label is left alone. |
| `ewma_series(values, alpha)` | The same filter on a single series, written out longhand. The tests use it to confirm the fast pandas version agrees. |

`alpha` must be above 0 and at most 1. Anything else raises.

### `normalization.py`

| Name | What it is |
|---|---|
| `OperatingConditionEncoder` | Works out how the aircraft was being flown at each row. `fit(frame)` learns from the **training rows only**; `transform(frame)` adds the condition column to any table; `fit_transform` does both. `n_detected` says how many conditions it settled on. Leave `n_conditions` as `None` to detect the number from the data. |
| `SensorNormalizer` | Rescales the sensors. `method` is one of the three above, `columns` is which columns to touch, and `feature_range` is the target range for the two min-max methods. `fit(frame)` measures the **training rows only**; `transform(frame)` applies what it measured to any table; `fit_transform` does both. After fitting, `statistics_` holds what it measured, one row per sensor and condition, so it can be inspected and saved. |
| `CONDITION_COL` | The name of the column the encoder adds: `"operating_condition"`. |

### `windowing.py`

| Name | What it is |
|---|---|
| `make_windows(frame, sensor_columns, ...)` | Cuts every engine's history into overlapping windows. Use for training data. |
| `make_last_windows(frame, sensor_columns, ...)` | Takes only the last window of each engine. Use for test data. |
| `split_units_by_id(units, validation_fraction=..., seed=...)` | Splits the engine numbers into the ones to train on and the ones to hold back. The two never share an engine. A fraction of 0 holds nothing back. |
| `WindowedDataset` | The result of the two window functions. Its fields are listed below. |

`WindowedDataset` holds five things, all lined up so that row *i* of each
describes the same example:

| Field | What it holds |
|---|---|
| `features` | The readings, shaped `(examples, sensors, numbers per sensor)`. |
| `targets` | The remaining life at the last cycle of each window, in flight cycles. |
| `units` | Which engine each example came from. Needed to group predictions by engine, and to check that no engine landed in two splits. |
| `end_cycles` | The cycle number of the last row of each window. |
| `sensor_columns` | The sensor names, in the order of the second axis. Kept with the array so the sensor order cannot be lost, which would quietly misalign the data with the sensor graph. |

`len(dataset)` gives the number of examples, and
`dataset.subset_by_units([...])` returns a copy holding only the engines listed.

### `pipeline.py`

| Name | What it is |
|---|---|
| `run_preprocessing(config=None, data_dir=None, keep_frames=False)` | Runs all nine steps and returns a `PreparedData`. `keep_frames=True` also keeps the intermediate tables, which costs memory but is what a notebook wants. |
| `PreparedData` | Everything the training stage needs, plus everything needed to check it. Its fields are listed below. |

| Field | What it holds |
|---|---|
| `config` | The exact settings that produced this object. |
| `train`, `validation`, `test` | The three sets of examples. `validation` is empty when nothing was held back, which is the default. |
| `adjacency_raw`, `adjacency_normalized` | The sensor graph before and after rescaling. |
| `sensor_symbols` | The sensor names, in the order of the graph's rows. |
| `normalizer`, `condition_encoder` | The two fitted objects, kept so what they measured can be inspected and saved. |
| `diagnostics` | A free-form record of what the run observed: constant sensors, the shape of the graph, the size of each split, and whether the paper's choice of sensors agrees with the data. |
| `raw`, `frames` | The intermediate tables, or `None` unless `keep_frames=True`. |

`summary()` returns a small table describing the three sets. `save(directory)`
writes the five files listed above.
