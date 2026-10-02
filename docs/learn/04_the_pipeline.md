# Preprocessing: from labelled rows to model inputs

The [Data page](01_the_problem.md) covers reading histories,
checking variation, and selecting sensor columns. We now change the labels and
readings, then arrange the readings into fixed-length histories called **windows**.

{{figure:pipeline}}

The order protects against **leakage**: information from test engines, held-back
engines or future cycles entering a calculation that should not use it.
Smoothing uses only a row's present and past; the engine split is chosen before
learning any rescaling statistics; test engines never supply those statistics.

The examples follow the first three rows of training engine 1 through steps
1–4. Every column is shown. Step 3 also shows engine 3 so the split is visible.
For step 5 we use five real rows and a three-cycle window to expose every
value; the normal run still uses 30-cycle windows. Decimal table values are
rounded to six places for reading; computations keep their precision.

## Example setup

The tables and explanations stand on their own. To reproduce the examples in
Python, open the setup below, then run the example blocks in page order.

<details class="process-example">
<summary>Set up the Python examples</summary>

Run from the repository root after installing the project. A pandas
`DataFrame` represents a table with named columns.

```python
import pandas as pd
from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.loading import load_subset
from thesis_gnn_rul_xai.data.labeling import apply_piecewise_rul
from thesis_gnn_rul_xai.data.smoothing import apply_ewma
from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_COLUMNS
from thesis_gnn_rul_xai.data.normalization import (
    OperatingConditionEncoder, SensorNormalizer,
)
from thesis_gnn_rul_xai.data.windowing import (
    make_windows, make_last_windows, split_units_by_id,
)
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing

data = load_subset("FD001", data_dir="data/raw")
columns = list(PAPER_SENSOR_COLUMNS)
```

</details>

<div class="algorithm-intro">
<p class="algorithm-label">Preprocessing algorithm</p>
<p>Start with the labelled tables from the Data page. Follow the steps below
in order to prepare sensor windows and their answers for training and testing.</p>
</div>

## 1 — Cap the labels

A very early reading need not distinguish reliably between different long
remaining lives. Capping uses the ceiling as **one shared label for all larger
remaining lives**; below it, the label counts down exactly. It changes the
answer we ask the model to predict; it does not change when the engine failed.

{{figure:rul-curve}}

The dashed line counts the recorded cycles remaining for training engine 1;
the solid line applies the project's ceiling of 125 cycles (assumption A-01).
They meet at cycle 67, where `192 − 67 = 125`; at cycle 68 both are 124.
This is a training-label curve, not a prediction.

### Process

#### Input

The labelled training table from the Data page. Its first three rows:

<!-- sample: cap_input -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | 641.82 | 1589.7 | 1400.6 | 14.62 | 21.61 | 554.36 | 2388.06 | 9046.19 | 1.3 | 47.47 | 521.66 | 2388.02 | 8138.62 | 8.4195 | 0.03 | 392 | 2388 | 100 | 39.06 | 23.419 | 191 |
| 1 | 2 | 0.0019 | -0.0003 | 100 | 518.67 | 642.15 | 1591.82 | 1403.14 | 14.62 | 21.61 | 553.75 | 2388.04 | 9044.07 | 1.3 | 47.49 | 522.28 | 2388.07 | 8131.49 | 8.4318 | 0.03 | 392 | 2388 | 100 | 39 | 23.4236 | 190 |
| 1 | 3 | -0.0043 | 0.0003 | 100 | 518.67 | 642.35 | 1587.99 | 1404.2 | 14.62 | 21.61 | 554.26 | 2388.08 | 9052.94 | 1.3 | 47.27 | 522.42 | 2388.03 | 8133.23 | 8.4178 | 0.03 | 390 | 2388 | 100 | 38.95 | 23.3442 | 189 |

#### Function

Compare each remaining-life answer with the chosen ceiling.
Keep the smaller of the two:

> Capped answer = the smaller of the original answer and the ceiling.

This rule applies to every row: answers above the ceiling are lowered to it;
answers at or below it stay unchanged. It groups long remaining lives under
one target while preserving the final countdown. With a ceiling of 125, the
three input answers become:

| Original answer | Comparison | Capped answer |
|---|---|---|
| 191 | 191 is above 125 | 125 |
| 190 | 190 is above 125 | 125 |
| 189 | 189 is above 125 | 125 |

For comparison, engine 1 at cycle 68 has 124 cycles left. That answer is below
the ceiling, so it remains 124. Sensor readings are never part of this comparison.

<details class="process-example">
<summary>Example in Python</summary>

Start with the [example setup](#example-setup):

```python
cap_input = data.train.head(3)
capped = apply_piecewise_rul(data.train, rul_max=125)
capped_test = apply_piecewise_rul(data.test, rul_max=125)
cap_output = capped.head(3)
assert capped.query("unit == 1 and cycle == 68").rul.item() == 124
```

</details>

#### Output

The same rows and sensor readings, with the capped answers in
the final `rul` column. The complete table still has 20,631 rows and 27 columns:

<!-- sample: cap_output -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | 641.82 | 1589.7 | 1400.6 | 14.62 | 21.61 | 554.36 | 2388.06 | 9046.19 | 1.3 | 47.47 | 521.66 | 2388.02 | 8138.62 | 8.4195 | 0.03 | 392 | 2388 | 100 | 39.06 | 23.419 | 125 |
| 1 | 2 | 0.0019 | -0.0003 | 100 | 518.67 | 642.15 | 1591.82 | 1403.14 | 14.62 | 21.61 | 553.75 | 2388.04 | 9044.07 | 1.3 | 47.49 | 522.28 | 2388.07 | 8131.49 | 8.4318 | 0.03 | 392 | 2388 | 100 | 39 | 23.4236 | 125 |
| 1 | 3 | -0.0043 | 0.0003 | 100 | 518.67 | 642.35 | 1587.99 | 1404.2 | 14.62 | 21.61 | 554.26 | 2388.08 | 9052.94 | 1.3 | 47.27 | 522.42 | 2388.03 | 8133.23 | 8.4178 | 0.03 | 390 | 2388 | 100 | 38.95 | 23.3442 | 125 |

The default also caps test answers (a separate choice, assumption A-02).
Turning **Cap test labels** off keeps those answers uncapped. Always name this
choice when reporting scores: the two choices evaluate different targets.
It is not a guarantee that every reported measure improves.

**Do not confuse label capping with sensor rescaling.** This stage changes the
answer; step 4 changes the readings from which an answer is predicted.

## 2 — Smooth each sensor

Successive readings can jump up and down. Smoothing gives a new reading only
part of the influence on the next value, reducing the effect of one sudden jump.
It is applied separately to each selected sensor and each engine.

{{figure:smoothing}}

### Process

#### Input

The capped table from step 1, with sensors still in their original
units. These are its first three rows:

<!-- sample: smooth_input -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | 641.82 | 1589.7 | 1400.6 | 14.62 | 21.61 | 554.36 | 2388.06 | 9046.19 | 1.3 | 47.47 | 521.66 | 2388.02 | 8138.62 | 8.4195 | 0.03 | 392 | 2388 | 100 | 39.06 | 23.419 | 125 |
| 1 | 2 | 0.0019 | -0.0003 | 100 | 518.67 | 642.15 | 1591.82 | 1403.14 | 14.62 | 21.61 | 553.75 | 2388.04 | 9044.07 | 1.3 | 47.49 | 522.28 | 2388.07 | 8131.49 | 8.4318 | 0.03 | 392 | 2388 | 100 | 39 | 23.4236 | 125 |
| 1 | 3 | -0.0043 | 0.0003 | 100 | 518.67 | 642.35 | 1587.99 | 1404.2 | 14.62 | 21.61 | 554.26 | 2388.08 | 9052.94 | 1.3 | 47.27 | 522.42 | 2388.03 | 8133.23 | 8.4178 | 0.03 | 390 | 2388 | 100 | 38.95 | 23.3442 | 125 |

#### Function

Blend the newest reading with the smoothed history so far.
Choose the share given to the newest reading; the remaining share belongs to
the previous smoothed value. These shares add to one, so the result lies
between the two values rather than adding their full sizes together.

> New smoothed value = (new-reading share × current reading)
> + (remaining share × previous smoothed value).

Start each sensor's history with its first recorded value. Here each share is
one half. This is an **exponentially weighted moving average**: older readings
keep contributing through the previous result, but their influence shrinks
with each new blend. The half-and-half weight and starting rule are project
choices (assumptions A-04 and A-09).

Follow `T24`, the `sensor_2` column, through all three rows:

```text
cycle 1: first reading                               = 641.82
cycle 2: 0.5 × 642.15 + 0.5 × 641.82                  = 641.985
cycle 3: 0.5 × 642.35 + 0.5 × 641.985                 = 642.1675
```

Cycle 3 uses 641.985, not the raw cycle-2 reading 642.15. The other selected
sensors follow the same calculation with their own values. Engine 2 starts
again from its own first row. No row reads a later cycle or another engine;
otherwise the current input could reveal information unavailable at that time.

<details class="process-example">
<summary>Example in Python</summary>

Continue after step 1:

```python
smooth_input = capped.head(3)
smoothed = apply_ewma(capped, columns, alpha=0.5)
smoothed_test = apply_ewma(capped_test, columns, alpha=0.5)
smooth_output = smoothed.head(3)
```

</details>

#### Output

The same 27 columns and row count. Only the selected 14 sensor
columns change; the first row of each engine stays as it was:

<!-- sample: smooth_output -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | 641.82 | 1589.7 | 1400.6 | 14.62 | 21.61 | 554.36 | 2388.06 | 9046.19 | 1.3 | 47.47 | 521.66 | 2388.02 | 8138.62 | 8.4195 | 0.03 | 392 | 2388 | 100 | 39.06 | 23.419 | 125 |
| 1 | 2 | 0.0019 | -0.0003 | 100 | 518.67 | 641.985 | 1590.76 | 1401.87 | 14.62 | 21.61 | 554.055 | 2388.05 | 9045.13 | 1.3 | 47.48 | 521.97 | 2388.045 | 8131.49 | 8.42565 | 0.03 | 392 | 2388 | 100 | 39.03 | 23.4213 | 125 |
| 1 | 3 | -0.0043 | 0.0003 | 100 | 518.67 | 642.1675 | 1589.375 | 1403.035 | 14.62 | 21.61 | 554.1575 | 2388.065 | 9049.035 | 1.3 | 47.375 | 522.195 | 2388.0375 | 8133.23 | 8.421725 | 0.03 | 391 | 2388 | 100 | 38.99 | 23.38275 | 125 |

Turning smoothing off skips this change. A weight of 1 also keeps the raw
readings. The filter runs before condition-based rescaling, so if an engine's
operating condition changes it can blend readings across that change.

## 3 — Hold engines back

We need to check a model on engines whose examples do not adjust its learned
numbers. Those held-back engines form the **validation set**. Their errors can
choose when to stop training and which saved version of the model to keep.

We split by engine, not by individual window. Neighbouring 30-cycle windows
share 29 cycles; putting them on opposite sides would test the model on almost
the same history it learned from. We also split **before** measuring rescaling
statistics so held-back readings cannot influence those statistics.

**Where this happens.** The playground's **Training → Procedure** control
chooses whether to reserve validation engines and how their errors will be
used during training. The reservation itself happens here, during
preprocessing. From this point onward, each engine belongs to training,
validation or the separate test file. Rescaling and windowing preserve that
membership; cutting a history into windows does not choose a new group.

### Process

#### Input

The unique engine identifiers in the smoothed training table, a
fraction to reserve (`0.2`), and a seed (`42`). A seed fixes the random draw so
repeating it gives the same split. For a visible example, here is the first
row of engines 1, 2 and 3, including every column:

<!-- sample: split_input -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | 641.82 | 1589.7 | 1400.6 | 14.62 | 21.61 | 554.36 | 2388.06 | 9046.19 | 1.3 | 47.47 | 521.66 | 2388.02 | 8138.62 | 8.4195 | 0.03 | 392 | 2388 | 100 | 39.06 | 23.419 | 125 |
| 2 | 1 | -0.0018 | 0.0006 | 100 | 518.67 | 641.89 | 1583.84 | 1391.28 | 14.62 | 21.6 | 554.53 | 2388.01 | 9054.72 | 1.3 | 46.93 | 522.33 | 2388.06 | 8137.72 | 8.3905 | 0.03 | 391 | 2388 | 100 | 38.94 | 23.4585 | 125 |
| 3 | 1 | 0.0008 | 0.0005 | 100 | 518.67 | 642.04 | 1584.2 | 1398.13 | 14.62 | 21.61 | 553.96 | 2388.06 | 9056.95 | 1.3 | 47.14 | 522.24 | 2388 | 8138.4 | 8.4207 | 0.03 | 391 | 2388 | 100 | 38.96 | 23.3205 | 125 |

#### Function

Imagine drawing engine identifiers from a shuffled list.
Reserve the chosen fraction of that list, rounded to a whole number of
engines, and keep the rest for learning. The unit being divided is an
**entire engine history**, so neighbouring readings can never end up on
opposite sides.

Here one fifth of 100 engines is 20. The repeatable draw numbered 42 keeps
engines 1 and 2 and reserves engine 3. Every row of those engines follows the
same decision; the first-row samples below make that division visible.
The split and the restriction on fitting statistics are recorded choices
(assumptions A-06 and A-25).

<details class="process-example">
<summary>Example in Python</summary>

Continue after smoothing:

```python
train_ids, held_ids = split_units_by_id(
    smoothed.unit.unique(), validation_fraction=0.2, seed=42,
)
split_input = smoothed.query("unit in [1, 2, 3] and cycle == 1")
held_ids_output = pd.DataFrame({"held-back engine": held_ids})
split_train_output = split_input[split_input.unit.isin(train_ids)]
split_validation_output = split_input[split_input.unit.isin(held_ids)]
fitting_rows = smoothed[smoothed.unit.isin(train_ids)]
assert len(fitting_rows) == 16779
assert smoothed.unit.isin(held_ids).sum() == 3852
```

</details>

#### Output

Two disjoint groups: 80 engines available for learning and 20
reserved for progress checks. No engine appears in both groups. The complete
reserved list is:

<!-- sample: held_ids_output -->
| held-back engine |
| --- |
| 3 |
| 5 |
| 19 |
| 22 |
| 25 |
| 26 |
| 28 |
| 34 |
| 43 |
| 51 |
| 52 |
| 53 |
| 57 |
| 60 |
| 62 |
| 72 |
| 82 |
| 93 |
| 98 |
| 100 |

Filtering the sample by those lists gives these rows for fitting:

<!-- sample: split_train_output -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | 641.82 | 1589.7 | 1400.6 | 14.62 | 21.61 | 554.36 | 2388.06 | 9046.19 | 1.3 | 47.47 | 521.66 | 2388.02 | 8138.62 | 8.4195 | 0.03 | 392 | 2388 | 100 | 39.06 | 23.419 | 125 |
| 2 | 1 | -0.0018 | 0.0006 | 100 | 518.67 | 641.89 | 1583.84 | 1391.28 | 14.62 | 21.6 | 554.53 | 2388.01 | 9054.72 | 1.3 | 46.93 | 522.33 | 2388.06 | 8137.72 | 8.3905 | 0.03 | 391 | 2388 | 100 | 38.94 | 23.4585 | 125 |

And this row held back from fitting:

<!-- sample: split_validation_output -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 1 | 0.0008 | 0.0005 | 100 | 518.67 | 642.04 | 1584.2 | 1398.13 | 14.62 | 21.61 | 553.96 | 2388.06 | 9056.95 | 1.3 | 47.14 | 522.24 | 2388 | 8138.4 | 8.4207 | 0.03 | 391 | 2388 | 100 | 38.96 | 23.3205 | 125 |

No sensor value or answer changed. Over the complete table, the lists select
16,779 training rows and 3,852 held-back rows. The full table remains available;
later stages use the lists to control which rows may teach their statistics.

**Do not confuse validation with test.** Validation influences the choice of
model. Test evaluates the finished choice on the separate test file. Here
validation includes every eligible window, while test includes only each
engine's last window; their errors cover different examples and are not
directly comparable. With fraction 0, all training engines are kept.

## 4 — Put readings on a comparable scale

Temperature readings and ratios have different units and numerical ranges.
Rescaling expresses each selected sensor relative to its training readings.
This keeps a large numerical unit from automatically producing a large input.

### Find operating conditions

Readings can change because the engine is being run differently, as well as
because it wears. A condition number groups similar operating settings so the
optional rescaling within each condition can use a separate scale for each group.

#### Process

##### Input

The three operating settings describing how the engines are run.
Only the 16,779 rows from engines kept in step 3 may define the groups. Once
those groups are established, every training, held-back and test row can be
assigned to one. The continuing engine-1 sample is:

<!-- sample: condition_input -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | 641.82 | 1589.7 | 1400.6 | 14.62 | 21.61 | 554.36 | 2388.06 | 9046.19 | 1.3 | 47.47 | 521.66 | 2388.02 | 8138.62 | 8.4195 | 0.03 | 392 | 2388 | 100 | 39.06 | 23.419 | 125 |
| 1 | 2 | 0.0019 | -0.0003 | 100 | 518.67 | 641.985 | 1590.76 | 1401.87 | 14.62 | 21.61 | 554.055 | 2388.05 | 9045.13 | 1.3 | 47.48 | 521.97 | 2388.045 | 8131.49 | 8.42565 | 0.03 | 392 | 2388 | 100 | 39.03 | 23.4213 | 125 |
| 1 | 3 | -0.0043 | 0.0003 | 100 | 518.67 | 642.1675 | 1589.375 | 1403.035 | 14.62 | 21.61 | 554.1575 | 2388.065 | 9049.035 | 1.3 | 47.375 | 522.195 | 2388.0375 | 8133.23 | 8.421725 | 0.03 | 391 | 2388 | 100 | 38.99 | 23.38275 | 125 |

##### Function

Look for repeated combinations of operating settings. Round
the settings to whole numbers when counting combinations so tiny variations
do not create a new condition for every row. The chosen procedure allows
between one and six groups. In the sample, all three combinations round to
the same values:

| Cycle | Original settings | Rounded combination |
|---|---|---|
| 1 | −0.0007, −0.0004, 100 | 0, 0, 100 |
| 2 | 0.0019, −0.0003, 100 | 0, 0, 100 |
| 3 | −0.0043, 0.0003, 100 | 0, 0, 100 |

All the kept FD001 rows give that same rounded combination, so one group
is enough. Label it 0 and assign each row to it. The original operating
settings stay unchanged; rounding was only used to count the groups.

Whole numbers are coarser than the settings of FD002 and FD004, the two
sub-datasets flown in six conditions. The three settings are the altitude in
thousands of feet, the Mach number (the speed as a fraction of the speed of
sound) and the throttle angle in degrees. Rounded, the Mach numbers 0, 0.25,
0.62, 0.70 and 0.84 become only 0 and 1, and with the throttle angle's two
values, 100 and 60, they make just three combinations. The count still comes
out at six because the altitudes round to six different values: 0, 10, 20, 25,
35 and 42. That holds in the training and the test file of both
sub-datasets, so no condition is merged. Two conditions at one altitude whose
Mach numbers round to the same whole number (0.62 and 0.84 both become 1)
would be counted as one; none of these files has such a pair. The
[C-MAPSS resource page](../resources/cmapss.md#which-flight-conditions-the-settings-are)
shows how the three settings were identified.

When there are several groups, their centres are found from the unrounded
training settings. Each row joins the nearest centre, then each centre moves
to the average of its members; repeating these steps forms clusters of similar
settings. This is called **k-means**. Held-back and test rows join the nearest
established centre without changing any centre.

<details class="process-example">
<summary>Example in Python</summary>

Continue after the engine split:

```python
condition_input = smoothed.head(3)
encoder = OperatingConditionEncoder(seed=42).fit(fitting_rows)
conditioned = encoder.transform(smoothed)
conditioned_test = encoder.transform(smoothed_test)
condition_output = conditioned.head(3)
assert encoder.n_detected == 1
```

</details>

##### Output

A new `operating_condition` column, making 28 columns. The new
integer is a group identifier, not a measure of health:

<!-- sample: condition_output -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul | operating_condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | 641.82 | 1589.7 | 1400.6 | 14.62 | 21.61 | 554.36 | 2388.06 | 9046.19 | 1.3 | 47.47 | 521.66 | 2388.02 | 8138.62 | 8.4195 | 0.03 | 392 | 2388 | 100 | 39.06 | 23.419 | 125 | 0 |
| 1 | 2 | 0.0019 | -0.0003 | 100 | 518.67 | 641.985 | 1590.76 | 1401.87 | 14.62 | 21.61 | 554.055 | 2388.05 | 9045.13 | 1.3 | 47.48 | 521.97 | 2388.045 | 8131.49 | 8.42565 | 0.03 | 392 | 2388 | 100 | 39.03 | 23.4213 | 125 | 0 |
| 1 | 3 | -0.0043 | 0.0003 | 100 | 518.67 | 642.1675 | 1589.375 | 1403.035 | 14.62 | 21.61 | 554.1575 | 2388.065 | 9049.035 | 1.3 | 47.375 | 522.195 | 2388.0375 | 8133.23 | 8.421725 | 0.03 | 391 | 2388 | 100 | 38.99 | 23.38275 | 125 | 0 |

### Rescale each sensor

The default subtracts each sensor's training mean and divides by its training
standard deviation, a measure of spread around that mean. This is called
**z-score rescaling**. A result of zero is at the training mean; a negative
result is below it. It does not mean negative physical temperature or pressure.
The scaling method is a project choice (assumption A-03).

#### Process

##### Input

The table from **Find operating conditions**. Its sample rows are:

<!-- sample: scale_input -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul | operating_condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | 641.82 | 1589.7 | 1400.6 | 14.62 | 21.61 | 554.36 | 2388.06 | 9046.19 | 1.3 | 47.47 | 521.66 | 2388.02 | 8138.62 | 8.4195 | 0.03 | 392 | 2388 | 100 | 39.06 | 23.419 | 125 | 0 |
| 1 | 2 | 0.0019 | -0.0003 | 100 | 518.67 | 641.985 | 1590.76 | 1401.87 | 14.62 | 21.61 | 554.055 | 2388.05 | 9045.13 | 1.3 | 47.48 | 521.97 | 2388.045 | 8131.49 | 8.42565 | 0.03 | 392 | 2388 | 100 | 39.03 | 23.4213 | 125 | 0 |
| 1 | 3 | -0.0043 | 0.0003 | 100 | 518.67 | 642.1675 | 1589.375 | 1403.035 | 14.62 | 21.61 | 554.1575 | 2388.065 | 9049.035 | 1.3 | 47.375 | 522.195 | 2388.0375 | 8133.23 | 8.421725 | 0.03 | 391 | 2388 | 100 | 38.99 | 23.38275 | 125 | 0 |

##### Function

First establish a reference for each sensor using only the
engines kept for learning. Find its mean reading and its spread around that
mean. To measure the spread, subtract the mean from each reading, square
those distances, average them, then take the square root. Squaring prevents
positive and negative distances from cancelling; the square root restores
the original unit. These are the resulting reference values for the example:

<!-- sample: scale_statistics -->
| column | mean | std | n_rows |
| --- | --- | --- | --- |
| sensor_2 | 642.675171 | 0.430308 | 16779 |
| sensor_3 | 1590.448957 | 5.118662 | 16779 |
| sensor_4 | 1408.82274 | 8.262315 | 16779 |
| sensor_6 | 21.6098 | 0.000878 | 16779 |
| sensor_7 | 553.378673 | 0.813975 | 16779 |
| sensor_8 | 2388.095956 | 0.065317 | 16779 |
| sensor_9 | 9064.656024 | 20.32118 | 16779 |
| sensor_11 | 47.537316 | 0.250992 | 16779 |
| sensor_12 | 521.421149 | 0.688006 | 16779 |
| sensor_13 | 2388.095602 | 0.066268 | 16779 |
| sensor_15 | 8.441602 | 0.033299 | 16779 |
| sensor_17 | 393.188889 | 1.326468 | 16779 |
| sensor_20 | 38.818372 | 0.159058 | 16779 |
| sensor_21 | 23.291152 | 0.095608 | 16779 |

Now express each reading as a distance from that reference:

> Rescaled reading = (reading − training mean) ÷ training spread.

Subtracting centres the scale on the mean. Dividing expresses the distance
in units of that sensor's typical spread, allowing different sensors to share
a comparable scale. For T24 at cycle 2, the smoothed input is 641.985; the
training mean is 642.6751709174424 and spread is 0.4303079837962261:

```text
distance from mean = 641.985 − 642.6751709174424 = −0.6901709174424
rescaled reading   = −0.6901709174424 / 0.4303079837962261 = −1.603900 (rounded)
```

The held-back and test tables use the **same** fitted numbers. Recalculating
them from those tables would let unseen engines influence preparation. A
column with effectively zero spread is set to zero to avoid division by zero.

<details class="process-example">
<summary>Example in Python</summary>

Continue after finding conditions:

```python
scale_input = conditioned.head(3)
normalizer = SensorNormalizer(method="zscore", columns=columns)
normalizer.fit(conditioned[conditioned.unit.isin(train_ids)])
scaled = normalizer.transform(conditioned)
scaled_test = normalizer.transform(conditioned_test)
scale_output = scaled.head(3)
scale_statistics = normalizer.statistics_[["column", "mean", "std", "n_rows"]]
```

</details>

##### Output

The same 28 columns, with only the selected sensors rescaled:

<!-- sample: scale_output -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul | operating_condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | -1.987346 | -0.146319 | -0.99521 | 14.62 | 0.227969 | 1.205599 | -0.550483 | -0.908708 | 1.3 | -0.268201 | 0.347164 | -1.140864 | 8138.62 | -0.663762 | 0.03 | -0.896282 | 2388 | 100 | 1.519113 | 1.337202 | 125 | 0 |
| 1 | 2 | 0.0019 | -0.0003 | 100 | 518.67 | -1.6039 | 0.060766 | -0.8415 | 14.62 | 0.227969 | 0.830894 | -0.703583 | -0.960871 | 1.3 | -0.228359 | 0.797742 | -0.763605 | 8131.49 | -0.47907 | 0.03 | -0.896282 | 2388 | 100 | 1.330503 | 1.361258 | 125 | 0 |
| 1 | 3 | -0.0043 | 0.0003 | 100 | 518.67 | -1.179785 | -0.209812 | -0.700499 | 14.62 | 0.227969 | 0.95682 | -0.473932 | -0.768707 | 1.3 | -0.646699 | 1.124774 | -0.876783 | 8133.23 | -0.596943 | 0.03 | -1.650164 | 2388 | 100 | 1.079023 | 0.958051 | 125 | 0 |

**Do not confuse this with rescaling to a fixed range.** The playground
option **Training range 0 to 1, all training rows** subtracts the training
minimum and divides by the training range (maximum minus minimum), placing
the training endpoints at 0 and 1. **Training range 0 to 1, per operating
condition** does that separately within each condition. New readings
outside the fitted range can fall outside 0–1. The default mean-and-spread
method pools training rows and does not use the condition number.

## 5 — Cut into windows

Engine histories have different lengths. A window gives the model a fixed
number of consecutive readings and the answer at the **end** of that history.
It must never cross from one engine to another.

{{figure:windowing}}

### Process

#### Input

The rescaled table. To show the whole operation, this teaching
example uses engine 1's first **five** rows with a window of **three** cycles
and a stride of **one** (move the start forward one row). These are real rows,
and the shorter teaching window does not change the normal run settings:

<!-- sample: window_input -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul | operating_condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | -1.987346 | -0.146319 | -0.99521 | 14.62 | 0.227969 | 1.205599 | -0.550483 | -0.908708 | 1.3 | -0.268201 | 0.347164 | -1.140864 | 8138.62 | -0.663762 | 0.03 | -0.896282 | 2388 | 100 | 1.519113 | 1.337202 | 125 | 0 |
| 1 | 2 | 0.0019 | -0.0003 | 100 | 518.67 | -1.6039 | 0.060766 | -0.8415 | 14.62 | 0.227969 | 0.830894 | -0.703583 | -0.960871 | 1.3 | -0.228359 | 0.797742 | -0.763605 | 8131.49 | -0.47907 | 0.03 | -0.896282 | 2388 | 100 | 1.330503 | 1.361258 | 125 | 0 |
| 1 | 3 | -0.0043 | 0.0003 | 100 | 518.67 | -1.179785 | -0.209812 | -0.700499 | 14.62 | 0.227969 | 0.95682 | -0.473932 | -0.768707 | 1.3 | -0.646699 | 1.124774 | -0.876783 | 8133.23 | -0.596943 | 0.03 | -1.650164 | 2388 | 100 | 1.079023 | 0.958051 | 125 | 0 |
| 1 | 4 | 0.0007 | 0 | 100 | 518.67 | -0.967728 | -0.853047 | -0.770999 | 14.62 | 0.227969 | 1.136493 | -0.129456 | -0.757757 | 1.3 | -1.134762 | 1.608055 | -0.556113 | 8133.83 | -1.400656 | 0.03 | -1.273223 | 2388 | 100 | 0.733238 | 0.911768 | 125 | 0 |
| 1 | 5 | -0.0019 | -0.0002 | 100 | 518.67 | -0.83846 | -1.168803 | -0.543006 | 14.62 | 0.227969 | 0.949909 | -0.339969 | -0.612773 | 1.3 | -1.079979 | 1.362781 | -0.697585 | 8133.8 | -0.883554 | 0.03 | -0.707812 | 2388 | 100 | 0.623215 | 1.048132 | 125 | 0 |

#### Function

Place a frame of the chosen length over one engine's history.
The readings inside it form a window; the answer belongs to its last cycle.
Move the frame forward by the chosen stride and repeat while a complete
window still fits. Start afresh for each engine.

For five rows, a length of three and a one-row stride, there are
5 − 3 + 1 = 3 complete windows. Their starting row positions, counted from
zero, are 0, 1 and 2:

```text
recorded cycles:   1   2   3   4   5
start position 0: [1   2   3]            answer at cycle 3
start position 1:     [2   3   4]        answer at cycle 4
start position 2:         [3   4   5]    answer at cycle 5
```

Position 0 means the first row, not cycle 0. A fourth start would need cycle 6,
which is outside this five-row sample. Within each window, keep each
sensor's readings in time order so its recent history remains visible.

<details class="process-example">
<summary>Example in Python</summary>

Continue after rescaling. The tables transpose each stored
sensor-by-cycle array only for display; they do not change the model input.

```python
window_input = scaled.query("unit == 1").head(5)
windows = make_windows(window_input, columns, window_size=3, stride=1)
window_0_output = pd.DataFrame(windows.features[0].T, columns=columns)
window_1_output = pd.DataFrame(windows.features[1].T, columns=columns)
window_2_output = pd.DataFrame(windows.features[2].T, columns=columns)
window_labels_output = pd.DataFrame({
    "window (from 0)": range(len(windows)), "unit": windows.units,
    "end_cycle": windows.end_cycles, "rul": windows.targets,
})
```

</details>

#### Output

Three windows. Each holds three consecutive readings for each
of fourteen sensors, together with the engine identifier, the last cycle and
the answer at that cycle. Here are all readings in each window, with cycles
as rows so you can compare them directly with the input:

Window 0 (cycles 1–3):

<!-- sample: window_0_output -->
| sensor_2 | sensor_3 | sensor_4 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_11 | sensor_12 | sensor_13 | sensor_15 | sensor_17 | sensor_20 | sensor_21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1.987346 | -0.146319 | -0.99521 | 0.227969 | 1.205599 | -0.550483 | -0.908708 | -0.268201 | 0.347164 | -1.140864 | -0.663762 | -0.896282 | 1.519113 | 1.337202 |
| -1.6039 | 0.060766 | -0.8415 | 0.227969 | 0.830894 | -0.703583 | -0.960871 | -0.228359 | 0.797742 | -0.763605 | -0.47907 | -0.896282 | 1.330503 | 1.361258 |
| -1.179785 | -0.209812 | -0.700499 | 0.227969 | 0.95682 | -0.473932 | -0.768707 | -0.646699 | 1.124774 | -0.876783 | -0.596943 | -1.650164 | 1.079023 | 0.958051 |

Window 1 (cycles 2–4):

<!-- sample: window_1_output -->
| sensor_2 | sensor_3 | sensor_4 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_11 | sensor_12 | sensor_13 | sensor_15 | sensor_17 | sensor_20 | sensor_21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1.6039 | 0.060766 | -0.8415 | 0.227969 | 0.830894 | -0.703583 | -0.960871 | -0.228359 | 0.797742 | -0.763605 | -0.47907 | -0.896282 | 1.330503 | 1.361258 |
| -1.179785 | -0.209812 | -0.700499 | 0.227969 | 0.95682 | -0.473932 | -0.768707 | -0.646699 | 1.124774 | -0.876783 | -0.596943 | -1.650164 | 1.079023 | 0.958051 |
| -0.967728 | -0.853047 | -0.770999 | 0.227969 | 1.136493 | -0.129456 | -0.757757 | -1.134762 | 1.608055 | -0.556113 | -1.400656 | -1.273223 | 0.733238 | 0.911768 |

Window 2 (cycles 3–5):

<!-- sample: window_2_output -->
| sensor_2 | sensor_3 | sensor_4 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_11 | sensor_12 | sensor_13 | sensor_15 | sensor_17 | sensor_20 | sensor_21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1.179785 | -0.209812 | -0.700499 | 0.227969 | 0.95682 | -0.473932 | -0.768707 | -0.646699 | 1.124774 | -0.876783 | -0.596943 | -1.650164 | 1.079023 | 0.958051 |
| -0.967728 | -0.853047 | -0.770999 | 0.227969 | 1.136493 | -0.129456 | -0.757757 | -1.134762 | 1.608055 | -0.556113 | -1.400656 | -1.273223 | 0.733238 | 0.911768 |
| -0.83846 | -1.168803 | -0.543006 | 0.227969 | 0.949909 | -0.339969 | -0.612773 | -1.079979 | 1.362781 | -0.697585 | -0.883554 | -0.707812 | 0.623215 | 1.048132 |

Each window is paired with this identification and answer:

<!-- sample: window_labels_output -->
| window (from 0) | unit | end_cycle | rul |
| --- | --- | --- | --- |
| 0 | 1 | 3 | 125 |
| 1 | 1 | 4 | 125 |
| 2 | 1 | 5 | 125 |

With the normal length of 30 and stride 1, engine 1 supplies cycles 1–30,
then 2–31, through 163–192: `192 − 30 + 1 = 163` windows. Across all training
histories there are 17,731 in total across the training and validation groups.
Their membership was already chosen in step 3. For testing, retain
only the final window of each engine: test engine 1 contributes cycles
2–31, paired with 112. An engine shorter than the chosen window is left out,
not padded with invented readings (the short-history choice, assumption A-23).

## Prepared windows

The algorithm above has produced windows for the engine groups chosen in
**Hold engines back**. There is no second split: a training engine supplies
training windows, a validation engine supplies validation windows, and a test
engine supplies a test window. Their readings and answers stay together.

The examples below trace that membership in the finished output. The Python
example collects windows by their engine identifiers after cutting them;
that is how the earlier decision is applied to the stored windows, not a new
decision about which engines to hold back.

### Three sets of windows

With some engines held back, there are training, validation and test sets.
This is the continuation of the seed-42 split used above.

#### Process

##### Input

The windows and the two engine lists from step 3. For a fully
visible example we take a three-cycle window from training-file engine 1, one
from training-file engine 3, and the final three cycles of test-file engine 1.
The following table contains **all input readings** of these sample windows.
`source` distinguishes the two files; `window` identifies a window within this
sample; each window occupies three rows. Those two columns are display labels,
not extra model features. `rul` is the answer at the window's final cycle.

<!-- sample: sets_input -->
| source | window | unit | cycle | sensor_2 | sensor_3 | sensor_4 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_11 | sensor_12 | sensor_13 | sensor_15 | sensor_17 | sensor_20 | sensor_21 | rul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| training file | 0 | 1 | 1 | -1.987346 | -0.146319 | -0.99521 | 0.227969 | 1.205599 | -0.550483 | -0.908708 | -0.268201 | 0.347164 | -1.140864 | -0.663762 | -0.896282 | 1.519113 | 1.337202 | 125 |
| training file | 0 | 1 | 2 | -1.6039 | 0.060766 | -0.8415 | 0.227969 | 0.830894 | -0.703583 | -0.960871 | -0.228359 | 0.797742 | -0.763605 | -0.47907 | -0.896282 | 1.330503 | 1.361258 | 125 |
| training file | 0 | 1 | 3 | -1.179785 | -0.209812 | -0.700499 | 0.227969 | 0.95682 | -0.473932 | -0.768707 | -0.646699 | 1.124774 | -0.876783 | -0.596943 | -1.650164 | 1.079023 | 0.958051 | 125 |
| training file | 1 | 3 | 1 | -1.476084 | -1.220818 | -1.294158 | 0.227969 | 0.714183 | -0.550483 | -0.379211 | -1.582983 | 1.19018 | -1.442671 | -0.627725 | -1.650164 | 0.890413 | 0.306957 | 125 |
| training file | 1 | 3 | 2 | -0.75567 | -0.943402 | -1.264505 | 0.227969 | 1.205599 | -0.627033 | -0.312286 | -1.164643 | 1.037565 | -1.367219 | -1.015129 | -2.027105 | 1.236198 | 0.915691 | 125 |
| training file | 1 | 3 | 3 | -1.081019 | -1.419112 | -1.0996 | 0.227969 | 1.297739 | -0.818408 | -0.506173 | -1.772232 | 1.055734 | -1.329493 | -1.402533 | -1.084753 | 1.28335 | 0.588836 | 125 |
| test file | 0 | 1 | 29 | -1.53328 | -0.368687 | -1.078744 | 0.227969 | 0.804907 | -0.263126 | -0.776748 | -0.922898 | 1.039849 | -0.466803 | -0.387229 | -0.902913 | 0.734379 | 0.756251 | 112 |
| test file | 0 | 1 | 30 | -0.633214 | -0.646276 | -1.014586 | 0.227969 | 0.845543 | -0.177153 | -0.811726 | -0.734996 | 0.780715 | -0.20022 | -0.653129 | -1.276539 | 1.221051 | 0.983448 | 112 |
| test file | 0 | 1 | 31 | -0.427191 | -1.224639 | -1.10717 | 0.227969 | 1.062427 | -0.210718 | -0.609001 | -0.979701 | 0.658416 | -0.368735 | -0.915213 | -0.709469 | 0.584207 | 0.826672 | 112 |

##### Function

Match every window to its engine's assigned group. A window
from a kept training engine goes to training; one from a held-back engine
goes to validation; one from the test file goes to test. Keep the readings
and their answer together. This rule applies to any window length.

In the sample, training-file engine 1 goes to training, training-file engine 3
goes to validation, and test-file engine 1 goes to test. The matching engine
number does not make the two engine-1 histories the same engine.

<details class="process-example">
<summary>Example in Python</summary>

Continue after step 5. The loop expands each window into
three display rows; it does not create new sensor values.

```python
mini_train = make_windows(
    scaled.query("unit in [1, 3] and cycle <= 3"), columns, window_size=3,
)
mini_test = make_last_windows(
    scaled_test.query("unit == 1"), columns, window_size=3,
)
rows = []
for source, dataset in (("training file", mini_train), ("test file", mini_test)):
    for i in range(len(dataset)):
        for t in range(3):
            rows.append({
                "source": source, "window": i, "unit": int(dataset.units[i]),
                "cycle": int(dataset.end_cycles[i]) - 2 + t,
                **dict(zip(columns, dataset.features[i, :, t].astype(float))),
                "rul": float(dataset.targets[i]),
            })
sets_input = pd.DataFrame(rows)
mini_kept = mini_train.subset_by_units(train_ids)
mini_held = mini_train.subset_by_units(held_ids)
sets_output = sets_input.copy()
sets_output["destination"] = [
    "test" if row.source == "test file" else
    "train" if row.unit in mini_kept.units else "validation"
    for row in sets_input.itertuples()
]
prepared = run_preprocessing(PreprocessingConfig())
three_shapes = pd.DataFrame([
    {"Set": name, "Windows": len(getattr(prepared, name)),
     "Sensors per window": 14, "Cycles per sensor": 30,
     "Answers": len(getattr(prepared, name).targets)}
    for name in ("train", "validation", "test")
])
```

</details>

##### Output

The same complete sample, with its destination attached. Each
window's readings and answer are unchanged:

<!-- sample: sets_output -->
| source | window | unit | cycle | sensor_2 | sensor_3 | sensor_4 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_11 | sensor_12 | sensor_13 | sensor_15 | sensor_17 | sensor_20 | sensor_21 | rul | destination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| training file | 0 | 1 | 1 | -1.987346 | -0.146319 | -0.99521 | 0.227969 | 1.205599 | -0.550483 | -0.908708 | -0.268201 | 0.347164 | -1.140864 | -0.663762 | -0.896282 | 1.519113 | 1.337202 | 125 | train |
| training file | 0 | 1 | 2 | -1.6039 | 0.060766 | -0.8415 | 0.227969 | 0.830894 | -0.703583 | -0.960871 | -0.228359 | 0.797742 | -0.763605 | -0.47907 | -0.896282 | 1.330503 | 1.361258 | 125 | train |
| training file | 0 | 1 | 3 | -1.179785 | -0.209812 | -0.700499 | 0.227969 | 0.95682 | -0.473932 | -0.768707 | -0.646699 | 1.124774 | -0.876783 | -0.596943 | -1.650164 | 1.079023 | 0.958051 | 125 | train |
| training file | 1 | 3 | 1 | -1.476084 | -1.220818 | -1.294158 | 0.227969 | 0.714183 | -0.550483 | -0.379211 | -1.582983 | 1.19018 | -1.442671 | -0.627725 | -1.650164 | 0.890413 | 0.306957 | 125 | validation |
| training file | 1 | 3 | 2 | -0.75567 | -0.943402 | -1.264505 | 0.227969 | 1.205599 | -0.627033 | -0.312286 | -1.164643 | 1.037565 | -1.367219 | -1.015129 | -2.027105 | 1.236198 | 0.915691 | 125 | validation |
| training file | 1 | 3 | 3 | -1.081019 | -1.419112 | -1.0996 | 0.227969 | 1.297739 | -0.818408 | -0.506173 | -1.772232 | 1.055734 | -1.329493 | -1.402533 | -1.084753 | 1.28335 | 0.588836 | 125 | validation |
| test file | 0 | 1 | 29 | -1.53328 | -0.368687 | -1.078744 | 0.227969 | 0.804907 | -0.263126 | -0.776748 | -0.922898 | 1.039849 | -0.466803 | -0.387229 | -0.902913 | 0.734379 | 0.756251 | 112 | test |
| test file | 0 | 1 | 30 | -0.633214 | -0.646276 | -1.014586 | 0.227969 | 0.845543 | -0.177153 | -0.811726 | -0.734996 | 0.780715 | -0.20022 | -0.653129 | -1.276539 | 1.221051 | 0.983448 | 112 | test |
| test file | 0 | 1 | 31 | -0.427191 | -1.224639 | -1.10717 | 0.227969 | 1.062427 | -0.210718 | -0.609001 | -0.979701 | 0.658416 | -0.368735 | -0.915213 | -0.709469 | 0.584207 | 0.826672 | 112 | test |

For the normal run, the groups have these sizes. Every window contains
fourteen sensor histories of thirty cycles, paired with one answer:

<!-- sample: three_shapes -->
| Set | Windows | Sensors per window | Cycles per sensor | Answers |
| --- | --- | --- | --- | --- |
| train | 14459 | 14 | 30 | 14459 |
| validation | 3272 | 14 | 30 | 3272 |
| test | 100 | 14 | 30 | 100 |

### Two sets of windows

With no engines held back, every training-file window is used for training;
the separate test file still supplies test windows. There is no validation
score to choose a saved model or a stopping time.

#### Process

##### Input

Windows prepared with **no training engines held back**. Their
rescaling references use all training engines. It is not the
same as taking previously rescaled validation windows and renaming them.
For comparison, the sample below uses the same engines and cycles as the
three-set example, but prepared with this setting. Every reading is shown:

<!-- sample: two_sets_input -->
| source | window | unit | cycle | sensor_2 | sensor_3 | sensor_4 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_11 | sensor_12 | sensor_13 | sensor_15 | sensor_17 | sensor_20 | sensor_21 | rul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| training file | 0 | 1 | 1 | -1.98771 | -0.145284 | -0.991541 | 0.228913 | 1.208116 | -0.544003 | -0.882684 | -0.267978 | 0.343758 | -1.129029 | -0.664694 | -0.897481 | 1.518021 | 1.340645 | 125 |
| training file | 0 | 1 | 2 | -1.603946 | 0.061111 | -0.838002 | 0.228913 | 0.832057 | -0.696109 | -0.932334 | -0.228095 | 0.795129 | -0.753886 | -0.480091 | -0.897481 | 1.329254 | 1.364765 | 125 |
| training file | 0 | 1 | 3 | -1.17948 | -0.208565 | -0.697157 | 0.228913 | 0.958438 | -0.467951 | -0.749423 | -0.64687 | 1.122736 | -0.866429 | -0.597907 | -1.651753 | 1.077564 | 0.960481 | 125 |
| training file | 1 | 3 | 1 | -1.476025 | -1.216198 | -1.290156 | 0.228913 | 0.714924 | -0.544003 | -0.378681 | -1.584128 | 1.188258 | -1.429143 | -0.628674 | -1.651753 | 0.888797 | 0.307649 | 125 |
| training file | 1 | 3 | 2 | -0.755014 | -0.939707 | -1.260537 | 0.228913 | 1.208116 | -0.620056 | -0.314978 | -1.165353 | 1.035374 | -1.354115 | -1.01589 | -2.028889 | 1.23487 | 0.918008 | 125 |
| training file | 1 | 3 | 3 | -1.080632 | -1.41383 | -1.095815 | 0.228913 | 1.300589 | -0.810188 | -0.499529 | -1.773574 | 1.053575 | -1.3166 | -1.403107 | -1.086049 | 1.282062 | 0.590281 | 125 |
| test file | 0 | 1 | 29 | -1.533268 | -0.366909 | -1.074982 | 0.228913 | 0.805976 | -0.258514 | -0.757077 | -0.923356 | 1.037662 | -0.458748 | -0.388295 | -0.904115 | 0.732633 | 0.758143 | 112 |
| test file | 0 | 1 | 30 | -0.632456 | -0.643573 | -1.010896 | 0.228913 | 0.846759 | -0.1731 | -0.790371 | -0.735258 | 0.778072 | -0.19366 | -0.654066 | -1.277934 | 1.219711 | 0.985946 | 112 |
| test file | 0 | 1 | 31 | -0.426263 | -1.220006 | -1.103376 | 0.228913 | 1.064427 | -0.206446 | -0.597407 | -0.980218 | 0.655557 | -0.36123 | -0.916023 | -0.710571 | 0.582336 | 0.828751 | 112 |

##### Function

The empty held-back list reserves no windows. All windows from
the training file go to training, including engine 3; the test window stays
test. Here the assignment itself does not change any input value.

<details class="process-example">
<summary>Example in Python</summary>

The preprocessing call below repeats steps 1–5 with no held-back
engines. It uses real three-cycle windows to match the displayed sample; the
second call gives the normal 30-cycle counts.

```python
mini_all = run_preprocessing(PreprocessingConfig(validation_fraction=0.0, window_size=3))
rows = []
for source, dataset in (("training file", mini_all.train), ("test file", mini_all.test)):
    indices = [i for i, (unit, end) in enumerate(zip(dataset.units, dataset.end_cycles))
               if (source == "training file" and unit in (1, 3) and end == 3)
               or (source == "test file" and unit == 1)]
    for window_number, i in enumerate(indices):
        for t in range(3):
            rows.append({
                "source": source, "window": window_number, "unit": int(dataset.units[i]),
                "cycle": int(dataset.end_cycles[i]) - 2 + t,
                **dict(zip(columns, dataset.features[i, :, t].astype(float))),
                "rul": float(dataset.targets[i]),
            })
two_sets_input = pd.DataFrame(rows)
two_sets_output = two_sets_input.copy()
two_sets_output["destination"] = [
    "train" if source == "training file" else "test" for source in two_sets_input.source
]
all_prepared = run_preprocessing(PreprocessingConfig(validation_fraction=0.0))
two_shapes = pd.DataFrame([
    {"Set": name, "Windows": len(getattr(all_prepared, name)),
     "Sensors per window": 14, "Cycles per sensor": 30,
     "Answers": len(getattr(all_prepared, name).targets)}
    for name in ("train", "validation", "test")
])
```

</details>

##### Output

These are the same sample windows after assigning their sets:

<!-- sample: two_sets_output -->
| source | window | unit | cycle | sensor_2 | sensor_3 | sensor_4 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_11 | sensor_12 | sensor_13 | sensor_15 | sensor_17 | sensor_20 | sensor_21 | rul | destination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| training file | 0 | 1 | 1 | -1.98771 | -0.145284 | -0.991541 | 0.228913 | 1.208116 | -0.544003 | -0.882684 | -0.267978 | 0.343758 | -1.129029 | -0.664694 | -0.897481 | 1.518021 | 1.340645 | 125 | train |
| training file | 0 | 1 | 2 | -1.603946 | 0.061111 | -0.838002 | 0.228913 | 0.832057 | -0.696109 | -0.932334 | -0.228095 | 0.795129 | -0.753886 | -0.480091 | -0.897481 | 1.329254 | 1.364765 | 125 | train |
| training file | 0 | 1 | 3 | -1.17948 | -0.208565 | -0.697157 | 0.228913 | 0.958438 | -0.467951 | -0.749423 | -0.64687 | 1.122736 | -0.866429 | -0.597907 | -1.651753 | 1.077564 | 0.960481 | 125 | train |
| training file | 1 | 3 | 1 | -1.476025 | -1.216198 | -1.290156 | 0.228913 | 0.714924 | -0.544003 | -0.378681 | -1.584128 | 1.188258 | -1.429143 | -0.628674 | -1.651753 | 0.888797 | 0.307649 | 125 | train |
| training file | 1 | 3 | 2 | -0.755014 | -0.939707 | -1.260537 | 0.228913 | 1.208116 | -0.620056 | -0.314978 | -1.165353 | 1.035374 | -1.354115 | -1.01589 | -2.028889 | 1.23487 | 0.918008 | 125 | train |
| training file | 1 | 3 | 3 | -1.080632 | -1.41383 | -1.095815 | 0.228913 | 1.300589 | -0.810188 | -0.499529 | -1.773574 | 1.053575 | -1.3166 | -1.403107 | -1.086049 | 1.282062 | 0.590281 | 125 | train |
| test file | 0 | 1 | 29 | -1.533268 | -0.366909 | -1.074982 | 0.228913 | 0.805976 | -0.258514 | -0.757077 | -0.923356 | 1.037662 | -0.458748 | -0.388295 | -0.904115 | 0.732633 | 0.758143 | 112 | test |
| test file | 0 | 1 | 30 | -0.632456 | -0.643573 | -1.010896 | 0.228913 | 0.846759 | -0.1731 | -0.790371 | -0.735258 | 0.778072 | -0.19366 | -0.654066 | -1.277934 | 1.219711 | 0.985946 | 112 | test |
| test file | 0 | 1 | 31 | -0.426263 | -1.220006 | -1.103376 | 0.228913 | 1.064427 | -0.206446 | -0.597407 | -0.980218 | 0.655557 | -0.36123 | -0.916023 | -0.710571 | 0.582336 | 0.828751 | 112 | test |

In the normal 30-cycle run, all training-file windows belong to training
and the validation group is empty:

<!-- sample: two_shapes -->
| Set | Windows | Sensors per window | Cycles per sensor | Answers |
| --- | --- | --- | --- | --- |
| train | 17731 | 14 | 30 | 17731 |
| validation | 0 | 14 | 30 | 0 |
| test | 100 | 14 | 30 | 100 |

Test answers remain for final evaluation. They must not replace validation
answers for choosing settings or deciding when to stop.

The model now receives windows of sensor readings with separate answers.
The [temporal stage](09_temporal.md) determines how each sensor's sequence is
turned into the numbers passed to the graph. The worked tables above are
checked by `python -m pytest tests/test_examples.py -k design_preprocessing`.

**Corrections.** The earlier claim that capping test answers improves all
reported measures was too broad; different target definitions need not move
every measure in the same direction. The RUL chart also previously stretched
an uncapped line from 125 and used row positions as cycles; it now uses the
recorded cycle numbers and actual remaining lives on the same scale.
