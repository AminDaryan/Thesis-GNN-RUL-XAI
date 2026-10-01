# Data: from engine histories to known answers

## Remaining useful life

**Remaining useful life (RUL)** is how many recorded cycles an engine has left
before failure. Predictive maintenance uses this estimate to plan servicing:
too early wastes usable life; too late risks a failure in service. A cycle is
the time step in these files, with one set of readings per engine per cycle.

## What the engine's sensors measure

{{figure:engine}}

Air passes through a fan, a low-pressure compressor (LPC), a high-pressure
compressor (HPC), a combustor, and high- and low-pressure turbines (HPT and LPT).
Sensors record temperatures, pressures, shaft speeds and other quantities along
this path. For example, `T24` measures temperature at the LPC outlet. Changes
upstream can affect readings downstream. The model's sensor graph represents
chosen connections between sensors; a connection is not proof of a causal effect.

## NASA's benchmark data: C-MAPSS

**C-MAPSS**, Commercial Modular Aero-Propulsion System Simulation, is NASA's
engine simulator. This tutorial uses the distributed **FD001** engine records
in `data/raw/`; it does not create replacement sensor readings.

Training histories continue to failure. Test histories stop earlier, and a
separate answer file gives the life remaining at each test engine's last
recorded cycle. Training engine 1 and test engine 1 are different engines:
their identifiers belong to separate files.

Each **Process** follows the same sample from input to output.
Tables show every column of the sample, with no omitted readings. Read each
process as a transformation of information: what we start with, what we do to
it, and what we have afterwards. The Python examples are optional. You can
understand the process from the tables and worked arithmetic alone.

<div class="algorithm-intro">
<p class="algorithm-label">Data preparation algorithm</p>
<p>Start with the raw engine histories. Follow the steps below in order to
produce labelled tables and select the sensor columns the model will receive.</p>
</div>

## 1 — Read the engine histories

### Training engines

Before there can be a prediction, each reading needs an engine identifier and
a cycle number. Otherwise readings from different engines could be mixed into
one history. Reading the file gives the numbers names and puts them in order.

#### Process

##### Input

`data/raw/train_FD001.txt`. Its first three complete lines are:

```text
1 1 -0.0007 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 21.61 554.36 2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 8.4195 0.03 392 2388 100.00 39.06 23.4190
1 2 0.0019 -0.0003 100.0 518.67 642.15 1591.82 1403.14 14.62 21.61 553.75 2388.04 9044.07 1.30 47.49 522.28 2388.07 8131.49 8.4318 0.03 392 2388 100.00 39.00 23.4236
1 3 -0.0043 0.0003 100.0 518.67 642.35 1587.99 1404.20 14.62 21.61 554.26 2388.08 9052.94 1.30 47.27 522.42 2388.03 8133.23 8.4178 0.03 390 2388 100.00 38.95 23.3442
```

Each line contains the engine number, cycle, three operating settings (how the
engine is being run), and all 21 sensor readings. There is no header in the file.

##### Function

Organise the readings into a table. Each position in a line
has a meaning: engine, cycle, operating settings, then sensors. Giving these
positions names lets us follow one engine through time without mixing it
with another. Keep all rows of an engine together and put its cycles in order.
No measurement changes during this operation.

For the first sample line, the first 1 identifies engine 1 and the second 1
identifies cycle 1. The three values after them describe how the engine was
being run. Among the sensor values, 641.82 is the temperature named T24. The
next two lines are cycles 2 and 3 of that same engine. The output below labels
engine and cycle as `unit` and `cycle`, and T24 as `sensor_2`.

<details class="process-example">
<summary>Example in Python</summary>

Run the code blocks on this page in order from the repository
root after installing the project. This block starts the example:

```python
from thesis_gnn_rul_xai.data.loading import (
    attach_rul, read_cmapss_frame, read_rul_file,
)
from thesis_gnn_rul_xai.data.sensors import (
    PAPER_SENSOR_COLUMNS, SENSORS, detect_uninformative_sensors,
)
import pandas as pd

training = read_cmapss_frame("data/raw/train_FD001.txt")
training_output = training.head(3)
```

</details>

##### Output

The complete training table has 20,631 rows and 26 columns. These
are the same three input lines, now as a named table; there is no answer column
yet:

<!-- sample: training_output -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | 641.82 | 1589.7 | 1400.6 | 14.62 | 21.61 | 554.36 | 2388.06 | 9046.19 | 1.3 | 47.47 | 521.66 | 2388.02 | 8138.62 | 8.4195 | 0.03 | 392 | 2388 | 100 | 39.06 | 23.419 |
| 1 | 2 | 0.0019 | -0.0003 | 100 | 518.67 | 642.15 | 1591.82 | 1403.14 | 14.62 | 21.61 | 553.75 | 2388.04 | 9044.07 | 1.3 | 47.49 | 522.28 | 2388.07 | 8131.49 | 8.4318 | 0.03 | 392 | 2388 | 100 | 39 | 23.4236 |
| 1 | 3 | -0.0043 | 0.0003 | 100 | 518.67 | 642.35 | 1587.99 | 1404.2 | 14.62 | 21.61 | 554.26 | 2388.08 | 9052.94 | 1.3 | 47.27 | 522.42 | 2388.03 | 8133.23 | 8.4178 | 0.03 | 390 | 2388 | 100 | 38.95 | 23.3442 |

### Calculate training labels

A model learns from examples paired with known answers, called **labels**.
For a training engine we know when its history ends in failure, so we can
count how many cycles remain at every earlier row. This is arithmetic on the
recorded history; it does not require a trained model.

#### Process

##### Input

The complete table from **Training engines**. The same three sample
rows enter unchanged:

<!-- sample: labels_input -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | 641.82 | 1589.7 | 1400.6 | 14.62 | 21.61 | 554.36 | 2388.06 | 9046.19 | 1.3 | 47.47 | 521.66 | 2388.02 | 8138.62 | 8.4195 | 0.03 | 392 | 2388 | 100 | 39.06 | 23.419 |
| 1 | 2 | 0.0019 | -0.0003 | 100 | 518.67 | 642.15 | 1591.82 | 1403.14 | 14.62 | 21.61 | 553.75 | 2388.04 | 9044.07 | 1.3 | 47.49 | 522.28 | 2388.07 | 8131.49 | 8.4318 | 0.03 | 392 | 2388 | 100 | 39 | 23.4236 |
| 1 | 3 | -0.0043 | 0.0003 | 100 | 518.67 | 642.35 | 1587.99 | 1404.2 | 14.62 | 21.61 | 554.26 | 2388.08 | 9052.94 | 1.3 | 47.27 | 522.42 | 2388.03 | 8133.23 | 8.4178 | 0.03 | 390 | 2388 | 100 | 38.95 | 23.3442 |

##### Function

Count the distance from the current cycle to that engine's
failure cycle. For any complete training history, the rule is:

> Remaining life = final cycle − current cycle.

Subtracting tells us how many cycles still lie ahead. The final cycle is
different for each engine, so the calculation must stay within one history.
Engine 1 ends at cycle 192; applying the rule to the sample gives:

| Row's cycle | Engine's final cycle | Subtraction | Remaining cycles |
|---|---|---|---|
| 1 | 192 | 192 − 1 | 191 |
| 2 | 192 | 192 − 2 | 190 |
| 3 | 192 | 192 − 3 | 189 |

The final cycle must be found from the **complete history**, not these three
displayed rows. Looking only at this sample would mistake cycle 3 for the end
and undercount the life remaining. Engine 2 uses its own final cycle, not
engine 1's.

<details class="process-example">
<summary>Example in Python</summary>

Continue from the [file-reading example](#training-engines):

```python
labels_input = training.head(3)
labelled = attach_rul(training)
labels_output = labelled.head(3)
assert training.query("unit == 1").cycle.max() == 192
assert labels_output.rul.tolist() == [191, 190, 189]
```

</details>

##### Output

The same rows and readings, with one extra column: the known
remaining life, labelled `rul`. The full table now has 20,631 rows and 27
columns. In the sample, only the answer column is new:

<!-- sample: labels_output -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | 641.82 | 1589.7 | 1400.6 | 14.62 | 21.61 | 554.36 | 2388.06 | 9046.19 | 1.3 | 47.47 | 521.66 | 2388.02 | 8138.62 | 8.4195 | 0.03 | 392 | 2388 | 100 | 39.06 | 23.419 | 191 |
| 1 | 2 | 0.0019 | -0.0003 | 100 | 518.67 | 642.15 | 1591.82 | 1403.14 | 14.62 | 21.61 | 553.75 | 2388.04 | 9044.07 | 1.3 | 47.49 | 522.28 | 2388.07 | 8131.49 | 8.4318 | 0.03 | 392 | 2388 | 100 | 39 | 23.4236 | 190 |
| 1 | 3 | -0.0043 | 0.0003 | 100 | 518.67 | 642.35 | 1587.99 | 1404.2 | 14.62 | 21.61 | 554.26 | 2388.08 | 9052.94 | 1.3 | 47.27 | 522.42 | 2388.03 | 8133.23 | 8.4178 | 0.03 | 390 | 2388 | 100 | 38.95 | 23.3442 | 189 |

**Do not confuse this with capping the labels.** These are the recorded cycles
left. [Step 1 of Preprocessing, Cap the labels](04_the_pipeline.md#1--cap-the-labels), changes
large answers to a ceiling. The answer stays separate from sensor inputs.

### Test engines

A test engine has not failed at its final recorded row. Subtracting its current
cycle from its last recorded cycle therefore gives only the distance to the
cut-off, not the life left. The separate answer file supplies the missing part.

#### Process

##### Input

`data/raw/test_FD001.txt` has the same 26-column format. These are
its first three rows after reading:

<!-- sample: test_input -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.0023 | 0.0003 | 100 | 518.67 | 643.02 | 1585.29 | 1398.21 | 14.62 | 21.61 | 553.9 | 2388.04 | 9050.17 | 1.3 | 47.2 | 521.72 | 2388.03 | 8125.55 | 8.4052 | 0.03 | 392 | 2388 | 100 | 38.86 | 23.3735 |
| 1 | 2 | -0.0027 | -0.0003 | 100 | 518.67 | 641.71 | 1588.45 | 1395.42 | 14.62 | 21.61 | 554.85 | 2388.01 | 9054.42 | 1.3 | 47.5 | 522.16 | 2388.06 | 8139.62 | 8.3803 | 0.03 | 393 | 2388 | 100 | 39.02 | 23.3916 |
| 1 | 3 | 0.0003 | 0.0001 | 100 | 518.67 | 642.46 | 1586.94 | 1401.34 | 14.62 | 21.61 | 554.11 | 2388.05 | 9056.96 | 1.3 | 47.5 | 521.97 | 2388.03 | 8130.1 | 8.4441 | 0.03 | 393 | 2388 | 100 | 39.08 | 23.4166 |

The second input is `data/raw/RUL_FD001.txt`: line 1 belongs to test engine 1,
line 2 to test engine 2, and so on. Its first three entries are:

<!-- sample: test_answers -->
| unit | rul_at_last_cycle |
| --- | --- |
| 1 | 112 |
| 2 | 98 |
| 3 | 69 |

##### Function

Split the remaining life into two stretches: the cycles from
the current row to the recording cut-off, and the cycles still left after
that cut-off. Add them to get the total life left at the current row:

> Remaining life = (last recorded cycle − current cycle) + life left at cut-off.

This works for any test engine with a known cut-off answer. The first stretch
comes from its history; the second comes from the separate answer file. Test
engine 1 stops at cycle 31 and still has 112 cycles left there:

| Cycle | Cycles until cut-off | Life left at cut-off | Answer for this row |
|---|---|---|---|
| 1 | 31 − 1 = 30 | 112 | 30 + 112 = 142 |
| 2 | 31 − 2 = 29 | 112 | 29 + 112 = 141 |
| 3 | 31 − 3 = 28 | 112 | 28 + 112 = 140 |

At cycle 31 the calculation is `31 − 31 + 112 = 112`. The earlier-row answers
can be calculated, but normal test evaluation uses only the final window of
each engine, paired with the answer at its end.

<details class="process-example">
<summary>Example in Python</summary>

Use the imports from the [file-reading example](#training-engines):

```python
history = read_cmapss_frame("data/raw/test_FD001.txt")
answers = read_rul_file("data/raw/RUL_FD001.txt")
test_input = history.head(3)
test_answers = answers.head(3).rename_axis("unit").reset_index()
test_labelled = attach_rul(history, rul_at_last_cycle=answers)
test_output = test_labelled.head(3)
assert history.query("unit == 1").cycle.max() == 31
assert test_output.rul.tolist() == [142, 141, 140]
```

</details>

##### Output

The original readings plus a remaining-life answer for each row.
The complete test table has 13,096 rows and 27 columns. For the sample:

<!-- sample: test_output -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.0023 | 0.0003 | 100 | 518.67 | 643.02 | 1585.29 | 1398.21 | 14.62 | 21.61 | 553.9 | 2388.04 | 9050.17 | 1.3 | 47.2 | 521.72 | 2388.03 | 8125.55 | 8.4052 | 0.03 | 392 | 2388 | 100 | 38.86 | 23.3735 | 142 |
| 1 | 2 | -0.0027 | -0.0003 | 100 | 518.67 | 641.71 | 1588.45 | 1395.42 | 14.62 | 21.61 | 554.85 | 2388.01 | 9054.42 | 1.3 | 47.5 | 522.16 | 2388.06 | 8139.62 | 8.3803 | 0.03 | 393 | 2388 | 100 | 39.02 | 23.3916 | 141 |
| 1 | 3 | 0.0003 | 0.0001 | 100 | 518.67 | 642.46 | 1586.94 | 1401.34 | 14.62 | 21.61 | 554.11 | 2388.05 | 9056.96 | 1.3 | 47.5 | 521.97 | 2388.03 | 8130.1 | 8.4441 | 0.03 | 393 | 2388 | 100 | 39.08 | 23.4166 | 140 |

**Do not confuse test answers with training inputs.** Test answers score the
finished predictions. Neither those answers nor `rul` enter the sensor array
given to the model.

## 2 — Check which sensors vary

A column that never changes cannot distinguish an engine approaching failure
from a healthy one. This check describes the data before selecting sensors;
it does **not** automatically remove columns or choose a different model.

### Process

#### Input

The complete training histories, with all sensor readings. The same three
engine-1 rows illustrate the input; the check must also examine every other row:

<!-- sample: diagnostic_input -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | 641.82 | 1589.7 | 1400.6 | 14.62 | 21.61 | 554.36 | 2388.06 | 9046.19 | 1.3 | 47.47 | 521.66 | 2388.02 | 8138.62 | 8.4195 | 0.03 | 392 | 2388 | 100 | 39.06 | 23.419 |
| 1 | 2 | 0.0019 | -0.0003 | 100 | 518.67 | 642.15 | 1591.82 | 1403.14 | 14.62 | 21.61 | 553.75 | 2388.04 | 9044.07 | 1.3 | 47.49 | 522.28 | 2388.07 | 8131.49 | 8.4318 | 0.03 | 392 | 2388 | 100 | 39 | 23.4236 |
| 1 | 3 | -0.0043 | 0.0003 | 100 | 518.67 | 642.35 | 1587.99 | 1404.2 | 14.62 | 21.61 | 554.26 | 2388.08 | 9052.94 | 1.3 | 47.27 | 522.42 | 2388.03 | 8133.23 | 8.4178 | 0.03 | 390 | 2388 | 100 | 38.95 | 23.3442 |

#### Function

Across all training histories, compare each sensor's smallest and largest
readings and count its distinct values. A zero range means no change at all.
Fewer than three distinct values means the reading can switch between at most
two levels. The result below uses the complete file; three sample rows alone
cannot establish that a sensor never varies:

<details class="process-example">
<summary>Example in Python</summary>

Continue with the loaded training table:

```python
diagnostic_input = training.head(3)
report = detect_uninformative_sensors(labelled)
diagnostic_output = pd.DataFrame([
    {"check": "constant", "sensors": ", ".join(report["constant"])},
    {"check": "fewer than three distinct values", "sensors": ", ".join(report["low_resolution"])},
])
assert sorted(labelled.sensor_6.unique().tolist()) == [21.6, 21.61]
```

</details>

#### Output

A report of the sensors that stay constant or take fewer than three distinct
values. The original readings remain unchanged.

<!-- sample: diagnostic_output -->
| check | sensors |
| --- | --- |
| constant | Nf_dmd, P2, PCNfR_dmd, T2, epr, farB |
| fewer than three distinct values | P15 |

For example, `P15` takes the values 21.60 and 21.61 in the complete training
file. That makes it a two-level reading rather than a gradually varying one.
The project retains it in the configured sensor set; this limitation is
recorded in the [sensor-selection finding](#/entry/F-01).

## 3 — Select 14 sensor columns for model inputs

The model needs sensor readings in a known order. Engine identifiers and cycle
numbers organise the histories; operating settings guide preparation; labels
are the answers. None of these is an extra sensor input.

{{figure:sensor-selection}}

The project uses one fixed list of 14 sensor columns. The six constant sensors
reported above are absent, as is `NRc`, the corrected core-speed reading
(the recorded selection choice, assumption A-10). In the full FD001 training
file, `Nc` and `NRc` have correlation 0.963157 when rounded to six decimal
places; 1 would mean they move perfectly together. They describe the same
core shaft's speed in different forms, so much of their variation overlaps.
The fixed list keeps `Nc`. `P15` stays in the list because removing it would
also change the chosen sensor graph; the diagnostic reports its limited
variation without silently redesigning the model.
The graph's rows and columns use the same sensor order, so changing the order
would connect the wrong readings to one another.

{{figure:correlation}}

This chart compares how readings rise and fall together across the training
file. A positive correlation means they tend to move in the same direction;
a negative one means opposite directions. Correlation describes co-variation,
not a causal connection. The chart helps inspect the selection; it does not
change the fixed list when preprocessing runs.

### Process

#### Input

The labelled table, still containing all 21 sensor columns. Here
are the same three training rows used in the label calculation:

<!-- sample: selection_input -->
| unit | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 | rul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.0007 | -0.0004 | 100 | 518.67 | 641.82 | 1589.7 | 1400.6 | 14.62 | 21.61 | 554.36 | 2388.06 | 9046.19 | 1.3 | 47.47 | 521.66 | 2388.02 | 8138.62 | 8.4195 | 0.03 | 392 | 2388 | 100 | 39.06 | 23.419 | 191 |
| 1 | 2 | 0.0019 | -0.0003 | 100 | 518.67 | 642.15 | 1591.82 | 1403.14 | 14.62 | 21.61 | 553.75 | 2388.04 | 9044.07 | 1.3 | 47.49 | 522.28 | 2388.07 | 8131.49 | 8.4318 | 0.03 | 392 | 2388 | 100 | 39 | 23.4236 | 190 |
| 1 | 3 | -0.0043 | 0.0003 | 100 | 518.67 | 642.35 | 1587.99 | 1404.2 | 14.62 | 21.61 | 554.26 | 2388.08 | 9052.94 | 1.3 | 47.27 | 522.42 | 2388.03 | 8133.23 | 8.4178 | 0.03 | 390 | 2388 | 100 | 38.95 | 23.3442 | 189 |

#### Function

Keep the same chosen sensor columns from every row, in the
same order. Think of this as selecting vertical strips of a table: every
chosen strip keeps all its readings, and every reading stays on its original
row. Selection changes which information is passed on, not the values.

For the first sample row, T24 contributes 641.82, T30 contributes 1589.7 and
T50 contributes 1400.6 as the first three values. Rows 2 and 3 contribute their
own readings from those same columns. The complete order is shown below.

The complete mapping between table names and sensor names is:

<!-- sample: selection_columns -->
| position (from 0) | column | sensor |
| --- | --- | --- |
| 0 | sensor_2 | T24 |
| 1 | sensor_3 | T30 |
| 2 | sensor_4 | T50 |
| 3 | sensor_6 | P15 |
| 4 | sensor_7 | P30 |
| 5 | sensor_8 | Nf |
| 6 | sensor_9 | Nc |
| 7 | sensor_11 | Ps30 |
| 8 | sensor_12 | phi |
| 9 | sensor_13 | NRf |
| 10 | sensor_15 | BPR |
| 11 | sensor_17 | htBleed |
| 12 | sensor_20 | W31 |
| 13 | sensor_21 | W32 |

<details class="process-example">
<summary>Example in Python</summary>

Continue from the [file-reading example](#training-engines):

```python
columns = list(PAPER_SENSOR_COLUMNS)
assert round(labelled.sensor_9.corr(labelled.sensor_14), 6) == 0.963157
selection_input = labelled.head(3)
selected = labelled[columns]
selection_output = selected.head(3)
sensor_names = {sensor.column: sensor.symbol for sensor in SENSORS}
selection_columns = pd.DataFrame([
    {"position (from 0)": i, "column": column, "sensor": sensor_names[column]}
    for i, column in enumerate(columns)
])
```

</details>

#### Output

Three rows with fourteen sensor readings each, in the chosen
order. The complete selection has 20,631 rows and fourteen columns:

<!-- sample: selection_output -->
| sensor_2 | sensor_3 | sensor_4 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_11 | sensor_12 | sensor_13 | sensor_15 | sensor_17 | sensor_20 | sensor_21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 641.82 | 1589.7 | 1400.6 | 21.61 | 554.36 | 2388.06 | 9046.19 | 47.47 | 521.66 | 2388.02 | 8.4195 | 392 | 39.06 | 23.419 |
| 642.15 | 1591.82 | 1403.14 | 21.61 | 553.75 | 2388.04 | 9044.07 | 47.49 | 522.28 | 2388.07 | 8.4318 | 392 | 39 | 23.4236 |
| 642.35 | 1587.99 | 1404.2 | 21.61 | 554.26 | 2388.08 | 9052.94 | 47.27 | 522.42 | 2388.03 | 8.4178 | 390 | 38.95 | 23.3442 |

The full table is still needed alongside this selection: engine identifiers
keep histories separate, operating settings describe conditions, and labels
supply the answers. Only the fourteen selected sensor readings become model
inputs. Later stages smooth and rescale those readings before grouping them
into windows.

Continue with [Preprocessing, starting with Cap the labels](04_the_pipeline.md#1--cap-the-labels).
The worked tables and code on this page are checked against the FD001 files by
`python -m pytest tests/test_examples.py -k design_data`.
