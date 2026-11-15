# Data: from engine histories to known answers

## The data: NASA's C-MAPSS engines

The data in this project come from **C-MAPSS**, the Commercial Modular
Aero-Propulsion System Simulation. C-MAPSS simulates a large commercial
turbofan engine of the 90,000 lb thrust class and is written in MATLAB and
Simulink (Saxena et al., 2008, p. 2); its user's guide was published by NASA
as a technical memorandum (Frederick et al., 2007, p. 45). Saxena et al. (2008)
used it to simulate engines that wear out until they fail, and the result became
the challenge data of the Prognostics and Health Management data competition at
PHM'08 (p. 1). They needed a simulation because records of real machines
followed all the way to failure are rare: collecting them is slow and
expensive, and those who hold them tend to keep them from public release
(Saxena et al., 2008, pp. 1–2).

NASA distributes the data as four subsets, FD001 to FD004, and its description
of them names Saxena et al. (2008) as their reference (NASA Prognostics Center
of Excellence [NASA PCoE], 2008). This project uses **FD001**: 100 training
engines and 100 test engines, all running in one operating condition (sea
level) with one fault mode, degradation of the high-pressure compressor (NASA
PCoE, 2008). FD001 is not the challenge data that paper describes, though:
those were simulated under six different flight conditions (Saxena et al.,
2008, pp. 6–7), and FD001 has one. Each subset comes as three text files: a
training file, a test file and a file of answers for the test engines, which
NASA's description calls "a vector of true Remaining Useful Life (RUL) values
for the test data" (NASA PCoE, 2008). This page reads the FD001 files in
`data/raw/` and never creates replacement sensor readings.

## What one cycle is

Every row of a data file is one engine at one moment. NASA's description of
the files says that "each row is a snapshot of data taken during a single
operational cycle", and that the second column is "time, in cycles" (NASA
PCoE, 2008). Saxena et al. (2008), the paper that description names as its
reference, also counts its simulated output in cycles, as "the time series
(cycles) of sensed measurements" (p. 1), and says what one snapshot is: a
simulated engine may fly in different flight conditions "from one flight to
another", and the scenario takes "one measurement snapshot per flight to
characterize the engine health during or right after that flight" (p. 5).

Read together, the two sources mean that **one cycle is one flight**, and an
engine has one row per flight. Neither source puts it in exactly those words;
it is this project's reading of the quotations above, and it assumes the files
were made the way Saxena et al. (2008) describe. That paper also describes its
output as measurements "at cruise snapshots" (p. 6), while FD001's one
condition is sea level (NASA PCoE, 2008), so the FD001 files do not match its
description exactly. A cycle is a count, and the files give it no
duration: the columns NASA lists are the engine number, the cycle, three
operational settings and the sensor measurements, none of them a duration
(NASA PCoE, 2008; Saxena et al., 2008, Table 2), so cycle 3 says only that it
was the engine's third recorded flight.

## Training engines and test engines

Each file holds many engines, and each engine starts with a different,
unknown amount of wear and manufacturing variation, which counts as normal
rather than as a fault (NASA PCoE, 2008). Every engine runs normally at first
and develops a fault at some point (NASA PCoE, 2008). In the **training**
file the fault grows until the engine fails; in the **test** file each history stops some time
before failure, and a separate answer file gives how many cycles each test
engine still had left after its last recorded one (NASA PCoE, 2008). Training
engine 1 and test engine 1 are different engines: each history comes from a
different engine (NASA PCoE, 2008), and the numbering starts again at 1 in
every file.

## Remaining useful life

The quantity the data are built to predict is the **remaining useful life
(RUL)**: how long a component will still work before it reaches the end of
its useful life, counted in units of time such as hours or cycles (Saxena et
al., 2008, p. 1). In these files it is counted in cycles. For a test engine it
is "the number of operational cycles after the last cycle that the engine will
continue to operate" (NASA PCoE, 2008).

This page needs the definition because everything after it turns the files
into examples with a known remaining life. A model learns by being shown
inputs together with the right answer, and the files contain no remaining-life
column for the training engines (NASA PCoE, 2008). That answer has to be
counted from each history, which is step 1 below. For the test engines the
answer file gives it only at the last recorded cycle.

The estimate is worth having because it lets maintenance be planned before a
failure: Saxena et al. (2008) write that remaining-life estimates allow
"planners to account for upcoming maintenance" (p. 2). Reliable estimates,
they add, promise two things: cost savings, "by avoiding unscheduled
maintenance and by increasing equipment usage", and better operational safety
(p. 2).

A wrong estimate costs something in either direction. Wen et al. (2026) write
that estimating too little is "generally safer" than estimating too much,
because it "leads to earlier maintenance rather than unexpected failures", but
that estimating far too little "results in unnecessary maintenance costs"
(p. 9). So an estimate that is too high lets an engine fly on towards a failure
in service, and one that is too low sends it for maintenance it did not yet
need.

## What the engine's sensors measure

{{figure:engine}}

Air enters through a fan. A splitter then sends part of it around the
engine's core along a bypass path, and the rest passes through a low-pressure
compressor (LPC), a high-pressure compressor (HPC), a combustor, and high- and
low-pressure turbines (HPT and LPT) (Frederick et al., 2007, p. 7 and
Figs. 1.1–1.2, p. 9; Saxena et al., 2008, Figs. 1 and 2). Sensors record
temperatures, pressures, shaft speeds and other quantities along this path (Saxena et al., 2008, Table 2). For
example, `T24` measures temperature at the LPC outlet (Saxena et al., 2008,
Table 2). Changes upstream can affect readings downstream: Wen et al. (2026)
build part of their sensor graph by linking sensors that follow one another
along the engine, "such as the temperature transfer chain T24→T30→T50"
(Section 2.3.1, p. 5). The model's sensor graph is the table of connections
those authors chose (Wen et al., 2026, Table 3); a connection is not proof of
a causal effect.

## How the steps below are written

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
engine is being run), and all 21 sensor readings (NASA PCoE, 2008; Saxena et
al., 2008, Table 2). There is no header in the file.

##### Function

Organise the readings into a table. Each position in a line
has a meaning: engine, cycle, operating settings, then sensors. Giving these
positions names lets us follow one engine through time without mixing it
with another. Keep all rows of an engine together and put its cycles in order.
No measurement changes during this operation.

For the first sample line, the first 1 identifies engine 1 and the second 1
identifies cycle 1. The three values after them describe how the engine was
being run. Among the sensor values, 641.82 is the second one. Wen et al.
(2026, Table 2) number the 21 sensors so that sensor 2 is the temperature
named T24; reading the file's sensor columns in that numbering is this
project's reading, checked on the
[C-MAPSS resource page](../resources/cmapss.md#which-flight-conditions-the-settings-are).
The next two lines are
cycles 2 and 3 of that same engine. The output below labels
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
For a training engine we know when its history ends in failure (NASA PCoE,
2008), so we can count how many cycles remain at every earlier row. This is
arithmetic on the recorded history; it does not require a trained model.

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

Subtracting tells us how many cycles still lie ahead. Wen et al. (2026, Eq. 19)
use the same subtraction before capping it. The final cycle is
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

A test engine has not failed at its final recorded row (NASA PCoE, 2008).
Subtracting its current
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
line 2 to test engine 2, and so on. NASA's description does not say which line
is which engine (NASA PCoE, 2008); matching them by position, one line for each
of the 100 test engines, is this project's reading. The file's first three
entries are:

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
each engine, paired with the answer at its end: the task NASA set asks for the
cycles left after each test engine's last cycle (NASA PCoE, 2008), and Wen et
al. (2026, Table 5) count 100 testing samples for FD001's 100 test engines.

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

The project uses one fixed list of 14 sensor columns: the 14 sensors of the
paper's sensor graph, in the order of its rows (Wen et al., 2026, Table 3).
The six constant sensors reported above are absent. The paper names all six
among the sensors it removes (writing PCNfR_dmd as PCNR_dmd there, unlike its
Table 2), by a rule that removes "features with high
correlational redundancy (above 0.95) and constant values" (Wen et al., 2026,
Section 3.2.2, p. 8). `NRc`, the corrected core-speed reading, is absent too:
the paper says seven sensors are removed but names only six, and taking `NRc`
as the seventh is this project's reading of its tables (assumption A-10). In
the full FD001 training file, `Nc` and `NRc` have correlation 0.963157 when
rounded to six decimal places, above the paper's 0.95; 1 would mean they move
perfectly together. They describe the same core shaft's speed in different
forms (Saxena et al., 2008, Table 2), so much of their variation overlaps.
The fixed list keeps `Nc`. `P15` stays in the list because it is one of the
paper's 14, and removing it would also change the chosen sensor graph; the
diagnostic reports its limited variation without silently redesigning the
model.
The graph's rows and columns use the same sensor order, so changing the order
would connect the wrong readings to one another.

{{figure:correlation}}

This chart compares how readings rise and fall together across the training
file. A positive correlation means they tend to move in the same direction;
a negative one means opposite directions. Correlation describes co-variation,
not a causal connection. The chart helps inspect the selection; it does not
change the fixed list when preprocessing runs.
Why most pairs of sensors correlate as strongly as they do, and why that is
not evidence that they are linked inside the engine, is taught on
[C-MAPSS, How far the sensors move together](../resources/cmapss.md#how-far-the-sensors-move-together).

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

## References

- Frederick, D. K., DeCastro, J. A., & Litt, J. S. (2007). *User's guide for
  the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)*
  (NASA/TM-2007-215026). NASA Glenn Research Center.
  https://ntrs.nasa.gov/citations/20070034949. Page numbers on this page are
  those of the PDF file, not the report's printed page numbers.
  [Copy](../references/frederick_2007_cmapss_users_guide.pdf)
- NASA Prognostics Center of Excellence. (2008). readme.txt, the description
  of the C-MAPSS data set distributed inside its zip file. NASA Prognostics
  Data Repository.
  [Copy](../references/nasa_2008_cmapss_readme.txt)
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation
  modeling for aircraft engine run-to-failure simulation. In *2008
  International Conference on Prognostics and Health Management* (pp. 1–9).
  IEEE. https://doi.org/10.1109/PHM.2008.4711414
  [Copy](../references/saxena_2008_damage_propagation_modeling.pdf)
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and
  Technology*, *168*, 110875. https://doi.org/10.1016/j.ast.2025.110875
  [Copy](../references/wen_2026_estgnn_dynamic_shap.pdf)
