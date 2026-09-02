# `loading` — read the three NASA text files, and work out a label for every row

This is the first step of the whole project. Nothing here is machine learning:
it opens three plain text files, turns them into a table, and adds one extra
column saying, for each row, how many flights that engine still has left in it.
That extra column is the number the model will later be asked to predict.

For what C-MAPSS is, what a flight cycle is and what remaining useful life
means, read [the data folder overview](../README.md) first. In one sentence:
the data is a set of simulated jet engines, each recorded once per flight until
it broke, and "remaining useful life" is how many flights are left before it
breaks.

---

## What this file is given, and what it hands back

| | |
|---|---|
| **Given** | A directory holding the three text files of one sub-dataset, for example `data/raw/` with `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` |
| **Hands back** | A `CMAPSSData` — the training table, the test table, and the official test labels, with a `rul` column already worked out on both tables |

Where it sits in the chain:

```
three raw .txt files  ->  THIS FILE  ->  table with a rul column  ->  labeling  ->  smoothing  ->  ...
```

---

## What you need to know first

### The three files, and why there are three

Each sub-dataset (`FD001` to `FD004`) arrives as three files:

| File | What it holds |
|---|---|
| `train_FD001.txt` | Complete histories. Every engine is recorded from new **until it fails**. |
| `test_FD001.txt` | Partial histories. Recording **stops before failure** — the engine is still running fine when the file ends. |
| `RUL_FD001.txt` | One number per test engine: how much life that engine had left at its **last recorded row**. |

The third file exists only because the second one is cut short. A training
engine needs no label file, because you can see it break. A test engine does,
because you cannot.

### A table, and a column of numbers

Two shapes of data appear throughout this page, and they are easy to mix up.

- A **DataFrame** is pandas' name for a table: named columns, numbered rows,
  each column holding one kind of value. `train` is a DataFrame.
- A **Series** is a single column of values with labels attached to each one.
  `test_rul` is a Series: 100 numbers, labelled by engine number.

### The 26 columns of a history file

The two history files have no header row. The columns are, in file order:

```
column 1     unit          which engine this row belongs to
column 2     cycle         which flight of that engine, counting from 1
columns 3-5  op_setting_*  how the aircraft was being flown
columns 6-26 sensor_1 .. sensor_21   the 21 sensor readings
```

The names come from [`sensors`](../sensors/README.md), which holds the list
`RAW_COLUMNS` that this file applies to the nameless columns it reads.

---

## The public names

| Name | What it is | What it gives you back |
|---|---|---|
| `load_subset(subset, data_dir=...)` | The only one you normally call. Reads all three files and attaches the labels. | A `CMAPSSData` |
| `read_cmapss_frame(path)` | Reads one history file. | A DataFrame with the 26 named columns, sorted by engine then cycle |
| `read_rul_file(path)` | Reads one `RUL_FD00X.txt` label file. | A Series of one number per engine, labelled by engine number |
| `attach_rul(frame, rul_at_last_cycle=None)` | Adds the `rul` column. | A **new** DataFrame; the one passed in is not changed |
| `CMAPSSData` | Holds one loaded sub-dataset. | An object with `.subset`, `.train`, `.test`, `.test_rul` and a `.summary()` |
| `EXPECTED_SHAPES` | How many rows and engines each sub-dataset should have, from the published descriptions. | A dictionary, used for a warning |
| `DOWNLOAD_URL` | Where the NASA files come from. | A string, quoted inside the missing-file error |

---

## `read_cmapss_frame` — one text file into one table

### The input

A path to a history file. Here is the very first line of the real
`train_FD001.txt`, printed as raw bytes so nothing is hidden:

```python
with open("data/raw/train_FD001.txt", "rb") as f:
    line = f.readline()
print(repr(line))
print("ends with a space before the newline?", line.endswith(b" \n"))
print("fields when split on whitespace:", len(line.split()))
```

```
b'1 1 -0.0007 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 21.61 554.36 2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 8.4195 0.03 392 2388 100.00 39.06 23.4190  \n'
ends with a space before the newline? True
fields when split on whitespace: 26
```

Reading that line by eye: engine `1`, cycle `1`, then the three settings
`-0.0007 -0.0004 100.0`, then 21 sensor values starting at `518.67`. Note the
**two spaces before the newline** at the end — that trailing whitespace is a
real feature of these files and is dealt with below.

### Step by step

**1. Refuse a path that is not there, and say where to get the file.**

```python
if not path.is_file():
    raise FileNotFoundError(...)
```

This is the error a first-time reader of the repository is most likely to
meet, so the fix travels with the problem. Triggered for real:

```python
read_cmapss_frame("data/raw/train_FD009.txt")
```

```
FileNotFoundError: C-MAPSS file not found: data\raw\train_FD009.txt

The NASA files are not included in this repository because they are
redistributed under NASA's own terms. Download the 'Turbofan Engine
Degradation Simulation Data Set' from the NASA Prognostics Center of
Excellence repository:

    https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

and put the three files for this sub-dataset into data/raw/. See README.md,
section 'Getting the data'.
```

**2. Read the file, splitting on runs of whitespace.**

```python
frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
frame = frame.dropna(axis=1, how="all")
```

`sep=r"\s+"` means "one or more whitespace characters separate one value from
the next". `header=None` says the file has no row of column names, so the
columns come out numbered `0` to `25` rather than named.

`dropna(axis=1, how="all")` throws away any column that is empty in **every**
row — `axis=1` means columns rather than rows, and `how="all"` means only if
the whole column is empty.

**A measured correction.** [The folder overview](../README.md) says the trailing
space "would otherwise be read as an empty 27th column". With the separator
this code actually uses, it is not. Measured on the real file:

```python
for sep in [r"\s+", " "]:
    f = pd.read_csv("data/raw/train_FD001.txt", sep=sep, header=None, engine="python")
    print(sep, "->", f.shape[1], "columns,", int(f.isna().all(axis=0).sum()), "entirely empty")
```

```
sep=r"\s+"  (what the code uses)       -> 26 columns, 0 of them entirely empty
sep=" "   (a single literal space)     -> 28 columns, 2 of them entirely empty
```

Because `\s+` swallows a run of spaces as one separator, the trailing
whitespace produces no extra field and `dropna` removes nothing on these files.
It would matter if the separator were a single literal space, which yields 28
columns of which 2 are empty. So the `dropna` line is a guard that does no work
on the official files — worth keeping, and worth not describing as though it
were load-bearing.

**3. Insist on exactly 26 columns.**

```python
if frame.shape[1] != len(RAW_COLUMNS):
    raise ValueError(...)
```

`frame.shape` is a pair, `(rows, columns)`, so `shape[1]` is the column count.
Triggered for real on a four-column file:

```
ValueError: Expected 26 columns in bad.txt, found 4. The file may be corrupted
or may not be an official C-MAPSS file.
```

**4. Name the columns, fix two types, and sort.**

```python
frame.columns = RAW_COLUMNS
frame[UNIT_COL] = frame[UNIT_COL].astype(int)
frame[CYCLE_COL] = frame[CYCLE_COL].astype(int)
return frame.sort_values([UNIT_COL, CYCLE_COL]).reset_index(drop=True)
```

The engine number and the cycle count are whole numbers that get compared and
grouped on constantly, so they are forced to whole-number type rather than left
as decimals. `sort_values([...])` orders the rows by engine first and by cycle
within each engine. `reset_index(drop=True)` renumbers the rows `0, 1, 2, …`
afterwards and throws the old numbering away.

That sort is a promise the rest of the project relies on: every later step —
smoothing a signal, cutting a window — assumes consecutive rows of one engine
are consecutive flights. Confirmed on the real file:

```
already sorted by (unit, cycle)? True
index is a clean 0..n-1 range? True

 unit  cycle
    1      1
    1      2
    1    191
    1    192
    2      1
    2      2
```

### The output

```python
train = read_cmapss_frame("data/raw/train_FD001.txt")
```

```
type: DataFrame  shape: (20631, 26)
columns: ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',
          'sensor_1', 'sensor_2', ... , 'sensor_21']
dtypes of the first four: {'unit': 'int64', 'cycle': 'int64',
                           'op_setting_1': 'float64', 'op_setting_2': 'float64'}

 unit  cycle  op_setting_1  op_setting_2  op_setting_3  sensor_1  sensor_2  sensor_3
    1      1       -0.0007       -0.0004         100.0    518.67    641.82   1589.70
    1      2        0.0019       -0.0003         100.0    518.67    642.15   1591.82
    1      3       -0.0043        0.0003         100.0    518.67    642.35   1587.99
```

20631 rows, 26 columns. `int64` means a whole number stored in 64 bits;
`float64` a decimal number in 64 bits.

---

## `read_rul_file` — the official test labels

### The input

`RUL_FD001.txt`: one number per line, nothing else. Line 1 belongs to test
engine 1, line 2 to test engine 2, and so on. **That positional rule is how the
dataset is defined** — the file contains no engine numbers of its own.

### Step by step

```python
values = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
values = values.dropna(axis=1, how="all")
if values.shape[1] != 1:
    raise ValueError(f"Expected a single column in {path.name}, found {values.shape[1]}.")

series = values.iloc[:, 0].astype(int)
series.index = pd.Index(range(1, len(series) + 1), name=UNIT_COL)
series.name = "rul_at_last_cycle"
```

`values.iloc[:, 0]` takes column number 0 — every row, first column — turning
the one-column table into a Series.

The two lines after it are what make this usable. `series.index = ...` attaches
the labels `1, 2, 3, …` to the values, so the engine number becomes the way you
look a value up rather than its position in a list. Naming the index `unit` and
the values `rul_at_last_cycle` means that when this Series is later matched
against a table, the join reads correctly.

### The output

```
type: Series  length: 100  dtype: int64
name: rul_at_last_cycle  index name: unit

unit
1    112
2     98
3     69
4     82
5     91
```

Read line 1 as: **test engine 1 had 112 flights left at its last recorded
row.** It says nothing about any other row of that engine.

---

## `attach_rul` — the heart of this file

This is the function worth reading slowly. Everything else here is plumbing.

### The input

A table from `read_cmapss_frame`, and — for test data only — the Series from
`read_rul_file`.

### Step by step, on a training engine

Training engine 1 of FD001:

```
rows: 192   first cycle: 1   last cycle: 192
```

It was recorded until it failed, so the last row **is** the failure. The label
at any cycle is how many rows are still to come:

```
RUL(t) = last_cycle_of_this_engine - t
```

In code:

```python
last_cycle = frame.groupby(UNIT_COL)[CYCLE_COL].transform("max")
```

`groupby(UNIT_COL)` splits the table by engine; `[CYCLE_COL].transform("max")`
computes the largest cycle within each engine and then **writes it back onto
every row of that engine**. So `last_cycle` is a column the same length as the
table, holding 192 on all 192 rows of engine 1, 287 on all rows of engine 2,
and so on. (`transform` is what distinguishes this from a plain `max`, which
would give one number per engine instead of one per row.)

With `rul_at_last_cycle=None`, the offset is zero and the final line is a
straight subtraction:

```python
frame["rul"] = (last_cycle - frame[CYCLE_COL] + offset).astype(int)
```

Real output:

```
 unit  cycle  rul
    1      1  191
    1      2  190
    1      3  189
   ...
    1    190    2
    1    191    1
    1    192    0

last row's label is 0 - zero by definition, because the engine failed there
```

Check the first row by hand: `192 - 1 + 0 = 191`. ✓

### Step by step, on a test engine — and why the training trick fails

Test engine 1:

```
rows: 31   first cycle: 1   last cycle: 31
```

Apply the training rule and you would get `31 - 31 = 0` at the last row: a
claim that the engine died at cycle 31. **It did not.** Recording simply
stopped. Every label built that way would be wrong, and wrong by a different
amount for every engine.

The label file fixes this, and the step that makes it work is worth stating
plainly: *the label file does not give you the labels — it gives you the
**end**, and from the end every label follows.*

```
label file, line 1                      = 112 flights left at cycle 31
so the cycle it would have failed at    = 31 + 112 = 143
```

Once you know it fails at 143, the engine is no different from a training
engine. Every row is the same subtraction as before:

| cycle | 143 − cycle | the `rul` column, measured |
|---|---|---|
| 1 | 142 | 142 |
| 10 | 133 | 133 |
| 20 | 123 | 123 |
| 31 | 112 | 112 ← equals the label file, as it must |

```
check: the label at the last recorded cycle equals the label file -> True (112 == 112)
```

**The code writes the same arithmetic the other way round.** Instead of
subtracting from the failure cycle, it adds two counts:

```python
offset = frame[UNIT_COL].map(rul_at_last_cycle).to_numpy()
frame["rul"] = (last_cycle - frame[CYCLE_COL] + offset).astype(int)
```

```
RUL(t) = (last_recorded_cycle - t) + life_left_after_the_file_stops
```

Read it as *"how many more rows of this engine are still in the file, plus how
much life it had after the file stops"*. Both of those lie ahead of cycle `t`,
which is why they add. At cycle 20: `(31 - 20) + 112 = 11 + 112 = 123` — the
same 123 as the table above.

`frame[UNIT_COL].map(rul_at_last_cycle)` is what turns one number per engine
into one number per row: for each row it looks the engine number up in the
Series and writes that engine's label. This is exactly why `read_rul_file`
bothered to set the index to the engine number.

The two forms are the same arithmetic rearranged, and they agree on every row
of the real file:

```
form A  (failure_cycle - t)           matches the rul column: True
form B  (rows_left + life_after_file) matches the rul column: True
rows checked: 13096
```

### It does not modify what it was given

```python
frame = frame.copy()
```

The first line of the function makes a copy, so the table you passed in is
untouched:

```
'rul' in the original?    False
'rul' in what came back?  True
same object?              False
rul dtype:                int64
```

This matters because the preprocessing chain hands the same table to several
steps, and a function that quietly added a column would change what the next
step sees.

### Why it is done this way

The alternative — computing the failure cycle explicitly and subtracting —
gives identical numbers, as shown above. The `map`-and-add form is used because
it needs no separate per-engine table: `last_cycle` and `offset` are both
already columns the same length as the frame, so the whole label column is one
arithmetic expression, and the training case falls out of it by setting the
offset to zero rather than by writing a second branch.

**Do not confuse** the `rul` column with `test_rul`. The `rul` column has a
value on **every** row. `test_rul` has one value per **engine**, describing only
that engine's last recorded row. The first is derived from the second.

---

## `load_subset` — all three files, in order

### Step by step

```python
train = read_cmapss_frame(data_dir / f"train_{subset}.txt")
test  = read_cmapss_frame(data_dir / f"test_{subset}.txt")
test_rul = read_rul_file(data_dir / f"RUL_{subset}.txt")

train = attach_rul(train)                              # no label file needed
test  = attach_rul(test, rul_at_last_cycle=test_rul)   # label file needed
```

Then a check that the files are the ones they claim to be. `EXPECTED_SHAPES`
holds the published row and engine counts:

```python
EXPECTED_SHAPES = {
    "FD001": {"train_rows": 20631, "train_units": 100, "test_units": 100},
    "FD002": {"train_rows": 53759, "train_units": 260, "test_units": 259},
    ...
}
```

The measured counts are compared against those, and any difference is
**warned** about rather than raised. Warning rather than stopping is a
deliberate choice: a legitimate variant of the data should not halt the work,
but a silent mismatch would be far worse — a result computed from truncated
files would look perfectly normal.

### The output

```
INFO    | thesis_gnn_rul_xai.data.loading | Loading FD001 from data\raw
INFO    | thesis_gnn_rul_xai.data.loading | FD001 matches the published shape of the official release.
INFO    | thesis_gnn_rul_xai.data.loading | Loaded FD001: 20631 training rows over 100 engines, 13096 test rows over 100 engines.

repr: CMAPSSData(subset='FD001', train_rows=20631, train_units=100, test_rows=13096, test_units=100)
```

And with 100 rows cut off the end of the training file, the check fires:

```
WARNING | thesis_gnn_rul_xai.data.loading | Loaded FD001 does not match the
published shape of the official release. Differences (expected, actual):
{'train_rows': (20631, 20531)}. The files may be a different version,
truncated, or modified. Investigate this before reporting any result computed
from them.
```

---

## `CMAPSSData` — what comes out

Four things: `subset` (which sub-dataset), `train` and `test` (the two tables,
each now carrying a `rul` column), and `test_rul` (the official label at each
test engine's last cycle, kept because scoring needs it later).

`summary()` returns a small table to check by eye:

```python
load_subset("FD001", data_dir="data/raw").summary()
```

```
        rows  engines  shortest_history  longest_history  median_history
split
train  20631      100               128              362           199.0
test   13096      100                31              303           133.5
```

Two things in that table matter later:

- The **shortest test history is 31 cycles.** The window used by
  [`windowing`](../windowing/README.md) is 30, so every FD001 test engine is
  long enough to yield one window — narrowly. That is finding **F-03**.
- Training histories are longer than test histories throughout, which is what
  you would expect: training engines are recorded to failure and test engines
  are cut off before it.

---

## The judgement calls recorded in this file

**None.** This file makes no choice the paper left open. Reading the files and
building the labels is fixed by how the dataset is defined, and everything here
follows from the published description of the format.

The choices start immediately afterwards. The `rul` column this file produces
counts all the way up — 191 for training engine 1 at cycle 1 — and the very
next step, [`labeling`](../labeling/README.md), stops it at a ceiling. That
ceiling is **A-01**, and applying it to the test labels as well is **A-02**.
Both are in [the assumptions log](../../../../docs/ASSUMPTIONS.md).

---

## What can go wrong

Every message below was produced by actually triggering it.

**The history file is missing** — the long message quoted earlier, carrying the
NASA download link.

**The label file is missing:**

```
FileNotFoundError: RUL label file not found: data\raw\RUL_FD009.txt
See README.md, section 'Getting the data'.
```

**The history file has the wrong number of columns:**

```
ValueError: Expected 26 columns in bad.txt, found 4. The file may be corrupted
or may not be an official C-MAPSS file.
```

**The label file has more than one column:**

```
ValueError: Expected a single column in badrul.txt, found 2.
```

**A test engine has no label.** With engines 99 and 100 removed from the label
file:

```
ValueError: No Remaining Useful Life label for engine(s) [np.int64(99), np.int64(100)].
The label file and the test file do not describe the same engines.
```

This is the check that catches the most damaging possible mistake in this
file — a label file and a test file that do not line up. Because engine
numbers are assigned **by line position**, a label file with one line missing
would otherwise shift every label by one engine and produce a table that looks
entirely reasonable and is entirely wrong.

**The files do not match the published shape** — a warning, not an error, shown
above.

---

The key idea is: a training engine's label is a subtraction you can do from the
file alone, because you can see it fail; a test engine's label needs the label
file to tell you where the end is, and once you know that, it is the same
subtraction.
