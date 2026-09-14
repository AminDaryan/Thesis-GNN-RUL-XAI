# `windowing` — cut each engine's history into fixed-length training examples

By the time data reaches this file it is a table: one row per engine per
flight, sensors rescaled, label attached. A neural network cannot be given a
table of ragged length — engine 1 flew 192 times, engine 2 flew 287 — so this
file cuts each history into equal-sized pieces and stacks them into one
rectangular pile of examples.

It also does the split into training and held-back engines, and that belongs
here rather than anywhere else for a reason this page measures.

For what a flight cycle and remaining useful life are, see
[the data folder overview](../README.md).

---

## What this file is given, and what it hands back

| | |
|---|---|
| **Given** | A table with sensor columns and a `rul` column, a window length, and a stride |
| **Hands back** | A `WindowedDataset` — one rectangular array of inputs, one array of answers, and enough bookkeeping to say where each example came from |

Where it sits in the chain:

```
normalization  ->  THIS FILE  ->  training
```

---

## What you need to know first

### What a window is

A **window** is a run of consecutive flights, treated as one training example.
The **stride** is how far the window jumps between one example and the next.

```
engine 1's flights   [1 2 3 4 5 6 7]        window size 3, stride 2

                     [1 2 3]                       ends at flight 3
                         [3 4 5]                   ends at flight 5
                             [5 6 7]               ends at flight 7
```

Three examples, starting at rows 0, 2 and 4.

**The label of a window is the remaining life at its last flight.** Not the
first, not the average. The window says "here is what the engine looked like
over these flights", and the answer is "and at the end of that stretch, this
much life remained". That matches the paper: its Equation 7 writes the window
as `[x(t-w+1), … , x(t)]` and its Section 2.1 attaches the target to time `t`.

### An array, and what its shape means

A **NumPy array** is a fixed-size grid of numbers of one type. Its **shape** is
the tuple of its side lengths, so shape `(3, 14, 30)` means 3 items, each a
14-by-30 grid. The arrays here are three-dimensional and each dimension means
something specific, stated at every step below.

---

## The public names

| Name | What it is | What it gives you back |
|---|---|---|
| `make_windows(frame, sensor_columns, window_size=30, stride=1, target_column="rul")` | Every window of every engine. For training data. | A `WindowedDataset` |
| `make_last_windows(frame, sensor_columns, window_size=30, target_column="rul")` | Only the final window of each engine. For test data. | A `WindowedDataset` |
| `split_units_by_id(units, validation_fraction=..., seed=42)` | Splits engine numbers into train and held-back. | Two arrays of engine numbers that never share an engine |
| `WindowedDataset` | The result of the two window functions. | An object with five fields, `len()`, a readable `repr` and `subset_by_units` |

Two private helpers do the real work and are documented here because you cannot
understand the public ones without them: `_extract_one_unit` (one engine) and
`_build` (all engines).

---

## `_extract_one_unit` — cutting one engine

### The input

One engine's rows, already sorted by flight. Using training engine 1's first 7
flights, three sensors, window 3, stride 2:

```
 unit  cycle  sensor_2  sensor_3  sensor_4  rul
    1      1    641.82   1589.70   1400.60  191
    1      2    642.15   1591.82   1403.14  190
    1      3    642.35   1587.99   1404.20  189
    1      4    642.35   1582.79   1401.87  188
    1      5    642.37   1582.85   1406.22  187
    1      6    642.10   1584.47   1398.37  186
    1      7    642.48   1592.32   1397.77  185
```

### Step by step

**1. Pull the three columns out as arrays.**

```python
values  = history[list(sensor_columns)].to_numpy(dtype=np.float32)
targets = history[target_column].to_numpy(dtype=np.float32)
cycles  = history[CYCLE_COL].to_numpy(dtype=np.int64)
n_rows  = values.shape[0]
```

```
  values: shape (7, 3) = (rows, sensors)
  n_rows = 7
```

`float32` is a decimal number stored in 32 bits — half the size of the usual 64
and what the network expects.

**2. Work out where each window starts.**

```python
starts = ([n_rows - window_size] if last_only
          else list(range(0, n_rows - window_size + 1, stride)))
```

With `n_rows = 7`, `window_size = 3`, `stride = 2`:

```
  starts = range(0, 7 - 3 + 1, 2) = range(0, 5, 2)  ->  [0, 2, 4]
```

**Why `n_rows - window_size + 1`?** A window starting at row `s` occupies rows
`s` to `s + 2`. The last row available is 6, so the last legal start is 4. The
expression gives `7 - 3 + 1 = 5`, and `range` stops *before* 5, so the last
value is 4. ✓ A start of 5 would need row 7, which does not exist.

**3. The three iterations, written out.**

```
    start 0: rows 0..2 -> cycles [1, 2, 3], label from row 2 = 189
    start 2: rows 2..4 -> cycles [3, 4, 5], label from row 4 = 187
    start 4: rows 4..6 -> cycles [5, 6, 7], label from row 6 = 185
```

The label always comes from `s + window_size - 1`, the **last** row of the
window.

**4. Stack the windows.**

```python
windows = np.stack([values[s : s + window_size] for s in starts])
```

```
  after np.stack: shape (3, 3, 3) = (windows, timesteps, sensors)
```

That shape is confusing here only because all three numbers happen to be 3 —
three windows, three flights each, three sensors each. With the real settings
it would be `(163, 30, 14)`.

**5. Put sensors in front of time steps.**

```python
windows = np.transpose(windows, (0, 2, 1))
```

```
  after transpose: shape (3, 3, 3) = (windows, sensors, timesteps)
```

`np.transpose(a, (0, 2, 1))` reorders the dimensions: keep dimension 0 where it
is, and swap dimensions 1 and 2. The reason is that the next stage treats each
sensor as one node of a graph, so each sensor needs to own one contiguous block
of numbers.

**6. Hand every cycle of the window on.**

Nothing is cut away here. Until 2026-09-09 this step kept only the last reading
of each sensor, `windows[:, :, -1:]`; it no longer does. How many of a window's
numbers a sensor node takes into the graph layers is a question about the
model, so the model's temporal stage in `models/temporal/` answers it. By
default that stage keeps the last reading (assumption A-07), which is the same
number this step used to keep — but now the reduction is named, where the model
is built, instead of hidden in the data.

Confirmed against the function itself, on the same seven real rows:

```
  from the function: (3, 3, 3)  targets [189. 187. 185.]  cycles [3 5 7]
  its first window:
  [[ 641.82  642.15  642.35]
   [1589.7  1591.82 1587.99]
   [1400.6  1403.14 1404.2 ]]
```

Row 0 of that first window is `sensor_2` at cycles 1, 2 and 3 — 641.82, 642.15
and 642.35, the first three values of its column in the input table — and rows 1
and 2 are the other two sensors over the same three cycles.

### The output

Three NumPy arrays, all aligned by position — entry `i` of each describes
window `i`:

| Returned | Type and shape | Value on this input |
|---|---|---|
| `windows` | `float32`, `(3, 3, 3)` = (windows, sensors, cycles) | every reading of each sensor in each window |
| `window_targets` | `float32`, `(3,)` | `[189. 187. 185.]` |
| `window_cycles` | `int64`, `(3,)` | `[3 5 7]` |

`np.ascontiguousarray` is applied to the first before returning it: the
transpose left the values scattered in memory rather than laid out in reading
order, and copying them into order once here is cheaper than letting every later
operation work around it.

An engine too short for even one window returns three **empty** arrays of the
same types instead, which is the signal `_build` looks for.

### Why the whole window is handed on, when the default model keeps one reading

This is the obvious objection, and it has a real answer. The reduction from 30
readings to one is a property of the model, and the paper's own model does it
with a recurrent network that reads all 30 (`LSTMStage` builds it). Doing the
reduction here would decide, inside the data, what every model is allowed to
see. And when it was done here, deleting one line of this function on 2026-08-28
silently widened the network's first weight matrix from `(1, 32)` to
`(30, 32)`, and two configuration comparisons reversed before anyone noticed.

The window also still decides **which moments are eligible**: the first example
of an engine ends at flight `window_size`, not at flight 1, because a window has
to fit behind it.

---

## `_build` — every engine, stacked into one dataset

`_extract_one_unit` handles one engine. `_build` runs it over **every** engine
in the table and glues the results into a single dataset.

### The input

A table holding more than one engine. Here are real FD001 rows — engines 1 and
2, their first 7 flights, three sensors — with `window_size=3` and `stride=2`:

```
 unit  cycle  sensor_2  sensor_3  sensor_4  rul
    1      1    641.82   1589.70   1400.60  191
    1      2    642.15   1591.82   1403.14  190
    1      3    642.35   1587.99   1404.20  189
    1      4    642.35   1582.79   1401.87  188
    1      5    642.37   1582.85   1406.22  187
    1      6    642.10   1584.47   1398.37  186
    1      7    642.48   1592.32   1397.77  185
    2      1    641.89   1583.84   1391.28  286
    2      2    641.82   1587.05   1393.13  285
    2      3    641.55   1588.32   1398.96  284
    2      4    641.68   1584.15   1396.08  283
    2      5    641.73   1579.03   1402.52  282
    2      6    641.30   1577.50   1396.76  281
    2      7    642.03   1587.49   1400.65  280
```

14 rows, 2 engines. Follow those 14 rows all the way through.

### Step by step

**1. Three guards, before any work.**

```python
missing = [c for c in sensor_columns if c not in frame.columns]
if missing: raise KeyError(...)
if target_column not in frame.columns: raise KeyError(...)
if window_size < 1: raise ValueError(...)
```

These fire before the loop, so a typo costs no time:

```
KeyError: Sensor columns not present in the frame: ['nope'].
KeyError: Target column 'nope' not present in the frame.
ValueError: window_size must be at least 1, got 0.
```

**2. Four empty lists.**

```python
feature_blocks, target_blocks, unit_blocks, cycle_blocks = [], [], [], []
```

Each engine adds one item to each. They stay **aligned by position**: item 0 of
all four describes engine 1.

| list | what it holds |
|---|---|
| `feature_blocks` | the sensor readings — the model's *input* |
| `target_blocks` | the remaining life at each window's end — the *answer* |
| `unit_blocks` | which engine each window came from |
| `cycle_blocks` | which flight each window ended at |

The last two are never fed to the model. They are bookkeeping, needed to group
predictions by engine and to check no engine landed in two splits.

**3. Sort, then group by engine.**

```python
for unit, history in frame.sort_values([UNIT_COL, CYCLE_COL]).groupby(UNIT_COL, sort=True):
```

The sort makes the result independent of the order rows arrived in. The
grouping is what stops a window from **straddling two engines** — without it, a
window starting near the end of engine 1 would run off into engine 2 and produce
one example containing two physical engines with a label belonging to neither.

**4. Record which engine each window came from.**

```python
unit_blocks.append(np.full(windows.shape[0], int(unit), dtype=np.int64))
```

`np.full(3, 1)` builds `[1, 1, 1]` — the engine number repeated once per window,
so it lines up one-to-one.

**5. Skip an engine too short to yield anything (A-23).**

```python
if windows.shape[0] == 0:
    continue
```

```
WARNING | Engine 3 has 2 row(s), fewer than the window size of 3, so it is left out.
Its readings are not padded: that would invent data. Any score computed from what
remains covers fewer engines.

  engines that survived: [1 2]   examples: 6
```

**6. Raise if nothing survived.**

```
ValueError: No engine has at least 3 rows, so every one was left out under A-23
and there is nothing to return. The window is longer than the longest history in
this frame.
```

Returning an empty dataset would let training run, produce nothing, and send you
hunting much later.

**7. Glue the blocks together.**

```python
features=np.concatenate(feature_blocks).astype(np.float32)
```

`np.concatenate` joins along the first dimension: `(3,3,3)` and `(3,3,3)` become
`(6,3,3)`, and `[1 1 1]` and `[2 2 2]` become `[1 1 1 2 2 2]`. All four lists are
joined the same way, so the alignment survives.

Collecting into Python lists and joining **once** is deliberate: a NumPy array
has a fixed size, so "appending" to one means allocating a bigger array and
copying everything across. Doing that per engine would copy the dataset 100
times.

### The output

First, what the loop collected — one block per engine, before anything is
joined:

```
  engine 1: windows(3, 3, 3)  targets [189. 187. 185.]  end cycles [3 5 7]  unit block [1 1 1]
  engine 2: windows(3, 3, 3)  targets [284. 282. 280.]  end cycles [3 5 7]  unit block [2 2 2]
```

Then the four `np.concatenate` calls turn those two blocks into one dataset:

```
WindowedDataset(examples=6, shape=(6, 3, 3), engines=2, target_range=(185, 284))

  features.shape : (6, 3, 3)  float32
  targets        : [189. 187. 185. 284. 282. 280.]
  units          : [1 1 1 2 2 2]
  end_cycles     : [3 5 7 3 5 7]
  sensor_columns : ('sensor_2', 'sensor_3', 'sensor_4')
```

Read those four arrays **downwards**, one position at a time — entry `i` of each
describes example `i`:

```
    example 0: engine 1, ends at cycle 3, features=[ 642.35 1587.99 1404.20], target=189.0
    example 1: engine 1, ends at cycle 5, features=[ 642.37 1582.85 1406.22], target=187.0
    example 2: engine 1, ends at cycle 7, features=[ 642.48 1592.32 1397.77], target=185.0
    example 3: engine 2, ends at cycle 3, features=[ 641.55 1588.32 1398.96], target=284.0
    example 4: engine 2, ends at cycle 5, features=[ 641.73 1579.03 1402.52], target=282.0
    example 5: engine 2, ends at cycle 7, features=[ 642.03 1587.49 1400.65], target=280.0
```

**Check example 0 by hand against the input table above.** Its window is flights
1, 2 and 3 of engine 1. The last of those is flight 3. The three sensor values
on that row are 642.35, 1587.99 and 1404.20 — exactly the features. The `rul`
on that row is 189 — exactly the target. Every one of the six rows is that same
check.

14 input rows became 6 examples: each engine had 7 flights, and a window of 3
with a stride of 2 fits starting at rows 0, 2 and 4 — 3 examples per engine.

### Why it is done this way

The four parallel lists could instead have been one list of small objects, one
per example. They are not, because what the network consumes is a single
rectangular block of numbers: building four flat arrays directly means the
result needs no rearranging afterwards, and the bookkeeping stays attached to
the data by position rather than by a lookup that could go wrong.

---

## `make_windows` — the training path

This is the public door for training data: `_build` with `last_only=False`, so
every engine yields every window it can.

### The input

The full FD001 training table — 20 631 rows over 100 engines — the paper's 14
sensor columns, and the real settings `window_size=30`, `stride=1`.

### Step by step

There are no steps of its own. It forwards its arguments to `_build` with
`last_only=False`, and everything above happens.

### The output, with the real settings

```python
make_windows(train, PAPER_SENSOR_COLUMNS, window_size=30, stride=1)
```

```
INFO | Built 17731 window(s) of shape (14, 30) from 100 engine(s).

WindowedDataset(examples=17731, shape=(17731, 14, 30), engines=100, target_range=(0, 332))
```

Shape `(17731, 14, 30)` reads as **17731 examples, 14 sensors each, thirty numbers
per sensor**.

**Check the count two ways.** Every engine loses its first 29 flights, because
no window can *end* there — there are not 30 flights behind them yet:

```
  rows 20631, engines 100
  20631 - 100*29 = 17731        matches the 17731 reported
  engine 1 has 192 rows -> 192 - 30 + 1 = 163 windows
```

---

## `make_last_windows` — the test path

### The input

The FD001 **test** table — 13 096 rows over 100 engines — the same 14 sensor
columns, and `window_size=30`. Note there is no `stride` argument: with one
window per engine there is nothing to step between.

### Step by step

Identical machinery with `last_only=True`, which changes exactly one line
inside `_extract_one_unit`:

```python
starts = [n_rows - window_size] if last_only else list(range(...))
```

So `starts` holds a single value. For a test engine with 31 rows and a window
of 30 that is `31 - 30 = 1`: rows 1 to 30, ending at the engine's final flight.

### The output

```python
make_last_windows(test, PAPER_SENSOR_COLUMNS, window_size=30)
```

```
INFO | Built 100 window(s) of shape (14, 30) from 100 engine(s).

WindowedDataset(examples=100, shape=(100, 14, 30), engines=100, target_range=(7, 145))

  one window per engine?                      True
  end_cycles equal each engine's last cycle?  True
```

**Why exactly one window per test engine.** The official label file describes
**one moment** per engine: how much life remained at that engine's last recorded
flight. There is no ground truth for any earlier moment, so scoring anything
else would be scoring against a number nobody supplied. On FD001 that is 100
predictions, one per engine.

---

## `split_units_by_id` — and the leakage it prevents

### Why the split must be by engine, not by window

This is the part that decides whether the held-back numbers mean anything.

With a stride of 1, two neighbouring windows of the same engine are almost the
same window:

```
  two neighbouring windows of engine 1 share 29 of their 30 cycles
  their labels: 162 and 161 - one cycle apart
```

Split *windows* at random and near-identical pairs land on both sides of the
split. The held-back set would then contain windows the model has effectively
already seen, and its error would be an optimistic measurement of nothing.
Splitting by **engine** makes that impossible: a held-back engine contributes
none of its windows to training.

### The input

A collection of engine numbers — here the 100 distinct engines of the FD001
training table — plus how large a fraction to hold back and a seed. Note it
takes **engine numbers**, not windows and not a table. It never sees a window
at all, which is what makes splitting by window impossible by construction.

### Step by step

```python
unique = np.unique(np.asarray(list(units), dtype=np.int64))
n_validation = round(len(unique) * validation_fraction)
if n_validation == 0:
    return unique, np.array([], dtype=np.int64)
shuffled = make_rng(seed).permutation(unique)
validation = np.sort(shuffled[:n_validation])
training   = np.sort(shuffled[n_validation:])
```

`make_rng(seed)` produces a random number generator started from a fixed point,
so the same seed always gives the same split. The two halves are taken from
opposite ends of one shuffled list, which is why they cannot overlap. Both are
re-sorted afterwards, purely so the output is readable.

### The output

```python
split_units_by_id(sorted(train.unit.unique()), validation_fraction=0.2, seed=42)
```

```
INFO | Split 100 engines into 80 for training and 20 for validation (seed=42).

  training engines: 80   validation engines: 20
  any engine in both? False
  first ten validation engines: [ 3  5 19 22 25 26 28 34 43 51]
  a different seed gives a different split: True
  fraction 0.0 -> validation is empty: True   training keeps all: 100
```

A fraction of 0 holds nothing back and returns an empty second array.
That was the project's default until 2026-09-02; it is now 0.2 (A-06).

---

## `WindowedDataset` — what comes out

### What it is given, and what it hands back

It is not a function but a container: `_build` fills it with five things, and
it offers three ways to use them — `__len__`, so `len(dataset)` gives the number
of examples; `__repr__`, so printing it gives the one-line description used
throughout this page; and `subset_by_units`, for keeping only some engines.

### The five fields

| Field | What it holds |
|---|---|
| `features` | The readings, shaped `(examples, sensors, numbers per sensor)` |
| `targets` | Remaining life at each window's last flight, in flights |
| `units` | Which engine each example came from |
| `end_cycles` | Which flight each window ended at |
| `sensor_columns` | The sensor names, in the order of the second dimension |

`sensor_columns` travelling with the array is a real safeguard. The sensor graph
is indexed by **position**, so losing the order would silently wire the model to
the wrong sensors — see [`sensors`](../sensors/README.md), where that failure is
demonstrated.

`subset_by_units` keeps only the engines listed:

```python
ds.subset_by_units(validation_engines)
```

```
WindowedDataset(examples=3272, shape=(3272, 14, 30), engines=20, target_range=(0, 239))

  only validation engines present?                     True
  windows in train part + validation part == whole?    True
```

An empty dataset is legitimate — it is what a validation fraction of zero
produces — so it gets its own description rather than failing on the minimum of
an empty array:

```
WindowedDataset(empty, sensors=14)      len: 0
```

---

## The judgement calls recorded in this file

### A-05 — windows are taken one flight apart

The paper gives a window size of 30 in its Table 6 and in Section 3.4, but
**never states a stride**. We use 1 (assumption A-05 in
[the assumptions log](../../../../docs/ASSUMPTIONS.md)).

### A-07 — the model, not this file, reduces a window to one number

Table 4 of the paper gives the input to the first graph layer as
`[batch, 14, 1]` — one number per sensor. In the paper that number is the output
of a per-sensor recurrent stage. This file hands on the whole window; the
model's default temporal stage then keeps the most recent reading in place of
the paper's stage, which is not the default (assumptions A-07 and A-11 in
[the assumptions log](../../../../docs/ASSUMPTIONS.md)).

### A-23 — an engine shorter than the window is left out, not padded

**The gap.** Some test engines are shorter than 30 flights, so no window can be
cut from them. The paper never says what happens to them.

**What we chose.** Leave the engine out, with a warning naming it.

**Why.** Padding invents readings — filling the missing rows by repeating the
engine's first reading puts sensor values into the record that the engine never
produced. That is what working rule 7 forbids. It is also the only published
practice found: Dida, Cheriet and Belhadj exclude engines with fewer than 30
samples on this same dataset.

**What it costs, and the cost is real.** The NASA score is a **sum** over
engines. Drop an engine and the score falls because there is one term fewer, not
because the model improved. Counted from the real files, no FD001 or FD003 test
engine is affected; 6 FD002 and 11 FD004 engines are. So this cannot touch any
number this project currently reports, and becomes live the moment FD002 or
FD004 is run. The full entry is in
[the assumptions log](../../../../docs/ASSUMPTIONS.md) (assumption A-23).

### A-06 — a validation split exists at all

The paper describes no validation split and trains for a fixed number of passes;
holding engines out is **our addition**. The consequence is settled and not up
for re-litigation: **validation error and test error are not comparable in this
project**, and neither should ever be presented as the other (assumption A-06
in [the assumptions log](../../../../docs/ASSUMPTIONS.md)).

---

## What can go wrong

```
KeyError:   Sensor columns not present in the frame: ['nope'].
KeyError:   Target column 'nope' not present in the frame.
ValueError: window_size must be at least 1, got 0.
ValueError: No engine has at least 3 rows, so every one was left out under A-23
            and there is nothing to return. The window is longer than the
            longest history in this frame.
```

From `split_units_by_id`:

```
ValueError: validation_fraction must be in [0, 1), got 1.0.
ValueError: validation_fraction must be in [0, 1), got -0.1.
ValueError: validation_fraction would leave no engines for training (2 of 2 selected).
```

**The silent one**, as always, is the dangerous one: splitting by window instead
of by engine raises nothing, trains fine, and reports a held-back error that is
meaningless. Nothing in the code can detect it — it is prevented by
`split_units_by_id` taking **engine numbers** as its input and never seeing a
window at all.

---

The key idea is: this file turns a ragged table of histories into one
rectangular pile of fixed-size examples, cutting one engine at a time so a
window can never cross between engines — and keeping four arrays aligned by
position so every example still knows its answer, its engine and its moment in
time.
