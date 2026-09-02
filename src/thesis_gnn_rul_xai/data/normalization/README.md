# `normalization` — put the sensors on a comparable scale, without letting the test engines in

Two classes live here, and they answer two different questions.

`OperatingConditionEncoder` asks **how was the aircraft being flown?** — because
a sensor reading means something different at cruise altitude than on the
ground. `SensorNormalizer` asks **how do we make 14 sensors with wildly
different units comparable?** — because a temperature near 1400 and a ratio
near 0.03 cannot be fed to the same network as they stand.

Underneath both sits one rule that matters more than either: **everything is
measured on the training engines only.** That rule is what makes the reported
error mean anything, and most of this page is about it.

For what a flight cycle is, see [the data folder overview](../README.md).

---

## What this file is given, and what it hands back

| | |
|---|---|
| **Given** | A table of sensor readings, and which columns to rescale |
| **Hands back** | A **new** table with those columns rescaled, plus a record of exactly what was measured |

Where it sits in the chain:

```
smoothing  ->  THIS FILE  ->  windowing
```

---

## What you need to know first

### Why rescaling is needed at all

The 14 sensors are on completely different scales. Measured on the real FD001
training data:

```
 sensor_4 (T50, a temperature)   mean 1408.93   spread 9.000
 sensor_6 (P15, a pressure)      mean   21.61   spread 0.001389
```

A neural network multiplies each input by a weight and adds the results up. If
one input arrives as ~1400 and another as ~21.6, the first dominates the sum
purely because of its units — before any learning happens. Rescaling removes
the accident of units so the network can decide what matters.

### The two ways to rescale, in plain terms

**Subtract the mean, then divide by the spread.** The **mean** is the average
value. The **spread** — its formal name is the standard deviation — is the
typical distance of a reading from that average. After this, the column has an
average of 0 and a typical distance from it of 1. Values above average come out
positive, below average negative. This is what statisticians call a
**z-score**: how many spreads away from the average a value sits.

```
   scaled = (value - mean) / spread
```

**Rescale to a fixed range.** Find the smallest and largest value, and stretch
everything so the smallest becomes 0 and the largest 1.

```
   scaled = (value - minimum) / (maximum - minimum)
```

**Do not confuse these two.** The first centres on the average and measures in
units of typical variation; it has no upper or lower limit, and an unusual
reading comes out as a large number. The second forces everything into a fixed
range; it has hard limits, and a single extreme reading squashes everything
else toward one end.

### What an operating condition is

Columns 3 to 5 of each row describe how the aircraft was being flown —
altitude, Mach number and throttle angle. Those settings change what a normal
sensor reading looks like, entirely apart from wear.

On **FD001** they barely vary:

```
      op_setting_1  op_setting_2  op_setting_3
min      -0.008700     -0.000600         100.0
max       0.008700      0.000600         100.0
mean     -0.000009      0.000002         100.0
```

Round those to whole numbers and there is exactly **one** distinct combination:
`[-0., -0., 100.]`. Every FD001 row was flown the same way.

**FD002 is different**, and it is what the machinery here exists for:

```
  n_detected: 6
  rows per condition:
    0    13458
    1     8122
    2     8002
    3     8044
    4     8096
    5     8037
```

Six genuinely different flight profiles. A `T50` of 1400 means something
different under each.

### k-means, before it is used

**k-means** is a way of sorting rows into `k` groups so that rows in a group
have similar values. It works by repeatedly (1) assigning each row to whichever
group centre is nearest, and (2) moving each centre to the average of the rows
assigned to it, until nothing moves. A **centre** is just the average of the
rows in its group. The `k` is fixed in advance — the algorithm never decides how
many groups there are.

Run on the real FD002 settings, the six centres it lands on:

```
[[ 42.     0.84 100.  ]
 [ 20.     0.7  100.  ]
 [ 25.     0.62  60.  ]
 [  0.     0.   100.  ]
 [ 10.     0.25 100.  ]
 [ 35.     0.84 100.  ]]
```

Each row is one flight profile: an altitude, a Mach number and a throttle
setting. These are recognisably distinct operating points, not noise.

---

## The public names

| Name | What it is | What it gives you back |
|---|---|---|
| `OperatingConditionEncoder` | Works out which flight profile each row belongs to. | Itself from `fit`; a table with a new `operating_condition` column from `transform` |
| `SensorNormalizer` | Rescales the sensor columns. | Itself from `fit`; a rescaled table from `transform` |
| `CONDITION_COL` | The name of the column the encoder adds. | `"operating_condition"` |

Both classes follow the same three-method shape, and it is the shape that
enforces the rule at the top of this page:

| Method | What it does | What you pass it |
|---|---|---|
| `fit(frame)` | **Measures.** Learns the numbers. | **Training rows only. Never test rows.** |
| `transform(frame)` | **Applies** what was measured. | Any table — training, validation or test |
| `fit_transform(frame)` | Both, in one call. | Training rows only |

---

## `OperatingConditionEncoder`

### Step by step — `fit`

```python
settings = frame[list(self.setting_columns)].to_numpy(dtype=float)

if self.n_conditions is None:
    rounded = np.round(settings, self.round_decimals)
    detected = len(np.unique(rounded, axis=0))
    detected = int(np.clip(detected, 1, 6))
```

With `n_conditions` left as `None`, the count is detected from the data. The
settings are rounded to whole numbers (`round_decimals=0`) and the distinct
combinations counted. `np.unique(..., axis=0)` means "distinct **rows**", not
distinct numbers.

The rounding is deliberate: FD001's settings wobble in the fourth decimal
place, so without it every row would look like its own condition. The
`np.clip(detected, 1, 6)` then holds the answer between 1 and 6 — 6 being the
number the dataset documentation describes — so measurement noise cannot invent
a seventh.

```python
if detected <= 1:
    self._model = None
    logger.info("A single operating condition: clustering is not needed.")
    return self
```

**With one condition, k-means is skipped entirely.** There is nothing to sort.
This is the FD001 path, and it means no clustering runs at all on the
sub-dataset this project reports.

```python
self._model = KMeans(n_clusters=detected, n_init=10, random_state=self.seed)
self._model.fit(settings)
```

Otherwise k-means runs. `n_init=10` restarts it 10 times from different starting
points and keeps the best, because the result depends on where the centres
start. `random_state=self.seed` fixes those starting points so two runs give
identical groups — without it, condition 3 in one run might be condition 5 in
the next, and nothing downstream would notice.

### Step by step — `transform`

```python
if self._model is None:
    frame[CONDITION_COL] = 0
else:
    settings = frame[list(self.setting_columns)].to_numpy(dtype=float)
    frame[CONDITION_COL] = self._model.predict(settings).astype(int)
```

One condition means every row is labelled 0. Otherwise each row is assigned to
whichever centre is nearest — using the centres **learned from training rows**,
which is the leakage rule applied here.

### The output

```python
enc = OperatingConditionEncoder()
marked = enc.fit_transform(train)
```

```
INFO | Detected 1 distinct operating condition(s) from the settings.
INFO | A single operating condition: clustering is not needed.

  n_detected: 1
  column added: operating_condition -> distinct values: [0]
  was k-means used? False
```

---

## `SensorNormalizer`

### Step by step — `fit` measures and stores

```python
if self.method == "minmax_per_condition":
    groups = frame.groupby(CONDITION_COL, sort=True)
else:
    groups = [(0, frame)]
```

The per-condition method measures separately within each flight profile;
the other two treat the whole table as one group, which is what
`[(0, frame)]` fakes — a single group numbered 0.

Then, for every group and every column, five numbers are recorded:

```python
records.append({
    "condition": int(condition), "column": column,
    "min": float(values.min()),   "max": float(values.max()),
    "mean": float(values.mean()), "std": float(values.std(ddof=0)),
    "n_rows": int(values.size),
})
```

That table **is** the fitted state. It can be printed, saved and checked, which
matters for a thesis: what the scaling did is inspectable rather than hidden.

```python
n = SensorNormalizer(method="zscore", columns=PAPER_SENSOR_COLUMNS).fit(marked)
n.statistics_
```

```
 condition   column     min     max        mean      std  n_rows
         0 sensor_2  641.21  644.53  642.680934 0.500041   20631
         0 sensor_3 1571.04 1616.91 1590.523119 6.131001   20631
         0 sensor_4 1382.25 1441.49 1408.933782 9.000387   20631
         0 sensor_6   21.60   21.61   21.609803 0.001389   20631
```

`std(ddof=0)` divides by the number of rows rather than by that number minus
one. With 20 631 rows the difference is in the fifth decimal place; the choice
is fixed so the number is reproducible rather than ambiguous.

### Step by step — `transform` applies

The three methods, all on the same real column. Raw first four values of
`sensor_4`:

```
  raw: [1400.60  1403.14  1404.20  1401.87]
```

**`zscore`** — using `mean = 1408.933782`, `spread = 9.000387`:

```python
scaled[mask] = (raw[mask] - float(stats["mean"])) / spread
```

```
  by hand (x-mean)/std : [-0.925936 -0.643726 -0.525953 -0.784831]
  from transform       : [-0.925936 -0.643726 -0.525953 -0.784831]
```

Check the first by hand: `(1400.60 - 1408.933782) / 9.000387 = -0.925936`. ✓
All four are negative because all four sit below the column's average — which
is expected, since these are a healthy engine's opening flights and `T50`
climbs as the engine wears.

Afterwards the whole column has exactly the intended shape:

```
  mean 0.0000000000   spread 1.0000000000
```

**`minmax_global` and `minmax_per_condition`** — using `min = 1382.25`,
`max = 1441.49`:

```python
unit = (raw[mask] - minimum) / span
scaled[mask] = unit * (high - low) + low
```

```
  by hand (x-min)/(max-min): [0.309757 0.352633 0.370527 0.331195]
  from transform           : [0.309757 0.352633 0.370527 0.331195]
  after rescaling, the column runs from 0.000000 to 1.000000
```

The extra `* (high - low) + low` maps the 0-to-1 result onto whatever
`feature_range` asks for; with the default `(0.0, 1.0)` it changes nothing.

All three side by side:

```
  raw                  : [1400.6    1403.14   1404.2    1401.87  ]
  zscore               : [-0.925936 -0.643726 -0.525953 -0.784831]
  minmax_global        : [ 0.309757  0.352633  0.370527  0.331195]
  minmax_per_condition : [ 0.309757  0.352633  0.370527  0.331195]
```

**The two min-max results are identical here**, and that is not a coincidence:
FD001 has one operating condition, so "per condition" and "global" are the same
grouping. They only diverge on FD002 and FD004.

### A constant sensor cannot be rescaled

Dividing by a spread of zero is undefined, so both paths check first:

```python
if spread > _ZERO_RANGE_TOLERANCE:  ... else: scaled[mask] = 0.0
```

Fitting on all 21 columns rather than the paper's 14 makes the warnings fire:

```
WARNING | Sensor column sensor_1 is constant within operating condition 0. It will
be mapped to a fixed value, and it carries no information for the model.
WARNING | Sensor column sensor_5 is constant within operating condition 0. ...
WARNING | Sensor column sensor_10 ...   sensor_16 ...   sensor_18 ...   sensor_19 ...
```

Six of them — exactly the six [`sensors`](../sensors/README.md) measures as
perfectly flat. What such a column becomes:

```
  sensor_1 raw distinct: 1  ->  rescaled distinct: 1   value: 0.0        (zscore)
  under minmax it becomes the bottom of the range:     0.0
```

The pipeline never hits these warnings, because it passes the paper's 14
columns and none of those is perfectly constant.

---

## The rule that protects the experiment: fit on training rows only

This is the most important idea on the page.

### What would go wrong

`fit` computes averages, spreads, minimums and maximums. If the test rows were
included in that computation, those numbers would carry information about the
test engines — and every training example would then be rescaled using a
quantity that depended on data the model is supposed never to have seen. The
reported error would be optimistic, and there would be no way to tell by how
much.

This is called **leakage**: information crossing from the held-back data into
the training process by a route nobody intended.

### It is not a hypothetical — measured on the real files

Fit the same rescaling twice, once on the 20 631 training rows and once on all
33 727 training-plus-test rows:

```
  column   mean_train_only  mean_train_plus_test   mean_shift   std_shift
sensor_2        642.680934            642.601005    -0.079929   -0.025259
sensor_3       1590.523119           1589.581927    -0.941192   -0.290764
sensor_4       1408.933782           1407.303559    -1.630222   -0.567855
sensor_6         21.609803             21.609763    -0.000040    0.000131
sensor_7        553.367711            553.519073     0.151362   -0.051113
sensor_8       2388.096652           2388.086677    -0.009974   -0.003753

  rows used: train only = 20631   train+test = 33727
  largest absolute shift in the mean: 2.654215 on sensor_9
  every column's mean moved? True
```

**Every single column's measured average changes.** And that changes the actual
numbers the network is given:

```
  first test reading of sensor_4: 1398.21
  rescaled with training-only numbers : -1.191480
  rescaled with train+test numbers    : -1.078390
```

The same physical reading becomes a different input depending on which rows
were allowed into the measurement. Nothing raises, nothing warns; the second
number is simply not one you are entitled to compute.

### How the code enforces it

By splitting `fit` from `transform`. `fit` is called once, on training rows.
`transform` is then called on each of training, validation and test — applying
the same stored numbers to all three. Calling `transform` before `fit` is
refused outright:

```
RuntimeError: Call fit() before transform().
```

The related question of *when* in the pipeline the split happens — before or
after scaling — is a separate recorded choice, **A-25**, and
[`pipeline`](../pipeline/README.md) is where it is carried out.

The key idea is: `fit` and `transform` are two methods rather than one because
the set of rows you are allowed to measure is smaller than the set you have to
apply the measurement to.

---

## The judgement calls recorded in this file

### A-03 — how the sensors are rescaled

**The gap.** The paper **never describes how it scales its inputs**, and never
mentions operating conditions in its preprocessing at all. Nothing in
`SensorNormalizer` reconstructs what Wen et al. did, because there is nothing to
reconstruct.

**What we chose.** `zscore` — subtract the mean, divide by the spread — as the
default, selected under the rule in A-22. The `minmax_per_condition` option
follows Jin et al. (2025) and was the default until 2026-08-29.

**How much it matters — and this is the largest measured effect in the whole
project.** Five seeds each, everything else identical (commit `09628fa`):

| Scaling | RMSE on FD001 |
|---|---|
| `minmax_per_condition` (default until 2026-08-29) | 20.353 ± 0.752 |
| `zscore` (the default now) | **16.524 ± 0.339** |

**3.83 cycles, paired t-test p = 0.0005.** That is 71 % of the entire gap to
the paper's 14.93, and larger than every other assumption tested combined.

Those numbers are quoted from **F-10** in
[the assumptions log](../../../../docs/ASSUMPTIONS.md); they were measured on a
named commit and are not re-derived on this page.

**What that means for any number this project reports.** It does not tell us
what Wen et al. did. It does mean this choice is the single biggest reason a
number from this code cannot be set beside a published one, and that sentence
belongs next to any result quoted from here.

### A-08 — the order of the preprocessing steps

Rescaling sits between smoothing and windowing, and that order is **not** a
choice: Figure 1 of the paper prints the chain explicitly. The entry is kept
under its original number, **A-08**, and now records why it is not an
assumption — along with a measured caution about what that order does on FD002
and FD004, where several operating conditions are in play. Read it before
running this pipeline on those two.

---

## What can go wrong

**An unknown method**, caught when the object is built rather than when it runs:

```
ValueError: Unknown normalisation method 'nope'. Valid options are:
['minmax_per_condition', 'minmax_global', 'zscore'].
```

**No columns given:**

```
ValueError: At least one column must be given to normalise.
```

**A backwards range:**

```
ValueError: feature_range must be increasing, got (1.0, 0.0).
```

**Columns that are not in the table:**

```
KeyError: Columns not present in the frame: ['nope'].
```

**Per-condition scaling without the condition column** — running the steps in
the wrong order:

```
KeyError: Column 'operating_condition' is required for per-condition scaling.
Run OperatingConditionEncoder first.
```

**Using either class before fitting it:**

```
RuntimeError: Call fit() before transform().
RuntimeError: The encoder must be fitted before use.
```

**A condition in the test data that never occurred in training.** This is the
subtle one, and it cannot happen on FD001:

```
KeyError: No statistics were learned for column 'sensor_4' under operating
condition 3. This happens when the test data contains a condition that never
occurred in training.
```

It is a real risk on FD002 and FD004, where six profiles are in play and a rare
one could in principle appear only on the test side. Failing loudly is the right
behaviour: there is no honest way to rescale a reading taken under conditions
never measured.

---

The key idea is: rescaling removes the accident of units so no sensor dominates
by size alone — and it must be measured on the training engines only, because
every number `fit` computes becomes part of what the model is told.
