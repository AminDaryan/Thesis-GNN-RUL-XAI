# `smoothing` — take the jitter out of each engine's sensor readings

Sensor readings jump about from one flight to the next, and most of that
jumping is not the engine wearing out. This file replaces each reading with a
blend of itself and everything before it, so the slow drift survives and the
flight-to-flight noise is damped.

Two functions, and the second is the one the pipeline actually calls. The first
exists to prove the second is doing what it claims.

For what a flight cycle is, see [the data folder overview](../README.md).

---

## What this file is given, and what it hands back

| | |
|---|---|
| **Given** | A table of sensor readings, a list of which columns to smooth, and one setting called `alpha` |
| **Hands back** | A **new** table where those columns are smoothed and everything else — including the label — is untouched |

Where it sits in the chain:

```
labeling  ->  THIS FILE  ->  normalization  ->  windowing
```

---

## What you need to know first

### The problem, on real numbers

Here are the first six readings of `T50` — the temperature at the low-pressure
turbine outlet — for training engine 1:

```
[1400.60  1403.14  1404.20  1401.87  1406.22  1398.37]
```

Up, up, down, up, sharply down. Across those six flights the engine did not
meaningfully degrade — a turbofan takes hundreds of flights to wear out. The
movement is measurement noise and normal variation between flights. What the
model needs to see is the slow climb that shows up over hundreds of flights,
not this.

### The first idea, and why it is not quite right

The obvious fix is a **moving average**: replace each reading with the average
of the last few. It works, but it has two awkward properties. You must choose
how many is "a few", and every one of those readings counts equally — a
measurement from 10 flights ago carries exactly the weight of the one from last
flight, and then drops out entirely.

What you actually want is for the recent past to matter most and older readings
to fade gradually rather than fall off a cliff.

### The formula

```
s(t) = alpha * x(t) + (1 - alpha) * s(t-1)
```

Every symbol:

- **`t`** — which flight, counting from 1.
- **`x(t)`** — the raw sensor reading at flight `t`.
- **`s(t)`** — the smoothed value at flight `t`; this is what comes out.
- **`s(t-1)`** — the smoothed value at the *previous* flight. This is what
  makes the formula **recursive**: each answer is built from the previous
  answer rather than from scratch.
- **`alpha`** — a number between 0 and 1 saying how much to trust the new
  reading. It is chosen before running, not learned.
- **`s(1) = x(1)`** — the first flight has no previous value, so it is left as
  it is.

**Why this fades the past gradually.** Substitute the formula into itself and
`s(t)` turns out to be `alpha·x(t) + alpha(1-alpha)·x(t-1) +
alpha(1-alpha)²·x(t-2) + …`. Each step further back is multiplied by another
factor of `(1-alpha)`, so the weights shrink geometrically. Nothing is ever
dropped; it just stops mattering. That geometric shrinking is what
"exponentially weighted" refers to.

**It only ever looks backwards.** `s(t)` uses `x(t)` and earlier values, never
later ones. That is not an accident of the formula — it is essential. A filter
that peeked forwards would put information from flight `t+5` into the window
ending at flight `t`, and the model would be predicting remaining life partly
from the future.

---

## The public names

| Name | What it is | What it gives you back |
|---|---|---|
| `ewma_series(values, alpha)` | The filter on one series, written out longhand with a plain loop. | A NumPy array the same length as the input |
| `apply_ewma(frame, columns, alpha=..., ...)` | The filter over a whole table, one engine at a time. `alpha=None` turns it off. | A **new** table |

---

## `ewma_series` — the formula, written out the slow and obvious way

This is the reference version. It is not what the pipeline runs — the pipeline
uses pandas' fast built-in — but the tests check that the two agree, so this
function is what fixes the meaning of "smoothed" in this project.

### The input

A sequence of numbers and an `alpha`. Using the six real readings above.

### Step by step

```python
if not 0.0 < alpha <= 1.0:
    raise ValueError(f"alpha must lie in (0, 1], got {alpha}.")

array = np.asarray(values, dtype=float)
if array.size == 0:
    return array

smoothed = np.empty_like(array)
smoothed[0] = array[0]
for index in range(1, array.size):
    smoothed[index] = alpha * array[index] + (1.0 - alpha) * smoothed[index - 1]
return smoothed
```

`alpha` must be greater than 0 and at most 1. Zero would mean "ignore every new
reading", which would return the first value forever.

`np.asarray(values, dtype=float)` accepts a Python list, a NumPy array or a
pandas column and produces a NumPy array of decimals — a fixed-size grid of
numbers of one type.

`np.empty_like(array)` makes room for the answer without filling it in: same
length, same type, contents undefined until written. Every position is written
by the loop, so nothing undefined survives.

**The first three iterations, with the real numbers, `alpha = 0.5`:**

```
  s(1) = x(1)                          = 1400.600000
  s(2) = 0.5*1403.14 + 0.5*1400.600000 = 1401.870000
  s(3) = 0.5*1404.20 + 0.5*1401.870000 = 1403.035000
  s(4) = 0.5*1401.87 + 0.5*1403.035000 = 1402.452500
```

Read `s(3)`: the raw reading was 1404.20 and the previous smoothed value was
1401.87, so the answer lands halfway between them. With `alpha = 0.5` each new
value is exactly the midpoint of the new reading and the running result.

Notice `s(4)`. The raw reading **fell** from 1404.20 to 1401.87, but the
smoothed value only fell from 1403.035 to 1402.45 — the filter absorbed most of
the drop. That is the whole point.

### The output

```python
ewma_series(v, alpha=0.5)
```

```
  from the code : [1400.6    1401.87   1403.035  1402.4525]
  by hand       : [1400.6    1401.87   1403.035  1402.4525]
  identical?    : True

  type: ndarray   dtype: float64   shape: (6,)
```

### What `alpha` actually does

Same real signal, twelve flights, three settings:

```
  raw          [1400.6 1403.14 1404.2 1401.87 1406.22 1398.37 1397.77 1400.97 1394.8 1400.46 1400.64 1400.15]
  alpha = 0.1  [1400.6 1400.85 1401.19 1401.26 1401.75 1401.41 1401.05 1401.04 1400.42 1400.42 1400.44 1400.41]
  alpha = 0.5  [1400.6 1401.87 1403.04 1402.45 1404.34 1401.35 1399.56 1400.27 1397.53 1399.   1399.82 1399.98]
  alpha = 0.9  [1400.6 1402.89 1404.07 1402.09 1405.81 1399.11 1397.9  1400.66 1395.39 1399.95 1400.57 1400.19]
```

Measured as the spread between the largest and smallest value over those twelve
flights:

```
    raw        11.420
    alpha=0.1   1.338
    alpha=0.5   6.803
    alpha=0.9  10.421
```

**`alpha` near 1 barely smooths** — at 0.9 the output nearly follows the raw
signal, keeping 10.4 of the original 11.4 spread. **`alpha` near 0 smooths
hard** — at 0.1 the signal is flattened to a spread of 1.3, which removes the
noise but would also flatten genuine degradation. `alpha = 1.0` is exactly no
smoothing at all:

```python
ewma_series([1., 5., 2.], alpha=1.0)  ->  [1. 5. 2.]
```

The key idea is: `alpha` sets how fast the past is forgotten, and every value
between the two extremes trades noise removal against responsiveness.

---

## `apply_ewma` — the filter over a whole table

### The input

The full training table, the 14 sensor columns the paper keeps, and `alpha`.

### Step by step

**1. `alpha=None` means "do not smooth", and returns a copy anyway.**

```python
if alpha is None:
    logger.info("Smoothing disabled (alpha is None); data passed through unchanged.")
    return frame.copy()
```

```
INFO | thesis_gnn_rul_xai.data.smoothing | Smoothing disabled (alpha is None); data passed through unchanged.

  identical values?         True
  but a different object?   True
```

This is how the filter's contribution is measured: run the pipeline once with
smoothing and once without, and compare. Returning a **copy** rather than the
original keeps the two paths behaving identically in every other respect, so
nothing downstream can accidentally depend on which one it got.

**2. Reject a bad `alpha`, and reject columns that are not there.**

```python
if not 0.0 < alpha <= 1.0:
    raise ValueError(...)
missing = [c for c in columns if c not in frame.columns]
if missing:
    raise KeyError(f"Columns not present in the frame: {missing}.")
```

The missing-column check reports **all** the bad names at once, not just the
first:

```
KeyError: Columns not present in the frame: ['sensor_99', 'nope'].
```

**3. Sort, then smooth within each engine.**

```python
frame = frame.sort_values([unit_column, cycle_column]).copy()
frame[list(columns)] = (
    frame.groupby(unit_column, sort=False)[list(columns)]
    .transform(lambda column: column.ewm(alpha=alpha, adjust=False).mean())
)
```

The sort guarantees consecutive rows of one engine are consecutive flights — a
recursive filter applied to shuffled rows would be meaningless.

`groupby(unit_column)` splits the table by engine. `.transform(...)` applies the
filter to each engine's block **separately** and reassembles the results into a
column of the original length, so the answer lines up row for row with the
table it came from.

`.ewm(alpha=alpha, adjust=False).mean()` is pandas' fast version of exactly the
loop in `ewma_series`. Confirmed on the real data:

```
  matches ewma_series?  True
```

### Why one engine at a time — shown with real numbers

This is the part that would be easy to get wrong and hard to notice. The
engines sit end to end in one table: engine 1's last flight is immediately
followed by engine 2's first.

```
  engine 1's last 3 readings : [1425.95 1425.77 1427.20]     <- a worn-out engine, running hot
  engine 2's first 3 readings: [1391.28 1393.13 1398.96]     <- a brand-new one, running cool
```

Smooth the column **without** grouping and the filter carries engine 1's dying
readings straight into engine 2:

```
  engine 2's first 3, smoothed PER ENGINE (what the code does): [1391.28   1392.205  1395.5825]
  engine 2's first 3, smoothed across the whole table         : [1408.9306 1401.0303 1399.9951]
  difference:                                                   [  17.6506    8.8253    4.4126]
```

Engine 2's opening reading would be recorded as **1408.93 instead of 1391.28** —
off by 17.65 degrees, and wrong in a specific direction: a new engine would look
worn, because the previous engine was. No error is raised; the numbers are
simply wrong, and the contamination fades over the following flights rather than
announcing itself.

### The output

```python
apply_ewma(train, PAPER_SENSOR_COLUMNS, alpha=0.5)
```

```
INFO | thesis_gnn_rul_xai.data.smoothing | Applied exponentially weighted moving average (alpha=0.500) to 14 columns across 100 engines.

  shape in: (20631, 26)   shape out: (20631, 26)

  engine 1, sensor_4, first 4 flights
    before: [1400.6   1403.14   1404.2    1401.87  ]
    after : [1400.6   1401.87   1403.035  1402.4525]

  a column NOT in the list is untouched (sensor_1): True
  the original frame is untouched:                  True
```

Same shape in and out — this changes values, never the number of rows or
columns. Only the named columns move. **The label is deliberately not in that
list**: smoothing the answer would be smoothing the thing being predicted.

---

## The judgement calls recorded in this file

### A-04 — the strength of the filter is 0.5

**The gap.** Section 3.2.3 of the paper contradicts itself here. It states that
"S(t-1) … is equal to 0.5 in this study", which attaches 0.5 to the *previous
smoothed value* rather than to `alpha`.

**What we chose.** Read it as `alpha = 0.5`.

**Why.** A fixed previous value makes no sense in a recursive formula — `s(t-1)`
is computed at every step, so it cannot be a constant. The only reading under
which that sentence describes a working filter is that 0.5 is the parameter.

**How much it matters.** That is an argument about what a sentence means, not a
measurement. The full entry is in
[the assumptions log](../../../../docs/assumptions/A-04.md) (assumption A-04).

### A-09 — which of pandas' two conventions is used

**The gap.** pandas offers two forms of this filter, chosen with `adjust=True`
or `adjust=False`, and they disagree. The paper prints one formula and does not
mention the distinction, because it is a pandas detail rather than a
mathematical one.

**What we chose.** `adjust=False`.

**Why, and what the two actually differ in.** They differ in how they handle the
opening values, where there is not yet much history to average. `adjust=False`
is the plain recursive formula. `adjust=True` re-normalises by the total weight
seen so far, which changes the early values. At element 1:

```
    adjust=False: 0.5*1403.14 + 0.5*1400.60           = 1401.870000
    adjust=True : (1*1403.14 + 0.5*1400.60)/(1 + 0.5) = 1402.293333
```

Across the first six real readings:

```
  raw            [1400.6  1403.14 1404.2  1401.87 1406.22 1398.37]
  adjust=False   [1400.6  1401.870 1403.035 1402.4525 1404.33625 1401.353125]   <- used here
  adjust=True    [1400.6  1402.293 1403.383 1402.5760 1404.45677 1401.365079]
  difference     [0.0     0.4233   0.3479   0.1235    0.12052    0.011954]
```

The gap shrinks as history accumulates — 0.42 at the second flight, 0.012 by
the sixth — but it never quite vanishes, and it is largest exactly where
engines are healthiest.

`adjust=False` is chosen because it is the formula the paper prints as its
Equation 18. `ewma_series` implements that same form, which is how the tests
pin it down:

```
  ewma_series matches adjust=False?  True
```

The entry is in [the assumptions log](../../../../docs/assumptions/A-09.md) (assumption A-09).

---

## What can go wrong

**`alpha` outside the allowed range.** Both functions check, with the same
message:

```
ValueError: alpha must lie in (0, 1], got 0.0.
ValueError: alpha must lie in (0, 1], got 1.5.
ValueError: alpha must lie in (0, 1], got -0.2.
```

Note the difference between `alpha=None`, which means "deliberately off and I
want a clean copy", and `alpha=0.0`, which is rejected. They are not the same
request, and conflating them is exactly the kind of thing this check exists to
stop.

**Columns that are not in the table:**

```
KeyError: Columns not present in the frame: ['sensor_99', 'nope'].
```

**An empty input** returns an empty array rather than failing:

```python
ewma_series([], alpha=0.5)  ->  []   shape (0,)
```

**The silent one.** Smoothing across engine boundaries, shown above, raises
nothing at all. It is prevented by the `groupby` and by nothing else, which is
why that line is worth reading carefully before changing.

---

The key idea is: each smoothed value is a blend of the newest reading and the
blend that came before it, so the past fades geometrically instead of being
dropped — and the blend must restart at every engine, because two engines' worth
of readings in one column are not one signal.
