# `labeling` — stop the label counting above a ceiling

This is the shortest file in the folder — 41 lines, two functions, and one
`np.minimum` call doing all the work. It carries one of the largest ideas in
the project, so this page is much longer than the code.

What it does: the previous step,
[`loading`](../loading/README.md), produced a `rul` column counting every
remaining flight, so a brand-new engine gets a label like 191. This file
replaces every label above a ceiling with the ceiling itself. A healthy engine
stops being labelled "191 flights left" and starts being labelled, in effect,
"healthy".

For what a flight cycle and remaining useful life are, see
[the data folder overview](../README.md).

---

## What this file is given, and what it hands back

| | |
|---|---|
| **Given** | A table with a `rul` column counting all the way up, and a ceiling |
| **Hands back** | A **new** table whose `rul` column never exceeds that ceiling |

Where it sits in the chain:

```
loading (rul counts to 361)  ->  THIS FILE  ->  rul capped at 125  ->  smoothing  ->  normalization  ->  windowing
```

---

## What you need to know first

### Why "cycles until failure" is a bad thing to predict

This looks like the obvious target, and it is not. The reason can be
**measured** rather than argued, which is what the rest of this section does.

Take training engine 1. It ran 192 flights, so its raw labels count down
191, 190, …, 0. Now look at what one of its sensors was doing while that
happened. `T50` is the temperature at the low-pressure turbine outlet, one of
the sensors the paper keeps:

```python
e1 = train[train.unit == 1]
e1[e1.cycle.isin([1, 20, 40, 60, 150, 180, 192])][["cycle", "rul", "sensor_4"]]
```

```
 cycle  rul  sensor_4
     1  191   1400.60
    20  172   1405.23
    40  152   1401.77
    60  132   1403.07
   150   42   1409.22
   180   12   1417.14
   192    0   1427.20
```

Read the two halves separately.

Over the **first 60 flights** the label falls from 191 to 132 — a change of 59
— while `T50` wanders between 1394 and 1406 with no trend. Over the **last 43
flights** the label falls from 42 to 0 and `T50` climbs steadily from 1408 to
1432.

Put a number on it. **Correlation** here means: as one quantity goes down, does
the other reliably go up or down? It runs from −1 (perfectly opposed) through 0
(no relationship at all) to +1 (perfectly together).

```python
early = e1[e1.cycle <= 60]
late  = e1[e1.cycle >= 150]
np.corrcoef(early.rul, early.sensor_4)[0, 1]
np.corrcoef(late.rul,  late.sensor_4)[0, 1]
```

```
  first 60 cycles  (60 rows): label falls 191 -> 132, T50 range 1394.75..1406.22, correlation -0.083
  last  43 cycles  (43 rows): label falls  42 ->   0, T50 range 1408.09..1432.52, correlation -0.715
```

**−0.083 is nothing.** Over the healthy stretch the sensor tells you
essentially nothing about how much life is left. Over the worn-out stretch,
−0.715 is a strong relationship.

One engine is not a result, so the same measurement over all 100 training
engines — each engine's own first 60 and last 43 flights:

```
  mean correlation over each engine's first 60 cycles : -0.094  (n=100)
  mean correlation over each engine's last 43 cycles  : -0.733  (n=100)
```

The pattern holds across the whole dataset.

So with a raw label, for the healthy part of an engine's life you are asking
the model: *"this window has 191 flights left and that one has 132 — tell them
apart."* The sensors do not distinguish them, because hardly any wear has
happened yet. **The model can only guess**, and that guessing does not stay
politely confined to the healthy windows: it becomes part of what the model
learns, and degrades the predictions it could otherwise have got right.

### The fix, and the shape it produces

The fix comes from Heimes (2008): stop the label counting above a fixed number.

```
label(t) = the smaller of (flights until failure) and the ceiling
```

Drawn against time, the label is now two straight pieces — flat, then a slope
down to zero. That shape is what the literature calls a **piecewise-linear**
target: "piecewise" meaning made of separate pieces, "linear" meaning each
piece is a straight line. The place where the two meet is often called the
*change point*.

```
label
  125 |*********------                  flat while the engine looks healthy
      |               \
      |                \
      |                 \               counting down once wear shows
      |                  \
    0 +-------------------\------------> flight
      1                68            192
                       ^
                    the corner
```

For engine 1 with a ceiling of 125, that corner sits at flight 68 — measured
below, not guessed.

---

## The public names

| Name | What it is | What it gives you back |
|---|---|---|
| `apply_piecewise_rul(frame, rul_max=..., column="rul", output_column=None)` | Caps a column of a table at a ceiling. | A **new** table; the one passed in is unchanged |
| `piecewise_rul_curve(total_cycles, rul_max)` | The label curve one engine of a given length *should* have. | A NumPy array of whole numbers, one per flight |

---

## `piecewise_rul_curve` — the curve one engine should have

This one touches no data. It builds the ideal label sequence for an engine of a
given length, which is what the tests compare against and what the figures draw.

### The input

Two whole numbers: how many flights the engine ran, and the ceiling.

### Step by step

```python
cycles = np.arange(1, total_cycles + 1)
return np.minimum(total_cycles - cycles, rul_max)
```

Two lines. Worked through with `total_cycles = 6` and `rul_max = 3`:

```
  cycles = np.arange(1, 7)   ->  [1 2 3 4 5 6]      the flight numbers
  total - cycles             ->  [5 4 3 2 1 0]      raw flights remaining
  np.minimum(that, 3)        ->  [3 3 3 2 1 0]      each one capped at 3
```

`np.arange(1, 7)` produces the whole numbers from 1 up to but **not
including** 7. `np.minimum(a, 3)` compares every element of `a` against 3 and
keeps the smaller — element by element, not a single overall minimum.

Notice that the first entry is `5`, not `6`: at flight 1 the engine has 5 more
flights ahead of it, and at flight 6 it has 0. The last value is always 0.

### The output

```python
piecewise_rul_curve(6, rul_max=3)
```

```
[3 3 3 2 1 0]     type: ndarray   dtype: int64
```

A NumPy array — a fixed-size grid of numbers of one type — of 6 whole numbers.
`int64` means each is a whole number stored in 64 bits.

On engine 1's real length:

```python
full = piecewise_rul_curve(192, rul_max=125)
```

```
  length: 192   first five: [125 125 125 125 125]   last five: [4 3 2 1 0]
  how many cycles sit flat at the ceiling? 67
  the corner is at cycle 68
```

Check that by hand: the raw label at flight `t` is `192 - t`, which first drops
below 125 when `192 - t < 125`, that is `t > 67`. So flights 1 to 67 sit at the
ceiling and flight 68 is the first one counting down — exactly what was
measured.

### An edge worth knowing

If the ceiling is higher than the engine is long, it never binds:

```python
piecewise_rul_curve(6, rul_max=10)
```

```
[5 4 3 2 1 0]
```

No error, no flat section — the curve is just the raw countdown. This is the
normal situation for a short test engine.

---

## `apply_piecewise_rul` — the cap applied to real data

### The input

A table carrying a `rul` column — the output of
[`attach_rul`](../loading/README.md) — and a ceiling.

### Step by step

**1. Refuse a ceiling below 1.**

```python
if rul_max < 1:
    raise ValueError(f"rul_max must be positive, got {rul_max}.")
```

A ceiling of 0 would label every row 0, which is to say "every engine is
failing right now". It is caught rather than allowed.

**2. Refuse a column that is not there.**

```python
if column not in frame.columns:
    raise KeyError(f"Column {column!r} not found. Available columns: {list(frame.columns)}.")
```

The message lists what *is* available, so a typo is diagnosable from the error
alone.

**3. Copy, then cap.**

```python
frame = frame.copy()
frame[output_column or column] = np.minimum(frame[column].to_numpy(), rul_max)
return frame
```

`frame.copy()` first, so the table handed in is never modified.

`frame[column].to_numpy()` pulls the column out as a plain NumPy array.
`np.minimum(array, 125)` compares every element against 125 and keeps the
smaller one — so 191 becomes 125, and 92 stays 92.

`output_column or column` is Python's "use the first if it is set, otherwise
the second". Pass `output_column="rul_capped"` and the capped values go into a
**new** column beside the original, which is what the figures use to draw the
before and after together. Leave it out and the `rul` column is written over.

### The output

With `output_column="rul_capped"`, so both can be seen side by side, on the
real engine 1:

```python
capped = apply_piecewise_rul(train, rul_max=125, output_column="rul_capped")
```

```
 cycle  rul  rul_capped
     1  191         125
    30  162         125
    60  132         125
    67  125         125
    68  124         124   <- from here on the cap does nothing
   100   92          92
   192    0           0
```

Without `output_column`, the `rul` column itself changes and no new column
appears:

```
  columns added: set()          (none - it wrote over 'rul')
  max rul before the call            : 361
  max rul in what came back          : 125
  max rul in the original, after the call: 361
  same object?                       : False
```

The original table still has its 361. That copy matters: the preprocessing
chain hands the same table to several steps, and a function that silently
rewrote a column would change what every later step sees.

### How much of the data this touches

```
  rows with a raw label above 125: 8031 of 20631 (38.9%)
  highest raw label anywhere: 361
```

Nearly two rows in five have their label changed by this step. It is not a
detail at the margins.

---

## The judgement calls recorded in this file

### A-01 — the ceiling is 125

**The gap.** The paper gives the piecewise-linear formula as its Equation 19,
but **never prints the ceiling it used**. Its Section 4.2 describes predicted
life holding "around 125" while an engine is healthy, which is suggestive but
is not a stated value.

**What we chose.** 125, set in `PreprocessingConfig`:

```python
PreprocessingConfig().rul_max  ->  125
```

**Why.** Heimes (2008) reported that any value between roughly 120 and 130
works well on this dataset, and Jin et al. (2025) use 125 on the same data.
The value here was settled by pixel-calibrating the plateau of the paper's own
Fig. 7 against that figure's axis ticks, which puts it at 125.2. That is a
measurement made from a figure, not a number the paper printed — and the
paper's schematic figure appears to disagree with its experimental one.

The full argument, including the disagreement between the two figures, is in
[the assumptions log](../../../../docs/ASSUMPTIONS.md) (assumption A-01).

### A-02 — the ceiling is applied to the test labels too

**The gap.** The paper does not say whether the official test labels are capped
before the error is computed.

**What we chose.** Cap them. This is controlled by `clip_test_labels` in the
configuration.

**Why.** The model's final layer is a sigmoid — an output squeezed into a fixed
range — scaled to the ceiling, so it **cannot** predict a value above 125 in
the first place. Leaving a test label at 180 would score the model against a
number it is structurally incapable of producing.

**How much it matters, and why it must always be stated.** Capping the test
labels can only move the reported error **favourably**: it removes exactly the
cases the model was going to be furthest from. So any number from this project
compared against a published one must carry that caveat. This is settled and
not up for re-litigation (assumption A-02 in
[the assumptions log](../../../../docs/ASSUMPTIONS.md)).

**Do not confuse** these two. The first is *where* the ceiling sits (assumption
A-01). The second is *who it is applied to* (assumption A-02). They are
independent choices and either could have gone the other way.

---

## What can go wrong

**A ceiling of zero or less:**

```
ValueError: rul_max must be positive, got 0.
```

**The column is not in the table:**

```
KeyError: "Column 'rul' not found. Available columns: ['unit', 'cycle',
'op_setting_1', 'op_setting_2', 'op_setting_3', 'sensor_1', 'sensor_2', ...]"
```

This is the error you get from running the steps in the wrong order — capping
before [`loading`](../loading/README.md) has attached the labels.

**A silent one, with no error at all:** calling this twice with different
ceilings. The second call caps an already-capped column, so a ceiling of 150
applied after a ceiling of 125 leaves everything at 125 and looks like it
worked. Nothing detects this. It is why the ceiling lives in the configuration
and is applied at exactly one point in
[`pipeline`](../pipeline/README.md), rather than being passed around.

---

The key idea is: the raw label asks the model to distinguish two healthy
engines that the sensors cannot distinguish — measured at a correlation of
−0.09 over the healthy stretch against −0.73 over the worn one — so the ceiling
replaces an unanswerable question with an answerable one.
