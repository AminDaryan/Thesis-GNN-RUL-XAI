# `sensors` — the 21 sensors, the 14 the paper keeps, and why their order carries meaning

This file holds no algorithm. It is a list: the names, descriptions and units
of all 21 sensors in the dataset, which 14 of them the paper uses, and in what
order. It also offers one diagnostic that looks at real data and reports which
sensors never actually move.

The order of those 14 is the single most fragile thing in this folder, and most
of this page is about why.

For what C-MAPSS is and what a flight cycle means, see
[the data folder overview](../README.md). In one sentence: each row of the data
is one engine on one flight, and columns 6 to 26 of that row are 21 sensor
readings taken during it.

---

## What this file is given, and what it hands back

| | |
|---|---|
| **Given** | Nothing, for the lists — they are written into the file. For the one diagnostic, a table of loaded data. |
| **Hands back** | Names and orderings that the rest of the project indexes by, and a three-way report on which sensors carry information |

Where it sits in the chain: everywhere. It has no step of its own — it is
imported by [`loading`](../loading/README.md), [`smoothing`](../smoothing/README.md),
[`normalization`](../normalization/README.md), [`windowing`](../windowing/README.md),
[`pipeline`](../pipeline/README.md) and the graph code, because all of them need
to agree on what a column is called and which sensors are in play.

```
                     +-- loading      (names the 26 raw columns)
                     +-- smoothing    (which columns to smooth)
      THIS FILE  ----+-- normalization(which columns to rescale)
                     +-- windowing    (which columns become node features)
                     +-- graph        (which sensor each graph position is)
```

---

## What you need to know first

### A sensor has two names, and both are used

Every sensor has a **symbol** — the short engineering name an aerospace
engineer would use, such as `T24` — and a **column name** — where it sits in
the file, `sensor_2`. The raw files carry neither; they are just 26 numbers per
line, so the names are supplied by this file.

```
symbol       T24                  what an engineer calls it
column       sensor_2             where it lives in the table
index        2                    its position among the 21, counting from 1
description  Total temperature at LPC outlet
unit         degR                 degrees Rankine, an absolute temperature scale
station      LPC outlet           which part of the engine it sits in
```

`LPC` is the low-pressure compressor, `HPC` the high-pressure compressor, `LPT`
and `HPT` the low- and high-pressure turbines. Those abbreviations appear in the
descriptions and are used for grouping sensors in figures.

### A tuple, and why these lists are tuples

A **tuple** in Python is a list that cannot be changed after it is made. The
sensor orderings are tuples rather than lists on purpose: an accidental
`.append` or `.sort` somewhere else in the project would be a silent disaster,
and a tuple refuses.

---

## The public names

| Name | What it is | What it gives you back |
|---|---|---|
| `SENSORS` | All 21 sensors, in file order. | A tuple of 21 `SensorInfo` |
| `SensorInfo` | The description of one sensor. | An object with `.index`, `.symbol`, `.description`, `.unit`, `.station` and a `.column` |
| `PAPER_SENSOR_SYMBOLS` | The 14 the paper keeps, **in the order of its Table 3**. Do not reorder. | A tuple of 14 symbols |
| `PAPER_SENSOR_COLUMNS` | The same 14, as column names. | A tuple of 14 column names |
| `PAPER_EXCLUDED_SYMBOLS` | The 7 the paper drops. | A tuple of 7 symbols |
| `symbol_to_column(symbol)` | `"T24"` → `"sensor_2"`. | A string; raises if the symbol is unknown |
| `column_to_symbol(column)` | `"sensor_2"` → `"T24"`. | A string; raises if the column is unknown |
| `detect_uninformative_sensors(frame)` | Looks at real data and sorts the 21 sensors three ways. | A dictionary with keys `"constant"`, `"low_resolution"`, `"informative"` |
| `UNIT_COL`, `CYCLE_COL` | The engine and cycle column names. | `"unit"` and `"cycle"` |
| `OP_SETTING_COLS`, `SENSOR_COLS`, `RAW_COLUMNS` | The 3 setting columns, the 21 sensor columns, and all 26 in file order. | Lists of strings |

---

## `SensorInfo` and `SENSORS` — the catalogue

### What it is given, and what it hands back

Nothing goes in — `SENSORS` is a fixed list written into the file. What comes
out is the description of any one of the 21 sensors, looked up by position.

### The output

```python
from thesis_gnn_rul_xai.data.sensors import SENSORS
s = SENSORS[3]
```

```
SensorInfo(index=4, symbol='T50', description='Total temperature at LPT outlet',
           unit='degR', station='LPT outlet')

  .index: 4   .symbol: T50   .column: sensor_4
  .description: Total temperature at LPT outlet
  .unit: degR   .station: LPT outlet
```

Note that `SENSORS[3]` has `index=4`. Python counts from 0, the dataset counts
from 1. `.column` is a computed property, not stored: it is built as
`f"sensor_{self.index}"`, so the column name can never drift out of step with
the index.

The names, descriptions and units are taken from Table 2 of Wen et al. (2026),
which in turn follows Saxena et al. (2008). **That is a statement about the
paper**, checkable in the PDF — not a choice made here.

---

## `PAPER_SENSOR_SYMBOLS` — the 14, in order

### What it is given, and what it hands back

Nothing goes in. What comes out is an ordering — and the ordering, not the
membership, is the part that carries meaning, as the demonstration below shows.

### The output

```
  position  0  T24        sensor_2   Total temperature at LPC outlet
  position  1  T30        sensor_3   Total temperature at HPC outlet
  position  2  T50        sensor_4   Total temperature at LPT outlet
  position  3  P15        sensor_6   Total pressure in bypass duct
  position  4  P30        sensor_7   Total pressure at HPC outlet
  position  5  Nf         sensor_8   Physical fan speed
  position  6  Nc         sensor_9   Physical core speed
  position  7  Ps30       sensor_11  Static pressure at HPC outlet
  position  8  phi        sensor_12  Ratio of fuel flow to Ps30
  position  9  NRf        sensor_13  Corrected fan speed
  position 10  BPR        sensor_15  Bypass ratio
  position 11  htBleed    sensor_17  Bleed enthalpy
  position 12  W31        sensor_20  HPT coolant bleed
  position 13  W32        sensor_21  LPT coolant bleed
```

The 7 dropped ones:

```
T2, P2, epr, NRc, farB, Nf_dmd, PCNfR_dmd
```

### Why the order is load-bearing — shown, not asserted

The sensor graph — which sensors are treated as connected — is stored as a
plain 14-by-14 grid of numbers. **Nothing in that grid records which sensor is
which.** Row 0 means "the sensor at position 0", and the only thing that says
what position 0 *is* is `PAPER_SENSOR_SYMBOLS`.

```python
from thesis_gnn_rul_xai.graph.adjacency import build_physics_adjacency
A, symbols = build_physics_adjacency()
```

```
adjacency matrix: type ndarray shape (14, 14) dtype float32
its symbols come from the same list? True

row 0 of the matrix: [0 1 0 1 1 1 0 1 0 1 1 0 0 0]
row 0 belongs to position 0, which the sensor list says is: T24
so position 0 is connected to positions [1, 3, 4, 5, 7, 9, 10]
   = ['T30', 'P15', 'P30', 'Nf', 'Ps30', 'NRf', 'BPR']
```

Now swap two entries of the sensor list — `T24` at position 0 with `P15` at
position 3 — and change nothing else:

```
correct list  position 0 = T24   position 3 = P15
swapped list  position 0 = P15   position 3 = T24

row 0 of A is unchanged either way: [0 1 0 1 1 1 0 1 0 1 1 0 0 0]

under the correct list it says T24 talks to ['T30', 'P15', 'P30', 'Nf', 'Ps30', 'NRf', 'BPR']
under the swapped list the SAME row would say P15 talks to ['T30', 'T24', 'P30', 'Nf', 'Ps30', 'NRf', 'BPR']

no error is raised; the graph is simply wired to the wrong sensors.
```

That is the whole danger in one transcript. The array is the same array. The
shapes still match, the code still runs, the training still converges, the
results are still plausible — and the model is now told that bypass-duct
pressure exchanges information with a set of sensors it has no physical
relationship to. **There is no error message for this.** The only defence is
that the list never changes.

Two defences exist besides discipline. `WindowedDataset` in
[`windowing`](../windowing/README.md) carries the sensor column names alongside
the array, so the order travels with the data rather than being re-derived. And
this file checks itself the moment it is imported.

### The import-time checks

```python
assert len(PAPER_SENSOR_SYMBOLS) == 14, "Table 3 defines exactly 14 sensors."
assert len(set(PAPER_SENSOR_SYMBOLS)) == 14, "Duplicate sensor in the kept set."
assert set(PAPER_SENSOR_SYMBOLS).isdisjoint(PAPER_EXCLUDED_SYMBOLS), ...
assert len(set(PAPER_SENSOR_SYMBOLS) | set(PAPER_EXCLUDED_SYMBOLS)) == 21, ...
```

These run when the file is first imported, so a contradiction stops the package
from loading at all rather than surfacing later as a wrong number. What each
one asserts, evaluated on the real lists:

```
  len(PAPER_SENSOR_SYMBOLS) == 14 -> True
  no duplicates                   -> True
  kept and dropped disjoint       -> True
  together they cover all 21      -> True
```

They cannot catch a **reordering**, though — a swapped pair still has 14 unique
symbols covering all 21. That is why the transcript above matters: the checks
guard the membership of the set, and only care and review guard its order.

---

## `symbol_to_column` and `column_to_symbol` — translating between the two names

### The input

One name, in either direction: a symbol such as `"T24"` for the first, or a
column name such as `"sensor_2"` for the second.

### Step by step

Both are dictionary lookups built once when the file loads:

```python
_SYMBOL_TO_SENSOR = {s.symbol: s for s in SENSORS}
_COLUMN_TO_SENSOR = {s.column: s for s in SENSORS}
```

Each wraps the lookup so an unknown name produces a message listing what *is*
valid, rather than a bare `KeyError` with just the missing key in it.

### The output

```
  symbol_to_column("T24")       = sensor_2
  column_to_symbol("sensor_2")  = T24

  PAPER_SENSOR_COLUMNS = ('sensor_2', 'sensor_3', 'sensor_4', 'sensor_6',
    'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13',
    'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21')
```

`PAPER_SENSOR_COLUMNS` is not typed out by hand — it is built by running
`symbol_to_column` over `PAPER_SENSOR_SYMBOLS`, so the two orderings cannot
disagree.

---

## `detect_uninformative_sensors` — a diagnostic, not a filter

**Read the heading again.** This function never changes which sensors the
pipeline uses. The pipeline always uses the paper's fixed 14. This exists so
the paper's choice can be *checked* against the data, and a disagreement
reported as a finding.

### The input

A loaded table — here, the real FD001 training frame, 20631 rows.

### Step by step

For each of the 21 sensors it computes three quantities and applies two tests
in order.

```python
values = frame[sensor.column]
value_range = float(values.max() - values.min())
scale = max(abs(float(values.mean())), 1e-12)
n_distinct = int(values.nunique())
```

**Test 1 — is it constant?**

```python
if value_range <= relative_tolerance * scale:
```

The test is **relative**, not absolute, and it has to be. The sensors are on
wildly different scales: a spread of 0.01 is nothing for a temperature near
1400 but enormous for a fuel-air ratio near 0.03. Dividing the spread by the
average size of the readings makes the comparison fair. `relative_tolerance`
defaults to `1e-9`, which is effectively "did not move at all".

`scale` is guarded with `max(..., 1e-12)` so a sensor whose average is zero
cannot cause a division by zero.

**Test 2 — does it take only a handful of values?**

```python
elif n_distinct < min_distinct_values:
```

`nunique()` counts how many different values appear. The default threshold is
3, so a sensor with 1 or 2 distinct levels is flagged. This catches a signal
that moves, but only between a few fixed steps — usually because the quantity
was rounded coarsely.

Anything that passes both tests is called informative.

### The output

```python
detect_uninformative_sensors(train)
```

```
  constant        ( 6): ['Nf_dmd', 'P2', 'PCNfR_dmd', 'T2', 'epr', 'farB']
  low_resolution  ( 1): ['P15']
  informative     (14): ['BPR', 'NRc', 'NRf', 'Nc', 'Nf', 'P30', 'Ps30', 'T24',
                         'T30', 'T50', 'W31', 'W32', 'htBleed', 'phi']
```

Each list is sorted alphabetically, so the output is stable between runs.

### Why "distinct values" and not "range" for the second test

This is the part that is easy to get wrong, and the data settles it. Compare
the physical fan speed `Nf` with the bypass-duct pressure `P15`:

```
  Nf    relative range 0.000276  over  53 distinct levels
  P15   relative range 0.000463  over   2 distinct levels
```

`P15` has the **larger** relative range of the two, and it is the useless one:
it takes exactly two values in the entire training set. `Nf` moves less overall
but does so across 53 levels, and it is genuinely informative. A range-based
test would keep the wrong one. Counting distinct levels gets it right.

The full table of both criteria on every sensor:

| symbol | range | mean | relative range | distinct | verdict |
|---|---|---|---|---|---|
| T2 | 0 | 518.67 | 0 | 1 | constant |
| T24 | 3.32 | 642.681 | 0.00517 | 310 | informative |
| T30 | 45.87 | 1590.52 | 0.0288 | 3012 | informative |
| T50 | 59.24 | 1408.93 | 0.042 | 4051 | informative |
| P2 | 0 | 14.62 | 0 | 1 | constant |
| P15 | 0.01 | 21.6098 | 0.000463 | 2 | low_resolution |
| P30 | 6.21 | 553.368 | 0.0112 | 513 | informative |
| Nf | 0.66 | 2388.1 | 0.000276 | 53 | informative |
| Nc | 222.86 | 9065.24 | 0.0246 | 6403 | informative |
| epr | 0 | 1.3 | 0 | 1 | constant |
| Ps30 | 1.68 | 47.5412 | 0.0353 | 159 | informative |
| phi | 4.69 | 521.413 | 0.00899 | 427 | informative |
| NRf | 0.68 | 2388.1 | 0.000285 | 56 | informative |
| NRc | 193.78 | 8143.75 | 0.0238 | 6078 | informative |
| BPR | 0.2599 | 8.44215 | 0.0308 | 1918 | informative |
| farB | 0 | 0.03 | 0 | 1 | constant |
| htBleed | 12 | 393.211 | 0.0305 | 13 | informative |
| Nf_dmd | 0 | 2388 | 0 | 1 | constant |
| PCNfR_dmd | 0 | 100 | 0 | 1 | constant |
| W31 | 1.29 | 38.8163 | 0.0332 | 120 | informative |
| W32 | 0.7242 | 23.2897 | 0.0311 | 4745 | informative |

The key idea is: a sensor is useless when its readings cannot distinguish one
moment in an engine's life from another, and the number of distinct levels
measures that directly while the size of the range does not.

---

## The judgement calls recorded in this file

### A-10 — the seventh dropped sensor is NRc

**The gap.** Section 3.2.2 of the paper says seven sensors are removed, and
then names only six: T2, P2, epr, farB, Nf_dmd and PCNfR_dmd. The seventh is
never named in that sentence.

**What we chose.** NRc, the corrected core speed. It is the one sensor present
in the paper's Table 2 and absent from its Table 3 without being named in that
list.

**Why, and what the data says about it.** The six the paper names are exactly
the six the data shows to be perfectly constant:

```
  paper names      : ['T2', 'P2', 'epr', 'farB', 'Nf_dmd', 'PCNfR_dmd']
  measured constant: ['Nf_dmd', 'P2', 'PCNfR_dmd', 'T2', 'epr', 'farB']
  the same set?  True
```

NRc is a different case, and this is worth stating precisely because it changes
what the assumption means:

```
  distinct values of NRc across 20631 training rows: 6078
  is NRc constant? False
  min, max: 8099.94 8293.72
```

NRc is **not** flat. So it was not dropped for being uninformative — it must
have been dropped for some other reason, most plausibly redundancy, since
corrected core speed is derived from the physical core speed `Nc`, which the
paper keeps. Identifying it is an elimination, not a measurement, and the full
entry is in [the assumptions log](../../../../docs/assumptions/A-10.md) (assumption
A-10).

### A finding, not an assumption: P15 is kept but carries almost nothing

```
  distinct values of P15: 2 -> [21.6, 21.61]
  is P15 in the paper's kept 14? True
  which bucket did the diagnostic put it in? ['low_resolution']
```

The bypass-duct pressure is one of the paper's 14 graph nodes, and on FD001 it
takes exactly two values across the whole training set. It cannot represent
gradual degradation.

**It is kept anyway.** The adjacency matrix is defined over the paper's fixed
14-sensor set, and removing a node would change the graph into something the
paper did not describe. So this is reported rather than quietly corrected —
[`pipeline`](../pipeline/README.md) emits a warning naming it on every FD001 run.

This is a **finding about the paper** established from data, so it is an `F-xx`
entry rather than an `A-xx` one: it is not a choice this project made.

---

## What can go wrong

**An unknown sensor symbol.** The message lists every valid one:

```
KeyError: Unknown sensor symbol 'T51'. Valid symbols are: ['BPR', 'NRc',
'NRf', 'Nc', 'Nf', 'Nf_dmd', 'P15', 'P2', 'P30', 'PCNfR_dmd', 'Ps30', 'T2',
'T24', 'T30', 'T50', 'W31', 'W32', 'epr', 'farB', 'htBleed', ...]
```

**An unknown column name.** The same, in the other direction:

```
KeyError: Unknown sensor column 'sensor_99'. Valid columns are: ['sensor_1',
'sensor_10', 'sensor_11', ...]
```

**A table missing one of the 21 sensor columns**, passed to the diagnostic. It
walks all 21 unconditionally, so a missing column surfaces as pandas' own bare
lookup failure rather than a helpful message:

```
KeyError('sensor_5')
```

That is the one rough edge on this page. It is only reachable by handing the
diagnostic a table that did not come from
[`read_cmapss_frame`](../loading/README.md), which always produces all 26
columns.

**A contradiction in the lists themselves** stops the import. Because the
checks are plain `assert` statements, running Python with optimisations turned
on (`python -O`) removes them, and the package would then load a contradictory
sensor list without complaint.

---

The key idea is: this file is a lookup table, and its one dangerous property is
that the graph is indexed by *position* in that table rather than by name — so
membership is checked automatically, and order is protected only by never
changing it.
