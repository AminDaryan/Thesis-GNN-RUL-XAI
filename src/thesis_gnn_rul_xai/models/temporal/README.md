# `models/temporal` — what each sensor node ends up carrying

A window of thirty flight cycles goes in, per sensor. One or more numbers come
out, per sensor. That is the whole job of this folder.

```
(batch, 14 sensors, 30 cycles)  ->  a temporal stage  ->  (batch, 14 sensors, k)
```

The graph layers that follow never see a cycle. They see whatever this stage
leaves them, and `k` — the width of the first graph weight matrix `W(1)` — is
read off **the stage**, never off the data.

---

## Why this folder exists at all

Until this stage was written, the reduction happened one line before the
pipeline returned:

```python
windows = np.transpose(windows, (0, 2, 1))   # (n, 14, 30)
windows = windows[:, :, -1:]                 # (n, 14, 1) -- 29 cycles discarded
```

and the model took its width from the array it was handed:

```python
ESTGNN(adjacency, in_features=prepared.train.features.shape[2])
```

Two things were wrong with that, and the second one cost the project real work.

**It put a decision about the model inside the data.** Whether a sensor node
carries the last cycle, the average of the window, or something a small network
computed is a question about the architecture. It was being answered in
`data/windowing/`, where it looked like a property of the dataset.

**It left a path from a preprocessing setting to the shape of the network.**
Delete that slice and `W(1)` silently becomes `(30, 32)` instead of `(1, 32)`.
That is what happened on 2026-08-28: 30 raw cycles reached the first graph
layer, the model scored *better*, and two of the project's three configuration
comparisons reversed direction before anyone noticed the architecture had
changed. The full story is in `docs/ASSUMPTIONS.md` (assumption A-07).

With the stage in place there is no such path. The pipeline always hands over
every cycle, the stage says what happens to them, and the network's shape
follows from the stage.

---

## The stages

| Name | What a node ends up carrying | Learns? | `out_features` |
|---|---|---|---|
| `last_cycle` | the most recent reading of the window | no | 1 |
| `mean_window` | the average of the window | no | 1 |
| `window_summary` | the last reading, the average, and the change across the window | no | 3 |
| `raw_window` | every cycle, untouched | no | 30 |
| `lstm` | what a recurrent network read out of the window | yes | 1 by default |
| `gru` | the same, with a simpler recurrent cell, which the paper does not have (assumption A-36) | yes | 1 by default |

`last_cycle` is the default and is what every result in this project before
2026-09-09 was produced with. It is this project's choice (assumption A-07):
Table 4 of the paper prints one number per sensor entering the first graph
layer, that number comes out of a recurrent stage in the paper, and when that
stage is not used the latest reading stands in for it.

### `lstm` — what the paper actually describes

Section 2.2 of Wen et al. is explicit, and says it four times:

> "Each sensor s ∈ {1, 2, …, n} is associated with a **dedicated** LSTM network"

with Equation 7 indexing the hidden state by sensor, the channels processed
"independently" through "separate specialized pathways", and the architectural
strengths opening with "Independent parallel LSTM networks for each sensor".

So `LSTMStage` builds **fourteen** recurrent networks, one per sensor, each
reading that sensor's thirty cycles. This is a **citation, not a choice**:
`shared_across_sensors=True` exists as a cheaper alternative, and any number
produced with it has to say so.

What the paper does *not* say is narrower than "the LSTM is unspecified". Table
4 has no recurrent row at all — it begins at the node features — and Table 6
lists the learning rate, 200 epochs, batch size 32, thirty time steps, Adam and
MSE loss. There is no hidden size, no depth, and no projection. Those three are
this project's choices (assumption A-26).

Nor does the paper give the numbers the stage starts from before training. The
stage sets none, so PyTorch's own apply: every multiplier and offset in a cell
is drawn evenly from −1/√32 to +1/√32, about ±0.177. That starts each cell's
forget gate, the part that decides how much of the cell's memory is kept from
one cycle to the next, near one half; Gers et al. (2000), who added that gate
to the LSTM, start theirs above one half (assumption A-34).

### `window_summary` — three numbers a person could read off a plot

Not in the paper (assumption A-28). For each sensor it hands on the last
reading, the average of the thirty, and the **change**: how far the best
straight line through the thirty readings rises from the first cycle to the
last.

That line is the one that makes the squared vertical distances from the points
smallest. With the cycles numbered from the middle of the window, so that
`t = −14.5, −13.5, …, +14.5` and those numbers add up to zero, its slope is

```
slope  = Σ t × reading  /  Σ t²          (Σ t² = 2247.5 for 30 cycles)
change = slope × 29
```

`t` is how many cycles a reading sits after the window's middle, `reading` is
the scaled value at that cycle, and the sum runs over all thirty. Numbering from
the middle is what lets the average drop out of the formula: moving every
reading up by the same amount adds that amount times `Σ t`, which is zero, so
the slope ignores where the series sits and measures only which way it goes.
The change is multiplied back up by 29, the number of cycles from the first
reading to the last, so it is in the readings' own units, like the other two.

On the real data, `T50` of engine 1 over its cycles 1 to 30 (after smoothing and
scaling) ends at −1.2938, averages −1.0449, and its line falls by 0.3598 across
the window: a slope of −0.012408 per cycle, times 29. `WindowSummary(30)` gives
`[−1.2938, −1.0449, −0.3598]` for that sensor, and
`tests/test_examples_torch.py` checks it term by term.

It learns nothing and hands over three numbers, so it trips the same check as
`raw_window` below and `build_model` lets it through only when it is named.
`W(1)` becomes `3 × 32`.

### `raw_window` — kept, named, and gated

This is the arrangement removed on 2026-08-28. It is kept because the comparison
that decided the last-cycle stand-in (assumption A-07) is otherwise re-runnable
only at a dead commit, and because a mode
you can only reach by writing a new class is *less* safe than one that names
itself in the run directory.

Building it is refused unless the caller says so in words:

```python
ESTGNN(adjacency, temporal=RawWindow(30))
# ValueError: RawWindow puts 30 numbers on each sensor without learning
# anything across the cycles, so the first graph weight matrix becomes
# (30, 32). Those are multipliers learned over time and shared across
# sensors -- a temporal encoder, inside a layer that is not named one. ...
```

**The model works this out rather than trusting the stage.** It asks whether
the stage holds any parameters:

```python
learns_across_time = any(p.numel() for p in self.temporal.parameters())
self.time_folded_into_graph_layer = self.in_features > 1 and not learns_across_time
```

A stage cannot lie about this, because nothing is asked of it. `LastCycle` and
`MeanWindow` are one number wide, so the answer is no. `LSTMStage` holds
parameters, so the answer is no and correctly so — the graph layer is then
mixing what an encoder produced, not raw cycles. `RawWindow` trips it, and so
does `WindowSummary`, which is three numbers wide and learns nothing; both are
built only when named. So would a hand-written passthrough someone added later.

**Its one honest limit.** A stage that holds parameters *and* still passes
cycles through — a learned per-cycle scaling, say — would get past the check.
The check closes the accident; this page closes the deliberate case.

---

## What it costs to put the paper's stage back

Measured, not estimated. `sum(p.numel() ...)` on the real objects:

| Configuration | Parameters |
|---|---|
| The network alone, Table 4 | 61,729 |
| The network with `window_summary` | 61,793 |
| One `nn.LSTM(input_size=1, hidden_size=32)` | 4,480 |
| One `nn.GRU(input_size=1, hidden_size=32)` | 3,360 |
| `LSTMStage`, fourteen cells at hidden 32, projected to 1 | 62,753 |
| The whole model with that stage | 124,482 |

So the paper's architecture is **roughly twice the size** of the graph stage on
its own, and almost all of the addition is the fourteen dedicated cells. Under
`shared_across_sensors=True` the same stage costs 4,513 instead of 62,753.

It is also about **thirteen times slower per batch**: 22.9 ms against 1.8 ms for
a batch of 32 on this machine, so an epoch over the 14,459 training windows
goes from roughly 0.8 s to roughly 10.4 s.

---

## Worked example

Real readings, few enough to check by hand: FD001 training engine 1, its first
four cycles, and two of its fourteen sensors, T24 (the temperature at the outlet
of the low-pressure compressor) and T50 (the temperature at the outlet of the
low-pressure turbine). Four cycles rather than thirty so that every reading can
be written out. The pipeline builds a four-cycle window when asked for one, and
its four readings are exactly the first four of engine 1's usual thirty-cycle
window.

The readings have been smoothed (assumption A-04) and rescaled: each sensor's
average over the training engines is subtracted, and the result is divided by
that sensor's standard deviation there, the typical size of its departures from
that average (assumption A-03). So −1.0 means one standard deviation below the
training average, not a temperature.

The training engines are the 80 of FD001's 100 that the seed, 42 here, does not
hold back for checking the model during training (assumption A-06), and the
averages and standard deviations are measured on those 80 alone (assumption
A-25). Another seed holds back a different fifth and gives slightly different
numbers: under seed 43, T24's first reading below is −2.032225 rather than
−1.987346. Under seeds 44 and 46 engine 1 is one of those held back, and the
code below fails, because it finds no window of engine 1 among the training
windows.

```python
import numpy as np
import torch
from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.models.temporal import LastCycle, MeanWindow, LSTMStage

prepared = run_preprocessing(PreprocessingConfig(seed=42, window_size=4))
row = int(np.flatnonzero(prepared.train.units == 1)[0])   # engine 1's first window: cycles 1 to 4
names = list(prepared.sensor_symbols)
pair = [names.index("T24"), names.index("T50")]           # [0, 2]
windows = torch.from_numpy(prepared.train.features[row:row + 1, pair])
windows.shape
# torch.Size([1, 2, 4])          one window, two sensors, four cycles
```

`windows` is a PyTorch tensor, an array like a NumPy array that the network's
code works on, of shape `(1, 2, 4)`. Its eight numbers:

| Sensor | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|
| T24 | −1.987346 | −1.603900 | −1.179785 | −0.967728 |
| T50 | −0.995210 | −0.841500 | −0.700499 | −0.770999 |

**`LastCycle` keeps the last column and drops the rest.**

```python
LastCycle(4)(windows)
# tensor([[[-0.9677],
#          [-0.7710]]])          shape (1, 2, 1): cycle 4 of each sensor
```

PyTorch prints four decimal places; the values are T24's −0.967728 and T50's
−0.770999, the last column of the table.

**`MeanWindow` adds up each row and divides by four.**

```python
MeanWindow(4)(windows)
# tensor([[[-1.4347],
#          [-0.8271]]])          shape (1, 2, 1): the average of each sensor
```

```
T24:  (−1.987346 − 1.603900 − 1.179785 − 0.967728) / 4  =  −5.738759 / 4  =  −1.434690
T50:  (−0.995210 − 0.841500 − 0.700499 − 0.770999) / 4  =  −3.308208 / 4  =  −0.827052
```

The two stages hand the graph different numbers from the same window. T24 rises
through all four cycles, so its last reading, −0.967728, sits well above its
average, −1.434690. T50 peaks at cycle 3 and falls back, so its last reading,
−0.770999, is closer to its average, −0.827052. Which of the two a sensor node
carries is therefore a real choice, and this project's (assumption A-07).

**`LSTMStage` learns its own reduction.**

```python
stage = LSTMStage(window_size=4, n_nodes=2, hidden_size=8)
stage(windows).shape
# torch.Size([1, 2, 1])          one number per sensor, from its own network
```

Two recurrent networks, one per sensor, each read their sensor's four readings
in order. `hidden_size=8` keeps the example small (the stage's default is 32,
assumption A-26). The numbers they hand on depend on their random starting
values (assumption A-34) and mean nothing until the stage has been trained, so
only the shape is shown.

The shape is the same in all three cases, and that is the point: whatever the
stage does, the graph layers behind it see one number per sensor unless the
stage says otherwise, and they are told how many by the stage.

`tests/test_examples_torch.py` runs this example and checks every number above
(`test_example_temporal_readme_on_two_real_sensors`).

**Correction, 2026-10-02.** Until this date this example ran on invented
readings, `[[1, 2, 3, 4], [10, 20, 30, 40]]`, which this project does not allow
in a worked example, and the sentence under it said the test file ran it
against the real FD001 windows. No test ran this example: the file's tests use
these stages on real windows elsewhere, but none checked the numbers printed
here. Found while the Resources pages were written and the cited papers checked
against the places that cite them.

---

## Adding another one

Four steps, and the third is the one people forget.

1. Subclass `TemporalStage`. Set `window_size` and `out_features` in
   `__init__`, and implement `forward` for
   `(batch, nodes, window_size) -> (batch, nodes, out_features)`. Call
   `self._check(windows)` first.
2. Add its name to `TemporalStageName` and a branch to
   `build_temporal_stage()`.
3. Add it to `--temporal` in `scripts/train.py`, so that a run using it says so
   in its directory name and in `summary.json`. A stage that cannot be named in
   a stored result is a stage whose numbers cannot be compared with anything —
   rule 3a.
4. Give it a row in the table above, and a worked example in
   `tests/test_examples_torch.py`.

If it departs from what the paper describes, it also needs a numbered entry in
`docs/ASSUMPTIONS.md` and that identifier quoted in the code, in words — rule 4.
