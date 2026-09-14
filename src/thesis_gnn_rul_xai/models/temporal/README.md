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
| `raw_window` | every cycle, untouched | no | 30 |
| `lstm` | what a recurrent network read out of the window | yes | 1 by default |
| `gru` | the same, with a simpler recurrent cell | yes | 1 by default |

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
mixing what an encoder produced, not raw cycles. `RawWindow` is the only thing
that trips it, and so would a hand-written passthrough someone added later.

**Its one honest limit.** A stage that holds parameters *and* still passes
cycles through — a learned per-cycle scaling, say — would get past the check.
The check closes the accident; this page closes the deliberate case.

---

## What it costs to put the paper's stage back

Measured, not estimated. `sum(p.numel() ...)` on the real objects:

| Configuration | Parameters |
|---|---|
| The network alone, Table 4 | 61,729 |
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

```python
import torch
from thesis_gnn_rul_xai.models.temporal import LastCycle, MeanWindow, LSTMStage

windows = torch.tensor([[[1.0, 2.0, 3.0, 4.0],      # sensor 0, four cycles
                         [10.0, 20.0, 30.0, 40.0]]])  # sensor 1

LastCycle(4)(windows)
# tensor([[[ 4.], [40.]]])       the last cycle of each sensor

MeanWindow(4)(windows)
# tensor([[[ 2.5], [25.0]]])     (1+2+3+4)/4 and (10+20+30+40)/4

stage = LSTMStage(window_size=4, n_nodes=2, hidden_size=8)
stage(windows).shape
# torch.Size([1, 2, 1])          one number per sensor, from its own network
```

The shape is the same in all three cases, and that is the point: whatever the
stage does, the graph layers behind it see one number per sensor unless the
stage says otherwise, and they are told how many by the stage.

`tests/test_examples_torch.py` runs this against the real FD001 windows.

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
