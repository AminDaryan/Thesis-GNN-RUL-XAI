# Temporal algorithm

Preprocessing supplies a window of consecutive readings for each sensor. The
temporal algorithm decides what to keep from that sequence before the graph
combines information between sensors. Time and connections do different jobs:
this stage reads cycles **within one sensor**; the graph then combines the
numbers carried by **different sensors**.

## What enters and leaves

For a 30-cycle window, one example enters as `[14, 30]`: fourteen sensor rows
and thirty cycle columns. The stage returns `[14, k]`, where `k` is the
number of values it keeps or computes per sensor. A batch adds a first
dimension counting how many windows are processed together.

| Playground choice | What each sensor passes to the graph | Values per sensor |
|---|---|---|
| Last cycle | Its latest reading in the window | 1 |
| Mean | The average of its readings in the window | 1 |
| Window summary | Last reading, average and change across the window from a fitted straight line | 3 |
| Full window | All readings, in cycle order | Window length |
| LSTM | Values computed by a learned memory-based sequence model | Chosen output width |
| GRU | Values computed by a different memory-based sequence model | Chosen output width |

The non-recurrent choices calculate fixed summaries; they do not learn new
multipliers. LSTM and GRU have multipliers adjusted during training, so
changing the temporal algorithm changes the model that must be trained.
The [implementation guide](../../src/thesis_gnn_rul_xai/models/temporal/README.md)
records the choices and their assumptions.

[Follow one real example](08_worked_examples.md#through-the-saved-model)
shows the input and output numbers for the Last cycle stage of a named saved
network. Keep that example's temporal choice in mind: its prediction is not
a demonstration of a network trained with LSTM.

## The stage left out by default

The per-sensor stage of the paper's Section 2.2 is a **long short-term memory**
network, or **LSTM**: a network built for sequences, which reads a window one
cycle at a time and keeps an internal memory of what it has already seen.

{{figure:lstm-cell}}

Three **gates** — each made of numbers between 0 and 1, produced by passing a
weighted sum of the new reading and the last output through a sigmoid — decide
how much of the old memory to keep, how much of a freshly proposed value to add
to it, and how much of the result to pass on as this cycle's output. The memory
line along the top is what lets the cell carry a slow trend across thirty
cycles without it being overwritten at every step.

The stage is built (`--temporal lstm`) but is not the default (assumption
A-11), because the paper never states its size, its depth, or how its memory
becomes the single number Table 4 requires. The version here fills those in
(assumption A-26), so a result from it says what those choices do, not what the
paper's stage did.

The paper's Table 11 scores its method with one part removed at a time. On
FD001, removing the LSTM raises its error from 11.78 to 14.93 cycles, a
difference of 3.15 that the table prints.

What the stage is worth here depends on how the model is trained (finding
F-16). With this project's procedure — a fifth of the engines held back, the
weights that scored best on them kept — it lowers the FD001 test error by 3.19
cycles, close to that 3.15. With the procedure the paper describes — nothing
held back and a fixed 200 epochs, so the weights from the last one — it lowers
it by 0.77, which five seeds cannot tell apart from nothing.

The procedure matters far more to the LSTM version. Its error on the held-back
engines was lowest between epochs 9 and 21, and its weights after 200 epochs
score much worse on the test engines, so holding engines back and keeping the
best weights is worth 2.75 cycles to it, against 0.32 to `LastCycle`. The paper
never mentions keeping the best weights.
