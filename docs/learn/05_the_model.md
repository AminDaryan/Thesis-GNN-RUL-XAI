# The model, layer by layer

## Before Table 4: the stage the paper does not tabulate

The paper's Table 4 begins with an input of shape `[batch size,14,1]`: for each
of the examples processed together, one number per sensor, where the previous
page ended with thirty cycles per sensor. The paper's Section 2.2 says what
closes that gap: fourteen dedicated recurrent networks, one per sensor, each
reading its sensor's window one cycle at a time. Table 4 gives that stage **no
row at all**.

So the network has three parts, and the paper tabulates the last two:

```
window of 30 cycles  ->  temporal stage  ->  graph layers  ->  fully connected
     (14, 30)               (14, 1)            (14, 32)          one number
```

By default the temporal stage is `LastCycle`: keep the most recent reading,
drop the other 29, learn nothing (assumption A-07). The paper's own stage is
built too; [Temporal algorithm](09_temporal.md) explains that choice. The layers below
are the part Table 4 does print, and they are the same either way.

---

## The six layers, and how many numbers each learns

{{figure:architecture-overview}}

The figure is drawn from the network this code builds with the default stage,
so its sensor-input column shows the last reading of each sensor, without the
memory-based LSTM stage. Fourteen sensors
arrive carrying one number each. Three rounds of message passing — each
sensor's value mixed with its neighbours', as on page 2 — grow that from 1
number per sensor to 32, then 64, then back down to 32. The 14 × 32 grid is
flattened into one long vector of 448 numbers, and two **fully connected**
layers, in which every output is a weighted sum of every input plus a constant
called a bias, reduce that to a single output. Each graph layer and the first
fully connected layer use ReLU, which turns negative values into zero. The
final layer uses the sigmoid described below.

The **parameters** are the numbers the network adjusts during training: the
learned numbers the figure counts beneath each stage. A graph layer holds one weight
matrix, shared by all fourteen sensors: GCN-2 turns 32
numbers per sensor into 64, so it holds 32 × 64 = 2 048 weights. A fully
connected layer holds a weight for every input–output pair plus one bias per
output: FC-1 holds 448 × 128 + 128 = 57 472. Table 4 prints the five
weight-matrix shapes and no total; they come to
1 × 32 + 32 × 64 + 64 × 32 + 448 × 128 + 128 × 1 = 61 600. The biases of the
two fully connected layers, which the paper's Equations 12 and 13 print, add
128 + 1 = 129, for 61 729 in all; the graph layers get no bias, because
Equations 8 to 10 print none (assumption A-12).

**The graph layers are tiny.** Together they hold 4 128 of the 61 729
parameters, under 7 %; FC-1 alone holds over 93 %. That is worth remembering
when reading any claim about what the graph contributes.

---

## The last layer, and what it makes impossible

The last layer ends in a **sigmoid**, a function that squashes any number into
the range 0 to 1: an input of 0 gives 0.5, and an input of 10 gives 0.99995.
That result is multiplied by the label ceiling of 125 to give an answer in
flight cycles — 62.5 and 124.99 in those two cases.

So the network is **structurally incapable of predicting more than 125
cycles**, no matter what it is trained on or for how long. Two consequences
follow:

- The **training targets are divided by the ceiling** so they lie in 0 to 1,
  where the sigmoid can reach them. The paper says its sigmoid "ensures
  normalized RUL values within [0,1]" without saying normalised by what, so the
  division is filled in here (assumption A-13).
- A test, `test_the_prediction_can_never_exceed_the_ceiling`, enforces the
  property, so it cannot be broken by accident.

---

## Why the shapes are checked so aggressively

The layers written for this project check the shape of what they are given,
and refuse rather than reshape. The sensor graph is refused too if it has a
zero anywhere down its diagonal. A zero there means that sensor is not joined
to itself, so the mixing step of page 2 would build its new value from its
neighbours alone and ignore its own reading. `normalize_adjacency()`, the
function that applies the normalisation of page 2, adds that join first, so a
zero on the diagonal means the graph never went through it.

The reason for the strictness: **a wrong answer here still looks plausible.** A
model that predicts 60 cycles when the truth is 40 raises no error and looks
fine on a plot, so the defence is to refuse impossible inputs and to test
properties that must hold, not just the numbers that come out. One such test,
`test_parameter_counts_match_table_4`, requires every layer to hold exactly the
count worked out above, and the whole network 61 729, so a result cannot come
from a network of a different size from the paper's.

---

## Training

The training settings come from the paper's Table 6:

- **mean squared error** as the loss: the average of the squared gaps between
  prediction and label, the number training tries to make small;
- the **Adam** optimiser, the rule that decides how each weight changes after
  a batch;
- **batches of 32**: the weights change after every 32 examples;
- **200 epochs** — an epoch is one pass over all the training examples;
- a **learning rate**, which scales the size of each change, starting at 1e-3
  and ending at 1e-4.

The paper gives both ends of that rate but not how it gets from one to the
other. This project drops it once, tenfold, at epoch 100, halfway through the
paper's 200, so it never uses a rate the paper does not print. Shrinking it
smoothly, by the same fraction every epoch, would fit the paper's words too, but
none of the three published studies on this dataset found to state a schedule
does that: they lower the rate in steps, or when the training loss stops
falling
(assumption A-17).

Two further gaps are filled here. The graph layers' weights start at random
values drawn from a range set by each layer's size, as in the paper that
introduced this layer type (assumption A-14). And where the paper trains for a
fixed 200 epochs, this project stops once the error on the held-back engines
has not improved for 20 epochs, then keeps the weights from the epoch that
scored best on them (assumption A-18). When the default settings were trained
with five seeds, every run stopped by epoch 75, so none reached the drop to
1e-4.

---

## The stage left out by default

The [Temporal algorithm page](09_temporal.md#the-stage-left-out-by-default)
now explains the memory-based stage, its settings and the distinction between
this project's default and its recurrent alternative.
