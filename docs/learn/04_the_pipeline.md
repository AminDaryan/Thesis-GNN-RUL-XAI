# The pipeline: three text files become training examples

{{figure:pipeline}}

Eight steps stand between the files NASA distributes and the arrays a network
can be trained on. Their order is not free. Three of them, shown in orange
above, decide whether the experiment is valid at all — get those wrong and the
model will report excellent numbers that mean nothing.

Below, each step with what it does and the shape the data has afterwards. Every
shape was measured by running the pipeline on the real FD001 files with the
project's default settings.

---

## 1 — Load

Three plain text files are read: the training engines, the test engines, and
the true remaining life of each test engine at its cut-off point. They are
space-separated, with no header row; the 26 columns are the engine number, the
cycle number, three operating settings and 21 sensors.

The remaining life at every training row is worked out here by counting
backwards from each engine's last cycle.

**After this step: 20 631 training rows, 27 columns.**

## 2 — Diagnose

Before anything is thrown away, the data is asked which sensors carry
information. Two tests: does the sensor ever change at all, and does it take at
least three distinct values?

This step **only reports**. It never changes which sensors are used. It exists
so the paper's selection can be checked against the data rather than taken on
trust — and it is what turned up finding F-01, that `P15` takes only two
distinct values yet is kept by the paper.

## 3 — Select

The paper's 14 sensors are kept, **in the exact order its Table 3 uses**.

That order is load-bearing and must never change. The sensor graph is stored as
a plain grid of numbers whose rows are matched to sensors by position alone, so
reordering the list would quietly connect the wrong sensors to each other and
leave everything apparently working. The code asserts the order the moment it
is imported.

**After this step: 14 sensor columns instead of 21.**

## 4 — Cap the labels

The piecewise-linear ceiling from page 1 is applied: while the engine is
healthy the label is held at 125, and it falls by one per cycle after that.

The same ceiling is applied to the **test** labels as well. That is worth
stating plainly, because it moves the reported numbers in the model's favour.
The network's final layer is a sigmoid multiplied by the ceiling, so it is
structurally incapable of predicting more than 125; leaving a test label at 140
would charge the model for an error it cannot avoid. Applying the cap is
assumption A-02, its effect has been measured, and the fact that it flatters the
numbers must be repeated whenever they are compared with published ones.

## 5 — Smooth *(leakage-critical)*

Sensor readings are noisy. The noise is damped with an **exponentially weighted
moving average**, a filter in which each smoothed value is a blend of the
current reading and the previous smoothed value:

$$S_t = \alpha X_t + (1 - \alpha) S_{t-1}, \qquad S_1 = X_1$$

{{figure:smoothing}}

Measured over all 100 training engines, the median sensor loses **59.0 %** of
its cycle-to-cycle jitter while the downward trend survives intact. Thirteen of
the fourteen sensors land between 56.9 % and 59.1 %. The fourteenth is `P15` at
49.4 % — the two-level sensor from page 1, behaving differently here as well.

Two things make this step safe, and both are enforced by tests:

- **The filter only looks backwards.** Every smoothed value depends on the
  current reading and earlier ones, never a later one. A filter that averaged
  forwards and backwards would pull information from an engine's future into
  its present — which is exactly the kind of mistake that produces excellent,
  meaningless results.
- **Each engine is filtered separately.** Running one filter across the
  concatenated data would blend the end of one engine's life into the start of
  the next.

The strength $\alpha = 0.5$ is assumption A-04. The paper's own sentence on
this is self-contradictory — it attaches the number 0.5 to the previous
*smoothed value* rather than to the smoothing parameter — and reading it as the
parameter is the only interpretation under which the sentence means anything.

## 6 — Split *(leakage-critical)*

If any engines are to be held back for validation, they are chosen **here**,
before any statistic has been computed from the data.

Two rules govern this, and each has a test:

- **Split by engine, never by window.** Two neighbouring windows from one
  engine share 29 of their 30 cycles. Splitting windows at random would put
  nearly identical examples on both sides of the divide, and the validation
  score would measure memorisation rather than generalisation.
- **Split before fitting anything.** A held-out engine must never influence the
  scale of a sensor.

Since 2026-09-02 this project holds back **20 %** of the training engines, which
reverses a decision taken on 2026-08-29 to hold nothing back. The paper
describes no held-out set and trains a fixed 200 epochs, so holding engines out
— and restoring the best-validation checkpoint, and stopping early — are all
additions to it. Measured over five seeds they are worth 0.32 cycles, which the
seed-to-seed spread cannot distinguish from zero (*p* = 0.234). See A-06, A-18
and A-22.

## 7 — Rescale *(leakage-critical)*

Sensors live on wildly different scales: a temperature near 1400 degrees
Rankine and a ratio near 8. Without rescaling, the temperature dominates
everything simply by being a larger number.

The rule that protects the experiment is that **every statistic is learned from
the training engines only** and then applied unchanged to validation and test
engines. A mean or a standard deviation computed over the test set is
information from the future leaking backwards.

The paper **never describes its input scaling at all** — not the method, not
where the statistics are fitted. This project uses z-scores, chosen under the
rule recorded in A-22, and the choice is worth 3.83 cycles against the
per-condition min-max scaling used by a closely related paper. That makes it
the largest measured effect of any single choice in the project, and it is
recorded as finding F-10: the input scaling the paper never describes accounts
for most of the remaining gap to its reported figure.

## 8 — Window

Engine histories have different lengths, and a network needs a fixed input
size. So each history is cut into overlapping slices of 30 consecutive cycles.

{{figure:windowing}}

Engine 1 ran 192 cycles, so a 30-cycle window moving one cycle at a time yields
**163 examples from that engine alone**. Across all 100 training engines the
pipeline produces **17 731 examples**.

The window length of 30 comes from the paper's Table 6. How far the window
moves between examples does not appear in the text; a stride of one cycle is
assumption A-05.

Test engines get exactly **one** window each — the last 30 cycles before the
cut-off — because there is one true label per test engine. That gives 100 test
examples for 100 test engines.

---

## The shape that surprises people

The pipeline hands over shape **(17 731, 14, 30)** — every cycle of every window.
What reaches the **first graph layer** is **(17 731, 14, 1)**: fourteen sensors,
and **one number each**.

Not thirty. One.

Those are two different statements, and the difference is the point. The reduction
from thirty cycles to one number is a decision about the **model**, so it is made
in the model, by a named stage in `models/temporal/`. It used to be made here,
in the preprocessing, and that is what allowed the mistake described below.

This follows from the paper's own Table 4, which prints an input of
`[batch, 14, 1]` and a first weight matrix in $\mathbb{R}^{1 \times 32}$. In
the paper's full method that single number per sensor is produced by a small
recurrent network that reads the 30-cycle window and summarises it. This
project does not implement that stage — see the next page — so each sensor node
carries **the last cycle of its window** instead, which keeps Table 4's shape
exactly. That is assumption A-07.

It is worth dwelling on, because getting it wrong is easy and expensive. An
earlier version of this project let each node carry all 30 cycles. That made the
first weight matrix `(30, 32)` instead of `(1, 32)` — 928 parameters the
paper's architecture does not have, computing a bank of filters over time. It
scored better. It was also, quietly, a different model: a temporal encoder,
which is exactly what omitting the recurrent stage was supposed to avoid. The
mode was removed and every affected number re-measured, and two of the
project's three configuration comparisons **reversed direction** as a result.

---

## The whole chain, with the counts filled in

The strip at the top of this page names the eight steps. This one is the same
chain with the data in it: what goes in, what comes out, how many rows and
engines survive each step, and where the training and test sets stop being
treated the same way.

{{figure:notebook-map}}

Every number in it was measured by running the pipeline while this page loaded,
so it cannot fall behind the code. It is also the picture that opens
`notebooks/01_preprocessing_fd001.ipynb`, which walks the same eight steps one
cell at a time and checks each one.

---

## What you now know

- Eight steps: load, diagnose, select, cap, smooth, split, rescale, window.
- Three of them are **leakage-critical**, and each is guarded by a test: the
  filter looks only backwards, the split happens before any statistic is
  learned, and the scaling is fitted on training engines alone.
- The sensor **order** is load-bearing, because the graph is indexed by
  position.
- Smoothing removes **59.0 %** of the median sensor's jitter; windowing turns
  100 engines into **17 731 examples** of shape **(14, 30)**, which the model's
  temporal stage reduces to the **(14, 1)** Table 4 prints.
- Four judgement calls live in this chain: the smoothing strength (A-04), the
  stride (A-05), the test-label cap (A-02) and the entire input scaling (A-03,
  measured in F-10).

Next: the network those arrays are fed into.
