# The pipeline: three text files become training examples

{{figure:pipeline}}

Eight steps stand between the files NASA distributes and the arrays a network
can be trained on. Three of them are where **leakage** can happen: information
the model would not have in real use — about the test engines, or from cycles
later than the one it predicts at — reaching it anyway. A leak produces
excellent numbers that mean nothing. Every shape and count below was measured
by running the pipeline on the real FD001 files with the project's default
settings.

---

## 1 — Load

Three plain text files are read: the training engines, the test engines, and
the true remaining life of each test engine at its cut-off point. They are
space-separated, with no header row; the 26 columns are the engine number, the
cycle number, three operating settings and 21 sensors.

The remaining life at every training row is worked out here by counting
backwards from each engine's last cycle, and added as a 27th column.

**After this step: 20 631 training rows, 27 columns.**

## 2 — Diagnose

Before anything is thrown away, each sensor is checked for whether it carries
any information. Two tests: does it ever change at all, and does it take at
least three distinct values?

This step **only reports**. It exists so the paper's selection can be checked
against the data rather than taken on trust — and it is what turned up that
`P15` takes only two distinct values yet is kept by the paper (finding F-01).

## 3 — Select

The paper's 14 sensors are kept, **in the exact order its Table 3 uses**.

The order matters because the sensor graph is stored as a plain grid of numbers
whose rows are matched to sensors by position alone: reordering the list would
quietly connect the wrong sensors to each other and leave everything apparently
working.

**After this step: 14 sensor columns instead of 21.**

{{figure:correlation}}

The paper's Fig. 5 prints the correlation between every pair of eighteen FD001
columns: the engine number, the cycle, the two operating settings that vary,
and the fourteen kept sensors. Its Section 3.2.2 says the selection removed
columns that never change and columns correlating above 0.95 with another. In
the figure above, recomputed from the training file, the most strongly
correlated pair of kept sensors is `Ps30` and `phi` at −0.85 (the minus means
one falls as the other rises), the same value the paper prints for that cell and
well short of the 0.95 limit.

## 4 — Cap the labels

The piecewise-linear label from page 1 is applied: each label is the smaller of
the cycles really left and 125, so it holds at 125 until the last 125 cycles and
then falls by one per cycle.

The same ceiling is applied to the **test** labels, and that moves the reported
numbers in the model's favour: the network cannot predict above 125 (the next
page shows why), so leaving a test label at 140 would charge it for an error it
cannot avoid. The cap is this project's choice, and it must be stated whenever
these numbers are compared with published ones (assumption A-02).

## 5 — Smooth *(leakage-critical)*

Sensor readings are noisy. The noise is damped with an **exponentially weighted
moving average**, a filter in which each smoothed value is a blend of the
current reading and the previous smoothed value:

$$S_t = \alpha X_t + (1 - \alpha) S_{t-1}, \qquad S_1 = X_1$$

Here $X_t$ is the raw reading at cycle $t$, $S_t$ is its smoothed value, and
$\alpha$, between 0 and 1, is the share of the blend given to the new reading.
With $\alpha = 0.5$, engine 1's `T50` reads 1400.60 at cycle 1 and 1403.14 at
cycle 2, so $S_1 = 1400.60$ and $S_2 = 0.5 \times 1403.14 + 0.5 \times 1400.60 = 1401.87$.

{{figure:smoothing}}

Over all 100 training engines, the filter removes **59.0 %** of the median
sensor's jitter, measured as the average size of the jump from one cycle's
reading to the next.

Two things make this step safe, and both are enforced by tests:

- **The filter only looks backwards.** Every smoothed value depends on the
  current reading and earlier ones, never a later one. A filter that also
  averaged forwards would carry an engine's future into its present.
- **Each engine is filtered separately.** Running one filter across the
  concatenated data would blend the end of one engine's life into the start of
  the next.

The value 0.5 is this project's reading of the paper, whose sentence attaches it
to the previous *smoothed value* instead. That value changes at every cycle and
so cannot be one number; reading 0.5 as $\alpha$ is the only way the sentence
means anything (assumption A-04).

## 6 — Split *(leakage-critical)*

A fifth of the training engines are **held back**: kept out of training, so
that while the model trains it can be scored on engines it has not learned
from. They are chosen **here**, before any statistic has been computed from the
data, under two rules that each have a test:

- **Split by engine, never by window.** Two neighbouring windows (the 30-cycle
  slices of step 8) from one engine share 29 of their 30 cycles. Splitting
  windows at random would put nearly identical examples on both sides, and the
  held-back score would measure memorisation rather than how the model copes
  with an engine it has never seen.
- **Split before fitting anything.** A held-back engine must never influence
  the scale of a sensor.

The paper describes no held-back engines and trains for a fixed 200 passes over
the data, so holding engines back, keeping the weights from the pass that
scored best on them, and stopping once that score stops improving are all
additions to it. Over five runs with different random seeds, together they
lower the error on the test engines by 0.32 cycles, which the run-to-run spread
cannot distinguish from zero (*p* = 0.234; assumptions A-06 and A-18).

## 7 — Rescale *(leakage-critical)*

Sensors live on very different scales: `T50`, a temperature, reads near 1400
degrees Rankine, while `BPR`, a ratio, reads near 8. Without rescaling, the
temperature would dominate only because its numbers are larger.

The rule that protects the experiment is that **every statistic is learned from
the 80 engines the model trains on** and then applied unchanged to the
held-back and test engines. A mean or a spread computed over the test engines
would carry information about them into training.

The paper **never describes its input scaling** — not the method, not where the
statistics are fitted. This project uses the **z-score**: subtract the sensor's
mean, then divide by its standard deviation (the typical distance of a reading
from that mean), so each reading becomes its distance from the mean counted in
standard deviations (assumption A-03). It was chosen by this project's rule for
settings the paper leaves open: take whichever option lands nearest the paper's
reported test error (assumption A-22). Against min-max scaling, which squeezes
each sensor into the range 0 to 1, it lowered the test error by 3.83 cycles,
towards the paper's figure, over five seeds under the project's earlier settings
for the graph and the learning rate: the largest effect measured for any such
setting (finding F-10).

## 8 — Window

Engine histories have different lengths, and a network needs a fixed input
size. So each history is cut into overlapping slices of 30 consecutive cycles.

{{figure:windowing}}

Engine 1, the one page 1 follows, ran 192 cycles. Its first window covers
cycles 1 to 30, the next 2 to 31, and the last 163 to 192:
192 − 30 + 1 = **163 examples**. Across all 100 training engines that makes
**17 731 examples**.

The window length of 30 comes from the paper's Table 6. How far the window
moves between examples is not in the paper's text; its Fig. 1 labels the slide
"t=1s", which this project reads as one cycle, since the data holds one reading
per cycle and no seconds (assumption A-05).

Test engines get exactly **one** window each — the last 30 cycles before the
cut-off — because there is one true label per test engine: 100 in all.

---

## What the model receives: one number per sensor, not thirty

Each of the 17 731 examples leaves the pipeline with shape **(14, 30)**:
fourteen sensors, thirty cycles each. The **first graph layer**, as the paper's
Table 4 prints it, takes **(14, 1)**: one number per sensor. Turning thirty
cycles into one is a decision about the model, so the model makes it, in a stage
the next page describes.
