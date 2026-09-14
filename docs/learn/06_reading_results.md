# Reading a result

A prediction is a number of cycles. So is the truth. The error is the
difference between them:

$$d = \hat{y} - y \qquad \text{positive means the model was optimistic}$$

A positive error means the model said the engine had more life left than it
did. That direction is dangerous: it leaves a failing engine flying. A negative
error means the model was pessimistic, and the cost is servicing an engine that
still had life in it. **The two are not equally bad**, and that asymmetry runs
through everything below.

---

## The four measures

### RMSE — root mean squared error

$$\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} d_i^{\,2}}$$

The typical size of the error, in flight cycles. An RMSE of 16 means the model
is typically about 16 cycles out.

Squaring makes it **symmetric**: 10 cycles too high counts exactly like 10 too
low. That is a limitation — it is blind to the very asymmetry that matters most
here — but it is the number nearly every paper quotes, so it is the one that
makes results comparable.

### The NASA score

$$s_i = \begin{cases}
e^{-d_i/13} - 1 & d_i < 0 \quad \text{(early — safe)} \\
e^{\;d_i/10} - 1 & d_i \ge 0 \quad \text{(late — dangerous)}
\end{cases}$$

The official score of the 2008 competition this dataset comes from, summed over
all engines. It is deliberately asymmetric: the divisor for a late prediction
is 10, smaller than the 13 for an early one, so **the same error costs more
when the model was optimistic**.

Three properties to keep in mind. It is exponential, so one badly wrong engine
can outweigh ninety-nine good ones. It is a **sum**, so it grows with the size
of the test set and can only be compared between runs on the same sub-dataset.
And lower is better, with 0 perfect.

> The paper's Equations 20 and 21 contradict each other on the sign — one of
> them would reward late predictions. This project follows the NASA convention,
> which is what the paper's own surrounding text describes. That is assumption
> A-15.

### R² — coefficient of determination

$$R^2 = 1 - \frac{\sum_i (\hat{y}_i - y_i)^2}{\sum_i (y_i - \bar{y})^2}$$

The fraction of the variation in the true labels the model accounts for. 1 is
perfect; **0 is exactly as good as always guessing the average**; negative is
worse than that. It is the measure that tells you whether a model has learned
anything at all, as opposed to how precise it is.

> The paper's Equation 23 puts the predicted values in the denominator, which
> is not the standard definition and gives a different quantity. This project
> uses the standard one — assumption A-16.

### ASUE — average safe underestimation error

$$\mathrm{ASUE} = \frac{1}{N_{\text{under}}}
\sum_{i\,:\,\hat{y}_i < y_i} \max\left(0,\; y_i - \hat{y}_i\right)$$

The average shortfall, counted **only over the engines the model was
pessimistic about**. Engines it was optimistic about do not contribute at all.

This is the cost of caution: how much life, on average, is being thrown away on
the engines that get serviced early. A model can look good on RMSE by being
uniformly cautious, and ASUE is what shows the price of that.

---

## One seed is not a result

Train the same network twice with different random seeds and you get two
different numbers. The weights start from random values, the batches arrive in
a random order, and — in this project — the choice of which engines are held
back varies too.

So **anything quoted as a result here is a mean over at least five seeds, with
its standard deviation**:

```bash
python scripts/train.py --seeds 5
```

The script says so when it is given only one. On FD001 under the current
default the reproduction gives:

| | RMSE | NASA score | R² | ASUE |
|---|---|---|---|---|
| Current default, 5 seeds | 16.82 ± 0.16 | 610 ± 25 | 0.824 ± 0.003 | 12.33 ± 0.60 |

That ± is not decoration. It is what tells you whether a difference between two
configurations means anything.

The varying split matters here. This project's seed changes the train/validation
division as well as the initialisation, so the spread is what a genuine
repetition of the whole experiment would show. There is a `--fixed-split` flag
that holds the split constant; it produces a smaller spread that **understates**
the real variability, and any number produced with it must say so.

---

## When is a difference real?

Before saying one configuration beats another, test whether the difference
survives the seed-to-seed spread.

This project has done that, and the results are a lesson in why it matters.

**How the sensor graph is rescaled.** Symmetric against row normalisation. The
difference turned out to be 0.35 cycles with a paired *p* of 0.047 — real, but
small, and it does not survive holding engines out. A settled question, and the
answer is "barely" (assumption A-21).

**What the paper's physics graph is worth.** This one has reversed twice. Set
against a graph joining every sensor to every other, it first read 5.7 cycles,
was withdrawn and re-measured as **0.044 cycles** (*p* = 0.913 —
indistinguishable from zero), then read **2.396 cycles** with nothing held back
(*p* = 0.000002, every one of the five seeds agreeing), and reads **2.784** under
the current default (finding F-04).

Nothing about the graph changed between those measurements. The preprocessing
and the split changed — and the paper specifies neither. Two things moved
between the zero result and the nothing-held-back one, the input scaling and the
split, so these runs do not isolate which caused the reversal, and the log says
so rather than guessing.

Then the question itself turned out to be badly posed. Joining every sensor to
every other makes every sensor receive the same average, so it removes far more
than the engineering knowledge. A graph built from the data, with the same 41
edges the paper's has, costs **0.096 cycles** (*p* = 0.655): nothing that five
seeds can see (assumption A-27). The comparison you choose decides the answer,
which is the rule below from another direction.

---

## The rule that catches people out

> **A comparison is a fact about one whole configuration — architecture,
> preprocessing and split together — not about the two options being
> compared.**

All three of this project's configuration comparisons were re-measured after a
correction to the architecture in August 2026, and **two of them reversed
direction**. Not "shrank" — reversed. F-04 has since reversed a second time,
under a change to the preprocessing and the split with the architecture held
fixed.

So a comparison must always name the configuration it was made under, and must
be re-run whenever anything moves. What the physics graph is worth under one
configuration tells you nothing reliable about what it is worth under another —
and that, rather than any single number, is the result worth reporting.

The **Stored runs** page on this site lists every run this repository has kept,
each with the commit that produced it, precisely so that a number can always be
traced back to the code that made it.

---

## What may and may not be compared with the paper

The paper reports several variants. Only one of them is comparable with this
code, because the LSTM stage is deliberately absent here:

| Paper's variant | RMSE | Comparable? |
|---|---|---|
| ESTGNN, full model | 11.78 | **No** — includes the LSTM stage |
| ESTGNN w/o LSTM | 14.93 | **Yes** — this is the matching row |
| ESTGNN w/o GCN | 16.42 | No |
| ESTGNN w/o physics prior | 12.34 | No |

And even that comparison needs three warnings attached:

1. **The settings were chosen to land near the target.** The options the paper
   leaves open were selected under a rule — pick the one closest to the paper's
   own figure — which means the configuration is not independent of the
   quantity it is being compared against. Under the previously chosen settings
   the same code gives 20.35 ± 0.75. The paper specifies neither, so the honest
   summary is that this reproduction **spans 16.50 to 20.35 cycles depending on
   choices the source leaves open**. The rule is written up in
   `docs/ASSUMPTIONS.md` (assumption A-22), and it should be read before quoting
   any single number.
2. **Capping the test labels flatters the result** by a measured amount, so the
   underlying gap is wider than it appears (A-02).
3. **The paper's numbers are single runs** with no seeds and no error bars, so
   they have an unknown spread of their own.

---

## What you now know

- Four measures: **RMSE** (typical error, symmetric), the **NASA score**
  (asymmetric, punishes optimism, a sum), **R²** (fraction of variation
  explained, 0 = guessing the mean) and **ASUE** (the cost of caution).
- Two of the four had to be resolved against contradictions in the paper's own
  equations (A-15, A-16).
- **One seed is not a result.** Five seeds, mean and standard deviation.
- A **comparison is a fact about one architecture** — two of this project's
  three comparisons reversed when the architecture was corrected.
- Only the paper's **"w/o LSTM"** row is comparable, and even then three
  caveats apply.

Next, and last: how this project keeps the paper, the choices and the
measurements from being mistaken for each other.
