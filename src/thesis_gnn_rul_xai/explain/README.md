# `explain/` — which sensor drove this prediction, and when

This folder is the "X" of the project's XAI. It takes a trained ESTGNN (the
graph neural network this repository re-implements, which predicts an engine's
remaining useful life from 14 sensors) and works out how much each sensor
contributed to a given prediction.

> **The `.py` files here carry one line of description each, and nothing more.**
> All of the explanation lives in this file. The section
> [What each name means](#what-each-name-means) at the end lists every function
> and what it gives you back.

It needs PyTorch, which is optional in this project. Install it with the
`model` extra:

```bash
pip install -e ".[model]"
```

## Files

- `shap.py` — everything: the coalition weights, the baseline, the Shapley
  values themselves, the per-cycle version, and the check that the numbers add
  up.
- `__init__.py` — re-exports the public names from `shap.py`.

## What a Shapley value is

The model gives one number, and fourteen sensors were involved in producing it.
"How much did each sensor contribute?" has no obvious answer, because the
sensors are not independent: removing one changes what the others are worth.

Cooperative game theory gives an answer. Imagine building the group of sensors
one at a time, in some order. Each time a sensor joins, the prediction changes
by some amount, and that change is what the sensor contributed *in that order*.
Do that for every possible order and average. That average is the sensor's
**Shapley value** (Shapley, 1953). It is the only way of splitting the credit
that satisfies a short list of fairness conditions at once.

Written out, for sensor `i`:

```
phi_i = sum over every subset S not containing i of
            |S|! (n - |S| - 1)! / n!  *  ( f(S + {i}) - f(S) )
```

That is Equation 15 of the paper this project re-implements (Wen et al., 2026).
The fraction is the share of orderings in which exactly the sensors of `S` come
before `i`. `coalition_weights(n)` returns those fractions, one per subset size.

## This implementation is exact, and most are not

The sum runs over every subset of the other sensors. With 14 sensors that is
2^14 = 16,384 subsets, which a computer can simply enumerate. Most published
SHAP work cannot: with 100 features there are more subsets than atoms in the
observable universe, so it approximates, either by sampling orders or by fitting
a weighted linear model (Lundberg & Lee, 2017).

Here no approximation is needed. Every number this folder returns is the
Shapley value as Equation 15 defines it, not an estimate of it. That is worth
stating in the thesis, because it removes a whole class of doubt: a
disagreement with the paper cannot be blamed on sampling noise.

The cost is `2 ** nodes` model evaluations per window — 16,384 for this
project's 14 sensors. `MAX_EXACT_NODES` is set to 20, above which enumeration
stops being reasonable (2^20 is about a million evaluations per window).
Ask for more nodes than that and `shap_values` raises instead of quietly
switching to an approximation. This module deliberately does not provide one.

## What "removing a sensor" means here

Equation 15 needs `f(S)`, "the ESTGNN prediction when only sensors in subset S
are considered". A trained network has no way to accept thirteen inputs; it
always needs fourteen. So a sensor is never actually removed. It is **replaced**
by an uninformative stand-in, and the choice of stand-in decides what the
resulting numbers mean.

The paper does not say what stand-in it uses. We replace an absent sensor's
reading with that sensor's average across the training windows, so the model
sees "this sensor is behaving typically" rather than some arbitrary value. That
choice is recorded as A-20 in `docs/ASSUMPTIONS.md`.

**Why zero would be the obvious alternative, and why that is a trap.** The
temptation is to feed a sensor 0 when it is left out. Whether that is safe
depends entirely on the rescaling, not on the baseline:

- Under the **min-max** scaling that was the default until 2026-08-29, zero was
  a sensor's lowest ever reading. Substituting it produced a ranking unrelated
  to the mean-window one (Spearman −0.16) and promoted `P15`, the least
  informative sensor, to first place.
- Under the **z-score** default now in force, zero *is* the training mean, so
  it agrees with the mean window almost exactly (Spearman +0.996).

A-20 records both measurements. The point it draws from them is that a SHAP
baseline cannot be justified on its own — only against the scaling in force —
and the mean window is kept because it is the one choice defensible under
either. Wen et al. state neither their baseline nor their scaling.

So a Shapley value here answers a specific question: *how much did knowing this
sensor's actual readings, rather than a typical engine's, change the
prediction?* The unit is flight cycles. A value of `+4` means this sensor's
actual readings pushed the predicted remaining life four cycles higher than a
typical engine's readings would have.

**Build the baseline from the training windows only.** `mean_window_baseline`
takes whatever dataset you hand it. Handing it validation or test windows leaks
information into the explanation, in the same way and for the same reason it
would leak into the scaling.

## Dynamic: the same calculation at every cycle

Equation 16 of the paper is Equation 15 applied to the window ending at each
cycle in turn. An engine that ran for 220 cycles yields 191 windows at a window
size of 30, so it yields 191 sets of 14 Shapley values — a table of how the
model's attention moved as the engine wore out. `dynamic_shap` produces that
table for one engine.

Give it the training or validation set. Those keep every window of every
engine. The test set holds only one window per engine, so it cannot show a
trajectory.

## Trap: our table and the paper's Table 12 are not comparable

The paper's Table 12 has the same *shape* as what `dynamic_shap` returns — one
engine, sensors down the side, time steps across, though its columns are
labelled `G1`, `G2` and so on instead of by cycle.

The *contents* are not comparable. Our values are signed and measured in flight
cycles. The paper's cannot be: finding F-08 in `docs/ASSUMPTIONS.md` shows that
Table 12's contents cannot be signed Shapley values satisfying the paper's own
Equation 14. Do not place the two side by side without saying so.

The same caution applies to `shap_summary`. Its `mean_abs` column — the average
size of a sensor's contribution, ignoring direction — is the usual measure of
"importance", and it is the order in which the rows of the paper's Fig. 9
appear, which is the convention for a SHAP summary plot although the paper does
not say so. It is **not** established that `mean_abs` is what Table 12 reports.
By F-08 that table holds magnitudes of something, computed in a way the paper
never states.

`shap_summary` also returns `mean`, the average signed contribution (a large
negative value means the sensor consistently pulled the predicted life *down*),
and `max_abs`, the single largest contribution the sensor ever made.

## The one property to check

Shapley values must add up: the fourteen contributions plus the baseline
prediction must equal the actual prediction, exactly. This is called
**efficiency**, it is Equation 14 of the paper, and it is what makes the
numbers trustworthy rather than merely suggestive. If it fails, the attribution
is wrong no matter how plausible it looks.

`efficiency_error` measures it. It returns the largest absolute violation of
`sum(phi) == f(x) - f(baseline)` across the given examples, in flight cycles.
Anything beyond floating-point noise means the attribution is wrong. The test
suite asserts this.

One more caveat: explaining an untrained network produces numbers that mean
nothing, and nothing in the code stops you doing it. The values will add up
correctly, because efficiency is a property of the arithmetic rather than of
the network, so a passing efficiency check is not evidence that the model
learned anything.

## Judgement calls made in this folder

- **A-20 — what "a sensor not considered" means.** Equation 15 needs a
  prediction with a sensor absent, but the network always needs all fourteen
  inputs, and the paper never says what it substitutes. We substitute that
  sensor's average window over the training set. Recorded as A-20 in
  `docs/ASSUMPTIONS.md`.

Finding F-08, also in `docs/ASSUMPTIONS.md`, is not a choice of ours — it is an
observation about the paper: its Table 12 cannot hold signed Shapley values
satisfying its own Equation 14.

## References

- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. *Advances in Neural Information Processing Systems 30
  (NeurIPS)*, 4765-4774. https://arxiv.org/abs/1705.07874
- Shapley, L. S. (1953). A value for n-person games. In H. W. Kuhn & A. W.
  Tucker (Eds.), *Contributions to the Theory of Games, Volume II*
  (pp. 307-317). Princeton University Press.
  https://doi.org/10.1515/9781400881970-018
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and
  Technology*, 168, 110875. https://doi.org/10.1016/j.ast.2025.110875

## What each name means

Everything `shap.py` offers.

| Name | What it is |
|---|---|
| `mean_window_baseline(dataset)` | The average reading of each sensor, which stands in for a sensor left out of the calculation. **Training windows only.** |
| `shap_values(model, features, baseline)` | How many cycles each sensor added to the prediction. Takes one window or many, and returns one number per sensor, or one row per window. Raises above `MAX_EXACT_NODES` sensors rather than silently switching to an estimate. |
| `dynamic_shap(model, dataset, baseline, unit=..., symbols=...)` | The same calculation at every cycle of one engine's life. Returns one row per cycle; the columns are listed below. |
| `shap_summary(values, symbols)` | Ranks the sensors by how much they mattered. One row per sensor; the columns are listed below. |
| `efficiency_error(model, features, baseline, values)` | The largest amount by which the numbers fail to add up, in cycles. Anything above rounding noise means a bug. |
| `coalition_weights(n_nodes)` | How much a group of each size counts, from Equation 15. Element `s` is the weight for groups of `s` sensors. |
| `MAX_EXACT_NODES` | 20. Above this, trying every combination stops being reasonable. |

`dynamic_shap` returns these columns, in cycle order:

| Column | What it holds |
|---|---|
| `end_cycle` | The cycle the window ends at. |
| `rul` | The true remaining life at that cycle. |
| `predicted` | What the model predicted there, in cycles. |
| `baseline` | What the model predicts for a wholly typical engine. Constant down the column. |
| one per sensor | That sensor's contribution at that cycle, in cycles. |

`shap_summary` returns one row per sensor, sorted by `mean_abs` downwards:

| Column | What it holds |
|---|---|
| `mean_abs` | The average size of the sensor's contribution, ignoring direction. This is the usual measure of importance — but read the trap above before comparing it with the paper. |
| `mean` | The average signed contribution. A large negative value means the sensor consistently pulled the predicted life down. |
| `max_abs` | The single largest contribution the sensor ever made. |

`shap_values` and `dynamic_shap` both take an optional `device`, and default to
wherever the model already is. `shap_values` also takes `batch_size`, which
changes speed and memory only, never the numbers.
