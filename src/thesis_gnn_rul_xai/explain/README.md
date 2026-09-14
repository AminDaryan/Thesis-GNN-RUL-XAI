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

- `shap.py` — the paper's method: the coalition weights, the baseline, the
  Shapley values themselves, the per-cycle version, and the check that the
  numbers add up.
- `occlusion.py` — a cheaper method that is not in the paper: replace one
  sensor at a time with the baseline and record how far the prediction moves.
- `plots.py` — the figure of every sensor's contribution at every cycle of one
  engine's life, for either method or both side by side.
- `__init__.py` — re-exports the public names from all three.

The notebooks in `notebooks/xai/` use all of these on a stored trained network,
one method per notebook.

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
sees "this sensor is behaving typically" rather than some arbitrary value
(assumption A-20 in `docs/ASSUMPTIONS.md`).

**Why zero would be the obvious alternative, and why that is a trap.** The
temptation is to feed a sensor 0 when it is left out. Whether that is safe
depends entirely on the rescaling, not on the baseline:

- Under the **min-max** scaling that was the default until 2026-08-29, zero was
  a sensor's lowest ever reading. Substituting it produced a ranking unrelated
  to the mean-window one (Spearman −0.16) and promoted `P15`, the least
  informative sensor, to first place.
- Under the **z-score** default now in force — every sensor rescaled to say how
  many standard deviations it sits from its own training average, so that zero
  means "exactly average" — the number zero is the training mean, and the zero
  baseline agrees with the mean window almost exactly (Spearman +0.996).

The log records both measurements (assumption A-20). The point it draws from
them is that a SHAP
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
cycles. The paper's cannot be: Table 12's contents cannot be signed Shapley
values satisfying the paper's own Equation 14 (finding F-08 in
`docs/ASSUMPTIONS.md`). Do not place the two side by side without saying so.

The same caution applies to `shap_summary`. Its `mean_abs` column — the average
size of a sensor's contribution, ignoring direction — is the usual measure of
"importance", and it is the order in which the rows of the paper's Fig. 9
appear, which is the convention for a SHAP summary plot although the paper does
not say so. It is **not** established that `mean_abs` is what Table 12 reports.
By the same finding, that table holds magnitudes of something, computed in a way
the paper never states.

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

## Occlusion: one sensor at a time

`occlusion_values` answers a simpler question than the Shapley value: how far
does the prediction move when this one sensor's readings are replaced by the
baseline, with every other sensor left as it is? For one window that is

```
occlusion(sensor) = f(every sensor real) − f(every sensor real but this one)
```

and it needs 15 runs of the network — the window itself and one per sensor —
where the Shapley values need 16,384. It is not in the paper.

**It is one term of the Shapley sum.** In Equation 15 the group `S` holding all
thirteen other sensors gives `f(S + {i}) − f(S)`, which is exactly the
occlusion value, and the sum weights that term by `13! 0! / 14! = 1/14`.
Occlusion keeps that one term at full weight and drops the rest. The two agree
only for a sensor that adds the same amount whichever others are present.

Two consequences follow, and neither is about this project's network in
particular:

- **The fourteen occlusion values do not add up** to the prediction minus the
  prediction at the baseline. With `f(a, b) = 2a + 3b + 4ab`, input `(1, 1)` and
  baseline `(0, 0)`, occlusion gives `a` 6 and `b` 7, which sum to 13 where the
  whole change is 9: each is measured with the other present, so each is
  credited with the whole `4ab` term.
- **Two sensors carrying the same information can both look worthless.** With
  `g(a, b) = 10 × max(a, b)`, replacing either input alone leaves the other to
  keep the output at 10, so both get 0, where the Shapley values are 5 each.

`notebooks/xai/02_occlusion_fd001.ipynb` measures how far the two methods part
on the stored network, and
`tests/test_examples_torch.py::test_example_occlusion_is_the_shapley_term_where_the_sensor_arrives_last`
checks the one-term relationship on a three-sensor network.

## The figure: contributions over an engine's life

`plot_contributions_over_time` draws what `dynamic_shap` returns. The top panel
is the prediction being explained: the true remaining life, the predicted one,
the prediction at the baseline, and a line at the cycle where the true remaining
life first drops below its ceiling. Below it is one small panel per sensor,
holding that sensor's contribution at every window, ordered by average size with
the largest first, and all on **one vertical scale**, so that a tall panel is a
large contribution and not just a large axis.

With one method, the area between the line and zero is blue where the sensor
pushed the predicted life up and red where it pushed it down. Given a second
table through `compare` — occlusion values for the same windows, say — each
panel draws the two methods as two lines instead, and the panel order still
follows the first. It returns the matplotlib figure and draws nothing itself,
so a notebook calls `plt.show()` and a script can save it.

It needs the columns `dynamic_shap` produces, and refuses a table without them
rather than drawing an empty chart. `tests/test_shap.py` checks the panel order
and the shared scale.

## Judgement calls made in this folder

- **What "a sensor not considered" means.** Equation 15 needs a prediction with
  a sensor absent, but the network always needs all fourteen inputs, and the
  paper never says what it substitutes. We substitute that sensor's average
  window over the training set (assumption A-20 in `docs/ASSUMPTIONS.md`).
  Occlusion uses the same substitute, so the two methods differ only in the
  arithmetic.
- **Offering occlusion at all.** It is not in the paper. It is here as the
  cheapest comparison, not as a replacement.

That the paper's Table 12 cannot hold signed Shapley values satisfying its own
Equation 14 is not a choice of ours but an observation about the paper (finding
F-08, also in `docs/ASSUMPTIONS.md`).

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

Everything `shap.py`, `occlusion.py` and `plots.py` offer.

| Name | What it is |
|---|---|
| `mean_window_baseline(dataset)` | The average reading of each sensor, which stands in for a sensor left out of the calculation. **Training windows only.** |
| `shap_values(model, features, baseline)` | How many cycles each sensor added to the prediction. Takes one window or many, and returns one number per sensor, or one row per window. Raises above `MAX_EXACT_NODES` sensors rather than silently switching to an estimate. |
| `dynamic_shap(model, dataset, baseline, unit=..., symbols=...)` | The same calculation at every cycle of one engine's life. Returns one row per cycle; the columns are listed below. |
| `shap_summary(values, symbols)` | Ranks the sensors by how much they mattered. One row per sensor; the columns are listed below. |
| `efficiency_error(model, features, baseline, values)` | The largest amount by which the numbers fail to add up, in cycles. Anything above rounding noise means a bug. |
| `coalition_weights(n_nodes)` | How much a group of each size counts, from Equation 15. Element `s` is the weight for groups of `s` sensors. |
| `MAX_EXACT_NODES` | 20. Above this, trying every combination stops being reasonable. |
| `occlusion_values(model, features, baseline, batch_size=512, device=None)` | From `occlusion.py`. The prediction with every sensor real minus the prediction with one sensor replaced by the baseline, one number per sensor. Takes one window or many, like `shap_values`; `batch_size` changes speed and memory only. |
| `plot_contributions_over_time(table, sensors, compare=None, method="Shapley value", compare_method="occlusion", title=None)` | From `plots.py`. The figure described above, from a table shaped like `dynamic_shap`'s. `compare` is a second table of values for the same windows, one column per sensor. |

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
