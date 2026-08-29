# `evaluation` — scoring a set of RUL predictions

This folder computes the four numbers the paper we are re-implementing (Wen et
al., 2026) reports for every experiment. RUL means "remaining useful life": how
many more flight cycles an engine has before it fails. Every function here takes
two equal-length arrays — the true RUL and the predicted RUL, one value per
engine — and returns a number.

> **The `.py` files here carry one line of description each, and nothing more.**
> All of the explanation lives in this file.

Everything works on plain NumPy arrays and nothing here imports PyTorch. That is
deliberate: you can score a baseline, a model written in another framework, or a
column of numbers copied out of a published table, using exactly the same code
that scores ours.

## Files

- `metrics.py` — the four measures, plus a helper that validates the two input
  arrays and an `evaluate` function that runs all four at once.
- `__init__.py` — re-exports the public names so you can write
  `from thesis_gnn_rul_xai.evaluation import rmse`.

## The sign convention, and why it is not the paper's

Every measure here is built from the difference between a prediction and the
truth, so the direction of that subtraction decides everything. This folder
uses:

```
d = predicted - actual
```

So `d > 0` means the model predicted **more** remaining life than the engine
actually had. That is the dangerous direction: the engine keeps flying past the
point where it should have been serviced.

The paper's Equation 20 defines the difference the other way round,
`delta = actual - predicted`. Its Equation 21 then scores `delta >= 0` with
`exp(delta/10) - 1` and `delta < 0` with `exp(-delta/13) - 1`. Substituting the
paper's own definition into the paper's own formula hands the gentler divisor
(13) to over-prediction and the harsher divisor (10) to under-prediction — the
reverse of what the text beside the equations says in words ("the penalty for
over-prediction is relatively large ... over-prediction introduces security
risks"), and the reverse of the scoring function published for the 2008 NASA PHM
Data Challenge that the paper cites.

The two printed equations are inconsistent with each other. We implement the
intent the paper states in words, which is also the standard function every
other paper on this dataset uses, so that our numbers can be compared with
theirs. That choice is recorded as A-15 in `docs/ASSUMPTIONS.md`.

## The four measures

### `rmse` — root mean squared error (the paper's Equation 22)

In flight cycles. Symmetric: it does not care whether an error is early or late.
This is the headline number in most papers on this dataset.

### `nasa_score` — the 2008 challenge score (the paper's Equation 21)

Per engine:

```
d < 0   (early, safe)      ->  exp(-d / 13) - 1
d >= 0  (late, dangerous)  ->  exp( d / 10) - 1
```

Lower is better; a perfect set of predictions scores 0.

Two traps here.

1. **It is a sum, not an average.** It grows with the size of the test set, so
   a score is only meaningful next to another score computed on the same
   sub-dataset. Never compare scores across test sets of different sizes.
2. **The penalty is exponential, so one badly wrong engine can outweigh every
   other one.** That is deliberate rather than a flaw: in maintenance, a single
   missed failure matters more than many small errors. But it does mean the
   score is far less stable than RMSE.

### `r_squared` — coefficient of determination (the paper's Equation 23)

The fraction of the variance in the true labels that the model explains. 1 is
perfect; 0 is no better than always predicting the mean.

We compute it the standard way:

```
1 - sum((predicted - actual)^2) / sum((actual - mean_of_actual)^2)
```

The paper's Equation 23 prints the denominator as
`sum((predicted_i - mean)^2)` instead — that is, how far the *predictions*
spread around the mean of the observations, rather than how far the
*observations* spread around it. That is not the coefficient of determination,
and it rewards the wrong thing: a model can raise its score simply by spreading
its output more widely, without becoming any more accurate. Predicting
`[-5, 15, 35, 55]` for a truth of `[10, 20, 30, 40]` is no better than
predicting the mean — the standard formula scores it 0.0 — but the paper's
version scores it 0.75. We use the standard definition, which is also what the
text beside the equation describes. That choice is recorded as A-16 in
`docs/ASSUMPTIONS.md`.

`r_squared` raises a `ValueError` when every true label is the same value,
because the denominator is then zero and the measure is undefined.

### `asue` — Average Safe Underestimation Error (the paper's Equation 24)

The mean of `actual - predicted`, counted only over the engines where the model
predicted *less* life than there was. Engines the model was optimistic about do
not contribute at all.

Underestimating is the safe direction, but doing it by a wide margin means
servicing engines that still had plenty of life left, so this measures the cost
of excess caution.

If the model never underestimated anything, `asue` returns `0.0`. That is the
**best possible value** of this measure, not a missing result. Because a bare
`0.0` is easy to misread as "no data", the function logs an informational
message whenever it takes that path.

## `evaluate` — all four at once

`evaluate(actual, predicted)` returns a dictionary with `"rmse"`, `"score"`,
`"r2"`, `"asue"`, `"n_engines"`, and two extras that make a result easier to
read at a glance:

- `"mean_error"` — the average of `predicted - actual`. Positive means the model
  is optimistic overall.
- `"n_late"` — how many engines the model was optimistic about, which is the
  count that matters for safety.

`n_late` counts engines with `d >= 0`, so an exactly-correct prediction is
counted as late. That looks odd, but it is the same boundary `nasa_score` uses;
counting only `d > 0` would make the two disagree on a zero error.

## Input validation

All the measures share one helper that flattens both inputs to float arrays and
rejects three things:

- arrays of different lengths;
- an empty set of predictions;
- any non-finite value. A `NaN` reaching this code usually means the model
  diverged during training, so the error message says so rather than quietly
  propagating the `NaN` into a metric.

## Judgement calls made in this folder

Both are cases where the paper contradicts itself and we had to pick a side.
Full write-ups are in `docs/ASSUMPTIONS.md`.

- **A-15 — the sign of the score function.** The paper's Equation 20 and
  Equation 21 imply the opposite penalty direction from the one its own prose
  and the cited 2008 NASA challenge describe. We follow the prose and the
  challenge, so late (optimistic) predictions are punished harder.
- **A-16 — the denominator of R-squared.** The paper's Equation 23 puts the
  predicted values in the denominator, which is not the coefficient of
  determination. We use the standard formula, with the observed values.

## References

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation
modeling for aircraft engine run-to-failure simulation. *2008 International
Conference on Prognostics and Health Management*, Denver, CO, pp. 1-9. IEEE.
https://doi.org/10.1109/PHM.2008.4711414

Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
168, 110875. https://doi.org/10.1016/j.ast.2025.110875
