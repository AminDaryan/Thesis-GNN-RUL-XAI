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
- `published.py` — the paper's own figures, copied out of its tables once, so
  that everything that sets a result beside the paper quotes the same numbers.
  Explained under [The paper's figures](#the-papers-figures-publishedpy) below.
- `__init__.py` — re-exports the names from `metrics.py` so you can write
  `from thesis_gnn_rul_xai.evaluation import rmse`. The names in `published.py`
  are imported from that file directly.

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
theirs (assumption A-15 in `docs/ASSUMPTIONS.md`).

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
text beside the equation describes (assumption A-16 in
`docs/ASSUMPTIONS.md`).

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

## The paper's figures (`published.py`)

### Why they are kept in one file

Two places in this project print the paper's results beside one of ours:
`scripts/train.py`, at the end of every run, and the playground on the
documentation site. If each held its own copy of the paper's tables, a typing
slip in one copy would make the two disagree about what the paper says, and a
reader would have no way to tell which was right. So the figures are typed in
once, here, and both read them from this file.

### What it holds

| Name | What it is | From the paper |
|---|---|---|
| `SUBSETS` | The four sub-datasets, `("FD001", "FD002", "FD003", "FD004")`. C-MAPSS is split into four, differing in how many operating conditions and fault types their engines see. | Table 5 |
| `DATASET_DESCRIPTION` | For each sub-dataset: how many engines are in the training and test files, how many operating conditions, and how many fault modes. | Table 5 |
| `TRAINING_SETTINGS` | The training settings the paper prints: a learning rate — how large each correction to the network's numbers is — of 1e-3 falling to 1e-4; 200 passes over the training data; 32 windows per correction; windows of 30 cycles; Adam, the rule that turns the measured errors into those corrections; and mean squared error, the average squared distance between prediction and label, as the error being reduced. | Table 6 |
| `PublishedRow` | One row of results for one sub-dataset: a key, a label, the table it came from, and the four measures. | — |
| `published_row(key, subset)` | Returns the `PublishedRow` for one version of the model on one sub-dataset. | Tables 7 to 11 |
| `ABLATION_ROWS` | The averages and differences Table 11 prints, kept only so a test can check the typing (below). | Table 11 |
| `SHAP_TABLE_12_GRAPHS` | The fifteen numbers that label Table 12's columns, `(1, 2, 3, 4, 5, 78, ..., 191)`. Each is one window of one engine the paper does not name, and the same number names one of the small graphs in Fig. 10. | Table 12, Fig. 10 |
| `SHAP_TABLE_12` | For each of the 14 sensors, its fifteen printed "SHAP importance values", one per column. Every one is positive, one of the two reasons they cannot be the signed contributions this project computes (finding F-08). | Table 12 |
| `FIG_10_RING` | The order in which Fig. 9 and Fig. 10 place the 14 sensors round their circle, clockwise from the node just left of the top. | Fig. 9, Fig. 10 |

Every value held by the first six names above was compared with the text of the
PDF on 2026-09-14. `SHAP_TABLE_12` and `SHAP_TABLE_12_GRAPHS` were not typed: they
were printed from the PDF's text layer by a script and pasted in, on the same
day. `FIG_10_RING` cannot come from the text layer, because the sensor names in
both figures are part of a picture; it was read by eye from graph 1 of Fig. 10
and from Fig. 9, each rendered at scale 8, and the two agree. How Fig. 10's
colours were then checked against Table 12, and the two graphs where they differ,
are recorded in `docs/ASSUMPTIONS.md` (finding F-17).

### The four versions of the model a row can describe

The paper reports its full network and, in Table 11, three versions with one
part removed, to show what each part was worth. `published_row` knows them by
these keys:

| Key | Label | What was removed | FD001 RMSE |
|---|---|---|---|
| `"full"` | ESTGNN (Full) | nothing | 11.78 |
| `"without_gcn"` | ESTGNN w/o GCN | the graph layers | 16.42 |
| `"without_lstm"` | ESTGNN w/o LSTM | the per-sensor LSTM stage | 14.93 |
| `"without_physics_prior"` | ESTGNN w/o Physics prior | the paper's sensor graph | 12.34 |

An unknown key or sub-dataset is refused with the list of valid ones:

```text
published_row("transformer", "FD001")
  -> ValueError: no published row called 'transformer'; the paper prints 'full' and ['without_gcn', 'without_lstm', 'without_physics_prior'].
```

**Which row a run may be set beside is not decided here.** It depends on what
the run built — above all whether it had a temporal stage — and the rule is
written out in `models/README.md`, under "What these results may be compared
with". This file only answers "what did the paper print for that row".

### A trap: Table 11 prints only RMSE

The full model has all four measures, spread over Tables 7 to 10. The three
versions with a part removed appear only in Table 11, and Table 11 gives root
mean squared error alone. So their other three fields are `None`:

```text
published_row("without_lstm", "FD001")
  -> PublishedRow(key='without_lstm', label='ESTGNN w/o LSTM', source='Table 11',
                  rmse=14.93, score=None, r2=None, asue=None)
```

`None` means "the paper does not print this", and it must stay `None` on the
way to any page. Turned into a 0 it would read as a score of zero, the best
possible score, which the paper never claimed.

### How the typing is checked

A mistyped figure would look exactly as plausible as a correct one. But Table 8
and Table 11 each print an average across the four sub-datasets, and Table 11
also prints each version's difference from the full model. Those are the
paper's own arithmetic on the numbers typed in here, so they can be recomputed
and compared. For the model without its LSTM stage:

```text
average:    (14.93 + 23.34 + 16.25 + 24.21) / 4 = 78.73 / 4 = 19.6825  ->  19.68, as printed
difference on FD001:   14.93 - 11.78 = 3.15,                           as printed
```

and for the full model, `(11.78 + 16.21 + 12.34 + 18.67) / 4 = 59.00 / 4 =
14.75`, the average Table 8 prints. `tests/test_playground.py` repeats this for
the RMSE of all four rows, so a slip in one of those digits fails a test.

Two things it does not catch. A slip that leaves the average unchanged, such as
two figures swapped between sub-datasets, passes; that is what the comparison
with the PDF was for. And the full model's score, R squared and ASUE are not
checked by any test, although Tables 7, 9 and 10 print averages that would
allow it.

**Table 12 prints no averages, so it is checked two other ways.** First, the
entry of the log showing that these values cannot be signed contributions added
up seven of the columns and recorded the sums (finding F-08). Adding up graph 1's
column as it sits in `SHAP_TABLE_12`:

```text
3.39 + 3.49 + 4.18 + 1.79 + 3.33 + 3.15 + 4.20 + 4.69 + 3.91 + 2.44 + 3.77 + 1.53 + 2.30 + 3.34 = 45.51
```

which is the 45.51 recorded. Second, the paper sets the four largest values of
each column in bold, and which cells are bold was read from the font of each
cell in the PDF. In graph 1 the four largest are Ps30 4.69, Nc 4.20, T50 4.18
and phi 3.91, with BPR fifth at 3.77, and those four are the bold ones.
`tests/test_playground.py` checks all seven sums and the four largest of all
fifteen columns, so a slip that changes a sum or moves a sensor into or out of
a column's top four fails a test. Eight of the fifteen columns have no recorded
sum, so a slip in one of them that leaves its top four alone — 1.81 typed as
1.18 in the middle of graph 80, say — passes.

## Judgement calls made in this folder

Both are cases where the paper contradicts itself and we had to pick a side.
Full write-ups are in `docs/ASSUMPTIONS.md`.

- **The sign of the score function.** The paper's Equation 20 and Equation 21
  imply the opposite penalty direction from the one its own prose and the cited
  2008 NASA challenge describe. We follow the prose and the challenge, so late
  (optimistic) predictions are punished harder (assumption A-15).
- **The denominator of R-squared.** The paper's Equation 23 puts the predicted
  values in the denominator, which is not the coefficient of determination. We
  use the standard formula, with the observed values (assumption A-16).

## References

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation
modeling for aircraft engine run-to-failure simulation. *2008 International
Conference on Prognostics and Health Management*, Denver, CO, pp. 1-9. IEEE.
https://doi.org/10.1109/PHM.2008.4711414

Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
168, 110875. https://doi.org/10.1016/j.ast.2025.110875
