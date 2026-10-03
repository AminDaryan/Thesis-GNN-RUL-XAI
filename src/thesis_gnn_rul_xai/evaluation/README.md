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

One script outside this folder is explained here because it is about what a
score is worth: `scripts/baselines.py`, which scores ordinary models beside a
stored run's networks. See
[Ordinary models beside the network](#ordinary-models-beside-the-network-scriptsbaselinespy)
below.

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
risks"). The dataset paper that published the 2008 NASA PHM Data Challenge's
scoring function prints the same swap in its Eq. (11), while its own text and
its Figure 9 give the harsher divisor, 10, to late predictions (finding F-20).

The two printed equations are inconsistent with each other. We implement the
intent the paper states in words, which is also the standard function every
other paper on this dataset uses, so that our numbers can be compared with
theirs (assumption A-15 in `docs/ASSUMPTIONS.md`).

## The four measures

What each measure means, its formula and a worked example of each are taught
on the page [Reading a result](../../../docs/learn/06_reading_results.md). This
section keeps what only the code adds: which of the paper's equations each
function stands for (two of them on purpose not as printed: the score follows
the NASA sign convention, assumption A-15, and R² divides by the spread of the
true values, assumption A-16), and how three of them behave at their edges.

| Function | Measure | Stands for the paper's equation |
|---|---|---|
| `rmse` | root mean squared error | 22 |
| `nasa_score` | the 2008 challenge score | 21 |
| `r_squared` | coefficient of determination | 23 |
| `asue` | Average Safe Underestimation Error | 24 |

### `nasa_score` — the 2008 challenge score (the paper's Equation 21)

Its exponential penalty, which lets one badly wrong engine outweigh all the
others, is deliberate rather than a flaw: in maintenance, a single missed
failure matters more than many small errors. It does make the score far less
stable than RMSE.

### `r_squared` — coefficient of determination (the paper's Equation 23)

The paper's version divides by how far the *predictions* spread around the
truth's mean, not by how far the truth itself does, and that rewards spreading
the predictions without making them any more accurate. A worked case:
predicting `[-5, 15, 35, 55]` for a truth of `[10, 20, 30, 40]`, whose mean is
25.

```
prediction minus truth, squared              15² +  5² +  5² + 15² =   500
truth minus its mean, squared                15² +  5² +  5² + 15² =   500   standard:  1 - 500/500  = 0.0
prediction minus the truth's mean, squared   30² + 10² + 10² + 30² = 2 000   paper's:   1 - 500/2000 = 0.75
```

The standard formula scores it 0.0, no better than predicting the mean every
time; the paper's version scores it 0.75. The code uses the standard definition,
which is also what the text beside the equation describes (assumption A-16 in
`docs/ASSUMPTIONS.md`).

`r_squared` raises a `ValueError` when every true label is the same value,
because the denominator is then zero and the measure is undefined.

### `asue` — Average Safe Underestimation Error (the paper's Equation 24)

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
| `SHAP_TABLE_12_GRAPHS` | The fifteen numbers that label Table 12's columns, `(1, 2, 3, 4, 5, 78, ..., 191)`, for one engine the paper does not name; the same number names one of the small graphs in Fig. 10. The paper never says which cycle a graph is; this project reads graph *g* as the engine's *g*-th window, the one ending at cycle *g* + 29 (assumption A-35). | Table 12, Fig. 10 |
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

## Ordinary models beside the network (`scripts/baselines.py`)

### The question it answers

Every result in this project is set beside the paper's tables and beside other
configurations of the same graph network. Neither comparison says whether the
graph network is worth its complexity: a regression model with no sensor graph
in it, fitted to the same training engines and given the same numbers, might do
as well. `scripts/baselines.py` answers that for one stored run. It trains no
graph network. It reads networks already trained and saved by
`scripts/train.py`, and fits a fixed list of ordinary models — the *ladder*,
from a constant up to models given a summary of the whole window — on exactly
the engines each of those networks was trained on, then scores everything on
the same 100 test engines with `evaluate` from this folder.

### What a stored run holds, and why the data is rebuilt for every seed

A stored run is a folder under `reports/` written by `scripts/train.py`. For
every seed — the number that fixes the random choices of one repetition of the
training — it holds the trained network (`seed_<n>/model.pt`), that network's
prediction for each of the 100 test engines (`seed_<n>/test_predictions.csv`),
and the four scores of those predictions (one row per seed in `results.csv`).

The seed also decides which training engines were held back. Of FD001's 100
training engines, a fifth are set aside to decide when training stops and which
pass over the data keeps its weights, and the rescaling of each sensor —
subtracting its average and dividing by its spread — is measured on the other
80 only (assumption A-25). So two seeds hand slightly different numbers to the
network even for the same test engine, and an ordinary model set beside seed
42's network has to be fitted on seed 42's 80 engines and scored on the test
engines as seed 42 rescaled them. The script rebuilds each seed's data with
`load_seed` from `training/stored.py`, which reruns the preprocessing under that
seed. The preprocessing cuts every engine's history into *windows*, 30
consecutive cycles each, labelled with the remaining life at the window's last
cycle. Under the default preprocessing settings and seed 42 that gives 14,459
training windows from 80 engines, an array of shape `(14459, 14, 30)`: one row
per window, 14 sensors, 30 cycles each (`tests/test_examples.py` checks the
count). The 20 held-back engines are not used at all.

### It refuses a run it cannot reproduce

Before anything is fitted, the stored run is checked against the data files on
disk, and each seed's rebuilt data against the run's own records. If any check
fails the script stops and writes nothing:

1. Where the run recorded a fingerprint of each of the three NASA files it
   read — a 64-character code computed from the file's bytes, which changes if
   any byte does — the files in the data folder give the same fingerprints.
   Older runs recorded none, the two of 2026-09-14 named below among them, and
   for those this check is skipped.
2. The saved test engines and their labels are the rebuilt ones.
3. Scoring the saved predictions again gives the four numbers in `results.csv`,
   up to the rounding of a number written to a text file (one part in a
   million).
4. The saved network, reloaded and run on the rebuilt test inputs, gives its
   saved predictions to within 0.001 cycles. This is the check that confirms
   the split: the test inputs depend on which 80 engines the rescaling was
   measured on, so data rebuilt under the wrong split hands the network
   different numbers and moves its predictions.
5. Where the run also saved its engine split (`split.json`) and its rescaling
   numbers (`scaling.csv`), the rebuilt ones match them. Runs from before
   these two files were written skip this check.

A run that saved no networks is refused, because nothing could then confirm
which engines it trained on. A second run given with `--also` (described under
"What it writes, and how to run it" below) goes through the same checks, and is
refused as well if its preprocessing settings or its seeds differ from the
first run's, or if, seed by seed, its rebuilt training engines and test inputs
are not the first run's.

`tests/test_baselines.py` edits copies of a stored run and requires every edit
to be refused: a `results.csv` that disagrees with the saved predictions; saved
predictions moved by half a cycle, with `results.csv` rewritten to agree with
them; one seed's network and records filed under another seed, so that every
score agrees and only the split is wrong; on a run that saved them, one engine
moved in `split.json` or one rescaling number changed in `scaling.csv`; a copy
with its saved network removed; a copy that records the fingerprints of the
real NASA files, pointed at a copy of those files with the answers changed; and,
given with `--also`, a copy with one preprocessing setting changed or with one
seed missing. The same copies left untouched have to pass, alone and with
`--also`.

### The ladder

Every model comes from scikit-learn, a Python library of standard statistical
models, with the seed of the stored network as its random seed. Nothing was
tuned: apart from one width noted below, every setting is the library's
default, so no setting was chosen by looking at a result.

The list of models was not drawn up blind, though. Five of the first eight in
the table below — linear regression, a small network with one layer of 64, a
random forest, gradient boosting and a constant — and that width of 64 follow
the audit of 2026-09-08, which had already measured those five on the same
fourteen numbers under an earlier configuration of the graph network, when that
network scored 16.82 ± 0.16. The audit's forest had 200 trees where this one
has the library's 100, and its constant was what it calls the best possible one
rather than the average training label. The other three of the first eight — the
shuffled labels, the engine's age and the 42 summaries — were added when the
comparison was first made outside the repository, on 2026-09-14. The six that
take the window summaries apart were added the next day, after the row with all
42 summaries had been seen, to find which summary carries the difference; they
are a follow-up question.

**Correction, 2026-10-02.** This section first said the first eight models were
listed before any result was seen. An independent review of the script on the
same day pointed out that the audit's table had measured five of them first,
and that this section itself took the width of 64 from that audit.

Each model is fitted on every window of the seed's 80 training engines, with
that window's capped remaining life as the label, and predicts the last window
of each test engine.

Apart from the constant, the models are of three kinds. **Linear regression**
predicts a weighted sum of its inputs plus a constant, with the weights chosen
to make the squared error on the training windows as small as possible. A
**decision tree** is a chain of yes-or-no questions about the inputs — "is the
T50 reading above 0.3?" — each answer leading to the next question and the last
one to a predicted value, and the ladder combines many trees in two ways: a
**random forest** averages 100 trees, each grown on a random resample of the
training windows, and **gradient boosting** adds 100 small trees one after
another, each fitted to what the trees before it still get wrong. The third
kind, a **network with one layer of 64**, computes 64 weighted sums of its
inputs, replaces each negative one by zero, and predicts a weighted sum of the
64.

| Key | Model | What it is given | Why it is on the ladder |
|---|---|---|---|
| `constant` | the average training label, for every engine | nothing | the floor: a model that does not beat it has learned nothing from the readings |
| `boosting_shuffled` | gradient boosting, after the labels were shuffled among the training windows | the 14 numbers the network gets | the readings with their link to remaining life cut |
| `boosting_age` | gradient boosting | the engine's age, the cycle number at the end of the window | what age alone tells, with no sensor at all |
| `linear_last` | linear regression | the 14 numbers the network gets | the simplest model that reads sensors |
| `network64_last` | a network with one layer of 64 | the same 14 numbers | a flexible model with no sensor graph |
| `forest_last` | random forest | the same 14 numbers | the same |
| `boosting_last` | gradient boosting | the same 14 numbers | the same |
| `boosting_summary` | gradient boosting | 42 numbers: the last value, the average and the slope of each sensor's window | what the window holds that the default network is never given |
| `boosting_mean`, `boosting_slope`, `boosting_last_mean`, `boosting_last_slope`, `boosting_mean_slope` | gradient boosting | one or two of the three summaries | which summary carries what |
| `linear_summary` | linear regression | the 42 summaries | whether a gain needs a flexible model, or only the information |

The width of 64 is the one setting that is not the library's: its default is
100, and 64 is the width the audit of 2026-09-08 used. The small network's
default limit is 200 passes over the training windows; if it stops there before
it has settled, scikit-learn warns, and the warning is written into
`summary.json` rather than hidden.

Every prediction of an ordinary model is clipped to between 0 and 125 cycles:
one below 0 becomes 0, one above 125 becomes 125. The network cannot predict
outside that range, because its last step squeezes its output between 0 and 1
and multiplies by 125, so an ordinary model is held to the same range and is
not charged for predictions the network is prevented from making. The cap on
the test labels at 125 is a logged choice (assumption A-02); this clip of the
ordinary models' predictions is a separate choice of this script (assumption
A-38).

Which way it can move a result follows from the labels, without a measurement.
Every capped test label lies between 0 and 125, so a prediction outside that
range is moved towards its label and never past it. A real case, from linear
regression on the 42 window summaries under seed 42: test engine 78 is
labelled 107 and was predicted 139.6, an error of +32.6 cycles before the clip
and +18.0 after it, since 125 − 107 = 18. So the clip can never raise an
ordinary model's RMSE: it lowers it when it moves a prediction and leaves it
alone when it moves none, so it can favour the ordinary models against the
network, never the other way round. How much it lowers it is measured, not
argued: `per_seed.csv` gives every model's RMSE without the clip beside the
one with it, and how many of its predictions the clip moved. Run on 2026-10-03
with the code at `19dceb5-dirty` (this script not yet committed), against the
default network's five-seed run from a clean `f4f40ad`
(`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z`),
and written to a scratch folder rather than `reports/`, so that run is not
kept, it moved 47 of the 500 predictions of linear regression on the 42 window
summaries, 35 of them below 0, and lowered that model's RMSE from 15.816 to
14.520 cycles; it lowered no other model's by more than 0.144, and changed no
model's comparison with the network on any seed (assumption A-38 has the
table, and the command that writes the same figures to a kept folder).

### The fourteen numbers, and a trap

The pipeline hands every model windows of shape `(windows, 14, 30)`. The default
network keeps only the most recent reading of each sensor (assumption A-07), so
"the 14 numbers the network gets" means `features[:, :, -1]`, the last of the
30 cycles. `features[:, :, 0]` runs without complaint and gives the *first*
cycle of the window, 29 cycles out of date. An earlier specification of this
comparison used index 0. It was written when the pipeline handed over one cycle
per window, so 0 was then the last; since 2026-09-09 the same line measures
something else. `tests/test_baselines.py` checks that the script takes the last.

### The window summaries

The three numbers per sensor describe its 30-cycle window, after smoothing and
rescaling: the last reading, the average of the thirty, and the slope — how fast
the readings rise or fall per cycle along the straight line that fits them best.
They are the three numbers the graph network's window-summary stage hands its
sensor nodes (assumption A-28), and the straight line, its formula and a worked
example on real data are in `models/temporal/README.md`, under
`window_summary`. One difference: that stage multiplies the slope by 29, so it
reads as the rise across the whole window, and this script does not. For linear
regression that changes no prediction, because a weight 29 times smaller undoes
it. For the tree models it would change nothing in exact arithmetic either — a
question "is the slope above 0.01?" becomes "is it above 0.29?" — but
scikit-learn rounds every input of a random forest or of gradient boosting to
single precision, about seven significant digits, before it builds a tree, and
the two scales round differently. Measured on 2026-10-02 with scikit-learn
1.9.0, under the default preprocessing settings and seed 42 — the settings and
the first seed of the stored default run named under "How the comparison is
scored" — gradient boosting given the slope alone scored an RMSE of 13.320 with
the slope per cycle and 13.238 with it times 29, one test engine's prediction
moving by 5.4 cycles, while linear regression's predictions agreed to within
1e-13. So a tree model's score on one seed can move by nearly a tenth of a
cycle for a reason that has nothing to do with what it is given.

The average is open to the same wobble, and the way the script takes it does
not remove it. The windows arrive in single precision. The script adds up the
thirty readings in double precision, about sixteen significant digits, so each
average is the average of those thirty numbers to about sixteen digits,
whatever order they are added in; an average taken in single precision carries
rounding of its own in the seventh digit. A tree model then rounds the
double-precision average back to single precision, so it never sees the extra
digits: rounding the average to single precision by hand before handing it to
gradient boosting changes none of its predictions. The choice only decides
which of two neighbouring single-precision values a tree is given, and neither
is more correct for a tree. Under the same settings and seed, the two differ
for 106,726 of the 202,426 training averages (14,459 windows times 14
sensors), by at most 7.2e-7, and gradient boosting given the average alone
scored 21.842 with the double-precision average and 21.689 with the
single-precision one. `tests/test_examples.py` checks every number in this
paragraph and the one before it, and the rounding by hand. The comparison
first made outside the repository on 2026-09-14 took the average in single
precision — recomputed that way, its rows that contain the average come out as
it printed them — so those rows differ slightly from this script's.

On the window of that worked example — T50, the temperature at the low-pressure
turbine outlet, over engine 1's cycles 1 to 30, under seed 42 — the script's
row for the window holds −1.2938, −1.0449 and −0.012408 for T50 in its three
groups of fourteen columns, the last of them the slope per cycle that the stage
multiplies up to −0.3598. `tests/test_examples.py` checks those numbers, and
that NumPy's `polyfit`, which fits the line by a different route, gives the same
slope.

**Correction, 2026-10-02.** This section first presented the double-precision
average as the remedy for the tree models' sensitivity to rounding. An
independent review the same day found, in scikit-learn's own code, that both
tree models convert their inputs to single precision before any tree is built,
so the double-precision average hands them nothing extra.

### How the comparison is scored

Each model gets the four measures on each seed. Then, for each stored network
set against each model, `results.csv` gives:

- the model's RMSE averaged over the seeds, with its standard deviation;
- the network's RMSE minus the model's, averaged over the seeds, so a negative
  number means the network is better;
- on how many seeds the network has the lower RMSE;
- the p-value of a paired t-test over the seeds;
- a 95 % interval for the difference from resampling the test engines, and the
  share of resamplings in which the network is better.

**The paired t-test.** For each seed, take the network's RMSE minus the model's.
The test asks whether the average of those differences lies far from zero
compared with how much they vary from seed to seed: it divides the average by
a measure of that variation, giving a ratio called `t` (written out below). Its
p-value is how often a `t` at least that far from zero would turn up if the two
were, on average over retrainings, equally good. With five seeds it rests on
five numbers.

It is *paired* because each seed gives one difference: the two numbers being
subtracted come from the same seed. With `n` seeds, `d` the difference on one
seed, `d̄` the average of the `n` differences and `s` their standard deviation
(each difference's distance from `d̄`, squared, added up, divided by `n − 1`,
and the square root taken),

```
t = d̄ / (s / √n)
```

The bottom line, `s / √n`, is how far an average of `n` such differences
typically strays from the true average: averaging `n` independent numbers
narrows their spread by `√n`. So `t` says how many of those typical strays the
observed average lies from zero. Worked through on two stored runs whose
comparison the log already records (finding F-16): the default network, which
hands its sensor graph only each sensor's last reading,
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z`,
set against the same network with the recurrent stage, which reads each
sensor's whole 30-cycle window one cycle at a time,
`…_lstm32_cfg72f506_f386291-dirty_20260914T134955Z`. Each seed's RMSE, to four
places:

```
seed                   42        43        44        45        46
last-cycle network  16.6949   16.1311   16.8432   15.9704   16.8427
recurrent stage     12.4274   13.8228   13.2695   13.4379   13.5790
difference d         4.2675    2.3083    3.5737    2.5325    3.2637

d̄   (4.2675 + 2.3083 + 3.5737 + 2.5325 + 3.2637) / 5           = 3.1891
s    √((1.1629 + 0.7759 + 0.1479 + 0.4312 + 0.0056) / (5 − 1))  = 0.7943
s / √5    0.7943 / 2.2361                                       = 0.3552
t         3.1891 / 0.3552                                       = 8.978
```

where 1.1629 is (4.2675 − 3.1891)², and so on. If the two networks were equally
good on average, five differences would give a `t` between −2.776 and +2.776 in
95 % of repetitions of the five-seed experiment. That range comes from
Student's t distribution with `n − 1 = 4` *degrees of freedom* — the number of
differences less the one used up in estimating `d̄` — which is the spread of `t`
that five differences with a true average of zero produce. 8.978 lies far
outside it; the share of that distribution at least 8.978 from zero, on either
side, is the p-value, 0.00085. The log writes the difference the other way
round, recurrent minus last-cycle, so it gives `t = −8.978`, with p = 0.0009,
the same p rounded. The test assumes the seeds' differences are independent
and roughly bell-shaped; with five of them, the second cannot be checked.
`tests/test_examples.py`
reproduces every line from the two runs' `results.csv`, and scipy's `ttest_rel`,
which the script uses, gives the same `t` and p from the unrounded scores.

**Resampling the test engines.** Draw 100 engines from the 100 test engines
with replacement, so that some appear twice and some not at all. On that draw,
compute each seed's RMSE, average over the seeds, do the same for the model,
and record the difference. Repeat 20,000 times; the middle 95 % of the
differences is the interval (statisticians call this a bootstrap). Every model
and every seed use the same 20,000 draws, so on each draw two models are
compared on the same engines. If the interval contains zero, a different set of
100 engines of the same kind could reverse the order. On a stored network's own
row, where there is nothing to subtract, the table gives the interval of its
RMSE instead. A check with exact numbers, in `tests/test_baselines.py`: give
one seed an error of exactly one
cycle on every test engine and another an error of exactly two. Whichever
engines a draw picks, the two RMSEs are 1 and 2, so every one of the draws
gives 1.5.

**Do not confuse the two kinds of uncertainty.** The seeds vary how the network
was trained and which engines were held back; they never vary the test engines,
which are the same 100 every time. So the standard deviation over seeds is how
much retraining moves a result, and says nothing about how much it depends on
which 100 engines happen to be in the test file. The interval over engines
answers that second question. A difference can be clear on one and not on the
other.

### What it writes, and how to run it

```bash
python scripts/baselines.py --run <stored run> --also <another stored run>
```

`--run` names the stored run every model is set against, by its folder or its
name under `reports/`. `--also`, which may be repeated, adds another stored run
prepared the same way — the same preprocessing settings apart from their notes,
and the same seeds — as one more row, and repeats every comparison against it
too; it is checked exactly as `--run` is. `--seeds` and `--models` run part of
the ladder for a quick check, and the folder name then says what was left out.
On the CPU, which is all scikit-learn uses and where the script also runs the
stored networks, fitting all fourteen models took between 2 and 5
minutes a seed on 2026-10-02, with training runs sharing the machine, and the
whole comparison of the two stored runs above, five seeds each, 17 minutes.

It writes one new folder, `reports/baselines_<settings>_<commit>_<time>/`, where
`<settings>` is the stored run's folder name without its own commit and time,
which are recorded inside instead:

| File | What it holds |
|---|---|
| `results.csv` | one row per stored network and model, with every column listed above and each seed's RMSE |
| `table.md` | the same as tables to read, one per stored network |
| `per_seed.csv` | the four measures of every model on every seed, how long each took to fit, how many of its predictions the clip to between 0 and 125 moved (`n_clipped`), and its RMSE without that clip (`rmse_unclipped`) |
| `predictions.csv` | every model's prediction for every test engine on every seed, so that any figure can be recomputed |
| `summary.json` | the commit and its `-dirty` mark, the time, the command, every library's version, fingerprints of the three NASA files, the stored runs it read, each model's exact scikit-learn settings, how the resampling was done, the largest gap found when the networks were reloaded, and any warning a model gave |
| `code_state.zip` | the uncommitted changes to the code, when there are any |

The folder is created new every time, and an existing one is never written into.

### What it does not show

- **Not the best an ordinary model can do.** Every model is untuned, so a row
  says what an ordinary model gets with no effort, not what the best one gets.
- **Not that the answers cannot leak.** The shuffled-label row lands where the
  constant does if the scoring rewards only real skill; it rules out a scoring
  bug that rewards any prediction, but not a leak of the test answers into the
  inputs, because a model trained on shuffled labels would not learn to read
  such a leak either. A separate test rules that out (next section).
- **Not a statement about one network design in general.** A row compares the
  ladder with the stored networks of one configuration, at one commit; the
  comparison has to be run again whenever the network or its training changes.
- **Not a claim about engines.** A model that predicts better from the window's
  slope has found the slope useful for these labels; that is not evidence that
  the slope causes anything in an engine.

### The test that the answer file cannot leak

`RUL_FD001.txt` holds the true remaining life of each test engine at its last
recorded cycle: the answers. If any input to any model depended on that file,
changing it would change the input. `tests/test_baselines.py` copies the three
FD001 files to a temporary folder, adds 7 to every answer, and runs the
preprocessing on the original files and on the copy, once with the paper's
sensor graph and once with the graph measured from correlations in the
training engines (assumption A-27). Every input has to be identical: the
training, held-back and test windows, the training labels, the rescaling
numbers and the sensor graph. So does a stored network's prediction for every
test engine, with its data rebuilt from each. Only the test labels, and with
them the scores, move: by 7 cycles for 86 engines, by less for 3 whose answers
were within 7 of the ceiling, and not at all for the 11 whose answers were
already above 125.
It covers the answer file only. That the held-back and test engines do not help
decide the rescaling is a different question, answered where the split is made
(assumption A-25).

## Judgement calls made in this folder

The first two are cases where the paper contradicts itself and we had to pick
a side; the last two fill gaps where the paper says nothing. Full write-ups are
in `docs/ASSUMPTIONS.md`.

- **The sign of the score function.** The paper's Equation 20 and Equation 21
  imply the opposite penalty direction from the one its own prose and the cited
  2008 NASA challenge describe. We follow the prose and the challenge, so late
  (optimistic) predictions are punished harder (assumption A-15).
- **The denominator of R-squared.** The paper's Equation 23 puts the predicted
  values in the denominator, which is not the coefficient of determination. We
  use the standard formula, with the observed values (assumption A-16).
- **Which window a graph of Table 12 is.** The paper numbers the columns of
  Table 12, and the small graphs of Fig. 10, from 1 to 191 without saying which
  cycle of the engine each one is. `published.py` reads graph *g* as the
  engine's *g*-th window, the one ending at cycle *g* + 29 (assumption A-35).
- **The ordinary models' predictions are clipped.** `scripts/baselines.py`,
  described above under "Ordinary models beside the network", holds every
  prediction of an ordinary model to between 0 and the label ceiling of 125
  before scoring it, the range the network cannot leave (assumption A-38).

## References

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation
modeling for aircraft engine run-to-failure simulation. *2008 International
Conference on Prognostics and Health Management*, Denver, CO, pp. 1-9. IEEE.
https://doi.org/10.1109/PHM.2008.4711414

Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
168, 110875. https://doi.org/10.1016/j.ast.2025.110875
