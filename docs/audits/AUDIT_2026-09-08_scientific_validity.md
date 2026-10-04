# Audit of 2026-09-08 — is the code faithful to the paper, and are its results defensible?

## What this file is

An independent audit run from outside the repository, on a fresh machine, with
three questions in front of it:

1. Does the code do what Wen et al. (2026) describe, and where it departs, is
   the departure recorded?
2. Are the numbers this project reports produced by a valid experiment?
3. Do the assumptions hold up against the C-MAPSS / GNN-for-RUL literature?

Method: the repository was reconstructed from the working tree in a clean
Python 3.11 environment (PyTorch 2.14, CPU); `pytest` was run; the FD001
pipeline was re-run from the raw NASA files; the headline five-seed result was
re-trained from scratch; and Table 3, Table 11 and Table 12 of the paper were
re-extracted from the PDF's text layer and compared line by line with the code.
Every number below was computed here, not copied.

Working rule 5 applies throughout: **paper says** / **we chose** / **measured**
are kept apart, and every measurement names the command that produced it.

---

## Part 0 — The one-paragraph summary

The reproduction is honest and unusually well documented, and its headline
number reproduces independently. But three things stand between it and a
defensible scientific claim, and one of them is new and serious. **New and
serious: the paper's own adjacency matrix is rank-deficient. Six of its
fourteen sensors are structurally invisible to any graph convolution built on
it — the network's prediction is provably unchanged when their readings are
redistributed. The model, and the paper's model, see eight numbers, not
fourteen. Every per-sensor SHAP claim in the paper and in this project inherits
that.** Second: the "without physics prior" control is degenerate — the fully
connected graph normalises to a rank-1 averaging operator, so the 2.40-cycle
figure in `README.md` §8 and F-04 measures the cost of averaging all sensors,
not the value of engineering knowledge; measured against a *random* graph with
the same 41 edges, the prior is worth far less. Third: every configuration
comparison made since 2026-08-29 was decided on the test set, which A-22's own
text forbids. None of these is fatal; all three are fixable, and two of them
turn into thesis contributions rather than corrections.

---

# Part 1 — What was verified, and reproduces

| Check | Result |
|---|---|
| `pytest` in a clean environment with the real FD001 files | **170 passed**, 0 failed |
| Table 3 of the paper against `PHYSICS_ADJACENCY_ROWS` | **exact, row by row**; 41 undirected edges, symmetric, zero diagonal |
| Table 11's `w/o LSTM` FD001 figure | **14.93**, as the repository states |
| Architecture against Table 4 and Equations 8–13 | matches: `A_hat H W` with no bias in the graph layers, biases in the two dense layers, sigmoid output |
| NASA score function | correct 2008 convention (`exp(-d/13)-1` early, `exp(d/10)-1` late); the paper's Eq. 20/21 sign contradiction is correctly resolved (A-15) |
| R², ASUE, RMSE | correct; A-16's departure from the paper's Eq. 23 is right |
| Label construction, test labels from `RUL_FD001.txt` | correct |
| Split hygiene | validation engines are separated **before** the scaler is fitted (A-25); no test data reaches any fitted statistic |
| SHAP implementation | coalition weights, the subset enumeration and the efficiency check are all correct; exact, not sampled |
| Headline five-seed result | **re-trained here: 16.706 ± 0.294** over three seeds, against the stored 16.818 ± 0.157. Reproduces. |

Two caveats on that last row. Reproduction is **statistical, not bit-exact**:
seed 42 came out at 16.565 here against 16.591 stored, and seed 43 at 16.513
against 16.838. Different PyTorch version and different hardware are enough.
F-04's claim of bit-identical reproduction is a claim about one machine, and
should say so.

---

# Part 2 — Defects, ranked by what they do to a conclusion

## D-1. The paper's graph is rank-deficient: the model sees 8 numbers, not 14

**This is the finding of the audit.** It is not recorded anywhere in the
repository, and it is a property of the paper, not of this code.

### The mechanism

Add self-loops to Table 3 and normalise, either way the code offers. Three
groups of sensors then have **identical rows in `A_hat`**:

```
{P15, Nf, NRf, BPR}     {P30, Ps30}     {Nc, W31}
```

In graph terms these are *true twins*: they have the same closed
neighbourhood, so `N[i] = N[j]`. Two consequences follow immediately and hold
for **any** weights, trained or not, and for **both** normalisations.

**First, their representations are identical at every layer.** One graph
convolution computes `(A_hat H)_i = sum_k A_hat[i,k] H_k`. If rows `i` and `j`
of `A_hat` are equal then that sum is the same number for both, whatever `H`
is. So after GCN-1, P15, Nf, NRf and BPR carry the *same* 32 numbers; after
GCN-2 the same 64; after GCN-3 the same 32 again. Measured on an untrained
network:

```
after GCN-1: 9 distinct node representations out of 14
after GCN-2: 9 distinct node representations out of 14
after GCN-3: 9 distinct node representations out of 14
max |h_i - h_j| within each group after GCN-3: 0.000e+00
```

The flattened 448-vector therefore contains only `8 x 32 = 256` independent
numbers, and 24,576 of `W_fc1`'s 57,344 weights — 40 % of the whole model —
are identifiable only as sums.

**Second — and this is the sharper statement — the columns are identical too**,
so the network's input is not the fourteen readings but eight numbers derived
from them. Because true twins have equal degree, `A_hat[k,i] = A_hat[k,j]` for
every `k`, so the first layer sees `x_i` and `x_j` only through `x_i + x_j`.
Redistributing a group's readings while holding the group's sum fixed changes
the prediction by exactly zero:

```
original prediction: 64.315575
  P15+Nf+NRf+BPR  redistributed (sum kept)  -> 64.315575   change 0.00e+00
  P30+Ps30        redistributed (sum kept)  -> 64.315575   change 0.00e+00
  Nc+W31          redistributed (sum kept)  -> 64.315575   change 0.00e+00
  T24+T30  (control, not twins)             -> 64.293121   change 2.25e-02
```

`A_hat` has **rank 8**; its singular values are
`[1.017, 0.918, 0.442, 0.223, 0.168, 0.126, 0.079, 0.074, 0, 0, 0, 0, 0, 0]`.
The six blind directions are the five twin differences plus one more:

```
T30 + htBleed - 0.5 * (P30 + Ps30 + Nc + W31)
```

Adding any multiple of that vector to the input — including 25x — leaves the
prediction bit-identical. The network is blind to a six-dimensional subspace
of its own input space.

### What it does to this project's own SHAP numbers

Measured with the repository's own `explain.shap_values`, on a trained network
(seed 42, current default), over all 100 FD001 test engines — mean `|phi|` in
cycles:

```
 1. T24      9.105     6. T30      4.977    11. W32      2.004
 2. T50      8.755     7. Nf       3.508    12. P30      1.959
 3. Nc       8.481     8. NRf      3.438    13. phi      1.593
 4. W31      7.547     9. BPR      2.935    14. P15      0.616
 5. htBleed  5.141    10. Ps30     2.260
```

Inside `{P15, Nf, NRf, BPR}` — four sensors the network reads only as a sum —
SHAP reports a **5.7-fold spread**, from P15 at 0.616 to Nf at 3.508. And the
ordering of those four attributions is exactly the ordering of their mean
distance from the background value:

```
mean |phi|        P15 0.616   Nf 0.775*  NRf 0.773*  BPR 0.714*     (*: |x-b|)
mean |x - b|      P15 0.265   Nf 0.775   NRf 0.773   BPR 0.714
Spearman(|phi| rank, |x-b| rank) over the four = +1.000
```

That is not a coincidence and it is not an empirical regularity — it is forced.
Since the network only sees `x_i + x_j + ...`, the marginal contribution of
sensor `i` to any coalition depends on `i` only through `x_i - b_i`. So within
a twin class, SHAP is measuring **how far each reading sits from the
background**, and nothing about the sensor's role in the model. (The
two-member groups also show `|rho| = 1`, but with n = 2 that is forced by
arithmetic and carries no information; the four-member group is the
informative case.)

### Why it matters, in order

1. **It disables the thesis's XAI target.** SHAP here masks node features. For
   twins, the marginal contribution of `i` to any coalition depends on `i` only
   through `x_i - b_i`, its displacement from the background value, because the
   network only ever sees the group sum. So the split of importance between
   P15, Nf, NRf and BPR is a fact about the *input distribution*, not about the
   model's use of those sensors. "The model relies on sensor X" is not a
   statement this architecture can support for 6 of its 14 sensors.
2. **It contradicts the paper's own interpretability narrative.** Table 12
   gives, at graph G1: `Nc 4.20` (bold, one of the four "most critical") and
   `W31 2.30` (near the bottom) — for two sensors the network cannot separate.
   Likewise `Ps30 4.69` against `P30 3.33`, and a spread of 1.79 to 3.77
   inside `{P15, Nf, NRf, BPR}`. Section 4.4's story — "early stage the model
   relies on T50, **Nc**, **Ps30**, phi… intermediate stage T50, **P30**,
   **Ps30**, phi" — describes a shift from Nc to P30 between two halves of
   channels the model reads as single sums.
3. **It applies to the paper's full model too.** Per-sensor LSTMs give each
   node its own scalar, but the collapse happens at the graph stage, which is
   identical. The paper's ESTGNN is also a function of eight numbers.
4. **It explains part of the parameter story.** 61,729 parameters act on an
   8-dimensional input.

### What to do with it

Do **not** "fix" the graph — the graph is the paper's, and reproducing it is
the point. Record it as a finding about the source (an `F-xx` entry), add an
import-time computation of `A_hat`'s rank and twin classes to
`graph/adjacency.py` so the fact is visible, add a regression test asserting
the three twin classes and the zero-change property, and make the explanation
chapter of the thesis start from it. It is a stronger result than any accuracy
number in this project: it says the source paper's headline contribution —
per-sensor dynamic explanations — is not identifiable for 43 % of its sensors,
and it is provable rather than empirical.

---

## D-2. The "without physics prior" control is degenerate

`build_fully_connected_adjacency` produces, after either normalisation, a
matrix in which **every entry is 1/14 and every row is identical**:

```
physics   row        distinct rows 9/14   rank 8
physics   symmetric  distinct rows 9/14   rank 8
complete  row        distinct rows 1/14   rank 1
complete  symmetric  distinct rows 1/14   rank 1
```

A rank-1 `A_hat` replaces every node by the mean of all fourteen. After GCN-1
the fourteen node representations are *all* identical, and the flattened
448-vector is one 32-vector repeated fourteen times. The variant does not
remove engineering knowledge; it removes thirteen of the fourteen degrees of
freedom in the input.

So the entry in `README.md` §8 — "**The physics prior is worth 2.40 cycles**…
paired `t = 41.6, p = 0.000002`" — and the same claim in F-04, do not measure
what they say. The p-value is real and the effect is real; what is being
measured is the cost of averaging every sensor into one number.

The paper's own figure supports this reading: Wen et al. report the same
ablation costing **0.56** cycles, an order of magnitude less than 2.40. A
control that is nine times more damaging than the original's is not the same
control.

**The correct control** is standard in the GNN literature and cheap here: a
random graph with the same number of edges, and a degree-preserving rewiring of
the physics graph itself. Both keep the message-passing budget and destroy only
the engineering content. Measured here (three seeds each, everything else at
the current default):

| Graph | RMSE | NASA score | Â rank |
|---|---|---|---|
| **physics (paper Table 3)** | **16.706 ± 0.291** | **624.4 ± 42.4** | 8 |
| degree-preserving rewire #1 | 16.975 ± 0.225 | 778.9 ± 67.6 | 14 |
| degree-preserving rewire #2 | 17.116 ± 0.171 | 758.9 ± 80.4 | 14 |
| degree-preserving rewire #3 | 17.506 ± 0.073 | 782.6 ± 46.1 | 14 |
| random, 41 edges #1 | 17.727 ± 0.258 | 771.8 ± 74.6 | 14 |
| random, 41 edges #2 | **16.820 ± 0.236** | 780.6 ± 54.1 | 14 |
| random, 41 edges #3 | 18.650 ± 0.267 | 1502.2 ± 124.6 | 14 |
| identity (self-loops only, no message passing) | 20.992 ± 1.163 | 1513.7 ± 292.3 | 14 |
| complete graph (this project's A-19 control, stored 5-seed run) | 19.214 ± 0.056 | 781.7 ± 9.0 | **1** |

Three seeds per graph; the degree-preserving rewirings keep the physics graph's
exact degree sequence and change only which sensors are joined to which.
Treating the **graph** as the unit of analysis, which is the right unit for the
question "does this particular wiring matter":

| Comparison with the physics graph | mean | sd | range | one-sample t (n = 3 graphs) | p |
|---|---|---|---|---|---|
| RMSE, degree-preserving rewirings | 17.199 | 0.275 | 16.975 – 17.506 | 3.10 | **0.090** |
| RMSE, random 41-edge graphs | 17.732 | 0.915 | 16.820 – 18.650 | 1.94 | 0.192 |
| NASA score, degree-preserving rewirings | 773.5 | 12.8 | 758.9 – 782.6 | 20.24 | **0.002** |
| NASA score, random 41-edge graphs | 1018.2 | 419.2 | 771.8 – 1502.2 | 1.63 | 0.245 |

**Read that carefully, because it is a better result than the one it replaces.**

- **On RMSE the physics prior cannot be distinguished from a random rewiring
  of itself** (0.49 cycles, p = 0.09 over three rewirings). One of the three
  random 41-edge graphs, drawn without any engineering input at all, landed at
  16.820 ± 0.236 against the physics graph's 16.706 ± 0.291 — the same number.
- **On the NASA score it can**: 624 against 773, with every one of the six
  randomised graphs worse, and the tightest comparison (degree-preserving,
  which controls for the degree sequence) at p = 0.002. The physics graph's
  advantage is that it makes *safer* errors, not smaller ones.
- Message passing itself is worth a great deal: the identity graph — every
  sensor seeing only itself — costs 4.3 cycles.
- And the complete graph sits at 19.21, worse than every randomised graph
  including the identity one on RMSE, which is what a rank-1 operator should
  do.

The honest headline is therefore: *under this configuration, the physics graph
is worth about half a cycle of RMSE against a matched random wiring — inside
the noise — and about 150 points of NASA score, which is not.* That is a
narrower claim than "2.40 cycles at p = 0.000002", and it is one that survives
a reviewer.

Report that instead. Keep the complete-graph number, relabelled as what it is:
an upper bound on what uniform mixing costs, useful as a sanity check and not
as an ablation of the prior.

---

## D-3. Every configuration comparison since 2026-08-29 was decided on the test set

A-22 says, in bold: *"any future comparison between configurations must be run
with `config/fd001_validated.yaml`, not this one."* The four runs stored since
then were all run on the defaults:

| Run directory | config digest | `validation_fraction` |
|---|---|---|
| `fd001_physics_row_cap_5seed_varsplit_cfg722cd2_…` | 722cd2 | **0.0** |
| `fd001_physics_symmetric_cap_5seed_varsplit_cfg2b37a4_…` | 2b37a4 | **0.0** |
| `fd001_complete_row_cap_5seed_varsplit_cfgd20b12_…` | d20b12 | **0.0** |
| `fd001_physics_row_cap_5seed_varsplit_exponential_cfg722cd2_…` | 722cd2 | **0.0** |

(`config/fd001_validated.yaml` hashes to `ba82e3`, and no run since 2026-08-28
carries it.) So physics-vs-complete, row-vs-symmetric and step-vs-exponential
were all settled by looking at the test engines. The results are not thereby
wrong, but they are **selected**, and any RMSE quoted from a configuration
chosen this way is optimistically biased by an unmeasured amount. A reviewer
will find this immediately, because A-22 tells them where to look.

The fix is not to redo everything. It is to run the comparisons that decide a
default under `fd001_validated.yaml`, then report the chosen configuration's
test number once. Two of the three comparisons cost eight minutes each.

## D-4. The provenance metadata contradicts the runs it describes

Every stored run is named `…_varsplit_…` and every `summary.json` says

```json
"split": "varies_with_seed",
"early_stopping": true
```

With `validation_fraction = 0.0` there is no split to vary and early stopping
can never fire — `results.csv` confirms it, with `validation_rmse` empty and
`best_epoch = 200` on all five seeds. `scripts/train.py` derives both fields
from command-line flags rather than from the configuration that was actually
used. Under working rule 3a, which exists precisely because a results file once
survived that no code produced, this is the same class of problem: the file
describes an experiment that did not happen. Both fields should be derived from
`preprocessing.validation_fraction`.

## D-5. Nothing in the project shows the graph network beats an ordinary model

The single most likely question from an examiner or a reviewer is: *given
fourteen numbers per example, does the graph buy anything over a standard
regressor on the same numbers?* Measured here on exactly the arrays the
pipeline hands the network (`prep.train.features`, `prep.test.features`, FD001,
current default):

| Model on the same 14 features | RMSE | NASA score | R² | ASUE |
|---|---|---|---|---|
| **ESTGNN, physics graph, 5 seeds (stored)** | **16.82 ± 0.16** | **610 ± 25** | 0.824 | 12.33 |
| Gradient boosting (scikit-learn defaults) | 16.94 | 860 | 0.821 | **9.32** |
| MLP, one hidden layer of 64 | 17.68 | 940 | 0.805 | 11.28 |
| Random forest, 200 trees | 17.87 | 1040 | 0.801 | 11.39 |
| Quadratic ridge regression | 18.75 | 1182 | 0.781 | 11.86 |
| Linear regression | 20.39 | 1040 | 0.741 | 14.25 |
| Best possible constant | 40.07 | 10623 | 0.000 | 31.39 |

**On RMSE the graph network's margin over an off-the-shelf gradient booster is
0.12 cycles, smaller than its own seed-to-seed standard deviation of 0.16.** It
wins clearly on the NASA score (610 against 860) and loses on ASUE. That is a
defensible, publishable set of statements — but it has to be stated, and right
now the project's results section compares only against the paper.

Add these rows to `README.md` §8. They cost one script and they turn "we got
16.82" into "we got 16.82, here is what 16.82 is worth".

## D-6. The model is not spatio-temporal, and the 30-cycle window is nearly inert

`window_size = 30` and A-07 takes the last cycle, so the only temporal
information reaching the network is what the exponentially weighted mean
carries. With `alpha = 0.5`:

```
weight on the reading k cycles back:  lag 0: 0.500   lag 1: 0.250   lag 2: 0.125
cumulative over the last 6 readings:  0.984
```

98.4 % of the weight sits in the last six cycles, and the value the network
receives is almost the raw reading at that cycle:

```
variance of the node feature explained by the single raw reading at t (r^2)
  T24 0.885   T30 0.867   T50 0.936   P15 0.750   P30 0.932   Nf 0.942
  Nc  0.990   Ps30 0.953  phi 0.946   NRf 0.941   BPR 0.909   htBleed 0.881
  W31 0.901   W32 0.902
```

So the model is a **static per-cycle regressor on lightly smoothed sensor
values**, wrapped in a graph. `window_size` does one thing only: it decides
that cycles 1–29 of each engine are never a prediction point. That is a correct
consequence of A-07 plus A-11 and it is defensible — but the words
"spatiotemporal" and "30-cycle window" should not appear in the thesis
describing *this* configuration without that sentence beside them.

There is also a consequence for the removed-LSTM decision that the log does not
draw: because the node feature is nearly the raw reading, the paper's
"w/o LSTM" row is not obviously the right comparison. Their ablation removes an
LSTM from a model that still receives a 30-cycle window somewhere; this one
removes the window as well.

## D-7. A-12 plus A-07 make the graph stage positively homogeneous

The graph layers have no bias (A-12) and each node carries one number (A-07).
Every graph layer is then `relu(A_hat H W)`, which satisfies
`relu(A_hat (cH) W) = c relu(A_hat H W)` for any `c > 0`. Measured:

```
max |G(cx) - c G(x)|   c = 0.5: 0.0    c = 2: 0.0    c = 10: 1.2e-07
control, c = -1:       5.5e-02
```

The whole three-layer graph stage can only compute a positively homogeneous
function of its input; every threshold-like behaviour in the network has to be
supplied by `b_fc1`. Worse, after GCN-1 each node's nominally 32-dimensional
representation lies on one of exactly two rays, chosen by the sign of the
scalar `(A_hat x)_i`, with length `|(A_hat x)_i|` — the 32 channels carry no
more information than the one scalar did.

This is a faithful reading of Equations 8–10, so it is not an error. It is
worth an entry, because it says the graph stage of the printed architecture
does far less work than its parameter count suggests, and because it predicts
that adding a bias (`gcn_bias=True`, already implemented) should help. That is
a one-command experiment and it has never been run.

## D-8. Seed spread is not uncertainty, and it is being used as though it were

With `validation_fraction = 0.0` the five seeds vary only the weight
initialisation and the batch order. The training set, the validation set (none)
and the test set are identical across seeds. So `± 0.157` measures
initialisation noise, and nothing else — not sampling variability over engines,
which is what a reader will assume it means.

The relevant uncertainty is over the **100 test engines**:

One trained network (seed 42, current default, RMSE 16.565), resampling the
100 test engines 20,000 times with replacement:

| Measure | Point estimate | 95 % bootstrap interval over the test engines | sd |
|---|---|---|---|
| RMSE | 16.56 | **[13.76, 19.21]** | 1.39 |
| NASA score | 611 | [314, 982] | 173 |

**The sampling uncertainty over engines is about nine times the seed-to-seed
spread** (1.39 against 0.157). Three consequences, and the first is good news:

1. **The paper's `w/o LSTM` figure of 14.93 lies inside that interval.** The
   reproduction is not statistically distinguishable from the published number
   on this test set. `README.md` §8 currently frames the difference as a
   "shortfall" — 5.4 cycles in the stale text, 1.89 under the current default —
   as though it were an established gap. It is not: with 100 engines, neither
   this project nor the paper can resolve a difference of that size. That is a
   better and more defensible statement than either version, and it also says
   something about the paper: single-run results on a 100-engine test set
   cannot support the 0.1–0.5 cycle margins its Tables 7–10 are built on.
2. **A-21's 0.35-cycle difference between row and symmetric normalisation is a
   quarter of one bootstrap standard deviation.** It is a fact about these
   hundred engines and these five initialisations, not about the two formulas.
3. Every paired-across-seeds p-value in the log — A-21's 0.047, F-04's
   0.000002, A-02's 0.00002, A-17's 0.39 — answers *"is this difference an
   initialisation artefact?"* and nothing more. None of them licenses a claim
   about engines the model has not seen. Say so wherever they appear.

Two rules follow. A paired test across seeds answers "is this difference an
initialisation artefact?" and nothing more — so A-21's `p = 0.047`, F-04's
`p = 0.000002` and A-17's `p = 0.39` should all be described that way. And a
difference much smaller than the bootstrap interval is not a difference between
*methods*, however small its p-value; it is a difference on these hundred
engines.

## D-9. Stale numbers presented as current

- **`README.md` §8** gives the current default as 16.82 ± 0.16 and then, two
  paragraphs later, says "the comparison is 20.35 ± 0.75 against 14.93 — a
  5.4-cycle shortfall", twice. Under the current default the shortfall is
  **1.89**. The section states two different answers to its own headline
  question.
- **A-02's** +1.311-cycle cost of capping the test labels was measured at
  `cdcc827-dirty` under the superseded min-max configuration (the one that
  produced 20.353). Its own text says the effect is configuration-dependent and
  must be re-measured; it has not been. `config/fd001_uncapped.yaml` exists and
  no run in `reports/` uses it.
- **F-06** still contains "**61,729 parameters — what Table 4's count
  exactly**", one of the five wordings the 2026-08-31 audit reports having
  fixed, and now ungrammatical as well.

## D-10. Training curves are discarded

`scripts/train.py` builds a `TrainingHistory` and never calls `.save()`. The
stored run keeps four test metrics and `best_epoch`, and throws away the
per-epoch training and validation loss. With 200 fixed epochs, no early
stopping, no weight decay and no dropout, the one question a reader will ask
about a stored run — *was it still improving, or had it overfitted?* — cannot
be answered from anything in `reports/`. One line fixes it.

## D-11. `ASUE = 0` is documented as the best possible value; it is the worst

`metrics.asue` returns `0.0` when no engine was underestimated and logs "That
is the best value this measure can take, not a missing result." ASUE is the
average size of the *conservative* errors. A model that never underestimates is
one that is late on every single engine — the most dangerous behaviour the NASA
score exists to punish. The correct treatment is to return `nan` with a warning
naming the reason, and to report `n_under` alongside. The case never fires on
FD001 (38–45 engines are underestimated), so no stored number is wrong; the
docstring is.

---

# Part 3 — The assumptions against the C-MAPSS and GNN-for-RUL literature

| ID | Choice | Standing in the field |
|---|---|---|
| A-01 | `RUL_max = 125` | **Standard.** 125 or 130; 125 dominates the post-2018 literature. Well supported by the Fig. 7 calibration. |
| A-02 | Cap the **test** labels too | **Divides the field, and moves the number favourably.** Handled correctly: measured, disclosed, and `fd001_uncapped.yaml` provided. The measurement is stale (D-9). |
| A-03 | z-score, global | **Standard for FD001.** Not adequate for FD002/FD004, which need per-condition scaling; the code offers per-condition min-max but no per-condition z-score. |
| A-04 | `alpha = 0.5` | Light by C-MAPSS standards, where the usual smoothing is a moving average over 5–10 cycles. Correct reading of a broken sentence. Its consequence (D-6) is under-stated. |
| A-05 | stride 1 | **Standard.** |
| A-06/A-18 | no validation split, no early stopping | Faithful to the paper, and the honest choice. But it removes the only legitimate basis for choosing between configurations, which is how D-3 happened. |
| A-07 | one number per node = last cycle | **Outside the field.** F-12 already establishes that none of twelve neighbouring papers does this. The planned mean-vs-last experiment (`docs/EXPERIMENT_A-07_mean_window.md`) is still not started, and should be. |
| A-11 | omit the LSTM | Reasonable, disclosed, and compared against the paper's own ablation row. Author's decision; unchanged. |
| A-12 | no bias in graph layers | Faithful to Equations 8–10 and matches Kipf & Welling's own default. Its consequence (D-7) has never been measured. |
| A-13 | targets scaled by `RUL_max` | **Standard**, and forced by the sigmoid. |
| A-15 | NASA sign convention | **Correct**, and the paper is wrong. |
| A-16 | standard R² | **Correct**, and the paper's Eq. 23 is not R². |
| A-17 | one x0.1 step at the halfway epoch | Well argued from Li et al.; measured indistinguishable from exponential. |
| A-19 | complete graph as the "no prior" control | **Wrong control** — see D-2. Replace with a random / degree-preserving graph. |
| A-20 | absent sensor = its mean training window | **Standard SHAP practice.** Correct, and the sensitivity to the choice of background is already measured. |
| A-21 | row normalisation | Both forms are in the literature; symmetric is the more common. Selected on test error (D-3). |
| A-23 | drop engines shorter than the window | **Standard**, correctly sourced, and irrelevant on FD001. |
| A-25 | split before fitting the scaler | **Correct, and stricter than the paper.** Keep. |

Two gaps in the assumption set itself, neither of which has an entry:

- **The rank deficiency of `A_hat` is not an assumption but a fact about the
  source, and it is not recorded** (D-1).
- **No entry covers regularisation.** There is none: no weight decay, no
  dropout, 200 fixed epochs on 17,731 windows with 61,729 parameters and no
  held-out set. The paper does not mention any either, so the choice is
  faithful; but "the paper is silent and we added nothing" is itself a decision
  under rule 4.

---

# Part 4 — What to do, in the order I would do it

1. **Write the rank finding up as `F-16`**, with the twin classes, the
   rank-8 fact, the sum-invariance demonstration and the sixth blind
   direction. Add the assertion to `graph/adjacency.py` and two regression
   tests. *(Half a day. This is the highest-value item in the project.)*
2. **Rewrite the physics-prior ablation** around random and degree-preserving
   graphs; relabel the complete-graph run as the uniform-mixing bound. Update
   F-04 and `README.md` §8. *(Runs already done below; the writing is the work.)*
3. **Re-run the three configuration comparisons under
   `config/fd001_validated.yaml`** and say which default each one chooses.
   *(~25 minutes of compute.)*
4. **Add the non-graph baselines** to `README.md` §8. *(One script.)*
5. **Fix the provenance fields**, save `TrainingHistory`, fix the three stale
   numbers, fix the ASUE docstring. *(An hour.)*
6. **Run the two one-command experiments that have never been run**:
   `--node-feature mean` (A-07) and `gcn_bias=True` (A-12).
7. Only then FD002/FD004, and only after the smoothing-before-scaling order is
   settled for multi-condition data — `README.md` §12 already flags it.

---

# Part 5 — What is explicitly not wrong

So that none of this gets re-litigated:

- the transcription of Table 3;
- the architecture, the parameter count and the layer widths;
- all four metric implementations, including both places where the paper is
  wrong and the code is right;
- the split hygiene and the order of splitting and scaling;
- the SHAP implementation, which is exact and passes its own efficiency check;
- the decision to omit the LSTM, and the decision to compare against the
  paper's own `w/o LSTM` row;
- the assumptions log itself, which is better than most published work.

The problems above are about what the numbers are being asked to *mean*, not
about whether the code computes what it says it computes. It does.

---

# Part 6 — Reproducing this audit

Environment: Python 3.11.15, PyTorch 2.14.0, NumPy/pandas/scikit-learn/networkx
current as of 2026-09-08, CPU only, 2 cores. Repository reconstructed from the
working tree; `pip install -e .`; `pytest` → 170 passed.

**The rank and twin structure** (no training needed, holds for any weights):

```python
from thesis_gnn_rul_xai.graph.adjacency import build_physics_adjacency, normalize_adjacency
from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_SYMBOLS as S
import numpy as np
A, _ = build_physics_adjacency()
for method in ("row", "symmetric"):
    An = normalize_adjacency(A, method=method).astype(np.float64)
    groups = {}
    for i in range(14):
        groups.setdefault(tuple(np.round(An[i], 12)), []).append(S[i])
    print(method, np.linalg.matrix_rank(An), [g for g in groups.values() if len(g) > 1])
```

**The sixth blind direction**: take the right null space of `A_hat`
(`Vt[s < 1e-10]` from its SVD), project out the five twin differences
`e_i - e_j`, and the remainder is
`T30 + htBleed - 0.5 (P30 + Ps30 + Nc + W31)`. Adding any multiple of it to a
trained model's input leaves the prediction unchanged to float32 precision
(`0.00e+00` at 1x and 10x, `1.5e-05` at 100x).

**The graph ablation**: 24 training runs, 8 graphs x 3 seeds, everything else
at `PreprocessingConfig()` defaults, `TrainingConfig(seed=s, device="cpu")`.
Random graphs drawn uniformly over 41-edge connected graphs on 14 nodes;
degree-preserving rewirings by `networkx.double_edge_swap(nswap=300)` on the
physics graph, retried until connected. About 4.7 minutes per run.

**The non-graph baselines**: scikit-learn defaults on
`prepared.train.features[:, :, 0]` and `prepared.test.features[:, :, 0]`,
scored with this repository's own `evaluation.metrics.evaluate`.

**The bootstrap**: 20,000 resamples with replacement of the 100 test engines,
percentile interval.

Every figure quoted in this file came from one of those five procedures. None
was carried over from an earlier document.
