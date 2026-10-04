# Audit of 2026-08-31: the paper against the code, line by line

## What this file is

A systematic comparison of Wen et al. (2026) against this repository: every
equation, every table, every figure and every numbered section of the paper,
checked against what the code does.

It exists because the previous two audits (`REVIEW_2026-08-27.md`,
`REVIEW_2026-08-29.md`) were reviews *of this repository*. This one runs in the
other direction — it starts from the paper and asks, for each thing the paper
says, whether the code does it and whether the difference is written down.

**It records agreements as well as disagreements.** An audit that lists only
what it found tells you nothing about what it looked at. The tables below name
everything checked, including the many places where the paper and the code
agree and no entry was needed.

Method: the PDF's text layer was extracted with `pypdfium2` and searched; every
table value and figure reading quoted here was taken from a page region
re-rendered at scale 9 to 11, per working rule 1. Where a number is measured
from the data, the command that produced it is named.

---

## What came out of it

**Three new assumptions, three new findings, one correction to our own wording.**

| New | About |
|---|---|
| **A-23** | An engine shorter than the window is padded by repeating its first reading |
| **A-24** | The attributions are measured against the prediction at the average window, not the average label |
| **A-25** | The scaling is fitted after the split, reversing the order Fig. 1 prints |
| **F-13** | Fig. 1 lists node features the paper's own feature selection removes |
| **F-14** | The text under Equations 9 and 10 misnumbers the weight matrices |
| **F-15** | Fig. 9 and Table 12 disagree about which sensors matter most |

**The correction.** Five places in this repository called 61,729 "Table 4's
count", "the count Table 4 gives" or "matching Table 4 to the digit". Table 4
prints no total. Its five weight-matrix shapes come to **61,600**; the remaining
**129** are the two dense-layer biases that Equations 12 and 13 print and
Table 4 does not; and the total stays at 61,729 rather than 61,857 only because
**A-12**, a choice of ours, gives the graph layers no bias. The figure is a
derivation from three places in the paper plus one of our own assumptions. All
five wordings are fixed and A-07 now carries the arithmetic.

**One entry strengthened.** A-10 deduced that the unnamed seventh dropped sensor
is `NRc` by elimination from Table 3. The paper's *own stated criterion* now
confirms it: applying "constant values, or correlation above 0.95" to the real
FD001 training set removes exactly the six sensors it names, and leaves exactly
one redundant pair among the remaining fifteen — `Nc`–`NRc` at **r = +0.9632**,
with `NRc`'s next strongest partner at +0.2466. This is one of the few places
where a procedure the paper describes reproduces the paper's own result exactly.

**One bug found in this repository's own tooling.** The documentation site read
an entry's status from the word "reversed" appearing anywhere in its title, so
A-25 — whose title says an order "reverses" Fig. 1 — was displayed as a retired
entry. Now anchored on the log's actual convention, a capitalised status word
opening the title, with three regression tests.

---

## Equations 1 to 24

| Equation | What it gives | Verdict |
|---|---|---|
| 1–6 | The LSTM gates | Not implemented — **A-11**. Nothing to check. |
| 7 | `h_i = LSTM([x_{t-w+1,s} … x_{t,s}])` — one LSTM per sensor over its own window | Confirms A-11's description of the omitted stage |
| 8 | `H1 = ReLU(A~ X W1)`, with `A~` described only as "normalized … incorporating self-loops" — a self-loop being a connection from a sensor to itself, so its own reading survives the mixing | Matches `layers.py`. The formula itself is never printed — **A-21** |
| 9, 10 | `H2`, `H3` with `W(2)`, `W(3)`, no bias term | Matches. No bias printed supports **A-12**. Text beneath misnumbers them — **F-14, new** |
| 11 | `z = Flatten(H3)` | Matches `torch.flatten(hidden, start_dim=1)` |
| 12, 13 | FC layers **with** biases `b_fc1`, `b_fc2`; sigmoid output | Matches. Biases printed here but not for the graph layers is exactly **A-12**'s argument |
| 14 | `g(z) = phi_0 + sum phi_i`, `phi_0` = "average value of target variable" | Code uses the prediction at the average window — **A-24, new**. Table 12 violates it — **F-08** |
| 15 | The Shapley weighting `|S|!(n−|S|−1)!/n!` | Matches `coalition_weights` exactly. `f(S)` undefined — **A-20** |
| 16 | The same, per time step, "n is equal to 14" | Matches `dynamic_shap` |
| 17 | Pearson correlation, used for feature selection | Reproduces the paper's 14 sensors exactly — see **A-10** |
| 18 | `S_t = a·X_t + (1−a)·S_{t−1}` | Recursive form — **A-09**. "equal to 0.5" attached to `S_{t−1}`, not `a` — **A-04** |
| 19 | `RUL_t = min(T_total − T_end, RUL_max)` | Matches `apply_piecewise_rul`. Value never printed — **A-01** |
| 20 | `d_i = y_i − y^_i` | Contradicts the text beside it about which sign is dangerous — **A-15** |
| 21 | `SF`, divisors 13 and 10 | Under Eq 20's sign these punish the safe direction harder — **A-15** |
| 22 | RMSE | Matches `metrics.rmse` |
| 23 | R², with the **predicted** values in the denominator | Not the standard definition — **A-16** |
| 24 | ASUE over underestimated engines only | Matches `metrics.asue`, including the `N_under` divisor |

---

## Tables 1 to 12

| Table | Verdict |
|---|---|
| 1 | Comparison of prior methods. Nothing to check against code. |
| 2 | 21 sensors, names and units. Matches `data/sensors.py` symbol for symbol. Fig. 1 names two of them differently — **F-13, new** |
| 3 | The adjacency matrix. Transcribed exactly; symmetry, zero diagonal, 41 edges and connectedness asserted at import. Its construction rules require an edge it lacks — **F-07** |
| 4 | The architecture. **Re-read at scale 11**: GCN-3's input is `[batch,14,64]`, consistent with GCN-2's output and `W(3) in R^(64x32)`. A low-resolution render made it look like `,32`; it is not. Every shape matches the code. Prints no parameter total — see the correction above |
| 5 | Dataset sizes. Verified against the real files: 100/100, 260/259, 100/100, 249/248 train/test engines. **All four match.** |
| 6 | The training settings: learning rate 1e-3 falling to 1e-4, 200 epochs (an epoch being one pass over the training data), batches of 32, 30 time steps, the Adam optimiser, mean squared error. All match `training/config.py`. Its **caption duplicates Table 2's** — already recorded |
| 7 | SF comparison. Internal arithmetic checked: the 17.4 % and 60.6 % improvements and the "third-best" ranking all follow from the printed values |
| 8 | RMSE comparison. The 11.3 %, 16.9 % and 21.7 % improvements follow from the printed values |
| 9 | R² comparison. **Transformer's 0.759 average exceeds ESTGNN's 0.742**, which Section 4.1 states outright and the Conclusions contradict — **F-11** |
| 10 | ASUE comparison. The 1.7 %, 11.0 % and 8.7 % improvements all follow from the printed values |
| 11 | The ablation study — the experiment that removes one component at a time and measures what the rest can do without it. Every delta, every average and every percentage recomputed from the printed values and consistent to rounding. `w/o LSTM` FD001 = **14.93**, the row this project compares against; `w/o Physics prior` = 12.34, a cost of **0.56** — the figure F-04 measures against |
| 12 | SHAP values. All 210 cells positive and the column sums span 45.51 to 16.67 without crossing zero — **F-08**. Its top-four ranking contradicts Fig. 9 — **F-15, new** |

---

## Figures 1 to 10

| Figure | Verdict |
|---|---|
| 1 | Preprocessing chain **and** the graph panel. Two new items: the chain puts normalisation before the train and test datasets — **A-25, new**; the node-feature list includes two removed sensors under names Table 2 does not use — **F-13, new**. The chain also puts the piecewise RUL after smoothing where this code puts it before; that is provably immaterial, because smoothing touches only sensor columns and the cap touches only the label column, and `pipeline.py` says so at the point of use |
| 2 | LSTM diagram. Not implemented — A-11 |
| 3 | The adjacency matrix as a heatmap, beside the graph. **Agrees with Table 3**, which is a finding rather than an assumption and is why the transcription can be trusted |
| 4 | The architecture diagram: three GCN+ReLU, Flatten, FC, FC, Sigmoid. Matches Table 4 and the code |
| 5 | Correlation heatmap "after removing the constant features". Shows exactly the 14 kept sensors; **`NRc` is absent**, independent support for A-10 |
| 6 | The truncation schematic. Draws `RUL_max` on the 120 gridline — A-01's recorded contrary indication |
| 7 | True-RUL trajectories. Plateau calibrated at **125.2** against its own ticks — A-01's principal evidence |
| 8 | True against predicted, all four sub-datasets. Nothing checkable against this code |
| 9 | SHAP beeswarm and the graph beside it. Its four "most critical" sensors contradict Table 12 — **F-15, new** |
| 10 | Fifteen dynamic graphs, G1–G191. Stage boundaries consistent with Table 12's columns and with Section 4.4's prose |

---

## Sections

| Section | Verdict |
|---|---|
| 2.1 Problem formulation | Consistent with the implementation |
| 2.3.1 Adjacency construction | The five stated rules generate 20 distinct edges; 19 are present, `phi`–`W32` is not — **F-07** |
| 3.1 Experiment environment | Lists NumPy, pandas, scikit-learn and **LightGBM 4.5.0** — a gradient-boosting library — and **no deep learning framework at all**, for a method built from LSTMs and graph convolutions. Recorded in `README.md` section 12; see the note below on why it did not become an entry |
| 3.2.2 Feature selection | Names six of seven removed sensors. Its stated criterion reproduces the paper's 14 exactly — **A-10**, now strengthened |
| 3.2.3 Data smoothing | Equation 18 and the `alpha` ambiguity — **A-04**, **A-09** |
| 3.3 Evaluation metrics | Equations 20–24 — **A-15**, **A-16** |
| 3.4 Implementation details | Both learning rates given, the schedule between them is not — **A-17**. No validation set is described anywhere; the word "validation" occurs once in the whole paper, in the Conclusions, about the dataset — **A-06**, **A-18** |
| 4.1–4.3 Results | Arithmetic checked as above |
| 4.4 Dynamic SHAP | Prose agrees with Table 12 and not with Fig. 9 — **F-15** |
| 5 Conclusions | Claims "the highest R² of 0.742" — **F-11** |

---

## Observations that did not become entries, and why

**The software environment lists LightGBM and no deep learning framework.**
Section 3.1 names NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.1.3 and LightGBM
4.5.0. None of those can build an LSTM or a graph convolution, and LightGBM is
a gradient-boosting library with no role in the method described. This is
already stated in `README.md` section 12 as part of the argument that exact
agreement with the published numbers is not a realistic target. It did not
become a finding because a finding should be a checkable statement about the
method, and this is a statement about a paragraph's completeness — the authors
plainly used *something*, and the omission is likelier editorial than
substantive.

**"Time" as the row header of Table 12's columns.** The entries are graph
indices, G1 to G191, not times. Cosmetic.

**`Ps30` jumps from 2.29 to 4.23 between G190 and G191** while every other
sensor continues to decline. Noted inside F-15 rather than separately, because
one anomalous cell in a table already shown not to satisfy its own defining
equation adds nothing.

**Equation 16 writes the subset condition as `S ⊆ N,i`** where Equation 15
writes `S ⊆ N\{i}`. A typesetting slip; the meaning is unambiguous from
Equation 15.

---

## What could not be checked

**Anything requiring the paper's code or data.** Neither was published; the data
availability statement reads "Data will be made available on request". So no
claim in the paper's results tables can be reproduced end to end, only checked
for internal consistency, which is what the arithmetic checks above do.

**The LSTM stage, and therefore the headline numbers.** A-11 declines to invent
it. Every comparison in this repository is against the paper's own
"ESTGNN w/o LSTM" row.

**A-25's cost.** Measuring what the paper's Fig. 1 ordering would be worth means
building the leaking variant and reporting a number from it. The direction is
not in doubt, so the measurement would only quantify an advantage this project
declines to take.

---

## Automated afterwards

The new numeric claims were added to `scripts/verify_claims.py`, which now
checks **41 figures** in `--fast` mode against the log, up from 30:

- A-10's redundancy analysis: that exactly one non-constant pair reaches 0.95,
  that it is `Nc`–`NRc`, and that its correlation is 0.9632;
- A-23's window counts: the shortest test history and the number of test
  engines under 30 cycles, for all four sub-datasets.

A-24's measurement needs a trained model and is recorded in that entry with its
seed and configuration rather than automated, in the same way A-01's figure
calibration is left to hand.
