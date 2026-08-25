# Assumptions log

## Why this file exists

This project re-implements the method of Wen et al. (2026). Their paper does
not describe every detail needed to write the code. Wherever a value or a
procedure had to be chosen by us instead of read from the paper, the choice is
recorded here.

Keeping this log is not paperwork. It is what separates a re-implementation
from a guess. When a result from this code differs from the published one, the
first question a reader (or an examiner) will ask is "what did you do
differently?", and this file is the answer. It also marks the boundary between
what the paper claims and what this code does, so that neither is mistaken for
the other.

**Rule followed in the code:** every one of these choices appears at the point
where it is used, in a comment beginning with `ASSUMPTION`, together with the
identifier below.

## Summary

| ID | Topic | What the paper says | What this project does |
|----|-------|---------------------|------------------------|
| A-01 | Label ceiling `RUL_max` | Defines it symbolically in Equation 19, never prints a value | 125 cycles |
| A-02 | Ceiling on test labels | Not discussed | Applied, because the model's output layer cannot exceed the ceiling |
| A-03 | Input rescaling | Not described at all | Per-condition min-max scaling to [0, 1], following Jin et al. (2025) |
| A-04 | Smoothing strength | Self-contradictory sentence in Section 3.2.3 | Read as `alpha = 0.5` |
| A-05 | Window stride | Not stated | 1 cycle |
| A-06 | Validation split | None; trains for a fixed 200 epochs | 20 % of engines held out |
| A-07 | Node feature content | Table 4 implies one number per sensor, produced by an LSTM that is never specified | The whole 30-cycle window per sensor (configurable) |
| A-08 | Order: smoothing and scaling | Not stated | Smooth first, then rescale |
| A-09 | Smoothing formula variant | Gives the recursive form as Equation 18 | Recursive form (`adjust=False` in pandas) |
| A-10 | Identity of the 7th dropped sensor | Says "seven", names six | The seventh is `NRc`, deduced from their Table 3 |
| A-11 | The LSTM stage | Central to their architecture | Omitted, matching their own published ablation |

---

## A-01 — The label ceiling is 125 cycles

**Where:** `config.PreprocessingConfig.rul_max`, `data/labeling.py`

**The gap.** Equation 19 of the paper reads
`RUL_t = min(T_total − T_end, RUL_max)`, and the text says only that `RUL_max`
"indicates the early RUL threshold". The number itself never appears.

**What we chose.** 125 cycles.

**Why.** Two independent indications point to it. First, Section 4.2 of the
paper describes the predicted life staying "relatively stable values around
125" while an engine is still healthy, which is exactly the flat part of the
capped label curve. Second, 125 is the value used by Jin et al. (2025) on the
same dataset, and it sits inside the range of 120 to 130 that Heimes (2008)
reported as effective when he introduced this kind of label.

**How much it matters.** Moderately. Changing the ceiling changes every label
above it and therefore every error figure. A sensitivity check over the values
120, 125 and 130 is planned so the effect can be measured rather than assumed.

---

## A-02 — The ceiling is applied to the test labels as well

**Where:** `config.PreprocessingConfig.clip_test_labels`, `data/pipeline.py`

**The gap.** The paper does not say whether the official test labels are capped
before the error is computed.

**What we chose.** Yes, cap them.

**Why.** The final layer of the model in Table 4 of the paper is a sigmoid
function, whose output lies between 0 and 1 and is then scaled by the ceiling.
The model therefore **cannot** predict a value above `RUL_max`, whatever the
input. Several engines in the official label file have a true remaining life
above 125. Leaving those labels uncapped would charge the model for an error
that its own architecture makes impossible to avoid, and the resulting figure
would not measure what it claims to measure.

**How much it matters.** On FD001 this affects only the small number of test
engines whose true remaining life exceeds the ceiling, but it shifts the error
measurably. This choice must be stated whenever a number from this code is
compared with a published one.

---

## A-03 — Sensor values are rescaled with per-condition min-max scaling

**Where:** `config.PreprocessingConfig.normalization`, `data/normalization.py`

**The gap.** The paper never describes how the sensor inputs are scaled. The
words "normalisation", "min-max" and "standardisation" do not appear in
connection with the inputs, and operating conditions are never mentioned in the
preprocessing section.

**What we chose.** Min-max scaling to the interval [0, 1], computed separately
for each operating condition, with all statistics learned from the training
engines only.

**Why.** Some form of rescaling is unavoidable: the 14 sensors differ by three
orders of magnitude, and a neural network trained on unscaled inputs of that
kind does not converge usefully. Among the possible choices, this one is taken
from Jin et al. (2025), who study the same dataset with a comparable
graph-based architecture and describe their procedure explicitly. Min-max
scaling to [0, 1] is also the choice that fits the rest of the architecture:
the sigmoid output layer already assumes targets in [0, 1].

**Honesty note.** This is a *substitution from a related paper*, not a
reconstruction of what Wen et al. did. Their procedure is unknown. On FD001
there is only one operating condition, so the per-condition variant and the
global variant give identical results; the distinction becomes important only
when FD002 and FD004 are added.

**How much it matters.** On FD001, little, as long as some sensible scaling is
used. On FD002 and FD004 it is likely to be the single largest source of
difference from the published numbers.

---

## A-04 — The smoothing parameter is 0.5

**Where:** `config.PreprocessingConfig.smoothing_alpha`, `data/smoothing.py`

**The gap.** Section 3.2.3 of the paper defines the filter as
`S_t = alpha * X_t + (1 − alpha) * S_(t−1)` and then writes: *"S(t-1) indicates
the previous smoothed value, which is equal to 0.5 in this study; alpha
constitutes the smoothing parameter"*. As written, the number 0.5 is attached
to the previous smoothed value rather than to `alpha`.

**What we chose.** Read the sentence as `alpha = 0.5`.

**Why.** In a recursive formula, `S_(t−1)` is a different quantity at every time
step; it cannot be a single fixed number "in this study". The only reading that
makes the sentence meaningful is that 0.5 belongs to the smoothing parameter,
which is the quantity a paper would normally report.

**How much it matters.** `alpha = 0.5` is light smoothing: half the weight
stays on the current reading. The setting can be disabled entirely by setting
it to `null` in the configuration, and the effect on the final error is
straightforward to measure.

---

## A-05 — Windows are taken one cycle apart

**Where:** `config.PreprocessingConfig.window_stride`, `data/windowing.py`

**The gap.** Table 6 of the paper gives a window of 30 time steps but no stride.

**What we chose.** A stride of 1 cycle.

**Why.** A stride of 1 is the usual choice on this dataset and produces the
largest number of training examples from a fixed amount of data. A larger
stride would reduce the training set without any stated benefit.

**How much it matters.** Small for accuracy, large for training time.

---

## A-06 — Twenty percent of engines are held out for validation

**Where:** `config.PreprocessingConfig.validation_fraction`, `data/windowing.py`

**The gap.** The paper describes no validation split. Table 6 states 200
epochs, with no early stopping and no held-out set mentioned.

**What we chose.** Hold out 20 % of the training **engines** (not windows).

**Why.** Without a validation set there is no honest way to decide when to stop
training or to compare settings; those decisions would have to be made by
looking at the test set, which invalidates the test set as a measure of
performance. The split is made by engine because consecutive windows from one
engine overlap by 29 of their 30 cycles, so splitting windows at random would
place nearly identical examples on both sides and give a falsely low validation
error.

**Honesty note.** This is a deliberate **improvement** on the source, not a
reconstruction. It should be described as such in the thesis.

---

## A-07 — Each sensor node carries its whole time window

**Where:** `config.PreprocessingConfig.node_feature_mode`, `data/windowing.py`

**The gap.** Table 4 of the paper gives the input to the first graph layer as
`[batch, 14, 1]`: one number per sensor. That number is produced by a per-sensor
LSTM whose size, depth, and output projection are never stated anywhere in the
paper.

**What we chose.** With the LSTM stage omitted (see A-11), each sensor node
carries the full 30-cycle window, giving `[batch, 14, 30]`. The alternative
`last_step` mode, which reproduces `[batch, 14, 1]` by keeping only the most
recent reading, is available in the configuration.

**Why.** With no temporal encoder, passing only the last reading would discard
29 of the 30 cycles in each window and make the window size meaningless. Giving
the graph layer the whole window is the standard form of this ablation and
keeps the temporal information available to the model.

**How much it matters.** Large. This changes the width of the first weight
matrix and the amount of information the model receives.

---

## A-08 — Smoothing is applied before rescaling

**Where:** `data/pipeline.py`

**The gap.** The paper does not state the order of its preprocessing steps.

**What we chose.** Smooth first, then rescale.

**Why.** Measurement noise lives in the sensor's own physical units, so the
filter is most naturally applied there. Rescaling is the last step before the
model. Both operations are close to linear, so the reverse order would give an
almost identical result, but with this order the training set maps exactly onto
the intended [0, 1] range, which makes the pipeline easier to check.

**How much it matters.** Very small.

---

## A-09 — The recursive form of the smoothing filter is used

**Where:** `data/smoothing.py`

**The gap.** The pandas function used for this filter has two modes, which
differ in how they treat the first few values of a series. The paper does not
mention the distinction.

**What we chose.** The plain recursive form (`adjust=False`), which is exactly
Equation 18 of the paper.

**Why.** It is the formula the paper prints. The two modes give visibly
different values at the start of each engine's history, so this is not a
cosmetic detail.

---

## A-10 — The seventh dropped sensor is NRc

**Where:** `data/sensors.py`

**The gap.** Section 3.2.2 of the paper states that "seven sensors are removed"
and then names only six: T2, P2, epr, farB, Nfdmd and PCNRdmd.

**What we chose.** The seventh is `NRc`, the corrected core speed.

**Why.** It is the only sensor that appears in the paper's full sensor list
(their Table 2) but not among the 14 rows of their adjacency matrix (their
Table 3). Since Table 3 defines what the model actually consumes, it is the
authoritative list.

**Confirmed against the data.** Running the diagnostic on the real FD001
training set shows that exactly the six *named* sensors are perfectly constant.
`NRc` is not: it takes 6078 distinct values across the 20 631 training rows. So
it was dropped for a different reason than the other six, most plausibly
redundancy — the corrected core speed is derived from the physical core speed
`Nc`, which the paper keeps. The paper's sentence conflates two different
reasons for exclusion.

**How much it matters.** None for the code, once identified: the retained set
is fully determined by Table 3. It is worth one sentence in the thesis, because
it shows the selection was not purely a variance filter.

---

## Findings about the source paper, established from the data

These are not choices made by this project. They are observations produced by
running the pipeline on the real FD001 files, and they belong in the thesis.

### F-01 — P15 is in the graph but carries almost no information

The bypass-duct pressure `P15` is one of the 14 sensors in the paper's
adjacency matrix. On the FD001 training set it takes only **two** distinct
values in 20 631 rows: 21.60 and 21.61. A signal with two levels cannot
represent gradual wear.

It is kept here anyway, because the adjacency matrix is defined over the
paper's fixed 14-sensor set and dropping a node would change the graph the
paper specifies. The pipeline reports it with a warning instead of silently
correcting it.

Why this is interesting for the thesis: `P15` has four neighbours in the
physics graph (`T24`, `Nf`, `NRf`, `BPR`). A node that carries no information
but still passes messages to four others is exactly the kind of thing a
graph-native explanation method should be able to expose, and a feature
attribution method such as SHAP should assign it near-zero importance. It is a
ready-made sanity check for the explainability stage.

Note that the right test here is the **number of distinct values**, not the
size of the range. `Nf` has a smaller relative range than `P15` (2.8e-4 against
4.6e-4) but spreads it over 53 levels and is genuinely informative.

### F-02 — Eleven percent of test engines are affected by the label ceiling

Of the 100 official FD001 test labels, 11 exceed 125 cycles (the maximum is
145). Assumption A-02 therefore changes the target for 11 test engines. Any
comparison with a published number has to state whether that paper capped its
test labels, and most do not say.

### F-03 — The shortest test history is exactly 31 cycles

This is why a window of 30 is the largest round number that works on FD001
without padding any engine, and it is a likely reason the value 30 has become
standard on this sub-dataset. With the configured window of 30, no test engine
needs padding.

---

## A-11 — The per-sensor LSTM stage is omitted

**Where:** project scope

**The gap.** This is not a gap in the paper but a deliberate decision about
this project's scope.

**What we chose.** Build the graph part of the model without the parallel LSTM
networks.

**Why.** Two reasons. First, the LSTM configuration is the single largest
undefined quantity in the paper (see A-07): its hidden size, its depth, and how
it reduces a 30-step window to one number are never given, so any
implementation of it would be invention rather than reproduction. Second, the
paper itself publishes this exact configuration: "ESTGNN w/o LSTM" is one of
the three variants in its ablation study (Table 11), reported at an average
error of 19.68 against 14.75 for the full model.

**Consequence to state clearly.** Results from this code must be compared
against the paper's **"ESTGNN w/o LSTM"** row, not against its headline figures.
The full ESTGNN has not been reproduced here and no claim about it should be
made from this code.

---

## References

Full citations for every work mentioned above are in `REFERENCES.md`.
