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
| ~~A-08~~ | Order: smoothing and scaling | **Fig. 1 specifies it**: smoothing then normalisation | Withdrawn as an assumption; the paper's order is followed |
| A-09 | Smoothing formula variant | Gives the recursive form as Equation 18 | Recursive form (`adjust=False` in pandas) |
| A-10 | Identity of the 7th dropped sensor | Says "seven", names six | The seventh is `NRc`, deduced from their Table 3 |
| A-11 | The LSTM stage | Central to their architecture | Omitted, matching their own published ablation |
| A-12 | Bias in the graph layers | Equations 8-10 print only `W` | No bias, as printed; configurable |
| A-13 | Scaling of the target | Sigmoid output called a "normalized RUL value" | Targets divided by `rul_max` for training |
| A-14 | Weight initialisation | Not stated | Glorot uniform, following Kipf & Welling (2017) |
| A-15 | Sign of the score function | Equations 20 and 21 contradict each other | The NASA convention the paper's own text describes |
| A-16 | R-squared denominator | Equation 23 uses the predicted values | The standard definition, using the observed ones |
| A-17 | Learning-rate schedule | Both endpoints given, mechanism not | Exponential decay reaching 1e-4 at the last epoch |
| A-18 | Early stopping | None; trains a fixed 200 epochs | Stop after 20 epochs without improvement |
| A-19 | What "without physics prior" removes | Names the ablation, never defines it | Replace the graph with a fully connected one |
| A-20 | What "a sensor not considered" means | Equation 15 needs `f(S)`, but a network cannot take 13 inputs | Replace the sensor with its average training window |

---

## A-01 — The label ceiling is 125 cycles

**Where:** `config.PreprocessingConfig.rul_max`, `data/labeling.py`

**The gap.** Equation 19 of the paper reads
`RUL_t = min(T_total − T_end, RUL_max)`, and the text says only that `RUL_max`
"indicates the early RUL threshold". The number itself never appears.

**What we chose.** 125 cycles.

**Why.** Three indications point to it, and one figure appears to contradict
them.

1. **The strongest evidence is Fig. 7.** Its four panels plot the *True RUL*
   for one engine from each sub-dataset. That curve is the ground-truth label
   itself, so its flat portion *is* `RUL_max`. Measuring the FD001 panel
   against its own axis ticks puts the plateau at **125.2**, which is 125 to
   within the width of the printed line.
2. Section 4.2 describes the predicted life holding at "relatively stable
   values around 125" while an engine is still healthy.
3. 125 is the value used by Jin et al. (2025) on the same dataset, and it sits
   inside the range of 120 to 130 that Heimes (2008) reported as effective when
   he introduced this kind of label.

**The contrary indication.** Fig. 6, the schematic that illustrates the
truncation, draws its `RUL_max` line on the 120 gridline. It is a diagram of
the *concept*, drawn with round numbers, and carries no experimental data;
Fig. 7 does. Where the two disagree, Fig. 7 is the record of what was actually
computed. This is noted here so that a reader who opens Fig. 6 first is not
left thinking the value was misread.

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

**The validation set does not mirror the test protocol.** The two measure
different things and their error figures are not comparable:

|            | what it contains                                        |
|------------|---------------------------------------------------------|
| validation | *every* window of 20 held-out engines (3272 on FD001)   |
| test       | *one* window per engine, at the truncation point NASA chose |

The important consequence is not that one is harder than the other, but that
**each engine carries a completely different weight**. In the validation set
the 20 engines contribute between 108 and 240 windows each, so three engines
alone supply about a fifth of the total and an unusual engine can move the
figure on its own. In the test set every engine contributes exactly one window,
so each is worth exactly 1 %.

Which of the two looks better is therefore not predictable in advance. On the
first full run of this pipeline the validation error was 16.2 cycles and the
test error 13.9 — the *opposite* of what the greater share of easy
ceiling-labelled windows in validation (24 % against 11 %) would suggest.

This is acceptable for what a validation set is for — choosing when to stop
training, and comparing settings, which are relative judgements made against a
fixed set — but a validation figure must never be reported as if it were a test
figure, and the gap between them must not be read as evidence of anything.

If the two need to be directly comparable, draw one random truncation point per
validation engine so that the validation set imitates the test protocol. That
is not done here, because the present form gives the early-stopping signal more
data to work with.

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

## A-08 — WITHDRAWN: the paper does state the order

**Where:** `data/pipeline.py`

**Status.** This was logged as an assumption in error. It is not one. Fig. 1 of
the paper prints the preprocessing chain explicitly:

> feature selection → **data smoothing** → piece-wise RUL →
> **data normalisation** → train and test datasets

Smoothing before normalisation is therefore *taken from the paper*, not chosen
here, and this project happens to match it. The entry is kept with its original
number so that references to A-08 elsewhere remain valid.

**Why it is still worth reading.** The order is safe on FD001 and only on
FD001. On FD002 and FD004 the operating condition changes from one cycle to the
next, and — as `data/normalization.py` explains — the jump between conditions is
far larger than the drift caused by wear. Smoothing in raw physical units
*before* the per-condition scaling has had a chance to remove those jumps blends
readings taken under different flight conditions together.

Measured on a six-condition signal built with a known wear curve:

| Order | correlation of the scaled signal with true wear |
|-------|--------------------------------------------------|
| smoothing → scaling (the paper's order, Fig. 1) | **0.045** |
| scaling → smoothing (condition-aware) | **0.997** |
| scaling only, no smoothing | 0.993 |

This is a property of the published method, not a defect introduced here, and
it is a candidate explanation for the paper's much weaker multi-condition
results (RMSE 16.21 on FD002 and 18.67 on FD004 against 11.78 on FD001; R²
falling to 0.603 and 0.510). Before this pipeline is run on FD002 or FD004 the
order must be revisited and the deviation from the paper recorded as a new
assumption.

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

**The number to aim at on FD001 is 14.93.** 19.68 is the average across all
four sub-datasets. Table 11's FD001 column gives the w/o-LSTM variant an RMSE
of **14.93** (against 11.78 for the full model). Since only FD001 is handled so
far, 14.93 is the figure any result from this code should be compared with.

**Consequence to state clearly.** Results from this code must be compared
against the paper's **"ESTGNN w/o LSTM"** row, not against its headline figures.
The full ESTGNN has not been reproduced here and no claim about it should be
made from this code.

---

## A-12 — The graph convolution layers carry no bias

**Where:** `models/layers.py`, `GraphConvolution(bias=False)`

**The gap.** Equations 8, 9 and 10 of the paper give the graph layers as
`H = ReLU(A_hat X W)`, with no bias term. Equations 12 and 13, which describe
the two fully connected layers, print biases explicitly as `W.z + b`.

**What we chose.** Follow the equations exactly: no bias in the three graph
layers, biases in the two dense layers.

**Why.** The paper writes biases where it means them. Within the same short
section it prints them for the dense layers and omits them for the graph
layers, which reads as a deliberate distinction rather than a shorthand. It is
also the convention of the layer the paper is citing: Kipf and Welling (2017)
define the propagation rule without a bias.

**How much it matters.** Small. A bias per output channel would add 128
parameters to a network of about 62 000. `GraphConvolution` takes `bias=True`
so the difference can be measured rather than argued about.

---

## A-13 — Targets are divided by the label ceiling for training

**Where:** `models/estgnn.py`, `ESTGNN.forward` and `ESTGNN.predict_cycles`

**The gap.** The paper's final layer is a sigmoid, and the text says it
"ensures normalized RUL values within [0,1]". It never states the arithmetic
that turns that number back into flight cycles, nor what the training targets
look like.

**What we chose.** The network predicts `RUL / rul_max`, and a prediction in
cycles is that number multiplied by `rul_max`. Training targets are divided by
the same constant.

**Why.** It is forced, not chosen. A sigmoid cannot leave `[0, 1]`, so a
network trained against targets measured in cycles could never reach a label of
40, let alone 125. Dividing by the ceiling is the only scaling that puts the
targets where the output layer can reach them and keeps the "normalized RUL"
description true.

**Consequence.** The model cannot predict a remaining life above `rul_max` for
any input whatsoever. This is the architectural fact that assumption A-02 rests
on, and it is checked directly by
`test_models.py::test_the_prediction_can_never_exceed_the_ceiling`.

---

## A-14 — Weights are initialised with Glorot uniform sampling

**Where:** `models/layers.py`, `GraphConvolution.reset_parameters`

**The gap.** The paper does not say how any weight matrix is initialised. Its
Table 6 lists the optimiser, learning rate, batch size and epoch count, and
stops there.

**What we chose.** Glorot (Xavier) uniform initialisation for the graph
convolution weights. The dense layers keep PyTorch's default, which is the
Kaiming-uniform variant that library has used for `Linear` for years.

**Why.** Glorot is what Kipf and Welling (2017) use for this exact layer. It
keeps the variance of the activations roughly constant as they pass through the
three stacked layers, which is what makes a deep stack trainable; an
initialisation that is too large or too small makes the first epochs behave
very differently even when everything else is identical.

**How much it matters.** Moderate for a single run and small on average.
Initialisation is one of the reasons a single seed is not evidence, which is
why this project reports results across several seeds (see `utils/seeding.py`).

---

## A-15 — The score function uses the NASA sign convention, not Equation 20

**Where:** `evaluation/metrics.py`

**The gap.** This is a contradiction inside the paper rather than an omission.
Equation 20 defines the error as `delta = actual - predicted`. Equation 21 then
scores `delta >= 0` with `exp(delta/10) - 1` and `delta < 0` with
`exp(-delta/13) - 1`.

Substituting the first into the second gives the *gentler* divisor of 13 to
over-prediction and the *harsher* divisor of 10 to under-prediction. The
surrounding text says the opposite in words: "the penalty for over-prediction
is relatively large, while the penalty for under-prediction is relatively
small ... over-prediction introduces security risks". The published function
from the 2008 NASA PHM Data Challenge, which the paper cites, also says the
opposite.

**What we chose.** The convention the paper describes in words and the
challenge defines: `d = predicted - actual`, with `exp(d/10) - 1` when the
model was late and `exp(-d/13) - 1` when it was early.

**Why.** Two of the paper's three statements agree with each other and with the
literature; only the printed sign of Equation 20 disagrees. Implementing the
outlier would produce numbers that cannot be compared with any other work on
this dataset, including the paper's own comparison table.

**How much it matters.** Large for the score, nothing for the other three
measures. Getting it backwards would reward exactly the behaviour the metric
exists to punish.

---

## A-16 — R-squared uses the observed values in its denominator

**Where:** `evaluation/metrics.py`

**The gap.** Equation 23 prints
`R^2 = 1 - sum(delta_i^2) / sum((predicted_i - mean_i)^2)`, with the
*predicted* values in the denominator.

**What we chose.** The standard definition, with the observed values:
`1 - sum((predicted - actual)^2) / sum((actual - mean)^2)`.

**Why.** As printed, the denominator measures how far the *predictions* spread
around the mean of the observations, rather than how far the observations
spread around it. That rewards the wrong thing: a model can raise its score by
spreading its output more widely without becoming any more accurate. For truth
`[10, 20, 30, 40]`, the prediction `[-5, 15, 35, 55]` is exactly as good as
predicting the mean, and the standard formula scores it 0.0 — but the paper's
version scores it 0.75.

That is not a coefficient of determination, and it is not what the text beside
the equation describes ("the proportion of variance in the dependent variable
that is predictable from the independent variables"). It reads as a typesetting
error.

---

## A-17 — The learning rate decays exponentially to its final value

**Where:** `training/config.py`, `TrainingConfig.lr_schedule`

**The gap.** Table 6 gives `1e-3 -> 1e-4` and Section 3.4 calls it "an adaptive
learning rate schedule". The mechanism is never stated: the sentence is
consistent with a smooth decay, a step at a fixed epoch, or a reduction
triggered by a plateau in the loss.

**What we chose.** Exponential decay, with the factor chosen so the last epoch
lands exactly on `1e-4`.

**Why.** It is the reading that reaches both stated endpoints and passes
through everything between them, and it needs no further invented constants —
a step schedule would require inventing the epoch to step at. The `"plateau"`
option implements the main alternative reading, and `"cosine"` and
`"constant"` are provided so the sensitivity to this choice can be measured.

**How much it matters.** Moderate. It changes the trajectory of a run and
therefore where early stopping lands, but the endpoints are fixed by the paper.

---

## A-18 — Training stops after 20 epochs without improvement

**Where:** `training/config.py`, `TrainingConfig.early_stopping_patience`

**The gap.** The paper trains for a fixed 200 epochs and describes no early
stopping, which follows from having no validation set (see A-06).

**What we chose.** Stop when the validation error has not improved for 20
epochs, and restore the weights from the best epoch.

**Why.** With a validation set available, training past the point where it
stops improving only fits noise. Restoring the best weights rather than keeping
the last ones is what makes the stopping rule useful rather than merely
cheaper.

**Honesty note.** This is a deliberate **improvement** on the source, like
A-06, and it must be described as one. Setting
`early_stopping_patience: null` reproduces the paper's procedure exactly, and
that comparison is worth running once so the size of the difference is known
rather than assumed.

---

## A-19 — The "without physics prior" ablation uses a fully connected graph

**Where:** `scripts/train.py --graph complete`

**The gap.** Table 11 reports a variant called "ESTGNN w/o Physics prior", but
the paper never says what replaces the hand-designed adjacency matrix. At least
three readings are possible: every sensor connected to every other, no edges at
all (self-loops only), or a graph learned or derived from the data.

**What we chose.** A fully connected graph, built by
`graph.adjacency.build_fully_connected_adjacency` and normalised the same way.

**Why.** It is the only one of the three that removes the engineering knowledge
while changing nothing else. Removing all edges would also remove message
passing itself, which would make the variant an ablation of the GCN rather than
of the prior — and the paper reports that separately, as "w/o GCN". A learned
graph would add a mechanism the paper does not describe.

**How much it matters.** A great deal, as F-04 below records.

---

### F-04 — The physics graph is worth far more here than the paper reports

Five seeds of each variant on FD001, identical in every other respect:

| Sensor graph | RMSE | NASA score | R-squared | ASUE |
|--------------|------|-----------|-----------|------|
| Physics (their Table 3) | **13.45 +/- 0.20** | 312 +/- 11 | 0.887 +/- 0.003 | 8.85 +/- 0.36 |
| Fully connected (A-19) | 19.17 +/- 0.45 | 743 +/- 53 | 0.771 +/- 0.011 | 13.44 +/- 0.27 |

Removing the physics prior costs **5.7 cycles of RMSE** in this configuration.
The paper reports the same ablation costing **0.56** cycles (12.34 against
11.78), and calls it "the smallest performance impact" of the three components
it tests.

The most likely explanation is the LSTM stage this project omits (A-11). In the
full architecture, a per-sensor temporal encoder extracts degradation
information before the graph layers ever run, so the graph is one of two routes
to the same information and the model can compensate when it is removed.
Without the encoder, the graph is the only structure the model has, and taking
it away removes far more.

If that explanation is right it is a point in favour of the paper's central
claim rather than against it: the physics prior does real work, and the paper's
own ablation *understates* how much because its LSTM masks the effect. It is
also a testable prediction — adding any temporal encoder should shrink the gap
— and therefore something the thesis can investigate rather than merely assert.

**Do not report the two ablation numbers as directly comparable with the
paper's.** They come from a configuration the paper does not run.

---

## A-20 — An absent sensor is replaced by its average training window

**Where:** `explain/shap.py`, `mean_window_baseline`

**The gap.** Equation 15 is defined in terms of `f(S)`, "the ESTGNN prediction
when only sensors in subset S are considered". A trained network cannot be
given thirteen inputs; it always requires fourteen. So a sensor is never
actually removed, only **replaced** by a stand-in, and which stand-in is used
decides what the resulting numbers mean. The paper does not say what it uses.

**What we chose.** The average window of that sensor across the training set,
so the model is shown "this sensor is behaving typically".

**Why.** The obvious alternative, substituting zero, is much worse than it
looks. After the min-max rescaling of A-03, zero is not a neutral value: it is
the lowest reading that sensor ever recorded in training. Feeding it would tell
the model something strong and false rather than nothing, and every Shapley
value would be measured against that fiction.

Using the training mean also keeps the explanation honest in the same sense
the scaling is: nothing from the validation or test engines enters it.

**What the numbers then mean.** A Shapley value here answers: *how much did
knowing this sensor's actual readings, rather than a typical engine's, change
the prediction?* It is measured in flight cycles, and by the efficiency
property the fourteen values plus the baseline prediction reconstruct the
actual prediction exactly.

**How much it matters.** Moderate, and measurable. The ranking of sensors is
fairly stable across reasonable baselines; the magnitudes are not. Any figure
quoted from this analysis must say which baseline produced it.

---

### F-05 — What the first SHAP run says about P15, and what it does not

Finding F-01 recorded that `P15` takes only two distinct values on FD001 and
predicted that a working attribution method should give it close to zero
importance. The first run of the exact Shapley computation, on the full life of
one held-out engine (240 windows, 16 384 coalitions each), gives a more
interesting answer than that.

**Averaged over the engine's life, P15 ranks last of the fourteen**, with a
mean absolute contribution of 4.2 cycles against 11.8 for `T24`. That agrees
with the paper's own Fig. 9, which also places `P15` at the bottom, and it is a
point of genuine agreement between this re-implementation and the original.

**But its contribution is not uniformly small.** It reaches 12 to 15 cycles in
the first five windows of the engine's history and then collapses to about
0.05 cycles for the remaining 235. A sensor carrying no information should not
be worth fifteen cycles at any moment.

The mechanism is in the preprocessing, not the attribution. `P15`'s two raw
levels are 0.01 psia apart, which is measurement quantisation rather than
signal. The smoothing filter of A-04 interpolates between those two levels, and
the min-max rescaling of A-03 then stretches the result across the full `[0, 1]`
range. Measured on the scaled training windows, `P15` ends up with a standard
deviation of 0.083, which is essentially the same as `Nc` at 0.081. A sensor
with nothing to say arrives at the model with the same amplitude as one that
has. The early windows are where the exponential filter is still settling from
its first reading, which is exactly where the spurious contribution appears.

**Consequences worth stating in the thesis.**

1. F-01's proposed sanity check is right on average and wrong window by window.
   The sharper version: `P15`'s attribution should be near zero *once the
   smoothing filter has settled*, and it is.
2. Min-max rescaling is unsafe for a near-constant sensor, because it maps
   quantisation noise onto the same range as real variation. Standardising by
   the standard deviation instead would have the same problem; only dropping
   the sensor, or scaling by a fixed physical range, avoids it. The pipeline
   keeps `P15` because the paper's adjacency matrix is defined over its fixed
   14-sensor set, so this is a limitation to report rather than to correct.
3. This is a concrete example of an explanation revealing a preprocessing
   defect rather than a model defect, which is a useful thing for a thesis on
   explainability to be able to show.

---

## References

Full citations for every work mentioned above are in `REFERENCES.md`.
