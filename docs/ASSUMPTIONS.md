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
| A-03 | Input rescaling | Not described at all | **`zscore`** — the largest measured assumption in the project (3.83 cycles, F-10); selected under A-22 |
| A-04 | Smoothing strength | Self-contradictory sentence in Section 3.2.3 | Read as `alpha = 0.5` |
| A-05 | Window stride | Not in the text; **Fig. 1 indicates one time step** | 1 cycle |
| A-06 | Validation split | None; trains for a fixed 200 epochs | **None in the reproduction** (A-22); 20 % in `fd001_validated.yaml`, for comparing configurations |
| A-07 | Node feature content | Table 4 **prints** one number per sensor (`W(1) in R^(1x32)`), produced by an LSTM that is never specified | The last cycle of the window, keeping Table 4's shape |
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
| A-18 | Early stopping | None; trains a fixed 200 epochs | **Disabled in the reproduction** (A-22); available with a validation split |
| A-19 | What "without physics prior" removes | Names the ablation, never defines it | Replace the graph with a fully connected one |
| A-20 | What "a sensor not considered" means | Equation 15 needs `f(S)`, but a network cannot take 13 inputs | Replace the sensor with its average training window |
| A-21 | Adjacency normalisation formula | Says only "normalized ... incorporating self-loops"; never prints it | **Row `D^(-1)(A+I)`** — 0.35 cycles closer *with nothing held back* (p = 0.047), selected under A-22 |

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

**How much it matters — measured, not argued.** Eleven of the 100 FD001 test
labels exceed 125 (F-02). Five seeds, the *same trained models* scored two
ways (identical seeds and identical best epochs, so this is a pure scoring
difference), commit `cdcc827-dirty`, 2026-08-28:

| Test labels | RMSE | NASA score | R-squared | ASUE |
|-------------|------|-----------|-----------|------|
| capped at 125 (`fd001.yaml`) | 20.353 ± 0.752 | 827 ± 85 | 0.742 | 15.67 |
| uncapped (`fd001_uncapped.yaml`) | 21.664 ± 0.818 | 914 ± 78 | 0.728 | 18.03 |

Per-seed penalty `[1.356, 1.333, 1.167, 1.204, 1.496]`; mean **+1.311 cycles**,
paired **t = 22.30, p = 0.00002**. Highly significant and one-directional:
capping can only help, and it helps by about 1.3 cycles.

**Therefore the gap to the paper's 14.93 is wider than the headline says.**
This code reports 20.35 with the cap; without it, 21.66. Whichever is quoted,
the cap must be quoted with it. (Under the removed `window` mode the same
measurement gave +1.762 cycles; the effect is configuration-dependent, so it
must be re-measured whenever the architecture changes.)

---

## A-03 — Sensor values are rescaled with z-scores

**Where:** `config.PreprocessingConfig.normalization`, `data/normalization.py`

**The gap.** The paper never describes how the sensor inputs are scaled. The
words "normalisation", "min-max" and "standardisation" do not appear in
connection with the inputs, and operating conditions are never mentioned in the
preprocessing section.

**What we chose.** The **z-score**: subtract the mean and divide by the
standard deviation, with both learned from the training engines only. Selected
under A-22 on 2026-08-29.

**What was chosen before, and why it changed.** Until then this entry recorded
min-max scaling to the interval [0, 1], computed separately for each operating
condition, taken from Jin et al. (2025). The argument for it was that it fits a
sigmoid output layer, which already assumes targets in [0, 1]. That argument is
about the *targets*, which are scaled separately under A-13, and it never
applied to the inputs. When the two were finally measured against each other
the z-score won by 3.83 cycles — the largest effect of any choice in this
project — and A-22's rule selected it. The measurement is below; the reasoning
in the rest of this entry is unchanged, because the *gap* it describes and the
honesty note both apply whichever option is taken.

`minmax_per_condition` remains available and remains the right choice to reach
for on FD002 and FD004, where several operating conditions coexist. See the
note at the end.

**Why.** Some form of rescaling is unavoidable: the 14 sensors differ by three
orders of magnitude, and a neural network trained on unscaled inputs of that
kind does not converge usefully. The paper offers nothing to reconstruct, so
the choice falls to A-22's rule — take the option that lands closest to the
paper's own figure — and that rule selects the z-score by a wide margin. The
z-score also has the property that it does not bound the values, so a test
reading outside the training range is represented faithfully instead of being
flattened onto 0 or 1.

**Honesty note.** Whichever option is in force, this is *our choice*, not a
reconstruction of what Wen et al. did. Their procedure is unknown. Until
2026-08-29 the choice was at least borrowed from a paper on the same dataset;
now it is selected by A-22's rule against the paper's own reported figure,
which is a weaker provenance, not a stronger one — see A-22's caveats.

On FD001 there is only one operating condition, so the per-condition min-max
variant and the global one give identical results; the distinction becomes
important only when FD002 and FD004 are added.

**How much it matters — measured, and it is the largest effect in the project.**
The claim that once stood here, "on FD001, little, as long as some sensible
scaling is used", is wrong. Five seeds each, everything else identical
(commit `09628fa`):

| Scaling | RMSE on FD001 |
|---------|---------------|
| `minmax_per_condition` (from Jin et al., the default until 2026-08-29) | 20.353 ± 0.752 |
| `zscore` (**the default now**) | **16.524 ± 0.339** |

**3.83 cycles, paired t-test p = 0.0005.** That is 71 % of the entire gap to
the paper's 14.93, and it is larger than every other assumption tested
combined. See F-10 for the full sweep.

This does not tell us what Wen et al. did — they describe no scaling at all —
but it does mean **the choice recorded in this entry is the single biggest
reason a number from this code cannot be compared with a published one.** That sentence belongs beside any result quoted from this project.

On FD002 and FD004, where the per-condition variant stops being equivalent to
the global one, the effect is likely larger still.

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

**The gap, and how much of one it really is.** Table 6 gives a window of 30
time steps and no stride, and the running text never mentions one. But
**Fig. 1 does**: the panel showing five overlapping windows sliding along the
`T24`, `T30` and `T50` traces is captioned *"Sliding window (t=1s)"*.

**What we chose.** A stride of 1 cycle, which is what that caption indicates.

**Why this is still logged as an assumption rather than withdrawn like A-08.**
The caption's unit is wrong. C-MAPSS is sampled once per flight cycle and
contains no seconds anywhere; Table 6 calls the same axis "time steps". So
the figure indicates a stride of one *sample*, and reading that as one cycle
is an inference, even if an easy one. A stride of 1 is also the usual choice
on this dataset and produces the largest number of training examples from a
fixed amount of data.

**How much it matters.** Small for accuracy, large for training time.

---

## A-06 — No engines are held out in the reproduction; twenty percent are for comparisons

**Superseded 2026-08-29 by A-22.** This entry originally held out 20 % of the
training engines unconditionally. Table 6 states 200 epochs, and holding
engines out in order to restore the best-validation checkpoint is
validation-based model selection, which the paper never describes; under A-22's
rule the paper wins. `config/fd001.yaml` therefore holds nothing out.

**The split has not been abandoned, only moved.** `config/fd001_validated.yaml`
keeps it, and **every comparison between configurations must use that file** —
without it there is no honest basis for preferring one setting to another, and
A-22 records one choice (A-21) that had to be made on the test set for exactly
this reason. The reasoning below is unchanged and still governs how the split
must be used when it is used.

---

### Original entry

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

**What the reported spread now covers.** Until this was corrected,
`scripts/train.py` prepared the data once, outside the seed loop, so all five
seeds shared one train/validation split: the quoted standard deviation covered
weight initialisation and batch order only. That understates the spread a real
repetition of the experiment would show, because a repetition also draws a
different set of validation engines — and with only 20 held out, which 20 they
are matters.

---

## A-07 — Each sensor node carries the last cycle of its window

**Where:** `data/windowing.py`; `models/estgnn.py` (`in_features`)

**The gap.** Table 4 of the paper gives the input to the first graph layer as
`[batch, 14, 1]` with `W(1) in R^(1 x 32)`: one number per sensor. That number
is produced by a per-sensor LSTM whose size, depth, and output projection are
never stated anywhere in the paper. What is *not* a gap is the shape: the
paper prints it.

**What we chose.** With the LSTM stage omitted (see A-11), each sensor node
carries the **most recent reading** of its window, giving `[batch, 14, 1]` and
`W(1) in R^(1 x 32)` exactly as printed. The parameter count matches Table 4
to the digit: **61,729**, asserted by
`test_models.py::test_the_default_width_reproduces_table_4`.

**Why.** Reproducing an ablation means keeping the printed architecture and
removing the named component. Any other filler for that one number — the mean
of the window, a learned 30 → 1 projection, or handing the graph layer the
whole window — requires deciding *how to summarise 30 cycles into a
representation*, which is precisely the decision the paper never states and
which A-11 declines to invent.

**Superseded reading, and why it was wrong.** Until 2026-08-28 the default was
a `window` mode in which each node carried all 30 cycles, making `W(1)` a
`(30, 32)` matrix. That was justified as "no temporal encoder, so keep the
history". It is not what it claimed to be. `A_hat X W(1)` with `X` of shape
`(14, 30)` computes, for each of the 32 output channels, `sum_t w(t,c) x(t)` —
**32 learned linear filters over the 30 time steps, shared across sensors**.
That is a temporal encoder, of exactly the kind A-11 refuses to invent, folded
into the graph layer where it was never labelled as a choice. The mode has been
removed. Its results remain valid *for the model they describe* and are
reproducible at commit `cdcc827`; they are no longer this project's numbers.

**How much it matters.** Large, and measured. See F-06: **20.35 ± 0.75** with
one reading per node against 13.45 ± 0.20 under the removed `window` mode, on
FD001, five seeds each. The removed mode was the more accurate of the two by a
wide margin — which is precisely why it needed removing rather than keeping:
those 928 extra parameters bought their accuracy by doing the temporal
filtering A-11 declines to invent.

**Note on what `window_size` still does.** It is not inert. It decides which
cycles are eligible — an engine's first example ends at cycle `window_size`,
so cycles 1 to 29 are never a prediction point — and it is what a temporal
encoder would consume if one were added later.

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
next, and — as `data/README.md` explains under "Rescaling, and what an
operating condition is" — the jump between conditions is far larger than the
drift caused by wear. Smoothing in raw physical units
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

**Why.** Seven of the 21 sensors in the paper's full list (their Table 2) are
absent from the 14 rows of their adjacency matrix (their Table 3): T2, P2,
epr, `NRc`, farB, Nf_dmd and PCNfR_dmd. Six of those seven are the ones
Section 3.2.2 names. `NRc` is the only one it does not, so it is the
unnamed seventh. Since Table 3 defines what the model actually consumes, it
is the authoritative list.

The paper's own Section 3.2.2 supports this independently. It gives the
removal criterion as "high correlational redundancy (above 0.95) and
constant values", and names `NRc` in the sentence "Rotational speed
parameters (Nf, Nc, NRf, NRc) show high correlation coefficients". `NRc` is
therefore removed by the redundancy half of the criterion rather than the
constancy half, which is exactly what the data shows.

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
graph-native explanation method should be able to expose.

**This entry originally predicted that SHAP would assign `P15` near-zero
importance. That prediction has since been tested and is only half right;
see F-05.**

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

**What this rules out, and did not always.** "No temporal encoder" has to mean
no temporal encoder *anywhere*, including one hidden inside a layer that is
named something else. Handing the first graph layer a 30-cycle vector per node
makes `W(1)` a `(30, 32)` matrix, which is a bank of 32 learned linear filters
over time. That was this project's default until 2026-08-28 and it contradicted
this entry. It has been removed; see A-07. The graph stage is now the only
learned stage, which is what makes a comparison against "ESTGNN w/o LSTM"
meaningful at all.

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

**Where:** `training/run.py`, `_as_loader` divides the targets;
`models/estgnn.py`, `ESTGNN.predict_cycles` multiplies the prediction back.

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
Table 6 lists six hyper-parameters — learning rate, epochs, batch size, time
steps, optimiser and loss function — and initialisation is not among them.

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

## A-18 — Early stopping is disabled in the reproduction

**Superseded 2026-08-29 by A-22.** Table 6 states 200 epochs. Early stopping
contradicts that, so the reproduction runs the full 200.

**A trap worth recording.** Passing `--no-early-stopping` is *not* by itself
enough to reproduce the paper's procedure. With a validation set present the
loop still tracks the best-validation epoch and ends with
`model.load_state_dict(best_state)`, so the reported model is the best
checkpoint rather than the epoch-200 model — validation-based model selection
without the name. Only `validation_fraction: 0.0` reaches the branch that keeps
the final weights. The two differ by about 0.45 cycles, and an earlier draft of
F-10 described the wrong one as "the paper's training procedure".

---

### Original entry

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

### F-04 — WITHDRAWN AND REVERSED: the physics graph is worth nothing measurable here

**Status.** This entry claimed the opposite of what it now reports. The
original claim was an artefact of the removed `window` node-feature mode
(A-07), and re-running under the corrected architecture reverses it. The
original text is preserved at the bottom because a corrected record is worth
more than a clean one (working rule 6).

**What it says now.** Five seeds of each variant on FD001, one feature per
node, identical in every other respect
(`cdcc827-dirty`, 2026-08-28, varying split):

| Sensor graph | RMSE | NASA score | R-squared | ASUE |
|--------------|------|-----------|-----------|------|
| Physics (their Table 3) | 20.353 +/- 0.752 | 827 +/- 85 | 0.742 +/- 0.019 | 15.67 +/- 2.05 |
| Fully connected (A-19) | 20.397 +/- 0.395 | 896 +/- 113 | 0.741 +/- 0.010 | 14.96 +/- 1.65 |

Removing the physics prior costs **0.044 cycles of RMSE**. Paired across
seeds, **t = 0.12, p = 0.913** — indistinguishable from zero. Per-seed
differences are `[+0.35, +0.56, +0.60, +0.16, -1.45]`: the complete graph is
*better* on one of the five seeds.

**How the old number arose.** Under `window` mode each node carried 30
features, so the first graph layer mixed 14 nodes x 30 features. A complete
adjacency matrix averages every sensor into every other, which destroys the
per-sensor identity of 30 rich traces and is far more destructive than
averaging 14 single numbers. The 5.7-cycle gap measured how much damage
uniform mixing does to a wide feature vector, not how much engineering
knowledge the graph contains.

**What this now says about the paper.** The paper reports the same ablation on
FD001 costing **0.56** cycles (12.34 against 11.78) and calls it "the smallest
performance impact" of the three components it tests. This project measures
**0.044** cycles and cannot distinguish it from zero. Both agree the physics
prior is worth little; this reproduction cannot confirm that it is worth
anything at all.

**The honest reading is that the question is out of reach here, not settled.**
With one number per sensor the model reaches only 20.35 RMSE, well short of
the paper's 14.93 for the same ablation. A model that far from the reference
may simply be too information-starved for graph structure to matter either
way. Whether the physics prior does real work is a question that needs a
temporal encoder to answer, which is exactly the component A-11 omits.

---

**Superseded text, kept for the record (produced under the removed `window`
mode, reproducible only at commit `cdcc827`):**

> Five seeds of each variant: physics 13.45 +/- 0.20, fully connected
> 19.17 +/- 0.45. "Removing the physics prior costs **5.7 cycles of RMSE** in
> this configuration... here it is the largest [of their three components]."
> The proposed mechanism was that without the LSTM the graph is the only
> structure the model has, so removing it costs more, and that the paper's own
> ablation therefore *understates* the prior's value. That mechanism predicted
> the gap should shrink when a temporal encoder is added. Removing the linear
> temporal encoder hidden in `W(1)` shrank it to nothing instead, which is the
> opposite direction and refutes the explanation.

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

**Why.** Two reasons, and only the second one survives a change of scaling.

The training mean keeps the explanation honest in the same sense the scaling
is: nothing from the test engines enters it. That holds whatever the inputs
look like, and it is the reason the choice stands.

The reason originally given was different — that substituting zero would be
actively misleading. That argument was true, but only of the scaling in force
at the time, and it is recorded below because the way it failed is the useful
part of this entry.

**What the numbers then mean.** A Shapley value here answers: *how much did
knowing this sensor's actual readings, rather than a typical engine's, change
the prediction?* It is measured in flight cycles, and by the efficiency
property the fourteen values plus the baseline prediction reconstruct the
actual prediction exactly.

**How much it matters — re-measured 2026-08-31 under the current default, and
the answer reverses.** The same trained model, the same 20 test windows, three
baselines. Commit `f150570`, `config/fd001.yaml` (z-score, row, nothing held
back):

| Baseline | Four most important sensors | mean \|φ\| | Spearman with the mean window |
|---|---|---|---|
| mean window (used here) | T50, Nc, W31, T24 | 3.244 | — |
| median window | T50, T24, W31, Nc | 2.950 | **+0.952** |
| all zeros | T50, Nc, W31, T24 | 3.032 | **+0.996** |

**Under z-score the zero baseline is very nearly the same as the mean window,**
which is exactly what it should be: the z-score puts each sensor's training
mean at zero, so "substitute zero" and "substitute the mean" are the same
instruction written two ways. The 0.004 that separates them is the difference
between the mean of the training *rows*, which the scaler centres, and the
mean of the training *windows*, which the baseline averages.

**Under the previous min-max default the same comparison said the opposite.**
Preserved from before the A-22 rework, measured under `minmax_per_condition`:

| Baseline | Four most important sensors | Spearman with the mean window |
|---|---|---|
| mean window | T24, Nf, phi, P30 | — |
| median window | T24, Nf, phi, P30 | 0.97 |
| all zeros | **P15**, phi, W31, P30 | **−0.16** |

There, zero was a sensor's *lowest recorded reading*, not its mean. Feeding it
told the model something strong and false, produced an unrelated ranking, and
promoted `P15` — the least informative sensor in the set, see F-01 — to first
place.

**The finding, which is more general than either measurement.** *The safety of
a baseline is a property of the scaling, not of the baseline.* Zero is
catastrophic under min-max and almost exactly right under z-score, and nothing
about the choice of stand-in changed in between. So a SHAP baseline cannot be
justified on its own; it has to be justified against the rescaling in force,
and any paper that names one without naming the other has not said enough to be
reproduced. Wen et al. name neither.

The mean window is kept because it is the one choice that is defensible under
both: it is the neutral value by construction, whatever the scaling maps
neutral onto.

Any figure quoted from this analysis must say which baseline **and which
scaling** produced it.

---

### F-05 — What SHAP says about P15, and what it does not

Finding F-01 recorded that `P15` takes only two distinct values on FD001, and
predicted that a working attribution method would give it close to zero
importance. Exact Shapley values, computed over the full life of three held-out
engines, show that the prediction is half right in a way that turns out to be
more useful than if it had simply been correct.

**Re-measured 2026-08-31** under the current default (z-score, row, nothing
held back), commit `f150570`, over every window of three engines. Because
nothing is held back now, these three are training engines rather than held-out
ones; the same three are used so the two measurements can be compared.

**The ranking prediction holds, and holds more strongly than before.** `P15`
comes last of the fourteen on every engine tested:

| Engine | Windows | `P15` rank | mean \|φ\| for `P15` | median \|φ\| | max \|φ\| | top sensor |
|---|---|---|---|---|---|---|
| 5  | 240 | 14 of 14 | 0.706 | 0.260 | 9.449 | 11.749 (`T24`) |
| 25 | 201 | 14 of 14 | 0.474 | 0.466 | 1.769 | 9.523 (`Nc`) |
| 82 | 185 | 14 of 14 | 0.677 | 0.255 | 9.526 | 14.927 (`Nc`) |

That agrees with the paper's own Fig. 9, which also places `P15` at the bottom,
and it is a point of genuine agreement between this re-implementation and the
original. It is the one part of this entry that has survived every change to
the architecture and the scaling.

**The magnitudes fell by roughly a factor of six**, and the shape of the
failure changed with them. Under min-max the same three engines gave mean
\|φ\| of 4.15, 0.15 and 6.29; they now give 0.706, 0.474 and 0.677. The
superseded table is preserved at the end of this entry.

**The engine-specific pattern half survives, and the half that does is the
one that matters.** Under min-max, engine 25 was the negligible case and
engines 5 and 82 were the loud ones, cleanly separated on every statistic.
Under z-score the *medians* no longer separate — engine 25's median (0.466) is
now the **highest** of the three, and engines 5 and 82 sit at 0.260 and 0.255.
But the *maxima* still separate sharply: 9.449 and 9.526 against 1.769.

That is the mechanism claim, and it still holds. Engine 25's `P15` never
changes level at all; engines 5 and 82 change 34 and 20 times. A quantisation
step falling inside a window is still worth several cycles of attribution, and
where no step can fall, nothing above 1.8 cycles ever appears.

**The mechanism is measurement quantisation.** `P15`'s two raw levels are 21.60
and 21.61 psia, 0.01 apart, which is the resolution of the recording rather
than anything physical. Engines differ in how often the reading crosses between
them — these counts are of the raw file and are unaffected by any scaling:

| Engine | Distinct `P15` values | Times the level changes |
|--------|----------------------|-------------------------|
| 5  | 2 | 34 |
| 25 | 1 | 0 |
| 82 | 2 | 20 |

Across all 100 training engines the level changes a median of 2 times, never
for 38 of them, and up to 34 times. Engine 25 is one of the 38 that never
change, and it is the one engine of the three on which `P15` never reaches 2
cycles.

**The rescaling still makes those steps loud, and the original entry predicted
exactly this.** Its second consequence read: "Min-max rescaling is unsafe for a
sensor whose real variation is at the resolution limit... **Standardising by
the standard deviation would do the same.**" That prediction is now measured.
Standard deviations over the training windows under the current z-score
scaling:

| Sensor | sd on the scaled windows |
|---|---|
| `Nc` | 1.053 |
| `T50` | 1.002 |
| `T24` | 1.000 |
| **`P15`** | **0.862** |

A sensor with two distinct readings arrives at the model at 82 % of the
amplitude of one that tracks the whole degradation. It is quieter than under
min-max, where it was marginally the *louder* of the pair (0.083 against `Nc`'s
0.081), but it is nowhere near the silence its information content deserves.
Changing the scaling moved the number and did not fix the problem, which is
what the original entry said would happen.

**Consequences worth stating in the thesis.**

1. F-01's check should be stated as a *ranking* claim, not a magnitude one.
   "`P15` should rank last" is confirmed on every engine tested, under both
   scalings and both architectures — it is the most durable result in this
   file. "`P15`'s attribution should be near zero" is not confirmed: under
   z-score its worst window still reaches 9.5 cycles on the engines whose
   reading crosses between its two levels.
2. **Rescaling by any per-sensor statistic is unsafe for a sensor whose real
   variation sits at the resolution limit**, because it maps quantisation onto
   the same range as genuine signal. This was written as a prediction about
   min-max, with the aside that standardising by the standard deviation "would
   do the same". Changing the default to z-score tested that prediction and
   confirmed it: `P15` still arrives at 0.86 against roughly 1.0 for the
   sensors that track degradation. Only dropping the sensor, or scaling by a
   fixed physical range, avoids it. The pipeline keeps `P15` because the
   paper's adjacency matrix is defined over its fixed 14-sensor set, so this is
   a limitation to report rather than to correct.
3. **The size of an attribution is not comparable across preprocessing
   choices, but its rank is.** The same sensor on the same engines moved by a
   factor of six when only the scaling changed, while its position never moved
   at all. Any SHAP magnitude quoted from this project — or from the paper —
   is meaningless without the scaling that produced it, which is the same
   conclusion A-20 reaches from the baseline side.
4. This is a concrete case of an explanation exposing a **preprocessing**
   defect rather than a model defect, which is a useful thing for a thesis on
   explainability to be able to demonstrate.

**Superseded measurements, kept for the record** (min-max scaling, removed
`window` node-feature mode, reproducible only at commit `cdcc827`):

> | Engine | Windows | `P15` rank | mean \|φ\| for `P15` | top sensor |
> |---|---|---|---|---|
> | 5  | 240 | 14 of 14 | 4.15 | 11.80 (`T24`) |
> | 25 | 201 | 14 of 14 | 0.15 | 11.64 (`T24`) |
> | 82 | 185 | 14 of 14 | 6.29 | 11.13 (`T24`) |
>
> Medians of 1.5 and 4.1 cycles on engines 5 and 82, maxima of 19.1 and 25.8,
> and roughly half of all windows above one cycle. Engine 25: median 0.15,
> maximum 0.4. Scaled standard deviations `P15` 0.083 against `Nc` 0.081.

**A caution about this entry.** Its first version was written from fourteen
sampled windows out of 240, and reported that `P15`'s contribution "collapses
to about 0.05 cycles" after the first five. Checking all 240 showed a median of
1.5 and a maximum of 19.1, and the large values spread across the first
three quarters of the engine's life rather than clustering at the start. The
sampled columns had happened to fall in quiet stretches. Attribution values
vary enough from window to window that any claim about them has to be computed
over every window and across more than one engine.

---

### F-06 — Where this code lands against the paper's FD001 figures

Five seeds of each configuration, everything else identical.

Current configuration, five seeds each, varying split, capped test labels,
commit `cdcc827-dirty`, 2026-08-28:

| Run | Configuration | RMSE | NASA score | R-squared | ASUE |
|-----|---------------|------|-----------|-----------|------|
| **B** | **physics graph (`fd001.yaml`) — the default** | **20.353 ± 0.752** | 827 ± 85 | 0.742 ± 0.019 | 15.67 ± 2.05 |
| C | complete graph (A-19 ablation) | 20.397 ± 0.395 | 896 ± 113 | 0.741 ± 0.010 | 14.96 ± 1.65 |

Run B reproduces the 2026-08-27 `last_step` run **bit for bit** — identical
RMSE to six decimals and identical best epochs `[85, 195, 172, 45, 14]` on all
five seeds. That is the regression check on the A-07 refactor: removing the
`window` option changed nothing about the configuration that was kept.

Two earlier runs used the removed `window` mode (physics 13.45 ± 0.20,
complete 19.17 ± 0.45). They are reproducible at commit `cdcc827` and nowhere
later, and are recorded in F-04 rather than deleted. They are not this
project's numbers.

The paper, on FD001:

| Variant | RMSE | Source |
|---------|------|--------|
| ESTGNN, full model | 11.78 | Tables 7-10 |
| **ESTGNN w/o LSTM** | **14.93** | Table 11 |
| ESTGNN w/o GCN | 16.42 | Table 11 |
| ESTGNN w/o physics prior | 12.34 | Table 11 |

**Only one number is directly comparable.** Table 11 reports RMSE alone, so
the score, R-squared and ASUE columns exist for the full model only. Since the
LSTM stage is not implemented here (A-11), the row to compare against is
"w/o LSTM", and the comparison is 20.353 ± 0.752 against 14.93.

#### Superseded by A-22 — the current default is z-score and row

The comparison below was made with the per-condition min-max scaling and
symmetric normalisation that were the defaults until 2026-08-29. Under the
selection rule recorded in **A-22** the reproduction now reports
**16.694 ± 0.331**, a shortfall of 1.76 rather than 5.42. The reasoning in this
section is unchanged and still applies to the architecture, which A-22 does not
touch; only the preprocessing and the adjacency formula moved.

#### The architecture-faithful comparison is 20.35 against 14.93

Run B is the paper's printed architecture: input `[batch, 14, 1]`,
`W(1) in R^(1 x 32)`, the same layer widths, and **61,729 parameters — Table 4's
count exactly**. The only thing removed is the LSTM, which is what "w/o LSTM"
means. So this is the comparison that carries weight:

| | RMSE on FD001 |
|---|---|
| this code, five seeds | 20.353 ± 0.752 |
| paper, "ESTGNN w/o LSTM" (Table 11) | 14.93 |

**A 5.4-cycle miss (+36 %), with no shape mismatch left to blame it on.** For
scale: it is worse than the paper's *worst* ablation, "w/o GCN" at 16.42 —
that is, worse than the paper's own model with the graph stage deleted
entirely. That is a
sharper statement than the earlier framing of this finding, which presented
13.45 and 19.76 as two readings bracketing 14.93. The bracket was misleading:
13.45 came from a wider first layer (`W(1)` of shape `(30, 32)`) that is not
the printed architecture and that smuggles in a linear temporal encoder — so
it was never a candidate reproduction of the ablation, whatever its accuracy.

What remains to explain the 5.42 cycles is the documented assumptions, in
roughly descending order of suspicion: A-03 (input scaling, which the paper
never describes at all), A-13/A-14/A-17 (target scaling, initialisation,
learning-rate schedule), A-18 (early stopping, an addition), and A-02 (label
capping, measured at 1.311 cycles and in the *favourable* direction, so the
underlying gap is larger still).

#### What run A was, and why beating the paper was a warning sign

Run A came in 1.5 cycles *better* than the published "w/o LSTM" figure. That
should have prompted more suspicion than it did. A re-implementation that is
missing a component and still beats the original is usually not a better
implementation; it is a different model. It was: `W(1)` of shape `(30, 32)`
gave it 928 parameters the paper's architecture does not have, and those
parameters do temporal filtering (A-07).

The lesson is worth keeping even though the configuration is gone: **check the
parameter count against the printed architecture before comparing accuracy.**
Table 4 gives 61,729. Run A had 62,657. That single number would have caught
the problem immediately, and it is now asserted by a test.

#### What the ablations say, re-measured

All three comparisons that had been taken under the removed `window` mode were
re-run on 2026-08-28 at commit `cdcc827-dirty`, five seeds each. Every one of
them changed, and two of them reversed:

| Comparison | under `window` (removed) | under the current architecture |
|---|---|---|
| physics vs complete graph (F-04) | 5.72 cycles apart | **0.044 apart, p = 0.913** — reversed |
| symmetric vs row normalisation (A-21) | symmetric better by 0.29, 6.7x more stable | **row better by 0.36 (p = 0.097), symmetric 1.81x *less* stable** — reversed |
| capped vs uncapped test labels (A-02) | +1.762 cycles | +1.311 cycles, p = 0.00002 — same direction, smaller |

**The general lesson is worth more than any of the three.** None of these
comparisons was robust to a change in what each node carries. A result of the
form "configuration X beats configuration Y on this dataset" evidently depends
on the rest of the architecture in ways this project cannot currently predict,
so each such claim has to name the configuration it was measured under and
must not be quoted as a general fact. That applies to the paper's ablations as
much as to ours.

---

## A-21 — The adjacency matrix is normalised by row

**Where:** `graph/adjacency.py`, `normalize_adjacency`

**The gap.** Equation 8 of the paper is `H = ReLU(A X W)`, and the text beside
it says only that "A denotes the normalized adjacency matrix incorporating
self-loops" and that the normalisation "ensures stable gradient propagation and
balanced feature aggregation across different node degrees". The formula itself
is never printed. Searching the paper for `D`, "degree matrix" or any symmetric
normalisation returns nothing.

**What we chose.** The **row** form, `D^(-1)(A + I)`, selected under A-22.

Note where that choice lives. `normalize_adjacency` still *defaults* to the
symmetric form, because that is the one Kipf and Welling define and the
function is a general tool. The project's default is set in
`config.PreprocessingConfig.adjacency_normalization` and in the YAML files, and
it is `row`.

**Why.** Two forms fit the paper's description equally well. The symmetric
`D^(-1/2)(A + I)D^(-1/2)`, and the row-normalised `D^(-1)(A + I)`, which also
adds self-loops and also balances across node degrees — it simply averages over
each node's neighbourhood instead of taking a geometric mean of the two
degrees. The paper favours neither, so the choice falls to A-22's rule: where
the paper is silent, take the option that lands closest to the paper's own
figure. Under the configuration this project reports, that is row.

**How much it matters — and it depends entirely on the configuration.** Five
seeds of each form, everything else identical, paired by seed on test RMSE:

| Configuration | row | symmetric | better | paired t | p (two-sided) |
|---|---|---|---|---|---|
| **current default** — z-score, nothing held back | **16.694 ± 0.331** | 17.045 ± 0.201 | **row by 0.352** | −2.830 | **0.047** |
| z-score, 20 % held back, early stopping | 16.616 ± 0.289 | 16.524 ± 0.339 | symmetric by 0.091 | +0.550 | 0.611 |
| z-score, 20 % held back, no early stopping | 16.616 ± 0.289 | 16.599 ± 0.275 | symmetric by 0.017 | +0.094 | 0.930 |
| superseded — min-max, 20 % held back (`cdcc827-dirty`) | 19.992 ± 0.416 | 20.353 ± 0.752 | row by 0.360 | −2.160 | 0.097 |

**What that table says.** Row's advantage exists **only when nothing is held
back**, which is the configuration the project reports, and it is marginal even
there. Hold 20 % of the engines back and the difference vanishes into noise and
changes sign, whether or not early stopping is on. The superseded min-max row
points the same way as the default but does not reach significance.

So: row is selected under A-22 for the configuration in `config/fd001.yaml`,
and **that selection must not be quoted as a general fact about the two
formulas.** It is a fact about one configuration, which is what working rule 3
requires be said. The earlier claim that symmetric was the more *stable* of the
two has been withdrawn outright — under min-max the ratio was 1.81 the other
way, and under z-score symmetric is the tighter of the two (0.201 against
0.331) while being the worse of the two.

**A correction to the audit of 2026-08-29.** Its item 3 reported that
`p = 0.047` "is not reproducible: two-sided 0.0968, one-sided 0.0484", and that
under z-score "the sign reverses — symmetric better by 0.091, p = 0.611". Both
halves compare the wrong runs. The 0.0968 is the **min-max** comparison, the
last row of the table above, which is what this entry's own body used to
describe. The 0.091 at p = 0.611 is the **20 %-held-back** configuration, the
second row — not the default that `config/fd001.yaml` describes and that the
disputed comment sits in. Recomputed from the stored runs, `p = 0.047` is
exactly reproducible for the default: 0.0473, two-sided, no one-sided test
needed. The audit's conclusion that A-22's rule now selects symmetric does not
follow.

What the audit was right about is that `config/fd001_validated.yaml` carried
the same comment, and there it *was* wrong: that file holds engines back, so
"under this configuration" the figures are the second row, not the first. That
comment has been corrected.

All four rows are reproducible from the stored runs:

```python
import pandas as pd, yaml
from scipy import stats

def compare(row_dir, symmetric_dir):
    r = pd.read_csv(f"reports/{row_dir}/results.csv")
    s = pd.read_csv(f"reports/{symmetric_dir}/results.csv")
    m = r.merge(s, on="seed", suffixes=("_r", "_s"))
    return (m["rmse_r"] - m["rmse_s"]).mean(), *stats.ttest_rel(m["rmse_r"], m["rmse_s"])

compare("fd001_physics_row_cap_5seed_varsplit_cfgee5482_6a7af8c_20260829T125025Z",
        "fd001_physics_symmetric_cap_5seed_varsplit_noearlystop_cfgb017ba_94617bf_20260828T164621Z")
# (-0.352, -2.830, 0.0473)
```

The row run above is the clean `6a7af8c` one whose five results are
bit-identical to the `94617bf-dirty` run of the same settings, so nothing here
rests on an unidentifiable tree.

---

## A-22 — The unspecified settings were chosen to land near the paper's figure

**Where:** `config/fd001.yaml`, `config.PreprocessingConfig` defaults

**Status.** This entry exists because the configuration it describes was not
arrived at independently, and a reader must be told that before reading any
number produced by it.

**The gap.** Several settings are unspecified by Wen et al. (2026) and were
originally chosen on other grounds: A-03 (input scaling) followed Jin et al.
(2025) because they describe their procedure explicitly; A-21 (adjacency
normalisation) followed Kipf and Welling (2017) because the paper cites that
layer. Neither choice was made by looking at an error figure.

**What we chose, and on whose instruction.** On 2026-08-29 the author directed
a different rule: *where the paper is silent, choose the option that brings the
result closest to the paper's; where the paper states something, follow the
paper regardless.* Applied assumption by assumption:

| Assumption | What the paper says | Outcome under the rule |
|---|---|---|
| A-03 input scaling | nothing at all | silent → **`zscore`**, 3.83 cycles closer than min-max |
| A-21 adjacency normalisation | "normalized … incorporating self-loops"; no formula | silent → **`row`**, 0.35 closer with nothing held back (p = 0.047) |
| A-06 validation split | describes none | contradicts Table 6's 200 epochs → **removed** |
| A-18 early stopping | Table 6: 200 epochs | contradicts → **removed** |
| A-04 smoothing | Eq. 18 defines it | stated → kept, and removing it is worse anyway |
| A-01 label ceiling | Eq. 19; value inferred from Fig. 7 | evidence-backed → kept at 125 |
| A-17 LR schedule | Table 6: 1e-3 → 1e-4 | stated decay → exponential kept |

Result: **16.694 ± 0.331** on FD001 against the paper's 14.93, from
20.353 ± 0.752 under the previous defaults.

**Why this is recorded as an assumption and not as a result.** Three reasons,
all of which bear on how the number may be described.

1. **The search was directed by the target.** The sweep behind F-10 was run
   *because* of the gap to 14.93. Even where a choice was later confirmed on
   validation error, the set of options considered was chosen by looking at the
   paper's number. The configuration is therefore not independent of the
   quantity it is being compared against.
2. **A-21 was decided on test error.** Removing the validation split (A-06)
   leaves no honest basis for choosing between configurations, so the
   symmetric-versus-row comparison was made on the test set. That is the thing
   a validation split exists to prevent, and it is why
   `config/fd001_validated.yaml` still exists: **any future comparison between
   configurations must be run with that file, not this one.**
3. **Following the paper more closely made the result worse, not better.**
   Removing the validation split and the best-checkpoint restore — which is
   what Table 6's "200 epochs" actually implies — costs about 0.45 cycles.
   The rule's two clauses pull in opposite directions here, and the paper won.

**How this number may and may not be described.** It may be described as *the
closest this re-implementation comes to the published figure, under settings
the paper does not specify*. It may **not** be described as a reproduction of
the paper's result, nor as evidence that these were the authors' settings.
Nothing here shows what Wen et al. did; the paper's silence is the finding
(F-10), and this entry is the consequence of choosing to navigate that silence
by proximity.

**The honest alternative, for the thesis to consider.** Report the range
instead of a point: the paper specifies neither the scaling nor the
normalisation formula, so this reproduction spans **16.69 to 20.35 cycles**
depending on choices the source leaves open. That framing needs no A-22 at all,
and it is a stronger claim about the paper.

---

### F-07 — The paper's construction rules require an edge its matrix does not contain

Section 2.3.1 gives five rules for building the adjacency matrix, each with a
worked example. Checking every edge those examples require against the matrix
as printed:

| Rule | Edges required | Present |
|------|----------------|---------|
| 1. same engine station | `P30`-`Ps30`, `T30`-`P30`, `T30`-`Ps30`, `P15`-`BPR`, `W31`-`W32` | 5 of 5 |
| 2. physical and corrected speed | `Nf`-`NRf` | 1 of 1 |
| 3a. temperature chain `T24`→`T30`→`T50` | `T24`-`T30`, `T30`-`T50` | 2 of 2 |
| 3b. pressure connections among `P15`, `NF`, `NRf`, `BPR` | all six pairs | 6 of 6 |
| 4. `phi` influences `T30`, `P30`, `Ps30`, then `htBleed`, `W31`, `W32` | six edges | **5 of 6** |
| 5. cooling interdependencies | `W31`-`W32`, `W31`-`htBleed`, `W32`-`htBleed` | 3 of 3 |

**Twenty-two of the twenty-three checks hold. The exception is `phi`-`W32`.**
Three pairs are required by two rules each — `P15`-`BPR` (rules 1 and 3b),
`Nf`-`NRf` (rules 2 and 3b) and `W31`-`W32` (rules 1 and 5) — so counting
distinct edges rather than checks gives **19 of 20**. Either way it is the same
single edge that is absent. The text names it explicitly — "phi influencing T30, P30, Ps30 and subsequently
affecting htBleed, W31, and W32" — and the matrix gives it as 0, in Table 3 and
in the Fig. 3 colour grid alike.

Rule 3b is worth noting separately, because it is stronger than it looks: all
six pairs among `P15`, `NF`, `NRf` and `BPR` are connected, so those four
sensors form a complete subgraph. That is consistent with the rule as stated,
and it is why the fan sensors behave as one tightly-coupled group.

**This project follows the matrix, not the text**, because the matrix is what
the model consumes and it is the version the paper prints twice. Adding the
missing edge would change nothing structural: the graph would have 42 edges
instead of 41, and its diameter (4), its four distance-4 pairs, and `T24`'s
status as the only articulation point would all be unchanged.

Every structural figure quoted above can be recomputed from the package rather
than taken on trust:

```python
from thesis_gnn_rul_xai.graph.adjacency import (
    adjacency_report, build_physics_adjacency, pairs_beyond_reach)

matrix, symbols = build_physics_adjacency()
report = adjacency_report(matrix, symbols)
report["diameter"]              # 4
report["articulation_points"]   # ['T24']
report["distance_histogram"]    # {1: 41, 2: 25, 3: 21, 4: 4}
pairs_beyond_reach(matrix, symbols, n_layers=3)
# [('BPR', 'W32', 4), ('NRf', 'W32', 4), ('Nf', 'W32', 4), ('P15', 'W32', 4)]
```

So can the edge counts in the table above:

```python
from itertools import combinations
from thesis_gnn_rul_xai.graph.adjacency import PHYSICS_ADJACENCY_ROWS
from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_SYMBOLS as S

def edge(a, b):
    return PHYSICS_ADJACENCY_ROWS[a][S.index(b)] == 1

required = (
    [("P30", "Ps30"), ("T30", "P30"), ("T30", "Ps30"), ("P15", "BPR"),
     ("W31", "W32")]                                          # rule 1
    + [("Nf", "NRf")]                                          # rule 2
    + [("T24", "T30"), ("T30", "T50")]                         # rule 3a
    + list(combinations(["P15", "Nf", "NRf", "BPR"], 2))       # rule 3b
    + [("phi", x) for x in ("T30", "P30", "Ps30", "htBleed", "W31", "W32")]
    + [("W31", "W32"), ("W31", "htBleed"), ("W32", "htBleed")] # rule 5
)
sum(edge(*p) for p in required), len(required)      # (22, 23)
distinct = {frozenset(p) for p in required}
sum(edge(*tuple(p)) for p in distinct), len(distinct)   # (19, 20)
[p for p in required if not edge(*p)]                   # [('phi', 'W32')]
```

**What it means for the thesis.** It is a small internal inconsistency, but it
bears on how much of the graph is really "physics-informed". If the text is the
intent and the matrix is the artefact, then the matrix was not generated from
the rules programmatically — it was written out by hand, and one cell was
missed. That is worth one sentence when describing how the prior was built.

A second, independent sign points the same way. **Table 6 carries Table 2's
caption.** Both are printed as "Characteristic description of C-MAPSS
sensors." — Table 2 on page 5, where it is correct and heads the sensor list,
and Table 6 on page 9, where the body beneath it is a hyperparameter table
(`Hyperparameter | Value`, giving the 200 epochs, batch size 32 and learning
rates this project takes from it). A caption duplicated onto the wrong table is
not evidence about the adjacency matrix specifically, but it is evidence that
the tables were assembled and proofed by hand rather than generated, which is
the assumption the paragraph above rests on. Verified against the PDF: the
twelve table captions were extracted and compared, and 2 and 6 are the only
pair that match.

---

### F-08 — The paper's Table 12 cannot hold signed Shapley values

This is the strongest objection available to the interpretability half of the
paper, and it follows from the paper's own equations.

**What Equation 14 requires.** It states

```
g(z) = phi_0 + sum_i phi_i
```

with `phi_0` "the average value of target variable across all samples" — a
single constant, not a per-time-step quantity. Rearranged, the fourteen
Shapley values at time `t` must satisfy

```
sum_i phi_i(t) = f(t)(x) - phi_0
```

**What that implies over an engine's life.** The paper's Table 12 covers graphs
G1 to G191, so the engine ran about 220 cycles. Across that life the prediction
`f` falls from the label ceiling to nearly zero. Two consequences follow, and
neither depends on any assumption of ours:

1. the column sums must span roughly the whole label range — about 125 cycles;
2. they must **change sign**, because `f` starts above the constant `phi_0` and
   ends below it.

**What the table actually contains.** Summing its printed columns:

| | G1 | G2 | G5 | G82 | G189 | G190 | G191 |
|---|----|----|----|-----|------|------|------|
| sum of the 14 values | 45.51 | 41.33 | 33.92 | 26.38 | 18.16 | 16.67 | 17.60 |

The sums fall from 45.51 to 16.67, a span of **28.8**, and never approach zero,
let alone cross it. Independently: **all 210 cells are positive**, with a
minimum of 0.20. A set of signed Shapley values for a regression that traverses
its label range essentially cannot be all-positive.

**Conclusion.** Table 12 reports magnitudes — `|phi|`, or some other summary —
not the signed values Equation 14 defines, and the paper never says which. Its
Equation 14 is printed but is not satisfied by the table it licenses.

**Three consequences for this project.**

1. `dynamic_shap` returns **signed** values in cycles. Its output must not be
   placed beside Table 12 without saying that the two quantities differ.
   `explain/README.md` says so under "Trap: our table and the paper's Table 12
   are not comparable", and `dynamic_shap` points at it. This warning first went
   into the docstring of that function, which had previously claimed the
   opposite; it moved to the README on 2026-08-29 when the explanation was
   taken out of the code.
2. The exact implementation here satisfies efficiency to about `1e-4` cycles,
   asserted by `test_shap.py::test_efficiency_holds_to_machine_precision` and
   again per time step by `test_dynamic_shap_rows_each_satisfy_efficiency`. So
   this project can **demonstrate the property the source paper states and its
   own table violates**. That is a result, not a complaint, and it is the
   clearest thing the reproduction has produced about the paper's XAI half.
3. Any comparison of sensor *rankings* with the paper remains legitimate, since
   a ranking by magnitude is well defined either way. F-05's agreement about
   `P15` is therefore unaffected.

---

### F-09 — A results file survived in the repository that no committed code produced

Not a finding about the paper, but about this project's own practice, and it is
recorded here because rule 2 exists to prevent exactly it.

`reports/fd001_physics_window_symmetric_5seed/results.csv` reported
**13.810 ± 0.161** with best epochs `[157, 148, 162, 148, 126]`, while
`README.md`, A-21, F-04 and F-06 all quoted **13.45 ± 0.20**. An outside review
found the disagreement and concluded, reasonably, that the documentation was
stale.

It was the other way round. Re-running settled it:

| Source | RMSE | best epochs |
|--------|------|-------------|
| documented in this repository | 13.452 ± 0.202 | — |
| re-run at the current commit | **13.452 ± 0.202** | 87, 135, 132, 113, 139 |
| re-run at commit `e5b15be`, seeds 42–43 | **13.408, 13.501** | 87, 135 |
| the stored `results.csv` | 13.810 ± 0.161 | 157, 148, 162, 148, 126 |

Three of the four agree to three decimals, per seed and per best epoch. Two
candidate explanations were tested and rejected: the intervening edit to
`data/windowing.py` is a pair of semantic no-ops (`round()` already returns an
`int`, and the other change sorts `__all__`), and training is bit-identical
across repeated runs regardless of thread count, so it is not non-determinism.
The stored file's `summary.json` records settings identical to the reproducing
run. **Its provenance could not be established**, and it has been deleted.

What that is worth: a results file whose origin cannot be traced is not
evidence, whichever way it points. It happened to contradict a figure that was
correct; it could as easily have corroborated one that was wrong, and nothing
in the repository would have caught it.

**The fix, now in `scripts/train.py`.** Every run records the commit it was
produced at — with a `-dirty` suffix when the working tree has uncommitted
changes — in both `summary.json` and the run directory name, together with a
UTC timestamp. Two runs of the same configuration can no longer overwrite one
another, and any stored result can be traced to the code that produced it.

**A second gap in the same scheme, closed 2026-08-28.** The directory name
carried the graph, the normalisation and the split, but not
`clip_test_labels`. The A-02 measurement therefore produced two directories
identical except for their timestamps, whose RMSEs differ by 1.3 cycles. The
timestamp made them unique, which is what the original fix promised, but a
name you cannot read is not provenance. The name now carries the cap, and
marks early stopping and the learning-rate schedule when they are not at their
defaults.

**And the first attempt at that fix was itself incomplete, which is the more
useful half of this entry.** Naming the fields that happen to matter today
covers command-line switches and nothing else. The assumption sweep of
2026-08-28 varied the normalisation method and the smoothing parameter from
YAML files, and produced two more directories differing only by timestamp
whose RMSEs are 5.0 cycles apart. Enumerating fields cannot fix a problem
whose cause is that someone has to remember to enumerate a field.

The name therefore also carries `cfg<digest>`, the first six hex characters of
a SHA-256 of the whole preprocessing configuration. Two runs whose settings
differ *anywhere* get different names, whether or not anyone remembered:

```
fd001_physics_symmetric_cap_5seed_varsplit_cfge50b20_<commit>_<utc>
fd001_physics_symmetric_cap_5seed_varsplit_cfgb8071d_<commit>_<utc>   (zscore)
fd001_physics_symmetric_uncap_5seed_varsplit_noearlystop_cfg00375a_<commit>_<utc>
```

The digest is also written into `summary.json`. The readable fields stay,
because a digest tells you two runs differ without telling you how.

The general point, which is the one worth keeping: **make the identifier
derive from the configuration, rather than from a list of the parts of the
configuration someone thought of.** A list has to be maintained; a digest does
not.

---

### F-10 — The input scaling the paper never describes accounts for most of the gap

Finding F-06 records this project landing at 20.353 ± 0.752 on FD001 against
the paper's 14.93 for "ESTGNN w/o LSTM" — a 5.42-cycle shortfall on an
architecture that matches Table 4 exactly. This entry asks where those 5.42
cycles are, and answers it by changing one assumption at a time.

Five seeds each, commit `09628fa`, physics graph, symmetric normalisation,
capped test labels, varying split. Every row differs from the baseline in
exactly one respect. `p` is a paired t-test across seeds.

| Configuration | RMSE | vs baseline | p | gap to 14.93 |
|---|---|---|---|---|
| **A-03: `zscore` scaling** | **16.524 ± 0.339** | **−3.828** | **0.0005** | **+1.59** |
| A-18: no early stopping (the paper's fixed 200 epochs) | 20.177 ± 0.932 | −0.175 | 0.374 | +5.25 |
| baseline (`fd001.yaml`) | 20.353 ± 0.752 | — | — | +5.42 |
| A-17: constant learning rate | 20.518 ± 0.493 | +0.165 | 0.254 | +5.59 |
| A-01: `rul_max = 130` | 20.822 ± 1.065 | +0.469 | 0.114 | +5.89 |
| A-04: smoothing disabled | 21.572 ± 0.315 | +1.219 | 0.007 | +6.64 |

**One assumption dominates, and it is the one the paper is silent about.**
Switching the input scaling from per-condition min-max to z-score removes
**3.83 of the 5.42 cycles — 71 % of the gap** — and it is the only change
tested that is both large and significant. A-03 was already flagged as "the
largest single unknown"; it is now the largest measured one.

**Correction, 2026-08-29.** An earlier version of this entry reported that
"combining it with the paper's own training procedure changes nothing further:
z-score with early stopping disabled and a fixed 200 epochs gives
16.599 ± 0.275". That run was not the paper's training procedure. With a
validation set present, `--no-early-stopping` stops the loop from *exiting*
early but the loop still ends with `model.load_state_dict(best_state)`, so the
reported model is the best-validation checkpoint — model selection on a
held-out set, which Table 6 does not describe. Only `validation_fraction: 0.0`
keeps the epoch-200 weights.

The genuinely paper-faithful training procedure is **17.045 ± 0.201**
(commit `94617bf`), 0.45 cycles worse than the 16.599 that was mislabelled.
So the A-18 and A-06 deviations *were* flattering the figure, by about half a
cycle, and the corrected statement is the opposite of the original: following
the paper's training procedure more closely makes the result worse.

| Configuration under z-score | RMSE | Reported model |
|---|---|---|
| val 0.2, early stopping | 16.524 ± 0.339 | best-validation checkpoint |
| val 0.2, 200 epochs | 16.599 ± 0.275 | best-validation checkpoint |
| **val 0.0, 200 epochs — the paper's procedure** | **17.045 ± 0.201** | **epoch 200** |

**What this does and does not establish.**

It does **not** show that Wen et al. used z-score scaling. They describe no
scaling at all; Fig. 1 has a "Data Normalization" box and the text never
returns to it. A-03's default is now the z-score, selected by A-22's rule
rather than borrowed from anywhere, and that choice is the single largest
reason a number from this code is not comparable with a published one.

It does show that **the paper's silence on preprocessing is worth more than
every architectural detail it does print.** Table 4 specifies the network to
the width of each weight matrix; Table 6 specifies the optimiser, the loss,
the batch size, the epoch count and the learning-rate range. Following all of
that exactly still leaves a 5.4-cycle gap, and one unstated preprocessing
choice closes three quarters of it. For the thesis this is the more useful
observation than any single number: **reproducibility here was lost in the
step that was not described, not in the steps that were.**

**Still unaccounted for: 1.59 cycles.** Candidates not yet separated, all
documented and none measured: A-13 (how the sigmoid output maps back to
cycles), A-14 (weight initialisation), A-05 (window stride), A-10 (the
identity of the seventh dropped sensor), and A-02 in reverse — capping the
test labels *helps* by a measured 1.311 cycles (see A-02), so an uncapped
comparison would put the residual gap at roughly 2.9 rather than 1.59.

**Do not read the ordering of the small rows as a result.** A-18, A-17 and
A-01 all have p > 0.1 across five seeds; only their sign is suggestive, and
A-04 (+1.219, p = 0.007) is the only other row that clears significance.

### F-11 — The Conclusions claim a "highest R²" the paper's own Table 9 refutes

**Where:** the paper, Conclusions, item (1); its Table 9; its Section 4.1

Conclusion (1) on the paper's final page reads:

> "The developed ESTGNN framework achieves superior performances with the
> lowest average SF of 1023.0, the best RMSE of 14.75, and the highest R² of
> 0.742."

Two of those three superlatives hold. Checked against the tables they refer to:

| Claim | Table | Best in that table | Verdict |
|---|---|---|---|
| lowest average SF, 1023.0 | 7 | ESTGNN 1023.0, then Transformer 1238.8 | **holds** |
| best RMSE, 14.75 | 8 | ESTGNN 14.75, then Transformer 16.62 | **holds** |
| highest R², 0.742 | 9 | **Transformer 0.759**, DCNN 0.757 | **fails** |

Table 9's average R² column, transcribed in full: SVR 0.551, CNN 0.655, DCNN
0.757, LSTM 0.718, TCN 0.704, Transformer **0.759**, GRU 0.703, Bi-LSTM 0.742,
ESTGNN 0.742. ESTGNN ties Bi-LSTM for **third place of nine**; two methods in
the paper's own comparison score higher.

**The paper concedes this in its own Section 4.1**, four pages earlier:

> "ESTGNN achieves competitive performance with an average R² value of 0.742,
> tied with Bi-LSTM. While the Transformer method demonstrates slightly
> superior average R² performance (0.759), ESTGNN maintains competitive
> variance explanation capability while excelling in other critical metrics."

That is an accurate description of Table 9. The Conclusions then restate it as
"the highest R²", which it is not.

**Why this is worth recording.** It is the second independent case in this
paper of the prose outrunning the tables — F-08 shows that Table 12 cannot hold
the signed Shapley values its own Equation 14 defines. Unlike F-08, this one
needs no assumption of ours and no reconstruction: it is a comparison between
two statements the paper makes about itself, one of which is checkable against
a table printed in the same paper.

It bears on how the source should be read. Where the paper's narrative and its
numbers disagree, this project follows the numbers — the same rule applied in
F-07 (the matrix over the construction text) and in A-15 and A-16 (the prose
over the misprinted equations, because there the prose was the coherent half).
The rule is not "prefer one or the other" but "prefer whichever survives being
checked".

**What it does not mean.** ESTGNN's R² is not bad, and the paper's headline
claims about SF and RMSE are unaffected — both were verified above. The defect
is one word in a summary, not a result. It should be quoted in the thesis as
evidence about the paper's editing rather than about its method.

**Verified from the PDF**, not from memory: Table 6 through Table 10 are on
pages 9 and 10, and the Conclusions list is on page 15.

---

---

## References

Full citations for every work mentioned above are in `REFERENCES.md`.
