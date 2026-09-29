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
| A-06 | Validation split | None; trains for a fixed 200 epochs | **20 % of the engines**, reinstated 2026-09-02 after the A-22 removal; worth 0.32 cycles, which five seeds cannot distinguish from zero (p = 0.234) |
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
| A-17 | Learning-rate schedule | Both endpoints given, mechanism not | **One drop of x0.1 at the halfway epoch**, so only the two printed rates are used; indistinguishable from exponential (p = 0.39) |
| A-18 | Early stopping | None; trains a fixed 200 epochs | **On**, since the split returned on 2026-09-02; fires between epochs 8 and 55 of a nominal 200 |
| A-19 | What "without physics prior" removes | Names the ablation, never defines it | Replace the graph with a fully connected one |
| A-20 | What "a sensor not considered" means | Equation 15 needs `f(S)`, but a network cannot take 13 inputs | Replace the sensor with its average training window |
| A-21 | Adjacency normalisation formula | Says only "normalized ... incorporating self-loops"; never prints it | **Row `D^(-1)(A+I)`** — 0.25 cycles closer *with nothing held back* under the step schedule (p = 0.068, **no longer significant**), selected under A-22 |
| A-23 | Engines shorter than the window | Not discussed; 17 test engines across FD002 and FD004 are affected | Left out, following the only published treatment found; no FD001 engine is affected |
| A-24 | What `phi_0` is in Equation 14 | Calls it "the average value of target variable across all samples" | The prediction at the average window, forced by A-20; **10.719 cycles** from the paper's wording |
| A-26 | Size and depth of the recurrent stage | Table 4 has **no recurrent row**; Table 6 lists no hidden size, depth or projection | One layer, hidden size 32, a linear projection to the one number Table 4 prints; **fourteen dedicated cells**, which the paper does state |
| A-25 | Where the scaling is fitted | Fig. 1 puts normalisation **before** the train and test datasets; never names the fitting set | Split first, fit on training engines only |
| A-27 | A sensor graph built from the data | Never builds one; its Conclusions claim its own graph goes "beyond data-driven correlations" | The 41 most strongly correlated pairs on the training engines, as many edges as Table 3; **indistinguishable from Table 3 by test RMSE** (+0.096, p = 0.655), 2.0 cycles better on the held-back engines (p = 0.012), but its test predictions run later |
| A-28 | Each sensor's last reading, average and change as node features | Never considered; Table 4 prints one number per sensor, from the LSTM stage | A stage outside the paper, not the default; **2.910 cycles below the last cycle** (p = 0.00025, engine interval [−5.19, −0.63]), indistinguishable from the recurrent stage on test error |
| A-29 | Explaining the sensor graph itself | Explains sensor nodes only (Section 2.4), yet says its rankings set "edge weights" and validate its graph (Section 4.4) | Walk shares from GNN-LRP, of the score before the sigmoid, gamma 0 by default and always at the first graph layer; on the default network about half the score is the dense layers' constant (51 ± 4 %), and how much stays within a sensor's group is mostly set by the graph |
| A-30 | What removing one connection means | Removes the whole graph once and retrains; never one connection | Delete it from the raw graph both ways, rescale, rerun without retraining; on the default network its effect is unrelated to what flowed along it (rank correlation −0.16 ± 0.20, p = 0.158) |
| A-31 | Placing explanation nodes on an engine | Table 2 names variables and component regions; Fig. 3 places a separate graph beside a schematic engine | An original schematic cutaway with section anchors, distinct markers for calculated and system quantities, and the stored model's graph overlaid; no claim of exact probe coordinates |

---

## A-01 — The label ceiling is 125 cycles

**Where:** `config.PreprocessingConfig.rul_max`, `data/labeling/`

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

**Where:** `config.PreprocessingConfig.clip_test_labels`, `data/pipeline/`

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

**Where:** `config.PreprocessingConfig.normalization`, `data/normalization/`

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

**Where:** `config.PreprocessingConfig.smoothing_alpha`, `data/smoothing/`

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

**Where:** `config.PreprocessingConfig.window_stride`, `data/windowing/`

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

## A-06 — Twenty percent of the training engines are held out

**Reinstated 2026-09-02, reversing the removal of 2026-08-29.**
`config/fd001.yaml` holds out 20 % of the training engines again, so a run
restores the weights that scored best on them and may stop early.

**What that costs, measured.** Five seeds each at commit `ae59bc1-dirty`,
physics graph, row normalisation, z-score scaling, capped test labels, step
schedule. The seed varies which engines are held back as well as the
initialisation, so the spread is what a real repetition would show.

| Configuration | RMSE | closer to 14.93 by | paired two-sided p |
|---|---|---|---|
| `validation_fraction: 0.2` — the default since 2026-09-02 | **16.496 ± 0.415** | 0.321 | 0.234 |
| `validation_fraction: 0.0` — the default from 2026-08-29 to 2026-09-02 | 16.818 ± 0.157 | — | — |

**The difference does not survive the spread, and that is the finding.**
Holding engines out lands 0.32 cycles closer to the paper's figure, but two
of the five seeds go the other way (−0.103 and −0.100 against +1.022,
+0.707 and +0.081) and the paired t-test gives p = 0.234. This is not
evidence that the split improves the number. It is evidence that five seeds
cannot tell, which is a different statement and the only one supported.

**Three things came back together, and this pair of runs cannot separate
them.** With a validation set present, `training/run.py` also stops early
and ends with `model.load_state_dict(best_state)`. The five runs stopped at
epochs 51, 55, 8, 18 and 35 of a nominal 200 — one of them at epoch 8. So
the comparison above is not "a split against no split"; it is "a split,
early stopping and best-checkpoint selection against a fixed 200 epochs".
Separating them needs a third arm with `early_stopping_patience: null`,
which has not been run.

**What this does to A-22.** That entry chose to remove the split because
Table 6 states 200 epochs and its rule lets the paper win where the paper
speaks. The reversal overrides that choice for the default configuration;
A-22's own headline figure of 16.818 ± 0.157 was measured under the removed
arrangement and is labelled there accordingly.

---

### What this entry said between 2026-08-29 and 2026-09-02

**Superseded 2026-08-29 by A-22.** This entry originally held out 20 % of the
training engines unconditionally. Table 6 states 200 epochs, and holding
engines out in order to restore the best-validation checkpoint is
validation-based model selection, which the paper never describes; under A-22's
rule the paper wins. `config/fd001.yaml` therefore held nothing out, until
the reversal recorded above put the split back on 2026-09-02.

**The split has not been abandoned, only moved.** `config/fd001_validated.yaml`
keeps it, and **every comparison between configurations must use that file** —
without it there is no honest basis for preferring one setting to another, and
A-22 records one choice (A-21) that had to be made on the test set for exactly
this reason. The reasoning below is unchanged and still governs how the split
must be used when it is used.

*Note added 2026-09-14.* Once the split returned to `config/fd001.yaml`, the
separate file became identical to it in every setting, and it was deleted.
Comparisons between configurations now use `config/fd001.yaml` itself. Stored
runs made with the old file are unaffected: each run folder keeps its own
copy of the settings it ran with, in `preprocessing_config.yaml`.

---

### Original entry

**Where:** `config.PreprocessingConfig.validation_fraction`, `data/windowing/`

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

**Amended 2026-09-09: the choice moved into the model.** It used to be made
in `data/windowing/`, one line before the pipeline returned, where it looked
like a property of the dataset. The pipeline now hands over every cycle of the
window and `models/temporal/LastCycle` performs the reduction, so the width of
the first graph weight matrix is read off the **stage** rather than off the
**data**. Nothing about the arrangement below changed: the same number reaches
the same layer, and the five-seed run recorded under A-26 reproduces the
previous figure. What changed is that there is no longer a path from a
preprocessing setting to a thirty-wide `W(1)`, which is how the 2026-08-28
mistake became possible. `ESTGNN` now refuses to build a stage that widens a
node without learning anything across the cycles, unless the caller asks for
it in words and the run records that it did.

**Where:** `data/windowing/`; `models/estgnn.py` (`in_features`)

**The gap.** Table 4 of the paper gives the input to the first graph layer as
`[batch, 14, 1]` with `W(1) in R^(1 x 32)`: one number per sensor. That number
is produced by a per-sensor LSTM whose size, depth, and output projection are
never stated anywhere in the paper. What is *not* a gap is the shape: the
paper prints it.

**What we chose.** With the LSTM stage omitted (see A-11), each sensor node
carries the **most recent reading** of its window, giving `[batch, 14, 1]` and
`W(1) in R^(1 x 32)` exactly as printed. The parameter count comes to
**61,729**, asserted by
`test_models.py::test_the_default_width_reproduces_table_4`.

**What that 61,729 is, exactly — corrected 2026-08-31.** Earlier drafts of this
entry, of `README.md` and of `models/README.md` called it "Table 4's count",
"the count Table 4 gives" and "matching Table 4 to the digit". **Table 4 prints
no total.** It prints five weight-matrix shapes, and those come to

```
1x32 + 32x64 + 64x32 + 448x128 + 128x1  =  61,600
```

The remaining **129** are the two biases `b_fc1` and `b_fc2`, which Table 4 does
not list but Equations 12 and 13 do print. And the total stays at 61,729 rather
than 61,857 only because **A-12** gives the three graph convolution layers no
bias — a choice of ours, made because Equations 8 to 10 print none.

So the figure is a *derivation* from three places in the paper plus one of this
project's own assumptions, not a number read off a table. It is still a real
and useful check — it caught the 62,657-parameter Run A described in F-06 — but
calling it "Table 4's count" let a derived quantity wear the clothes of a
printed one, which working rule 5 exists to prevent. Found by the audit of
2026-08-31.

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

**Where this choice sits in its field — and it does not sit comfortably.**
**F-12** read the twelve other papers filed alongside the source. Not one of
them puts a single number on a graph node: every one either hands the graph the
raw window or runs a temporal encoder first and passes its **output vector**,
the smallest of which is thirty numbers wide. Neither the last reading nor the
mean appears anywhere in that corpus. So the choice made here is not the
field's default — it is a consequence of refusing to depart from the shape
Table 4 prints, and the honest way to describe it is "unusual, and adopted for
a stated reason", not "natural". The nearest comparable design, HAGCN, gives
each of the same fourteen sensor nodes thirty numbers and lets its recurrent
stage widen them to sixty; the source paper's stage has to narrow thirty to
one.

**The mean has still never been measured.** The figures above compare this
choice against the removed `window` mode, not against the other option that
keeps Table 4 intact. Averaging the window instead of taking its last cycle
leaves `W(1) in R^(1 x 32)` and the parameter count untouched, so it is a
one-line change and a five-seed run. Until it is run, "how much it matters"
answers a different question from the one this entry poses.

**Note on what `window_size` still does.** It is not inert. It decides which
cycles are eligible — an engine's first example ends at cycle `window_size`,
so cycles 1 to 29 are never a prediction point — and it is what a temporal
encoder would consume if one were added later.

---

## A-08 — WITHDRAWN: the paper does state the order

**Where:** `data/pipeline/`

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

**Measured on the real FD002 engines**, which have six operating conditions.
For each of the 14 sensors, the absolute Pearson correlation between its
processed values and the remaining-life label, averaged over the 14. Regenerate
with `python scripts/verify_claims.py --only ordering`:

| Order | mean \|correlation\| with remaining life |
|-------|------------------------------------------|
| smoothing → scaling (the paper's order, Fig. 1) | **0.054** |
| scaling → smoothing (condition-aware) | **0.591** |
| scaling only, no smoothing | 0.485 |

Smoothing first costs **more than a factor of ten**. It also shows that the
smoothing is worth having once it is applied in the right place: 0.591 against
0.485 without it.

**An earlier version of this measurement used a constructed six-condition
signal with a known wear curve**, and reported 0.045 / 0.997 / 0.993. Those
figures are not reproducible here and are not quoted as a result, because
working rule 7 forbids synthetic data in this project — a pipeline that appears
to work on invented signals says nothing about whether it works on engines. The
real-data measurement above reaches the same conclusion by a wider margin than
is comfortable, and it is the one to cite.

This is a property of the published method, not a defect introduced here, and
it is a candidate explanation for the paper's much weaker multi-condition
results (RMSE 16.21 on FD002 and 18.67 on FD004 against 11.78 on FD001; R²
falling to 0.603 and 0.510). Before this pipeline is run on FD002 or FD004 the
order must be revisited and the deviation from the paper recorded as a new
assumption.

---

## A-09 — The recursive form of the smoothing filter is used

**Where:** `data/smoothing/`

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

**Where:** `data/sensors/`

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
it was dropped for a different reason than the other six. The paper's sentence
conflates two different reasons for exclusion.

**The paper's own criterion settles it — measured 2026-08-31.** Section 3.2.2
gives the rule as removal for "high correlational redundancy (above 0.95) and
constant values". Applying exactly that rule to the real FD001 training set:

- **six** sensors are constant — `T2`, `P2`, `epr`, `farB`, `Nf_dmd`,
  `PCNfR_dmd` — which are precisely the six the paper names;
- among the fifteen sensors that are *not* constant, **exactly one pair**
  reaches the 0.95 threshold: **`Nc`–`NRc`, at r = +0.9632.** No other pair
  comes close; `NRc`'s next strongest partner is `htBleed` at +0.2466.

So the paper's stated rule, applied to the paper's stated dataset, removes six
sensors by constancy and exactly one more by redundancy, and that one can only
be `Nc` or `NRc`. Table 3 keeps `Nc`. The seventh removed sensor is therefore
`NRc` by the paper's own criterion, not merely by elimination from its tables.

This is worth noting for a second reason. It is **one of the few places where a
procedure the paper describes can be run and shown to reproduce the paper's own
result exactly.** Most of this log records the opposite. Recording the
agreement as carefully as the disagreements is what keeps the log evidence
rather than advocacy.

**How much it matters.** None for the code, once identified: the retained set
is fully determined by Table 3. It is worth a sentence in the thesis, because
it shows the selection was not purely a variance filter — and a second
sentence, because the selection is reproducible where much of the paper is not.

---

## Findings about the source paper, established from the data

These are not choices made by this project. They are observations produced by
running the pipeline on the real FD001 files, and they belong in the thesis.

### F-01 — P15 is in the graph but carries almost no information

The bypass-duct pressure `P15` is one of the 14 sensors in the paper's
adjacency matrix. On the FD001 training set it takes only **two** distinct
values in 20 631 rows: 21.60 and 21.61. A single reading with two possible
levels cannot follow gradual wear.

**Corrected 2026-09-16.** This entry first said a two-level signal "cannot
represent gradual wear", which is too strong. How *often* the sensor sits at
its lower level does change with wear. Measured on the FD001 training rows,
grouped by cycles left before failure:

| Cycles left | more than 125 | 101–125 | 76–100 | 51–75 | 26–50 | 0–25 |
|---|---|---|---|---|---|---|
| Rows at 21.60 | 4.00 % | 1.52 % | 1.28 % | 0.52 % | 0.08 % | 0 % |

So `P15` carries a weak sign of wear, not none, which is what the heading
already says. How weak: its correlation with the capped label is −0.108,
against −0.678 for `T24`, −0.757 for `T50` and −0.775 for `Ps30`. It was
found while checking the Learn pages, by counting the rows at each level.

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

## A-11 — The per-sensor LSTM stage is omitted from the default

**Amended 2026-09-09: it can now be built.** `models/temporal/LSTMStage` is
the paper's Section 2.2 stage, fourteen dedicated networks, one per sensor.
It is **not** the default: `--temporal last_cycle` remains what a run does
unless told otherwise, so every figure in this log produced before that date
still describes what it says it describes, and results from the default
still compare against the paper's own "ESTGNN w/o LSTM" row and never its
headline. What the paper leaves unstated about that stage is now A-26 rather
than a reason not to build it. The reasoning below is unchanged and is why
the omission was right at the time.

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

## A-17 — The learning rate drops once, by a factor of ten, at the halfway epoch

**Where:** `training/config.py`, `TrainingConfig.lr_schedule`

**The gap.** Table 6 gives `1e-3 -> 1e-4` and Section 3.4 calls it "an adaptive
learning rate schedule". The mechanism is never stated: the sentence is
consistent with a smooth decay, a step at a fixed epoch, or a reduction
triggered by a plateau in the loss.

**What we chose.** One drop, multiplying the rate by `0.1` at the halfway
epoch — epoch 100 of the paper's 200. The run spends its first half at `1e-3`
and its second half at `1e-4`, and uses no other value at any point.

**Why.** The paper's two rates are exactly one factor of ten apart, which is
precisely one `x0.1` step. A single step reaches `1e-4` and holds it there; an
exponential curve reaches it only on the very last epoch and spends the whole
run at values the paper never mentions. Reading the paper as stepping requires
it to have omitted *when* the step happens. Reading it as decaying smoothly
requires it to have omitted the mechanism entirely.

And no paper on this benchmark decays it exponentially. Of three C-MAPSS papers
that state a schedule at all, two step at fixed epochs and one reduces when the
loss stops falling; all three are quoted with page numbers in
`docs/LITERATURE_BASIS.md`. Li, Zhao, Sun, Yan and Chen (2021) write on their
page 6 that "learning rate decay is adopted, which decreases the learning rate
by **multiplying 0.1 in epochs 120 and 160**".

Where the step falls is still ours to choose, and the paper says nothing about
it. The halfway point gives equal time to each of the two rates the paper does
print, which adds the least on top of what is stated. `lr_step_epoch` moves it
for anyone who wants to measure that choice separately.

**How much it matters — measured, and the answer is: not measurably.** Five
seeds of each schedule, everything else identical, paired by seed. FD001,
physics graph, row normalisation, z-score, capped test labels, commit
`989f3d8-dirty`:

| Measure | step | exponential | difference | paired t | p (two-sided) |
|---|---|---|---|---|---|
| **RMSE** | **16.818 ± 0.157** | **16.694 ± 0.331** | +0.124 | +0.96 | **0.393** |
| NASA score | 609.8 ± 25.5 | 613.4 ± 34.5 | −3.6 | −0.19 | 0.860 |
| R-squared | 0.824 ± 0.003 | 0.826 ± 0.007 | −0.003 | −0.95 | 0.397 |
| ASUE | 12.333 ± 0.598 | 11.638 ± 0.468 | +0.695 | +2.36 | 0.078 |

On the measure this project reports, the two cannot be told apart: 0.124 cycles
against a seed-to-seed spread of 0.331. The score and R-squared agree. **The
choice therefore rests on the argument above and not on the result** — and the
step schedule is the slightly *worse* of the two on RMSE, so it is plainly not
being adopted because it flatters the reproduction.

**The one measure that is not silent.** ASUE differs by 0.695, larger than
either standard deviation, at `p = 0.078`. That does not reach significance at
five seeds, and it is the only one of the four to come close. It is recorded
rather than rounded away: it is weak evidence that the step schedule is a
little worse at the late-prediction penalty, and anyone relying on ASUE
specifically should measure it with more seeds first.

**What this replaced, and why it changed.** Until 2026-09-01 this entry chose
exponential decay, arguing that it reaches both stated endpoints and needs no
invented constant, where a step needs an invented epoch. Half of that still
holds — the step epoch *is* invented. What it missed is that the exponential
reading invents more, not less: an entire trajectory through rates the paper
never names. The audit of 2026-08-31 then found that no C-MAPSS paper decays
exponentially, and the entry was rewritten to follow the evidence rather than
the earlier reasoning.

**What moved downstream.** Because this is a default, every figure measured
under the old one was re-measured at the new one. F-04 keeps its direction and
its strength; A-21 keeps its direction but **no longer reaches significance**.
Both tables below carry a row for each schedule.

---

## A-18 — Early stopping ends training when the held-out error stops improving

**Live again since 2026-09-02, with the split of A-06.** Early stopping was
never switched off in code — `training/config.py` has held
`early_stopping_patience = 20` throughout — so it was disabled only in the
sense that `validation_fraction: 0.0` left it nothing to measure. With 20 %
of the engines held back it fires again, and in the five-seed run recorded
under A-06 it fired at epochs 51, 55, 8, 18 and 35 of a nominal 200.
**Table 6's "200 epochs" is therefore not what the default configuration
runs**, and any number produced by it must be described that way.

---

### What this entry said between 2026-08-29 and 2026-09-02

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

*Correction, 2026-09-14.* "Changing nothing else" was wrong. A fully connected
graph also changes how many edges there are, and once its rows are rescaled
every sensor receives the same average of all fourteen, which the audit of
2026-09-08 raised as its item D-2. A graph built from the data with the same 41
edges as Table 3 (assumption A-27) removes the knowledge while keeping the
count, and under the current default it costs 0.096 cycles where this graph
costs 2.784. So this choice measures mostly the averaging, not the knowledge.
It is kept because it is still the most literal reading of "without physics
prior", but no number from it should be described as the value of the
engineering knowledge without A-27's figure beside it.

**How much it matters.** A great deal, as F-04 below records — and, as the
correction above says, for a reason other than the one intended.

---

### F-04 — What the physics graph is worth has now been measured three times, with three answers

**Status.** This entry has reversed twice. It first claimed the physics prior
was worth 5.7 cycles, was withdrawn and reversed to "worth nothing
measurable", and now measures 2.4 cycles at `p = 0.000002`. Nothing about the
graph changed between those measurements. What changed was the preprocessing
and the split — choices the paper does not specify.

**Added 2026-09-14: the current default, and what the fully connected graph
really measures.** The table below gains a row for the configuration
`config/fd001.yaml` now sets, 20 % of engines held back, which this entry did
not have: **+2.784 cycles, p = 0.00046**, the same direction as the
nothing-held-back row. In the same week a third graph was measured beside
these two — the 41 sensor pairs whose readings correlate most strongly, as many
edges as Table 3 (assumption A-27) — and it costs only **+0.096 cycles,
p = 0.655**. That graph removes the engineering knowledge but keeps the number
of edges, so the large figures in this table are mostly the cost of joining
every sensor to every other, which makes all fourteen rows of the rescaled
table identical, and not the value of the knowledge. Read A-27 before quoting
any row here as what the physics prior is worth.

That instability is the finding. The individual numbers are less interesting
than the fact that an ablation result moved from *decisive*, to *zero*, to
*decisive again* under settings the source leaves open.

**All five measurements, five seeds each, paired by seed on test RMSE.** The
second and third rows differ only in the learning-rate schedule of A-17 and
agree with each other, which is the useful part: the instability below came
from the preprocessing, not from every setting the paper leaves open.

| Configuration | physics | fully connected | cost of removing it | p |
|---|---|---|---|---|
| **current default** — z-score, row, **20 % held back**, step schedule (`f386291-dirty`, 2026-09-14) | **16.496 ± 0.415** | 19.281 ± 0.432 | **+2.784** | **0.00046** |
| z-score, row, nothing held back, **step** schedule (`bc20edd-dirty`) — the default until 2026-09-02 | 16.818 ± 0.157 | 19.214 ± 0.056 | +2.396 | 0.000002 |
| the same under the superseded **exponential** schedule (`9548104`) | 16.694 ± 0.331 | 19.213 ± 0.056 | +2.520 | 0.000057 |
| min-max, symmetric, 20 % held back (`cdcc827-dirty`) | 20.353 ± 0.752 | 20.397 ± 0.395 | +0.044 | 0.913 |
| superseded `window` mode (`cdcc827`) | 13.45 ± 0.20 | 19.17 ± 0.45 | +5.7 | — |

**Both z-score measurements are unambiguous within their own configurations.**
Under the current default every one of the five seeds is worse with the fully
connected graph — per-seed costs `[2.125, 2.822, 2.997, 3.636, 2.342]` — and all
four measures agree (runs
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z`
and
`fd001_complete_row_cap_5seed_varsplit_cfgf21547_f386291-dirty_20260914T150527Z`):

| Measure | physics | fully connected | difference | paired t | p |
|---|---|---|---|---|---|
| RMSE | 16.496 ± 0.415 | 19.281 ± 0.432 | +2.784 | +10.51 | 0.00046 |
| NASA score | 539.8 ± 44.7 | 796.1 ± 78.8 | +256.3 | +5.46 | 0.0055 |
| R-squared | 0.830 ± 0.008 | 0.768 ± 0.010 | −0.062 | −10.48 | 0.00047 |
| ASUE | 11.651 ± 0.995 | 13.759 ± 0.489 | +2.108 | +5.66 | 0.0048 |

With nothing held back the per-seed costs were
`[2.569, 2.387, 2.244, 2.309, 2.470]`, and again all four measures agreed:

| Measure | physics | fully connected | difference | paired t | p |
|---|---|---|---|---|---|
| RMSE | 16.818 ± 0.157 | 19.214 ± 0.056 | +2.396 | +41.65 | 0.000002 |
| NASA score | 609.8 ± 25.5 | 781.7 ± 9.0 | +171.8 | +19.79 | 0.000038 |
| R-squared | 0.824 ± 0.003 | 0.770 ± 0.001 | −0.054 | −45.63 | 0.000001 |
| ASUE | 12.333 ± 0.598 | 13.941 ± 0.148 | +1.608 | +6.70 | 0.002576 |

**What cannot be concluded from the reversal.** Two things changed between the
zero result and the nothing-held-back one — the input scaling (min-max to
z-score) and the split (20 % held back to nothing held back) — so those runs do
not isolate which caused it. The current-default row narrows this: it holds 20 %
back like the min-max row, so the split is ruled out, but it also rescales the
graph by rows where the min-max row used the symmetric form, so two differences
remain. No stored run varies only one. Saying "the z-score revealed the
prior's value" would be a guess, and this file does not make those.

**How this compares with the paper.** Wen et al. report the same ablation
costing **0.56** cycles on FD001 (12.34 against 11.78) and call it the smallest
of the three components they test. This project has now measured 0.044, and
2.396, for the same ablation on the same sub-dataset. The paper's figure sits
between the two. Every one of the three is a defensible measurement of its own
configuration, and none of them reproduces the paper's.

**What this means for the thesis, and it is the useful part.** An ablation is
usually reported as a property of an architecture: *the physics prior is worth
X*. These measurements say it is not that. It is a property of the architecture
**and** the preprocessing **and** the split, and the paper specifies only the
first. A reader cannot tell from the paper whether its 0.56 would survive a
different input scaling, because the paper never says what its input scaling
was (A-03).

This is a stronger result than either individual number. It is direct evidence
that the paper's silence on preprocessing is not a cosmetic omission: it makes
the paper's own ablation table unreproducible in principle, not merely in
practice. It is the same conclusion F-10 reaches from the accuracy side and
A-20 reaches from the explanation side, arrived at independently.

**Reproduce with:**

```bash
python scripts/train.py --seeds 5                    # physics
python scripts/train.py --graph complete --seeds 5   # fully connected
```

The stored runs are
`fd001_physics_row_cap_5seed_varsplit_cfg722cd2_9548104_20260831T121451Z` and
`fd001_complete_row_cap_5seed_varsplit_cfgd20b12_9548104-dirty_20260831T123001Z`.
Their saved configurations differ in exactly one key, `graph`. The `-dirty`
on the second is a false positive: three untracked files unrelated to this
project's code appeared in the working tree between the two runs, and
`git status --porcelain --untracked-files=no` was empty throughout. The
physics arm reproduced the earlier `6a7af8c` run bit-identically on all five
seeds, which is direct evidence the code was unchanged.

---

**Superseded accounts, kept for the record.**

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

**How much it matters — re-measured 2026-08-31 under the default of that date,
and the answer reverses.** The same trained model, the same 20 test windows, three
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

**Correction, 2026-09-14: the table above depends on how the network was
trained, and did not say so.** It names the scaling, the graph rescaling and
that nothing was held back, but not the learning-rate schedule. On 2026-08-31
that was the default of the day: the rate shrank by the same factor after every
pass over the training data. On 2026-09-01 the default became a single drop at
the halfway pass (assumption A-17). `scripts/verify_claims.py` reads today's
defaults, so from then on it trained a different network from the one this
table describes, and on 2026-09-14 its check of this table failed.

How the cause was established: the same network, seed 42, was trained under
each schedule with everything else as above, at commit `8b487ab-dirty` (the
uncommitted changes were to documents, tests, the transcribed tables of
`evaluation/published.py` and the verification script, none of them on the
path from the data to a trained network). Under the old schedule every figure
in the table above came back exactly. Under the current one:

| Baseline | Four most important sensors | mean \|φ\| | Spearman with the mean window |
|---|---|---|---|
| mean window (used here) | Nc, T50, W31, T24 | 3.340 | — |
| median window | T50, T24, W31, Nc | 2.952 | **+0.956** |
| all zeros | T50, Nc, W31, T24 | 3.192 | **+0.965** |

**What survives.** The same four sensors lead under every stand-in and both
schedules, and the zero baseline still agrees closely with the mean window —
nowhere near the −0.16 that min-max gave below.

**What does not.** Under the mean window, `Nc` and `T50` swap first and second
place. And the agreement between zeros and the mean window falls from 0.996 to
0.965, which shows the explanation of the 0.004 in the paragraph above is
incomplete. All three stand-ins are computed from the training data before any
network exists, so they are identical under both schedules, and so is every
difference between them. Yet the gap in agreement grew from 0.004 to 0.035 with
only the training changed. How far a given difference in the stand-in moves the
ranking is a property of the trained network, not of the data alone.

Both tables come from one network, seed 42. How much a different seed moves them
is not measured.

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
scaling** produced it — and, since the correction of 2026-09-14 above, **which
training**: the learning-rate schedule and whether engines were held back.

---

### F-05 — What SHAP says about P15, and what it does not

Finding F-01 recorded that `P15` takes only two distinct values on FD001, and
predicted that a working attribution method would give it close to zero
importance. Exact Shapley values, computed over the full life of three held-out
engines, show that the prediction is half right in a way that turns out to be
more useful than if it had simply been correct.

**Re-measured 2026-08-31** under the default of that date (z-score, row, nothing
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

**Correction, 2026-09-14: the table above depends on how the network was
trained, and did not say so.** Exactly as in A-20: it was measured when the
learning rate shrank by the same factor after every pass, and the default became
a single drop at the halfway pass the next day (assumption A-17). The check in
`scripts/verify_claims.py` reads today's defaults, and on 2026-09-14 it
disagreed with the largest contributions for engines 5 and 82. Under today's
other default, a fifth of the engines held back, it stopped altogether, because
seed 42 holds back all three of these engines.

The same network was trained under each schedule with nothing held back, at
commit `8b487ab-dirty`, as A-20's correction describes. Under the old schedule
every figure in the table above came back exactly. Under the current one:

| Engine | Windows | `P15` rank | mean \|φ\| for `P15` | median \|φ\| | max \|φ\| | top sensor |
|---|---|---|---|---|---|---|
| 5  | 240 | 14 of 14 | 0.750 | 0.363 | 6.814 | 11.146 (`T24`) |
| 25 | 201 | 14 of 14 | 0.464 | 0.445 | 1.675 | 9.383 (`Nc`) |
| 82 | 185 | 14 of 14 | 0.731 | 0.420 | 6.287 | 14.799 (`Nc`) |

The rank survives this change too. The sizes do not all move alike. `P15`'s
largest contribution fell by 27.9 % on engine 5 and 34.0 % on engine 82, the
two engines whose reading crosses between its levels, but only 5.3 % on engine
25, where it never does. Its average contribution moved by between −2.1 % and
+8.0 %, and the top sensor's by at most 5.1 %.

Every argument that follows still holds under the current schedule, with its
numbers changed:

- the medians still do not single out engine 25 as the quiet one: 0.445 against
  0.363 and 0.420;
- the maxima still do, 6.814 and 6.287 against 1.675;
- nothing on engine 25 reaches 1.8 cycles.

Where the text below quotes 9.449, 9.526 or "9.5 cycles", those are figures
under the old schedule. As in A-20, this is one network, seed 42, and how much a
different seed moves it is not measured.

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
   reading crosses between its two levels (6.8 under the learning-rate
   schedule in force since 2026-09-01; see the correction above).
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
   conclusion A-20 reaches from the baseline side. The correction of
   2026-09-14 adds the training to the scaling: changing only the learning-rate
   schedule moved `P15`'s largest contribution by up to a third, and again its
   rank did not move.
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
**16.818 ± 0.157**, a shortfall of 1.89 rather than 5.42. The reasoning in this
section is unchanged and still applies to the architecture, which A-22 does not
touch; only the preprocessing and the adjacency formula moved.

#### The architecture-faithful comparison is 20.35 against 14.93

Run B is the paper's printed architecture: input `[batch, 14, 1]`,
`W(1) in R^(1 x 32)`, the same layer widths, and **61,729 parameters — what Table 4's
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
Table 4's shapes imply 61,729. Run A had 62,657. That single number would have caught
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
| z-score, **nothing held back**, **step** schedule — the default until 2026-09-02 | **16.818 ± 0.157** | 17.071 ± 0.325 | **row by 0.253** | −2.478 | 0.068 |
| the same under the superseded **exponential** schedule | 16.694 ± 0.331 | 17.045 ± 0.201 | row by 0.352 | −2.830 | 0.047 |
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
comment has been corrected. (The file itself was deleted on 2026-09-14, after
it had become identical to `config/fd001.yaml`.)

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
| A-06 validation split | describes none | contradicts Table 6's 200 epochs → removed, then **reinstated 2026-09-02**; worth 0.32 cycles, p = 0.234 |
| A-18 early stopping | Table 6: 200 epochs | contradicts → removed, then **live again 2026-09-02** with A-06 |
| A-04 smoothing | Eq. 18 defines it | stated → kept, and removing it is worse anyway |
| A-01 label ceiling | Eq. 19; value inferred from Fig. 7 | evidence-backed → kept at 125 |
| A-17 LR schedule | Table 6: 1e-3 → 1e-4 | mechanism silent → **one x0.1 step**, the only form any C-MAPSS paper uses |

Result: **16.818 ± 0.157** on FD001 against the paper's 14.93, from
20.353 ± 0.752 under the previous defaults. **That figure was measured with
nothing held back.** Since the A-06 reversal of 2026-09-02 the default holds
out 20 % of the engines and gives **16.496 ± 0.415**, a difference the seeds
cannot distinguish from zero (p = 0.234; the table under A-06).

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
   a validation split exists to prevent. `config/fd001_validated.yaml` was
   kept for that reason while the default held nothing back; since the split
   returned to `config/fd001.yaml` the two files were identical, and the
   separate one was deleted on 2026-09-14. **Any future comparison between
   configurations must hold engines back**, which the default now does — and
   must be decided on the held-back error, not on the test error (the
   independent audit of 2026-09-08 reports that this has not always been so).
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
normalisation formula, so this reproduction spans **16.82 to 20.35 cycles**
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

### F-12 — No other paper in the surrounding literature gives a graph node one number

**Where:** twelve papers alongside the source in
`D:\Main\University\Thesis\Papers\Predictive Maintanace Papers\GNN\GNN Predictive Maintenance\RUL\MAPSS\`

A-07 had to invent a stand-in for the node feature, because the paper never
says what the "ESTGNN w/o LSTM" ablation feeds its graph. This entry asks a
narrower question that *can* be answered from sources: **what does everyone
else put on a node?**

Every PDF in that folder was read — full text extracted from all thirteen, and
the deciding tables and figures rendered and read as images.

**The corpus.** Ten method papers report on C-MAPSS FD001–FD004; one is a 2025
survey; one (`MAGF-SPA`) uses only N-CMAPSS and is excluded from every count
below; the thirteenth is the source paper itself.

**What each one puts on a node:**

| Paper | Node is | What the node carries into the graph layers | Page |
|---|---|---|---|
| GLSTGN (ESWA 2025) | sensor | output of a gated temporal convolution | 1, 3 |
| IMTSO-GCN (QRE 2025) | sensor | raw values over a sliding window; size tuned, 20 in the reported run | 10, 11 |
| DVGTformer (MSSP 2024) | sensor | the sensor's whole window, `Xw in R^(Tw x N)` | 6 |
| Zheng et al. (Adv. Eng. Inf. 2026) | sensor | per-sensor LSTM output, `R^(1 x F)`, **F = 30** | 4, 7 |
| HAGCN (RESS 2021) | sensor | the raw 30-cycle window | 5, 6 |
| HierCorrPool (IEEE) | sensor | 1-D CNN output over sub-windows | 3 |
| ADFG (IEEE TIM) | sensor | multiscale-CNN output, same length as the window | 1, 8 |
| GCN + Graph Transformer (2025) | sensor x time step | 1D-CNN features plus positional encoding | 3 |
| THGNN (IEEE TNNLS 2025) | sensor, continuous time | per-event features | 2 |
| AGCTE (Aerospace 2024) | sensor | the raw 1-D series itself; the GCN runs *first* | 3, 4 |
| **ESTGNN (the source)** | sensor | **one number** | Table 4, p. 7 |

Two families, and no third. Either the raw window goes straight in, or a
temporal encoder runs first and its **output vector** becomes the node feature.
The smallest node feature found anywhere in the corpus is 30 numbers.

**Nobody uses the last reading, and nobody uses the mean.** All thirteen texts
were searched for every phrasing of both — "last time step", "final time step",
"mean over", "average over time", "single value", "one value per", "current
reading". Twelve papers returned nothing. The two hits in HierCorrPool are a
sentence about averaging *inference times* and a reference title. **Both of the
options A-07 was choosing between are absent from the literature.**

**The closest structural analogue is HAGCN**, and it is worth quoting because
it is the same benchmark at the same window length. Its Table 4, read off page
6, opens:

```
1   Input     14 x 30 x 256
2   BiLSTM    14 x 60 x 256
3   HGRL_1    14 x 64 x 256
```

and its page 6 text states the input is "number of nodes x feature length x
batch size". So: fourteen sensor nodes, each carrying **thirty** numbers, and
its recurrent stage **widens** them to sixty before any graph layer runs.
ESTGNN's Table 4 requires the mirror image — a recurrent stage that narrows
thirty cycles to **one**. That is the unstated output projection A-11 declines
to invent, and this is direct evidence that it is an unusual thing to ask of an
LSTM rather than a routine one.

**Three independent sources choose a different fourteen, and the one sensor
they swap is P15.** Every C-MAPSS study reduces the 21 sensors to 14 by
dropping seven that do not vary. Three of the sources here name their seven,
and all three agree with each other and disagree with the source paper:

| Source | Page | The fourteen kept |
|---|---|---|
| HAGCN (RESS 2021) | 5 | 2, 3, 4, 7, 8, 9, 11, 12, 13, **14**, 15, 17, 20, 21 |
| AGCTE (Aerospace 2024) | 6 | 2, 3, 4, 7, 8, 9, 11, 12, 13, **14**, 15, 17, 20, 21 |
| Survey (MSSP 2025) | 19 | the same, stated as removing 1, 5, **6**, 10, 16, 18, 19 |
| **ESTGNN (the source)**, Table 3 | 5 | 2, 3, 4, **6**, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21 |

Thirteen sensors are shared by all four. The disagreement is a single swap:
the field keeps `NRc` (column 14) and discards `P15` (column 6) as
constant-valued; the source paper does the exact opposite.

The survey's wording is the pointed part — it removes column 6 among "sensors
with constant values". **F-01** measured `P15` in FD001 and found it takes two
distinct values across the whole sub-dataset — 21.60 and 21.61 in 20 631 rows
— and **F-05** found it ranked last of the fourteen on every engine tested. Three independent groups reached by inspection the
conclusion this project reached by measurement: `P15` carries almost nothing.
The source paper nevertheless admits it to the graph and gives it four edges.

This also bears on **A-10**, which deduced `NRc` as the source's seventh
dropped sensor from its Table 3. That deduction stands — but it is now clear
that dropping `NRc` is the source's own departure from its field, not a
convention it inherited.

**How the field orders the computation.** The survey (MSSP 2025, page 13)
splits the literature into *spatial-first* — the graph runs on the raw data and
a temporal encoder follows — and *temporal-first*, and states:

> "Most of the existing works adopt the spatial-first approach."

The source paper is temporal-first. That is the minority arrangement, and it is
the only one in which removing the temporal stage leaves a hole where the node
feature used to be. In a spatial-first design there is nothing to remove: the
graph was always reading the window directly.

The survey also defines node features loosely (page 3) — they "could be the raw
signals themselves or features extracted through some preprocessing or feature
engineering methods" — and THGNN (page 2) argues *against* what the source
does, criticising methods that compress "MTS data from a time window into a
single 'flat' graph, often leading to the loss of critical temporal
information".

**What this does and does not settle.** It does not overturn A-07. The
reasoning there — keep the architecture Table 4 prints, delete only the named
component — still stands, and no paper here faced the same question, because
none of them removes a temporal stage and reports the result. What it does is
locate A-07 honestly: **its choice is outside the practice of its field**, the
alternative it rejected is also outside that practice, and the option the field
would reach for is the one this project removed on 2026-08-28 for re-introducing
a temporal encoder. A-07 should say so rather than leave the impression that
the last reading is a natural default. It is not; it is a consequence of
insisting on Table 4's printed shape.

**Method, so this can be checked.** Text was extracted from all thirteen PDFs
with `pypdfium2`; every quotation above carries the page it came from; the
tables and figures cited (HAGCN Table 4, the survey's Figs. 4 and 6, DVGTformer
Fig. 4, AGCTE Fig. 2) were rendered as page images and read, not inferred from
the text layer, because two-column extraction interleaves table rows.

---

## A-23 — An engine shorter than the window is left out, not padded

**Where:** `data/windowing/`, `_extract_one_unit`

**The gap.** Table 6 fixes the window at 30 time steps, and the paper reports
results on all four sub-datasets. Some test engines are shorter than 30 cycles,
so no window can be cut from them at all. The paper never says what happens to
those engines.

**What we chose.** Leave the engine out. It yields no window and no prediction,
and the run writes a warning naming the engine and its length. If *every*
engine in a table is too short, the code raises rather than returning an empty
result.

**Why.** Padding invents readings. Filling the missing rows by repeating the
engine's first reading puts sensor values into the record that the engine never
produced, and then asks a model to predict from them — which is what working
rule 7 forbids, and the reason this project contains no generated data
anywhere else either.

It is also the only published practice found. Dida, Cheriet and Belhadj,
*Remaining Useful Life Prediction Using Attention-LSTM Neural Network of
Aircraft Engines*, write on their page 6, working at the same window length on
the same dataset:

> "Engines in sub-datasets FD002 and FD004 that contained fewer than 30 samples
> were excluded from the study."

A search of 25 papers on this benchmark found no other treatment of short
engines at all, and none that pads. See `docs/LITERATURE_BASIS.md`.

**What this costs, and the cost is real.** The NASA score is a **sum** over
engines, not a mean. Leave an engine out and the score falls because there is
one term fewer, not because the model improved, so a score computed this way
cannot be set beside a published FD002 or FD004 figure. Root mean squared error
is an average and survives better, but its denominator changes too. Dida et al.
accepted that cost; a reproduction that wants to sit beside published numbers
for those two sub-datasets may not be able to. **Anyone extending this pipeline
to FD002 or FD004 inherits this and must say so beside any number they quote.**

**How much it matters — measured.** Counted from the real files, the number of
**test** engines with fewer than 30 cycles:

| Sub-dataset | shortest test history | test engines under 30 cycles | test engines |
|---|---|---|---|
| FD001 | 31 | **0** | 100 |
| FD002 | 21 | **6** | 259 |
| FD003 | 38 | **0** | 100 |
| FD004 | 19 | **11** | 248 |

So this choice **cannot touch any number this project currently reports**,
because FD001 is the only sub-dataset run and no FD001 engine is short enough
to be affected — that is finding F-03. It becomes live the moment FD002 or
FD004 is run, where 17 test engines between them cannot yield a window.

**What this entry said until 2026-09-01, and why it changed.** It said the
opposite: keep the engine and repeat its first reading backwards. The argument
was that dropping an engine changes the size of the test set and breaks
comparability with the published tables — which is true, and is recorded above
rather than dropped. What that argument missed is that the alternative breaks a
rule this project treats as absolute: padding is generated data. The audit of
2026-08-31 found no published support for padding and one clear precedent for
exclusion, and the entry was rewritten to follow the evidence. Both the choice
and the cost it carries are now stated, which the earlier entry did not do for
the option it rejected.

---

## A-24 — The attributions are measured against the prediction at the average window, not the average label

**Where:** `explain/shap.py`, `shap_values` and `efficiency_error`

**The gap.** Equation 14 reads

```
g(z) = phi_0 + sum_i phi_i
```

and defines `phi_0` as "the average value of target variable across all
samples" — the mean of the **labels**. The usual definition in the SHAP
literature is the mean of the model's **predictions**, and neither is the same
as the model's prediction at an average input. The paper prints one of the
three and uses none of them explicitly.

**What we chose.** `phi_0 = f(baseline)`, the model's prediction on the average
training window. This is forced by A-20: a sensor left out of a group is
replaced by its average training reading, so the group containing no sensors at
all *is* the average window, and its prediction is what the fourteen
contributions are measured from. Choosing anything else for `phi_0` would break
the adding-up property, because the marginal contributions are already defined
relative to that substitution.

**Why it cannot simply follow the paper.** Equation 14 with the mean label
requires `sum_i phi_i = f(x) - mean(y)`, and no Shapley calculation built on a
substitution baseline satisfies that except by coincidence. The paper's own
Table 12 does not satisfy it either — that is F-08, reached from the same
equation by a different route.

**How much it matters — measured 2026-08-31**, on a model trained at seed 42
under `config/fd001.yaml` (test RMSE 16.68, in line with the five-seed
16.694 ± 0.331 of the exponential schedule then in force — see A-17), over the
17 731 training windows:

| Candidate for `phi_0` | value, in cycles |
|---|---|
| `f(average training window)` — used here, forced by A-20 | **91.357** |
| mean prediction `E[f(x)]` — the usual SHAP definition | 80.223 |
| mean label `E[y]` — the paper's Equation 14 wording | 80.638 |

The choice made here sits **10.719 cycles** from the paper's wording. That is
8.6 % of the 125-cycle label range and about two thirds of the model's whole
RMSE, so it is not a rounding matter.

Two things are worth reading off that table. The paper's `E[y]` and the usual
`E[f(x)]` agree to 0.4 cycles, because the trained model is nearly unbiased on
average; the outlier is the third row, and the reason is that the network is
not linear, so the prediction at the average input is not the average
prediction. And because every attribution is measured from `phi_0`, the whole
table this project produces is offset from the paper's definition by that
constant — **rankings by magnitude are unaffected, individual values are not.**

Reproduce with the script recorded in this entry's commit message.

---

## A-25 — The scaling is fitted after the split, which reverses the order the paper's Fig. 1 prints

**Where:** `data/pipeline/`, steps 6 and 7

**The gap.** Fig. 1 prints the preprocessing chain as

> feature selection → data smoothing → piece-wise RUL → **data normalisation**
> → **train and test datasets**

with normalisation *before* the train and test datasets are formed. It never
says which engines the scaling statistics are fitted on. Those are two
different silences and both matter: the order in the diagram, and the fitting
set the diagram does not name.

**What we chose.** The opposite order. Engines are divided first — step 6, and
before any statistic is computed from the data — and the mean and standard
deviation are then fitted on the **training engines only** and applied unchanged
to the held-back and test engines.

**Why.** A statistic computed over engines that will later be scored is
information from those engines reaching the model, which is the ordinary
definition of leakage. It flatters the reported figures and cannot be detected
by looking at them. Two tests hold the line —
`test_normalization.py::test_test_data_never_influences_the_scaling` and
`test_pipeline.py::test_validation_engines_never_influence_the_scaling` — and
working rule 1 of `README.md` section 11 states it.

**What this is not.** It is not a claim that the paper leaks. C-MAPSS ships its
training and test engines as separate files, so an implementation reading that
diagram could still fit on the training file alone and match both the diagram
and good practice. The diagram is simply not specific enough to tell, and this
entry exists so that a reader comparing the two orders is not left thinking the
difference was overlooked.

**How much it matters.** Unmeasured, and deliberately so: measuring it would
mean building the leaking variant and reporting a number from it, which is not
a number this project wants in its history. The direction is not in doubt —
fitting on the test engines can only improve the reported figures — so the
comparison would tell us the size of an advantage this project declines to
take. A-08 covers the *other* ordering question in Fig. 1, between smoothing
and scaling, and is withdrawn because there the paper's order is the one
followed here.

---

### F-13 — Fig. 1 lists node features the paper's own feature selection removes

**Where in the paper.** Fig. 1, the "Node features" list in the
physics-based-graph panel, read at scale 11.

The list reads, in order:

> T20 (°R), T24 (°R), T30 (°R), T50 (°R), P20 (psia), P15 (psia), P30 (psia),
> Nf (rpm), Nc (rpm), …

Two problems, and they compound.

**The names T20 and P20 appear nowhere else in the paper.** Table 2 lists all
21 C-MAPSS sensors and calls the corresponding ones **T2** ("Total temperature
at fan inlet") and **P2** ("Pressure at fan inlet"), which are the names Saxena
et al. (2008) use. Nothing in the paper defines T20 or P20.

**Those two sensors are ones the paper removes.** Section 3.2.2 states that
seven sensors are removed by feature selection and names six of them: "T2, P2,
epr, farB, Nf_dmd and PCNR_dmd". So T2 and P2 — under whichever spelling — are
excluded, and they are correspondingly absent from Table 3's fourteen rows,
which begin at T24, and from Fig. 5's correlation heatmap, which shows only the
kept sensors.

**The figure contradicts itself.** The graph drawn immediately beside that list,
inside the same panel, has fourteen nodes labelled with the Table 3 set — P30,
P15, T50, T30, T24, Nf, Nc, Ps30 and the rest. So Fig. 1's caption text and
Fig. 1's own drawing disagree about what enters the graph.

**Why it is recorded.** The list is the only place in the paper that enumerates
what a node carries, and it is wrong in a way that would mislead a
re-implementer into building a 16-node graph. This project follows Table 3 and
Section 3.2.2, which agree with each other and with the data (see A-10). Like
F-07 and the duplicated Table 6 caption, this is evidence that the figures were
assembled by hand and not generated from the code that produced the results.

---

### F-14 — The text under Equations 9 and 10 misnumbers the weight matrices

**Where in the paper.** Section 2.3.2, the sentence immediately following
Equation 10, read at scale 11.

Equations 8, 9 and 10 print, in order, `W(1)`, `W(2)` and `W(3)` — the first,
second and third graph convolution layers. The sentence beneath them reads:

> where **W(1)** and **W(2)** are the second and third learnable weight
> matrixes of GNN, respectively.

The superscripts are off by one. Equation 8 has already used `W(1)` for the
*first* layer, so calling `W(1)` "the second" contradicts the equation three
lines above it. The sentence should name `W(2)` and `W(3)`, which are the
matrices Equations 9 and 10 actually contain.

**Consequence: none for the architecture.** Table 4 prints the shape of every
matrix — `W(1) in R^(1x32)`, `W(2) in R^(32x64)`, `W(3) in R^(64x32)` — and
those are unambiguous, mutually consistent, and what this project implements.
Nothing here changes a layer.

**Why it is recorded anyway.** F-07's argument that the paper's tables and
figures were assembled by hand rests on accumulating exactly this kind of
evidence, and an audit that found it and left it out would be selecting its
evidence. Counted so far: the duplicated Table 6 caption, Fig. 1's node feature
list (F-13), this sentence, the sign contradiction between Equation 20 and its
own text (A-15), the R-squared denominator (A-16), and the Conclusions'
"highest R²" (F-11).

---

### F-15 — Fig. 9 and Table 12 disagree about which sensors matter most

Fig. 9 and Table 12 show different sensor rankings. The paper does not identify
which engines or windows are represented by Fig. 9, so the two results cannot
be treated as a comparison on the same examples.

**What Fig. 9 says.** Its caption reads "Red nodes represent the four most
critical sensors, blue nodes show remaining sensors with importance-based color
gradation". The four in the red box, read at scale 11, are

> **Nc, P30, NRf, T24**

and the beeswarm beside them is ordered top-to-bottom by importance, with
`Ps30` first in the blue "remaining" box, immediately below the red four.

**What Table 12 says.** It gives SHAP importance for all 14 sensors at 15 time
steps. Ranking each column by value:

| | in the top four, of 15 columns |
|---|---|
| T50 | **15** |
| Ps30 | **15** |
| Phi | 11 |
| Nc | 10 |
| P30 | 5 |
| BPR | 4 |
| T24, T30, P15, Nf, **NRf**, htBleed, W31, W32 | **0** |

**The difference in the printed rankings.** Two of Fig. 9's four most critical sensors — **NRf and
T24** — are in Table 12's top four in **none** of its fifteen columns. Both are
outside that leading group; T24 ranks from sixth to tenth across the printed
columns. T24 falls to 0.76 and NRf to 0.89 by G191.
Meanwhile **T50 and Ps30 are in the top four in every single column**, and
Fig. 9 places both in the "remaining sensors" box.

The prose agrees with Table 12, not with Fig. 9: Section 4.4 says the model
"predominantly relies on sensors T50, Nc, Ps30, and phi" early, and names T50,
P30, Ps30, phi in the middle and Nc, Ps30, BPR, T50 late. NRf and T24 are never
named as important anywhere in the text.

**What cannot be concluded.** That either is wrong. Fig. 9 may show a different
engine, a different sub-dataset, or a ranking over a different collection of
windows from the representative unit Table 12 describes. The paper does not
give that information. Even for one engine, a ranking averaged over its life
can differ from the ranking at any of the selected windows. The cause of the
difference remains unresolved.

**Correction, 2026-09-28.** This entry previously said that T24 and NRf were
both in the bottom half of every Table 12 column. Re-ranking the printed
values shows that T24 sometimes ranks sixth or seventh, so that statement was
wrong. It also claimed that the two results could not describe the same
quantity for the same engine; that conclusion was too strong without knowing
which windows Fig. 9 includes. The corrected comparison above was checked
against PDF pages 13 and 14 rendered at scale 8. The transcription gives these
T24 ranks, with rank 1 assigned to the largest value and ties sharing a rank:

```text
G1 G2 G3 G4 G5 G78 G79 G80 G81 G82 G187 G188 G189 G190 G191
 7  6  7  7  8   9   9  10   9   9    8    8    6    6    9
```

Recompute them without training a model:

```bash
python -c "from thesis_gnn_rul_xai.evaluation.published import SHAP_TABLE_12 as t; print([1 + sum(v[i] > t['T24'][i] for v in t.values()) for i in range(len(t['T24']))])"
```

**Consequence for this project.** F-05 compares this project's sensor ranking
with the paper's. It must say *which* of the paper's two rankings it is
comparing against, because they differ.

F-05 happens to concern `P15`, and there the two agree: Fig. 9 places it last
of fourteen, and in Table 12 it is in the **bottom three of fourteen in every
one of the fifteen columns** — second-lowest in nine of them, lowest in three,
third-lowest in three. So that particular comparison survives either way. **Any
future comparison about a highly ranked sensor does not**, and must name which
of the two it is made against.

---

---

### F-16 — What the paper's LSTM is worth depends on a training procedure the paper does not describe

**Corrected 2026-09-14.** This entry was first written on 2026-09-09 under the
title *"Building the paper's LSTM reproduces its stated benefit almost exactly,
and leaves a constant gap that has nothing to do with it"*. Both halves of that
title were true only under this project's training procedure, and two further
runs showed that the paper's own procedure gives a different answer. What the
entry first said is kept below, under its own heading, because the way it was
wrong is instructive.

**How it was found.** The original entry noticed that early stopping fired at
epochs 10, 13, 9, 14 and 21 of a nominal 200, and suspected it was cutting the
recurrent stage short. Two arms were run to test that. Neither result was the
one expected.

| Arm | Held back | Training | Reported model | RMSE (5 seeds) |
|---|---|---|---|---|
| A | 20 % | stops early | best-validation checkpoint | 13.307 ± 0.532 |
| C | 20 % | full 200 epochs | best-validation checkpoint | 13.307 ± 0.532 |
| D | nothing | full 200 epochs | **epoch 200 — the paper's procedure** | **16.052 ± 0.975** |

**Arm C is bit-identical to arm A**, seed for seed, with the same best epochs.
Early stopping was not cutting anything short: the recurrent stage's
validation error genuinely bottoms out between epochs 9 and 21 and never
improves over the remaining 180. It is also a clean demonstration of the trap
A-18 records — `--no-early-stopping` alone does not change the reported model
while a validation set exists, because the run still ends by restoring the
best checkpoint.

**Arm D is the paper's procedure**, and it is where the correction comes from.
Table 6 states 200 epochs; the paper describes no held-out set and no early
stopping (A-06, A-18), so its reported model is the weights at the end of
training. Set beside the stand-in under the same procedure, paired by seed:

| Training procedure | `last_cycle` | `lstm` | LSTM worth | paired p |
|---|---|---|---|---|
| **the paper's**: nothing held back, 200 epochs, final weights | 16.818 ± 0.157 | 16.052 ± 0.975 | **+0.766** | **0.117** |
| this project's: 20 % held back, best checkpoint | 16.496 ± 0.415 | 13.307 ± 0.532 | +3.189 | 0.0009 |
| *the paper's Table 11, FD001* | *14.93* | *11.78* | *+3.150* | — |

**Under the paper's own procedure the stage is worth 0.77 cycles, and that
does not survive even the seed-to-seed spread.** The 3.19 that agreed so
closely with Table 11's 3.15 appears only when the best checkpoint is
restored — a step the paper never mentions.

**Why the two procedures disagree so much for this stage.** Restoring the best
checkpoint is worth very different amounts to the two models:

| | best checkpoint worth | paired p |
|---|---|---|
| `last_cycle`, 61,729 parameters | +0.321 | 0.234 |
| `lstm`, 124,482 parameters | **+2.745** | **0.004** |

The recurrent model is twice the size, and over 200 epochs it gets much worse
after its best epoch; the smaller model barely does. So checkpoint selection is
not a neutral bookkeeping choice here. It interacts with the temporal stage,
and it accounts for most of what the stage appeared to be worth. The spread
says the same thing from another side: at epoch 200 the recurrent model's
seed-to-seed standard deviation is 0.975, against 0.157 for the stand-in.

That the recurrent model overfits is **inferred**, from where its best epochs
fall and from how much worse its final weights are — not observed. The
per-epoch training and validation curves that would show it directly are built
during training and never saved, a gap the audit of 2026-09-08 raises as D-10.

**What this does and does not establish.**

- It does **not** show the paper's stage is worthless. It shows that under the
  one procedure the paper describes, this implementation of it — with the
  hidden size A-26 had to choose — cannot be told apart from the stand-in
  across five seeds.
- It does **not** show the paper selected checkpoints without saying so. That
  would explain the agreement with Table 11, and it is the obvious suspicion,
  but the paper does not say, so it is speculation and must be written that
  way wherever it appears.
- **Every p-value above answers only "is this an artefact of initialisation?"**
  The independent audit of 2026-09-08 (`docs/AUDIT_2026-09-08_scientific_validity.md`,
  D-8) makes the point that the five seeds share one test set of 100 engines,
  so seed spread is not uncertainty over engines, and a paired test across
  seeds says nothing about engines the model has not seen. The per-engine
  uncertainty cannot be computed from these runs at all: each run directory
  holds `preprocessing_config.yaml`, `results.csv` and `summary.json`, and no
  per-engine prediction.
- **The "w/o LSTM" row is not a clean comparison for `last_cycle`**, for a
  reason the same audit gives in D-6: the paper's ablation removes a recurrent
  stage from a model that still reads a 30-cycle window somewhere, while
  `last_cycle` discards the window as well.

**Provenance.** Arms A and C, and the `last_cycle` run at 20 %, are at commit
`f82c76c-dirty`, whose uncommitted changes were the component architecture
later committed as `3a62e7b`. Arm D is at `c703693-dirty`. The `last_cycle`
run at nothing-held-back is at `ae59bc1-dirty`, before that refactor; the
refactor was gated on reproducing the previous five-seed run bit-identically
and did, so the training path is the same, but the pairing does cross it and a
re-run at one commit would remove the question. The run directories:

- `reports/fd001_physics_row_cap_5seed_varsplit_lstm32_cfgba82e3_f82c76c-dirty_20260909T125552Z` (A)
- `reports/fd001_physics_row_cap_5seed_varsplit_lstm32_noearlystop_cfgba82e3_f82c76c-dirty_20260909T154345Z` (C)
- `reports/fd001_physics_row_cap_5seed_varsplit_lstm32_noearlystop_cfg00be26_c703693-dirty_20260909T193330Z` (D)
- `reports/fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f82c76c-dirty_20260909T123135Z` (`last_cycle`, 20 %)
- `reports/fd001_physics_row_cap_5seed_varsplit_cfg00be26_ae59bc1-dirty_20260908T224841Z` (`last_cycle`, nothing held back)

---

#### What this entry said on 2026-09-09, before arms C and D

**Superseded 2026-09-14.** The analysis below is correct as far as it goes —
every number in it is right for the procedure it was measured under — but it
did not know the answer depended on that procedure, and so it drew two
conclusions the paper's own procedure does not support: that the stage's
contribution "reproduces to within 0.04 cycles", and that "the remaining gap is
a constant". Under the paper's procedure the contribution is 0.77, and the gap
to Table 11 is 1.89 without the stage and 4.27 with it.

`models/temporal/LSTMStage` builds the stage of the paper's Section 2.2:
fourteen dedicated recurrent networks, one per sensor, which is what the paper
states rather than anything chosen here (A-26 covers the size, which it does
not state). That made the paper's own ablation runnable for the first time in
this project.

**What was run.** Five seeds each, commit `f82c76c-dirty`, FD001, physics
graph, row normalisation, z-score scaling, capped test labels, step schedule,
20 % of engines held back. The two arms differ in the temporal stage and in
nothing else.

| Temporal stage | RMSE | Score | R² | ASUE |
|---|---|---|---|---|
| `last_cycle` — the A-07 stand-in | 16.496 ± 0.415 | 539.8 | 0.8305 | 11.651 |
| `lstm`, hidden 32 — the paper's stage | **13.307 ± 0.532** | **330.1** | **0.8896** | **8.938** |

Paired across seeds, the stage is worth **3.189 cycles**, t = −8.978,
two-sided p = 0.0009. Every one of the five seeds improves. Unlike A-06's
split, this difference is far larger than the seed-to-seed spread.

**Set beside the paper's own ablation.** Table 11 (page 12) prints the FD001
column of its ablation study, including the delta:

| | Paper, Table 11 | Here, 5 seeds | Difference |
|---|---|---|---|
| ESTGNN (Full) | 11.78 | 13.307 ± 0.532 | +1.53 |
| ESTGNN w/o LSTM | 14.93 | 16.496 ± 0.415 | +1.57 |
| **What the LSTM is worth** | **3.15** | **3.189** | **0.04** |

The paper's 3.15 is not computed here: Table 11 prints it, in its "LSTM" delta
row, alongside "Δ ( %) v.s. Full (+27)".

**Two things follow, and they point in opposite directions.**

**The stage's contribution reproduces to within 0.04 cycles.** That is a
stronger corroboration than the paper's silence would lead you to expect: A-26
had to choose the hidden size, the depth and the projection, and the *value* of
the stage came out the same anyway. It is the closest this project has come to
reproducing a number the paper reports, and it is worth stating that it is a
reproduction of a **difference**, not of a level.

**The remaining gap is a constant, and the LSTM is not in it.** 1.53 cycles
with the stage, 1.57 without: unchanged across a doubling of the model, from
61,729 parameters to 124,482. Whatever produces it is therefore not the
temporal stage, and no amount of work on that stage will close it. The
candidates are the settings the paper does not specify at all, and F-10 already
measures the largest of them: the input scaling, worth 3.83 cycles between the
two readings tried. A-21, A-01 and A-17 are the others.

**How this comparison may and may not be described.** The paper reports single
runs with no error bars, so its 11.78 and 14.93 have no spread to compare
against; the ± here is this project's and covers seed, split and batch order.
The two arms above differ from the paper in every way recorded in A-01 through
A-26 except the one being tested. So this is evidence that **the stage does
what the paper says it does**, not evidence that this code reproduces the
paper.

**Where the runs are.**
`reports/fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f82c76c-dirty_20260909T123135Z`
and
`reports/fd001_physics_row_cap_5seed_varsplit_lstm32_cfgba82e3_f82c76c-dirty_20260909T125552Z`.

---

### F-17 — Fig. 10 is drawn from Table 12, except in two of its fifteen graphs

**What Fig. 10 shows.** Fifteen small graphs, numbered 1 to 5, 78 to 82 and 187
to 191. Each places the fourteen sensors round a circle, joins them with edges,
and colours them the way Section 4.4 describes: "the four most influential
sensors" red, "the remaining sensors" blue, each shade by importance. The text
says Fig. 10 "visualizes these importance patterns", meaning those of Table 12,
whose columns carry the same fifteen numbers.

**How it was checked, on 2026-09-14.**

- Page 14 rendered with `pypdfium2` at scale 8. The node circles were found as
  connected patches of coloured pixels, and a circle fitted through the node
  centres of each graph; the fitted radius is 317 or 318 pixels in all fifteen,
  so no graph was mis-located. Each node's colour is the median of the pixels
  within 30 pixels of its centre, leaving out the white label and the black
  outline. A node counts as red when its red channel exceeds its blue one.
- The order round the circle was read by eye from graph 1, and from the graph in
  Fig. 9, both at scale 8, and is the same in both: clockwise from the node just
  left of the top, P30, P15, T50, T30, T24, W32, W31, htBleed, BPR, NRf, phi,
  Ps30, Nc, Nf.
- Which cells of Table 12 are bold was read from the font of each cell in the
  PDF's text layer: bold cells are set in `CharisSIL-Bold`, the rest in
  `CharisSIL`.

**What agrees.**

- **Table 12 agrees with itself.** In all fifteen columns the four bold cells
  are the four largest values.
- **Thirteen graphs have exactly Table 12's four largest in red**, and their
  shades follow the values. Ordering the red nodes by how dark they are, and
  separately the blue ones, and comparing each order with Table 12's by
  Spearman's rank correlation — 1 when two orderings agree exactly — gives 1.00
  for both groups in eleven of the thirteen, and for the blue group 0.95 in
  graph 188 and 0.88 in graph 190.
- **The edges are Table 3's.** For every pair of sensors, the pixels along the
  straight line between their nodes were tested for a dark line. No graph shows
  an edge Table 3 lacks, and graph 4 shows all 41 of Table 3's. In the other
  fourteen, one to three edges went undetected, always among the same three
  pairs, each joining two sensors two places apart on the circle: Nf and P15,
  Nc and phi, W32 and htBleed. Such a line passes 317 × (1 − cos 25.7°) ≈ 31
  pixels from the node between its ends, inside that node's radius of about 40
  pixels, so the node is drawn over it.

**What does not agree.** Graphs 81 and 82, confirmed by eye on a crop at scale 8:

| Graph | Red in Fig. 10 | Four largest in Table 12, and its bold cells |
|---|---|---|
| 81 | P30, Ps30, **T30**, T50 | P30, **phi**, Ps30, T50 |
| 82 | **Nc**, Ps30, **T30**, T50 | **P30**, **phi**, Ps30, T50 |

In graph 81, Table 12 puts T30 fifth, at 2.46 against phi's 2.58, and the figure
draws phi in deep blue. In graph 82, T30 and Nc tie for fifth at 2.21, below
phi's 2.29 and P30's 2.34, and the figure draws both of those blue. The prose
sides with the table: it names "T50, P30, Ps30, and phi" for graphs 78 to 82.

**What cannot be concluded.** Which is right about those two moments. The paper
keeps no other record of them.

**Consequences for this project.**

1. Of the paper's two pictures of its explanation, **Fig. 10 follows Table 12
   and Fig. 9 does not** (F-15). A comparison with "the paper's ranking" can use
   Table 12 and Fig. 10 as one source, except at graphs 81 and 82.
2. The documentation site's playground redraws Fig. 10 from Table 12's numbers,
   above a stored explanation of one engine at the same graph numbers, and says
   beside it that the printed figure differs at graphs 81 and 82. Table 12, its
   graph numbers and the order round the circle are in
   `evaluation/published.py`; `tests/test_playground.py` checks the
   transcription against F-08's column sums and against the bold cells.

---

## A-26 — The recurrent stage is one layer of hidden size 32, projected to one number

**Where:** `models/temporal/LSTMStage`, `scripts/train.py --temporal-hidden`

**The gap.** The paper describes the recurrent stage's *arrangement* precisely
and its *size* not at all.

What it does state, in Section 2.2 and repeated four times, is that every
sensor gets its own network: *"Each sensor s ∈ {1, 2, …, n} is associated with
a **dedicated** LSTM network"*, with Equation 7 indexing the hidden state by
sensor, the channels processed *"independently"* through *"separate specialized
pathways"*, and the architectural strengths opening with *"Independent parallel
LSTM networks for each sensor"*. **That is a citation, not an assumption**, and
this project follows it: fourteen cells, not one shared.

What it never states is how big each one is. **Table 4 has no recurrent row at
all** — it begins at `Input | Node features | [batch size,14,1]`, so the stage
sits entirely outside the only architecture table the paper prints. Table 6
lists the learning rate, 200 epochs, batch size 32, thirty time steps, Adam and
MSE loss, and nothing else. The word "hidden" appears three times in the whole
paper: once in the related-work section, and twice inside the gate equations
meaning `h_{t-1}`, the previous hidden state.

So three quantities are missing: the **width** of the hidden state, the
**depth** in layers, and how a hidden state becomes the one number `W(1) ∈
ℝ(1×32)` consumes.

**What we chose.** One layer, hidden size 32, and a single linear projection
from the last cycle's hidden state to one number per sensor.

**Why.** The width matches the first graph layer's output width, which is the
only number of that kind the paper prints, and 32 is the smallest of the three
widths in Table 4. One layer is the minimum that is still an LSTM, and adding
depth would be inventing more than is needed to have the stage at all. Taking
the last cycle's hidden state is what Equation 7 describes — it defines
`h_t^(s)` at time `t`, the end of the window — and the projection exists
because Table 4 requires one number and an LSTM produces `hidden_size` of them.
The projection is the part with least support in the paper: it is required by
Table 4's shape, but the paper never prints it.

**How much it matters.** Large, and measured.

| | Parameters |
|---|---|
| The graph stage alone, Table 4 | 61,729 |
| One `nn.LSTM(input_size=1, hidden_size=32)` | 4,480 |
| Fourteen of them, plus the projection | 62,753 |
| **The whole model with the stage** | **124,482** |

The paper's own architecture is therefore **about twice the size** of what this
project trained before, and about **thirteen times slower per batch** (22.9 ms
against 1.8 ms for a batch of 32, so an epoch over the 14,459 training windows
goes from roughly 0.8 s to roughly 10.4 s).

Everything in the table is `sum(p.numel() ...)` on the real objects, not
arithmetic on paper.

**What this does not settle.** A different hidden size is a different model,
and this entry fixes one point in a space the paper leaves open. Any number
produced with the stage names the width in its run directory and in
`summary.json`, so two such numbers can be told apart. Whether 32 is a good
choice has not been measured against 16, 64 or two layers, and until it is, a
result from this stage says what this configuration does and not what the
paper's stage would do.

---

## A-27 — The data-driven graph joins the 41 most strongly correlated sensor pairs

**Where:** `graph/adjacency.py`, `build_correlation_adjacency(..., n_edges=41)`;
`data/pipeline/`; `graph: correlation` in the configuration, or
`scripts/train.py --graph correlation`

**The gap.** The paper never builds a graph from the data. It uses the Pearson
correlation coefficient in one place, to decide which sensors to keep at all
(Section 3.2.2, Equation 17: sensors that never change, or that correlate above
0.95 with another, are dropped). Its graph is built by hand (Section 2.3.1,
Table 3). Yet item 2 of its Conclusions says the hand-built matrix captures
sensor relationships "beyond data-driven correlations". Nothing in the paper
tests that claim against a graph built from correlations, and its "without
physics prior" row of Table 11 never says what replaced its graph (assumption
A-19). Checked on 2026-09-14 by searching the PDF's text for every occurrence
of "correlat": none of them describes building a graph from correlations or
comparing one with Table 3.

**What we chose.** A graph with exactly as many edges as Table 3 has, 41,
joining the 41 pairs of sensors whose readings have the largest absolute
Pearson correlation. The correlations are measured on the rows of the training
engines only, after smoothing and rescaling — the same table the network is
trained on — and the graph is rebuilt for every seed, because the seed decides
which engines are held back. A tie goes to the pair that comes first in sensor
order.

**Why.**

1. **A cut-off lets the number of edges float, and on FD001 it floats to a
   graph that cannot test anything.** Measured on the 80 training engines of
   seed 42 under `config/fd001.yaml`:

   | Cut-off on the absolute correlation | Edges | Sensors joined to nothing | Largest group of joined sensors |
   |---|---|---|---|
   | 0.5, 0.6 or 0.7 | 66 | P15, Nc | 12, every pair of them joined |
   | 0.8 | 62 | P15, Nc | 12 |
   | 0.85 | 42 | P15, Nc | 12 |
   | 0.9 | 12 | T24, T30, P15, Nc, htBleed | 7 |
   | 0.95 | 0 | all 14 | — |

   From 0.5 to 0.7 the graph is twelve sensors each joined to all the others,
   66 edges being every pair of twelve. Once the rows are rescaled to sum to one
   (assumption A-21), those twelve rows are identical, so the first graph layer
   hands all twelve sensors the same average — the flattening the
   every-sensor-joined graph causes, which the audit of 2026-09-08 raised as its
   item D-2. A comparison against that graph would measure density, not which
   relationships were chosen.
2. **Fixing the count at the paper's 41 leaves one difference: which pairs are
   joined.** The engineer chose 41; the data chooses 41.
3. **Pearson, because the paper uses Pearson** for the neighbouring decision of
   which sensors to keep. The absolute value, because a strong inverse
   relationship is as much a relationship as a direct one.
4. **Training engines only**, for the reason the rescaling is fitted on them
   (assumption A-25): otherwise the held-back and test engines would help decide
   the model that is then scored on them. `tests/test_pipeline.py` checks that
   the graph handed to the network is exactly the one built from those rows.

**What the graph looks like.** Recomputed on each seed's training engines:

| Seed | Weakest kept correlation | Edges shared with Table 3 | Joined to nothing | Independent rows after rescaling |
|---|---|---|---|---|
| 42 | 0.8550 | 17 | P15, Nc | 10 |
| 43 | 0.8498 | 18 | P15, Nc | 12 |
| 44 | 0.8516 | 18 | P15, Nc | 12 |
| 45 | 0.8507 | 18 | P15, Nc | 12 |
| 46 | 0.8484 | 18 | P15, Nc | 11 |

Fewer than half of the 41 edges are ones the engineer also drew. `P15` and
`Nc` are joined to nothing on every seed, so in this network those two sensors
only ever combine with themselves. "Independent rows" counts the rows of the
rescaled table that cannot be built by adding up other rows: 14 would mean
every sensor can end up with its own mixture of its neighbours. The paper's own
graph gives 8 (the audit's item D-1).

**How much it matters — measured.** Five seeds of each graph at commit
`f386291-dirty`, under the current default: z-score scaling, row rescaling,
capped test labels, 20 % of engines held back with early stopping, the step
schedule, and the last-cycle temporal stage, so no recurrent stage. Paired by
seed on the test engines:

| Measure | Table 3, the paper's graph | 41 strongest correlations | difference | paired t | p |
|---|---|---|---|---|---|
| RMSE | 16.496 ± 0.415 | 16.592 ± 0.354 | +0.096 | +0.48 | 0.655 |
| NASA score | 539.8 ± 44.7 | 1032.7 ± 292.2 | +492.9 | +3.91 | 0.017 |
| R-squared | 0.830 ± 0.008 | 0.829 ± 0.007 | −0.002 | −0.48 | 0.657 |
| ASUE | 11.651 ± 0.995 | 9.865 ± 0.153 | −1.786 | −4.35 | 0.012 |

**By RMSE and R-squared the two graphs cannot be told apart.** The per-seed
RMSE differences are `[+0.238, +0.311, +0.146, +0.457, −0.672]`.

**On the held-back engines the correlation graph does better.** Each seed holds
back the same 20 engines whichever graph it trains on, so the held-back error
pairs by seed too, and a comparison between configurations should be judged on
it rather than on the test engines (the audit of 2026-09-08 found this project
had not always done so):

| | Table 3 | 41 strongest correlations | difference | p |
|---|---|---|---|---|
| held-back RMSE | 21.168 ± 0.870 | 19.169 ± 1.473 | −1.999 | 0.012 |

Lower on every seed: `[−1.798, −2.391, −3.429, −1.734, −0.644]`. **Read it
with one caution.** The held-back error a run records is the lowest that error
reached over all the passes, and it is also what chose which pass's weights to
keep (assumption A-18), so it is flattered — by an amount that can differ
between two networks whose errors wander differently from pass to pass. The
test error was not used to choose anything. The two figures also weight
engines differently and are never comparable with each other (assumption
A-06). So the held-back result says the correlation graph is at least not
worse; it does not settle that it is better.

**They do make different errors.** The correlation graph predicts more life
than the engine had, on every seed. Read from the `test_predictions.csv` each
seed saved:

| | Table 3 | 41 strongest correlations |
|---|---|---|
| mean of predicted − actual, cycles | 0.860 | 2.611 |
| test engines predicted late, of 100 | 50.8 | 55.2 |
| largest late error, cycles | 45.7 | 59.3 |

(means over the five seeds). The shift in the mean error is
`[+1.504, +1.656, +1.924, +1.253, +2.417]` by seed, paired p = 0.0009. The
NASA score punishes a late prediction exponentially harder than an early one
(assumption A-15) and ASUE counts only early ones, which is why the first moves
up and the second down. Two cautions about those two results: four measures
were compared, and at 0.05 divided among four tests (0.0125) only ASUE's
difference clears the bar; and the mean-error comparison was run after seeing
the score and ASUE, to explain them, so it is a follow-up and not a test
decided in advance. One seed of the correlation run, seed 42, kept its weights
from the third pass over the data.

**What this says about the paper's claim.** In this configuration the
engineering knowledge did not make the network more accurate than the data's
own 41 strongest relationships — not on the test engines, and not on the
held-back engines, where the correlation graph came out ahead; it did make its
test errors less often and less severely late. That is a fact about this architecture — one number per sensor
and no recurrent stage — and not about the full model the paper's claim is
made for. Two of this project's configuration comparisons have reversed when
the architecture moved (working rule 3), so it should be re-measured with
`--temporal lstm` before being quoted about ESTGNN.

**What this says about the "without physics prior" experiment.** Under the same
configuration, the every-sensor-joined graph costs 2.784 cycles of RMSE
(finding F-04). The correlation graph removes the same engineering knowledge
but keeps the number of edges, and costs 0.096, indistinguishable from zero. So
in this configuration nearly all of what the every-sensor-joined comparison
measures is the flattening that graph causes, not the knowledge it removes.
That supports the audit's item D-2, and it is an inference from two
measurements rather than a third measurement of its own.

**Reproduce with:**

```bash
python scripts/train.py --config config/fd001.yaml --seeds 5
python scripts/train.py --config config/fd001.yaml --graph correlation --seeds 5
```

The stored runs are
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z`
and
`fd001_correlation_row_cap_5seed_varsplit_cfg7fdfce_f386291-dirty_20260914T150213Z`.
The first reproduces the `ae59bc1-dirty` run of 2026-09-08 on all four measures
and all five seeds, so the tree it ran from trains the network as before. The
threshold table and the per-seed graph table are recomputed by
`python scripts/verify_claims.py --only correlation` for seed 42.

---

## A-28 — The window-summary stage gives each sensor its last reading, its average and its change

**Where:** `models/temporal/WindowSummary`, `scripts/train.py --temporal window_summary`

**The gap.** This is not a gap in the paper; it is a stage the paper does not
have, added to answer a question the paper raises. Table 4 prints one number per
sensor entering the first graph layer, `W(1) ∈ ℝ(1×32)`, and Section 2.3.2 says
where that number comes from: *"X represents the input node features from LSTM
networks"*. The default here passes the last reading instead (assumption A-07),
which throws away 29 of the 30 cycles. The recurrent stage keeps them but brings
in sizes the paper never states (assumption A-26). Neither answers the plain
question: does the direction a reading is moving in carry information the last
reading alone does not?

**What we chose.** A stage with no learned parameters that gives each sensor
node three numbers from its own 30 readings:

1. the last reading;
2. the average of the 30;
3. the **change**: how far the best straight line through the 30 readings rises
   from the first cycle to the last, which is its slope per cycle times 29.

The first graph weight matrix becomes `(3, 32)`, so the network learns 61,793
numbers against Table 4's 61,729. It is not the default, and a run using it says
so in its directory name and in `summary.json`.

**Why these three, and why the change rather than the slope.** They are the
simplest description of a short series: where it ends, where it sits, and which
way it is going. The change is given in the readings' own units rather than per
cycle because all three enter one weight matrix whose starting values are drawn
on the same scale for every input (assumption A-14), and there is no bias in the
graph layers to absorb a difference in size (assumption A-12). Measured on seed
42's training windows for `T50`, averaged over each band of remaining life:

| Remaining life | Windows | Last reading | Average | Change |
|---|---|---|---|---|
| 0–25 | 2000 | +1.787 | +1.181 | +1.066 |
| 25–50 | 2000 | +0.815 | +0.457 | +0.673 |
| 50–75 | 2000 | +0.215 | +0.003 | +0.386 |
| 75–100 | 1999 | −0.131 | −0.268 | +0.264 |
| 100–124 | 1920 | −0.368 | −0.479 | +0.196 |
| 125, the capped plateau | 4540 | −0.659 | −0.708 | +0.089 |

Per cycle the same change would be the last column divided by 29, from +0.003 to
+0.037, while the last reading's band averages run from −0.659 to +1.787. That
the scale matters is an argument, not a measurement: the per-cycle version was
not trained. The scale was fixed before
either run below was started, and no other version was tried.

**Is it physically admissible on C-MAPSS?** Yes, on three counts. All three
numbers are computed from readings up to the cycle being predicted at and none
after it; the smoothing before them only looks backwards
(`tests/test_smoothing.py::test_the_filter_only_looks_backwards`). Every FD001
test engine has at least 31 cycles (finding F-03), so every test window is a
full 30 readings. And an operator predicting an engine's remaining life at a
given flight has that engine's last 30 flights of readings.

**How much it matters.** Measured, five seeds each, both runs at commit
`5602773-dirty` from the same tree, under `config/fd001.yaml`: z-score scaling,
row rescaling, capped test labels, a fifth of the engines held back with early
stopping and the best pass restored, the step schedule.

| Stage | Test RMSE | NASA score | R² | ASUE | Held-back RMSE |
|---|---|---|---|---|---|
| last cycle (default) | 16.496 ± 0.415 | 539.8 ± 44.7 | 0.830 | 11.651 | 21.168 ± 0.870 |
| **window summary** | **13.586 ± 0.675** | **374.7 ± 71.1** | **0.885** | **8.643** | **15.823 ± 0.461** |

Paired by seed, the window summary is **2.910 cycles lower** on test RMSE, lower
on all five seeds, paired `p = 0.00025`. Resampling the 100 test engines 20,000
times gives a 95 % interval for the difference of **[−5.19, −0.63]**, lower in
99.4 % of resamples, so the difference is not a matter of which engines happen
to be in the test set. On the held-back engines, which is the error a choice of
configuration should rest on, it is 5.345 lower (`p = 0.00002`); that figure
also chose which pass was kept, for both stages, so it is flattered for both.

Against the recurrent stage's stored run
(`…_lstm32_cfg72f506_f386291-dirty_20260914T134955Z`, 13.307 ± 0.532, the same
settings, one commit earlier), the window summary is +0.279 on test RMSE, lower on
2 of 5 seeds, `p = 0.506`, interval over engines [−1.62, +2.29]: the two cannot be
told apart on the test engines. On the held-back engines the recurrent stage is
1.689 lower, on all five seeds (`p = 0.0018`). The pairing crosses one commit;
the last-cycle run above reproduced the stored last-cycle run of `f386291-dirty`
exactly on every seed, and every stored network, the recurrent ones included,
reloads to its saved predictions under
`tests/test_run_records.py::test_a_stored_network_still_gives_the_predictions_it_saved`.

**What this does and does not show.**

- It shows that, under this configuration, most of the gap between the default
  and the recurrent stage is information the default never receives: a
  parameter-free summary closes it on the test engines.
- It does not show that the paper's recurrent stage is unnecessary. The paper's
  claim is about its own stage and procedure, and on the held-back engines the
  recurrent stage is still ahead.
- It does not show which of the three numbers the network uses. No explanation
  method has been run on this stage.
- It is one configuration. Two of this project's comparisons reversed when the
  configuration moved (working rule 3).
- It inherits the paper's graph. Sensors whose rows of the rescaled matrix are
  identical still reach the network only as sums, now of three numbers each
  instead of one (the audit of 2026-09-08, its item D-1).

**Which published row it may be set beside.** None. Every row of Tables 7 to 11
describes either a model with the paper's recurrent stage or one with no
temporal stage at all.

**Reproduce with:**

```bash
python scripts/train.py --config config/fd001.yaml --temporal window_summary --seeds 5
python scripts/train.py --config config/fd001.yaml --seeds 5
```

The stored runs are
`fd001_physics_row_cap_5seed_varsplit_window_summary_cfgba82e3_5602773-dirty_20260914T233435Z`
and `fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z`,
each carrying its command, library versions, data-file fingerprints, the
uncommitted code as a patch, and each seed's engine split and scaling.

---

## A-29 — Walks through the sensor graph are explained with GNN-LRP, on the score before the sigmoid, with gamma = 0 by default

**Where:** `explain/walks.py` (`walk_relevance`)

**The gap.** Not a gap in a method the paper describes, but a question it raises
and does not answer. Section 2.4 applies SHAP to the fourteen sensor nodes and
calls the result *"node-level contribution analysis"* (p. 7). Section 4.4 then
says the rankings *"determine both the node coloring and edge weights in the
interpretability graph"* (p. 13) and ends by saying the changing patterns
validate *"the effectiveness of the physics-informed graph structure"* (p. 14).
No calculation in the paper attributes anything to a connection between two
sensors: the edge weights of Fig. 10 are derived from node rankings by a rule
the paper's text does not state. So the paper has no method for asking how much of a
prediction travelled through its graph, which is what its claims about the
graph need.

**What we chose.**

1. **GNN-LRP** (Schnake et al., 2022), which divides a graph network's output
   into one share per walk — the list of sensors a contribution passes
   through, one per graph layer — rather than GNNExplainer (Ying et al., 2019)
   or PGExplainer (Luo et al., 2020), which learn a mask over the connections.
2. **The score before the sigmoid** is what is divided, not the prediction in
   cycles.
3. **gamma = 0** by default. Gamma is GNN-LRP's one setting; a positive value
   shrinks negative shares while keeping each layer's total. It is always 0 at
   the first graph layer; from the second layer on a positive value may be
   passed, one per layer.
4. **The added constants** of the two dense layers, and of the graph layers
   when they have them, are reported as one number per window, not spread over
   the walks.
5. **Walks start at the temporal stage's output**, the numbers entering the
   first graph layer, not at the raw readings.

**Why.**

- **GNN-LRP over a mask.** Both mask methods state their objective for
  classification and neither paper mentions regression; both start from a
  random mask and depend on penalty weights, a learning rate and a number of
  steps that no source fixes for this network, and a replication of
  PGExplainer (Holdijk et al., 2021) could not reproduce its results from the
  paper and found them sensitive to exactly those settings; both need the
  network changed to accept a weakened graph per window; and both force a
  choice between weakening the raw graph and rescaling again, which changes
  the rest of the row, or weakening the rescaled graph, which the network
  never saw. GNN-LRP needs none of that, changes nothing in the network, and
  its shares add up.
- **Why the shares add up exactly here.** Our own derivation, not the source's
  theorem. The source's exact decomposition (its Proposition 1) assumes every
  step, including the layers after the graph, has no added constant; ours has
  them in `fc1` and `fc2`. What makes the split exact for this network is that
  every step between the graph input and the score is multiplication, addition
  or a ReLU, the graph layers add no constant (assumption A-12), and the dense
  layers keep each sensor's numbers apart rather than averaging them. With each
  ReLU held on or off as the window set it, the score is a sum of products,
  one per walk, plus a remainder from the constants. The remainder is computed
  separately, by carrying the constants forward, so that walks plus remainder
  equalling the score is a check and not a definition. The remainder changes
  from window to window with `fc1`'s ReLUs, so the walks add up to the score
  minus that remainder, not to the score minus the score of some reference
  engine, which is what Shapley values add up to.
- **The score, not cycles.** The sigmoid bends, so shares of the score cannot
  become shares of cycles that still add up. The source explains outputs before
  the final squashing step in its own examples: the difference of two class
  scores before the softmax (its Fig. 7), and a non-linear output made linear
  first (its Section 5.2). It never discusses a sigmoid.
- **gamma = 0.** It is the one setting at which each share is exactly what the
  network computes at this window, it needs no value chosen, and no source
  gives one for this network. At the first graph layer it is also the only
  setting the source's rule supports: that rule's derivation (its appendix C)
  assumes the numbers entering a layer are never negative, and this network's
  first graph layer receives rescaled readings, which are negative below the
  training average. The cost is stated by the source itself: with gamma = 0
  GNN-LRP is identical to its own simpler baseline, GNN-GI, which it reports as
  noisier and less selective than a positive gamma; its experiments use
  positive values that fall towards the output.
- **Walks from the temporal stage's output.** Every implementation by the
  source's authors stops at the input of the first graph layer. Going further
  back through the recurrent stages would need rules for their gates that this
  project does not have; for the stages that learn nothing, the starting number
  is a reading or an average of readings.

**Where the source's section numbers come from.** The arXiv version 3 of the
paper, since the journal's own text could not be retrieved; the journal version
may number them differently.

**How much it matters.** The choice of method is an argument, not a
measurement. What is measured is, first, the arithmetic, in `tests/test_walks.py`:
walks plus constant give back the score to within 1e-9 for every temporal stage,
with and without the graph layers' constants; each sensor's total equals
PyTorch's own differentiation, at both ends of the walks and for each graph
entry; and for 30 walks at gamma 0 and at gamma (0, 0.5, 0.25) the shares equal
the source's own procedure, which runs the network backwards once per walk with
every other route blocked.

Second, what the walks say about the stored networks, measured on 2026-09-26 by
`scripts/measure_walks.py` at `14c1d64-dirty`, on the last window of each of the
100 test engines, for all five saved seeds of five stored runs. Beside each
trained network the script explains an **untrained** one, built exactly as
`scripts/train.py` builds that seed's network before training it, so it is the
trained network's own starting point. A number says something about training
only where the two differ. Each cell is the mean ± spread over the five seeds of
each seed's median over the 100 windows:

| Stored run | Network | Walks that stay within their starting sensor's group, share of all walk size | Constant, share of score's size | Walk sizes ÷ size of their sum | Connection ranking, gamma 0 against (0, 0.5, 0.25) |
|---|---|---|---|---|---|
| Paper's graph, last cycle (the default) | trained | 5.51 ± 3.04 % | 51 ± 4 % | 8.6 ± 3.5 | 0.902 ± 0.040 |
|  | untrained | 15.43 ± 3.74 % | 82 ± 14 % | 8.2 ± 5.2 | 0.992 ± 0.007 |
| Paper's graph, window summary | trained | 13.99 ± 3.55 % | 57 ± 4 % | 4.5 ± 1.5 | 0.966 ± 0.011 |
|  | untrained | 13.55 ± 2.25 % | 74 ± 32 % | 8.1 ± 3.3 | 0.975 ± 0.015 |
| Paper's graph, recurrent stage | trained | 15.44 ± 3.23 % | 11 ± 5 % | 2.2 ± 0.3 | 0.992 ± 0.005 |
|  | untrained | 16.46 ± 6.36 % | 97 ± 4 % | 9.2 ± 4.9 | 0.995 ± 0.011 |
| Correlation graph, last cycle | trained | 9.65 ± 2.68 % | 36 ± 4 % | 9.9 ± 1.7 | 0.928 ± 0.024 |
|  | untrained | 7.18 ± 2.65 % | 86 ± 13 % | 11.0 ± 3.7 | 0.935 ± 0.070 |
| Every sensor joined, last cycle | trained | 100 % | 54 ± 3 % | 3.6 ± 0.0 | 1.000 ± 0.000 |
|  | untrained | 100 % | 90 ± 7 % | 19.4 ± 19.0 | 1.000 ± 0.000 |

The columns: the size, ignoring signs, of the walks that never leave their
starting sensor's group, as a share of all walks' size, where sensors the graph
cannot tell apart form one group (finding F-18) and every other sensor is a
group of its own; the size of the dense layers' constant against the constant
plus the walks' sum; every walk's size added up and divided by the size of the
walks' sum, so 8.6 means the shares are large and mostly cancel; and Spearman's
rank correlation, over the connections, between what flowed along each at
gamma 0 and at gamma (0, 0.5, 0.25), with what arrives at a group at the last
graph layer split equally among its members.

**Correction, same day.** The first version of this table counted a walk as
staying only if it stayed on one sensor, and gave 0.37 ± 0.04 % trained against
0.46 ± 0.03 % untrained on the default. That split P30 -> P30 -> P30 -> P30
(staying) from P30 -> P30 -> P30 -> Ps30 (moving), although which of the two a
share lands on follows `fc1` weights that training never separated (finding
F-18), so part of that figure was the random start. It was caught by the review
that planned the playground view, before the figure was used anywhere.

What this shows, and what it does not:

- **How much stays within a group is mostly the graph's doing.** With the window
  summary, the recurrent stage and the correlation graph, the trained and the
  untrained networks cannot be told apart (paired p = 0.86, 0.62 and 0.13). On
  the default, training moved weight away from staying: 5.5 % against 15.4 %,
  lower on all five seeds (paired p = 0.023). Where everything is joined to
  everything, every sensor is in one group and every walk stays, by definition.
  A share of this kind cannot, on its own, show that the network uses its graph.
- **About half of the default network's score belongs to no sensor**: the dense
  layers' constant is 51 ± 4 % of the score's size (82 % before training,
  paired p = 0.0088). With the recurrent stage it is 11 ± 5 %.
- **The shares cancel heavily and training does not make that worse**: 8.6
  against 8.2 before training on the default (paired p = 0.85). A list of the
  largest walks, read without that ratio, shows mostly cancellation.
- **Rankings of connections move little with gamma**: 0.902 ± 0.040 at
  (0, 0.5, 0.25) and 0.866 ± 0.054 at (0, 1, 1) on the default, above 0.98 with
  the recurrent stage. Where everything is joined to everything, gamma changes
  nothing, because every row is the same.
- **Seeds agree on which connections carry most, except with the recurrent
  stage.** Spearman's rank correlation between every pair of seeds, over the
  connections they share: 0.71 to 0.96 on the default, 0.86 to 0.99 on the
  correlation graph, 0.30 to 0.79 with the recurrent stage.
- Nothing here says the paper's graph helps or hurts. That needs networks
  retrained on another graph (assumption A-27), not a reading of one network.

The five stored runs, each measured in its own folder under `walks/`:
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z`
(`14c1d64-dirty_20260926T160933Z`),
`fd001_physics_row_cap_5seed_varsplit_window_summary_cfgba82e3_5602773-dirty_20260914T233435Z`
(`…T161011Z`),
`fd001_physics_row_cap_5seed_varsplit_lstm32_cfg72f506_f386291-dirty_20260914T134955Z`
(`…T160959Z`, the best pass restored, not the paper's 200-pass procedure),
`fd001_correlation_row_cap_5seed_varsplit_cfg7fdfce_f386291-dirty_20260914T150213Z`
(`…T160945Z`) and
`fd001_complete_row_cap_5seed_varsplit_cfgf21547_f386291-dirty_20260914T150527Z`
(`…T161022Z`). Before the correction above, the stored run of the default
settings at `f386291-dirty` (`…T150004Z`) gave every figure identical to the
`5602773-dirty` run, as it should: that run reproduced it exactly.

**Reproduce with:**

```bash
python scripts/measure_walks.py --run <stored run directory>
```

**What a walk's share is not.** It is not what removing a connection would do —
that changes the row rescaling and can switch ReLUs, which is a different
network — and it is not a statement that one sensor affects another in the
engine.

---

## A-30 — Removing a connection deletes it from the raw graph and rescales the whole graph again

**Where:** `explain/connections.py` (`connection_removal`), `scripts/measure_walks.py`

**The gap.** The paper removes its whole graph once, in its variant "without
physics prior" (Table 11), and retrains; it never removes a single connection,
and never says what removing one would mean. A walk's share (assumption A-29)
describes a small change to one entry of the rescaled graph, so asking what a
connection is worth needs a second, separate measurement, and a definition.

**What we chose.** Delete the connection from the raw graph, in both directions,
rescale the whole graph with the run's own rescaling, and run the same trained
network again, without retraining. The value is the prediction with the graph as
trained minus the prediction without the connection, in cycles, so a positive
value means the connection raised the prediction.

**Why.** It is how the network's graph was made: the raw graph is built first
and rescaled after, so removing a connection from the raw graph gives a graph of
the same kind the network was trained on. The alternative, zeroing one entry of
the rescaled graph, leaves a row that no longer adds up to 1, which the network
has never seen. The cost is that removal also changes entries nobody removed:
under row rescaling a sensor with seven neighbours goes from eight entries of
1/8 to seven of 1/7. `tests/test_examples_torch.py::test_example_removing_a_connection_by_hand_on_the_same_three_sensors`
works one case through by hand. Not retraining keeps the measurement about one
trained network, as every explanation here is; retraining would measure a
different network.

**How much it matters, measured.** The same runs, seeds, windows and commit as
the table in assumption A-29. Spearman's rank correlation, over the connections,
between the average size of what flowed along each connection (its walk shares,
summed over the graph layers and both directions) and the average size of what
removing it does:

| Stored run | Trained | Untrained | Paired p | Trained against zero, p |
|---|---|---|---|---|
| Paper's graph, last cycle (the default) | −0.16 ± 0.20 | 0.42 ± 0.25 | 0.009 | 0.158 |
| Paper's graph, window summary | 0.34 ± 0.37 | 0.13 ± 0.46 | 0.507 | 0.112 |
| Paper's graph, recurrent stage | 0.40 ± 0.14 | 0.11 ± 0.29 | 0.159 | 0.003 |
| Correlation graph, last cycle | 0.63 ± 0.08 | 0.45 ± 0.21 | 0.094 | 0.0001 |
| Every sensor joined, last cycle | 0.25 ± 0.01 | 0.20 ± 0.12 | 0.344 | < 0.0001 |

What flowed is added up with what arrives at a group of indistinguishable
sensors at the last graph layer split equally among them (finding F-18).

On the default network, **what flowed along a connection says nothing about what
removing it does**: the rank correlation cannot be told from zero (p = 0.158),
and training moved it down from 0.42. On the correlation graph and with the
recurrent stage the two agree moderately. So "the connections that matter most"
names two different things, and a claim of that kind has to say which.

**A trap for both measures.** Removing a connection that touches one of the
sensors the paper's graph cannot tell apart (finding F-18) makes that sensor's
row differ from its group's, so the prediction starts to depend on weights that
training never separated. What such a removal does is partly the random start.

---

### F-18 — The paper's graph cannot tell three groups of sensors apart, and training never separates their weights

**Established from.** The rows of the paper's rescaled graph, and the five saved
seeds of five stored runs.

**The observation.** Under both rescaling formulas, the rows of the rescaled
graph are identical for P15, Nf, NRf and BPR, for P30 and Ps30, and for Nc and
W31. Each member of a group is joined to exactly the same sensors as the others,
itself and the others in its group included. The fourteen rows can all be
written as sums of multiples of just eight of them (the audit of 2026-09-08, its
item D-1, recorded the eight; `scripts/verify_claims.py --only same_rows`
recomputes both).

**What follows, shown with the network.** A graph layer gives each sensor a mix
of its row of neighbours, so from the first graph layer on every member of a
group holds exactly the same numbers. The first dense layer, `fc1`, therefore
receives the same numbers from every member, each member's block of `fc1`
weights receives the same correction at every training step, and the difference
between two members' blocks stays what the random start made it.

Measured on 2026-09-26 by `scripts/measure_walks.py` at `14c1d64-dirty`, against
each network's starting point rebuilt from its seed, over the five seeds of the
default stored run
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z`:
for every pair of members, the difference between their blocks moved by at most
**4.65e-06** during training, while every member's block itself moved by at
least 0.30, and the same difference for T24 and T30, which are in no group,
moved by 0.32 to 1.31. The window-summary, recurrent-stage and all-joined runs
gave at most 2.73e-06, 3.06e-06 and 1.46e-06. The correlation graph has groups of
its own, different in each seed, with at most 2.12e-06.
`tests/test_walks.py::test_training_never_changes_the_difference_between_their_fc1_weights`
shows the mechanism in 20 training steps.

**Why it matters.** The prediction depends only on each group's total, which
training does shape, so the network's accuracy and every explanation made by
replacing whole sensors, the Shapley values and occlusion among them, are
unaffected. But anything read off the split *within* a group at the dense layer
is partly the random start: where a walk ends inside a group (assumption A-29),
and what removing a connection that touches a group member does (assumption
A-30). `explain/walks.py` finds the groups for any graph with
`indistinguishable_sensors` and adds such walks together with
`top_walks(..., pool=...)`.

**What it does not show.** That the graph is wrong, or that these sensors carry
the same information in the engine: it is a property of which sensors Table 3
joins, not of the readings.

---

## A-31 — The engine overlay locates variables by section, not by probe coordinates

**The gap.** Table 2 of Wen et al. names `T30` as "Total temperature at HPC
outlet" and `phi` as "Ratio of fuel flow to Ps30". Section 2.3.1 distinguishes
measurements at the same engine station from speed correction, combustion and
cooling relationships. Figure 3 shows an engine schematic and a separate graph.
Neither supplies installation coordinates, dimensions or routing for physical
probes. Checked against page 5 on 2026-09-27.

**What we chose.** The playground adds an original SVG cutaway: a browser drawing
of the fan, bypass passage, compressors, combustor, turbines and shafts. Sensor
nodes sit on callouts around it. Dashed leaders anchor them to a component or
system; solid graph lines retain the model's connections. Circles denote
measurement regions; rounded boxes denote shaft, calculated or system quantities.
The geometry, blade counts, callout offsets and cooling branches are illustrative,
not an engineering drawing of a named production engine. Node positions do not
enter any model calculation.

**Sources and existing implementations checked.**

- Saxena et al. (2008), [NASA report 20090029214](https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf),
  Figure 1 and Figure 2 supply the component and flow-path arrangement; Table 2
  supplies the dataset variable definitions. These are the placement authority.
- NASA's [Engine Parameter Interactive, EngineSimU](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/enginesimu/)
  already implements selecting engine components on a schematic. NASA publishes
  its source in [nasa/BGA](https://github.com/nasa/BGA). It is an educational engine
  simulator, not a C-MAPSS sensor-coordinate map or a GNN-LRP implementation.
  Its documented interaction informed this view; no source code, artwork or
  engine-performance equations were copied into the repository.
- Frederick et al. (2007), [NASA/TM-2007-215026](https://ntrs.nasa.gov/api/citations/20070034949/downloads/20070034949.pdf),
  Tables 1.2 and 1.3 distinguish model outputs from other calculated variables.
  Table 1.3 labels both W31 and W32 as HPT coolant bleed. The dataset paper's
  Table 2 and Wen et al.'s Table 2 instead identify W32 with the low-pressure
  turbine. The overlay follows those two dataset definitions; it does not infer
  a physical measuring station from the earlier guide's conflicting label.

The search found a relevant NASA implementation, but no verified existing
C-MAPSS cutaway with a GNN-LRP overlay. That is a description of the search,
not a claim that no such implementation exists.

| Variables | Anchor in this drawing | Limit of the evidence |
|---|---|---|
| T24 | Low-pressure compressor outlet | Region documented; position on the casing unknown |
| T30, P30, Ps30 | High-pressure compressor outlet | Same region, separated callouts for readability |
| T50 | Low-pressure turbine outlet | Region documented; exact probe position unknown |
| P15 | Bypass duct | Passage documented; exact point unknown |
| Nf, NRf | Fan shaft | Physical speed and its corrected quantity, not two located probes |
| Nc | Core shaft | Shaft speed, not a specified measuring point |
| phi | Fuel-system callout | A ratio involving fuel flow and Ps30, not a point sensor |
| BPR | Bypass/core split | A ratio of flow rates, not a point sensor |
| htBleed | Schematic bleed system | The definition supplies an energy quantity, not a unique station |
| W31, W32 | Cooling branches associated with the high- and low-pressure turbines | Neither exact extraction nor measuring points are established |

**Why.** A region anchor lets a reader connect a variable name to the part of
the engine it describes. Separate callout shapes avoid suggesting that every
input is a physical probe. Graph layout remains available because placing nodes
by engine section can make their connections overlap. A selected sensor limits
the emphasis to its neighbours. These are design arguments; improved reader
understanding has not been measured.

**How much it matters.** The change only reads stored explanations. GNN-LRP
edges come from the stored connection columns, including connections with a
zero share. Other methods read the selected seed's saved adjacency matrix;
if that cannot be read, connections are omitted with an explanation. A layout
switch must leave those edges and every displayed contribution unchanged.
Marker strengths and line widths share fixed scales across the stored windows.
The node values retain their method's units: cycles for Shapley and occlusion,
shares of the score before conversion to cycles for graph walks.

The saved largest walks combine sensors the graph cannot distinguish (finding
F-18) and store averages over all explained windows, not a separate value for
the selected window. The overlay highlights all compatible connections between
successive groups and names the layer at each step. It does not choose one
member of a group or assign the average share to a particular window. Repeated
groups are shown in the layer sequence, including the possibility of remaining
on one sensor. Connection widths still describe totals over all walks and
layers, not the selected walk alone. None of these marks is a fault location,
physical flow of relevance or causal claim about engine damage.

---

## References

Full citations for every work mentioned above are in `REFERENCES.md`.
