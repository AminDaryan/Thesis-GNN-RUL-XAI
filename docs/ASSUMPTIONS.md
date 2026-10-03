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
| A-06 | Validation split | None; trains for a fixed 200 epochs | **20 % of the engines**, reinstated 2026-09-08 after the A-22 removal (date corrected 2026-10-02 from 2026-09-02; see A-06); worth 0.32 cycles, which five seeds cannot distinguish from zero (p = 0.234) |
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
| A-18 | Early stopping | None; trains a fixed 200 epochs | **On**, since the split returned on 2026-09-08 (date corrected 2026-10-02 from 2026-09-02; see A-06); fires between epochs 8 and 55 of a nominal 200 |
| A-19 | What "without physics prior" removes | Names the ablation, never defines it | Replace the graph with a fully connected one |
| A-20 | What "a sensor not considered" means | Equation 15 needs `f(S)`, but a network cannot take 13 inputs | Replace the sensor with its average training window |
| A-21 | Adjacency normalisation formula | Says only "normalized ... incorporating self-loops"; never prints it | **Row `D^(-1)(A+I)`** — 0.25 cycles closer *with nothing held back* under the step schedule (p = 0.068, **no longer significant**), selected under A-22 |
| A-23 | Engines shorter than the window | Not discussed; 17 test engines across FD002 and FD004 are affected | Left out, following the only published treatment found; no FD001 engine is affected |
| A-24 | What `phi_0` is in Equation 14 | Calls it "the average value of target variable across all samples" | The prediction at the average window, forced by A-20; **10.719 cycles** from the paper's wording |
| A-26 | Size and depth of the recurrent stage | Table 4 has **no recurrent row**; Table 6 lists no hidden size, depth or projection | One layer, hidden size 32, a linear projection to the one number Table 4 prints; **fourteen dedicated cells**, which the paper does state |
| A-25 | Where the scaling is fitted | Fig. 1 puts normalisation **before** the train and test datasets; never names the fitting set | Split first, fit on training engines only |
| A-27 | A sensor graph built from the data | Never builds one; its Conclusions claim its own graph goes "beyond data-driven correlations" | The 41 most strongly correlated pairs on the training engines, as many edges as Table 3; **indistinguishable from Table 3 by test RMSE** (+0.096, p = 0.655), lower held-back error on every seed, by 1.8 to 2.0 cycles on average on each of four summaries of it (p 0.010 to 0.020), but its test predictions run later; with the recurrent stage, indistinguishable again by test RMSE (+0.002, p = 0.995) and no held-back advantage that survives a summary no pass was chosen by |
| A-28 | Each sensor's last reading, average and change as node features | Never considered; Table 4 prints one number per sensor, from the LSTM stage | A stage outside the paper, not the default; **2.910 cycles below the last cycle** (p = 0.00025, engine interval [−5.19, −0.63]), indistinguishable from the recurrent stage on test error |
| A-29 | Explaining the sensor graph itself | Explains sensor nodes only (Section 2.4), yet says its rankings set "edge weights" and validate its graph (Section 4.4) | Walk shares from GNN-LRP, of the score before the sigmoid, with the authors' own setting of gamma (0, 1.5, 0), 0 at the first graph layer; on the default network about half the score is the dense layers' constant (51 ± 4 %), and how much stays within a sensor's group is mostly set by the graph |
| A-30 | What removing one connection means | Removes the whole graph once and retrains; never one connection | Delete it from the raw graph both ways, rescale, rerun without retraining; on the default network its effect ranks the connections against what flowed along them (rank correlation −0.26 ± 0.14, p = 0.013) |
| A-31 | Placing explanation nodes on an engine | Table 2 names variables and component regions; Fig. 3 places a separate graph beside a schematic engine | An original schematic cutaway with section anchors, distinct markers for calculated and system quantities, and the stored model's graph overlaid; no claim of exact probe coordinates |
| A-32 | Learning-rate sensitivity levels | Table 6 gives `0.001` at the start and `0.0001` at the end; it gives no sensitivity study | Test those rates, one tenth of both, and ten times both, keeping the ten-to-one ratio |
| A-33 | Running the GNN-LRP paper's own check on this network | Not in the paper; the GNN-LRP paper's node flipping never says how a network on a fixed rescaled graph runs with nodes missing | A sensor's row and column of the rescaled graph zeroed, nothing rescaled; 20 random orders; walks beat random orders on every seed of all five configurations |
| A-34 | Starting values of the recurrent stage | Prints the gate equations with one offset each (Equations 1, 2, 3 and 5); never says what any number starts from | PyTorch's default, every number drawn evenly from ±1/√32, so each forget gate starts between 0.412521 and 0.587479 at a zero reading; Gers et al. (2000) start it above one half, at 0.622459 and 0.731059 for their first two blocks; effect not measured |
| A-35 | Which window a graph of Table 12 and Fig. 10 is | Numbers the graphs 1 to 191 under a column headed "Time"; never says which cycle of the engine a graph is | The engine's *g*-th window, ending at cycle *g* + 29; under every placement one window apart the default network's four largest values still change more often than Table 12's (8 to 28 joins against 1) |
| A-36 | A GRU in place of each sensor's LSTM | Has no such stage; its "GRU [55]" row is another published method | An optional stage at the LSTM stage's size, PyTorch's cell; on FD001 against the LSTM stage −0.416 test RMSE (p = 0.455) and −0.999 held-back (p = 0.0047), neither clearing the bar |
| A-37 | The bar when several comparisons are read together | Tests no difference: one number per cell, no seeds, no spread | p below 0.05 divided by the number of comparisons; among the seventeen of the FD001 sensitivity study it sets aside one test and two held-back differences |
| A-38 | Ordinary models' predictions outside 0 to the label ceiling | Never says how its compared methods' numbers were obtained or bounded | Clipped to 0 to 125 before scoring; lowers linear regression on the 42 window summaries by 1.295 cycles of RMSE, every other model by at most 0.144; no comparison with the network changes side |

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
paired **t = 22.30, p = 0.00002**. Highly significant, and in the only
direction RMSE can move under capping: it fell by about 1.3 cycles. (Until
2026-10-02 this sentence said "capping can only help", naming no measure; the
correction below says which measures that holds for.)

**Therefore the gap to the paper's 14.93 is wider than the headline says.**
This code reports 20.35 with the cap; without it, 21.66. Whichever is quoted,
the cap must be quoted with it. (Under the removed `window` mode the same
measurement gave +1.762 cycles; the effect is configuration-dependent, so it
must be re-measured whenever the architecture changes.)

**Re-measured on the default since 2026-09-08 (added 2026-10-02).** Z-score
input scaling, the graph rescaled by rows, a fifth of the training engines
held back with early stopping, and the single drop in learning rate
(assumptions A-03, A-21, A-06 and A-17). Stored runs
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_9856794-dirty_20260908T135935Z`
(capped) and
`fd001_physics_row_uncap_5seed_varsplit_cfge49671_9856794-dirty_20260908T141351Z`
(uncapped), both at `9856794-dirty`; their best passes (51, 55, 8, 18, 35) and
held-back errors agree on every seed, so they are the same trained networks
scored two ways. RMSE **16.496 ± 0.415** capped against **18.005 ± 0.438**
uncapped; per-seed penalty `[1.605, 1.356, 1.467, 1.596, 1.516]`, mean
**+1.508 ± 0.102 cycles**, paired **t = 32.97, p = 5.0e-06**. So on today's
default the number to quote is 16.50 with the cap and 18.01 without it. Added
because the figure above was the only one this entry gave, on a configuration
superseded since 2026-09-08; the audit of 2026-09-08 had flagged it (its D-9),
and it was set beside the current default on 2026-10-02 while the label cap's
comment in `config/fd001_uncapped.yaml` was being corrected.

**Correction, 2026-10-02.** "Capping can only help", as the paragraph above
said until this date, is true of two of the four measures and was written as
though it held for all four; the table above cannot show the difference,
because there all four happened to improve. The
review of the playground's information cards on 2026-09-26 found it and
corrected the cards, but not this entry. The argument, for one engine with
true remaining life `y`, ceiling `c = 125` and a prediction `p` that the
sigmoid keeps at or below 125:

- If `y` is 125 or less, capping changes nothing.
- If `y` is above 125, then `p ≤ 125 < y`, so the error `p − y` is negative
  before capping: the prediction was early. After capping the error is
  `p − 125`, still negative or zero, and smaller in size by exactly `y − 125`.
  Engine 25, labelled 145, is charged 20 cycles less whatever the model
  predicted for it.

So every engine's error keeps its direction and can only shrink. That
guarantees two of the measures. **RMSE** is built from squared errors, none of
which grows, so it cannot rise. **The NASA score** charges an early error of
size `s` the amount `exp(s/13) − 1` (assumption A-15), which grows with `s`;
the error stays early and shrinks, so its charge falls.

It guarantees neither of the other two:

- **ASUE** averages the shortfall over the engines predicted below their
  label. While every prediction is below 125, the same engines stay in that
  average with smaller shortfalls, so it cannot rise. But exactly 125 is
  reachable: in 32-bit arithmetic PyTorch's sigmoid returns exactly 1 once its
  input reaches about 16.64. An engine above the ceiling predicted at exactly
  125 has no shortfall after capping and leaves the average. At the same
  moment every other engine above the ceiling loses its own `y − 125` from its
  shortfall, so the average can rise or fall. With the real labels of FD001
  test engines and predictions chosen to show the arithmetic: engine 74
  (labelled 126) predicted 125 and engine 1 (labelled 112) predicted 100 give
  an ASUE of (1 + 12) / 2 = 6.5 uncapped and 12 capped, a rise, because
  engine 1 is below the ceiling and keeps its shortfall of 12. Put engine 25
  (labelled 145) predicted 123.5 in engine 1's place and the ASUE is
  (1 + 21.5) / 2 = 11.25 uncapped and 1.5 capped, a fall, because engine 25's
  shortfall shrinks by 20 while engine 74 leaves.
- **R-squared** is 1 minus the squared errors divided by the labels' squared
  distances from their own mean (assumption A-16), and capping shrinks both
  parts. On the 100 FD001 test labels the second part falls from 172,686.96
  to 160,586.75. Predicting 75.52, the mean uncapped label, for every engine
  scores exactly 0 against the uncapped labels and −0.000713 against the
  capped ones, so capping made it worse.

`tests/test_examples.py` reproduces these examples from `RUL_FD001.txt`, and
checks every stored FD001 prediction file in `reports/` from a run whose
ceiling is 125 against the two guarantees. For RMSE the direction never needed the t-test above, since it
follows from the argument; what the five seeds measured is its size. The
R-squared and ASUE improvements in the table are outcomes measured on that
configuration, not properties of capping.

This entry caps the answers. Holding an ordinary model's predictions to the
same range, in the comparison with models that have no sensor graph, is a
separate choice with its own entry, assumption A-38 (pointer added 2026-10-03).

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

**Held-back error beside test error (added 2026-10-02).** This choice was made
on 2026-08-29 by the rule assumption A-22 records, which sets test error beside
the paper's figure. The audit of 2026-09-08 objected that choices between
configurations were being decided on the test engines (its item D-3). The two
runs in the table above held a fifth of the training engines back, so each also
recorded its error on those held-back engines, and that error pairs by seed,
because the seed decides which engines are held back:

| Scaling | Test RMSE | Held-back RMSE |
|---|---|---|
| per-condition min-max | 20.353 ± 0.752 | 23.072 ± 1.096 |
| z-score | 16.524 ± 0.339 | 21.034 ± 0.775 |
| z-score minus min-max, paired | −3.828, p = 0.0005 | −2.038, p = 0.017 |

Runs `fd001_physics_symmetric_cap_5seed_varsplit_09628fa_20260828T155320Z`
(min-max) and `fd001_physics_symmetric_cap_5seed_varsplit_09628fa_20260828T153058Z`
(z-score), both at `09628fa`: physics graph, symmetric rescaling, the
exponential learning-rate schedule then in force, the last reading of each
sensor and no recurrent stage, capped test labels. The z-score is lower on the
held-back engines on all five seeds (`[−3.185, −2.302, −2.467, −2.140, −0.094]`),
so **for this architecture the choice comes out the same on held-back error.**
The two kinds of error are never compared with each other (assumption A-06).

**With the recurrent stage it does not.** The sensitivity study (finding F-19,
`reports/sensitivity_fd65a3b-dirty_20260929T101059Z`, at `fd65a3b-dirty`) swaps
the scaling around a reference that has the paper's recurrent stage and z-score
scaling, everything else at the default. There, per-condition min-max is lower
on the held-back engines by 1.234 cycles (12.950 ± 0.522 against
14.184 ± 0.389), on all five seeds, paired p = 0.00286, under the study's
threshold of 0.05 / 17 = 0.00294 for its seventeen comparisons (assumption
A-37). On the test engines the same change is −1.048 (12.186 ± 1.462 against 13.234 ± 0.723,
p = 0.332), which five seeds cannot resolve. So a choice made on held-back error
would take the z-score for the default network and min-max for the network with
the recurrent stage: one more comparison that is a fact about one architecture
(working rule 3).

Two cautions. The held-back error a run records is the lowest that error reached
over all passes, and it also chose which pass to keep, so it flatters, by an
amount that can differ between the two arms and was not measured for these
runs. Where it was measured, for the three graphs compared under assumption
A-27, it was 0.2 to 0.7 cycles on average, and five seeds could not tell those
amounts apart.
And each p-value here is a paired test across five seeds scored on the same 100
test engines, or on held-back engines drawn from the same 100 training engines:
it says whether the difference could be an accident of how the networks started
and which engines each seed held back, not how it would come out on engines
outside this data (the audit's item D-8). The paired tests are recomputed from
each run's `results.csv` by the code in `docs/AUDIT_2026-09-08_response.md`.

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

**Reinstated 2026-09-08, reversing the removal of 2026-08-29.**
`config/fd001.yaml` holds out 20 % of the training engines again, so a run
restores the weights that scored best on them and may stop early.

**What that costs, measured.** Five seeds each at commit `ae59bc1-dirty`,
physics graph, row normalisation, z-score scaling, capped test labels, step
schedule. The seed varies which engines are held back as well as the
initialisation, so the spread is what a real repetition would show.

| Configuration | RMSE | closer to 14.93 by | paired two-sided p |
|---|---|---|---|
| `validation_fraction: 0.2` — the default since 2026-09-08 | **16.496 ± 0.415** | 0.321 | 0.234 |
| `validation_fraction: 0.0` — the default from 2026-08-29 to 2026-09-08 | 16.818 ± 0.157 | — | — |

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

**Correction, 2026-10-02.** Until this date this entry gave 2026-09-02 as the
day the split came back: in its first line, in both rows of the table above,
in the heading below and in that section's last sentence. Nothing about
holding engines back changed that day. What the history shows, every time in
UTC (git prints these commits in the author's local time, two hours ahead):

- **2026-08-29, 12:37.** The split was removed, in commit `6a7af8c`.
- **2026-09-02.** The one commit of that day, `e3bfd23`, still sets
  `validation_fraction: 0.0` and documents holding nothing back as the default.
- **2026-09-08.** The only record of a reason is an independent audit dated
  that day (`docs/AUDIT_2026-09-08_scientific_validity.md`, added to the
  repository on 2026-09-14; the project's answer to each of its items, written
  on 2026-10-02, is `docs/AUDIT_2026-09-08_response.md`). Its section D-3 objects that every comparison
  between configurations since 2026-08-29 had been decided on the test
  engines, because nothing was held back to decide it on; its section D-9,
  that the cost of capping the test labels had not been re-measured on the
  current configuration and that no stored run used
  `config/fd001_uncapped.yaml`. It asks for those comparisons to be re-run
  holding engines back, and does not itself propose changing the default.
  Its statement that
  no run since 2026-08-28 had used `config/fd001_validated.yaml` stops being
  true at 13:59 that day, which fits its date.
- **2026-09-08, 12:30.** Commit `9856794` still sets 0.0, in both
  `config/fd001.yaml` and `config/fd001_uncapped.yaml`.
- **2026-09-08, 14:13.** A stored run of `config/fd001_uncapped.yaml` holds
  engines back. Its settings match that file as `ee26e88` later committed it
  (0.2), not as `9856794` had it (0.0), and no command-line option of the
  training script changes the fraction held back, so the edit was already in
  the working tree. The run that held engines back at 13:59 does not show the
  change: its settings are those of `config/fd001_validated.yaml`, a separate
  file that held 20 % back throughout the removal.
- **2026-09-08, 22:29** (00:29 on 2026-09-09 in local time). The change was
  committed as `ee26e88`, whose message says its edits "were already in the
  working tree". The comments it adds say 2026-09-08.
- **2026-09-09, 07:15.** This entry was brought up to date in `8ef7b4c`, whose
  message names `ee26e88` and 2026-09-02 together; the same commit wrote that
  date into this log, and from here it spread to the code comments, the
  READMEs and the tables added on 2026-09-14.

So 2026-09-08 is the date of the only recorded reason, of the change in the
working tree and, in UTC, of the commit; only in local time does the commit
fall on 2026-09-09. No record of the decision itself was found apart from the
change. Found on 2026-10-02 while the quotation of assumption A-21's figure
in `config/fd001.yaml` was being corrected, which set that file's 2026-09-08
beside this entry's date. The heading below
now names the arrangement it describes, holding nothing back from 2026-08-29
to 2026-09-08, rather than the time its text stood in this log, which ran to
`8ef7b4c` on 2026-09-09.

---

### What this entry said while nothing was held back (2026-08-29 to 2026-09-08)

**Superseded 2026-08-29 by A-22.** This entry originally held out 20 % of the
training engines unconditionally. Table 6 states 200 epochs, and holding
engines out in order to restore the best-validation checkpoint is
validation-based model selection, which the paper never describes; under A-22's
rule the paper wins. `config/fd001.yaml` therefore held nothing out, until
the reversal recorded above put the split back on 2026-09-08.

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
place nearly identical examples on both sides, and the validation error would
stop measuring performance on engines the model never saw. The usual effect of
leakage of this kind is a lower, flattering error, but nothing guarantees that
direction, and this project has never built a window-level split to measure
it. (Until 2026-10-02 the sentence ended "give a falsely low validation
error", stating the usual direction as certain: the same kind of claim the
correction to A-25 removed. Found by the search for claims of this kind after
the review of the playground's information cards on 2026-09-26.
`src/thesis_gnn_rul_xai/data/windowing/README.md` gives the argument and its
source.)

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
raw window or runs a temporal encoder first and passes its **output vector**.
At least two put fewer than thirty numbers on a node: 18 on a DVGTformer node
standing for one time step, and a raw window of 20 cycles in IMTSO-GCN, both on
FD002. (Until 2026-10-02 this sentence said the smallest was thirty numbers
wide; F-12 records the correction.) Neither the last reading nor the mean
appears anywhere in that corpus. So the choice made here is not the
field's default — it is a consequence of refusing to depart from the shape
Table 4 prints, and the honest way to describe it is "unusual, and adopted for
a stated reason", not "natural". The nearest comparable design, HAGCN, gives
each of the same fourteen sensor nodes thirty numbers and lets its recurrent
stage widen them to sixty; the source paper's stage has to narrow thirty to
one.

**The mean was not measured until 2026-09-29** (the correction below has the
result; until 2026-10-02 this sentence said "The mean has still never been
measured"). The figures above compare this
choice against the removed `window` mode, not against the other option that
keeps Table 4 intact. Averaging the window instead of taking its last cycle
leaves `W(1) in R^(1 x 32)` and the parameter count untouched, so it is a
one-line change and a five-seed run. Until it is run, "how much it matters"
answers a different question from the one this entry poses.

**Note on what `window_size` still does.** It is not inert. It decides which
cycles are eligible — an engine's first example ends at cycle `window_size`,
so cycles 1 to 29 are never a prediction point — and it is what a temporal
encoder would consume if one were added later.

**Correction, 2026-10-02: the mean has been measured.** "The mean has still
never been measured", above, stopped being true on 2026-09-29. The sensitivity
study (finding F-19, `reports/sensitivity_fd65a3b-dirty_20260929T101059Z`, at
`fd65a3b-dirty`) trained both one-number stages from the same tree, with every
other setting at the default: z-score scaling, row rescaling, a fifth of the
engines held back with early stopping, the single drop in learning rate.
Paired by seed:

| What each sensor node carries | Test RMSE | NASA score | Held-back RMSE |
|---|---|---|---|
| its last reading (this entry) | 16.690 ± 0.229 | 572.6 ± 35.9 | 21.142 ± 0.817 |
| the average of its 30 readings | 22.434 ± 0.516 | 9973.3 ± 2462.4 | 21.725 ± 1.939 |
| average minus last | +5.744, p = 0.000019 | +9400.8, p = 0.00099 | +0.583, p = 0.399 |

On the test engines the last reading is better on all five seeds
(`[4.994, 6.269, 6.112, 5.994, 5.351]`); on the held-back engines the two cannot
be told apart. The last-reading arm here, 16.690 ± 0.229, is not the stored
default's 16.496 ± 0.415 under the same settings: the study's runs record
PyTorch 2.5.1 on a graphics card, the stored default PyTorch 2.13.0 on the
processor. Found while checking the audit of 2026-09-08 (its item D-6) against
this entry.

The two runs also come from different commits, which the paragraph above
first left out (added the same day): the study's arm at `fd65a3b-dirty`, whose
only uncommitted file was the study's own script, and the stored default
(`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z`)
at `5602773-dirty`. What makes the code an unlikely cause is a third run: the default trained
again on the processor under PyTorch 2.13.0 from a clean checkout of `f4f40ad`,
a commit after `fd65a3b`
(`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z`),
gives the stored default's five results exactly. So the code at a commit
before the study's and at one after it trains the same networks on the
processor, and that run's settings files match the study's arm except for the
device. The likeliest cause is the device and the PyTorch version together; no
stored run separates the two, and none was trained at `fd65a3b` itself on the
processor.

**What reaches the network from the window (added 2026-10-02, from the same
audit item).** With the smoothing weight of 0.5 (assumption A-04), the smoothed
value at a cycle is 0.5 times that cycle's reading, plus 0.25 times the one
before, plus 0.125 times the one before that, and so on: the last six readings
carry 0.984 of the weight between them, and the thirtieth carries 9.3e-10. On the
17,731 rows of the 100 FD001 training engines that can end a 30-cycle window,
the raw reading at that cycle alone accounts for this share of the smoothed
value's spread (the squared correlation between the two, which rescaling does
not change):

| T24 | T30 | T50 | P15 | P30 | Nf | Nc |
|---|---|---|---|---|---|---|
| 0.885 | 0.867 | 0.936 | 0.750 | 0.932 | 0.942 | 0.990 |

| Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|
| 0.953 | 0.946 | 0.941 | 0.909 | 0.881 | 0.901 | 0.902 |

So the default network reads lightly smoothed current readings, one per sensor.
The 30-cycle window decides which cycles can be prediction points and enters
the number itself only through the smoothing. "Spatiotemporal" and "a 30-cycle
window" describe the paper's model; wherever this configuration is described,
that has to be said beside them. The window-summary stage (assumption A-28) is
the measured alternative that gives the network the window's trend.

```python
import numpy as np
from thesis_gnn_rul_xai.data.loading import read_cmapss_frame
from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_SYMBOLS, PAPER_SENSOR_COLUMNS
from thesis_gnn_rul_xai.data.smoothing import ewma_series

frame = read_cmapss_frame("data/raw/train_FD001.txt")
for symbol, column in zip(PAPER_SENSOR_SYMBOLS, PAPER_SENSOR_COLUMNS):
    raw, smooth = [], []
    for _, engine in frame.groupby("unit"):
        values = engine.sort_values("cycle")[column].to_numpy(float)
        raw.append(values[29:]); smooth.append(ewma_series(values, 0.5)[29:])
    print(symbol, round(np.corrcoef(np.concatenate(raw), np.concatenate(smooth))[0, 1] ** 2, 3))
```

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

**Correction, 2026-10-02.** Wen et al. do not cite Kipf and Welling: the name
appears nowhere in the paper's text, and its reference list (pages 14–16 of
the copy in `docs/references/`) has no entry for them. Found while writing the
GCN page under Resources. Kipf and Welling's rule (their Eq. 2) does have no
bias, so it remains a precedent for the reading above, but it is not "the layer
the paper is citing"; the reading rests on Wen et al.'s own Equations 8–10.

**How much it matters.** Small. A bias per output channel would add 128
parameters to a network of about 62 000. `GraphConvolution` takes `bias=True`
so the difference can be measured rather than argued about.

**What having no offset does, and the one measurement of adding one (added
2026-10-02).** "Small" above counts parameters, not what the layers can
compute. With no offset and one number per sensor, the three graph layers can
only scale with their input, and after the first layer each sensor's 32
numbers come from one number (finding F-21, which the audit of 2026-09-08
raised as its item D-7). An offset would end the first of those, and with it
the two fixed directions the second takes, because `ReLU(s · w + b)` is 32
differently placed bent lines in `s` rather than one pattern scaled up and
down. It would not end the second itself: the first layer still works on one
number per sensor, which comes from Table 4's input shape, not from the
missing offset (corrected the same day, 2026-10-02, from "An offset is the
change that would lift that limit"; finding F-21 sets out both halves).
It has been trained once, in the sensitivity study (finding F-19,
`reports/sensitivity_fd65a3b-dirty_20260929T101059Z`, at `fd65a3b-dirty`),
around the reference with the paper's recurrent stage: test RMSE
13.516 ± 0.613 with offsets against 13.234 ± 0.723 without, +0.282, paired
p = 0.648; held-back RMSE 13.931 ± 0.478 against 14.184 ± 0.389, −0.253,
p = 0.105, lower on four of five seeds. Neither difference survives the
spread. It has not been trained with the default's last-reading stage, so
what an offset is worth to the default network is not known.

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

**Correction, 2026-10-02.** The second sentence above claims more than Glorot
and Bengio derived. Their argument assumes activation functions that behave
linearly near zero, and these graph layers apply ReLU, which sets every
negative value to zero. He et al. (2015, page 1029 of the copy in
`docs/references/`) say of exactly this derivation that its linearity
assumption "is invalid for ReLU and PReLU", and derive a larger starting spread
for ReLU layers. The choice stands, because it is the one the GCN paper names
for this layer (their Section 5.2), but the reason it keeps the activations'
spread constant through three ReLU layers is not established. Found while
collecting the cited papers. PyTorch's own default for the dense layers, kept
above, is also not He et al.'s distribution despite its name: PyTorch's source
states it equals drawing evenly from ±1/√(number of inputs).

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

**Correction, 2026-10-02.** The sentence above saying the challenge's published
function "also says the opposite" is wrong about its printed equation. The
dataset paper's Eq. (11) prints `a1 = 10` for early and `a2 = 13` for late,
which agrees with Wen et al.'s equations read literally. Only that paper's
sentence and its Figure 9 support the convention chosen here (finding F-20).
The choice stands; the evidence for it is the two papers' words and one figure,
against two printed equations.

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

And no paper on this benchmark decays it exponentially. Of the three papers the
literature search found that state a schedule at all, two work on C-MAPSS and
both step at fixed epochs; the third, Li and Dong (2026), reduces the rate when
the loss stops falling, but it works on N-CMAPSS, a newer and larger simulated
engine dataset, and on nothing else. All three are quoted with page numbers in
`docs/LITERATURE_BASIS.md`. Li, Zhao, Sun, Yan and Chen (2021) write on their
page 6 that "learning rate decay is adopted, which decreases the learning rate
by **multiplying 0.1 in epochs 120 and 160**".

**Correction, 2026-10-02.** Until this date the paragraph above said "three
C-MAPSS papers" state a schedule. Li and Dong (2026) say they chose N-CMAPSS as
their data (PDF page 3, journal page 400; the copy in `docs/references/` has a
publisher's cover sheet in front, so its PDF page numbers are not the
journal's) and report results on no other, so only two of the three are on
this benchmark, and both of those step. Found when the cited papers were
collected into `docs/references/` and checked against the places that cite
them. The argument for a step is unchanged; the only precedent found for a
reduction on a plateau is on a different dataset.

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

**Held-back error (added 2026-10-02).** The table above held no engines back,
so it has no held-back error, and the choice was not made on a measured
difference at all. The audit of 2026-09-08 asked for choices between
configurations to be judged on held-back engines (its item D-3); two stored
comparisons that held a fifth of the engines back say the same as the table.
Around the recurrent-stage reference of the sensitivity study (finding F-19,
`reports/sensitivity_fd65a3b-dirty_20260929T101059Z`, at `fd65a3b-dirty`),
exponential minus step is +0.422 on test RMSE (13.656 ± 0.172 against
13.234 ± 0.723, p = 0.232) and +0.019 on held-back RMSE (14.203 ± 0.471
against 14.184 ± 0.389, p = 0.931). With the last reading, z-score scaling and
row rescaling, a fifth of the engines held back with early stopping, runs
`fd001_physics_row_cap_5seed_varsplit_exponential_cfgba82e3_f4f40ad_20261002T152750Z`
(exponential) and
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z`
(step), both trained on 2026-10-02 from a clean checkout of `f4f40ad`, give
+0.119 on test RMSE (16.616 ± 0.289 against 16.496 ± 0.415, p = 0.632) and
−0.014 on held-back RMSE (21.154 ± 0.868 against 21.168 ± 0.870, p = 0.758).
(Corrected the same day: this note first named
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_94617bf-dirty_20260828T210918Z`
and `fd001_physics_row_cap_5seed_varsplit_cfgba82e3_9856794-dirty_20260908T135935Z`,
a pair that crosses the commits of 2026-08-28 to 2026-09-08 and so was called
the weaker of the two. The clean runs above reproduce those two seed for seed,
so the figures stand and the pair no longer crosses a commit.) On held-back
error too the two schedules cannot be told apart, so the choice comes out as it
did: on the argument.

---

## A-18 — Early stopping ends training when the held-out error stops improving

**Live again since 2026-09-08, with the split of A-06.** Early stopping was
never switched off in code — `training/config.py` has held
`early_stopping_patience = 20` throughout — so it was disabled only in the
sense that `validation_fraction: 0.0` left it nothing to measure. With 20 %
of the engines held back it fires again, and in the five-seed run recorded
under A-06 it fired at epochs 51, 55, 8, 18 and 35 of a nominal 200.
**Table 6's "200 epochs" is therefore not what the default configuration
runs**, and any number produced by it must be described that way.

**Correction, 2026-10-02.** Until this date the first line above and the
heading below gave 2026-09-02; the split, and with it early stopping, came
back on 2026-09-08. The correction under A-06 sets out the history and how the
wrong date was found. The heading below now names the arrangement it
describes rather than the time its text stood in this log, which ran to
2026-09-09.

---

### What this entry said while nothing was held back (2026-08-29 to 2026-09-08)

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

**The averaging, measured (added 2026-10-02, the audit's item D-2).** The
correction above says every sensor receives the same average once the rows are
rescaled. This is the measurement behind it. Joining every sensor to every
other, and each to itself, gives a 14 × 14 table of ones; both rescalings then
put 1/14 = 0.071429 in every entry, so all fourteen rows are the same.
Multiplying the table by any fourteen readings hands every sensor their
average: the readings 1, 2, …, 14 come out as 7.5, fourteen times. The rank of
a table is how many of its rows cannot be built by adding up multiples of the
others:

| Graph | Rescaling | Distinct rows | Rank |
|---|---|---|---|
| the paper's (Table 3) | row | 9 | 8 |
| the paper's (Table 3) | symmetric | 9 | 8 |
| every sensor joined to every other | row | 1 | 1 |
| every sensor joined to every other | symmetric | 1 | 1 |

The paper's 8 is finding F-18. On a trained network: in seed 42 of
`fd001_complete_row_cap_5seed_varsplit_cfgf21547_f386291-dirty_20260914T150527Z`,
over the 100 test windows, every sensor holds exactly the same numbers after
each of the three graph layers (largest difference between any two sensors 0),
and shuffling which sensor's window goes to which node, or moving 1.0 of
rescaled reading from one sensor to another, changes no prediction by more than
1.53e-05 cycles (exactly 2 to the power −16), the rounding of 32-bit
arithmetic. (This said "1.5e-05" until it was recomputed the same day; the
largest change is 1.52587890625e-05 in both cases.) So this control hands the
network one number, the average, where the paper's graph hands it eight.

That collapse costs 2.784 cycles when each sensor carries its last reading
(finding F-04), but not measurably around the recurrent-stage reference of the
sensitivity study (finding F-19, at `fd65a3b-dirty`): +0.402 on test RMSE
(13.636 ± 0.601 against 13.234 ± 0.723, p = 0.372) and +0.112 on held-back RMSE
(p = 0.221). Why the two differ has not been examined.

The controls the audit proposed instead — random graphs with 41 edges, and
rewirings of the paper's graph that keep each sensor's number of connections —
have never been run in this repository. `scripts/train.py --graph` offers the
paper's graph, every sensor joined, and the 41 strongest correlations
(assumption A-27); the audit's own three-seed figures for random graphs were
measured outside the repository and name no commit here. Running them needs a
new graph option and new training.

```python
import numpy as np
from thesis_gnn_rul_xai.graph.adjacency import (
    build_fully_connected_adjacency, build_physics_adjacency, normalize_adjacency)

for name, raw in (("physics", build_physics_adjacency()[0]),
                  ("complete", build_fully_connected_adjacency()[0])):
    for method in ("row", "symmetric"):
        table = normalize_adjacency(raw, method=method).astype(np.float64)
        rows = {tuple(np.round(row, 12)) for row in table}
        print(name, method, len(rows), np.linalg.matrix_rank(table))
```

The trained network, and the recurrent-stage pair (added 2026-10-02):

```python
import numpy as np, pandas as pd, torch
from scipy import stats
from thesis_gnn_rul_xai.training.stored import read_run, load_seed

model, prepared = load_seed(read_run(
    "fd001_complete_row_cap_5seed_varsplit_cfgf21547_f386291-dirty_20260914T150527Z"), 42)
x = torch.from_numpy(prepared.test.features.astype(np.float32))      # (100 engines, 14 sensors, 30 cycles)
with torch.no_grad():
    hidden = model.temporal(x)
    for layer in model.graph_layers:                                  # 0.0: every sensor holds the same numbers
        hidden = torch.relu(layer(hidden, model.adjacency))
        print(float((hidden - hidden[:, :1]).abs().max()))
    before = model.predict_cycles(x)
    shuffled = x[:, torch.randperm(14, generator=torch.Generator().manual_seed(0))]
    moved = x.clone(); moved[:, 0, -1] += 1.0; moved[:, 1, -1] -= 1.0  # 1.0 from T30's last reading to T24's
    print(float((model.predict_cycles(shuffled) - before).abs().max()),
          float((model.predict_cycles(moved) - before).abs().max()))

study = "reports/sensitivity_fd65a3b-dirty_20260929T101059Z/runs/FD001/"
a, b = (pd.read_csv(study + arm + "/results.csv") for arm in ("graph_complete", "reference"))
for column in ("rmse", "validation_rmse"):
    print(column, round(a[column].mean(), 3), round(a[column].std(), 3), round(b[column].mean(), 3),
          round(b[column].std(), 3), round(float((a[column] - b[column]).mean()), 3),
          round(float(stats.ttest_rel(a[column], b[column]).pvalue), 3))
```

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
| z-score, row, nothing held back, **step** schedule (`bc20edd-dirty`) — the default until 2026-09-08 (date corrected 2026-10-02 from 2026-09-02; see A-06) | 16.818 ± 0.157 | 19.214 ± 0.056 | +2.396 | 0.000002 |
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

**Held-back error beside test error (added 2026-10-02).** The audit of
2026-09-08 asked for comparisons between configurations to be judged on the
held-back engines (its item D-3). Three of the configurations here held a fifth
of the engines back, so their held-back error exists and pairs by seed; the two
with nothing held back have none. Cost of joining every sensor to every other:

| Configuration | On test RMSE | On held-back RMSE |
|---|---|---|
| current default (the runs at `f386291-dirty` above) | +2.784, p = 0.00046 | +1.447 (22.615 ± 0.620 against 21.168 ± 0.870), p = 0.0041, all five seeds; on summaries no pass was chosen by (assumption A-27), +1.172, p = 0.0278 on the mean of the last 20 passes and +0.966, p = 0.0184 on the mean of passes 1 to 23, still all five seeds |
| min-max, symmetric (`cdcc827-dirty`) | +0.044, p = 0.913 | +0.539 (23.611 ± 0.899 against 23.072 ± 1.096), p = 0.040, four seeds of five |
| recurrent-stage reference of the sensitivity study (finding F-19, `fd65a3b-dirty`) | +0.402, p = 0.372 | +0.112, p = 0.221 |

The min-max row is the one that moves. On the test engines the every-sensor
graph cost nothing measurable there; on the held-back engines it was worse on
four seeds of five, per seed `[−0.019, +0.532, +0.383, +0.745, +1.056]`. So
the "zero" in this entry's history is a result on the test engines, and on
held-back error the direction did not reverse between min-max and z-score
scaling. Two cautions keep this from being a reversal of its own: the held-back
error a run records is the lowest reached over all passes and also chose which
pass to keep, so it flatters, by an amount that can differ between the two
graphs and that the min-max runs, which saved no error after each pass, cannot
show (for the current default's graphs it was measured under assumption A-27,
at 0.2 to 0.7 cycles on average, by amounts five seeds could not tell apart);
and p = 0.040 is one of several tests in this note. The min-max runs are
`fd001_complete_symmetric_5seed_varsplit_cdcc827-dirty_20260828T144045Z` and
`fd001_physics_symmetric_5seed_varsplit_cdcc827-dirty_20260828T143735Z`.

Every p-value in this entry is a paired test across seeds that are all scored
on the same 100 test engines. It says whether a difference could be an
accident of how the networks started, and, where engines were held back, of
which ones; it does not say how the difference would come out on engines
outside this data (the audit's item D-8).

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

**Read this first (added 2026-10-02).** The table below, and the 20.353 and
5.42-cycle shortfall that follow it, describe the configuration of 2026-08-28:
per-condition min-max scaling, symmetric rescaling of the graph, the
exponential learning-rate schedule, a fifth of the engines held back. It was
called "current" and "the default" when written, and has been superseded
since: from 2026-08-29 by z-score scaling and row rescaling with nothing held
back, under the step schedule from 2026-09-01 (16.818 ± 0.157, the section
headed "Superseded by A-22" below), and from 2026-09-08 by holding a fifth of
the engines back again (assumption A-06).
Today's default gives **16.496 ± 0.415**, 1.57 above the paper's 14.93
(`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z`;
the passage "the figure since 2026-09-08" below). The audit of 2026-09-08 lists
figures presented as current after they stopped being so as its item D-9. The
older text is kept as it was, because the comparisons in it were measured on
that configuration.

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

**Added 2026-10-02: the figure since 2026-09-08.** 16.818 ± 0.157 was the
default's figure while nothing was held back, from 2026-08-29 to 2026-09-08.
Since then the default holds a fifth of the training engines back
(assumption A-06) and gives **16.496 ± 0.415**
(`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z`),
1.57 above 14.93. The paragraph above said "now" of a figure that had stopped
being current; the audit of 2026-09-08 lists numbers of that kind as its item
D-9.

**What 100 test engines can resolve (added 2026-10-02, the audit's item D-8).**
Every ± in this entry is the spread over five seeds, and the five seeds are all
scored on the same 100 test engines, so it is not the uncertainty a reader
usually means by it. Resampling those 100 engines 20,000 times with
replacement, and recomputing the five-seed mean RMSE of the run above each
time from its stored predictions, gives a 95 % interval of **[13.98, 18.91]**
and a standard deviation of 1.262, three times the seed spread of 0.415. The
paper's 14.93 lies inside the interval: 11.7 % of the resampled means are at or
below it. So the 1.57-cycle gap describes these 100 engines and these five
networks; the test set is too small to establish it as a difference between
this re-implementation and the paper's model. The paper gives its own figure
with no spread, so its uncertainty over the same 100 engines is not reported.

```python
import numpy as np, pandas as pd

run = "reports/fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z"
files = [pd.read_csv(f"{run}/seed_{s}/test_predictions.csv").sort_values("unit") for s in range(42, 47)]
predicted = np.array([f["predicted"].to_numpy(float) for f in files])   # (5 seeds, 100 engines)
actual = files[0]["rul"].to_numpy(float)
draws = np.random.default_rng(20261002).integers(0, 100, size=(20000, 100))
means = np.sqrt(((predicted[:, draws] - actual[draws]) ** 2).mean(axis=2)).mean(axis=0)
print(np.percentile(means, [2.5, 97.5]).round(2), means.std(ddof=1).round(3), (means <= 14.93).mean().round(4))
```

#### The architecture-faithful comparison is 20.35 against 14.93

Run B is the paper's printed architecture: input `[batch, 14, 1]`,
`W(1) in R^(1 x 32)`, the same layer widths, and **61,729 parameters — what
Table 4's five weight shapes (61,600 numbers), the two dense-layer offsets
Equations 12 and 13 print (129) and no offset in the graph layers
(assumption A-12) add up to**. (Corrected 2026-10-02 from "61,729 parameters —
what Table 4's count exactly", one of the wordings the audit of 2026-08-31
meant to remove; the audit of 2026-09-08 found it still here, its item D-9.
Assumption A-07 sets out the arithmetic.) The only thing removed is the LSTM,
which is what "w/o LSTM" means. So this is the comparison that carries weight:

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
Table 4's shapes, with the offsets of Equations 12 and 13 and none in the graph
layers, imply 61,729 (corrected 2026-10-02 from "Table 4's shapes imply 61,729";
the shapes alone give 61,600). Run A had 62,657. That single number would have caught
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
| z-score, **nothing held back**, **step** schedule — the default until 2026-09-08 (date corrected 2026-10-02 from 2026-09-02; see A-06) | **16.818 ± 0.157** | 17.071 ± 0.325 | **row by 0.253** | −2.478 | 0.068 |
| the same under the superseded **exponential** schedule | 16.694 ± 0.331 | 17.045 ± 0.201 | row by 0.352 | −2.830 | 0.047 |
| z-score, 20 % held back, early stopping, **step** schedule — today's `config/fd001.yaml` (`f4f40ad`) | 16.496 ± 0.415 | 16.599 ± 0.494 | row by 0.102 | −0.362 | 0.735 |
| z-score, 20 % held back, no early stopping, **step** schedule (`f4f40ad`) | 17.003 ± 0.324 | 16.804 ± 0.463 | symmetric by 0.199 | +1.260 | 0.276 |
| z-score, 20 % held back, early stopping, superseded **exponential** schedule (`f4f40ad`) | 16.616 ± 0.289 | 16.524 ± 0.339 | symmetric by 0.091 | +0.550 | 0.611 |
| z-score, 20 % held back, no early stopping, superseded **exponential** schedule (`f4f40ad`) | 16.710 ± 0.157 | 16.599 ± 0.275 | symmetric by 0.111 | +0.887 | 0.425 |
| superseded — min-max, 20 % held back (`cdcc827-dirty`) | 19.992 ± 0.416 | 20.353 ± 0.752 | row by 0.360 | −2.160 | 0.097 |

**What that table says.** Row's advantage exists **only when nothing is held
back**, and it is marginal even there. Hold 20 % of the engines back and the
difference vanishes into noise under both schedules, with or without early
stopping (p between 0.276 and 0.735); which form comes out nominally ahead
depends on the schedule and on early stopping, and under today's default it
is row, by 0.102 cycles at p = 0.735. The superseded min-max row points the
same way as the default but does not reach significance.

**Correction, 2026-10-02.** Until this date the table had two rows for 20 %
held back, naming neither a commit nor a schedule, and the second was wrong.
All eight runs behind the four rows above were trained again on 2026-10-02 from
a clean checkout of `f4f40ad`, five seeds each: the runs
`fd001_physics_{row,symmetric}_cap_5seed_varsplit[_noearlystop][_exponential]_cfg{ba82e3,5d7c10}_f4f40ad_20261002T*`.
The old "early stopping" row (16.616 ± 0.289 against 16.524 ± 0.339, p = 0.611)
is reproduced to every printed digit by the **exponential** schedule, so it was
measured under the schedule replaced on 2026-09-01. The old "no early
stopping" row gave row rescaling 16.616 ± 0.289, the same figure as the row
above it: no run of row rescaling without early stopping was stored, so the
row set the early-stopping run beside a symmetric run without it (the held-back
passage below names both). Under the exponential schedule without early
stopping row rescaling gives 16.710 ± 0.157, so that row's difference (0.017)
and p (0.930) compared two different procedures; its symmetric figure,
16.599 ± 0.275, is reproduced exactly. The sentence above also said the difference "changes sign"
whatever early stopping does; under today's step schedule with early stopping
row is nominally ahead. The conclusion, that no difference survives holding
engines back, holds under both schedules. Found when the rows were re-run at
one commit to settle which schedule produced them; the re-run default
reproduces the stored default's predictions exactly on every seed.

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
describe. The 0.091 at p = 0.611 is the **20 %-held-back** configuration under
the exponential schedule, now the fifth row (it was the third when this was
written) — not the default that `config/fd001.yaml` described and that the
disputed comment sits in. Recomputed from the stored runs, `p = 0.047` is
exactly reproducible for the default: 0.0473, two-sided, no one-sided test
needed. The audit's conclusion that A-22's rule now selects symmetric does not
follow.

What the audit was right about is that `config/fd001_validated.yaml` carried
the same comment, and there it *was* wrong: that file holds engines back, so
"under this configuration" the figures were those of a row holding engines
back, not the first. That
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

**Held-back error beside test error (added 2026-10-02).** The comparison that
chose row is the first two rows of the table, which held nothing back, so the
runs that decided it have no held-back error: it was decided on the test
engines, which the audit of 2026-09-08 objected to (its item D-3). Stored runs
that held a fifth of the engines back give both errors, paired by seed:

| Configuration | Test: row minus symmetric | Held-back: row minus symmetric |
|---|---|---|
| z-score, step schedule, early stopping: the default's settings since 2026-09-08 (both runs at `9856794-dirty`; the table's third row, re-run at `f4f40ad`) | −0.102 (16.496 ± 0.415 against 16.599 ± 0.494), p = 0.735 | +0.045 (21.168 ± 0.870 against 21.123 ± 0.791), p = 0.311 |
| z-score, exponential schedule, early stopping (the table's fifth row) | +0.091, p = 0.611 | +0.120, p = 0.376 |
| z-score, step schedule, no early stopping (the table's fourth row, `f4f40ad`) | +0.199, p = 0.276 | −0.014 (21.009 ± 0.856 against 21.023 ± 0.824), p = 0.565 |
| z-score, exponential schedule, no early stopping (the table's sixth row, `f4f40ad`) | +0.111, p = 0.425 | +0.050 (20.975 ± 0.848 against 20.925 ± 0.861), p = 0.178 |
| min-max, exponential schedule (the table's last row, `cdcc827-dirty`) | −0.360, p = 0.097 | −0.226, p = 0.0103, all five seeds |
| the recurrent-stage reference of the sensitivity study (finding F-19, `fd65a3b-dirty`) | +0.332, p = 0.307 | +0.103, p = 0.454 |

Under z-score scaling with engines held back, the held-back error cannot tell
the two forms apart either: in none of the five z-score rows does the
difference come near the seed-to-seed spread (the smallest p is 0.178), and
its direction is mixed, nominally lower for symmetric in four rows and for row
in one, the step schedule without early stopping, by 0.014 cycles. (Corrected
the same day: this sentence first had three z-score rows and said all three
were nominally lower for symmetric; the two rows without early stopping were
added once both forms had been trained under those settings at `f4f40ad`.)
**A choice made on held-back error would not have come out as row**: it would
have been a tie, which leaves the choice to an argument rather than a
measurement. Only under the superseded min-max scaling does the held-back
error favour row, on every seed. And since 2026-09-08 the default holds engines
back, so the configuration the project reports is the first row of this one,
where row has no measured advantage on either error.

The first row's runs are
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_9856794-dirty_20260908T135935Z`
and `fd001_physics_symmetric_cap_5seed_varsplit_cfg5d7c10_9856794-dirty_20260908T140637Z`;
the same comparison re-run at `f4f40ad`, the table's third row, gives the same
five results for each form, seed by seed. The second row's are
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_94617bf-dirty_20260828T210918Z`
(row, exponential schedule, early stopping) and
`fd001_physics_symmetric_cap_5seed_varsplit_09628fa_20260828T153058Z`; the
same pair re-run at `f4f40ad`
(`fd001_physics_row_cap_5seed_varsplit_exponential_cfgba82e3_f4f40ad_20261002T152750Z`
and `fd001_physics_symmetric_cap_5seed_varsplit_exponential_cfg5d7c10_f4f40ad_20261002T181841Z`)
gives the same five results for each form. The two rows without early stopping
are `fd001_physics_row_cap_5seed_varsplit_noearlystop_cfgba82e3_f4f40ad_20261002T170226Z`
against `fd001_physics_symmetric_cap_5seed_varsplit_noearlystop_cfg5d7c10_f4f40ad_20261002T183412Z`
(step), and `fd001_physics_row_cap_5seed_varsplit_noearlystop_exponential_cfgba82e3_f4f40ad_20261002T170506Z`
against `fd001_physics_symmetric_cap_5seed_varsplit_noearlystop_exponential_cfg5d7c10_f4f40ad_20261002T183413Z`
(exponential); with engines held back, "no early stopping" still keeps the pass
with the lowest held-back error (assumption A-18). Until
2026-10-02 the same row run also supplied the row figure of a table row for
engines held back with early stopping off: no run of row rescaling under those
settings was stored, so that row set an early-stopping run beside the
symmetric run without early stopping,
`fd001_physics_symmetric_cap_5seed_varsplit_noearlystop_cfg5d7c10_09628fa-dirty_20260828T160523Z`,
and those two runs give its old +0.094 and 0.930 exactly. That row has been
replaced by runs of both forms without early stopping (the correction under
the table). The p of the first row here said 0.736 until 2026-10-02; it is
0.73546, so 0.735. Each figure here comes from
`compare` above, with `validation_rmse` in place of `rmse` for the held-back
column. The held-back error flatters, by an amount that can differ between the
two arms and was not measured for these runs (where it was measured, for the
three graphs compared under assumption A-27, it was 0.2 to 0.7 cycles on
average, by amounts five seeds could not tell apart), and every p-value is a paired
test across five seeds scored on the same 100 test engines, so it says nothing
about engines outside this data (the audit's item D-8).

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
layer. Neither choice was made by looking at an error figure. (Corrected
2026-10-02: Wen et al. do not cite Kipf and Welling, as the correction under
assumption A-12 records; A-21's first choice followed the paper that introduced
that layer's rescaling, not one Wen et al. point to.)

**What we chose, and on whose instruction.** On 2026-08-29 the author directed
a different rule: *where the paper is silent, choose the option that brings the
result closest to the paper's; where the paper states something, follow the
paper regardless.* Applied assumption by assumption:

| Assumption | What the paper says | Outcome under the rule |
|---|---|---|
| A-03 input scaling | nothing at all | silent → **`zscore`**, 3.83 cycles closer than min-max |
| A-21 adjacency normalisation | "normalized … incorporating self-loops"; no formula | silent → **`row`**, 0.35 closer with nothing held back (p = 0.047) under the exponential schedule then in force; 0.25 (p = 0.068) under the step schedule since 2026-09-01 (A-21's table; schedules named here 2026-10-02) |
| A-06 validation split | describes none | contradicts Table 6's 200 epochs → removed, then **reinstated 2026-09-08** (date corrected 2026-10-02 from 2026-09-02; see A-06); worth 0.32 cycles, p = 0.234 |
| A-18 early stopping | Table 6: 200 epochs | contradicts → removed, then **live again 2026-09-08** with A-06 (date corrected 2026-10-02 from 2026-09-02) |
| A-04 smoothing | Eq. 18 defines it | stated → kept, and removing it is worse anyway |
| A-01 label ceiling | Eq. 19; value inferred from Fig. 7 | evidence-backed → kept at 125 |
| A-17 LR schedule | Table 6: 1e-3 → 1e-4 | mechanism silent → **one x0.1 step**, the only form any C-MAPSS paper uses |

Result: **16.818 ± 0.157** on FD001 against the paper's 14.93, from
20.353 ± 0.752 under the previous defaults. **That figure was measured with
nothing held back.** Since the A-06 reversal of 2026-09-08 the default holds
out 20 % of the engines and gives **16.496 ± 0.415**, a difference the seeds
cannot distinguish from zero (p = 0.234; the table under A-06). (Corrected
2026-10-02: this paragraph and the table above gave the reversal as
2026-09-02, a day on which the held-back setting did not change; the
correction under A-06 has the history.)

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
   On 2026-10-02 the held-back error was set beside the test error for the
   two choices this table made on test error: the z-score comes out the same
   for the default network (assumption A-03), and row rescaling does not, the
   two forms being a tie on held-back error (assumption A-21).
   `docs/AUDIT_2026-09-08_response.md` answers each of the audit's items.
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

**Changed 2026-10-03: the digest no longer counts the notes.** "The whole
preprocessing configuration" above included `notes`, the one field of free
prose, so the digest broke the rule it was built for in the other direction:
runs with identical settings got different names because a sentence differed.
`PreprocessingConfig()` and `config/fd001.yaml` agree on all eleven settings
and digested to `e3a331` and `ba82e3`. Found on 2026-09-09 while the
swappable-stages design was written (`docs/DESIGN_swappable_stages.md`,
sections 0 and 7), and fixed after the author decided on 2026-10-02 to make
that change and to defer the design's four unbuilt steps (steps 5, 6, 8 and 9
of its section 6). (Corrected 2026-10-03: this said "that change and no other
from the design", which the design's own note contradicts, since the author
reversed its fourth guard the same day; found by an independent review of this
change.) The digest is now
`PreprocessingConfig.digest()`, computed from every setting except `notes`
through `settings_without_notes()` in `config.py`, the function `find_runs`,
the playground's comparison of two runs, `scripts/baselines.py`'s check of a
second run and the store of trained networks in `scripts/verify_claims.py`
also use to decide whether two sets of settings are alike. The same day
`PreprocessingConfig` began storing every number as the type its field
declares, because the digest reads the settings as JSON text, where `0` and
`0.0` differ although Python counts them as equal:
`PreprocessingConfig(validation_fraction=0)` gave `e950d9` and
`validation_fraction=0.0` gave `2c9b5b` before that, and both give `2c9b5b`
since. None of the 141 settings files stored under `reports/` that day held a
number of another type, so no stored digest moved. Measured on 2026-10-03:

| Configuration | Digest before | Digest from 2026-10-03 |
|---|---|---|
| `PreprocessingConfig()` | `e3a331` | `818b40` |
| `config/fd001.yaml` | `ba82e3` | `818b40` |
| `config/fd001_uncapped.yaml` | `e49671` | `7c2c29` |

This is a one-time break. Stored runs keep their names, because a stored run
is never edited after it is written, so a run made with the code from now on
carries a different `cfg` part from a stored run with identical settings; a
run trained from a checkout of an older commit still gets the old code. Over
the 50 runs stored in `reports/` on 2026-10-03 whose names carry a digest, the
old rule had filed 10 distinct sets of settings under 16 codes, and none
carried a new code yet; for example the default settings appear as
`cfgba82e3` (15 runs) and as `cfg72f506` (4 runs, whose notes the playground
wrote). `src/thesis_gnn_rul_xai/training/README.md`
lists every old code beside its new one, and
`tests/test_run_records.py::test_a_stored_run_name_carries_the_digest_of_the_settings_it_saved`
checks every stored name against one rule or the other. To gather the runs of
one configuration across the break, compare saved settings, as `find_runs`
does, not names.

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

**Held-back error beside test error, and a correction to the ceiling row
(added 2026-10-02).** Every run in the first table held a fifth of the training
engines back, so each also recorded its error on those engines, which pairs by
seed with the baseline
(`fd001_physics_symmetric_cap_5seed_varsplit_09628fa_20260828T155320Z`). The
audit of 2026-09-08 asked for configurations to be judged on that error (its
item D-3):

| Change from the baseline | Test RMSE | Held-back RMSE |
|---|---|---|
| z-score scaling (`…_09628fa_20260828T153058Z`) | −3.828, p = 0.0005 | −2.038, p = 0.017 |
| no early stopping (`…_noearlystop_09628fa_20260828T152617Z`) | −0.175, p = 0.374 | −0.090, p = 0.374 |
| constant learning rate (`…_constant_09628fa_20260828T154334Z`) | +0.165, p = 0.254 | +0.078, p = 0.159 |
| ceiling of 130 (`…_09628fa_20260828T154003Z`) | +0.469, p = 0.114 | +1.167, p = 0.0001; see below |
| smoothing disabled (`…_09628fa_20260828T153446Z`) | +1.219, p = 0.007 | +1.866, p = 0.0002 |

(Each run name begins `fd001_physics_symmetric_cap_5seed_varsplit`; all are at
`09628fa`.) Every row keeps its direction on the held-back engines. Without
early stopping four seeds of five end on the same pass as the baseline and
give identical errors, so that row rests on one seed.

**The ceiling row changes the answers as well as the model.** The run with a
ceiling of 130 caps its test labels at 130 rather than 125 (assumption A-02
caps them at whatever the ceiling is), and its held-back labels likewise. Of
the 100 FD001 test labels, 3 lie between 126 and 130 and keep their value
instead of becoming 125, and 8 lie above 130 and become 130 instead of 125. Its
+0.469 and +1.167 therefore mix a change in the network with a change in the
answers it is scored against, and neither measures what the ceiling does to
accuracy. The stored run saved no predictions (runs save them from 2026-09-14
on), so it cannot be rescored against the baseline's answers. Found on
2026-10-02 while setting the held-back errors beside this table.

**The "roughly 2.9" above adds two configurations together.** The 1.311 it adds
was measured under min-max scaling (assumption A-02); on today's default
capping is worth 1.508. Neither was measured on the z-score configuration of
this table, so "roughly 2.9" is a sum across configurations, not a measurement
(noted 2026-10-02).

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

**The corpus.** Ten method papers report on C-MAPSS, not all of them on all
four sub-datasets FD001–FD004 (IMTSO-GCN runs only FD002 and FD004); one is a
2025 survey; one (`MAGF-SPA`) uses only N-CMAPSS and is excluded from every
count below; the thirteenth is the source paper itself.

**What each one puts on a node:**

| Paper | Node is | What the node carries into the graph layers | Page |
|---|---|---|---|
| GLSTGN (ESWA 2025) | sensor | output of a gated temporal convolution | 1, 3 |
| IMTSO-GCN (QRE 2025) | a column of the data table, not only a sensor: of the 28 columns its graph is built over on FD002 (Fig. 6), the 19 its filtering leaves connected are 14 sensors, two operating settings, the engine number, the cycle count and the remaining life (RUL) | raw values over a sliding window whose length is tuned between 10 and 31: 20 cycles on FD002, 25 on FD004 (Table 4) | PDF 5, 7, 9–12 (journal 178, 180, 182–185) |
| DVGTformer (MSSP 2024) | two views of one graph: sensor or operating setting (spatial view), time step (temporal view) | the window `Xw in R^(Tw x N)`, 30 x 17 on FD002, mixed by a learned linear map along each axis: a sensor node carries a column, 31 numbers, a time-step node a row, 18 numbers (in each, the one number beyond the window's 30 or 17 belongs to the extra learned node that the other view adds and joins to all its nodes) | 6, 7, 9, 12, 15 |
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
DVGTformer belongs to the second family, though its encoder is only a learned
linear map: its input layer mixes the window along each of its two axes and
keeps its shape. It then puts the result in either way round, a column on each
sensor or operating-setting node and, in its second view, a row on each
time-step node. At least two papers put fewer than 30 numbers on a node. A
DVGTformer time-step node starts as the 17 readings taken at that time step on
FD002 and reaches the graph layers as 18 numbers: the mixing makes each of the
17 depend on every reading in the window, and the input layer then adds an
18th number for the extra learned node that the spatial view joins to every
sensor and operating-setting node. An IMTSO-GCN sensor node carries a window
of 20 cycles on FD002 and 25 on FD004. Neither is one number.

**Correction, 2026-10-02.** Until this date the corpus paragraph said the ten
method papers "report on C-MAPSS FD001–FD004", the table gave DVGTformer's
node as "sensor" only, carrying "the sensor's whole window", which placed it
among the papers that pass the raw window in although its input layer mixes
the window first, and gave IMTSO-GCN's window as "20 in the reported run" on
pages "10, 11", and this paragraph said "The smallest node feature found
anywhere in the corpus is 30 numbers". DVGTformer's Fig. 4 (page 6) and its
text on page 7 make each time step a node of a second, temporal graph; its
Table 4 (page 15) gives the window as 30 time steps by 17 channels on FD002,
which are the 14 sensors and the 3 operating settings (page 12), and 31 by 18
once its input layer (Equations 21 to 24, page 9) has mixed the window along
both axes and added one extra learned node to each view; in that copy the PDF and printed page
numbers are the same. IMTSO-GCN's Table 4 (PDF page 12, journal page 185)
gives a window of 20 on FD002 and 25 on FD004, the only two sub-datasets the
paper runs (PDF page 9, journal page 182); the pages cited were the PDF's,
whose page 1 is the journal's page 174, so the old "10, 11" are the journal's
183 and 184. Both tables were rendered and read as images. Found when the
cited papers were collected into `docs/references/` and checked against the
places that cite them. The finding itself stands: no paper here gives a node
one number.

**Correction, 2026-10-02.** Until this date the table gave IMTSO-GCN's node as
"sensor". The paper does say its nodes "are organized by the sensors of the
engine" (PDF page 5, journal page 178), but it later feeds "all raw measured
parameters" of FD002 and FD004 into the graph as node features (PDF page 11,
journal page 184), and the matrix its graph is built from, drawn for FD002 in
Fig. 6 (PDF page 12, journal page 185), has 28 rows and columns: the three
operating settings, the 21 sensors, the engine number (`id`), the cycle count
(`cycle`), and two columns that are not among the 26 of NASA's files
(`docs/references/nasa_2008_cmapss_readme.txt`), `fault` and `RUL`, the
second being the remaining life the model predicts. Each cell of that matrix
holds the mutual information of two columns: a number measuring how much
knowing one column's values tells you about the other's, which is zero when
the two are unrelated. The paper uses each value as the weight of the
connection between the two columns (PDF page 11, journal page 184), so a cell
at zero is no connection. Its filtering
step removes connections according to their mutual information, by a setting
the tuning chooses (`Thr`, 0.2 on FD002 in Table 4 on the same page; the step
is described on PDF pages 6 and 7, journal pages 179 and 180). In Fig. 6B, the
matrix after filtering, 19 of the 28 keep at least one connection:
`setting_1`, `setting_2`, the fourteen sensors that HAGCN, AGCTE and the
survey keep (the table further down; so `P15`, sensor 6, is among those cut
off), `id`, `cycle` and `RUL`, which stays connected to nine sensors. The other
nine, `setting_3`, sensors 1, 5, 6, 10, 16, 18 and 19, and `fault`, are left
with no connection at all, not even to themselves; the text names six of them
as "redundant connected information" the filtering removes. The paper does not
say whether a column cut off in this way also leaves the input, or whether
`RUL` was ever an input. Fig. 6B was rendered at eight times its printed size
and the colour of every cell compared with the zero end of its colour bar.
Found when the cited papers were collected into `docs/references/` and checked
against the places that cite them. The finding itself stands: a sensor node
there still carries a window, not one number.

**Three questions about IMTSO-GCN's results, added 2026-10-02.** Reading
Fig. 6 for the correction above raised three questions that the rest of the
paper had to answer. The whole paper was read in the copy, and its Figs. 2, 3,
6 and 7 and Tables 9 and 10 were rendered at eight times their printed size
(PDF pages 7, 8, 12, 13 and 14, journal pages 180, 181, 185, 186 and 187).
None of the three changes this finding, since a sensor node there carries a
window either way. They bear on the paper's published results, none of which
this project sets beside its own.

*Is the remaining life an input?* The paper never says, and its pages point
both ways. Against: its nodes "are organized by the sensors of the engine"
(PDF page 5, journal page 178), and it introduces the remaining life only as
the label, cut off at 125 (PDF page 9, journal page 182). For: it feeds "all
raw measured parameters" in as node features (PDF page 11, journal page 184);
it calls the 28 columns of Fig. 6A, `RUL` among them, "the original
parameters"; and it presents Fig. 6B as "the results for the adjacency matrix
optimized by IMTSO-GCN" (both PDF page 12, journal page 185). An adjacency
matrix is the table of which nodes the graph layers let exchange values. In
Fig. 6B `RUL` keeps a connection to itself and to sensors 2, 3, 4, 7, 9, 12,
14, 15 and 21, so as drawn it is a node, and the threshold that left it
connected is one of the settings the paper's search chose (`Thr`, its Table 4).
Whether it was then taken out of the input, the paper does not say.

If it was an input, the network was handed the answer to within one
subtraction, because below the cap of 125 the remaining life falls by exactly
one per cycle. NASA's test record of FD002 engine 74 is 103 cycles long, and
NASA's answer file gives it 86 cycles left at the last of them. A window of
20 cycles, the length its Table 4 gives for FD002, over cycles 83 to 102 then
holds the remaining lives 106, 105, …, 88, 87, and the value to predict at
cycle 103 is 86: the last of them less one. A
window of twenty 125s is followed by 125 at every cycle but the one where the
decline starts (cycle 65 for this engine, where it is 124). The error of such a
network would say nothing about predicting from sensor readings. For the test
engines there is a second consequence. NASA's test files hold 26 columns, the
engine number, the cycle count, three settings and 21 sensors, and no
remaining life; for a test engine it exists only by adding the answer file's
value to the cycles still to come. A remaining-life input on the test set
would therefore mean the answer file was in the input. The paper's FD002 RMSE
of 2.8577 (its Tables 5 and 9) rests on which reading is true. The paper also
never says how many predictions that RMSE averages over. Its Eq. 26 (PDF page
9, journal page 182) sums over K predictions without defining K, and its Fig. 7
draws a prediction at every cycle of each test engine (next paragraph), not
one per engine, so even apart from the remaining life the 2.8577 cannot be set
beside a number computed either way. That holds for the paper this project
re-implements too: Wen et al. report 16.21 for their own method on FD002, and
no row of their comparison table goes below 16.11 (Wen et al.'s Table 8, PDF
page 10), but with K undefined the gap between those numbers and 2.8577
cannot by itself be read as evidence of anything, a remaining-life input
included.

*Where the search scored its candidates.* The paper's swarm search keeps a
population of candidate settings (the window length, the learning rate, the
number of neurons, the number of graph layers and the threshold above), trains
and scores each, and moves them round after round towards the best-scoring
ones. Its scoring step (Step 2, PDF page 7, journal page 180) reads: "A trained
model is obtained by training the training set according to the
hyperparameters in U. The results obtained from the trained model are compared
with the real results. In this paper, R2 is used as the fitness function." It
never says which data "the real results" belong to. The paper names two sets
and no third (PDF pages 8 and 9, journal pages 181 and 182): "the GCNbest is
trained in training set, the testing set is adopted to achieve the RUL
prediction". It uses "validate" and "verify" only for showing that the method
works (PDF pages 1, 3, 9 and 13), never for engines held back. Its Fig. 3 (PDF
page 8, journal page 181) draws one arrow into the search and a second into its
"Error analysis", and prints a box labelled "Training sets" beside each. Both
arrows are light grey. The box beside the first is filled in the green of the
figure's own "Training sets" box, and the box beside the second in the orange
of its "Testing sets" box; the colours were sampled from the page rendered at
eight times its printed size.

Its Fig. 7 shows what the testing set is. Each of its nine panels, FD002
engines 7, 10, 20, 30, 55, 74, 100, 120 and 250, draws the true remaining life
at 125 and then falling by one per cycle. The figure is a single picture
embedded in the page, 1700 by 1190 pixels, and was measured at that size. Each
panel's two scales were set from its frame, whose edges are the ends of its
axes; checked against the panel's dashed gridlines and tick marks, every
gridline lands within 0.2 of a cycle of its labelled value and every tick
within 0.1 of its own. A straight line falling by exactly one per cycle was
then fitted to the blue "True" curve: to the pixels whose colour lies within a
distance of 10 of the legend's blue (red, green and blue values 74, 114 and
176), and whose height on the remaining-life axis lies between 1.5 above the
frame's lower edge and 121, which leaves out the bend at 125. The legend box,
which carries a blue sample of its own, was left out as well. In all
nine panels, the point where that line meets 125, read off the horizontal
axis, is the last cycle at which NASA's test record and answer file put that
engine's remaining life at 125, less 21, to within half a cycle: within 0.1 in
eight panels, and 0.40 in panel D. Widening the colour distance to 15 or 25,
or ending the fitted stretch at 118 instead of 121, keeps every panel but D
within 0.2 and moves D's gap to at most 0.45. A line whose slope is fitted as
well comes out falling by 0.99 to 1.07 per cycle, but where it meets 125 is
less steady, because in panels D and I most of the blue curve is hidden under
the other four: across the same choices, and whether the fit minimises the
pixels' vertical or their horizontal distance from the line, its largest gap
ranges from 0.22 to 0.74 of a cycle, and at 0.74, in panel I, it lies nearer
the last cycle at 125 less 20 than less 21. So the axis counts from the 21st
cycle: one more than the 20-cycle window, and Fig. 3's sketch of the window
marks the step after it "Predict". Read against NASA's training file instead,
the engines with the same numbers come close in only one panel of the nine:
engine 20, about one cycle off (0.94 to 1.25 across all the fits above); the
other eight are at least 15 cycles off. In panel F the curves stop inside the
frame. Their rightmost coloured pixel lies at 82.5 on its horizontal axis, and
the lines are drawn about one cycle thick there, so their last point is at 82:
engine 74's 103 test cycles less the same 21. The blue curve is hidden under
the others over its last stretch, but the fitted line reaches 85.95 at 82, the
answer file's 86. So FD002's testing set is NASA's test file, predicted at
every cycle. Which data the search's candidates were scored on, the paper does
not say. Its text leaves open the training engines themselves (as Fig. 3's
label suggests), a part of them held back without mention, or that testing
set, the only other data it describes; the last would be consistent with
choosing the settings on the test set. It prints nothing from its search but
the settings chosen (its Table 4), so none of the three can be checked against
its test results. The passage on holding engines out in
`docs/LITERATURE_BASIS.md` sets this beside three papers whose search results
can be checked.

*Tables 9 and 10.* Confirmed from the rendered tables (PDF page 13, journal
page 186). The rows for TSO-GCN (R² 0.9766, RMSE 4.6304, Score 0.9980) and
WOA-XGBoost (0.9781, 4.4783, 0.9985) are the same to the last digit in Table 9,
for FD002, and in Table 10, for FD004; only the GCN row differs between the
two. In Table 10 IMTSO-GCN's own row (0.9658, 4.9416, 0.9986), printed in bold
in every column, has a lower R² and a higher RMSE than both of those methods.
Only its Score is ahead of theirs, by 0.0006 and 0.0001. That score (Eq. 27,
PDF page 9, journal page 182) averages, over the predictions, a number that is
1 for an exact prediction and smaller the further off it is relative to the
true remaining life, so higher is better. The text argues superiority from
FD002 ("it is obvious that IMTSO-GCN can obtain better predicted results
compared with other algorithms as shown in Table 9"), makes no claim about
FD004 beyond introducing Table 10 together with Table 9 ("Table 9 and Table 10
are the comparison results of different algorithms") and saying "Figure 8
shows the predicted results of different methods on the FD004 dataset" (all
PDF page 13, journal page 186), and concludes that IMTSO-GCN "could provide
better predicted performance in contrast to other methods" (PDF page 14,
journal page 187). Its abstract says its R² "is greater than 0.99, RMSE is less than 3"
(PDF page 1, journal page 174), which is true of FD002 only. The paper does not
explain the repeated rows, and nothing in it recovers what those two methods
scored on FD004.

All three were raised on 2026-10-02 while the correction above was being
written, and answered the same day from the copy in `docs/references/`. The
numbers about NASA's engines in this note were computed from the files in
`data/raw/`. The positions and colours in Figs. 3 and 6 were read from the
pages rendered at eight times their printed size, and those in Fig. 7 from the
picture embedded in the page at its own size, all by sampling pixel colours.
A review the same day, before this note was first committed, found that its
first version gave Fig. 7's agreement as "to within 0.3 of a cycle" without
saying how the line was fitted. Refitted under the choices now written down,
the largest gap is 0.40 to 0.45 with the slope held at one per cycle and up to
0.74 with it fitted, so a precision read off a figure cannot be checked unless
the fitting choices are written beside it. The same review found that the first version gave
Fig. 3's arrows the colours of the boxes beside them, put the question of
where the search was scored as a choice between two possibilities when the
paper leaves three open, and said the paper's text mentions FD004 only through
Fig. 8.

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
they swap is P15.** Reducing the 21 sensors to 14 by dropping seven that do
not vary is a common step in C-MAPSS studies; the survey takes it "Following
previous studies" (page 19). It is not universal. IMTSO-GCN builds its graph
over all 21 sensors and seven other columns, and only its tuned filtering
leaves fourteen sensors connected; they are the fourteen of the first three
rows below, an outcome of the tuning rather than a choice the paper states
(see the correction about IMTSO-GCN above). Jin et al. (2025), outside this
corpus, drop the five sensors that fail their two tests, being related to the
other sensors and trending over time, and keep sixteen (PDF page 7), `P15`
and `NRc` among them. Three of the sources here name their seven, and all
three agree with each other and disagree with the source paper:

| Source | Page | The fourteen kept |
|---|---|---|
| HAGCN (RESS 2021) | 5 | 2, 3, 4, 7, 8, 9, 11, 12, 13, **14**, 15, 17, 20, 21 |
| AGCTE (Aerospace 2024) | 6 | 2, 3, 4, 7, 8, 9, 11, 12, 13, **14**, 15, 17, 20, 21 |
| Survey (MSSP 2025) | 19 | the same, stated as removing 1, 5, **6**, 10, 16, 18, 19 |
| **ESTGNN (the source)**, Table 3 | 5 | 2, 3, 4, **6**, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21 |

Thirteen sensors are shared by all four. The disagreement is a single swap:
these three keep `NRc` (column 14) and discard `P15` (column 6) as
constant-valued; the source paper does the exact opposite.

**Correction, 2026-10-02.** Until this date this passage said "Every C-MAPSS
study reduces the 21 sensors to 14 by dropping seven that do not vary", and
the paragraph above said "the field" keeps `NRc` and discards `P15`. Two
studies this project holds do otherwise. IMTSO-GCN, in this corpus, builds its
graph over all 21 sensors (its Fig. 6, PDF page 12, journal page 185, rendered
at eight times its printed size). Jin et al. (2025) keep sixteen sensors,
`P15` and `NRc` both among them: their five dropped sensors are named by
symbol on PDF page 7 and were matched to sensor numbers through the table of
all 21 in `src/thesis_gnn_rul_xai/data/sensors/__init__.py`. Found when the
cited papers were collected into `docs/references/` and checked against the
places that cite them. The finding itself stands: the three sources in the
table name the same seven, and they differ from the source paper by the one
swap of `P15` for `NRc`.

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
engineering methods" — and THGNN argues *against* what the source does. The
caption of its Fig. 1 (PDF page 2, journal page 19749) says that earlier
snapshot-based methods compress "MTS data from a time window into a single
'flat' graph", which it says often loses "critical fine-grained temporal
details"; MTS is the paper's short form for multivariate time series, several
sensors recorded over the same cycles. Its abstract (PDF page 1, journal page
19748) calls capturing such snapshots a coarse-grained approach that "leads to
a loss of temporal information".

**Correction, 2026-10-02.** Until this date the THGNN passage above was printed
as a single quotation from page 2 ending "often leading to the loss of critical
temporal information". The paper has no such sentence. In the caption of
Fig. 1 on page 2 that clause ends "critical fine-grained temporal details",
and the caption then runs on for several more lines. The quotation had also
dropped, without marking the gap, a parenthesis the caption has between
"graph" and "often". The ending "loss of temporal information" is the wording
of the abstract on page 1 (and of the body text beside the figure on page 2).
The two are now separate quotations, each with its page. Found when the cited
papers were collected into `docs/references/` and checked against the places
that cite them. What the paragraph argues is unchanged.

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
average; the outlier is the first row, and the reason is that the network is
not linear, so the prediction at the average input is not the average
prediction. And because every attribution is measured from `phi_0`, the values
this project produces cannot be turned into values measured from the paper's
`phi_0` by a shift that keeps their order by size; the correction below works
through each way of making that change.

Reproduce with the script recorded in this entry's commit message.

**Correction, 2026-10-02.** Until this date the paragraph above ended with a
claim in bold: that the whole table is "offset from the paper's definition by
that constant", so that "rankings by magnitude are unaffected, individual
values are not". That is wrong under each of the three things measuring from a
different `phi_0` could mean. The review of the playground's information cards
on 2026-09-26 found it. The same paragraph also called the outlier "the third
row"; it is the first, `f(average training window)` at 91.357, and that slip
was found while correcting the claim.

- **Changing only the constant.** Write the mean label where `f(baseline)`
  stood and keep the fourteen values. No value moves, so nothing about them is
  offset, but Equation 14 stops adding up: at every window, the mean label
  plus the fourteen values falls short of the prediction by the same amount,
  `f(baseline) − E[y]`, which for the model in the table above is the 10.719
  cycles.
- **Shifting every value by the same amount**, the most literal reading of
  "offset by that constant". Spreading `f(baseline) − E[y]` evenly, adding a
  fourteenth of it to each value, makes Equation 14 add up from the mean label
  again. Every value moves by the same signed amount, so the order from most
  positive to most negative is kept, but the order by size is not: a shift
  upwards makes a positive value larger and a negative one smaller in size. On
  the stored explanation of held-back engine 5 by the seed-42 network of run
  `fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z`
  (explanation `shap_engine_unit5_seed42_c2bf7d4-dirty_20260916T070850Z`,
  240 windows), `f(baseline)` is 91.885 and the mean label over the 17 731
  training windows is 80.638, so each value moves up by 0.803 cycles. The
  order by size changes in 236 of the 240 windows, and so does the order of
  the sizes averaged over the engine's life: W31 and T30 swap second and third
  place, and BPR and Nf swap twelfth and thirteenth.
- **Changing what stands in for an absent sensor.** A different stand-in
  changes every prediction in which some sensor is absent, which is every
  prediction the calculation is built from except `f(x)` itself, so each
  sensor's value is recomputed by its own amount, and nothing keeps the order
  of their sizes. The SHAP page under Resources shows this on real FD001
  readings, with a straight line fitted to four sensors and its values
  computed against two stand-ins: the average of all training rows, and the average of the rows at
  the 125-cycle ceiling. T24 moves from 5.984611 to 1.883631 cycles and P30
  from −0.863789 to −6.969030, and the order by size changes from T50, T24,
  P30, T30 to T50, P30, T30, T24.

So the choice recorded here is not a uniform shift that leaves rankings alone.
The rankings this project reports belong to its stand-in (assumption A-20),
and would have to be recomputed, not shifted, under any other.
`tests/test_examples.py` checks all three points on real data: the first on
every stored Shapley explanation in `reports/`, the second on those from runs
whose data were prepared by `config/fd001.yaml` (with the counts above on the
one named), skipping when there are none, and the third from the FD001 files.

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
definition of leakage. It moves the reported figures, by an amount and in a
direction that cannot be detected by looking at them. (Until 2026-10-02 this
said it "flatters" them; see the correction at the end of this entry.) Two
tests hold the line —
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
a number this project wants in its history. Its direction is not known either:
leakage usually inflates measured performance, but nothing guarantees that
here, so the comparison would tell us the size and the sign of an effect this
project declines to introduce. (Until 2026-10-02 this said "the direction is
not in doubt", and that fitting on the test engines "can only improve the
reported figures"; see the correction below.) A-08 covers the *other*
ordering question in Fig. 1, between smoothing and scaling, and is withdrawn
because there the paper's order is the one followed here.

**Correction, 2026-10-02.** Until this date two sentences above promised a
direction that nothing guarantees: that the leak "flatters the reported
figures", and that fitting on the test engines "can only improve" them. The
review of the playground's information cards on 2026-09-26 found the second,
and the first was found while correcting it. Fitting the scaling on the test
engines changes the numbers every training reading is turned into
(the normalisation folder's README shows every sensor's mean moving), so the
network trained on them is a different network, and nothing in the arithmetic
of training says whether that network then does better or worse on the test
engines. What is certain is the reason for refusing it: the reported figure
would stop measuring performance on engines the preparation never saw, and
nothing in the figure would say in which direction, or by how much, it had
moved. Studies of leakage find that it usually inflates measured performance
(Kapoor and Narayanan's 2022 preprint, the copy in `docs/references/`, PDF
page 4, Section 2.1: leakage "usually leads to inflated estimates"), which is
a tendency across published work, not a guarantee for this pipeline. The
decision not to build the leaking variant stands, for the first reason given
above, and its effect stays unmeasured in both size and direction.

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

**Seen since, for arm A (added 2026-10-02).** Since commit `c8ba1f1` of
2026-09-14 every run saves, for each seed, `training_history.csv`: one row per
pass over the training windows, with the training loss (the average squared
error over that pass, on lives divided by 125, assumption A-13) and the RMSE on
the held-back engines. Arm A was trained again after that, at `f386291-dirty`
with the same settings:
`fd001_physics_row_cap_5seed_varsplit_lstm32_cfg72f506_f386291-dirty_20260914T134955Z`
gives the same five test RMSEs to every stored digit and the same kept passes
(10, 13, 9, 14, 21), so its curves are arm A's. From the pass that was kept to
the last pass run, twenty passes later when early stopping ended the run, the
training loss falls on every seed and the held-back RMSE rises on every seed:

| Seed | Kept pass | Last pass | Training loss, kept → last | Held-back RMSE, kept → last |
|---|---|---|---|---|
| 42 | 10 | 30 | 0.01014 → 0.00665 | 13.449 → 14.911 |
| 43 | 13 | 33 | 0.00949 → 0.00631 | 13.491 → 14.390 |
| 44 | 9 | 29 | 0.00968 → 0.00622 | 14.804 → 18.313 |
| 45 | 14 | 34 | 0.00874 → 0.00636 | 14.625 → 15.796 |
| 46 | 21 | 41 | 0.00757 → 0.00514 | 14.302 → 15.636 |

The network keeps fitting the training engines more closely while it gets worse
on engines it never learns from, which is what over-training is, so for the
first 41 passes it is now observed rather than inferred. All of those passes
ran at the first learning rate, 0.001; the single drop (assumption A-17) comes
at pass 100. What happens after pass 41 is still not seen: arm C ran all 200
passes but predates the saved curves, and arm D, the paper's procedure, holds
no engines back, so it has no held-back curve to save. Seeing the held-back
error out to pass 200 needed arm C trained again, which the next paragraph
reports. (Corrected the same day: this
note first said the arms' curves exist only if they are trained again; the
re-run of arm A above was already stored, and the log cites it under
assumption A-28.)

```python
import pandas as pd

run = "reports/fd001_physics_row_cap_5seed_varsplit_lstm32_cfg72f506_f386291-dirty_20260914T134955Z"
kept = pd.read_csv(f"{run}/results.csv").set_index("seed")["best_epoch"]
for seed in range(42, 47):
    curve = pd.read_csv(f"{run}/seed_{seed}/training_history.csv").set_index("epoch")
    a, b = curve.loc[kept[seed]], curve.iloc[-1]
    print(seed, kept[seed], curve.index[-1], round(a.train_loss, 5), round(b.train_loss, 5),
          round(a.validation_rmse_cycles, 3), round(b.validation_rmse_cycles, 3), curve.learning_rate.max())
```

**Seen to pass 200, for arm C (added 2026-10-03).** Arm C trained again from a
clean checkout of `19dceb5`, on the graphics card:
`fd001_physics_row_cap_5seed_varsplit_lstm32_noearlystop_cfgba82e3_19dceb5_20261003T090451Z`
(a fifth of the engines held back, all 200 passes, the best pass restored). It
does not reproduce the stored arm seed for seed — test RMSE 13.209 ± 0.708
against 13.307 ± 0.532, kept passes 17, 24, 9, 37 and 5 against 10, 13, 9, 14
and 21 — because the graphics card and PyTorch 2.5.1 compute slightly
differently from the processor and PyTorch 2.13.0 the stored arm used, the same
movement the sensitivity study's runs show beside the stored ones (assumption
A-28). The shape of every curve is what matters here:

| Seed | Kept pass | Training loss, kept → 200 | Held-back RMSE at the kept pass | at 50 | at 100 | at 150 | at 200 |
|---|---|---|---|---|---|---|---|
| 42 | 17 | 0.00867 → 0.00039 | 14.069 | 16.922 | 17.765 | 18.318 | 18.592 |
| 43 | 24 | 0.00755 → 0.00036 | 13.549 | 14.800 | 15.746 | 16.475 | 16.468 |
| 44 | 9 | 0.00976 → 0.00034 | 14.439 | 17.905 | 18.050 | 18.137 | 18.258 |
| 45 | 37 | 0.00596 → 0.00044 | 14.211 | 15.381 | 16.506 | 16.848 | 16.993 |
| 46 | 5 | 0.01111 → 0.00032 | 14.430 | 16.022 | 17.401 | 18.005 | 18.257 |

On every seed the training loss falls about twentyfold after the kept pass
while the held-back error climbs, by 2.78 to 4.52 cycles by pass 200, and keeps
climbing after the single drop in learning rate at pass 100. So the over-training
the paper's 200-pass procedure exposes the stage to is now observed over the
whole run, not inferred. Reproduce with the snippet above, pointed at this run,
with the 50th, 100th, 150th and 200th rows of each `training_history.csv`.

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
  per-engine prediction. (Added 2026-10-02: runs since 2026-09-14 save each
  seed's `test_predictions.csv`, and the resampling over engines has since been
  done for the current default, in finding F-06, and for the window-summary
  stage, in assumption A-28. `docs/AUDIT_2026-09-08_response.md` records the
  project's answer to each of the audit's items.)
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

The GRU stage, a recurrent cell the paper does not have, borrows these three
sizes unchanged; it has its own entry, assumption A-36 (pointer added
2026-10-03).

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

**On the held-back engines the correlation graph does better, on every seed,
on each of four ways of summarising that error.** Each seed holds back the
same 20 engines whichever graph it trains on, because which engines are held
back is drawn from the seed alone, so the held-back error pairs by seed too,
and a comparison between configurations should be judged on it rather than on
the test engines (the audit of 2026-09-08 found this project had not always
done so).

The held-back figure a run records needs care. After every pass over the
training data the run measures its error on the held-back engines. The figure
it records is the lowest of those, and that same figure chose which pass's
weights to keep (assumption A-18). The lowest of many readings sits below where
the network usually was, and further below for a network whose error wanders
more from pass to pass, so two networks can be flattered by different amounts.
Each seed's `training_history.csv` keeps the error after every pass, so the two
graphs can also be compared on summaries that no pass was chosen by. Two were
decided on before either was computed, and a third was added as a check:

- **The mean of the last 20 passes each run made.** A run stops once 20 passes
  in a row have failed to beat its best by more than 0.0001 cycles (assumption
  A-18), so in each of the fifteen runs read here — five seeds each of Table 3,
  the correlation graph and the every-sensor-joined graph further down — these
  are exactly the 20 passes after the one it kept. No pass is picked out, but
  the stopping rule placed the window. The rule alone does not keep those 20
  above the kept pass: one could sit below it by less than 0.0001 cycles
  without counting as better. In all fifteen runs none of them is below it
  (checked against the pass each run's `results.csv` names as kept; the
  closest sits 0.0012 cycles above it, Table 3's seed 45), so if this summary
  leans at all, it leans against the network.
- **The mean of passes 1 to 23.** The shortest of the fifteen runs, the
  correlation graph's seed 42, made 23 passes, so every run has all of them. A
  run is the same up to the pass at which it stops whether or not it stops
  early, so nothing in this summary depends on the held-back error. It does
  include the first passes, while the error is still falling fastest, so it
  reflects how quickly a network gets somewhere as well as where it gets to.
- **The error after pass 23 alone**, one reading, as the check.

Every pass in all fifteen runs used the same learning rate, 0.001: the step
schedule lowers it only after pass 100 (assumption A-17), and the longest run
made 75.

| Held-back RMSE, cycles | Table 3 | 41 strongest correlations | difference | paired t | p |
|---|---|---|---|---|---|
| lowest over all passes, as each run records it | 21.168 ± 0.870 | 19.169 ± 1.473 | −1.999 | −4.39 | 0.0118 |
| mean of the last 20 passes | 21.665 ± 1.068 | 19.852 ± 1.622 | −1.813 | −4.54 | 0.0105 |
| mean of passes 1 to 23 | 21.890 ± 0.832 | 19.910 ± 1.628 | −1.981 | −4.39 | 0.0118 |
| pass 23 alone | 21.672 ± 1.004 | 19.709 ± 1.568 | −1.964 | −3.74 | 0.0202 |

The correlation graph is lower on every seed in every row. By seed, the
recorded figure differs by `[−1.798, −2.391, −3.429, −1.734, −0.644]`, the
mean of the last 20 passes by `[−1.847, −2.378, −2.811, −1.563, −0.468]` and
the mean of passes 1 to 23 by `[−2.231, −2.754, −2.858, −1.672, −0.387]`.
Pass by pass, it is lower after every pass both runs made on seeds 42 to 45
(23, 46, 28 and 38 passes) and after 44 of 55 on seed 46, and lower after the
very first pass on all five, by 1.756 to 2.632 cycles.

**How much the choice of pass flattered each graph.** Seed by seed, the
recorded figure sits below the mean of the last 20 passes by
`[0.328, 0.190, 0.932, 0.148, 0.886]` cycles for Table 3 (mean 0.497) and by
`[0.279, 0.203, 1.551, 0.320, 1.062]` for the correlation graph (mean 0.683).
The correlation graph is nominally flattered more, by 0.186 cycles on average,
but on seed 42 it is flattered less, and five seeds cannot resolve the
difference: per seed `[−0.049, +0.013, +0.619, +0.172, +0.176]`, paired
t = 1.59, p = 0.19. Either way it is under a tenth of the recorded gap between
the graphs, 1.999 cycles. The held-back advantage does not come from the choice
of pass, and that conclusion rests not on this difference but on the two
summaries no pass was chosen by, which put the correlation graph lower on every
seed, by 1.813 and 1.981 cycles on average.

Two limits. The four rows summarise the same ten runs, so they are four views
of one comparison rather than four experiments. Treated as separate tests
anyway, the bar is 0.05 ÷ 3 = 0.0167: three, because the recorded row is the
figure under question and the three rows below it were added to test it
(assumption A-37 states the rule). At that bar the two summaries decided on in
advance clear it and the single
reading at pass 23 does not. And the held-back and test figures weight engines
differently and are never comparable with each other (assumption A-06), so the
advantage here and the absence of one on the test engines are two separate
results, and this entry does not explain the difference between them.

**Every sensor joined to every other, on the same summaries.** Against the
every-sensor-joined graph (assumption A-19,
`fd001_complete_row_cap_5seed_varsplit_cfgf21547_f386291-dirty_20260914T150527Z`,
the same commit and settings), Table 3 is better on every seed in every row:
the every-sensor graph is +1.447 cycles as recorded (p = 0.0041), +1.172 on the
mean of the last 20 passes (p = 0.0278), +0.966 on the mean of passes 1 to 23
(p = 0.0184) and +1.143 at pass 23 (p = 0.0155). At the same bar of 0.0167,
neither summary decided in advance clears it (0.0278 and 0.0184), so the
every-sensor graph's held-back cost is less firmly established than the
recorded p = 0.0041 suggests, though it has the same sign on every seed: by
seed, `[+1.745, +1.917, +0.415, +1.522, +0.262]` on the mean of the last 20
passes. Its recorded figure sits below its last-20 mean by
`[0.195, 0.064, 0.253, 0.178, 0.424]`, mean 0.223, the least of the three
graphs, against Table 3's 0.497. So it is nominally flattered less than the
paper's graph, by 0.274 cycles on average, which is how much wider the recorded
difference is than the one on the last 20 passes (1.447 − 1.172, before
rounding). Five seeds cannot resolve that either: per seed
`[−0.134, −0.126, −0.679, +0.029, −0.462]`, paired t = −2.13, p = 0.10, and on
seed 45 it goes the other way. Nor can they resolve the largest of the three
gaps, between the correlation graph and the every-sensor graph: 0.460 cycles on
average, the same sign on every seed (`[+0.085, +0.139, +1.298, +0.142, +0.638]`),
paired t = 1.98, p = 0.12.

*Correction, 2026-10-02.* From 2026-09-14 until this date the held-back table
here was followed by a caution in place of the measurements above: that the
recorded figure is flattered by an amount that can differ between two networks,
so "the held-back result says the correlation graph is at least not worse; it
does not settle that it is better". Nothing in the caution was wrong, but it
stopped short of an answer that needed no new training. It was settled by
reading the held-back error after every pass from each seed's
`training_history.csv` in the stored runs named under "Reproduce with" below.
The first version of the measurements, written the same day and never
committed, said the correlation graph was flattered more than Table 3 and that
the choice of pass favoured Table 3 over the every-sensor graph, without testing
either difference against the seed-to-seed spread; it also said the window of
the last 20 passes could not hold a pass below the kept one "by construction",
which the stopping rule's 0.0001-cycle margin does not guarantee. A review the
same day tested both differences (p = 0.19 and 0.10), checked the window in all
fifteen runs, and the paragraphs above now say so.

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
were compared, and at 0.05 divided among four tests (0.0125, assumption A-37)
only ASUE's difference clears the bar; and the mean-error comparison was run after seeing
the score and ASUE, to explain them, so it is a follow-up and not a test
decided in advance. One seed of the correlation run, seed 42, kept its weights
from the third pass over the data.

**What this says about the paper's claim.** In this configuration the
engineering knowledge did not make the network more accurate than the data's
own 41 strongest relationships — not on the test engines, and not on the
held-back engines, where the correlation graph's error was lower on every seed,
by 1.8 to 2.0 cycles on average on each of four summaries of that error
(p 0.010 to 0.020); it did make its test errors less often and less severely
late. That is a fact about this architecture — one number per sensor
and no recurrent stage — and not about the full model the paper's claim is
made for. Two of this project's configuration comparisons have reversed when
the architecture moved (working rule 3), so it was re-measured with the
recurrent stage.

**With the recurrent stage (added 2026-10-02).** The same comparison with the
paper's Section 2.2 stage in front of the graph layers, at the size assumption
A-26 chose, with a fifth of the engines held back, early stopping and the best
pass restored (so not the paper's 200-pass procedure, finding F-16), the step
schedule, both graphs trained on the GPU from a clean checkout of `19dceb5`:
`fd001_physics_row_cap_5seed_varsplit_lstm32_cfgba82e3_19dceb5_20261002T210700Z`
and `fd001_correlation_row_cap_5seed_varsplit_lstm32_cfg7fdfce_19dceb5_20261002T214210Z`.
Five seeds each, paired by seed:

| Error | Table 3 | 41 correlations | correlation minus Table 3 | p |
|---|---|---|---|---|
| test RMSE | 13.234 ± 0.723 | 13.237 ± 0.967 | +0.002 | 0.995 |
| held-back, the recorded lowest | 14.184 ± 0.389 | 13.494 ± 0.362 | −0.690, all five seeds | 0.0027 |
| held-back, mean of the last 20 passes run | 15.463 ± 0.738 | 14.846 ± 0.525 | −0.617, four of five seeds | 0.113 |
| held-back, mean of passes 1 to 25 (the fewest any run made) | 15.322 ± 0.757 | 14.853 ± 0.548 | −0.469, all five seeds | 0.037 |
| held-back, pass 25 alone | 15.220 ± 0.559 | 14.741 ± 0.467 | −0.479, four of five seeds | 0.188 |

On the test engines the two graphs are again indistinguishable. On the
held-back engines the correlation graph is lower on every summary, but only
the recorded lowest — the figure the choice of pass flatters — clears the bar
of 0.05 / 3 = 0.0167 set above for the three summaries no pass was chosen by;
none of those three does. So with the recurrent stage the engineering
knowledge again does not make the network more accurate than the 41 strongest
correlations, and the held-back advantage of the correlation graph that the
last-cycle network showed is smaller here and not established. The best passes
were 17, 24, 9, 15 and 5 for Table 3 and 10, 12, 9, 15 and 13 for the
correlation graph; the runs stopped after 25 to 44 passes. The held-back rows
come from the snippet above with `runs` set to these two runs and the
every-sensor entry and its lines removed, which prints them exactly; the test
row from each run's `results.csv`. The FD002 and FD004 comparison is part of the
sensitivity study (finding F-19) and is reported there once it is complete.

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

The held-back summaries, and the every-sensor-joined comparison beside them,
are recomputed from those two runs and
`fd001_complete_row_cap_5seed_varsplit_cfgf21547_f386291-dirty_20260914T150527Z`
(made by the same command with `--graph complete`) by this, run from the
repository root with no training:

```python
from pathlib import Path

import pandas as pd
from scipy import stats

runs = {
    "Table 3": "fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z",
    "correlation": "fd001_correlation_row_cap_5seed_varsplit_cfg7fdfce_f386291-dirty_20260914T150213Z",
    "every sensor": "fd001_complete_row_cap_5seed_varsplit_cfgf21547_f386291-dirty_20260914T150527Z",
}
seeds = range(42, 47)
# The held-back RMSE after every pass, one array per graph and seed.
error = {
    (graph, seed): pd.read_csv(Path("reports", name, f"seed_{seed}", "training_history.csv"))
    ["validation_rmse_cycles"].to_numpy()
    for graph, name in runs.items() for seed in seeds
}
shortest = min(len(passes) for passes in error.values())
summaries = {
    "lowest over all passes": lambda e: e.min(),
    "mean of the last 20 passes": lambda e: e[-20:].mean(),
    f"mean of passes 1 to {shortest}": lambda e: e[:shortest].mean(),
    f"pass {shortest} alone": lambda e: e[shortest - 1],
}
for label, summarise in summaries.items():
    value = {graph: pd.Series([summarise(error[graph, s]) for s in seeds]) for graph in runs}
    for graph in ("correlation", "every sensor"):
        gap = value[graph] - value["Table 3"]
        test = stats.ttest_rel(value[graph], value["Table 3"])
        print(f"{label:26s} {graph:12s}  Table 3 {value['Table 3'].mean():.3f} "
              f"(sd {value['Table 3'].std():.3f})  {graph} {value[graph].mean():.3f} "
              f"(sd {value[graph].std():.3f})  difference {gap.mean():+.3f}  "
              f"t {test.statistic:+.2f}  p {test.pvalue:.4f}  "
              f"per seed {gap.round(3).tolist()}")
for seed in seeds:
    ours, paper = error["correlation", seed], error["Table 3", seed]
    both = min(len(ours), len(paper))
    print(f"seed {seed}: the correlation graph is lower after "
          f"{(ours[:both] < paper[:both]).sum()} of the {both} passes both runs made, "
          f"and by {paper[0] - ours[0]:.3f} after the first")
# How far each run's recorded figure sits below the mean of its last 20 passes,
# and whether that amount differs between graphs by more than the seeds disagree.
flattered = {
    graph: pd.Series([error[graph, s][-20:].mean() - error[graph, s].min() for s in seeds])
    for graph in runs
}
for graph in runs:
    print(f"{graph:12s} flattered by {flattered[graph].mean():.3f} on average, "
          f"per seed {flattered[graph].round(3).tolist()}")
for one, other in (("correlation", "Table 3"), ("every sensor", "Table 3"),
                   ("correlation", "every sensor")):
    shift = flattered[one] - flattered[other]
    test = stats.ttest_rel(flattered[one], flattered[other])
    print(f"{one} minus {other}: {shift.mean():+.3f}  t {test.statistic:+.2f}  "
          f"p {test.pvalue:.4f}  per seed {shift.round(3).tolist()}")
# Each run made exactly 20 passes after the one its results.csv names as kept,
# and none of them is below it. Passes are numbered from 1.
margins = []
for graph, name in runs.items():
    kept = pd.read_csv(Path("reports", name, "results.csv")).set_index("seed")["best_epoch"]
    for s in seeds:
        e = error[graph, s]
        after = e[kept[s]:]
        assert len(after) == 20 and e[kept[s] - 1] == e.min()
        margins.append(after.min() - e[kept[s] - 1])
print(f"the closest of those passes sits {min(margins):.4f} cycles above the kept one")
```

The later stored run of Table 3,
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z`,
has the same held-back error after every pass on all five seeds, so a summary
read from it gives the same figures.

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

**Pairing across commits, checked against the other stored runs of the
recurrent stage (added 2026-10-03).** Five stored five-seed runs train the
recurrent stage in this configuration, each at a different state of the code,
and they fall into two groups that follow where they were trained, not the
commit. Three agree with each other exactly: `f82c76c-dirty` (2026-09-09), the
`f386291-dirty` run above, and `70521f9-dirty` (2026-09-28). Their
`results.csv` files agree on every seed in every column they share, except the
number of passes run and the time taken: `70521f9-dirty` waited 50 passes
without improvement before stopping instead of 20, which changed no kept pass.
The two of them that saved their predictions give the same prediction for
every test engine on every seed, and the same held-back error after every pass
both ran. Only `70521f9-dirty` records where it ran, on the processor under
PyTorch 2.13.0; the other two record neither, so that they ran the same way is
inferred from the exact agreement, not read from a record. Two ran on a
graphics card under PyTorch 2.5.1, and record it: the sensitivity study's
reference at `fd65a3b-dirty` (finding F-19) and the run from a clean `19dceb5`
in assumption A-27; they agree with each other in the same way. Between the two
groups single seeds move. Graphics card minus processor, test RMSE by seed is
`[+0.588, +0.062, −1.197, +0.184, −0.001]`, while the means differ by only
0.073 (13.234 ± 0.723 against 13.307 ± 0.532, p = 0.82); held-back RMSE moves
by up to 0.620 on one seed (mean +0.050, p = 0.78), and one test engine's
prediction by up to 30.6 cycles. So among these five runs, pairing by seed
across commits paired the same networks when the device and the PyTorch
version were the same, and different networks when they were not: one seed's
difference can then hold a cycle that comes from the change of device and
library alone, which these runs cannot split between the two. That is a
finding about these five runs, not a rule: a commit that changes the training
or the preprocessing could give different networks on the same device and
library. The window-summary run above
records PyTorch 2.13.0 on the processor, the group its partner belongs to.
Paired with the graphics-card run instead, it is +0.352 on test RMSE (lower on 2
of 5 seeds, p = 0.46) and +1.639 on held-back RMSE (higher on all five seeds,
p = 0.00003), so neither conclusion above depends on which group's run it is set
beside. Each run's `results.csv`, `summary.json`, saved predictions and
`training_history.csv` files, read on 2026-10-03, give these figures.

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

## A-29 — Walks through the sensor graph are explained with GNN-LRP, on the score before the sigmoid, with the authors' own setting of its rule

**Where:** `explain/walks.py` (`walk_relevance`, `paper_gammas`); the named
settings of gamma in `scripts/explain.py` (`gamma_values`) and
`docsite/playground/settings.py` (`GAMMA_SETTINGS`)

**The gap.** Not a gap in a method the paper describes, but a question it raises
and does not answer. Section 2.4 applies SHAP to the fourteen sensor nodes and
calls the result *"node-level contribution analysis"* (p. 7). Section 4.4 then
says the rankings *"determine both the node coloring and edge weights in the
interpretability graph"* (p. 13) and ends by saying the changing patterns
validate *"the effectiveness of the physics-informed graph structure"* (p. 14).
No calculation in the paper attributes anything to a connection between two
sensors: the edge weights of Fig. 10 are derived from node rankings by a rule
the paper's text does not state. So the paper has no method for asking how much
of a prediction travelled through its graph. The explanation added here answers
that question with a method of its own, applied as its authors define it;
nothing in it is taken from the paper's SHAP analysis or tuned to its results.

**What we chose.**

1. **GNN-LRP** (Schnake et al., 2022), which divides a graph network's output
   into one share per walk — the list of sensors a contribution passes
   through, one per graph layer — rather than GNNExplainer (Ying et al., 2019)
   or PGExplainer (Luo et al., 2020), which learn a mask over the connections.
2. **The score before the sigmoid** is what is divided, not the prediction in
   cycles.
3. **The authors' own setting of the rule's gamma.** Their published code,
   given no setting, uses `linspace(3, 0, depth)`: 3 at the first graph layer
   falling in equal steps to 0 at the last. Gamma is 0 at the first graph layer
   here, because the rule's derivation needs every number entering a layer to
   be zero or more and this network's first graph layer receives rescaled
   readings; with one number per sensor that changes nothing, because gamma
   cancels there. Gamma is 0 in the dense layers after the graph, following the
   one network for which the paper says what such layers get. For three graph
   layers the setting is 0, 1.5 and 0. Any other setting may be passed to
   `walk_relevance`, one per layer with 0 first. *Added 2026-10-02:* beside
   it, for comparison and never as the default, `scripts/explain.py --gamma`
   and the playground's Explain tab offer three named settings, each 0 at the
   first graph layer (`gamma_values` in `scripts/explain.py` turns a name into
   numbers): `gnn-gi`, 0 at every layer, the source's own simpler baseline;
   `falling`, 0.5 at the second graph layer and halving at each later one,
   which is the halving the source uses for VGG-16 (its Section 5.3) started at
   the second graph layer; and `flat`, 1 at every layer after the first. For
   three graph layers the last two are 0, 0.5 and 0.25 and 0, 1 and 1, the two
   settings the measurements below compare with the authors'. For another
   number of graph layers `falling` goes on halving and `flat` stays at 1;
   that carry-over is ours and has not been measured. *Added 2026-10-03:* the
   authors' setting carries over by its own rule, 3 falling in equal steps to 0
   with the first value replaced by 0, so the four names do not always give
   four settings. With four graph layers the authors' setting is 0, 2, 1 and 0,
   `falling` 0, 0.5, 0.25 and 0.125, and `flat` 0, 1, 1 and 1; with two graph
   layers the authors' setting is 0 and 0, the same as `gnn-gi`; with one, all
   four names give 0. (Printed by `gamma_values` for one to five graph layers
   on 2026-10-03. The number of graph layers is set by
   `scripts/train.py --gcn-hidden`; every network record under `reports/`,
   the `model_config.yaml` files, checked on 2026-10-03, has three graph
   layers, of 32, 64 and 32 numbers per sensor.)
4. **The added constants** of the two dense layers, and of the graph layers
   when they have them, are reported as one number per window, not spread over
   the walks.
5. **Walks start at the temporal stage's output**, the numbers entering the
   first graph layer, not at the raw readings.
6. **Sensors the graph cannot tell apart count as one wherever a walk ends**
   when walks are added up or drawn (finding F-18); the walks themselves are
   kept unpooled.

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
  one per walk, plus a remainder from the constants; the gamma rule rescales
  each layer so it still passes on exactly what it computed, so the sum holds
  at any setting. The remainder is computed separately, by carrying the
  constants forward, so that walks plus remainder equalling the score is a
  check and not a definition. It changes from window to window with `fc1`'s
  ReLUs, so the walks add up to the score minus that remainder, not to the
  score minus the score of some reference engine.
- **The score, not cycles.** The sigmoid bends, so shares of the score cannot
  become shares of cycles that still add up. The source explains outputs before
  the final squashing step in its own examples: the difference of two class
  scores before the softmax (its Fig. 7), and a non-linear output made linear
  first (its Section 5.2). It never discusses a sigmoid.
- **The authors' setting of gamma.** With gamma 0 everywhere the method is the
  source's own simpler baseline, GNN-GI, which it reports as noisier and less
  selective (its Section 4); its method proper uses gamma above 0, set per
  network: 2 then 1 for its GIN, 3 for its sentiment model, 0.5 halving per
  block and 0 in the classifier for VGG-16, and, in its version 1, 0.25 at
  every layer. With no single value given, the default of the authors'
  published implementation is the closest thing to their protocol. The first
  graph layer is the exception their own derivation forces: its appendix C
  assumes the numbers entering a layer are never negative, and where a network
  lacks that structure — a ReLU, then a weighted sum, then a ReLU — the authors
  use gamma 0 themselves, as for their SchNet model (their Section 4.1). That
  exception has an effect only when the temporal stage hands the graph more
  than one number per sensor. With one, every unit of the first graph layer
  takes all its inputs through a single weight, so the factor 1 + gamma of
  their Table 1 sits above and below every share and cancels. Measured on the
  five stored runs below, every seed, the last window of every test engine:
  giving the first layer the authors' 3 instead of 0 changes no walk at all on
  the default, correlation-graph, complete-graph and recurrent networks, and
  moves the walks of the three-number window summary by a median of
  106 ± 42 % of their total size (mean and spread over the five trained seeds
  of each seed's median; `first_layer_gamma_change_median` in the run's
  `walks/19dceb5_20261002T182022Z/per_seed.csv`, from
  `scripts/measure_walks.py` at a clean `19dceb5`). Until 2026-10-02 this said
  103 ± 40 %, measured with the code at `ce7e305-dirty` before the measure was
  stored in a run folder; the stored figure replaces it, and what accounts for
  the three points between them was not traced. On the other four runs the
  stored measure is exactly 0 on every seed.
- **Gamma 0 in the dense layers after the graph.** The paper sets what such
  layers get for one network only: for its image model, VGG-16, gamma 0 in the
  top-level classifier (its Section 5.3). Its synthetic-graph code has no
  layers after the graph part. This is a choice following that one case, not a
  rule the paper states in general.
- **The named settings beside the authors'.** They exist so that a reader can
  explain the same windows twice and see, one window at a time, what the
  ranking comparison below measures over the last windows of the 100 test
  engines. The baseline is the
  source's own; the other two are the settings that comparison already used,
  so on three graph layers the playground offers nothing the log has not
  measured. None of them is offered as a better setting.
- **Walks from the temporal stage's output.** Every implementation by the
  source's authors stops at the input of the first graph layer. Going further
  back through the recurrent stages would need rules for their gates that this
  project does not have; for the stages that learn nothing, the starting number
  is a reading or an average of readings.

**Where the source's section numbers come from.** The arXiv version 3 of the
paper, since the journal's own text could not be retrieved; the journal version
may number them differently. The default schedule is read from `explain` in
`xaigraph-synth-code/explainers.py` of the authors' repository
(git.tu-berlin.de/thomas_schnake/paper_gnn_lrp).

**Correction, 2026-10-02.** Three readings of the source were wrong when this
entry was rewritten on 2026-10-01, and were found when another session checked
every citation against a copy of arXiv version 3. It said the paper keeps
gamma 0 "in the layers after the graph"; the paper says so only for VGG-16's
top-level classifier. It said the rule of Table 1 also adds gamma to a layer's
positive constant; Table 1 has no constant, and that treatment comes from the
authors' code. And it justified gamma 0 at the first graph layer without
noticing that with one number per sensor gamma cancels there, so on four of
the five stored runs the choice changes nothing. None of these changes a
number in this entry: none of the five stored runs it measures has
graph-layer constants, and the setting stays as it was.

**Correction, 2026-10-01.** From 2026-09-26 the default here was gamma 0 at
every layer, defended as "the one setting at which each share is exactly what
the network computes". That is the source's baseline, GNN-GI, under the name of
GNN-LRP. An audit against the source and its code, asked for by the author,
found it. It mattered: on the stored default network, seed 42, the walks of the
held-back engine 5 move by a median of 34 % of their total size between the two
settings on 184 of its 240 windows, and are unchanged only on the 56 where every
sensor's input to the first graph layer has the same sign — the one case where
gamma cancels, since every sensor then leaves that layer holding the same part
of the same weight row (`notebooks/xai/03_walks_fd001.ipynb`, step 4). The
tables below were re-measured under the authors' setting and replace the ones
measured at gamma 0, which had also been corrected once already on 2026-09-26,
when a walk that only changed member within a group of indistinguishable
sensors was still counted as leaving its starting sensor.

**How much it matters.** The choice of method is an argument, not a
measurement. What is measured is, first, the arithmetic, in `tests/test_walks.py`:
walks plus constant give back the score to within 1e-9 for every temporal stage,
with and without the graph layers' constants; at gamma 0 each sensor's total
equals PyTorch's own differentiation, at both ends of the walks and for each
graph entry; and for 30 walks at gamma 0 and at gamma (0, 0.5, 0.25) the shares
equal the source's own procedure, which runs the network backwards once per
walk with every other route blocked. The source's own check of an explanation
is assumption A-33.

Second, what the walks say about the stored networks, measured on 2026-10-01 by
`scripts/measure_walks.py` at `692d2d4-dirty` and again on 2026-10-02 from a
clean checkout of `19dceb5`, which reproduced every figure below and every
connection of assumption A-30 exactly, on the last window of each of
the 100 test engines, for all five saved seeds of five stored runs. Beside each
trained network the script explains an **untrained** one, built exactly as
`scripts/train.py` builds that seed's network before training it, so it is the
trained network's own starting point. A number says something about training
only where the two differ. Each cell is the mean ± spread over the five seeds of
each seed's median over the 100 windows:

| Stored run | Network | Walks that stay within their starting sensor's group, share of all walk size | Constant, share of score's size | Walk sizes ÷ size of their sum | Connection ranking, the authors' setting against gamma 0 |
|---|---|---|---|---|---|
| Paper's graph, last cycle (the default) | trained | 6.42 ± 3.20 % | 51 ± 4 % | 7.1 ± 2.6 | 0.901 ± 0.043 |
|  | untrained | 16.59 ± 3.56 % | 82 ± 14 % | 7.7 ± 5.0 | 0.994 ± 0.006 |
| Paper's graph, window summary | trained | 18.13 ± 3.54 % | 57 ± 4 % | 2.8 ± 0.6 | 0.966 ± 0.011 |
|  | untrained | 15.48 ± 3.01 % | 74 ± 32 % | 6.4 ± 2.5 | 0.979 ± 0.013 |
| Paper's graph, recurrent stage | trained | 15.78 ± 2.86 % | 11 ± 5 % | 2.1 ± 0.3 | 0.994 ± 0.003 |
|  | untrained | 16.59 ± 6.54 % | 97 ± 4 % | 9.1 ± 4.9 | 0.996 ± 0.010 |
| Correlation graph, last cycle | trained | 11.96 ± 3.17 % | 36 ± 4 % | 7.5 ± 1.1 | 0.932 ± 0.026 |
|  | untrained | 8.03 ± 2.87 % | 86 ± 13 % | 10.4 ± 3.7 | 0.942 ± 0.060 |
| Every sensor joined, last cycle | trained | 100 % | 54 ± 3 % | 3.6 ± 0.0 | 1.000 ± 0.000 |
|  | untrained | 100 % | 90 ± 7 % | 19.4 ± 19.0 | 1.000 ± 0.000 |

The columns: the size, ignoring signs, of the walks that never leave their
starting sensor's group, as a share of all walks' size, where sensors the graph
cannot tell apart form one group (finding F-18) and every other sensor is a
group of its own; the size of the dense layers' constant against the constant
plus the walks' sum; every walk's size added up and divided by the size of the
walks' sum, so 7.1 means the shares are large and mostly cancel; and Spearman's
rank correlation, over the connections, between what flowed along each under
the authors' setting and under gamma 0, with what arrives at a group at the last
graph layer split equally among its members.

What this shows, and what it does not:

- **How much stays within a group is mostly the graph's doing.** With the window
  summary and the recurrent stage, the trained and the untrained networks cannot
  be told apart (paired p = 0.37 and 0.75); with the correlation graph the
  trained networks keep a little more within groups (p = 0.049). On the default,
  training moved weight away from staying: 6.4 % against 16.6 %, lower on all
  five seeds (paired p = 0.021). Where everything is joined to everything, every
  sensor is in one group and every walk stays, by definition. A share of this
  kind cannot, on its own, show that the network uses its graph.
- **About half of the default network's score belongs to no sensor**: the dense
  layers' constant is 51 ± 4 % of the score's size (82 % before training,
  paired p = 0.0088). With the recurrent stage it is 11 ± 5 %. The constant does
  not depend on gamma.
- **The shares cancel heavily on the default, and training does not make that
  worse**: 7.1 against 7.7 before training (paired p = 0.75). A list of the
  largest walks, read without that ratio, shows mostly cancellation.
- **Rankings of connections move little between the authors' setting and the
  others**: 0.901 ± 0.043 against gamma 0, 0.994 ± 0.004 against
  (0, 0.5, 0.25) and 0.984 ± 0.009 against (0, 1, 1) on the default; above
  0.99 with the recurrent stage. Where everything is joined to everything,
  gamma changes nothing, because every row is the same.
- **Seeds agree on which connections carry most, except with the recurrent
  stage.** Spearman's rank correlation between every pair of seeds, over the
  connections they share: 0.77 to 0.97 on the default, 0.78 to 0.98 on the
  correlation graph, 0.27 to 0.77 with the recurrent stage.
- Nothing here says the paper's graph helps or hurts. That needs networks
  retrained on another graph (assumption A-27), not a reading of one network.

The five stored runs, each measured in its own folder under `walks/`:
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z`
(`19dceb5_20261002T174613Z`),
`fd001_physics_row_cap_5seed_varsplit_window_summary_cfgba82e3_5602773-dirty_20260914T233435Z`
(`19dceb5_20261002T182022Z`),
`fd001_physics_row_cap_5seed_varsplit_lstm32_cfg72f506_f386291-dirty_20260914T134955Z`
(`19dceb5_20261002T181845Z`, the best pass restored, not the paper's 200-pass procedure),
`fd001_correlation_row_cap_5seed_varsplit_cfg7fdfce_f386291-dirty_20260914T150213Z`
(`19dceb5_20261002T175956Z`) and
`fd001_complete_row_cap_5seed_varsplit_cfgf21547_f386291-dirty_20260914T150527Z`
(`19dceb5_20261002T180317Z`). Until 2026-10-02 this list named the folders
measured at `692d2d4-dirty` on 2026-10-01; they are kept beside the new ones,
and every column they share with them is identical. The folders measured at
gamma 0 on 2026-09-26 are kept too, as the record of what that code computed.
`notebooks/xai/03_walks_fd001.ipynb` reads the newest stored run of the
default, which since 2026-10-02 is
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z`, the
default re-trained from a clean `f4f40ad` (assumption A-21). Its five networks
give the same test predictions as the first run above on every seed, and its
walk measurement, `19dceb5_20261002T205544Z`, equals that run's in every column.

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
| Paper's graph, last cycle (the default) | −0.26 ± 0.14 | 0.41 ± 0.26 | 0.003 | 0.013 |
| Paper's graph, window summary | 0.25 ± 0.35 | 0.06 ± 0.45 | 0.534 | 0.178 |
| Paper's graph, recurrent stage | 0.37 ± 0.14 | 0.12 ± 0.31 | 0.224 | 0.004 |
| Correlation graph, last cycle | 0.62 ± 0.05 | 0.39 ± 0.17 | 0.036 | < 0.0001 |
| Every sensor joined, last cycle | 0.25 ± 0.01 | 0.20 ± 0.12 | 0.344 | < 0.0001 |

What flowed is measured under the authors' setting of gamma (assumption A-29),
added up with what arrives at a group of indistinguishable sensors at the last
graph layer split equally among them (finding F-18). These figures replace the
ones measured at gamma 0 on 2026-09-26, when the default network's trained
correlation was −0.16 ± 0.20 and could not be told from zero (p = 0.158).

On the default network, **what flowed along a connection does not predict what
removing it does; if anything the two rank the connections in opposite
orders**: the rank correlation is −0.26 ± 0.14, below zero (p = 0.013), and
training moved it down from 0.41. On the correlation graph and with the
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

**What it means for the Shapley values inside a group (added 2026-10-02).**
"Unaffected" above means unaffected by the random start. The audit of
2026-09-08 drew a second consequence from the same fact (its item D-1), which
this entry had not recorded. A Shapley value adds one sensor at a time to sets
of others and watches the prediction; a sensor left out is replaced by its
average training window (assumption A-20). With the last-reading stage a group
member reaches the prediction only through the group's total, so adding member
`i` moves the prediction by an amount that depends on `i` only through how far
its reading sits from that stand-in, its displacement. Two members with the same
displacement get the same Shapley value, and swapping two members'
displacements swaps their Shapley values. **The split of importance inside a
group is set by where each reading sits, not by anything the network does
differently with one member**, because it can do nothing differently.

On test engine 1, with seed 42 of the default stored run
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z`
loaded on 2026-10-02:

| | P15 | Nf |
|---|---|---|
| displacement of the last reading, rescaled units | +0.174072 | −0.301754 |
| Shapley value, cycles | −0.427663 | +0.844460 |
| Shapley value after swapping the two displacements | +0.844460 | −0.427663 |

The prediction, 108.618858 cycles, does not change, and no other sensor's
Shapley value moves by more than 1.2e-07 cycles. Over the 100 test engines:

| | P15 | Nf | NRf | BPR |
|---|---|---|---|---|
| mean size of the Shapley value, cycles | 0.923 | 4.414 | 4.285 | 3.743 |
| mean size of the displacement | 0.259 | 0.780 | 0.777 | 0.714 |

Both rank the four the same way, and on 97 of the 100 engines the four
members' order by the size of their Shapley value is their order by the size of
their displacement. It is not all 100 because the network's response to the
group total is not a straight line, so a Shapley value is not a fixed multiple
of its displacement. So a statement such as "the network relies on Nf more
than on P15" is not one this architecture can support for two members of a
group; eight of the fourteen sensors are in one. P15's place at the bottom of
the ranking (finding F-05) says its readings sit close to their average, which
finding F-01 explains, not that the network uses it less than Nf.

With the window-summary stage (assumption A-28) the same holds for each
member's three numbers. With the paper's recurrent stage it does not hold as
stated: each sensor's own recurrent network turns its window into the number
the graph receives, so members can be treated differently before the graph, and
the displacement that matters is in that number. That is an argument; it has not
been measured. `tests/test_examples_torch.py::test_example_swapping_two_group_members_swaps_their_shapley_values`
shows the swap on a real test window.

```python
import numpy as np, torch
from thesis_gnn_rul_xai.training.stored import read_run, load_seed
from thesis_gnn_rul_xai.explain.shap import shap_values, mean_window_baseline

model, prepared = load_seed(read_run(
    "fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z"), 42)
stand_in = mean_window_baseline(prepared.train)
window = prepared.test.features[0].astype(np.float32)           # test engine 1
i, j = (list(prepared.sensor_symbols).index(s) for s in ("P15", "Nf"))
swapped = window.copy()
swapped[i] = stand_in[i] + (window[j] - stand_in[j])
swapped[j] = stand_in[j] + (window[i] - stand_in[i])
with torch.no_grad():
    print(model.predict_cycles(torch.from_numpy(np.stack([window, swapped]))))
print(shap_values(model, window, stand_in)[[i, j]], shap_values(model, swapped, stand_in)[[i, j]])
```

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

- Saxena et al. (2008), [the dataset paper](references/saxena_2008_damage_propagation_modeling.pdf),
  Figure 1 and Figure 2 supply the component and flow-path arrangement; Table 2
  supplies the dataset variable definitions. These are the placement authority.
  (Corrected 2026-10-02: this line linked NASA report 20090029214, which turned
  out, when the cited papers were collected, to be an earlier draft with a
  different title and numbering. Figures 1 and 2 and Table 2 carry the same
  numbers in both versions, so the choice above is unaffected.)
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

## A-32 — The learning-rate sensitivity uses one decade below and above the paper's rates

**Where:** `scripts/run_sensitivity.py`

**The gap.** Table 6 of Wen et al. gives a learning rate that starts at 0.001
and ends at 0.0001. It does not report whether the result is sensitive to the
size of those rates. The schedule between them is a separate missing detail,
recorded in assumption A-17.

**What we chose.** The sensitivity study keeps the ten-to-one ratio and tests
three scales: `0.0001 -> 0.00001`, the paper's `0.001 -> 0.0001`, and
`0.01 -> 0.001`. Every other setting stays at the reference value for this
comparison. The lower and upper pairs are experiment settings, not values
attributed to the paper and not new defaults.

**Why.** Moving both endpoints by one power of ten gives one clearly smaller
and one clearly larger update scale while preserving the relationship between
the two numbers printed in the paper. A narrower interval could miss a real
effect; a wider one is more likely to compare failed training with successful
training rather than measure ordinary sensitivity.

**How much it matters, measured.** Over seeds 42 through 46 on FD001, one tenth
of both rates raised mean RMSE by 0.428 cycles (13.663 ± 0.676, paired
p = 0.274), which cannot be distinguished from zero. Ten times both rates
raised it by 1.372 cycles (14.607 ± 0.307, p = 0.0094). That second p-value is
below 0.05 on its own, but not below the conservative threshold of 0.00294 for
all seventeen comparisons in the sensitivity study (assumption A-37). The other C-MAPSS
sub-datasets were not run. The complete record is finding F-19.

---

### F-19 — On FD001, the temporal stage dominates the measured one-factor changes

**Established from.** Eighteen configurations at `fd65a3b-dirty`, each trained
with seeds 42 through 46, stored under
`reports/sensitivity_fd65a3b-dirty_20260929T101059Z`. Each comparison changes
one available choice around the recurrent-stage reference. The graph widths,
dense-layer width, batch size, epoch limit, first seed, recurrent width,
numbers passed per sensor, label ceiling, window length, stride and smoothing
weight were held fixed. `summary.csv` holds all four test measures and their
paired differences; `REPORT.md` explains the complete table.

**The reference.** One dedicated long short-term memory network per sensor,
the paper's physics graph, row graph rescaling, no graph-layer offset, z-score
sensor scaling, smoothing, the one-drop learning-rate path, 20% of training
engines held back and early stopping gave **13.234 ± 0.723 RMSE**.

**The resolved changes.** Passing only the final sensor reading raised RMSE by
**3.456 cycles** (16.690 ± 0.229, paired p = 0.00097). Passing the window
average raised it by **9.200 cycles** (22.434 ± 0.516, p = 0.000032). Those are
the only differences below `0.05 / 17 = 0.00294`, a conservative correction
for the seventeen comparisons (assumption A-37).

Multiplying both learning rates by ten raised RMSE by 1.372 cycles
(14.607 ± 0.307, unadjusted p = 0.0094), which does not pass that correction.
None of the other temporal, graph, rescaling, bias, smoothing, scaling,
learning-rate-path or lower-rate comparisons could be distinguished from zero
over these five paired seeds. This is not evidence that those components never
matter: two earlier comparisons in this project reversed when the surrounding
architecture changed.

**Two equalities need mechanisms, not rankings.** Per-condition and global
range scaling are identical on FD001 because it has one operating condition.
The step, plateau and constant learning-rate paths also produced identical
predictions: every reference run selected pass 5 to 24 and stopped by pass 44,
before the step at pass 100, and the plateau path did not reduce the rate before
stopping.

**A scoring change, measured without retraining.** Scoring the same reference
predictions against uncapped test answers raised RMSE from 13.234 ± 0.723 to
**14.398 ± 0.732**, a paired increase of 1.164 cycles (p = 0.000027). It changes
the answers used to score the predictions, not the predictions themselves.

**The held-back engines (added 2026-10-03).** Everything above is measured on
the test engines. Every run of the study also recorded its error on its
held-back engines, which pair by seed in the same way, because the seed alone
decides which engines are held back. That recorded figure is the lowest
held-back error over all passes, and it also chose which pass was kept
(assumption A-18), so it flatters every run, not necessarily equally. Paired
with the reference's 14.184 ± 0.389 on it, five of the seventeen comparisons
fall below the same bar of 0.00294 (assumption A-37), each on all five seeds.
They are four changes, because the two range scalings are identical on FD001:
passing only the final reading, +6.958 (p = 0.000004), and the window average,
+7.541 (p = 0.00053); the 41 most strongly correlated pairs as the graph,
−0.690 (p = 0.0027, assumption A-27); and range scaling, per condition or
global, −1.234 (p = 0.0029, assumption A-03). Two more are lower on all
five seeds without clearing the bar: the GRU stage, by 0.999 cycles
(13.185 ± 0.634, p = 0.0047), where its test difference is −0.416 (p = 0.455);
and one tenth of both learning rates, by 0.179 (p = 0.0065). Assumption A-36
sets the GRU's held-back error out on summaries no pass was chosen by.
Held-back and test errors are never compared with each other (assumption A-06).
Recomputed from each run's `results.csv`.

---

## A-33 — The GNN-LRP paper's own check is run on this network by cutting a sensor's connections

**Where:** `explain/flipping.py` (`node_flipping`, `subgraph_scores`), `scripts/explain.py`, `scripts/measure_walks.py`

**The gap.** The GNN-LRP paper checks every explanation against the network it
describes, by node flipping (its Section 4.1): sensors are added to an empty
graph in the order the explanation rates highest (its Equation 14), or removed
from the full one in the order it says matters least (its Equation 15), and the
network's output is followed at each step; the mean of each curve, the area
under the flipping curve, is set beside the same check with simpler rankings
and random orders. For walks, the paper rates a set of nodes by every bag of
edges lying wholly inside it. It does not say how a network built on a fixed,
rescaled graph is to be run with nodes missing, nor how many random orders make
its random baseline, and it allows, without requiring it, an occasional coarser
step that adds the node with the largest share on its own.

**What we chose.**

1. **A sensor is taken out by zeroing its row and its column of the rescaled
   graph the network was trained with, its link to itself included, and nothing
   is rescaled again.** Its readings stay in the window, but with no connection
   left nothing of them reaches the score through the graph layers.
2. **What is followed is the score before the sigmoid**, the quantity the walks
   divide (assumption A-29): while adding, the score itself; while removing,
   its distance from the full graph's score.
3. **The rankings set beside the walks are the paper's own**: the walks added
   up by their starting sensor, which the paper says matches the node-level
   method of Pope et al.; the paper's simpler GNN-GI and its starting sensors;
   and random orders, here **20**, averaged, each drawn from a generator seeded
   by the window's position.
4. **No coarser steps**: every step is the greedy one.

**Why.** The paper only says the network is run on the smaller graph: its
Fig. 6 plots the change from the output on "the input graph before pruning".
Its authors' published code holds no step that runs a network with nodes
removed; the one place it masks a graph, for GNNExplainer (the paper's
Section 4, item (c)), it multiplies the mask into the graph it has already
rescaled and does not rescale again. Zeroing a sensor's entries of the
rescaled graph does the same here. Rescaling again, as assumption A-30 does to
remove one connection, would also change what every remaining sensor
receives, so a sensor's removal would be mixed with a change to all the
others. The random count and the absence of coarse steps are the plainest
readings of what it leaves open. The order of flipping follows the paper's
Equations 14 and 15, not the authors' published code, which instead scores
randomly drawn orders and, while adding, draws only sensors next to one
already added.

**Correction, 2026-10-02.** Until this date the reason above said the paper
"feeds its networks the masked connectivity, `f(M * Lambda)`". That expression
appears in the paper only where it describes GNNExplainer, not in its
Section 4.1; another session found it while checking every citation against a
copy of arXiv version 3. The procedure is unchanged; only its reason was
wrong.

**How much it matters, measured.** The same runs, seeds, test windows and
commit as the table in assumption A-29. Mean ± spread over seeds of each seed's
average over the 100 test windows, trained networks; adding is better higher,
removing is better lower:

| Stored run | Task | Walks, authors' setting | Their starting sensors | Walks, GNN-GI | Random | Walks against random, paired p (seeds better) | Walks against starting sensors, p | Authors' setting against GNN-GI, p |
|---|---|---|---|---|---|---|---|---|
| Paper's graph, last cycle (the default) | adding | 1.408 ± 0.094 | 1.195 ± 0.138 | 1.410 ± 0.096 | 0.982 ± 0.162 | 0.0014 (5/5) | 0.013 | 0.31 |
|  | removing | 0.645 ± 0.085 | 0.896 ± 0.095 | 0.653 ± 0.092 | 1.053 ± 0.100 | < 0.0001 (5/5) | < 0.0001 | 0.14 |
| Paper's graph, window summary | adding | 2.156 ± 0.266 | 2.045 ± 0.226 | 2.134 ± 0.254 | 1.551 ± 0.238 | 0.0001 (5/5) | 0.069 | 0.24 |
|  | removing | 0.679 ± 0.069 | 0.924 ± 0.079 | 0.679 ± 0.079 | 1.320 ± 0.126 | < 0.0001 (5/5) | < 0.0001 | 0.99 |
| Paper's graph, recurrent stage | adding | 0.886 ± 0.168 | 0.748 ± 0.120 | 0.882 ± 0.167 | 0.438 ± 0.115 | 0.0003 (5/5) | 0.012 | 0.28 |
|  | removing | 0.824 ± 0.037 | 1.154 ± 0.038 | 0.823 ± 0.037 | 1.343 ± 0.082 | < 0.0001 (5/5) | < 0.0001 | 0.073 |
| Correlation graph, last cycle | adding | 1.860 ± 0.169 | 1.371 ± 0.076 | 1.869 ± 0.173 | 0.810 ± 0.089 | < 0.0001 (5/5) | 0.0007 | 0.026 |
|  | removing | 0.352 ± 0.029 | 0.535 ± 0.035 | 0.360 ± 0.025 | 0.993 ± 0.032 | < 0.0001 (5/5) | 0.0005 | 0.16 |
| Every sensor joined, last cycle | adding | 1.267 ± 0.066 | 1.267 ± 0.066 | 1.267 ± 0.066 | 1.082 ± 0.065 | < 0.0001 (5/5) | 0.0068 | — |
|  | removing | 0.732 ± 0.024 | 0.880 ± 0.028 | 0.732 ± 0.024 | 0.935 ± 0.031 | < 0.0001 (5/5) | < 0.0001 | — |

- **By the paper's own check, the walks describe every network here better than
  chance**: on all five configurations and both tasks they beat the random
  orders on every seed.
- **The walks beat their own starting sensors** on every configuration's
  removing task and on four of five adding tasks (the window summary's, at
  p = 0.069, cannot be told apart), so what the walks add over a node-level
  ranking is visible in the network's own output, as the paper reports for its
  networks.
- **The authors' setting and GNN-GI cannot be told apart** on nine of the ten
  rows; on the correlation graph's adding task GNN-GI is slightly ahead
  (1.869 against 1.860, p = 0.026). So on these networks the setting changes
  the walks (assumption A-29) but not how well they pass this check. Where
  every sensor is joined to every other the two settings give identical walks,
  so no test is possible.
- The check shows that the ranking describes the network; it says nothing about
  engines, and nothing about whether the paper's graph is useful.

**Reproduce with:**

```bash
python scripts/measure_walks.py --run <stored run directory>
```

Each walk explanation the playground or `scripts/explain.py` makes stores the
same check for every window it explains, in `flipping.csv`.

---

### F-20 — The dataset paper's scoring equation prints its two constants the other way round from its own text and figure

**Established from.** Saxena et al. (2008), the copy in `docs/references/`
(`saxena_2008_damage_propagation_modeling.pdf`, SHA-256 beginning `e7aaef80`),
page 7 rendered at eight times its size and read, and Figure 9 on page 8
rendered at eight times and measured against its own axis labels, on
2026-10-02.

**What the paper prints.** Its Eq. (11), with
`d = estimated RUL - true RUL`, scores `d < 0` with `exp(-d/a1) - 1` and
`d >= 0` with `exp(d/a2) - 1`, and states `a1 = 10, a2 = 13`. As printed, a
prediction 50 cycles late scores `exp(50/13) - 1 = 45.8` and one 50 cycles
early scores `exp(50/10) - 1 = 147.4`: late predictions cost **less**.

**What the same paper says otherwise.** The sentence directly above the
equation says late predictions were penalised more heavily than early ones.
Figure 9 agrees with the sentence: the x-axis lies at pixel row 1189 and the
label 150 is centred at row 537, so one unit of score is 4.35 pixels; the curve
ends at 45.5 at an error of -50 and at 147.2 at +50. The steep side is the late
side, so the figure divides late errors by 10 and early ones by 13.

**What this changes.** Substituted the same way, the printed Eq. (11) gives
exactly what Wen et al.'s Equations 20 and 21 give when read literally: the
gentler 13 to late predictions. Assumption A-15 said the challenge's published
function "also says the opposite" of Wen et al.'s equations; its printed
equation does not. What supports the convention this project implements is the
words of both papers and the dataset paper's Figure 9, against both papers'
printed equations. The convention is unchanged; its support is narrower than
A-15 stated.

---

## A-34 — The recurrent stage starts from PyTorch's own default values, so its forget gates start near one half

**Where:** `models/temporal/LSTMStage` (its recurrent cells and its `project` layer); the seed is set by `scripts/train.py` just before `build_model`

**The gap.** The paper prints the recurrent cell's equations and never the
values its numbers start from. Its Section 2.2 (page 4; in this copy the PDF
page and the printed page are the same) writes the forget gate as Equation 1,
`f_t = σ(W_f·[h_{t−1}, x_t] + b_f)`, and says of `b_f` only that it
"represents the bias vector": an offset, a number added before the sigmoid
squashes the result into the range 0 to 1. Equations 2, 3 and 5 give the input
gate, the candidate values and the output gate one offset each in the same way,
and Fig. 2 draws the cell with no values on it. Section 3.4 and Table 6
(page 9) list the learning rate, 200 passes over the training data, batches of
32, 30 time steps, Adam and the mean squared error, and nothing about starting
values. Searching the copy's text for "initial", "Xavier", "Glorot", "Kaiming",
"uniform" and "random" finds no sentence on the subject. "Xavier", "Glorot",
"Kaiming" and "uniform" do not occur. "Random" occurs once, of test records
that end at random points before failure (page 7). "Initial" occurs six times:
once of the node features the first graph layer receives (page 6), and five
times of the early part of an engine's life or of its wear (twice on page 9,
once on page 11, twice on page 13).
Assumption A-14 records how the graph layers and the two dense layers after
them start, and assumption A-26 the stage's width, depth and projection; until
this entry nothing recorded how the recurrent cells start.

The paper the forget gate comes from does say. Gers, Schmidhuber and Cummins
(2000), in the copy in `docs/references/` (the authors' manuscript, which has no
text layer, so its pages were rendered at three times their size and read on
2026-10-02):

- **Section 3.1** (PDF page 6, the manuscript's page 5): the offsets of the
  input and output gates start negative and those of the forget gates
  positive, so that at the start of training the forget gate is close to 1 and
  the cell behaves like the 1997 cell, which keeps its whole memory, "until it
  has learned to forget".
- **Section 4.2** (PDF page 12, the manuscript's page 11): the values used.
  The input and output gates of successive blocks of cells start at −0.5,
  −1.0, −1.5 and so on; the forget gates at +0.5 for the first block, +1 for
  the second, and so on; every other weight is drawn from −0.2 to +0.2. The
  same paragraph says the exact starting offsets are not critical and other
  values work as well.
- **Section 4.5** (PDF page 16, the manuscript's page 15): on a task where the
  network is reset after every sequence, forget gates starting at +5.0 learned
  as fast as the 1997 cell, and gates starting at +1.0 took about three times
  as long on average, because the network first had to learn not to forget.

**What we chose.** Nothing in the code sets a starting value in the stage, so
PyTorch's own rule applies:

1. **Every number in each recurrent cell**, LSTM or GRU, is drawn evenly from
   −1/√h to +1/√h, where h is the hidden size: from −0.176777 to +0.176777 at
   the default h = 32 (assumption A-26). That is the whole of the cells'
   `reset_parameters` in both versions of PyTorch this project's stored runs
   record: 2.13.0, in the five runs stored directly under `reports/` since late
   on 2026-09-14, three of them five-seed runs of the recurrent stage, and
   2.5.1, in all 64 runs the sensitivity study (finding F-19), whose reference
   uses the recurrent stage, had stored by 2026-10-02. Earlier runs record no
   version, among them those behind finding F-16 and the recurrent-stage run
   assumption A-28 is compared with. The table below comes out the same,
   number for number, under both versions.
2. **Each gate has two offsets** where Equation 1 prints one: one added to the
   reading's term and one to the previous hidden state's (`bias_ih` and
   `bias_hh`). Both are drawn the same way, so a gate's total offset starts
   between −2/√h and +2/√h, centred on zero, and the forget gate gets no
   positive start. The GRU cell has no forget gate; its update gate, which
   sets how much of the previous memory is kept, starts in the same range.
3. **The projection** from the last hidden state to the one number per sensor
   (assumption A-26) is a PyTorch linear layer and keeps that library's
   default for one: its multipliers and its offset drawn evenly from
   ±1/√(number of inputs), the same ±0.176777 for 32 inputs. Assumption A-14
   states this default for the two dense layers after the graph layers; it
   does not name the projection.
4. **The draws follow the run's seed.** `scripts/train.py` sets it just before
   it builds the network, and the stage is built before the graph layers, so
   its numbers are drawn first.

**Why.** The paper gives no starting values, so any choice here is this
project's, and keeping the library's default adds no number that neither Wen et
al. nor anyone reproducing them could know. Gers et al.'s values were not
carried over. They are set per block of cells and grow with the block's
position, while PyTorch gives each of the 32 cells its own gates, so "the first
block" has no counterpart here; taken literally, +0.5 per block would start the
32nd cell's forget gate at +16. Their own Section 4.2 calls the exact values not
critical. This is not a claim that the default is the better start: their
Section 4.5 is evidence the other way for a network reset after every sequence,
and this stage starts every 30-cycle window from an empty memory.

**How much it matters.** Not measured. The table measures the starting values
only; what they do to training and to the test error is an argument.

Measured on the default stage, built the way `scripts/train.py` builds it, for
the five seeds of a five-seed run (42 to 46). Fourteen sensors with 32 cells
each give 448 forget gates per seed. "At a zero reading" is the gate's value at
the first cycle of a window, where the previous hidden state is zero, for a
rescaled reading of zero, the sensor's training average: the sigmoid of the two
summed offsets.

| Seed | Largest starting number in the cells | Summed forget offsets, lowest to highest | Positive, of 448 | Forget gate at a zero reading: lowest, average, highest |
|---|---|---|---|---|
| 42 | 0.176776 | −0.319376 to +0.328242 | 211 | 0.420828, 0.499410, 0.581332 |
| 43 | 0.176775 | −0.339548 to +0.330730 | 219 | 0.415919, 0.498105, 0.581937 |
| 44 | 0.176776 | −0.303028 to +0.341626 | 231 | 0.424818, 0.499787, 0.584586 |
| 45 | 0.176775 | −0.345936 to +0.338435 | 197 | 0.414368, 0.494837, 0.583810 |
| 46 | 0.176776 | −0.300566 to +0.324980 | 253 | 0.425419, 0.504403, 0.580537 |

Every forget gate on every seed starts between 0.414368 and 0.584586, inside
the bounds the rule allows, 0.412521 and 0.587479, which
`docs/resources/lstm.md` computes the same way. Gers et al.'s offsets would
start their first two blocks at 0.622459 and 0.731059.

The argument: at the start of training a cell multiplies its memory by its
forget gate at every cycle, and with that gate near one half, what the first
cycles of a window put into the memory has largely faded by the 30th until
training raises the gates. The gate also depends on the reading and on
the previous hidden state, through multipliers drawn from the same small range,
so this is a description of the start, not a calculation of it. Whether the
stage learns more slowly for it, or ends at a different test error, has not
been measured, and every number this log gives for the recurrent stage
(findings F-16 and F-19, assumption A-28) was produced with PyTorch's default
starting values, as far as the record shows: no committed version of the stage
has set its own.
Measuring it needs an option that sets the forget offsets, which the code does
not have, and five seeds with and without it under the configuration of the
day.

**Reproduce with:**

```python
import torch
from thesis_gnn_rul_xai.graph.adjacency import build_physics_adjacency, normalize_adjacency
from thesis_gnn_rul_xai.models.config import ModelConfig, build_model

graph = normalize_adjacency(build_physics_adjacency()[0], method="row")
h = 32
for seed in range(42, 47):
    torch.manual_seed(seed)                       # as scripts/train.py does
    stage = build_model(graph, ModelConfig(temporal="lstm"), window_size=30, rul_max=125).temporal
    largest = max(p.abs().max().item() for p in stage.cells.parameters())
    forget = torch.stack([c.bias_ih_l0[h:2 * h] + c.bias_hh_l0[h:2 * h] for c in stage.cells]).detach()
    gate = torch.sigmoid(forget)
    print(seed, round(largest, 6), round(forget.min().item(), 6), round(forget.max().item(), 6),
          int((forget > 0).sum()), round(gate.min().item(), 6), round(gate.mean().item(), 6),
          round(gate.max().item(), 6))
```

PyTorch orders each cell's gates input, forget, cell, output, so rows 32 to 63
of an offset vector belong to the forget gate.
`python scripts/verify_claims.py --only starts` recomputes the table, the
lowest and highest gate over all five seeds and the two bounds, and fails if
one disagrees with what is written here.

---

### F-21 — As printed, the paper's graph layers can only scale with their input, and after the first one each sensor's 32 numbers come from one

**Established from.** The paper's Equations 8 to 10 (page 6) and Table 4
(page 7), read in the copy in `docs/references/` on 2026-10-02, and seed 42 of
the default stored run
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z`,
loaded at that day's code. The audit of 2026-09-08 raised it (its item D-7)
and said it was worth an entry; until this date it had none.

**What the paper prints.** Each graph layer is `H = ReLU(Ã H W)`: mix each
sensor's numbers with its neighbours' through the rescaled graph `Ã`, multiply
by a weight matrix `W`, then replace every negative result by zero, which is
what ReLU does. Equations 12 and 13 add an offset, a fixed learned number, to
each of the two dense layers after the graph; Equations 8 to 10 add none to the
graph layers, and this project follows them (assumption A-12). Table 4 gives the
first graph layer one number per sensor, `[batch, 14, 1]`, and `W(1)` of shape
1 × 32.

**What follows, for any weights.**

1. **Scaling the input scales the output.** Multiply every sensor's number by a
   positive `c`. Mixing through `Ã` and multiplying by `W` are multiplications,
   so the result is `c` times as large; and replacing negatives by zero commutes
   with a positive `c`, because a negative entry stays negative and a positive
   one stays positive. With numbers: ReLU([2, −1]) = [2, 0], and
   ReLU([6, −3]) = [6, 0] = 3 × [2, 0]. Layer after layer, the whole graph stage
   `G` obeys `G(c·x) = c·G(x)` for every `c > 0`. It fails for `c = −1`:
   ReLU([−2, 1]) = [0, 1], not −[2, 0]. With no offset there is no level the
   graph stage can compare an input against; the first place one can enter is
   the offset of the first dense layer.
2. **After the first layer, 32 numbers carry one.** Sensor `i` receives one
   number `s_i`, its row of `Ã` times the fourteen inputs, and the layer returns
   `ReLU(s_i · w)`, where `w` is the one row of `W(1)`. If `s_i` is positive that
   is `s_i · ReLU(w)`; if negative, `|s_i| · ReLU(−w)`. With three numbers in
   place of 32: for `w = [0.5, −2, 1]`, `s = 2` gives ReLU([1, −4, 2]) =
   [1, 0, 2] = 2 × ReLU(w), and `s = −3` gives ReLU([−1.5, 6, −3]) = [0, 6, 0]
   = 3 × ReLU(−w). Every sensor's 32 numbers lie along one of two fixed
   directions, chosen by the sign of `s_i`, at a length set by its size.

**Measured on the stored network**, over the 100 FD001 test windows, where the
largest number the graph stage returns is 0.826:

| `c` | 0.5 | 2 | 10 | −1, the control |
|---|---|---|---|---|
| largest difference between `G(c·x)` and `c·G(x)` | 0 | 0 | 1.4e-06 | 0.91 |

The 1.4e-06 is the rounding of 32-bit arithmetic. After the first layer, all
1,400 pairs of an engine and a sensor lie exactly on one of the two directions
(largest difference 0); 743 of them have a negative `s_i`. Of the 32 entries of
`w`, 15 are positive and 17 negative, so one direction has 15 non-zero numbers
and the other 17. The whole network does not scale, because its dense layers
have offsets and its output passes a sigmoid: engine 1 is predicted at 108.619
cycles from its readings and 118.737 from twice them.

**Why it matters.** The width of the first graph layer suggests 32 numbers per
sensor; the layer can produce no more than one number and a sign. The
network's 61,729 parameters therefore overstate what its graph stage can
compute, from the side of the weights, as finding F-18 does from the side of
the graph.

**What an offset in the graph layers would change, and what it would not.**
Until a correction the same day (2026-10-02) this paragraph said an offset
"would lift the limit", as though both properties above came from having none.
Only the first does. Give the first layer an offset `b`, 32 more learned
numbers, and it returns `ReLU(s_i · w + b)`. That ends the scaling and the two
directions: each of the 32 numbers becomes its own bent line in `s_i`, zero
until `s_i` passes its own threshold, so which of them are switched on changes
with the size of `s_i` and not only with its sign. With three numbers again,
`w = [0.5, −2, 1]` and `b = [−1, 1, 0.5]`: `s = 2` gives
ReLU([0, −3, 2.5]) = [0, 0, 2.5], and `s = 4` gives ReLU([1, −7, 4.5]) =
[1, 0, 4.5], which is not twice the first. What an offset cannot change is that
the first layer works on one number per sensor, `s_i`: each sensor's 32 numbers
are still all worked out from that one number, and everything after them from
the fourteen inputs. That comes from Table 4's one input number per sensor, not
from the missing offset; only a stage that hands each sensor more than one
number changes it, as the next paragraph says. The one measurement of an
offset, around the recurrent stage, is under assumption A-12, and finds no
difference five seeds can resolve.

**An offset with the default's last-reading stage (added 2026-10-03).** The
default trained again with an offset in each graph layer (`--gcn-bias`),
everything else as `config/fd001.yaml` sets it, five seeds, from a clean
checkout of `19dceb5`:
`fd001_physics_row_cap_5seed_varsplit_net9f3068_cfgba82e3_19dceb5_20261002T225228Z`,
paired by seed with the default re-trained at `f4f40ad`,
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z`
(between those two commits nothing that training runs changed but comments).
Test RMSE 16.735 ± 0.435 with the offset against 16.496 ± 0.415 without,
+0.239 cycles, per seed `[0.016, 0.106, 0.587, 0.601, −0.116]`, paired
p = 0.185; held-back RMSE 21.465 ± 0.756 against 21.168 ± 0.870, +0.297, higher
on all five seeds, p = 0.025; NASA score 522.0 ± 95.0 against 539.8 ± 44.7,
p = 0.72. So for the default network the offset buys nothing five seeds can
show on the test engines, and is nominally worse on the held-back ones.

**Which models it applies to.** The paper's own: its recurrent stage hands the
graph layers one number per sensor (Table 4), and the graph layers are those
printed. With the window-summary stage (assumption A-28), three numbers per
sensor, the first property still holds and the second does not, because
`W(1)` then has three rows.

**What it does not show.** That the printed architecture is a mistake. (Until
2026-10-03 this also said it does not show how much accuracy an offset would
buy the default network, "which has not been trained with one"; it has now,
in the paragraph above.)

`tests/test_examples_torch.py::test_example_the_graph_layers_scale_with_their_input_and_keep_two_directions`
checks both properties on real test windows.

```python
import numpy as np, torch
from thesis_gnn_rul_xai.training.stored import read_run, load_seed

model, prepared = load_seed(read_run(
    "fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z"), 42)
x = torch.from_numpy(prepared.test.features.astype(np.float32))

def graph_stage(inputs):
    hidden = model.temporal(inputs)
    for layer in model.graph_layers:
        hidden = torch.relu(layer(hidden, model.adjacency))
    return hidden

with torch.no_grad():
    for c in (0.5, 2.0, 10.0, -1.0):
        print(c, float((graph_stage(c * x) - c * graph_stage(x)).abs().max()))
    s = (model.adjacency @ model.temporal(x)).squeeze(-1)
    w = model.graph_layers[0].weight.reshape(-1)
    rays = torch.where(s[..., None] >= 0, s[..., None] * torch.relu(w), -s[..., None] * torch.relu(-w))
    first = torch.relu(model.graph_layers[0](model.temporal(x), model.adjacency))
    print(float((first - rays).abs().max()), int((s < 0).sum()))
```

---

## A-35 — Graph *g* of the paper's Table 12 and Fig. 10 is read as the engine's *g*-th window

**Where:** `evaluation/published.py` (`SHAP_TABLE_12_GRAPHS`); `docsite/assets/playground.js` (`paperComparison`, which sets the stored engine beside the paper's graphs); `explain/README.md`, under "Trap: on the default network, the four largest values change almost every window"; `notebooks/xai/01_shap_fd001.ipynb`, step 8; `docsite/playground/README.md`, under "Beside the paper's Fig. 10"

**The gap.** Table 12 (page 14) prints Shapley values of the 14 sensors for
what the text calls "a representative engine unit" (page 13), without naming
the engine or its sub-dataset. Its first column is headed "Time" and its
fifteen others G1 to G5, G78 to G82 and G187 to G191. Fig. 10, on the same
page, draws one small graph of the sensors per column, titled "Graph #1" to
"Graph #191", and the text calls graphs 1 to 5 the early degradation stage and
187 to 191 the advanced one. Section 2.4 says the dynamic analysis "operates
on each time step of the prediction sequence", computing each sensor's value
"at the t-th time step" (the paragraph introducing Equation 16, page 7), and
Section 3.4 sets the window to "30-time steps" (page 9). Nowhere does the paper
say which cycle of the engine a graph number is: whether graph 1 is the first
full window, the first cycle, or a window counted back from the engine's
failure. Checked on 2026-10-03 by rendering page 14 at scale 8 with
`pypdfium2` (the header cell reads "Time", the figure's titles "Graph #1" to
"Graph #191") and by searching the text layer of every page for "time step",
"window", "Graph #" and "representative engine".

**What we chose.** Graph *g* is the engine's *g*-th window. Windows are 30
cycles long and one cycle apart (assumption A-05), and the first ends at the
engine's 30th cycle, so graph *g* is the window of cycles *g* to *g* + 29:
graph 1 ends at cycle 30, graph 78 at cycle 107 and graph 191 at cycle 220. An
engine needs at least 220 cycles for graph 191 to exist; 27 of the 100 FD001
training engines and 4 of the 100 FD001 test engines have that many (counted
from `train_FD001.txt` and `test_FD001.txt` on 2026-10-03).

**Why.** It is the plainest reading of what is printed: one explanation per
time step of the prediction sequence, numbered from 1 at the first prediction.
A prediction needs a full window, and this project leaves out what is shorter
rather than padding it (assumption A-23), so reading graph *g* as cycle *g*
would put graphs 1 to 29 at windows that do not exist. Counting back from the
engine's failure fits the text equally well, and the measurement below covers
it.

**How much it matters, measured.** The reading matters only where this
project sets its own explanations beside Table 12's: the count of **joins**
over the 12 steps from one graph to the next inside graphs 1 to 5, 78 to 82
and 187 to 191. The **red four** of a graph is its four sensors with the
largest values, by size, computed from Table 12's numbers; Fig. 10 colours
those four red in thirteen of its fifteen graphs, and colours other sensors
red at graphs 81 and 82 (finding F-17). A join is a sensor that is in the red
four of one graph and was not in the red four of the graph before
(`explain/README.md` sets this out with an example). Counted on Table 12's
values there is one join, phi at graph 191; counted on Fig. 10's colours there
would be three, T30 at graph 81 and Nc at graph 82 as well. Every count below
takes the red four from values, Table 12's or this project's, never from
colours. Every reading that keeps neighbouring graph numbers one window apart
puts graph *g* at the window ending at cycle *g* + 29 + *k*, for one
shift *k* from 0, which is ours, to the number of the engine's windows minus
191, which puts graph 191 at the engine's last window. Every such shift was
counted on the two stored runs `explain/README.md` uses, which differ only in
the temporal stage: the default network
(`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z`)
and the recurrent one
(`fd001_physics_row_cap_5seed_varsplit_lstm32_cfg72f506_f386291-dirty_20260914T134955Z`),
both at `f386291-dirty`, each seed explained at every window of its held-back
engine with the most windows (engines 5, 96, 95, 92 and 67 for seeds 42 to 46,
with 240 to 312 windows, so 50 to 122 shifts each and 447 in all). The
explanations were computed on 2026-10-03 at `19dceb5-dirty`, on the processor:

| | Joins in the 12 steps, our reading | Graph 191 at the engine's last window | Fewest and most over every shift | Shifts with one join or none, of 447 |
|---|---|---|---|---|
| The paper's Table 12 | 1 | — | — | — |
| Default network (`last_cycle`) | 20.2 ± 4.8 | 13.6 ± 3.2 | 8 and 28 | 0 |
| Recurrent network (`lstm`) | 4.6 ± 2.1 | 3.4 ± 0.9 | 0 and 11 | 27 |

(Means ± standard deviations over the five seeds.) At all 447 shifts the
default network has more joins than the recurrent one on the same seed and
engine, and more than Table 12's one. So what `explain/README.md` concludes
about the default network, that its red four changes far more often than
Table 12's and than the recurrent network's, holds under every reading of this
kind. Its statement that the recurrent network has more joins than Table 12 on
every seed holds under ours and under the reading from the engine's failure
(2, 4, 3, 4 and 4 joins), but not under every shift: at 27 of the 447, on
seeds 42, 43, 44 and 46, it has one join or none. That remaining gap was
never put down to the network anyway, since Table 12's values are magnitudes
computed in a way the paper does not state (finding F-08).

**What this does not settle.** A reading in which neighbouring graph numbers
are more than one window apart, for instance if the paper explained only
every few windows, is not covered by the shifts. And the engines are not the
paper's, which it does not name, so no reading makes the comparison with
Table 12 a reproduction.

**Reproduce with**, from the repository root with no training, about four
minutes on the processor:

```python
import numpy as np

from thesis_gnn_rul_xai.evaluation.published import SHAP_TABLE_12, SHAP_TABLE_12_GRAPHS
from thesis_gnn_rul_xai.explain.seeds import shap_for_each_seed
from thesis_gnn_rul_xai.training.stored import read_run

sensors = list(SHAP_TABLE_12)
# The 12 steps inside graphs 1-5, 78-82 and 187-191, as pairs of Table 12's columns.
steps = [(k, k + 1) for start in (0, 5, 10) for k in range(start, start + 4)]


def red_four(values):
    order = sorted(range(len(values)), key=lambda i: -abs(values[i]))
    return {sensors[i] for i in order[:4]}


def joins(rows):
    return sum(len(red_four(rows[after]) - red_four(rows[before])) for before, after in steps)


print("Table 12:", joins([[SHAP_TABLE_12[s][c] for s in sensors] for c in range(15)]))
runs = {"last_cycle": "fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z",
        "lstm": "fd001_physics_row_cap_5seed_varsplit_lstm32_cfg72f506_f386291-dirty_20260914T134955Z"}
counted = {}
for stage, name in runs.items():
    # Each seed's held-back engine with the most windows, every window explained.
    for explained in shap_for_each_seed(read_run(name), device="cpu"):
        (unit, table), = explained.tables.items()
        table = table.sort_values("end_cycle")
        assert table["end_cycle"].iloc[0] == 30 and (np.diff(table["end_cycle"]) == 1).all()
        values = table[sensors].to_numpy()
        # Shift k puts graph g at the window ending at cycle g + 29 + k; k = 0 is the reading used,
        # the last k puts graph 191 at the engine's last window.
        counts = np.array([joins(values[[g - 1 + k for g in SHAP_TABLE_12_GRAPHS]])
                           for k in range(len(values) - SHAP_TABLE_12_GRAPHS[-1] + 1)])
        counted[stage, explained.seed] = counts
        print(stage, explained.seed, "engine", unit, "shifts", len(counts), "first", counts[0],
              "last", counts[-1], "fewest", counts.min(), "most", counts.max(), "at 1 or fewer", (counts <= 1).sum())
for stage in runs:
    rows = [c for (s, _), c in counted.items() if s == stage]
    first, last = np.array([c[0] for c in rows]), np.array([c[-1] for c in rows])
    print(f"{stage}: first {first.mean():.1f} ± {first.std(ddof=1):.1f}, last {last.mean():.1f} ± {last.std(ddof=1):.1f}, "
          f"over every shift {min(c.min() for c in rows)} to {max(c.max() for c in rows)}, "
          f"at 1 or fewer {sum((c <= 1).sum() for c in rows)} of {sum(len(c) for c in rows)}")
higher = sum((counted["last_cycle", seed] > counted["lstm", seed]).sum() for seed in range(42, 47))
print("shifts at which the default network has more joins than the recurrent one:", higher)
```

`tests/test_examples.py::test_example_reading_graph_g_as_another_window_moves_the_red_four_count_but_not_its_order`
checks seed 42's counts on the two stored explanations of engine 5.

---

## A-36 — The GRU stage puts a gated recurrent unit of the LSTM stage's size where each sensor's LSTM goes

**Where:** `models/temporal/` (`LSTMStage` built with `cell="gru"`, by `build_temporal_stage("gru", ...)`); `scripts/train.py --temporal gru`; `notebooks/temporal/05_gru_fd001.ipynb`

**The gap.** This is not a gap in the paper but a stage it does not have.
Section 2.2 (page 4) builds the temporal stage from LSTM networks only: "Each
sensor s∈{1, 2, …, n} is associated with a dedicated LSTM network", and
Equations 1 to 6 are that network's gates and memory. "GRU" appears in the
paper only as a row of Tables 7 to 10 (page 10), "GRU [55]", one of the
published methods the paper sets its network beside, with an RMSE of 18.31 on
FD001 in Table 8. Its reference 55 (page 16) is Zou and Lin (2025),
"Multichannel attention-based TCN-GRU network for remaining useful life
prediction of aero-engines", so that row is not a plain GRU either, and it is
not the paper's network with another recurrent cell. Checked on 2026-10-03 by
searching the text layer of every page for "GRU" and "gated recurrent".

**What we chose.**

1. **A gated recurrent unit** (Cho et al., 2014) in place of each sensor's
   LSTM: fourteen of them, one per sensor, as Section 2.2 gives the LSTM, or
   one shared by all with `--shared-across-sensors`, as for the LSTM. Where the
   LSTM carries two lists of numbers from cycle to cycle, its hidden state and
   its cell state, and has three gates, the GRU carries one list and has two
   gates: a reset gate, which decides how much of the old memory goes into a
   proposed new one, and an update gate, which decides slot by slot how much of
   the old memory to keep and how much to replace by the proposal.
2. **The LSTM stage's size, with no choice of its own:** hidden size 32
   (`--temporal-hidden`), one layer, and one linear projection, shared by the
   fourteen sensors, from the last cycle's hidden state to the one number per
   sensor that Table 4 prints (assumption A-26).
3. **PyTorch's version of the cell** (`nn.GRU`). It multiplies the reset gate
   into the old memory's contribution after the memory's multipliers and offset
   have acted, `r ⊙ (U h + c)`, where `⊙` multiplies slot by slot; Cho et al.'s
   Equation 8 (Section 2.3, page 3 of the copy in `docs/references/`)
   multiplies it in before, `U (r ⊙ h)`, and has no offset; the comparison
   below adds PyTorch's offset `c` afterwards, unscaled. The two orders
   generally give different numbers, because PyTorch scales each slot's mixed
   memory and its offset by that slot's own gate, while Cho et al.'s order
   scales each slot of the old memory by its own gate before the slots are
   mixed. They agree only in special cases, for instance if every slot's reset
   gate were exactly 1.
   On real data: T50's trained
   GRU in seed 42 of the stored run below, run over the 30 readings of the
   last window of held-back engine 5 (cycles 240 to 269), gives memories that
   match PyTorch's to within 1.7e-07 at every cycle when computed by hand in
   PyTorch's order, and differ from them by up to 0.505 in Cho et al.'s order;
   the number handed to T50's node would be 2.267 instead of 2.072 (checked
   2026-10-03; notebook `temporal/05` works through the same window by hand).
4. **Starting values** are PyTorch's own, as for the LSTM (assumption A-34,
   which covers both cells).

**Why.** The stage asks whether the paper's particular recurrent cell matters,
at a lower cost. A GRU has three blocks of multipliers and offsets where an
LSTM has four, so one of hidden size 32 reading one number learns 3,360
numbers against 4,480; the fourteen with their projection learn 47,073 against
62,753, and the whole network 108,802 against 124,482 (summed over the
networks `build_model` builds, 2026-10-03). Borrowing assumption A-26's sizes
leaves the cell as the only difference between the two stages; a GRU sized on
its own would leave any difference between them impossible to put down to the
cell. PyTorch's version is taken because it is what the library builds, and
the stage is not offered as Cho et al.'s exact cell.

**How much it matters, measured.** One stored five-seed run uses the stage:
the sensitivity study's `temporal_gru` (finding F-19),
`reports/sensitivity_fd65a3b-dirty_20260929T101059Z/runs/FD001/temporal_gru`,
at `fd65a3b-dirty`, trained beside the study's reference, the same settings
with the LSTM stage, from the same commit on a graphics card under PyTorch
2.5.1: z-score scaling, the paper's graph rescaled by rows, capped test labels,
a fifth of the training engines held back with early stopping and the best
pass restored (so not the paper's 200-pass procedure, finding F-16), and the
step schedule. Paired by seed:

| Error, cycles | LSTM per sensor (the reference) | GRU per sensor | GRU minus LSTM | Seeds with the GRU lower | Paired p |
|---|---|---|---|---|---|
| test RMSE | 13.234 ± 0.723 | 12.818 ± 0.681 | −0.416 | 4 of 5 | 0.455 |
| held-back, the recorded lowest | 14.184 ± 0.389 | 13.185 ± 0.634 | −0.999 | 5 of 5 | 0.0047 |
| held-back, mean of the last 20 passes | 15.463 ± 0.738 | 14.510 ± 0.787 | −0.954 | 4 of 5 | 0.042 |
| held-back, mean of passes 1 to 25 | 15.322 ± 0.757 | 14.477 ± 0.711 | −0.845 | 5 of 5 | 0.012 |
| held-back, pass 25 alone | 15.220 ± 0.559 | 14.475 ± 0.568 | −0.745 | 4 of 5 | 0.107 |

The held-back rows are the four summaries assumption A-27 uses: the lowest
error over all passes, which each run records and which also chose the pass
kept, so it flatters every run (assumption A-18); and three that no pass was
chosen by, 25 being the fewest passes either of the ten runs made.

On the test engines the two cells cannot be told apart, on RMSE (per seed
`[−1.121, −1.445, +1.393, −0.812, −0.097]`) or on the other three measures:
NASA score 309.4 ± 62.6 for the GRU against 316.3 ± 57.2 (p = 0.857),
R-squared 0.897 against 0.891 (p = 0.448), ASUE 8.310 against 8.146
(p = 0.725), from each run's `results.csv`. On the held-back engines the GRU is
lower on every summary, by 0.75 to 1.00 cycles, and on the recorded figure on
every seed (`[−1.370, −1.061, −0.661, −0.531, −1.374]`), but no p clears the
study's bar of 0.05 / 17 = 0.00294 (assumption A-37). The GRU kept passes 10,
8, 9, 12 and 6 and stopped after 26 to 32 passes; the LSTM kept passes 17, 24,
9, 15 and 5 and stopped after 25 to 44.

So on FD001, in this configuration, the GRU stage cannot be told apart from
the LSTM stage on the test engines, and on the held-back engines its recorded
error is lower on every seed, by about a cycle, which five seeds at the study's
bar cannot establish. No test was run of whether the GRU is at least as good as
the LSTM; on one seed its test RMSE is 1.393 higher, and its mean ASUE is
higher.

**What this does not settle.** Any other size; the paper's own procedure of
200 passes with nothing held back, under which the LSTM stage is worth much
less (finding F-16) and the GRU has not been trained; a run on the processor,
whose networks can differ seed by seed from these graphics-card ones
(assumption A-28 measures how much for the LSTM stage); and FD002 to FD004.
The study of those three sub-datasets, still running in
`reports/sensitivity_fd65a3b-dirty_20260930T160040Z`, has a GRU run on each;
they are not reported here, because that study is unfinished and its runs do
not all record the same state of the code. No published row may be set beside
this stage: the paper's "GRU [55]" is another method.

**Reproduce with:**

```bash
python scripts/train.py --config config/fd001.yaml --temporal gru --seeds 5
```

The stored run was made by `scripts/run_sensitivity.py` with `--device cuda`;
its full command is in its `summary.json`. The table, from the repository root
with no training:

```python
from pathlib import Path

import pandas as pd
from scipy import stats

study = Path("reports/sensitivity_fd65a3b-dirty_20260929T101059Z/runs/FD001")
seeds = list(range(42, 47))
arms = ("reference", "temporal_gru")   # the LSTM stage, then the GRU stage
error = {(arm, s): pd.read_csv(study / arm / f"seed_{s}" / "training_history.csv")
         ["validation_rmse_cycles"].to_numpy() for arm in arms for s in seeds}
shortest = min(len(e) for e in error.values())
summaries = {
    "lowest over all passes": lambda e: e.min(),
    "mean of the last 20 passes": lambda e: e[-20:].mean(),
    f"mean of passes 1 to {shortest}": lambda e: e[:shortest].mean(),
    f"pass {shortest} alone": lambda e: e[shortest - 1],
}
rows = {"test RMSE": [pd.read_csv(study / arm / "results.csv").set_index("seed").loc[seeds, "rmse"]
                      .reset_index(drop=True) for arm in arms]}
rows.update({label: [pd.Series([f(error[arm, s]) for s in seeds]) for arm in arms]
             for label, f in summaries.items()})
for label, (lstm, gru) in rows.items():
    gap = gru - lstm
    print(f"{label:26s} LSTM {lstm.mean():.3f} ± {lstm.std():.3f}  GRU {gru.mean():.3f} ± {gru.std():.3f}  "
          f"{gap.mean():+.3f}, GRU lower on {(gap < 0).sum()} of 5, p {stats.ttest_rel(gru, lstm).pvalue:.4f}, "
          f"per seed {gap.round(3).tolist()}")
```

`tests/test_examples_torch.py::test_example_pytorch_gru_multiplies_the_reset_gate_in_after_the_memory_multipliers`
checks the comparison of the two orders on T50's trained GRU.

---

## A-37 — When several comparisons are read together, a difference counts only below 0.05 divided by their number

**Where:** finding F-19, its held-back addition, and assumptions A-03, A-32 and A-36 (0.05 / 17); assumption A-27 (0.05 / 3 and 0.05 / 4); `notebooks/temporal/07_comparison_fd001.ipynb`, step 3 (`BAR = 0.05 / tests`); `notebooks/graph/04_comparison_fd001.ipynb` (0.05 / 4 and 0.05 / 3)

**The gap.** The paper reports one number per method, measure and
sub-dataset (Tables 7 to 11) and tests none of its differences: it gives no
repetition with other seeds, no spread and no test. Searching the copy's text
layer on 2026-10-03 for "p-value", "t-test", "Bonferroni", "significan", "seed",
"±", "standard deviation", "statistic" and "confidence" finds "significant"
only as an ordinary adjective, as in "significant improvements over the
best-performing baseline methods" (page 10); "confidence intervals" once, in a
sentence saying SHAP does not address them (page 7); "standard deviation" once,
describing what RMSE measures (page 9); and "statistical measure" once, of the
Pearson correlation (page 8). So the paper sets no bar for calling a difference
real. This project compares configurations with a paired t-test over five
seeds (working rule 3), and several of its studies make many such comparisons
at once, so it has to choose one.

**What we chose.** When a set of comparisons is read together, a difference is
called established only if its p is below 0.05 divided by the number of
comparisons in the set, the correction known as Bonferroni's, after the
inequality it rests on (Dunn, 1961, set it out for many comparisons at once).
Wherever the bar is used, the set and its size are named:

| Where | The comparisons read together | Bar |
|---|---|---|
| finding F-19; assumptions A-03, A-32 and A-36 | the seventeen changes of the study of one setting at a time on FD001, each against its reference | 0.05 / 17 = 0.00294 |
| assumption A-27; notebook `graph/04` | the three summaries of held-back error no pass was chosen by, added to test the recorded one | 0.05 / 3 = 0.0167 |
| assumption A-27; notebook `graph/04` | the four test measures of one comparison | 0.05 / 4 = 0.0125 |
| notebook `temporal/07` | every pair of temporal stages trained at one commit, each on test and on held-back error: 7 pairs and 14 tests when it was last executed | 0.05 / 14 = 0.0036 |

A p below 0.05 but above the bar is reported with its value and called
unresolved, never a result.

**Why.** Start from one test. If two configurations do not differ on average,
a paired test at 0.05 still calls their difference real in one experiment in
twenty; that 0.05 is its false-alarm rate. Read seventeen such tests together
and the chance that at least one of them raises a false alarm is far larger: if
the seventeen were independent, it would be 1 − 0.95^17 = 1 − 0.418 = 0.582.
Dividing 0.05 by the number of tests, k, brings that chance back to 0.05 or
below, because the chance that at least one of k events happens is never more
than the sum of their chances, here k × 0.05 / k = 0.05. That bound needs
nothing about how the tests depend on each other, which matters here: the
seventeen comparisons of the study all use the same five reference networks and
the same 100 test engines, so they are not independent, and a correction that
assumed they were would not hold.

The bar is conservative, in two ways. Tests that share a reference tend to err
together, and then the real chance of any false alarm is below 0.05, so the
bar is stricter than its guarantee needs. And a stricter bar asks much more of
five seeds. The paired t-test takes each seed's difference between the two
configurations and divides their average by its standard error, the typical
distance of such an average from the true one; that ratio is called t. The p is
the chance of a t at least that far from zero, on either side of it (so
"two-sided"), if the two configurations were equally good on average. With five
seeds that chance is read from a curve whose tails are heavier the fewer seeds
there are: the one for four "degrees of freedom", the five differences less
the one used up in estimating their average. `evaluation/README.md`, under
"The paired t-test", works one through with real numbers. On that curve a p
below 0.05 needs |t| above 2.776, below 0.0167 above 3.961, below 0.0125 above
4.315, below 0.0036 above 6.138, and below 0.00294 above 6.469.
A real difference of the size these studies look for is therefore missed more
often than at 0.05 alone. That is the error this project can afford: a missed
difference is reported with its p as unresolved and can be measured again with
more seeds, while a false one would be quoted.

**How much it matters, measured** on the study of finding F-19, against two
other ways of handling many tests: Holm's procedure (Holm, 1979), which tests
the smallest p against 0.05 / 17, the next against 0.05 / 16, and so on,
stopping at the first that fails, with the same guarantee and never fewer
differences called real; and Benjamini and Hochberg's (Benjamini & Hochberg,
1995), which bounds the expected share of false alarms among the differences
called real rather than the chance of any.

| Measure | Below 0.05 alone | Below 0.05 / 17 | Holm | Benjamini and Hochberg |
|---|---|---|---|---|
| test RMSE | 3 | 2 | 2 | 2 |
| held-back RMSE, as recorded | 7 | 5 | 5 | 7 |

On the test engines the bar sets aside one change, ten times both learning rates
(p = 0.0094). On the held-back engines it sets aside two, the GRU stage
(p = 0.0047, assumption A-36) and one tenth of both learning rates
(p = 0.0065), which Benjamini and Hochberg's procedure would keep. The
seventeen include three comparisons that are not separate experiments: global
and per-condition range scaling are identical on FD001, and the plateau and
constant schedules reproduce the reference exactly. Counted once each, there
are fourteen, and at 0.05 / 14 = 0.00357 the same changes clear the bar and
the same ones do not. Elsewhere the bar decides three held-back summaries in
assumption A-27, which clear 0.05 and not 0.0167: the every-sensor graph's mean
of the last 20 passes (p = 0.0278) and of passes 1 to 23 (p = 0.0184), and,
with the recurrent stage, the correlation graph's mean of passes 1 to 25
(p = 0.037); and it leaves the correlation graph's NASA score (p = 0.017)
unresolved among four test measures. Those four figures are the ones that entry
reports.

**What it does not do.** It says nothing about how a difference would come out
on engines outside these 100: every p here is over five seeds scored on one set
of engines (the audit of 2026-09-08, its item D-8). And it does not turn a
comparison chosen after seeing the results into a test decided in advance: the
mean-error comparison in assumption A-27, run to explain two measures already
seen, stays a follow-up whatever its p.

**Reproduce with**, from the repository root with no training:

```python
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

study = Path("reports/sensitivity_fd65a3b-dirty_20260929T101059Z/runs/FD001")
reference = pd.read_csv(study / "reference" / "results.csv").set_index("seed")
changes = sorted(p.name for p in study.iterdir() if p.name != "reference")
for measure in ("rmse", "validation_rmse"):
    p = {}
    for name in changes:
        other = pd.read_csv(study / name / "results.csv").set_index("seed")
        # Two schedules give the reference's predictions exactly; their test is undefined.
        p[name] = np.nan_to_num(stats.ttest_rel(other[measure], reference[measure]).pvalue, nan=1.0)
    k = len(p)
    ranked = sorted(p, key=p.get)
    holm = []
    for place, name in enumerate(ranked):          # smallest p against 0.05 / k, the next 0.05 / (k - 1) ...
        if p[name] >= 0.05 / (k - place):
            break
        holm.append(name)
    passing = [place for place, name in enumerate(ranked, start=1) if p[name] <= 0.05 * place / k]
    fdr = ranked[:max(passing, default=0)]         # Benjamini and Hochberg
    print(f"{measure}: {k} comparisons; below 0.05 {sum(v < 0.05 for v in p.values())}, "
          f"below 0.05 / {k} = {0.05 / k:.5f} {sum(v < 0.05 / k for v in p.values())}, "
          f"Holm {len(holm)}, Benjamini-Hochberg {len(fdr)}")
    print("   ", ", ".join(f"{name} {p[name]:.2g}" for name in ranked[:8]))
for bar in (0.05, 0.05 / 3, 0.05 / 4, 0.05 / 14, 0.05 / 17):
    print(f"two-sided p below {bar:.5f} with five seeds needs |t| above {stats.t.ppf(1 - bar / 2, df=4):.3f}")
print(f"chance of at least one false alarm in 17 independent tests at 0.05: {1 - 0.95 ** 17:.3f}")
```

---

## A-38 — The ordinary models' predictions are clipped to between 0 and the label ceiling

**Where:** `scripts/baselines.py` (`fit_and_predict`); `evaluation/README.md`, under "Ordinary models beside the network", "The ladder"

**The gap.** The paper sets its network beside published methods in Tables 7
to 10 (page 10), each row citing a published source. Section 4 introduces
them as "established baseline methods" and Section 4.1 as "evaluated methods"
(both page 10), without saying whether their numbers were copied from those
sources or obtained again, how they were trained, or whether their predictions
were held to any range. This project's comparison with ordinary models fits
its own, with no sensor graph, on the same training engines as each stored
network (`scripts/baselines.py`), and has to decide what to do with a
prediction outside the range of the labels. Assumption A-02 caps the test
labels at 125 because the network cannot predict more; it says nothing about a
model that can.

**What we chose.** Every prediction of an ordinary model is clipped to between
0 and the run's label ceiling, 125 on FD001 (assumption A-01), before it is
scored: one below 0 becomes 0, one above 125 becomes 125. The stored networks'
predictions are not touched. Each model's RMSE without the clip, and how many
of its predictions the clip moved, are written beside the clipped figures
(`rmse_unclipped` and `n_clipped` in `per_seed.csv`). This is a separate choice
from the cap on the test labels (assumption A-02), not part of it: the cap
decides what the answers are, the clip decides what an ordinary model is
allowed to answer.

**Why.** The network's last step squeezes its output between 0 and 1 and
multiplies it by the ceiling (assumption A-13), so it cannot predict outside
that range. The comparison asks whether the graph network does better than an
ordinary model given the same information, not whether an ordinary model can
be charged for answers the network is prevented from giving.

Which way the clip can move a result follows from the labels. Every capped
FD001 test label lies between 7 and 125, so a prediction outside 0 to 125 is
moved towards its label and never past it: its error keeps its direction and
shrinks, or reaches 0 at a label of 125. Two real cases, from linear regression
on the 42 window summaries under seed 42 (the run below): test engine 78,
labelled 107 and predicted 139.6, has an error of +32.6 before the clip and
+18.0 after it; test engine 34, labelled 7 and predicted −26.85, has −33.85
before and −7.0 after. No engine's error grows, so RMSE cannot rise, R-squared
cannot fall (its denominator depends on the labels alone, assumption A-16), and
the NASA score, which charges each error more the larger it is on either side
(assumption A-15), cannot rise. ASUE, the average shortfall over the engines
predicted below their label, cannot rise either: a prediction clipped down to
125 is below no label, and one clipped up to 0 is still below its label, so the
same engines are averaged, with smaller shortfalls. On that seed the clip moved
11 of the 100 predictions and took RMSE from 15.933 to 14.642, the NASA score
from 417.3 to 365.1, R-squared from 0.842 to 0.866 and ASUE from 12.930 to
10.865. The clip can therefore favour the ordinary models against the network,
or leave a model's scores as they were when it moves none of its predictions,
but never the other way round, which is the safe direction for a
comparison whose question is whether the network does better.

**How much it matters, measured.** `scripts/baselines.py` run on 2026-10-03,
finishing at 08:42 UTC, against the default network re-trained from a clean
`f4f40ad` (`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z`),
with the code at `19dceb5-dirty` and scikit-learn 1.9.0, written to a scratch
folder rather than `reports/`, so it is not kept; a run at 22:39 UTC on
2026-10-02, at the same commit, gave the same `per_seed.csv` in every column
but the seconds. Five seeds of 100 test engines give 500 predictions per
model:

| Model | Predictions the clip moved, of 500 | RMSE without the clip | RMSE with it | Lowered by |
|---|---|---|---|---|
| linear regression, 42 window summaries | 47: 35 below 0, 12 above 125 | 15.816 ± 0.135 | 14.520 ± 0.120 | 1.295 |
| linear regression, the network's 14 numbers | 10: 1 below 0, 9 above 125 | 20.601 ± 0.225 | 20.457 ± 0.219 | 0.144 |
| one-hidden-layer network of 64, the same 14 numbers | 5, all above 125 | 17.317 ± 0.198 | 17.213 ± 0.223 | 0.104 |
| gradient boosting, each of its seven sets of window summaries | 4 to 13, all above 125 | | | 0.004 to 0.024 |
| the constant, boosting on shuffled labels, boosting on engine age, and the random forest | 0 | | | 0 |

No model's comparison with the network changes direction on any seed. The one
it moves is linear regression on the 42 window summaries: without the clip it is
ahead of the network by 0.681 cycles (p = 0.020), with it by 1.976
(p = 0.0002), and ahead on every seed both ways. So the clip makes the strongest
linear model look about 1.3 cycles better, and the conclusion that a linear
model given the window summaries beats the default network on every seed does
not rest on it.

**Reproduce with:**

```bash
python scripts/baselines.py --run fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z
```

It writes a new folder under `reports/`; the columns `rmse`, `rmse_unclipped`
and `n_clipped` of its `per_seed.csv` give the table, and its `predictions.csv`
holds every clipped prediction. It took 499 seconds on the processor, with
other work sharing the machine. The two engines above come from the script's
own `rebuild_and_verify` and `fit_and_predict` for seed 42, which return the
prediction before and after the clip.

---

## References

Full citations for every work mentioned above are in `REFERENCES.md`.
