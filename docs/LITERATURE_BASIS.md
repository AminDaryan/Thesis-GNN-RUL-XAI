# Does each assumption have a basis in a published paper?

## What this file is

`docs/assumptions/` records every choice made where Wen et al. (2026) are
silent, one file per choice. Each entry says what we chose and why. This file asks a harder
question of the same list:

> **Has anyone else published the same choice, on this dataset, and can we
> point at them?**

An assumption that some other group has made and reported is a defensible
reading of a silent paper. An assumption nobody has published is an invention,
and a thesis should say which of the two it is rather than leave a reader to
guess.

This file was written on 2026-09-01, after the observation that **A-23 — pad a
short engine by repeating its first reading — modifies the recorded data and
had no precedent behind it.** That turned out to be right, and the search below
found a paper doing the opposite.

## How the search was done

The library at `D:/Main/University/Thesis/Papers/All papers in one` holds 74
PDFs. **25 of them mention C-MAPSS or turbofan engines** and were searched;
the rest are surveys of graph networks, explainability or water networks and do
not touch this dataset.

Each paper's full text was extracted and searched for the practice in question
— sliding-window stride, label ceiling, short sequences, smoothing, test-label
capping, and so on. Every quotation below was read in the PDF. Where a paper is
named only because another paper's reference list mentions it, this file says
so and does not treat it as evidence.

---

## The finding that prompted this: A-23, short engines

**The choice at the time of this search.** When an engine had fewer rows than
the 30-cycle window, `data/windowing/` repeated its first reading backwards
until the window was full, and kept the engine. **This search is what changed
it:** A-23 was rewritten on 2026-09-01 and the code now leaves such an engine
out instead. The paragraph is left in the past tense rather than updated,
because the finding below is what prompted the change and reads as nonsense
without the choice it argued against.

**No paper in the 25 does this.** The search for padding, repeating a first
reading, or handling short sequences returned nothing on this practice. The
only hits for "padding" in these papers are *convolutional* padding — the
zeros a 1-D convolution adds at the edge of its kernel, which is a different
operation entirely and says nothing about short engines.

**One paper addresses the situation directly, and excludes those engines.**
Dida, Cheriet and Belhadj, *Remaining Useful Life Prediction Using
Attention-LSTM Neural Network of Aircraft Engines*, International Journal of
Prognostics and Health Management, write on their page 6:

> "A window size of 30 was chosen for this study. For sub-datasets FD001 and
> FD003, the window size was determined based on the engine with the shortest
> data sequence, ensuring uniformity across the training and testing datasets
> for each engine. The stride for the time windowing process was set to one. In
> the initial stages, RUL labels were assigned a constant value of 125.
> **Engines in sub-datasets FD002 and FD004 that contained fewer than 30
> samples were excluded from the study.**"

So the published practice is **exclusion**, not padding. That matters, because
padding invents sensor readings that the engine never produced and then asks a
model to predict from them.

**The cost of switching, which is real and must be weighed.** Excluding test
engines changes the size of the test set. The NASA score is a **sum** over
engines, not a mean, so it shrinks purely because there are fewer terms, and it
stops being comparable with any published score. RMSE is a mean and survives,
but its denominator changes too. Dida et al. accepted that cost; a
reproduction that wants to sit beside published FD002 and FD004 numbers may
not be able to.

**It does not affect any number this project currently reports.** FD001's
shortest test history is 31 cycles (finding F-03), so no FD001 engine is
padded. The question only becomes live when FD002 or FD004 is run, where 6 and
11 test engines respectively fall below 30.

**Recommendation.** Follow the published practice and exclude, or keep padding
and defend it explicitly against Dida et al. — but do not leave it as an
unsourced invention. Either way the entry should quote the passage above, since
it is the only published treatment found.

---

## The same paper supplies a basis for three more entries

Dida et al.'s page 6 also settles, for their own work:

| Entry | This project's choice | Dida et al. |
|---|---|---|
| **A-05** | stride of 1 | "The stride for the time windowing process was set to one." |
| **A-01** | label ceiling of 125 | "RUL labels were assigned a constant value of 125." |

A-01 already cites Heimes (2008) and Jin et al. (2025). **A-05 currently cites
nobody**, and this is a direct precedent on the same dataset with the same
window length.

## And it corroborates two findings, by disagreeing with the paper

Dida et al. also reduce the 21 sensors to 14, on page 6:

> "the readings from sensors numbered 1, 5, 6, 10, 16, 18, and 19 remain
> constant throughout the cycles. Consequently, these sensors are excluded from
> the analysis, and the remaining 14 sensor readings are selected as raw
> features"

Those seven are `T2, P2, P15, epr, farB, Nf_dmd, PCNfR_dmd`. Wen et al.'s seven
are `T2, P2, epr, NRc, farB, Nf_dmd, PCNfR_dmd`.

**Both papers keep exactly 14 sensors. They are not the same 14.**

- Dida et al. drop **`P15`**; Wen et al. keep it.
- Wen et al. drop **`NRc`**; Dida et al. keep it.

This bears on two existing entries:

- **F-01** records that `P15` takes two distinct values on FD001 and carries
  almost no information. An independent group dropped it *for that reason*.
  F-01 is no longer only our observation.
- **A-10** identifies `NRc` as the seventh sensor Wen et al. remove without
  naming, and argues it was dropped for redundancy rather than constancy.
  Dida et al. keeping `NRc` while dropping the genuinely flat `P15` supports
  that reading: `NRc` is not one of the obviously constant channels.

Note also that Dida et al. call `P15` constant, which is not exactly true — it
takes two values, 21.60 and 21.61. Our own measurement is the more precise one.

---

## A-17, A-18, A-19 and A-13, read on 2026-09-01

These were listed below as "candidates not yet read closely enough to quote".
They have now been read. One gained a basis (A-18). One did not, and the
evidence points away from the current choice (A-17). For the other two (A-19
and A-13) the survey passages first read as support turn out not to give it:
the A-19 passage describes a different graph, and the A-13 passage is about
another dataset and does not say whether its labels are divided at all, as the
corrections under each explain.

**Correction, 2026-10-02.** Until this date this paragraph said "Three gained a
basis", counting A-19 and A-13 among them. Found when the cited papers were
collected into `docs/references/` and checked against the places that cite
them.

### A-17 — the learning rate schedule. **No paper decays it exponentially.**

The paper gives 1e-3 and 1e-4 as the first and last rates and never says how it
travels between them. When this section was written, this project shrank it by
the same factor every epoch (it no longer does; see the end of this section).
Three papers state a schedule, two of them on C-MAPSS and one on its
newer and larger successor, N-CMAPSS, and **none of them does that**:

| Paper | What it does |
|---|---|
| Zhang, Liu, Xie & Liu (2025), Table 4 | `Learning rate decay stepLR: 40` — a **step** every 40 iterations |
| Li, Zhao, Sun, Yan & Chen (2021), p6 | "learning rate decay is adopted, which decreases the learning rate by **multiplying 0.1 in epochs 120 and 160**" |
| Li & Dong (2026), PDF page 10, journal page 407; N-CMAPSS only | "**When the loss is not reduced**, the learning rate is reduced at a rate of 0.1 times" |

The two on C-MAPSS both use a step decay at fixed epochs; the third reduces on
a plateau, on N-CMAPSS. Exponential decay is not among them.

**Correction, 2026-10-02.** Until this date this section said "Three C-MAPSS
papers state a schedule" and cited Li and Dong's sentence as "p10". Li and Dong
say they chose N-CMAPSS as their data (PDF page 3, journal page 400) and report
results on nothing else, so only two of the three are C-MAPSS papers. And "p10"
is the page of the PDF copy, whose first page is the publisher's cover sheet;
the sentence is on the journal's page 407. Found when the cited papers were
collected into `docs/references/` and checked against the places that cite
them. The conclusion, that no paper found decays the rate exponentially, is
unchanged.

**And there is a reason to prefer a step here.** The paper's two endpoints,
1e-3 and 1e-4, are **exactly one factor of ten apart** — precisely one `×0.1`
step of the kind Li et al. (2021) describe. An exponential curve reaches 1e-4
only at the final epoch and spends the whole run at values the paper never
mentions; a single `×0.1` step reaches it exactly and holds it. The step
reading requires the paper to have omitted *when* the step happens; the
exponential reading requires it to have omitted the mechanism entirely.

When this section was written the project implemented `"plateau"`, which
matches Li and Dong (2026) on N-CMAPSS, and `"constant"`, but no step schedule,
and the section asked that A-17 either gain a step option and be re-measured,
or state that exponential decay was chosen despite no C-MAPSS paper using it.
The first was done the same day. The project now also has `"step"`, and it is
the default: the rate is multiplied by 0.1 once, at the halfway epoch (epoch
100 of 200), so a run uses only the two rates the paper prints. A-17 records
the change and the comparison with exponential decay that was measured for it.

**Correction, 2026-10-02.** Until this date this section said, in its first
paragraph, that the project "chose to shrink it by the same factor every
epoch", and in this last one that the project "does not implement a step
schedule" and that A-17 "does neither". All three stopped being true on
2026-09-01, later on the day this section was written, when the step was
added and made the default; the section was not updated with it. Found when
the cited papers were collected into `docs/references/` and checked against
the places that cite them, while this section's count of papers was being
corrected.

### A-18 — early stopping. **Two papers do it; one gives a patience.**

- Dida, Cheriet & Belhadj, p7: "an early stopping mechanism is implemented,
  which halts the training process once the model begins to overfit the
  training data."
- Wen, Fang, Wei, Liu, Chen & Wu (2025), p7: "We set the training epochs to 50
  for both datasets, **incorporating an early stopping mechanism with a
  patience level of 10.**" (A different Wen from the paper being reproduced.)

This project uses a patience of 20. The practice is established; the particular
value is still ours, and A-18 can now cite both.

### A-19 — the fully connected comparison. **The survey's dense graphs are a different graph.**

Wen et al. name an ablation "without physics prior" and never say what replaces
the hand-designed matrix. Wang, Wu, Li, Xie and Chen (2025), a survey of graph
networks for this exact task, reports on its page 8:

> "most works simply constructed **fully-connected graphs**, where every node is
> connected to every other node, resulting in dense adjacency matrices."

The sentence begins "With these distances", and "these" are the six measures
the survey has just listed, on the same page, for setting how strong each
connection is (its Equations 1 to 6): dot product, cosine similarity, Pearson
correlation, Euclidean distance, a Gaussian kernel and Mahalanobis distance,
each computed between the two nodes' features. So in those graphs every pair
is joined, but each connection is as strong as the two nodes are alike, and
the strengths come from the data. A-19's graph gives every pair the same
strength, 1; once each sensor's link to itself is added and every row is
rescaled to add up to one, every entry is 1/14 (0.071429), so each sensor
receives the plain average of all fourteen. A graph whose strengths are
derived from the data is one of the readings A-19 set aside, so the survey
describes a different graph and gives A-19 no basis. A-19 rests on its own
argument, that a graph with no knowledge in it at all is the most literal
reading of "without physics prior".

**Correction, 2026-10-02.** Until this date this section was headed "A survey
says it is the default" and concluded that "a complete graph is what the field
reaches for when it has no structure to impose", calling that "direct support
for the substitution A-19 makes". That read the quotation without the words in
front of it. Found when the cited papers were collected into
`docs/references/` and checked against the places that cite them. (In that
copy of the survey the PDF page and the printed journal page are the same, so
"page 8" here, and pages 19 and 20 below, are both.)

### A-13 — dividing the targets by the ceiling. **Not established by the passage cited.**

The same survey, page 20: "The **maximum RUL is set to 88 to normalize the RUL
labels**, and max–min normalization is used to normalize the data." That
sentence is about N-CMAPSS, the newer and larger successor to C-MAPSS (its
subset DS02, which the survey first thins to every hundredth time step), not
about C-MAPSS. And it does not say what the 88 does to the labels. In its
C-MAPSS paragraph, on page 19, the survey adopts the piecewise label "to
normalize the sample labels" and defines that as replacing any remaining life
above a chosen maximum by the maximum, there 125: there "normalize" names a
cap, not a division. Its N-CMAPSS paragraph says it "applied similar
processing as in C-MAPSS", but the sentence gives as its reason that the
dataset records each engine's whole life, which is the C-MAPSS paragraph's
reason for cutting windows out of each engine's life, so it does not settle
what happens to the labels. Table 3 (page 19) prints 88 in its "Max RUL" row
beside C-MAPSS's cap of 125, which suggests a cap; but the same row heading
in Tables 4 and 5 (pages 19 and 20) holds each bearing's own longest remaining
life, which the bearing paragraphs on page 20 use for what reads as a
division (each bearing's labels are "normalized by its own maximal RUL"). So
88 may cap the labels, as 125 does, or divide them, and the survey does not
say which. Each engine paragraph then rescales "the data" into the range 0 to
1 by its minimum and maximum, without saying whether the labels are included.

The passage therefore does not establish the practice A-13 follows. If 88
divides the labels, the survey describes dividing them by a chosen maximum,
but on N-CMAPSS, not on C-MAPSS: a precedent on another dataset. If it caps
them, it describes a cap like the one at 125 and no division at all. The only
divisions the survey states plainly are on its two bearing datasets, PHM 12
and XJTU-SY (page 20), and they divide by a maximum read off each bearing
rather than a chosen ceiling, on another kind of machine. A-13 rests on the
argument in its entry: the network's last step (a sigmoid) can only output
numbers between 0 and 1, so targets measured in cycles have to be brought into
that range, and dividing by the ceiling does it.

**Correction, 2026-10-02.** Until this date this section was headed
"Supported" and introduced the page 20 sentence as the survey "describing its
own preprocessing", read as dividing its labels by a chosen maximum. It
describes the survey's N-CMAPSS preprocessing, which the section did not say,
and that sentence does not say whether 88 caps the labels or divides them; in
the survey's C-MAPSS paragraph "normalize the sample labels" names the cap at
125. Found when the cited papers were collected into `docs/references/` and
checked against the places that cite them.

### A-06 — holding engines out. **Nothing found, and now measured to be absent.**

The original search returned only surveys counting how often train/test splits
appear in the field, not C-MAPSS papers describing their own validation
engines, so A-06 sat in the "no precedent found" class.

**Re-checked on 2026-09-08 by reading all twelve papers directly**, the same
corpus as F-12, asking each one whether it describes holding anything out.

> **Twelve of twelve describe no validation split for C-MAPSS.**

For C-MAPSS, not one names held-out engines, a validation fraction, or what an
early-stopping criterion would monitor. The one genuine validation set in the
corpus is on another kind of data altogether: HierCorrPool (Wang, Wu, Li, Xie
and Chen 2024) splits a dataset of people's movements "into 60%, 20%, and 20%
for training, validation, and testing" (PDF page 6, journal page 326).
Elsewhere the word
means something else. Zheng et al. (2026), on a *wind-turbine* dataset, write on
page 15: "The data collected from wind turbines No. 7 and No. 18 were used for
training, while data from turbine No. 15 was used for validating model
performance." Turbine No. 15 is the only data the paper keeps out of training,
and it names no other, so the results it reports for the turbines (its Table 6,
page 14, and Fig. 16, page 15) are on that turbine. "Validating" there means
measuring the finished model, which in this project's terms makes turbine
No. 15 a test set, not a validation set used to choose settings or when to
stop. DVGTformer (Wang, Cao, Ye, Xu and Yan 2024, page 12) uses the word for
C-MAPSS: its grid search over the learning rate and the batch size took FD001
as the "validation task", a whole sub-dataset whose test results the paper
then reports beside the other three (its Tables 7 and 8, page 17), without
saying which of FD001's engines the search was scored on. So neither is
described as training data held back to choose settings: Zheng et al.'s
turbine is the data the results are reported on, and DVGTformer never says
whether its search held any engines back.

**Correction, 2026-10-02.** Until this date this paragraph called Zheng et
al.'s turbine No. 15 "the single occurrence of a genuine validation set in the
corpus". It is the set the paper's wind-turbine results are reported on. The
corpus's one genuine validation set is HierCorrPool's, which the 2026-09-08
reading missed; like the turbines it is not C-MAPSS, so the count above is
unchanged. Because HierCorrPool's split does name a validation fraction, the
sentence saying that not one paper names one was limited to C-MAPSS at the
same time; until then it carried no such limit. Found when the cited papers
were collected into `docs/references/` and checked against the places that
cite them, by searching all twelve copies for the word; DVGTformer's
"validation task", which the paragraph did not mention, was found the same
way. (In the Zheng et al. and DVGTformer copies the PDF page and the printed
journal page are the same.)

**Two of them score the test set more than once, and neither says its
settings were chosen from those scores:**

- Zhang et al. (GLSTGN) say their settings "were obtained through the grid
  search method" (Section 3.4, p. 7) and never say what the search was scored
  on. Their sensitivity study (Section 4.2, p. 9; the figure it describes,
  Fig. 9, is on p. 11) varies five settings and reports "the sensitivity
  analysis of the remaining hyperparameters **on the test results** for the
  FD001 dataset", concluding that the model is "generally robust" to them.
  That reports how the test score moves with each setting; it does not say
  the settings were picked from it. But the figure's lowest points fall on
  the settings the paper uses. In each panel of Fig. 9 that plots the error
  (RMSE), the shortest bar, the lowest error, stands at the value Table 4
  (p. 9) prints. In panel (a), which the text describes as the prediction
  results for the four sub-datasets, that is at window lengths 50, 65, 65 and
  70 for FD001 to FD004. In panels (c), (e) and (f), the FD001 test results,
  it is at 3 residual blocks, a dropout rate of 0.5 and a learning rate of
  0.01. For the number of attention heads Table 4 prints "3/5" without saying
  which number applies where, and panel (d) is lowest at 3. (Read from p. 11
  rendered at eight times its printed size, by comparing the heights of the
  bars within each panel, not by reading values off its axis.)
- Li T. et al. (HAGCN), Section 4.1.5, p. 6: "In the experiments, we **test the
  model at per epoch**, and to avoid randomness of the results, the final
  values of the two evaluation metrics are the average of the last 10 epochs."
  So the test set is scored after every pass over the training data, but the
  number reported is the average over a fixed window, the last 10 of its 180
  passes, not the best pass picked by its test score. Separately, Section 5.3
  (p. 8) varies α, the weight that sets how much an extra error term on the
  attention scores counts beside the prediction error (Eq. 10, p. 4), from 0
  to 1000, and observes that at 100 the model "can achieve the best results on
  all datasets". 100 is the value the paper trains with (p. 6, and Table 3 on
  the same page). Section 5.3 does not say which data its results are measured
  on, and the paper describes no data besides a training set and a test set.

Picking the best epoch by its test score, or sweeping settings against the
test score and keeping the winner, is model selection on the test set, and it
produces a number that cannot be compared with one produced any other way.
Neither paper says it did either. What both print is consistent with the
second: each uses the settings its own sweep shows to be best, and with no
held-out engines described, the test score is the only score either paper
reports that a grid search or a sweep could have been judged on. That is an
inference about what they did, not something either one states.

**So the honest position for this project is:** holding engines out has no
precedent in this corpus, and the corpus offers no stated alternative to it
either: the two papers above score the test set repeatedly and use the
settings their own sweeps show to be best, but never say what their choices
were made against. This project will not make them against the test set. A-06
therefore rests on the argument rather than on a citation, and the entry says
so.

**Correction, 2026-10-02.** Until this date this passage said that two papers
"state outright that the test set is what guides their choices", and that "the
alternative the corpus actually practises is tuning on the test set". Neither
paper states that, so the count of papers that say they chose settings on the
test set falls from two to none. The GLSTGN sentence was cited to "Section
4.2, p. 11"; it is on p. 9, and it describes a study of how sensitive the test
results are to each setting, not a choice made from them. The HAGCN quotation
stopped at "the two evaluation" and called the rest an average, which hid that
the average is over the last 10 epochs, a fixed window, while the paragraph
after it spoke of reporting the best. What the passage now says instead is
weaker but not empty: the lowest points of GLSTGN's Fig. 9 fall on the
settings of its Table 4, and HAGCN's Section 5.3 reports the value of α it
trains with as the best of its sweep. Both are consistent with choosing on the
test set, and neither paper says so. Found when the cited papers were
collected into `docs/references/` and checked against the places that cite
them.

**A paper outside this corpus shows the same pattern.** Jin et al. (2025),
whose scaling of each operating condition to the range 0 to 1 this project
followed until 2026-08-29 (assumption A-03), are not among the twelve papers
above, but this project cites them and holds a copy. They describe each
sub-dataset as divided into a training set and a testing set and nothing else
(PDF page 6), define both of their error measures over the test samples
(Eqs. 23 and 24, PDF page 7), and train for a fixed 100 passes over the
training data (Table II, PDF page 8), so they describe neither held-out
engines nor a rule for when to stop. Their study of the window length
(Section III-C5 and Fig. 14, PDF page 12) tries windows of 10, 30, 50 and 70
cycles on all four sub-datasets and concludes that FD001 and FD003 "achieved
optimal results" with 30 and FD002 and FD004 with 50, the windows the paper
then uses (Table II). It never says which data Fig. 14 was measured on. At
those four windows, though, the figure prints exactly the test results the
paper sets beside twelve published methods in its Tables III and IV (PDF
page 9): an RMSE of 12.25, 19.06, 12.56 and 18.27 and a NASA score of 243.91,
1343.24, 306.91 and 1214.81 on FD001 to FD004 (its Tables V and VII print
FD002's score as 1314.24 instead). And on every sub-dataset the window it
adopts has both the lowest RMSE and the lowest score of the four it tried. So
the study's numbers at the adopted windows are the paper's test results, and
the adopted windows are the best of the study: consistent with choosing the
window on the test set, which the paper does not say it did. Jin et al.'s
pages carry the article number 3538014 and no page numbers, so the pages
given are the PDF's; Fig. 14 was read rendered at eight times its printed
size.

**Added 2026-10-02.** Jin et al. were not in this passage until this date.
The lead came up when the cited papers were collected into
`docs/references/` and checked against the places that cite them; the
paper's experimental setup and results were then read in full in the copy.

**Another paper in the corpus names what its search scored (R-squared), but
not on which data.** IMTSO-GCN (Yuan et al. 2025), one of the twelve, chooses
its window length, learning rate, number of neurons, number of graph layers
and the threshold that decides which connections its graph keeps by a swarm
search: a population of candidate settings, each trained and scored, moved
round after round towards the best-scoring ones. Its scoring step reads: "A
trained model is obtained by training the training set according to the
hyperparameters in U. The results obtained from the trained model are compared
with the real results. In this paper, R2 is used as the fitness function" (PDF
page 7, journal page 180). R-squared is the share of the spread of the true
values around their mean that the predictions account for, 1 when every
prediction is exact. The paper never says which data "the real results" are.
It names a training set and a testing set and nothing else (PDF pages 8 and
9, journal pages 181 and 182); its Fig. 7 shows its FD002 testing set to be
NASA's test file, predicted at every cycle; and its Fig. 3 (PDF page 8,
journal page 181) prints a box labelled "Training sets" beside the one arrow
into the search. That leaves three possibilities open: the training engines
themselves (as Fig. 3's label suggests), a part of them held back without
mention, or the testing set, the only other data it describes. The last would
be consistent with choosing the settings on the test set, and the paper does
not say it did. Unlike GLSTGN, HAGCN and Jin et al., it prints nothing from
its search but the settings chosen (its Table 4, PDF page 12, journal page
185), so its test results cannot be checked against the search either way.
How Figs. 3 and 7 were read, and two further problems with this paper's
results, are in the finding about what other papers put on a graph node
(finding F-12, `docs/findings/F-12.md`).

**Added 2026-10-02.** IMTSO-GCN was counted among the twelve above, but its
search was not discussed in this passage until this date. The question came up
when its Fig. 6 was checked for that finding; the paper was then read in full
in the copy, with Fig. 3 rendered at eight times its printed size and Fig. 7
measured in the picture embedded in the page, at its own size. In that copy a
journal page number is the PDF page number plus 173. A review the same day,
before the passage was first committed, found that its first version called
this "a third paper" (it is not the third of anything above: GLSTGN never
says what its search scored, and DVGTformer's search is described earlier in
this section) and put the question of where the search was scored as a choice
between two possibilities when the paper leaves three open.

---

## Where the rest of the list stands

Of the numbered entries in `docs/assumptions/` and `docs/findings/`, roughly two in five cite a
supporting work — Heimes (2008) for the piecewise label, Kipf and Welling
(2017) for the graph layer, its normalisation and its initialisation, Jin et
al. (2025) for the input scaling, Shapley (1953) and Lundberg and Lee (2017)
for the attribution, Saxena et al. (2008) for the dataset itself.

The rest do not, and they fall into three different cases, which should not be
treated alike:

**1. A precedent exists and simply has not been cited yet.** A-05 is the clear
case, settled above. A-17 (learning-rate decay), A-18 (early stopping) and
A-19 (a fully connected graph as the "no-prior" comparison) were also listed
here, as having candidate hits among the 25 papers that had not been read
closely enough to quote. All three were read on 2026-09-01, and the section
above headed with that date says what each one found.

**Correction, 2026-10-02.** Until this date this paragraph still described
A-17, A-18 and A-19 as having candidate hits "not yet been read closely enough
to quote", although the section above records reading them on 2026-09-01.
Found when the cited papers were collected into `docs/references/` and checked
against the places that cite them.

**2. No paper describes the practice at all.** The search for any C-MAPSS paper
that says whether it caps the **test** labels returned **zero** — which is
exactly what A-02 already claims ("most do not say"). Here the honest entry is
not a citation but a statement that a search of 25 papers found none, with the
search recorded so someone can repeat it. Demanding a citation in this case
would only produce a decorative one.

**3. The choice is ours by design and a precedent would not justify it.**
A-11 omits the LSTM stage because the paper does not specify it; the
justification is the paper's own "w/o LSTM" ablation row, not somebody else's
practice. A-22 is the rule this project uses to pick among options the paper
leaves open — a precedent for it would be a precedent for a methodology, not
for a setting.

**A note on one paper this file does not cite as evidence.** Several of the 25
list, in their references, Shi et al. (2024), *A dual attention LSTM
lightweight model based on exponential smoothing for remaining useful life
prediction*, Reliability Engineering & System Safety 243, 109821. That is a
plausible precedent for A-04 and A-09, the two smoothing entries. It is named
here so it can be followed up, and it is **not** treated as evidence, because
the paper itself is not in the library and has not been read.

---

## What would make this file finished

1. Decide A-23 — exclude, following Dida et al., or keep padding and say why.
2. Add the Dida et al. citation to A-05, and to A-01 alongside the two it has.
3. Add the corroboration to F-01 and A-10.
4. ~~Read the candidate papers behind A-17, A-18 and A-19.~~ **Done above.**
   A-18 gained a basis; A-19, A-13 and A-06 did not; **A-17 turned up evidence
   against the current choice** and needs a decision of its own. (Until
   2026-10-02 this item counted A-19 and A-13 as gaining a basis; the survey
   passages behind both turn out not to give one, as corrected above.)
5. Obtain Shi et al. (2024) and settle A-04 and A-09.
6. For every entry where a search found nothing, say so in the entry, naming
   the search, rather than leaving the silence unexplained.

## References read for this file

Dida, M., Cheriet, A., & Belhadj, M. *Remaining Useful Life Prediction Using
Attention-LSTM Neural Network of Aircraft Engines.* International Journal of
Prognostics and Health Management. Read from
`Papers/All papers in one/Remaining Useful Life Prediction Using Attention-LSTM Neural.pdf`.

Li, T., Zhao, Z., Sun, C., Yan, R., & Chen, X. (2021). *Hierarchical attention
graph convolutional network to fuse multi-sensor signals for remaining useful
life prediction.* Reliability Engineering and System Safety, 215, 107878.

Li, T., & Dong, J. (2026). *Multi-level attention graph feature fusion smooth
prognostics approach for aircraft engines remaining useful life prediction.*
Journal of Control and Decision, 13(2), 399–415.

Wang, Y., Wu, M., Li, X., Xie, L., & Chen, Z. (2025). *A survey on graph neural
networks for remaining useful life prediction: methodologies, evaluation and
future trends.* Mechanical Systems and Signal Processing, 229, 112449.

Wen, Z., Fang, Y., Wei, P., Liu, F., Chen, Z., & Wu, M. (2025). *Temporal and
heterogeneous graph neural network for remaining useful life prediction.* IEEE
Transactions on Neural Networks and Learning Systems, 36(11). **Not** the Wen
et al. this project reproduces.

Zhang, Q., Liu, J., Xie, K., & Liu, Q. (2025). *A dynamic spatio-temporal graph
neural network with global and local perception for remaining useful life
prediction.* Huazhong University of Science and Technology.

Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). *Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction.* Aerospace Science and Technology,
168, 110875. The paper being reproduced.

Full citations for everything else named above are in `docs/REFERENCES.md`.
