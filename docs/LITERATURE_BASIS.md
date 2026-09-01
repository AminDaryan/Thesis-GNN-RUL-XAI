# Does each assumption have a basis in a published paper?

## What this file is

`docs/ASSUMPTIONS.md` records every choice made where Wen et al. (2026) are
silent. Each entry says what we chose and why. This file asks a harder
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

**The current choice.** When an engine has fewer rows than the 30-cycle window,
`data/windowing.py` repeats its first reading backwards until the window is
full, and keeps the engine.

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

## Where the rest of the list stands

Of the numbered entries in `docs/ASSUMPTIONS.md`, roughly two in five cite a
supporting work — Heimes (2008) for the piecewise label, Kipf and Welling
(2017) for the graph layer, its normalisation and its initialisation, Jin et
al. (2025) for the input scaling, Shapley (1953) and Lundberg and Lee (2017)
for the attribution, Saxena et al. (2008) for the dataset itself.

The rest do not, and they fall into three different cases, which should not be
treated alike:

**1. A precedent exists and simply has not been cited yet.** A-05 is the clear
case, settled above. Others worth searching for properly: A-17 (learning-rate
decay), A-18 (early stopping) and A-19 (a fully connected graph as the
"no-prior" comparison) all have candidate hits among the 25 papers that have
not yet been read closely enough to quote.

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
4. Read the candidate papers behind A-17, A-18 and A-19 and quote them or
   record that nothing usable was found.
5. Obtain Shi et al. (2024) and settle A-04 and A-09.
6. For every entry where a search found nothing, say so in the entry, naming
   the search, rather than leaving the silence unexplained.

## References read for this file

Dida, M., Cheriet, A., & Belhadj, M. *Remaining Useful Life Prediction Using
Attention-LSTM Neural Network of Aircraft Engines.* International Journal of
Prognostics and Health Management. Read from
`Papers/All papers in one/Remaining Useful Life Prediction Using Attention-LSTM Neural.pdf`.

Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). *Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction.* Aerospace Science and Technology,
168, 110875.

Full citations for everything else named above are in `docs/REFERENCES.md`.
