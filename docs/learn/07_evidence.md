# Evidence: keeping three things apart

Everything on this site is one of three kinds of statement, and the value of a
re-implementation study depends entirely on a reader being able to tell which
is which.

---

## The three kinds

### "The paper says X"

A statement about the paper cites it the way a journal article cites its
sources: by authors and year, with the table, equation, section or page where
it can be checked, as in (Wen et al., 2026, Table 3). It is checked against the
PDF rather than stated from memory. The References list at the end of this
page gives each work cited here in full.

Figures need more care. A value read off a small page image is a guess, so the
region is rendered at high resolution before a number is read off an axis. The
label ceiling of 125 was settled that way, by measuring the flat top of the true
remaining-life curve in the paper's Fig. 7 (Wen et al., 2026, p. 11) against
that figure's own axis ticks. The paper's Fig. 6 draws the ceiling at 120 (Wen
et al., 2026, p. 9), but it is a schematic drawn with round numbers, not a plot
of data (assumption A-01).

Where the paper prints something twice, both copies are checked. The sensor
graph's matrix appears as numbers in Table 3 and as a colour grid in Fig. 3
(Wen et al., 2026, pp. 5–6); both were checked against this project's
transcription of the matrix, and all 196 cells, 14 sensors by 14, agree.

### "We chose Y" — an assumption

The paper does not specify everything needed to rebuild the method. Where a
value or a procedure had to be chosen, it gets a numbered entry containing four
things:

- **The gap** — quoting what the paper does and does not say.
- **What was chosen.**
- **Why** — the reasoning, not just the conclusion.
- **How much it matters** — and if that is an argument rather than a
  measurement, the entry says so plainly.

The entry's number then appears in the code where the choice is used, in a
comment that **says what the choice was in words**, not just its number. The
one for the learning rate reads: "drop the learning rate once, by a single
factor, at the halfway point. The paper gives the first and last rate but never
how it gets between them." The paper prints the two rates in its Table 6 and
describes the route between them only as "an adaptive learning rate schedule"
(Wen et al., 2026, p. 9).

Documents for readers follow the same rule: a sentence says what the choice is,
and the number follows in parentheses, as in "one drop of the learning rate at
the halfway point (assumption A-17)". `python scripts/check_prose.py` fails when
a number is used as a word.

### "We measured Z" — a result

A number this code produced, given with the command that produced it and
traceable to the commit, the saved version of the code, that made it. A trained
model's number is never one training run's, for the reason
[One seed is not a result](06_reading_results.md#one-seed-is-not-a-result)
gives. Nothing is estimated, and no figure is carried over from an earlier
draft without being recomputed.

---

## Assumptions are ours; findings are the paper's

An **assumption** is a choice this project made. A **finding** is a checked
fact, nearly always about the paper. Confusing the two would present this
project's own decisions as facts about its source. Some of the findings:

- **The paper's own rules for building its graph require an edge that its
  printed matrix does not contain** (Wen et al., 2026, Section 2.3.1 and
  Table 3; finding F-07).
- **The paper's Table 12 prints only positive Shapley values**, although its
  own Equation 14 requires their sum to turn negative late in an engine's life
  (Wen et al., 2026, Table 12 and Eq. 14; finding F-08).
- **The paper's Conclusions claim a "highest R²" that its own Table 9 refutes**
  (Wen et al., 2026, p. 15 and Table 9; finding F-11).
- **The paper gives each graph node a single number** (Wen et al., 2026,
  Table 4), which none of the ten other graph methods on the same NASA data
  read for this project does (finding F-12).

---

## Corrections stay visible

Several claims here were wrong when first written. Each was corrected with what
was wrong left on the record, because that shows what kind of mistake this
project is prone to:

- **How much `P15` contributes to each prediction, by its Shapley value,** was
  first written from 14 of one engine's 240 windows, and both how large the
  contribution was and when in the engine's life it was large came out wrong
  (finding F-05).
- **The order of smoothing and scaling** was logged as a choice when the
  paper's Fig. 1 states it outright (Wen et al., 2026, p. 3). It was withdrawn
  (assumption A-08).

One correction changed how results are stored. A results file that no
committed version of the code could reproduce stayed in the repository (finding
F-09). It happened to contradict a figure that was correct; it could as easily
have supported one that was wrong. So every run now writes its commit, marked
`-dirty` when the code had uncommitted changes, into both its summary file and
its own directory name.

---

## The machinery that keeps this true

Besides `check_prose.py`, two checks guard the claims. They run automatically on
every push to the main branch and every pull request, but on a machine without
the NASA files, where every check that needs the data skips itself:

**`pytest`** checks that the code does what it claims. Some tests check
*properties* rather than outputs, because a wrong answer in this domain still
looks plausible: that the smoothing filter only looks backwards, as the paper's
formula does (Wen et al., 2026, Eq. 18); that the test engines never influence
the scaling (assumption A-25); that held-back engines never overlap the
training engines (assumption A-06); that the parameter counts match the weight
shapes in the paper's Table 4 (Wen et al., 2026, p. 7); and that information
spreading along the graph's edges, one hop per layer (Kipf & Welling, 2017,
p. 3), cannot reach further than the graph allows.

**`python scripts/verify_claims.py`** recomputes the figures the log of
assumptions and findings quotes — the graph's edge counts, `P15`'s two levels,
how many test labels exceed the ceiling — prints each beside the recorded
value, and fails on any disagreement, because a number written into the log
goes stale silently when the code changes.

## References

- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with
  graph convolutional networks. *International Conference on Learning
  Representations (ICLR)*. https://arxiv.org/abs/1609.02907v4
  [Copy](../references/kipf_2017_graph_convolutional_networks.pdf)
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and
  Technology*, *168*, 110875. https://doi.org/10.1016/j.ast.2025.110875
  [Copy](../references/wen_2026_estgnn_dynamic_shap.pdf)
