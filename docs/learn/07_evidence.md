# Evidence: keeping three things apart

Everything on this site is one of three kinds of statement. The value of a
re-implementation study depends entirely on a reader being able to tell which
is which, so this page sets out how the distinction is kept.

---

## The three kinds

### "The paper says X"

Verifiable by opening the PDF, and always cited to a specific table, equation
or section.

The rule behind these is strict: **never state a fact about the paper from
memory.** Open it and check — every table value, every equation, every sentence
quoted.

Figures need more care than text. A value read off a page thumbnail is a guess.
The region has to be rendered at high resolution before a number is read off an
axis, and how it was obtained has to be said. The label ceiling of 125 was
settled that way — by calibrating a plateau in the paper's figure against its
own axis ticks — and in the course of doing so it emerged that the paper's
schematic figure disagrees with its experimental one.

Where the paper prints something twice, both copies are checked. The adjacency
matrix appears in its Table 3 and again in its Fig. 3. They agree. **That is a
finding, not an assumption** — it is something learned about the source, and
recording it is what makes it possible to say the transcription is sound.

### "We chose Y" — an assumption

The paper does not specify everything needed to rebuild the method. Where a
value or a procedure had to be chosen, it gets a numbered entry, `A-01` through
`A-22`, containing four things:

- **The gap** — quoting what the paper does and does not say.
- **What was chosen.**
- **Why** — the reasoning, not just the conclusion.
- **How much it matters** — and if that is an argument rather than a
  measurement, the entry says so plainly.

The identifier then appears in the code at the point of use, in a comment that
**says what the choice was in words**, not just its number. A bare
`# ASSUMPTION A-17` tells a reader nothing. The comment in the code reads
"shrink the learning rate by the same factor every epoch; the paper gives the
first and last rate but never how it gets between them" — everything needed,
without leaving the file.

### "We measured Z" — a result

A number this code produced, with the command that produced it, averaged over
at least five seeds, and traceable to the commit that made it.

The rule is: **never report a number you have not computed.** No estimates, no
"approximately", no figures carried over from an earlier draft without being
re-derived.

---

## Assumptions are ours; findings are the paper's

There is a fourth label, `F-01` through `F-12`, and keeping it separate from
`A-xx` is the point.

An **assumption** is a choice this project made. A **finding** is something
established *about the paper* from the data. Confusing the two would let a
project's own decisions be presented as facts about its source, which is the
central thing a re-implementation study must not do.

A few of the findings, to show the range:

- **F-01** — `P15` is in the paper's graph but takes only two distinct values
  in the training data.
- **F-07** — the paper's own stated construction rules for its graph require an
  edge that its printed matrix does not contain.
- **F-08** — the paper's Table 12 cannot hold signed Shapley values, given what
  it reports elsewhere.
- **F-10** — the input scaling the paper never describes accounts for most of
  the remaining gap between its figure and this reproduction's.
- **F-11** — the paper's Conclusions claim a "highest R²" that its own Table 9
  refutes.
- **F-12** — none of the twelve neighbouring papers on this benchmark gives a
  graph node a single number; the smallest node feature any of them uses is
  thirty wide.

None of those is a criticism dressed as a result. Each is a checkable statement
about a published document, recorded because a reader comparing the two needs
to know it.

---

## Corrections stay visible

Several claims in this repository were wrong when first written. They were
fixed after checking, and the record says so rather than being quietly
overwritten.

- **F-05** was written from 14 sampled windows out of 240, and had both its
  magnitude and its mechanism wrong.
- **A-10** claimed a sensor was "the only" one meeting a condition when seven
  qualified.
- **A-08** was logged as an assumption when the paper's Fig. 1 states the thing
  outright. It was withdrawn.
- **F-04** claimed the physics graph was worth 5.7 cycles. Re-measured, the
  difference was 0.044 cycles and not significant. It was reversed.

The withdrawn and reversed entries are still there, marked as such, with their
original text below the correction. A corrected record is worth more than a
clean one — partly because it tells you what kind of mistake this project is
prone to, which is information a clean record destroys.

**F-09** is the sharpest of these. A results file survived in the repository
that no committed code could produce. It happened to contradict a figure that
was correct. It could as easily have corroborated one that was wrong. That is
why every run now writes its commit — with a `-dirty` suffix when the working
tree had uncommitted changes — into both its summary file and its own directory
name, and why the **Stored runs** page shows those commits rather than only the
numbers.

---

## The machinery that keeps this true

Documentation drifts away from code. This project has suffered from it, so
three things run automatically:

**`pytest`** checks that the code does what it claims. Some tests check
*properties* rather than outputs, because a wrong answer in this domain still
looks plausible: that the filter only looks backwards, that test data never
influences the scaling, that training and validation engines never overlap,
that the parameter counts match the paper's table, that message passing — the
spreading of information along the graph's edges, one hop per layer — cannot
reach further than the graph allows, that the attributions reconstruct the
prediction exactly.

**`ruff check`** catches unused imports, dead code and style drift.

**`python scripts/verify_claims.py`** recomputes the figures the assumptions log
quotes — the graph's edge counts, `P15`'s two levels, how many test labels the
ceiling touches — prints each beside the recorded value, and exits non-zero on
a disagreement. It exists because the log had drifted from the code more than
once. When it was first written, **five of the entries it checks had already
drifted.**

---

## What this site adds

Every page here is rendered from the repository's own files at the moment you
request it. There is no build step and no generated copy, so a page cannot
disagree with the file it came from.

The figures go further: each is **computed** rather than drawn. The sensor
graph is laid out from the transcribed matrix, its edge count and diameter
measured by the same code the tests use. The network diagram is produced by
building the model and reading its layers back. The label curve, the smoothing
and the windowing are drawn from the real NASA files, and if those files are
absent the figure says so and points at the download rather than substituting
something with the right shape.

That is the same rule the rest of the project follows, applied to its
documentation: **show the thing, or say you cannot.**

---

## Where to go next

- **The assumptions and findings explorer** lists all 34 entries as cards you
  can scan, each linking into the full log.
- **The stored runs page** shows every result this repository has kept, with
  its commit and whether the tree was clean.
- **`docs/ASSUMPTIONS.md`** is the log itself — long, and the most important
  document here.
- **The notebooks** run the whole chain on real data with every number visible.
- **The folder READMEs** hold the full argument for each step of the method.
