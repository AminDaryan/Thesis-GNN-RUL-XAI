# Working rules for this repository

This is a master's thesis. Its subject is a **re-implementation of a published
paper**, and its value depends entirely on the reader being able to tell which
claims come from the paper, which come from measurement, and which are
judgement calls. Everything below exists to protect that distinction.

Read `README.md` section 9 for the layout and `docs/ASSUMPTIONS.md` for every
judgement call made so far.

---

## The paper

Wen, J.-R., et al. (2026). *Explainable spatiotemporal graph neutral network
with dynamic SHAP enhancement for aeroengine RUL prediction.* Aerospace Science
and Technology, 168, 110875.

The PDF is at
`D:\Main\University\Thesis\Papers\Predictive Maintanace Papers\GNN\GNN Predictive Maintenance\RUL\MAPSS\Explainable Spatio-Temporal Graph Neural Network with Dynamic SHAP Enhancement for Aeroengine RUL Prediction.pdf`

No source code was published with it, and its data availability statement reads
"Data will be made available on request".

---

## Rules

### 1. Never state a fact about the paper from memory

Open the PDF and check. This applies to every table value, every equation, every
figure and every sentence you intend to quote.

Figures need care: a value read from a page thumbnail is a guess. Render the
region at high resolution (`pypdfium2`, scale 8 or more) before reading a number
off an axis, and say how you obtained it. The value of `RUL_max` in this project
was settled by pixel-calibrating a plateau against its own axis ticks, and the
paper's schematic figure disagrees with its experimental one.

If the paper prints something twice — the adjacency matrix appears in Table 3
and again in Fig. 3 — check both. They have agreed so far; that is a finding,
not an assumption.

### 2. Never report a number you have not computed

No estimated results, no "approximately", no numbers carried over from an
earlier message without re-deriving them. If a claim needs a number, run the
code that produces it and show the output.

Numbers already in `docs/ASSUMPTIONS.md` and `README.md` have been measured.
Numbers anywhere else have not, until you check.

### 3. One seed is not a result

Anything quoted as a result of this project is a mean over at least five seeds,
with its standard deviation:

```bash
python scripts/train.py --seeds 5
```

Before saying that one configuration beats another, test whether the difference
survives the seed-to-seed spread. A-21 was settled that way and turned out not
to be significant, which is a real answer and was worth having.

Then remember it may not stay settled. All three of this project's
configuration comparisons — A-21, A-02 and F-04 — were re-measured after the
A-07 correction of 2026-08-28, and two of them **reversed direction**. A
comparison is a fact about one architecture, not about the two options. Name
the configuration whenever you quote one, and re-run it whenever the
architecture moves.

The seed varies the train/validation split as well as the initialisation and
the batch order, so the spread is what a genuine repetition would show.
`--fixed-split` holds the split constant, which understates it; if you use that
flag, say so wherever the number appears.

### 3a. A stored result must name the code that produced it

Every run writes its commit — with a `-dirty` suffix when the tree has
uncommitted changes — into `summary.json` and into its own directory name,
alongside a UTC timestamp. Never hand-edit a results file, never quote a number
from a run whose commit you cannot identify, and never let two runs share a
directory.

This rule exists because a results file once survived here that contradicted
the documentation and could not be reproduced from any committed state. It
happened to contradict a figure that was correct. It could as easily have
corroborated one that was wrong. See F-09.

### 4. Every deviation from the paper gets a numbered entry

If the paper does not specify something and you have to choose, add an entry to
`docs/ASSUMPTIONS.md` with:

- **The gap** — quote what the paper does and does not say.
- **What we chose.**
- **Why** — the reasoning, not just the conclusion.
- **How much it matters** — and if that is an argument rather than a
  measurement, say so.

Then put the identifier in the code at the point of use, as a comment beginning
`ASSUMPTION A-xx`. The header of that file promises this; keep it true.

That comment must say **what the choice was, in words**, not just cite its
number. `# ASSUMPTION A-17` tells a reader nothing; `# ASSUMPTION A-17: shrink
the learning rate by the same factor every epoch. The paper gives the first and
last rate but never how it gets between them.` tells them everything they need
without leaving the file.

Observations *about the paper* established from data are findings (`F-xx`), not
assumptions. Keep the two apart: assumptions are our choices, findings are
facts about the source.

### 5. Distinguish the three kinds of statement, always

- **"The paper says X"** — verifiable in the PDF; cite the table or equation.
- **"We chose Y"** — an assumption; it needs an `A-xx` entry.
- **"We measured Z"** — a result; it needs the command that produced it.

Never let one wear the clothes of another. "This reproduces the paper" is a
claim about all three at once and is almost always wrong here: the LSTM stage is
not implemented, so results compare against the paper's own **"ESTGNN w/o
LSTM"** row (14.93 RMSE on FD001), never its headline figures.

### 6. Correct errors plainly, including your own

Several claims in this repository were wrong when first written and were fixed
after checking: F-05 was written from 14 sampled windows out of 240 and had both
its magnitude and its mechanism wrong. A-10 said "the only sensor" when seven
qualified. A-08 was logged as an assumption when Fig. 1 states it outright.

When you find one, fix it, say what was wrong and how it was found, and leave a
note if the error is instructive. Do not quietly overwrite. A corrected record is
worth more than a clean one.

### 7. Generated data is not allowed

The project deliberately contains no synthetic data. A pipeline that appears to
work on invented signals says nothing about whether it works on engines. If the
NASA files are missing, the code says so and points at the download; it never
substitutes something with the right shape.

### 8. Do not commit or push without being asked

Make the changes, run the checks, report what changed, then stop. Committing is
the author's decision, not yours.

### 9. Explanation lives in the folder READMEs, not in the code

Every folder under `src/thesis_gnn_rul_xai/` has a `README.md`, and that is
where the reasoning, the traps, the vocabulary and the meaning of every
argument belong. A `.py` file carries **one line** of description per module,
class and function, plus the `ASSUMPTION` comments required by rule 4.

Do not put the explanation back into the code. No `Parameters` / `Returns` /
`Raises` / `Notes` / `Examples` sections, and no Sphinx cross-reference roles
(`:class:`, `:func:`, `:meth:`, `:attr:`, `:data:`) — this project has no Sphinx
build, so they render as noise to everyone who reads them.

Write for someone who does not already know the project. Spell a term out the
first time it appears rather than assuming it. Where a README has to name a
class, say what it is: "a `TrainingHistory`, the record of a finished run", not
just the name.

#### The check, which is not the same as the rule

The paragraph above has been in this file from the start and was broken anyway,
twice. Stating the rule again does not work. Before calling any document,
comment or explanation finished, **run this check**:

> Re-read every sentence as one specific person: they know Python, they have
> never used PyTorch, and they have never read the paper. Any word that person
> would have to look up gets replaced, or explained in the same sentence.

The words that fail are the ones that feel ordinary from inside the code
because they were borrowed from PyTorch or from graph theory: *buffer*,
*broadcast*, *state dictionary*, *logits*, *self-loop*, *ablation*,
*hyperparameter*, *epoch*, *z-score*, *Glorot*. Invented words are worse still
— "an unrescaled table is rejected" survived here for weeks and means nothing
to anyone. `python scripts/check_prose.py` catches this specific list; it
cannot judge whether writing is clear, so it is a floor and not a substitute
for the check above.

Second half of the check: **a claim about how something works needs a worked
example with real numbers.** "A column of the weight matrix is a list of
multipliers" explains nothing on its own. Load the real data, print the actual
values, do the arithmetic term by term, and show the code producing the same
answer. Placeholders like `x1 ... x32` are not an example.

Worked examples are tests, in `tests/test_examples.py` and
`tests/test_examples_torch.py`, not doctests. When you add a function, add its
example there.

---

## Before saying a task is done

```bash
ruff check .          # must be clean
pytest                # 203 tests; must all pass
python scripts/check_prose.py   # borrowed words must be explained where used
```

`check_prose.py` is the floor under rule 9, not the ceiling. It knows a fixed
list of words — *buffer*, *broadcast*, *self-loop*, *ablation* and the rest —
and fails if one is used for the first time in a document with no explanation
beside it. It cannot tell whether a sentence is clear, so a green run means
"none of the known traps", not "this reads well". The check in rule 9 is still
yours to run.

If the change touches `docs/ASSUMPTIONS.md`, a default in `config/`, or
anything the log quotes a number from, also run:

```bash
python scripts/verify_claims.py --fast
```

That recomputes every figure in the log that needs no trained model and fails
if one disagrees with what is written. `--fast` takes seconds; without it, two
more checks train a network and it takes minutes. This exists because the log
has drifted from the code more than once — see the header of that script.

`pytest` needs the real FD001 files in `data/raw/`. Without them the
data-dependent tests skip with a download link rather than failing, leaving
137 passing and 66 skipped. The model, training and explanation tests
additionally need PyTorch; without it they are not collected and 150 still
run.

If a change touches a notebook, re-execute it before committing, so its stored
outputs match its code:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/02_model_fd001.ipynb
```

---

## Environment

- Python 3.10+; the package is importable only after `pip install -e .`
  (the `src/` layout is deliberate — see README section 9).
- PyTorch is optional and only `models/`, `training/` and `explain/` need it.
- All four C-MAPSS sub-datasets are in `data/raw/`, verified against the
  published row and engine counts.
- Long training runs belong in the background; a 5-seed run takes about
  8 minutes on CPU.

## Things that are settled — do not re-litigate

- The 14 sensors and their order come from Table 3 and are load-bearing: the
  adjacency matrix is indexed by position. Import-time assertions guard this.
- The LSTM stage is omitted on purpose (A-11). Implementing it would be
  invention; the paper never states its size, depth or output projection.
- Test labels are capped at 125 (A-02) because the sigmoid output makes higher
  predictions impossible. This can only move reported numbers favourably, and
  that must be stated whenever they are compared with published ones.
- Validation error and test error are not comparable (A-06). Never present one
  as the other, and never read anything into the gap.
