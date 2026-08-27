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
to be significant (p = 0.20), which is a real answer and was worth having.

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

---

## Before saying a task is done

```bash
ruff check .          # must be clean
pytest                # 163 tests; must all pass
```

`pytest` needs the real FD001 files in `data/raw/`. Without them the
data-dependent tests skip with a download link rather than failing. The model,
training and explanation tests additionally need PyTorch; without it they skip
and 107 still run.

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
