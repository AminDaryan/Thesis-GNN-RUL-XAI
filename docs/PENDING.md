# Pending work

Things known to be wrong, unmeasured or unfinished, written down so they are not
rediscovered from scratch. Every entry says **what** is wrong, **how to check**
it, and **what finishing it looks like**, so a session with the budget for it can
pick one up without re-deriving the context.

Add to this file whenever you find something you are not fixing now. Delete an
entry when it is done — a stale entry here is worse than no entry, because the
next reader will spend budget confirming it is already fixed.

Ordered by consequence, not by effort. Identifiers are never reused, so a
gap in the numbering means an entry was finished and removed.

---

## P-17 — Nothing in the repository shows the network beats an ordinary model

The thesis has to answer "would any reasonable method have done as well?", and
no committed code, log entry or README row answers it. Every result is set
beside the paper and beside other configurations of the same network, never
beside a model with no graph in it. The audit of 2026-09-08 raised this (its
D-5) and it was never carried out.

**A scratch measurement on 2026-09-14 says the answer matters.** It set the two
stored five-seed runs — the current default,
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z`,
and the same settings with the recurrent stage,
`…_lstm32_cfg72f506_f386291-dirty_20260914T134955Z` — beside scikit-learn
models trained, for each seed, on exactly that seed's training engines (the
held-back engines unused) and scored on the same 100 test engines. The inputs
were `features[:, :, -1]`, the fourteen numbers the default network receives,
or for the last row the last value, mean and least-squares slope of each
sensor's 30-cycle window. Library defaults throughout, chosen before any result
was seen; predictions clipped to [0, 125]. Before comparing, the stored
predictions were checked to reproduce `results.csv` exactly, and all ten stored
networks were reloaded at `8b487ab` and reproduced their saved predictions to
3.8e-06 cycles.

| Model | RMSE, 5 seeds | Network's RMSE minus this | Seeds network better | Paired t over seeds | 95 % interval over 20,000 resamplings of the test engines |
|---|---|---|---|---|---|
| network, last cycle (stored) | 16.496 ± 0.415 | — | — | — | own RMSE [14.00, 18.90] |
| network + recurrent stage, best pass restored (stored) | 13.307 ± 0.532 | +3.189 | 0/5 | 0.0009 | [+1.03, +5.35] |
| constant: mean training label | 40.568 ± 0.089 | −24.072 | 5/5 | < 0.0001 | [−29.00, −19.12] |
| gradient boosting trained on shuffled labels | 40.846 ± 0.292 | −24.349 | 5/5 | < 0.0001 | [−29.33, −19.35] |
| gradient boosting on engine age alone | 31.239 ± 0.100 | −14.742 | 5/5 | < 0.0001 | [−18.92, −10.56] |
| linear regression, same 14 numbers | 20.457 ± 0.219 | −3.961 | 5/5 | < 0.0001 | [−6.30, −1.70] |
| one-hidden-layer network of 64, same 14 numbers | 17.702 ± 0.469 | −1.206 | 5/5 | 0.0023 | [−2.89, +0.64] |
| random forest, same 14 numbers | 17.870 ± 0.308 | −1.373 | 5/5 | 0.0057 | [−3.20, +0.56] |
| gradient boosting, same 14 numbers | 17.253 ± 0.329 | −0.756 | 5/5 | 0.0472 | [−2.56, +1.17] |
| gradient boosting, 42 window summaries | 12.134 ± 0.364 | +4.363 | 0/5 | 0.0001 | [+2.04, +6.69] |

The recurrent-stage network against the last row: +1.174 cycles, the booster
ahead on all five seeds (paired p = 0.0099), interval over engines
[−0.13, +2.45].

**Which summary does it**, measured the same way the next day (five seeds, no
paired tests run): boosting on the last value alone 17.253 ± 0.329, the mean
alone 21.207 ± 0.447, **the slope alone 13.693 ± 0.488**, last and mean
15.086 ± 0.595, **last and slope 11.835 ± 0.391**, mean and slope
12.827 ± 0.320; and **linear regression** on all 42, 14.520 ± 0.120, against
20.457 ± 0.219 on the last value. The slope carries the gain, and a linear
model given it beats the network, so it is the information the default network
never receives, not the boosting algorithm.

**The graph network given the same three numbers per sensor has since been
measured properly** (assumption A-28): 13.586 ± 0.675, 2.910 cycles below the
last-cycle default on the same code and seeds. So the default's gap to the
booster above is mostly the information it is not given. How it compares with
the booster itself stays open until the booster is committed and paired with it.

Read with care, and not quoted anywhere until finished:

- **The scores are not chance.** A model trained on shuffled labels lands on the
  constant: what the fitted models achieve needs the link between readings and
  remaining life. That is all the shuffled-label row shows.

  **Correction, 2026-09-15.** This bullet first said the shuffled row showed
  "no path other than the sensor-to-life relationship produces a good score",
  and the reply that proposed it said a leak from the test engines would let a
  model trained on shuffled labels score well. Both overstate it. If the answer
  leaked into the inputs, a model trained on shuffled labels would not learn to
  read it, and would still land on the constant. The row therefore rules out a
  scoring bug that rewards any prediction, not a leak. The author questioned it
  on 2026-09-15, rightly. A check that does catch a leak from the test answers
  is to change `RUL_FD001.txt` and require every test input, and every
  prediction of a trained network, to stay identical; only the scores may move.
- **The graph network's edge over an ordinary model on the same fourteen
  numbers is not established.** It wins on every seed against the three
  flexible models, but every one of those intervals over engines contains zero.
- **An untuned booster given the window's trend beats both networks** — the
  default clearly, the recurrent one on seeds but not resolvably over engines.
  That bears on assumption A-07 and on the audit's D-6.

**The numbers above came from a script outside the repository**, so under
working rule 3a they name no committed code and must be re-produced before any
of them is used.

**What finishing looks like:** a committed `scripts/baselines.py` that reads a
stored run, rebuilds each seed's split, fits the ladder above and writes its
own run directory with commit and timestamp; the answer-file test in the
correction above; the table written into
`docs/ASSUMPTIONS.md` as a finding with both kinds of uncertainty; the rows
added to `README.md` section 8; and the same ladder run for the recurrent stage
under the epoch-200 procedure (F-16) and for any configuration the thesis
quotes.

---

## P-12 — The data-driven graph on FD002 and FD004

Assumption A-27 sets the paper's Table 3 beside a graph of the 41 most strongly
correlated sensor pairs. On FD001 it is now measured with and without the
recurrent stage, at one commit each, and on held-back summaries no pass was
chosen by (both added to A-27 on 2026-10-02): the two graphs cannot be told
apart on the test engines either way. What is left:

**FD002 and FD004 are untouched by A-27.** There the correlations are measured
after per-condition scaling. The sensitivity study (entry P-26) trains the
correlation graph beside its reference on FD002, FD003 and FD004 with the
recurrent stage; its FD002 and FD004 cells are finished.

**What finishing looks like:** once the study is complete and its code state
checked (entry P-26), the correlation-graph cells set beside the reference
cells, paired by seed, with held-back error beside test error, added to A-27,
and this entry deleted.

---

## P-20 — The Shapley tables of A-20 and F-05 come from one trained network

Both entries, including the tables added to them on 2026-09-14, are computed
from a single network: seed 42, nothing held back, all 200 passes. That day's
correction showed how loose that is. Changing only the learning-rate schedule
moved `P15`'s largest contribution by 27.9 % on engine 5 and 34.0 % on engine
82, and swapped the first two sensors of A-20's ranking under the mean window.
Every rank F-05 relies on held.

A different seed may move these figures as much, or more. It has not been
measured, so every size in both entries is one draw, and the rank claims have
been tested against two trainings rather than against the spread of five
(working rule 3).

**Finishing:**

1. Train five seeds under the entries' settings. Runs since 2026-09-14 save each
   seed's network.

   ```bash
   python scripts/train.py --config <a copy of config/fd001.yaml with validation_fraction: 0.0> --no-early-stopping --seeds 5
   ```

2. Recompute both tables on each saved network; `training/stored.py` loads them.
3. Write into each entry the spread of every figure, and whether `P15` ranks
   last on every seed.

---

## P-10 — The audit of 2026-09-08: four items still need code or training

`docs/AUDIT_2026-09-08_response.md` (2026-10-02) answers every one of the audit's
eleven items, D-1 to D-11, in plain words, with pointers to where each was
written into the log or why it does not apply; finding F-21 and dated
amendments in the entries concerned carry the measurements. On 2026-10-03 the
offset in the graph layers was trained with the default's last-reading stage
(finding F-21), the parameter-count wording was corrected in the models README
and notebook 02, and the recurrent stage's curves to pass 200 were stored
(finding F-16). Still open, as the response's last section lists them:

- **D-2:** random graphs with 41 edges and rewirings of the paper's graph, which
  need a new graph option in `config.py` and `scripts/train.py`, then five-seed
  runs.
- **D-4:** `split` and the `varsplit` in a run's directory name still come from
  the `--fixed-split` option alone, so a run that holds nothing back is
  recorded as varying its split (`scripts/train.py`); `training/stored.py` reads
  `split`, so the fix must keep stored runs loadable.
- **Regularisation:** an assumption entry for the training's regularisation,
  with its comment in `training/`.
- **D-11:** `evaluation/metrics.py` returns an ASUE of 0 when no engine is
  predicted late, and calls 0 the best value; it is undefined. Return "not a
  number" with a warning, report the count of early engines beside it, and
  record the choice as an assumption.

**What finishing looks like:** each of the four done or answered in the
response, and this entry deleted.

---

## P-11 — F-16's pairing crosses a refactor

**The over-training is now observed** (finding F-16, 2026-10-03): arm C trained
again from a clean `19dceb5`, all 200 passes stored, the held-back error rising
on every seed after the kept pass. **The last-cycle arm reproduces:**
`fd001_physics_row_cap_5seed_varsplit_noearlystop_cfg00be26_19dceb5_20261002T183904Z`
gives 16.818 ± 0.157 at a clean `19dceb5`, the figure F-16 pairs against. What
is left is its partner: the recurrent arm with nothing held back and all 200
passes (`--temporal lstm --no-early-stopping`, the same configuration copy), at
the same commit, on the processor, which was still training on 2026-10-03.

**What finishing looks like:** the two paired by seed at one commit, the pairing
in F-16 replaced with a dated note, and this entry deleted.

---

## P-13 — The matrix explorer, deferred

The component notebooks are built: `notebooks/temporal/` and `notebooks/graph/`
(2026-10-02 and 2026-10-03), each reading stored five-seed runs, and notebook
02's graph comparison moved into the graph folder. On 2026-10-02 the author
decided to build the notebooks and not, for now, the documentation site's
matrix explorer: a grid of temporal stage x sensor graph x graph rescaling in
which every cell shows the stored five-seed result with its commit, and every
unmeasured cell is visibly empty and shows the command that would fill it. The
playground must not train for it (argued on 2026-09-09: a number shown in a
browser and never written to `reports/` with its commit cannot be quoted or
reproduced).

**What finishing looks like:** a decision by the author to build it, then the
grid in the playground reading `training/stored.py`.

---

## P-14 — Codes used as words outside the documents the check covers

`scripts/check_prose.py` now fails when a code such as `A-07` is used as a word
instead of as a pointer, in the READMEs, `docs/learn/` and the notebooks. It does
not look at:

- **`docs/ASSUMPTIONS.md`** itself, where entries refer to one another by code
  hundreds of times ("under A-22's rule", "see F-10"). It is the log, so a code
  there has its entry one click away, but the author's instruction covers it.
- **`docs/LITERATURE_BASIS.md`** and the dated audits and reviews. The dated
  documents are records of what was written then and should not be rewritten.
- **`scripts/train.py --help`**, whose option texts end in "(A-19)" and the like.

The playground's cautions were on this list until 2026-09-17; they now say the
thing in words and put the code after it in parentheses, as a link to the entry.

**What finishing looks like:** the log's cross-references rewritten with the
thing said in words; the check extended to `docs/ASSUMPTIONS.md` with entry
headings and table identifier columns exempt.

---

## P-26 — Complete the fixed-setting sensitivity study on FD002, FD003 and FD004

The 18-scenario, five-seed study completed on FD001 on 2026-09-29 has not yet
been run on the other three C-MAPSS sub-datasets.  The author requested the
same scenarios while keeping the graph layer widths, first fully connected
width, windows per update, passes over the training data, first seed,
recurrent hidden size, numbers handed to the graph per sensor, label ceiling,
window length, cycles between windows and smoothing weight on the newest
reading unchanged.

**The running study is mixing code states (found 2026-10-01).** The study
started on 2026-09-30 in `reports/sensitivity_fd65a3b-dirty_20260930T160040Z/`
is still running. It trains each scenario in a fresh `scripts/train.py`
process, and since the package is installed in editable mode, each process
uses the code in the working tree as it stands when that process starts, not
the code the study started with. Its first 34 runs record `fd65a3b-dirty`.
`runs/FD003/scaling_minmax_global`, finished at 22:59 on 2026-10-01, records
`692d2d4-dirty`: a commit made that evening, plus whatever several sessions had
uncommitted at that moment. Counted again at 16:57 on 2026-10-02, 46 of its 54
runs had finished, and 12 of them record a later state than the folder's name:
`692d2d4-dirty` 3, `148f260-dirty` 1, `ce7e305-dirty` 6, `bbc6618` 2. Each
run's own `code_state` keeps that run
reproducible, but the folder name no longer describes everything inside it, and
`test_the_directory_name_of_every_run_carries_its_commit` fails because of it.
Either restart the remaining scenarios from a committed tree in a folder of
their own, or compare runs only within one recorded commit.

**What finishing looks like:** run all 18 predefined scenarios with five seeds
on FD002, FD003 and FD004; retain each run's code identity and measured files;
make grouped runs visible in the website's Run History; report the mean and
seed-to-seed spread of every measure; record the resulting cross-dataset
finding in `docs/ASSUMPTIONS.md`; run the relevant tests; and delete this
entry.

---

## P-27 — Citations still to check, found while collecting the cited papers

On 2026-10-01 and 2026-10-02 every cited paper was located, copied into
`docs/references/` and checked against the places this repository cites it
(`docs/REFERENCES.md` says which version each copy is). The readings that check
turned up in `docs/LITERATURE_BASIS.md`, finding F-12, assumptions A-13, A-17
and A-19, the folder READMEs, notebook 01 and the Resources pages were
confirmed in the copies and corrected on 2026-10-02, each with a dated note;
the recurrent cells' starting values became assumption A-34. What is left:

**Works cited but not held.** Heimes (2008) and Li, Ding and Sun (2018) are
IEEE and Elsevier papers; Shi et al. (2024) is Elsevier; Shapley (1953) is a
book chapter; Young (1985), to whom Lundberg and Lee credit the uniqueness of
the Shapley value, is a journal article; Kapoor and Narayanan's 2023 article is
held only as its 2022 preprint (cell.com refuses scripted downloads); Holdijk
et al. (2021) is free on Zenodo, which refused this network. Until they are
obtained, the claims attributed to Heimes (the 120–130 ceiling, assumption
A-01), Holdijk et al. (PGExplainer's sensitivity, assumption A-29), and Shapley
and Young (the uniqueness conditions) are unchecked against their sources.

- `README.md`'s glossary entry on data leakage credits Kapoor and Narayanan
  with more than the preprint's sentence says; check it against the 2023
  article once held.

**What finishing looks like:** the missing works obtained and added to
`manifest.json` and `docs/REFERENCES.md` (or the claims resting on them marked
as unchecked), the Kapoor sentence checked, and this entry deleted. (The IMTSO-GCN leads this
entry also held were confirmed in the copy and written into finding F-12 and
`docs/LITERATURE_BASIS.md` on 2026-10-02 and 2026-10-03.)
