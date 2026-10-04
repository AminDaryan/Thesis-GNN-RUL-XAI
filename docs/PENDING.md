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

## P-10 — The audit of 2026-09-08: the control graphs are built but not trained

`docs/audits/AUDIT_2026-09-08_response.md` answers all eleven of the audit's items. On
2026-10-03 the last code items were done: the run record's split field
(item D-4), ASUE's undefined case (item D-11, assumption A-39), the
regularisation entry (assumption A-40), and the control graphs of item D-2 —
random graphs with 41 edges and degree-preserving rewirings of Table 3, built
as `--graph random` and `--graph rewired` (assumption A-41). What is left is
their training:

- from a clean checkout of the commit that adds them, all six on one device,
  five seeds each, as assumption A-41 lists: the default and the two controls,
  with the last-cycle stage and with the recurrent stage;
- then A-41's "How much it matters", its summary row, the dated note in A-19,
  `README.md`, `graph/README.md` and the response's D-2 paragraphs filled from
  them, and `tests/test_graph_controls.py`'s stored-run test run once.

**What finishing looks like:** those runs stored and written up, and this entry
deleted.

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

- **The log itself**, `docs/assumptions/` and `docs/findings/`, where entries
  refer to one another by code hundreds of times ("under A-22's rule", "see
  F-10"). It is the log, so a code there has its entry one click away, but the
  author's instruction covers it.
- **`docs/LITERATURE_BASIS.md`** and the dated audits and reviews. The dated
  documents are records of what was written then and should not be rewritten.
- **`scripts/train.py --help`**, whose option texts end in "(A-19)" and the like.

The playground's cautions were on this list until 2026-09-17; they now say the
thing in words and put the code after it in parentheses, as a link to the entry.

**What finishing looks like:** the log's cross-references rewritten with the
thing said in words; the check extended to `docs/assumptions/` and `docs/findings/` with entry
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
finding in `docs/findings/`; run the relevant tests; and delete this
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
