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
Spearman (1904), cited since 2026-11-14 on the C-MAPSS page for the name
"attenuation" (finding F-23), is free from the Internet Archive, which refused
this network; its two passages were checked against the Internet Archive's
text of the scan, made by character recognition, not against the scan itself.

- `README.md`'s glossary entry on data leakage credits Kapoor and Narayanan
  with more than the preprint's sentence says; check it against the 2023
  article once held.

**What finishing looks like:** the missing works obtained and added to
`manifest.json` and `docs/REFERENCES.md` (or the claims resting on them marked
as unchecked), the Kapoor sentence checked, and this entry deleted. (The IMTSO-GCN leads this
entry also held were confirmed in the copy and written into finding F-12 and
`docs/LITERATURE_BASIS.md` on 2026-10-02 and 2026-10-03.)

---

## P-30 — Sentences on the documentation pages that no held source supports

On 2026-10-18 the author asked for every statement on the documentation pages
to carry a source, cited the way papers cite theirs. All 17 pages under Learn
and Resources now do, checked against the held copies; the record is
`docs/audits/AUDIT_2026-10-19_docs_sources.md`. That check left 58 sentences
that teach something the page needs but that no held copy states. The audit
lists every one, in four groups, with what is missing:

- **Plain-word explanations of standard terms** (39): a spool, the Rankine
  scale, a paired *t*-test, a softmax and the like. Textbook definitions that
  no held copy gives.
  The section "What the FD001 training engines look like", added to the
  C-MAPSS page on 2026-11-14, adds four more: a median, a percentile, a
  standard deviation, and that the variance of a sum or a difference of two
  unrelated draws is the sum of their variances. Its own reasoning is marked
  as the page's.
- **Why information must not leak from the test engines** (6): the reading
  list's source, Kapoor and Narayanan (2023), is held only as its 2022
  preprint (see the entry above).
- **Sources needed but not held** (2): the GNN-LRP authors' code, which sets
  the explanation rule's gamma per layer, and any paper on peephole
  connections.
- **The project's own reasoning, not marked as such** (11). Two of them
  describe the Mean and Full window options for the step before the graph
  layers, which this project added and the worked examples page describes,
  but which no assumption entry records, unlike the GRU stage (assumption
  A-36).

**How to check:** read the audit's four tables; `tests/test_citations.py`
cannot see any of this, because it compares citations only with the lists.

**What finishing looks like:** the author decides, group by group, whether a
sentence stays as the page's own explanation (marked as such where it is
reasoning), gets a source added to `docs/REFERENCES.md` and its copy to
`docs/references/`, or is reworded; the Mean and Full window options get an
assumption entry each, or one between them; the pages are changed and
`pytest tests/test_citations.py tests/test_links.py` and
`python scripts/check_prose.py` pass; and this entry is deleted.

## P-31 — The default's five-seed figure moves with the PyTorch version, and the Reading results page does not say which it quotes

`docs/learn/06_reading_results.md`, under "One seed is not a result", gives
16.496 ± 0.415 as what "the current default settings" produce, from
`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z`,
trained under PyTorch 2.13.0 on the processor. The settings are still the
default's: `ba82e3` and `818b40` are the same settings under the digest rule
before and after 2026-10-03, which stopped counting the prose notes (finding
F-09). But on 2026-10-19 the same settings at `a05da94`, under PyTorch 2.14.1
on the processor, gave 16.689 ± 0.246 over the same seeds
(`reports/fd001_physics_row_cap_5seed_varsplit_cfg818b40_a05da94_20261019T091343Z`,
trained to measure smoothing for assumption A-04), close to the 16.690 ± 0.229
that assumption A-07 records under PyTorch 2.5.1 on a graphics card. A-07
says no stored run separates the device from the PyTorch version; this one is
on the processor, so the device alone does not explain the 16.496, but it also
differs from the 2.13.0 runs in commit, so the code is not ruled out either.

**How to check:** train the default at `f4f40ad`, whose clean checkout
reproduced 16.496 under 2.13.0, again under PyTorch 2.14.1 on the processor,
and compare seed by seed.

**What finishing looks like:** A-07's paragraph on the two figures says which
of code, version or device moves them, and the Reading results page names the
PyTorch version (and device) its figure was trained under.

## P-32 — The network without graph layers is built, not measured, and its clean control is not built

Since 2026-11-13 `scripts/train.py --spatial none` removes the three graph
layers and puts nothing in their place (assumption A-42), so the paper's
"ESTGNN w/o GCN" row (16.42 on FD001, Table 11) can be set beside a run. Only
smoke runs of one seed and three passes exist, which are not results.

**What is left.**

- **The five-seed runs**, from a clean checkout of the commit that adds the
  option, each beside the same configuration with its graph layers at the same
  commit, paired by seed: `--spatial none --temporal lstm` against
  `--temporal lstm`, and `--spatial none` against the default. Test RMSE is the
  measure; both comparisons read together at 0.05 / 2 (assumption A-37). The
  recurrent pair needs saying which weights were kept (finding F-16). Then A-42's
  "How much it matters", its summary row and `README.md` section 12 filled from
  them.
- **The control that keeps every learned number.** Removing the graph layers
  removes 59,680 of the default's 61,729 learned numbers, so it cannot separate
  the connections from the size. A graph in which each sensor is joined only to
  itself, run through the same three graph layers, keeps all 61,729 and removes
  only the passing between sensors. The audit of 2026-09-08 measured one over
  three seeds outside any stored run (item D-2); here it would be a new
  `--graph` value with its own assumption entry, then the same paired runs.

**What finishing looks like:** the runs stored and written up in A-42, the
control built, trained and written up (or the author deciding against it), and
this entry deleted.

---

## P-34 — Figures keep the text's line, not the 1080 pixels the site's README describes

The website's README says, under "How every page fits the screen", that on a
wide screen "tables, code and figures may grow beside it to 1080" and that "At
1920 a figure is 1080 pixels wide beside 848 of text"; the comment above
`.prose` in `assets/app.css` says the same. The rule under that comment,
`.prose > .figure { width: 100%; max-width: min(var(--text-w), 100%); }`, holds
every figure to the text's line instead, and `.prose .figure-svg { min-width:
820px; }` never lets a drawing shrink below 820 pixels. A figure therefore fits
only where the serif font's 74 characters come to 846 pixels or more (820 for
the drawing and 26 for the box's padding and border), as the README's 848 for
Sitka Text on Windows does. Where they come to less, every figure on the
documentation pages scrolls sideways at every screen width. Measured on
2026-11-15 in Chromium on Linux, where the page's font list falls through to
Caladea: inside its border a figure's box is 687 pixels wide at 1024, 1280,
1366, 1440, 1536 and 1920 pixels, while its drawing and padding take 844, so
157 pixels of each figure are out of sight until it is scrolled, although the
article's column beside it is 864 to 1080 pixels wide at those widths. Which
font a Mac takes from the list, whose first entry is Charter, and how wide its
74 characters are, has not been measured.

**How to check:** on any page with a figure, `docs/resources/cmapss.md` for
one, compare each `figure.figure` element's `clientWidth` with its
`scrollWidth`.

**What finishing looks like:** the author decides which of the two is meant.
Either figures go back to `min(var(--measure-wide), 100%)`, which in the
measurement above fits every figure at every one of those widths, or the
README and the comment are corrected to say figures keep the text's line. Then
the README's 16 widths are measured again, with pictures before and after, and
this entry is deleted.
