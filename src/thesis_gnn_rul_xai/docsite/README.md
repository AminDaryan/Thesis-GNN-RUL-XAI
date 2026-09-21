# `docsite` — the documentation, in a browser

This folder holds a small website for this repository. It opens on the
playground, where a run of the network is designed, trained, set beside the
paper and explained. Its menu holds everything written here that a reader
needs: eight taught pages that build the method from nothing, the notebooks,
one page per folder of the code, the assumptions log, the audits and reviews.
It runs on your own machine, reads the files from disk as you ask for them,
and publishes nothing.

Start it with:

```bash
python scripts/docs_ui.py
```

That prints an address, opens a browser at it, and keeps running until you
press Ctrl+C.

Nothing here computes a result. This folder is a **reader** for work done
elsewhere in the package, plus a set of diagrams drawn from that work. It is
the only folder under `src/thesis_gnn_rul_xai/` that the pipeline never calls.
Its playground page can *start* the project's own scripts, `scripts/train.py`
and `scripts/explain.py`, but what they compute is theirs, stored by them in
`reports/`, and only read back here.

---

## The files

| File | What it is |
|------|------------|
| `server.py` | The web server. Receives a request for an address such as `/api/doc?path=README.md` and sends back the answer. |
| `catalog.py` | Finds every document in the repository and arranges them into the groups the menu shows. |
| `markup.py` | Turns one Markdown file into the HTML the browser displays. |
| `figures.py` | Draws the diagrams the taught pages and the notebooks show, each from this repository's own data. |
| `results.py` | Reads the entries of `docs/ASSUMPTIONS.md`, and the last re-check of the numbers the log quotes. |
| `notebooks.py` | Turns a notebook into HTML for reading, and starts JupyterLab for running. |
| `playground.py` | Checks the playground's settings, builds the command that runs them, queues the jobs, and reads stored runs and explanations back. |
| `assets/playground.js` | The playground page: its form, its job list, its charts. |
| `assets/shell.html` | The page the browser loads first: a menu, a search box, and an empty middle. |
| `assets/app.css` | How it looks. The settings at the top decide it; the rules below only apply them. |
| `assets/app.js` | What happens when you click something. |

---

## The idea, in one sentence

**Every page is built from the repository at the moment you ask for it**, so a
page can never disagree with the file it came from.

That is the whole design constraint, and it rules out the obvious alternative.
The usual way to publish documentation is to run a tool that converts a folder
of Markdown into a folder of HTML, and then serve the HTML. The trouble is that
the HTML is a *copy*. Edit a Markdown file, forget to run the tool, and the
website now says something the repository does not. This project has already
been bitten by a stored copy disagreeing with the code it described — a results
file survived that no committed code could produce (finding F-09 in the
assumptions log) — so a second copy of the documentation was not worth the risk.

Reading a file from disk on every request costs a few milliseconds and removes
the failure mode entirely. There is no build step, nothing to regenerate, and
nothing to forget.

The figures follow the same rule, one step further: they are **computed**, not
drawn. The graph drawings are laid out from the transcribed matrix in
`graph/adjacency.py`, and the distances the reach figure counts come from the
same `adjacency_report()` the tests use. The network diagram is produced by
building a real `ESTGNN` in memory and reading its layers back. If someone
changes the layer widths, the diagram changes with them.

---

## How a page reaches your screen

Say you click on the assumptions log. Six things happen.

1. The browser's address becomes `#/doc/docs/ASSUMPTIONS.md`. Nothing is
   requested from the server yet — that part after the `#` never leaves the
   browser.
2. `app.js` notices the change and asks the server for
   `/api/doc?path=docs/ASSUMPTIONS.md`.
3. `server.py` checks the path is inside the project directory, refuses it if
   not, and reads the file.
4. `markup.py` converts the Markdown to HTML, gives every heading an anchor,
   rewrites the links, and returns the headings as a list.
5. The server puts the page's menu name, "Assumptions log", in place of the
   words of its first heading, and sends back a small parcel: the HTML, that
   name, the headings, and how long the document takes to read.
6. `app.js` puts the HTML in the middle of the page, names the browser tab with
   the same menu name, and builds the "On this page" list down the right-hand
   side from the headings.

The whole exchange is one request. There is no framework involved on either
side — the server is Python's own `http.server`, and the browser code is plain
JavaScript with no libraries.

Step 5 is why a page has one name. The menu has to use short names, and a file
often opens with a longer heading of its own, so the reader used to meet two
names for one page, side by side. The Markdown files are not edited for this:
on GitHub they keep their own headings. A page reached only through a link,
such as the top-level `README.md`, is not in the menu and keeps its own
heading.

---

## Three things `markup.py` has to get right

### The mathematics has to survive the Markdown parser

The documents in this repository write formulas between dollar signs, in the
notation called LaTeX — for example `$\mathrm{RUL}_{\max} = 125$`.

A Markdown parser does not know that is a formula. It sees the underscore in
`RUL_{\max}` and, following its ordinary rules, treats it as the start of
italic text. What comes out the other end is mangled.

So the formulas are taken out **before** the parser runs. Each one is replaced
by a meaningless placeholder word that the parser has no opinion about, the
document is converted, and then the formulas are put back into the finished
HTML. A worked example:

```python
from thesis_gnn_rul_xai.docsite.markup import render_markdown

rendered = render_markdown(r"Then $\mathrm{RUL}_{\max} = 125$ holds.")
# The formula comes back exactly as written, and no italics were invented:
assert r"\mathrm{RUL}_{\max} = 125" in rendered.html
assert "<em>" not in rendered.html
```

Code blocks are protected first, so `echo $HOME` in a shell example is left
alone rather than being read as the start of a formula.

The browser then typesets the formulas with MathJax, which it fetches from the
internet. If you are offline that fetch fails, and the page falls back to
showing the LaTeX source in a fixed-width face — which is what a plain text
editor shows too, and is still readable. Everything else on the site works
offline.

### The links have to keep working

The documents link to each other the way GitHub expects: `docs/ASSUMPTIONS.md`,
or `../README.md`, or `#8-results-so-far`. Those are paths on disk, and they
mean nothing to a browser looking at a single-page website.

So every link is rewritten as it is rendered, relative to the document it was
found in. A link to `../README.md` inside `src/thesis_gnn_rul_xai/data/README.md`
becomes `#/doc/src/thesis_gnn_rul_xai/README.md`. Links to `.ipynb` files go to
the notebook viewer, links to `.py` files to the source viewer, and links
starting `http` are left exactly as they are.

### The anchors have to match the ones already written down

The top-level README links to its own sections with anchors like
`#8-results-so-far`, written for GitHub. If this site invented its own naming
scheme, every one of those links would break.

So `slugify()` follows GitHub's rule: lower-case the heading, drop everything
that is not a letter, digit, space or hyphen, and turn spaces into hyphens.
`## 8. Results so far` becomes `8-results-so-far`, exactly as GitHub produces.
Where two headings share the same words — several folder READMEs have two
sections called `Files` — the second gets `files-1`, again matching GitHub.

One extra rewrite happens that GitHub does not do. Any `A-07` or `F-11`
appearing in ordinary prose becomes a link to that entry in the log, so you can
jump from a mention to the reasoning without hunting. Identifiers inside code
spans and inside existing links are left alone, so `` `ASSUMPTION A-07` `` in a
code sample stays plain text.

### And code has to be readable in both themes

Code is coloured by Pygments, the library that splits source code into words
and gives each kind of word a colour. Each of its colour schemes is chosen for
one background. The light scheme's colours once showed through on the dark
theme's code panel, where they fell below the contrast of 4.5 to 1 that
ordinary text needs: its blue measured 2.01 to 1, its red 2.74 and its green
3.36. So `markup.py` gives each theme its own scheme, Pygments' `default` for
the light theme and `github-dark` for the dark one, and mixes any colour that
still falls short towards the theme's ordinary text colour, five per cent at a
time, until it reaches 4.5 to 1 on that theme's panel. The panel colours it
measures against are copied from `app.css`; one test fails if the two ever
differ, and another if any colour on the site falls short.

---

## The figures, and the ones that cannot always be drawn

Each figure is one function in `figures.py`, listed by name in `FIGURES`. A
taught page asks for one with `{{figure:name}}` on a line of its own, and a
notebook with `display(HTML(render_figure("name")))`. Before a figure is
deleted, both kinds of caller have to be searched for its name.

**A notebook must hand a figure over as HTML, never as an SVG image.** The
site reads a notebook through `nbconvert`, which drops an output of the type
`image/svg+xml` and keeps one of the type `text/html`, so a figure displayed as
an SVG object shows in JupyterLab and is simply missing here, with nothing
failing. That was measured on nbconvert 7.17.1 against its `basic`, `lab` and
`classic` templates, after notebook 01's map went missing that way.
`tests/test_docsite.py` renders notebook 01 as the site does and requires the
map to be in the page.

Five of them read the **real NASA files**: `rul-curve`, `correlation`,
`smoothing`, `windowing` and `notebook-map`. If those files are not in
`data/raw/`, the figure is replaced by a note saying so and pointing at the
download. It does **not** draw an invented signal that happens to have the right
shape. That is the same rule the rest of the project follows, applied to
pictures: show the thing, or say you cannot.

`notebook-map` is the one that answers "what does the preprocessing notebook
actually do?" in a single picture. It runs the pipeline while the page loads
and prints the count the data really has at every step, so it cannot drift away
from the notebook it describes the way a hand-drawn diagram would.

`architecture` needs PyTorch, because it builds a real model to read the layer
shapes off it. Without PyTorch it says so, and nothing else on the site is
affected.

Two of the figures move. `message-passing` fills the nodes in waves, one wave
per layer, and `windowing` slides a box along an engine's history. Both are
animated with SVG's own timing tags rather than with JavaScript, so they work
in a saved copy of the page.

### The animation is the argument

`message-passing` earns its place because the thing it shows is hard to state
in a sentence and obvious in a picture: after three layers, `W32` is still
grey. It is four edges from `P15` and three layers cannot cross four edges, so
those two sensors are **never** mixed, whatever the training does. The four
pairs in that situation, which the `reach` figure names, are computed by
`pairs_beyond_reach()` in `graph/adjacency.py`, which `tests/test_adjacency.py`
checks against the four pairs worked out by hand.

That the network itself behaves that way is a separate test,
`tests/test_models.py::test_three_layers_cannot_reach_four_hops`: it adds 100
to every reading of `P15`, runs the three graph layers, and requires `W32`'s
output not to move. An earlier version of this page said that test used
`pairs_beyond_reach()`; it does not, and was corrected on 2026-09-17.

---

## Reading a notebook and running a notebook are different things

The site does both, and the difference matters.

**Reading** uses `nbconvert` to turn the stored notebook into HTML. You see the
code and the outputs **as they were saved**, produced on the real FD001 files.
Nothing runs. This always works, needs no extra software, and is what you want
if you are following the argument. The line above a notebook says so in two
words, "Saved outputs.", beside the button that runs it.

**Running** needs JupyterLab, which is a separate program. The **Open in
JupyterLab** button above a notebook starts it as a child process, waits for it
to print the address it is serving on — including the one-time token it
generates for itself — and opens that notebook in a new tab. From then on you
are in JupyterLab, not on this site, and you can execute cells. The new tab is
the confirmation; a message gives the address only when the browser blocked the
tab from opening.

If JupyterLab is not installed, the line above the notebook says how to install
it in place of the button:

```bash
pip install -e .[notebook]
```

The server looks for JupyterLab again on every request rather than once when it
starts, so installing it is not a reason to restart the site.

Only one JupyterLab is ever started. Asking again returns the address of the
one already running, and it is shut down when the documentation server stops.

### A trap: the address of one notebook is not the address plus a path

JupyterLab announces itself as `http://localhost:8899/lab?token=...`. To open a
particular notebook the address has to become

```text
http://localhost:8899/lab/tree/notebooks/01_preprocessing_fd001.ipynb?token=...
```

Two things are easy to get wrong there, and the first version of this code got
both.

**The token must survive.** It is a one-time password JupyterLab generates for
itself at startup. Drop it and JupyterLab shows a login page asking for a
password nobody has been told.

**The path is rebuilt, not appended to.** The announced address already ends in
`/lab`, so gluing `/lab/tree/...` onto the end produces `/lab/lab/tree/...`,
which is a page that does not exist. `labAddress()` in `app.js` therefore
strips a trailing `/lab` before adding the rest. That bug survived until
JupyterLab was first installed on a machine here, because until then the code
that builds the address had never run with a real address to build from.

### A trap: nbconvert's stylesheet is not used

`nbconvert` returns a stylesheet along with the HTML, and that stylesheet
styles bare tags — `pre`, `table`, `div`. Dropped into the page as it is, it
would restyle the entire site, not just the notebook, and its code colours are
chosen for a light background only.

`render_notebook()` therefore keeps the HTML and discards the stylesheet.
`app.css` styles everything inside the box the notebook is drawn in,
`.notebook-frame`, and the code colours of the section above apply to a
notebook's cells in both themes. If a notebook ever looks unstyled, that box is
where to look.

---

## The playground: design, results, explain, jobs

The page at `#/explore/playground` is where the parts of the method can be
swapped and the effect seen. It has three tabs, in the order the work happens,
and a fourth for the jobs:

1. **Design a run.** Choose the sub-dataset, the preprocessing, the temporal
   stage, the sensor graph and the widths of the graph layers, and the
   training settings. The graph layers themselves stay the three of the
   paper's Table 4; only their sizes change.
2. **Results.** Every stored run, its four measures beside the paper's, every
   seed's curve and predictions, and any two runs paired seed by seed.
3. **Explain.** Explain one stored run's predictions, with the paper's method
   or a cheaper one, and read back how much each sensor moved them.

The design tab has twenty-nine settings, and shown all at once they ran to five
screens. So the form is laid out as numbered steps. Step 1, the sub-dataset,
stands on its own at the top. Under it the **pipeline** draws the network as a
flowchart, in the order the data passes through it: the preprocessing, the
temporal stage, the graph layers with the sensor graph they run on, and the
remaining life that comes out. Below that, the other settings sit in five
groups that fold, numbered 2 to 6 — Preprocessing, Temporal stage, Graph
layers, Training and Repeats. Each box of the pipeline shows its group's number
and what that step is now, "3 LSTM" or "3 GRU", so the group's name is said
once, on the group, and the number ties the two together. Choosing a box opens
its group and brings it into view; choosing the sensor graph inside the graph
layers' box goes straight to that setting.

The sensor graph is built while the data is prepared, but its setting sits in
the Graph layers group, beside the layers that use it, because that is where
the pipeline draws it.

A folded group's heading shows only its number, its name and how many of its
settings differ from their defaults. It does not repeat what the group holds:
the pipeline shows the choice that matters for each stage, and the count says
whether anything else was moved. An open group shows the controls, and a
setting that differs carries a Reset. Which groups are open is remembered in the
reader's own browser, as the menu's folds are.

One step is the current one: step 1 when the page opens, then whichever group
was last opened, reached from the pipeline, or worked in. Its number is filled
in, on its group and on its box in the pipeline, and every other number is an
outline, which also keeps the form's numbers from reading as the numbers of
the Design, Results and Explain tabs above them.

The form ends, below step 6, with the card that runs it: the network the
settings build, what stops or qualifies the run, and the button that starts it.
It stands last so the page reads from top to bottom in the order the work is
done, and so each caution sits directly above the button it is about.

### Why the page never computes a result of its own

The easy design would train in the server and draw the number. It would also
break the two rules this project leans on hardest. A number that appears in a
browser and is never written down with the commit that produced it cannot be
quoted or reproduced (rule 3a of `CLAUDE.md`, and finding F-09 records what
happened here the one time that slipped). And a page that answers quickly
invites one-seed numbers, which rule 3 forbids as results.

So the page does not train. It starts **`scripts/train.py`** — the same script,
with the same arguments, that anyone would type, and the job's card in the Jobs
tab shows that command. What the run produces lands in `reports/` with its
commit in its directory name, exactly as if it had been typed. An explanation
is `scripts/explain.py` in the same way, and lands inside the run it explains.
The command is handed to the operating system as a list of separate arguments,
never as one line of text for a shell to split, so no setting typed into the
page can become a second command.

Fewer than five seeds is allowed, because trying things out is the point. It is
marked where it matters: above the Train button before the run, as a
warning-coloured seed count in the list of runs, and among the cautions on the
run's own page.

### What the design tab shows before anything runs

Some things need no training, and the page shows them as the settings change:

- **The network these settings build**, built for real by `models/`, with how
  many numbers it learns and, folded under it, each layer's output shape and
  count. For the paper's LSTM stage at hidden size 32 that is 124 482; for the
  w/o LSTM network, which keeps only each window's last cycle, it is 61 729.
  `ESTGNN.layer_summary()` lists the temporal stage and then Table 4's "Input"
  row, but both describe the same array of shape `(B, 14, 1)`, one number per
  sensor for each of the `B` windows in a batch — Table 4's Input row, which
  learns nothing of its own, is what the stage hands to the first graph layer —
  so the page shows them as one Input row and counts the stage's learned
  numbers there. Settings the network refuses — thirty raw cycles handed
  straight to a graph layer, for one — are refused with the network's own
  message.
- **What stops or qualifies the run**, directly above the Train button: that
  PyTorch or the NASA files are missing, the network's refusal, how many test
  engines are shorter than the window and will be left out (counted from the
  real file), and two cautions about settings that conflict — fewer than five
  seeds, and the learning rate that waits for the held-back error to stall with
  no engines held back. Those two are here because the settings involved sit in
  groups that may be folded, and "Set these settings up again" on a stored run
  can bring either in without the reader opening a group.

The cautions come from one function, `caveats()`, and each carries a `where`:
`"run"` for a caution that qualifies a stored run's numbers, which the run's
own page lists in full, and `"design"` for the two above. Most cautions are
`"run"` only, because on the design tab the i beside the setting already says
the same thing, while a stored run's page shows no settings and the caution is
the only copy there.

### Which row of the paper a run is set beside

Removing one part of a model to see what it was worth is called an
**ablation**, and the pipeline makes the two that can be built here. The cross
on the temporal stage's box keeps only each window's last cycle, which is the
w/o LSTM row's stage, and the box leaves the flowchart. The **physics prior**
is the paper's own sensor graph, the matrix of its Table 3, which the authors
built by hand from how a turbofan engine works; the form calls it "The paper's
physics graph". The cross on it joins every sensor to every other instead, the
w/o Physics prior row's graph. Each leaves a dashed button in its place that puts
the part back as it was, however it was taken out: a GRU, the smaller recurrent
network this project offers beside the LSTM, comes back as a GRU. The graph
layers' box has a cross drawn faint that does nothing and says why when
pointed at: a network without graph layers is not built here, so "w/o GCN" has
no settings.

A stored run is set beside one row of the paper, decided by what its settings
remove, one part at a time, the way the paper's Table 11 does:

| Temporal stage | Sensor graph | Published row |
|---|---|---|
| LSTM | the paper's | ESTGNN (Full), Tables 7 to 10 |
| w/o LSTM | the paper's | ESTGNN w/o LSTM, Table 11 |
| LSTM | every sensor joined | ESTGNN w/o Physics prior, Table 11 |
| w/o LSTM | every sensor joined | none: two parts removed at once |
| anything | the 41 most correlated pairs | none: the paper builds no such graph |
| anything else | the paper's or every sensor joined | none: the paper has no row for the stage |
| not recorded, in an older stored run | the paper's or every sensor joined | none: no row can be chosen |

The list of runs gives that row's RMSE in its Paper column, and a run's own
page names the row and sets its figures beside the run's, or gives the reason
from the table's last column. The design tab names no row: the pipeline already
shows which parts are in, and the paper's number belongs beside a result, not
beside settings that are still being chosen.

### Where each default comes from

Every setting has an i beside its name. Pointing at the i, reaching it with
the keyboard, or choosing it opens a small card with a tag saying which of four
things set the default, because that is the distinction the thesis depends on,
and a few words on why. The form itself shows only names, so it reads as a list
of choices rather than a list of citations. Escape closes the card.

| Tag | Meaning | Example |
|---|---|---|
| Paper | the paper prints the value | 200 passes over the training data, Table 6 |
| Our choice | the paper is silent, so a numbered assumption chose | the label ceiling of 125 (assumption A-01) |
| Project rule | neither: a rule of this repository | five seeds |
| Not in the paper | an option offered for comparison | the GRU stage |

Each fact is said once: when every option of a setting shares one source, the
tag is behind the setting's i; when the options differ, as for the sensor graph
and the temporal stage, each option has an i of its own. An option's i is not a
stop for the Tab key. Moving onto the option with the arrow keys opens its card,
and a screen reader reads the card as the option's description. The cards of
the Explain tab's options work the same way.

A pass over the whole training set is what PyTorch calls an **epoch**; the page
says "pass" throughout.

The training procedure is the first choice in the Training group: **the
paper's procedure** (nothing held back, every pass, the final weights) or
**held back, best pass** (a fifth of the training engines held back, and the
weights of the pass that scored best on them restored). It is the only control
for those two things; while the second is chosen, the share of engines held
back and how many passes may go by without improvement sit under it. It does
not set the number of passes, which is the next setting in the group. The two
things are one choice because on FD001 it changed what the LSTM stage was worth
from 3.19 cycles to 0.77 (finding F-16).

### Jobs: one at a time, and what stopping means

A `JobManager`, the queue behind the page, runs one command at a time in the
order they were asked for. Two trainings at once would compete for the same
processor and both finish later. Each job has a folder in
`reports/.playground/` holding the settings file it wrote, everything the
command printed, a progress file of one JSON line per event — a pass
finished, a seed finished, a window explained — and, since 2026-09-16, a
`job.json` with its name, its command and how it ended. The page reads the
progress file every two seconds while a job is live, to draw the progress bar
and the held-back error of the current seed. The folder starts with a dot so
the run listing never mistakes it for a run.

A job's name is built by the server alone, in `job_label()` for a training run
and `explanation_label()` for an explanation — "FD001 · LSTM 32 · physics
graph · best pass · 5 seeds", "Exact Shapley values · engine 5 · seed 42" — and
the same way for a job this server started and for a folder read back from
disk, so one job never appears under two names. A folder read back that asked
for no particular engine takes the engine from the explanation it stored.

The job running now is shown in one line above whichever tab is open, with a
Stop button, and nothing at all when nothing is live, so the tabs never move.
On the Jobs tab that line is hidden while the list below shows the same job.
The Jobs tab's badge counts the jobs waiting behind it. The tab itself holds
every job this server ran and every one an earlier server left on disk, read
back from its folder. A job whose server stopped before it finished is listed
as interrupted, since a run writes nothing until its last seed is done. One
that **another server is still running** is listed as running instead, and
refused for removal: every job writes down the number of the process running
it, and a server that finds that process still alive knows the command is
writing into that folder at this moment. Without that check the job would read
as ended, and removing all the ended ones would delete the folder from under a
running command. The tab filters by kind and by state, and a job that has ended
can be removed, singly or all at once; removing one deletes its folder under
`reports/.playground/` and nothing else, so the run or explanation it wrote
stays where it is.

**Stopping a job writes nothing to `reports/`.** `scripts/train.py` keeps every
seed's weights, curves and predictions in memory and writes them only when the
last seed has finished, so a run stopped part-way leaves no directory that could
be read as a finished one.

**A short message is shown only where nothing else says the same.** "Run
queued." and "Explanation queued." appear only when the line above the tabs is
scrolled out of sight:
that line does not stay on screen, while the column holding the Train button
does, so a reader who has scrolled down to it would otherwise see nothing
happen. A job that ends is
announced by name unless the Jobs tab is open, where its card changes in front
of the reader. One trap here: a job is announced when a poll finds it ended
after an earlier poll saw it live, and a short explanation can finish between
the request that starts it and the next poll. So a job this page started is
also announced when the very first poll already finds it ended.

### A run, and two runs

Click a run and it opens on its own, in place of the list, with a button back
to every run; the list is long, and a run's curves and predictions drawn
underneath it were a screen's scroll away. A commit ending in `-dirty` had
uncommitted changes, so no committed state of the code reproduces that run;
the list colours such a commit, and its column heading says so when pointed at.

The run's predictions are drawn as the paper's Fig. 8 draws them, test engines
in order of unit number, or ordered by true remaining life instead, which shows
how the error changes along the life.

Tick two runs and the page lists **every setting that differs** between them,
then pairs their seeds. For each seed it takes the second run's number minus
the first's; the paired *t* statistic is the average of those differences
divided by how much they vary, and *p* is how often a *t* that large would turn
up if the true average difference were zero.

A worked case, from two runs stored on 2026-09-09 that differ in one setting
only: the w/o LSTM network, and the paper's LSTM at hidden size 32, both with a
fifth of the engines held back, five seeds each. The page computes a mean RMSE
difference of −3.189 cycles, *t* = −8.978, *p* = 0.00085 — the same figures the
assumptions log records for this pair (finding F-16), computed here by
`compare_runs` rather than copied.

If more than one setting differs, the page says the difference cannot be put
down to any one of them, and if only the commit differs, that any difference
comes from the code or the machine. And under every comparison it says what a
paired *p* does not establish: every seed is scored on the same test engines, so
it answers only whether the difference could be an accident of how the network
started, never how the model would do on engines it has not seen (defect D-8 of
the audit of 2026-09-08).

### Explanations

The Explain tab explains the run chosen in Results with "Explain this run", or,
when none is chosen, the newest run that kept its weights; another run is
chosen back in Results. It starts `scripts/explain.py` with one of two methods,
exact Shapley values or occlusion, which `explain/README.md` explains. Measured
on this machine's processor with an untrained network, one window of exact
Shapley values takes 0.07 s for the w/o LSTM network and 0.50 s with the LSTM
stage, so an engine's whole life is minutes, not hours.

A stored explanation holds one row per window: the true and predicted
remaining life, one value per sensor in cycles, and, since 2026-09-16, what
each sensor was reading at the window's last cycle, because the paper's Fig. 9
colours every value by whether the sensor read high or low and nothing else in
the file could say which. That last number is the reading **after** the noise
filter and the rescaling, not the raw one: it is what the model was given, and
it is the only version of it the stored windows hold. An explanation stored
before the readings were kept is coloured by the window's place in the sequence
instead, and its legend says so.

Every caption of an explanation opens by naming the paper's figure the chart
follows, as in "As the paper's Fig. 9, left.", or with "Not in the paper.", so
a reader can tell the paper's pictures from this project's additions (working
rule 5).

**Beside the paper's Fig. 10.** For an engine explained with Shapley values,
the page redraws the paper's Fig. 10 from the numbers of its Table 12, above
the stored engine at the same graph numbers, where graph `g` is the engine's
`g`-th window, and counts **how often a sensor joins the red four**, the four
largest values, from one graph to the next. Within the three stages there are
12 such steps. At graph 190 of Table 12 the four largest are Nc 2.37, Ps30
2.29, BPR 1.84 and T50 1.65, with phi fifth at 1.59. At graph 191 BPR has
fallen to 1.18 and the four largest are Ps30 4.23, Nc 2.60, phi 1.51 and T50
1.42. So phi joins the red four there, once, and in the other eleven steps
nothing joins. The page counts the stored engine the same way, and, when every
window was explained, over all of that engine's steps as well. A large count
means the sensors this network leans on move from one window to the next, which
is not something the paper's table shows.

---

## What the server will and will not serve

It listens on `127.0.0.1`, which reaches your own machine and nothing else. It
is a reader for your own files, not a way to publish them, and it is worth
being clear about the difference because the project directory contains a
paper's worth of unpublished work.

Five rules keep it that way:

- **Every path is resolved and checked to be inside the project directory.**
  `../../../etc/passwd` is refused, and so is anything else that climbs out.
  There is a test for this.
- **The source viewer shows a fixed list of file types** — `.py`, `.yaml`,
  `.toml`, `.json`, `.txt` and two more. Anything else is refused, so the
  server cannot be turned into a way of reading the whole disk.
- **Files above 400 kB are refused** by the source viewer. Every source file in
  the project is far below that; the raw NASA files are far above it, and a
  3.5 MB table of numbers is not something to read in a browser.
- **Only the addresses that start work accept a POST**, the kind of request a
  browser sends to ask for something to be done rather than read: the one that
  starts JupyterLab, and the playground's five, which preview a network, start
  a training run, start an explanation, stop a job and remove one. Everything
  else is read-only.
- **A request that starts work must come from this site's own page and from
  this computer.** A browser attaches the address of the page that sent a POST
  in a header called `Origin`, so any other web page open in the same browser
  could otherwise ask this server to start a training run. The server compares
  that header with its own address, and it also refuses a request from any
  machine but this one, even when `--host` has opened the site to a network.
  Both refusals are tested.

The `--host` option exists and will accept any address, but `docs_ui.py` prints
a warning when it is given anything other than the loopback address, because
that makes every file in the project directory readable by anything that can
reach your machine.

---

## Things that will catch you out

**The site opens on the playground.** The playground is what the site is for,
so the address `#/` sends the reader straight to `#/explore/playground`, and
the playground is the one link above the menu's sections. There is no separate
home page and no trail of links above a page's title: every page sits one level
below the menu, so a trail would only repeat the page's name.

**Removed pages still have addresses.** The site once had separate pages for
the sensor graph, the network's layers, every figure, the stored runs and the
assumptions as cards. Each repeated something another page already said, and
each was removed. Their old addresses, which bookmarks may hold,
open the page that now says the same thing: the sensor graph and every figure
open the taught page on graphs, the network's layers the taught page on the
model, the stored runs the playground, and the assumptions the log.

**The menu is written by hand, in four sections named for what a reader goes
there to do.** They are *Learn the method*, *Notebooks*, *Code* and *Choices
and checks*. `catalog.py` gives every page a short name of its own, which the
menu, the page's first heading and the browser tab all use. The code pages
follow the order the pipeline runs them — data, graph, models, training,
evaluation, explain, utils — not the alphabet, and the parts of a folder are
indented under it. The dated audits and reviews are the one group found on
disk, newest first, so a new one needs no edit. Two files are left out of the
menu on purpose, and still open from any link to them: the top-level
`README.md`, the repository's front page on GitHub, whose sections each have a
menu page of their own, and the paper-against-code audit of 2026-08-31, the
record of how each row of `docs/PAPER_TO_CODE.md` was first checked.

**Every menu section starts folded.** There are 43 pages in the four sections.
Whatever the reader opens or folds is remembered in their own browser — except
that going to a page inside a folded section always unfolds it, since a menu
that hides the page you are reading is worse than one that ignores your
preference.

**The server loads its Python once, and says so when that is out of date.**
Documents and the page's own scripts are read from disk on every request, so
an edit to them shows on the next reload. The Python behind the addresses is
loaded when the server starts and never again. Edit `playground.py` and a
running server keeps answering with the old code while serving the new script,
which then asks for things the old code does not send. That happened here on
2026-09-16: a server started two days earlier served a page that expected the
paper's Table 12 and did not get it. The server now compares the modification
time of every Python file in the package with the moment it started, reports
the answer as `stale` in `/api/state`, and the page shows a banner under the
top bar telling the reader to restart it. A server too old to report the
answer at all is treated as out of date, since the page asking is newer than
it — that is the case the banner was first needed for. Every reply has always
carried `Cache-Control: no-store`, which covers the other half of the same
promise: the browser must ask again each time rather than keep its own copy of
a script or a figure, or the page would be stale on the reader's side instead
of the server's.

**Two servers could hold one port, and Windows would not complain.** This is
what made the trap above hard to see, and it was found on 2026-09-16 with
three documentation servers listening on the same port at once, the oldest
started two days earlier. Python's own web server turns on a network setting
called SO_REUSEADDR so that a server stopped and started again can reclaim its
port immediately instead of waiting for the operating system to let go of it.
On Windows that setting means something else: take the port **even if another
program is listening on it**. So starting a second server printed the usual
address and looked fine, while each request went to whichever server Windows
picked — sometimes the one started days ago, answering with the code it loaded
then. `DocsServer` now leaves that setting off on Windows and turns on its
exclusive counterpart, so a second server fails to start and `docs_ui.py`
says to stop the first rather than to pick another port, which is what its
advice used to say. `tests/test_docsite.py` holds a server on a port and
requires a second one to be refused.

**The menu can be closed, and the two screen sizes close it differently.** The
button at the left of the top bar hides or shows it. On a wide screen the page
takes the width the menu gave up, and the choice is remembered in the browser,
so a reader who works in the playground with the menu shut finds it shut next
time. On a narrow screen the menu is a drawer over the page: it starts closed,
opens over a dimmed page, and closes when the dimmed page is tapped, when
Escape is pressed, or when a page is chosen from it. The state is one
attribute on the document, `data-nav`, and a few lines in `shell.html` set it
before the first paint so a closed menu does not flash open while the page
loads.

**"About n minutes" is a word count, not a promise.** It divides the prose word
count by 220 words a minute. It is there to warn you that the assumptions log
is an hour's reading, not to be accurate about you.

**The assumptions log is parsed, not just displayed.** `results.py` reads the
headings of `docs/ASSUMPTIONS.md`, picking out each entry's identifier, title
and anchor, so that a link such as `#/entry/A-07` lands on that entry's heading
in the log. That parsing depends on the log's existing heading style —
`## A-01 — The label ceiling is 125 cycles`. A test checks that the entries come
out numbered consecutively with no gaps, so if a future entry is written
differently the test fails rather than the link quietly going nowhere. The log's
page also shows, under its title, the result of the last run of
`scripts/verify_claims.py`, which recomputes the numbers the log quotes: how
many it re-checked, how many disagree, and when.

**A run with one seed keeps its blank standard deviation blank.** The stored
summary files write `null` there, because one run has no spread. It would be
easy to turn that into a `0.0` on the way through, and it would be a lie: it
would read as "measured, and it was zero" instead of "not measured". The page
prints no spread at all.

**A job is marked running a moment before its process exists.** The queue
takes the next job, marks it running, and only then asks the operating system to
start the command. Until 2026-09-14 a Stop that arrived in that moment found no
process to end, did nothing, and the job ran to completion while the page
waited for it to stop. It was first written up as a Windows problem — that
ending a process there does not reach its children — and that diagnosis was
wrong: checked directly, the process did not yet exist when Stop arrived, and
once it did exist it ended in a tenth of a second. Now starting the process and
checking for a Stop happen under the same lock, so a Stop is either seen before
the process starts, which then never starts, or finds the process and ends it.
`tests/test_playground.py` covers the first case.

**The site adds nothing to what the repository says.** Everything on it is
either a file from the repository, a number computed by the package's own code,
or a caption written for a figure. It is a way of reading the work, not a
second account of it — so if a page here and a file in the repository ever
disagree, the file is right and this folder has a bug.

---

## Judgement calls made in this folder

**None of them are assumptions about the paper.** The `A-xx` numbers exist for
choices forced by gaps in the source being re-implemented, and a documentation
reader is not that. Nothing in this folder affects a result, and nothing in it
appears in the assumptions log.

The choices made here are ordinary software ones, recorded for the next person
rather than for an examiner:

- **Python's own `http.server` rather than a web framework.** It is in the
  standard library, so the site adds no dependency that must be installed
  before the documentation can be read. A framework would be more comfortable
  to write and would make reading the documentation conditional on a successful
  install.
- **Markdown converted on the server, not in the browser.** The two libraries
  used, `mistune` and `pygments`, arrive with `nbconvert`, which the project
  already needs for the notebooks. Converting in the browser would have meant
  shipping a large JavaScript library into the repository, which is a thing
  nobody reviews.
- **Diagrams drawn as SVG by hand, not by a plotting library.** The project
  already depends on `matplotlib`, and it could have produced these. SVG
  written directly is a few hundred lines longer, and in exchange the diagrams
  scale to any window, take the page's own colours in both the light and the
  dark themes, carry text you can select and search, and can animate. A
  generated image does none of those.

---

## What each name means

### `serve(port, root, host, open_browser)`

Runs the website until interrupted. Prints the address, and a short summary of
what this machine can do — whether the NASA files are present, whether PyTorch
is installed, whether JupyterLab is available — so that a missing piece is
visible before you start clicking.

### `build_catalog(root)`

Returns a `Catalog`: every page the site can show, the playground as its one
primary page, and the rest in the four sections of the menu, each with the
short name the menu shows and, for a document, its length measured. A document
that is not on disk is simply absent from the result, so the site works on a
partial checkout.

### `read_document(path, root)`

The text of one file, or a refusal. Raises `ValueError` if the path points
outside the project directory and `FileNotFoundError` if there is nothing
there.

### `search(query, catalog, root)`

Every place a phrase appears across the documents in the menu, as a list of
dictionaries holding the document's path, its menu name and the surrounding
text. The notebooks and the playground are not searched. It is a plain
substring search over the raw Markdown — no index, no ranking. On this
repository's 300 kB of prose it takes long enough to notice and not long enough
to mind. The page names each document once, over the lines that matched in it.

### `render_markdown(source, source_path, link_identifiers)`

One Markdown document as a `RenderedDocument` — the HTML, the title, the list
of headings, and whether it contained any mathematics. `source_path` is what
the relative links are resolved against, so pass it whenever you have it.

### `slugify(text, seen)`

The anchor GitHub would give a heading. Pass a dictionary as `seen` to number
repeats; pass nothing to get the plain form.

### `render_figure(name)`

One diagram as a piece of SVG, or — if the NASA files or PyTorch are missing —
a block of HTML explaining what is absent and how to get it. It never raises,
because a missing figure should not take the page down with it.

### `load_entries(root)` and `latest_verification(root)`

The first returns every assumption and finding in the log as an `Entry`, with
its identifier, title, status and the anchor to link to. The second returns the
most recent run of `scripts/verify_claims.py` stored in `reports/`, as a
`Verification` holding how many claims it re-checked and how many failed, or
`None` if it never ran.

### `render_notebook(path, root)` and `start_jupyter(port, root)`

The first returns a notebook's cells as HTML, with the outputs it was saved
with. The second starts JupyterLab and returns a `JupyterProcess`, the record of
a running server — where it is serving, how it was started, and a `stop()` that
shuts it down.

### `validate_settings(payload)` and `paper_row_for(subset, temporal, graph)`

The first takes the settings the page sent, a dictionary, and returns every
setting checked against its range and filled from its default; an unknown
setting or a value out of range raises a `SettingsError`, whose message is what
the page shows. The second returns the row of the paper's results those
settings may be set beside, as a dictionary of its four measures, or `None`
together with the reason there is none.

### `JobManager(root)`

The queue of the playground's commands for one project directory. Its
`submit_training(settings)` and `submit_explanation(request)` add a job and
return it, named by `job_label` or `explanation_label`; `cancel` removes a
queued job or ends a running one, `forget` removes ended jobs and deletes their
folders, and `snapshot` returns every job with its progress — this server's and
those read back from earlier servers' folders by `job_history` — for the page to
draw.

### `list_runs(root)`, `run_detail(root, name)` and `compare_runs(root, first, second)`

Every finished run in `reports/`, newest first, each with the published row it
may be set beside and the cautions that apply; one run in full, with every
seed's per-pass record and test predictions and every stored explanation; and
two runs paired seed by seed, with the settings that differ and a paired *t*
and *p* for each measure.
