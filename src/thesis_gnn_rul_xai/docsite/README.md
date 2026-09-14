# `docsite` — the documentation, in a browser

This folder holds a small website that shows everything written in this
repository: the top-level README, one page per folder of the code plus one
per file under `data/`, the assumptions log, the audits, both notebooks, and a
set of figures. It runs on your own
machine, reads the files from disk as you ask for them, and publishes nothing.

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
| `figures.py` | Draws the thirteen diagrams, each from this repository's own data. |
| `results.py` | Reads the stored runs in `reports/` and the entries in `docs/ASSUMPTIONS.md`. |
| `notebooks.py` | Turns a notebook into HTML for reading, and starts JupyterLab for running. |
| `playground.py` | Checks the playground's settings, builds the command that runs them, queues the jobs, and reads stored runs and explanations back. |
| `assets/playground.js` | The playground page: its form, its job list, its charts. |
| `assets/shell.html` | The page the browser loads first: a menu, a search box, and an empty middle. |
| `assets/app.css` | How it looks. |
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
drawn. The sensor graph is laid out from the transcribed matrix in
`graph/adjacency.py`, and its edge count and diameter come from the same
`adjacency_report()` the tests use. The network diagram is produced by building
a real `ESTGNN` in memory and reading its layers back. If someone changes the
layer widths, the diagram changes with them.

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
5. The server sends back a small parcel: the HTML, the title, the headings, and
   how long the document takes to read.
6. `app.js` puts the HTML in the middle of the page and builds the "On this
   page" list down the right-hand side from the headings.

The whole exchange is one request. There is no framework involved on either
side — the server is Python's own `http.server`, and the browser code is plain
JavaScript with no libraries.

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

---

## The figures, and the two that cannot always be drawn

There are thirteen: eight need nothing but the repository, four read the real
NASA files, and one needs PyTorch.

| Figure | What it shows |
|---|---|
| `engine` | A turbofan, with the station each sensor sits at |
| `pipeline` | The eight preprocessing steps, and the three that decide validity |
| `sensor-graph` | The paper's Table 3 drawn as the network it describes |
| `adjacency` | The same graph as a table, before and after rescaling |
| `degrees` | How many neighbours each sensor has |
| `message-passing` | Information spreading from one sensor, one edge per layer |
| `reach` | How far apart every pair of sensors is |
| `shapley` | Every order three players could join a group in |
| `architecture` | Every layer of the network, with its real parameter count |
| `rul-curve`, `smoothing`, `windowing`, `notebook-map` | The label, the filter, the window, and the whole of notebook 01 |

`notebook-map` is the one that answers "what does the preprocessing notebook
actually do?" in a single picture. It runs the pipeline while the page loads
and prints the count the data really has at every step, so it cannot drift away
from the notebook it describes the way a hand-drawn diagram would.

The last four read the **real NASA files**. If those files are not in
`data/raw/`, the figure is replaced by a note saying so and pointing at the
download. It does **not** draw an invented signal that happens to have the right
shape. That is the same rule the rest of the project follows, applied to
pictures: show the thing, or say you cannot.

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
pairs in that situation are computed by `pairs_beyond_reach()`, the same
function `tests/test_models.py` uses to check the network really behaves that
way.

---

## Reading a notebook and running a notebook are different things

The site does both, and the difference matters.

**Reading** uses `nbconvert` to turn the stored notebook into HTML. You see the
code and the outputs **as they were saved**, produced on the real FD001 files.
Nothing runs. This always works, needs no extra software, and is what you want
if you are following the argument.

**Running** needs JupyterLab, which is a separate program. The **JupyterLab**
button in the top bar starts it as a child process, waits for it to print the
address it is serving on — including the one-time token it generates for
itself — and opens that address in a new tab. From then on you are in
JupyterLab, not on this site, and you can execute cells.

If JupyterLab is not installed the button says so and tells you how to install
it, rather than failing silently:

```bash
pip install -e .[notebook]
```

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

### A trap: nbconvert's stylesheet is not scoped

`nbconvert` returns a stylesheet along with the HTML, and that stylesheet
styles bare tags — `pre`, `table`, `div`. Dropped into the page as-is it
restyles the entire site, not just the notebook.

`app.js` therefore rewrites every rule in it to apply only inside the box the
notebook is drawn in, before adding it to the page. If a notebook ever looks
right but the rest of the site suddenly does not, that rewriting is where to
look.

---

## The playground: choose, train, compare, explain

The page at `#/explore/playground` is where the parts of the method can be
swapped and the effect seen. It has three tabs, in the order the work happens:

1. **Design a run.** Choose the sub-dataset, the preprocessing, the temporal
   stage, the widths of the graph layers and the training settings. The graph
   layers themselves stay the three of the paper's Table 4; only their sizes
   change.
2. **Results and comparisons.** Every stored run, its four measures beside the
   paper's, every seed's curve and predictions, and any two runs paired seed by
   seed.
3. **Explanations.** Pick a stored run and a seed, choose the method and what to
   explain, and read back how much each sensor moved the predictions.

### Why the page never computes a result of its own

The easy design would train in the server and draw the number. It would also
break the two rules this project leans on hardest. A number that appears in a
browser and is never written down with the commit that produced it cannot be
quoted or reproduced (rule 3a of `CLAUDE.md`, and finding F-09 records what
happened here the one time that slipped). And a page that answers quickly
invites one-seed numbers, which rule 3 forbids as results.

So the page does not train. It starts **`scripts/train.py`** — the same script,
with the same arguments, that anyone would type — and shows that command beside
the button. What the run produces lands in `reports/` with its commit in its
directory name, exactly as if it had been typed. An explanation is
`scripts/explain.py` in the same way, and lands inside the run it explains. The
command is handed to the operating system as a list of separate arguments,
never as one line of text for a shell to split, so no setting typed into the
page can become a second command.

Fewer than five seeds is allowed, because trying things out is the point, and
is marked **exploratory** everywhere it appears: in the cautions before a run,
in the results table, and on the run's own page.

### What is shown before anything runs

Some things need no training, and the page shows them as the settings change:

- **The network these settings build**, built for real by `models/`, with each
  layer's output shape and how many numbers it learns. For the paper's LSTM
  stage at hidden size 32 that is 124,482; with the stage removed it is 61,729.
  Settings the network refuses — thirty raw cycles handed straight to a graph
  layer, for one — are refused here with the network's own message.
- **The row of the paper a result may be set beside.** Which row is decided by
  what the settings remove, one part at a time, the way the paper's Table 11
  does:

  | Temporal stage | Sensor graph | Published row |
  |---|---|---|
  | LSTM | the paper's | ESTGNN (Full), Tables 7 to 10 |
  | removed | the paper's | ESTGNN w/o LSTM, Table 11 |
  | LSTM | every sensor joined | ESTGNN w/o Physics prior, Table 11 |
  | anything else | anything | none, and the page says why |

  Removing one part of a model to see what it was worth is called an
  **ablation**, and the variant buttons at the top of the form set exactly the
  settings of each row. "w/o GCN" is shown and disabled: a model without graph
  layers is not built.
- **The cautions that apply**, each with the entry of `docs/ASSUMPTIONS.md` it
  comes from as a pointer: that capped test labels can only flatter the numbers,
  that held-back engines mean the weights of the best pass over the training
  data are restored — a step the paper never describes — and that on FD002 and
  FD004 engines shorter than the window are left out, so the score, a sum over
  engines, stops being comparable.
- **What the files say about the sub-dataset**: Table 5's counts, and how many
  test engines are shorter than the chosen window, counted from the real file.

### Where each default comes from

Every setting carries a label saying which of three things set its default,
because that is the distinction the thesis depends on:

| Label | Meaning | Example |
|---|---|---|
| Paper | the paper prints the value | 200 passes over the training data, Table 6 |
| Our choice | the paper is silent, so a numbered assumption chose | the label ceiling of 125 (assumption A-01) |
| Project rule | neither: a rule of this repository | five seeds |
| Not in the paper | an option offered for comparison | the GRU stage |

A pass over the whole training set is what PyTorch calls an **epoch**; the page
says "pass" throughout.

The training procedure gets a switch of its own, **the paper's** (nothing held
back, all 200 passes, the last pass's weights) against **this project's
default** (a fifth of the training engines held back, the best pass restored).
It is a separate switch because on FD001 it changed what the LSTM stage was
worth from 3.19 cycles to 0.77 (finding F-16).

### Jobs: one at a time, and what stopping means

A `JobManager`, the queue behind the page, runs one command at a time in the
order they were asked for. Two trainings at once would compete for the same
processor and both finish later. Each job has a folder in
`reports/.playground/` holding the settings file it wrote, everything the
command printed, and a progress file of one JSON line per event — a pass
finished, a seed finished, a window explained. The page reads that file every
two seconds to draw the progress bar and the held-back error of the current
seed. The folder starts with a dot so the run listing never mistakes it for a
run.

**Stopping a job writes nothing to `reports/`.** `scripts/train.py` keeps every
seed's weights, curves and predictions in memory and writes them only when the
last seed has finished, so a run stopped part-way leaves no directory that could
be read as a finished one.

The "train it and its w/o LSTM twin" button queues the same settings twice, the
second time with the temporal stage removed. Both runs use the same seeds, so
the Results tab can pair them seed by seed.

### Comparing two runs

Tick two runs and the page lists **every setting that differs** between them,
then pairs their seeds. For each seed it takes the second run's number minus
the first's; the paired *t* statistic is the average of those differences
divided by how much they vary, and *p* is how often a *t* that large would turn
up if the true average difference were zero.

A worked case, from two runs stored on 2026-09-09 that differ in one setting
only: the temporal stage removed, and the paper's LSTM at hidden size 32, both
with a fifth of the engines held back, five seeds each. The page computes a
mean RMSE difference of −3.189 cycles, *t* = −8.978, *p* = 0.00085 — the same
figures the assumptions log records for this pair (finding F-16), computed here
by `compare_runs` rather than copied.

If more than one setting differs, the page says the difference cannot be put
down to any one of them. And under every comparison it says what a paired *p*
does not establish: every seed is scored on the same test engines, so it answers
only whether the difference could be an accident of how the network started,
never how the model would do on engines it has not seen (defect D-8 of the
audit of 2026-09-08).

### Explanations

Two methods, and the page labels which is the paper's:

- **Exact Shapley values** (Equations 14 and 15). Every one of the 16,384
  subsets of the 14 sensors is evaluated for every window, with an absent sensor
  replaced by its average training window (assumption A-20). Measured on this
  machine's processor with an untrained network, one window takes 0.07 s with
  the temporal stage removed and 0.50 s with the LSTM stage, so an engine's
  whole life is minutes, not hours.
- **Occlusion**, not in the paper: one sensor at a time replaced by the same
  baseline, 15 evaluations per window. `explain/README.md` explains both.

Either can explain **one engine at every cycle** — the dynamic explanation of
the paper's Section 4.4 — or **the last window of every test engine**. For one
engine the page draws the prediction along its life, every sensor at every
window as a grid of colours, and one chosen window on the sensor graph coloured
the way the paper's Fig. 9 colours it: the four largest in red, the rest in blue
from deep to light. It also checks Equation 14 on every stored explanation — a
window's Shapley values must add up to its prediction minus the prediction at
the baseline — and prints the largest gap.

---

## What the server will and will not serve

It listens on `127.0.0.1`, which reaches your own machine and nothing else. It
is a reader for your own files, not a way to publish them, and it is worth
being clear about the difference because the project directory contains a
paper's worth of unpublished work.

Four rules keep it that way:

- **Every path is resolved and checked to be inside the project directory.**
  `../../../etc/passwd` is refused, and so is anything else that climbs out.
  There is a test for this.
- **The source viewer shows a fixed list of file types** — `.py`, `.yaml`,
  `.toml`, `.json`, `.txt` and two more. Anything else is refused, so the
  server cannot be turned into a way of reading the whole disk.
- **Files above 400 kB are refused** by the source viewer. Every source file in
  the project is far below that; the raw NASA files are far above it, and a
  3.5 MB table of numbers is not something to read in a browser.
- **Only the addresses that start work accept a POST**: the one that starts
  JupyterLab, and the playground's four (below). Everything else is read-only.
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

**The menu order is not the directory order.** `catalog.py` lists the folder
READMEs in the order the pipeline runs them — data, graph, models, training,
evaluation, explain, utils — not alphabetically. A reader following the method
wants `data` first and `evaluation` after `training`, and alphabetical order
gives neither.

**The menu sections fold, and two of them start open.** There are thirty
documents. Listed flat they run together and a reader cannot see where one
group ends, so each section is a block that folds shut, showing its title and
how many documents it holds. **Learn** and **Explore** start open because that
is where someone new begins; the long reference lists start shut. Whatever the
reader folds is remembered in their own browser, so the menu opens the way they
left it — except that navigating to a page inside a folded section always
unfolds it, since a menu that hides the page you are reading is worse than one
that ignores your preference.

**"About n minutes" is a word count, not a promise.** It divides the prose word
count by 220 words a minute. It is there to warn you that the assumptions log
is an hour's reading, not to be accurate about you.

**The assumptions log is parsed, not just displayed.** `results.py` reads the
headings of `docs/ASSUMPTIONS.md` to build the cards on the explorer page,
picking out the identifier, the title, the `**Where:**` line and whether the
entry has been withdrawn, reversed or superseded. That parsing depends on the
log's existing heading style — `## A-01 — The label ceiling is 125 cycles`. A
test checks that the entries come out numbered consecutively with no gaps, so
if a future entry is written differently the test fails rather than the card
quietly going missing.

**A run with one seed keeps its blank standard deviation blank.** The stored
summary files write `null` there, because one run has no spread. It would be
easy to turn that into a `0.0` on the way through, and it would be a lie: it
would read as "measured, and it was zero" instead of "not measured". The table
shows a dash.

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

Returns a `Catalog`: every document the site can show, grouped into the six
sections of the menu, with each one's title read from its first heading and its
length measured. A document that is not on disk is simply absent from the
result, so the site works on a partial checkout.

### `read_document(path, root)`

The text of one file, or a refusal. Raises `ValueError` if the path points
outside the project directory and `FileNotFoundError` if there is nothing
there.

### `search(query, catalog, root)`

Every place a phrase appears across the documents, as a list of dictionaries
holding the document, the line number and the surrounding text. It is a plain
substring search over the raw Markdown — no index, no ranking. On this
repository's 300 kB of prose it takes long enough to notice and not long enough
to mind.

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

### `load_runs(root)` and `load_entries(root)`

Every run in `reports/` as a `Run`, newest first, each carrying the commit that
produced it and whether that commit was marked `-dirty`. And every assumption
and finding in the log as an `Entry`, with its identifier, title, status and
the anchor to link to.

### `render_notebook(path, root)` and `start_jupyter(port, root)`

The first returns a notebook as HTML and the stylesheet it needs. The second
starts JupyterLab and returns a `JupyterProcess`, the record of a running
server — where it is serving, how it was started, and a `stop()` that shuts it
down.

### `validate_settings(payload)` and `paper_row_for(subset, temporal, graph)`

The first takes the settings the page sent, a dictionary, and returns every
setting checked against its range and filled from its default; an unknown
setting or a value out of range raises a `SettingsError`, whose message is what
the page shows. The second returns the row of the paper's results those
settings may be set beside, as a dictionary of its four measures, or `None`
together with the reason there is none.

### `JobManager(root)`

The queue of the playground's commands for one project directory. Its
`submit_training` and `submit_explanation` add a job and return it, `cancel`
removes a queued job or ends a running one, and `snapshot` returns every job
with its progress, for the page to draw.

### `list_runs(root)`, `run_detail(root, name)` and `compare_runs(root, first, second)`

Every finished run in `reports/`, newest first, each with the published row it
may be set beside and the cautions that apply; one run in full, with every
seed's per-pass record and test predictions and every stored explanation; and
two runs paired seed by seed, with the settings that differ and a paired *t*
and *p* for each measure.
