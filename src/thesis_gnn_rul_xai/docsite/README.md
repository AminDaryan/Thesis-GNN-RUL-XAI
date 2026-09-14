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
- **Only one address accepts a POST**, the one that starts JupyterLab.
  Everything else is read-only.

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
