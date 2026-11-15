# `docsite` — the documentation, in a browser

This folder holds a small website for this repository. It opens on the
playground, where a run of the network is designed, trained, set beside the
paper and explained. Its menu holds everything written here that a reader
needs: taught pages ordered as Data, Preprocessing, Temporal algorithm, GNN and
XAI (with Dynamic SHAP and GNN-LRP beneath it), the notebooks,
one page per folder of the code, the assumptions and the findings, the audits
and reviews,
and, under Resources, the dataset, the LSTM, the GCN, SHAP and GNN-LRP each
explained from its original paper, ending with the reading list.
It runs on your own machine, reads the files from disk as you ask for them,
and publishes nothing.

On desktop, the header and any server-status notice occupy the top of one
screen-sized frame. The navigation, article and contents list share the height
left below them and scroll independently. Design a run keeps its existing
two scrollable panels instead of adding another scroll around them. Long
articles remain fully readable; the browser page itself does not grow taller
than the screen. Below the desktop breakpoint, ordinary page scrolling remains.
Changing documents resets the article's scroll, and section links scroll that
article rather than moving the surrounding frame.
Screen-reader table captions are positioned inside their table wrappers.
Previously those visually hidden, absolutely positioned captions could still
extend the page beyond a panel's visible bottom and create a stray scrollbar.

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
| `worked_example.py` | Follows one real FD001 window through a named saved network and exposes its additions for inspection. |
| `results.py` | Reads the entries in `docs/assumptions/` and `docs/findings/`, one file each, and the last re-check of the numbers they quote. |
| `notebooks.py` | Turns a notebook into HTML for reading, and starts JupyterLab for running. |
| `playground/` | The playground's Python, in five files: its settings and their checks, what the Design tab shows, the commands, the stored runs, and the job queue. Its own README says which file holds what. |
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
file survived that no committed code could produce (finding F-09, in
`docs/findings/`) — so a second copy of the documentation was not worth the risk.

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

Say you click on Assumptions in the menu. Six things happen.

1. The browser's address becomes `#/doc/docs/assumptions/README.md`. Nothing is
   requested from the server yet — that part after the `#` never leaves the
   browser.
2. `app.js` notices the change and asks the server for
   `/api/doc?path=docs/assumptions/README.md`.
3. `server.py` checks the path is inside the project directory, refuses it if
   not, and reads the file.
4. `markup.py` converts the Markdown to HTML, gives every heading an anchor,
   rewrites the links, and returns the headings as a list.
5. The server puts the page's menu name, "Assumptions", in place of the
   words of its first heading, and sends back a small parcel: the HTML, that
   name, the headings, and how long the document takes to read.
6. `app.js` puts the HTML in the middle of the page, names the browser tab with
   the same menu name, and builds the "On this page" list from the headings:
   down the right-hand side where there is room beside the text, folded at the
   top of the page where there is not.

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
offline. MathJax may arrive after the page is drawn, on a first visit; once it
is ready it typesets whatever the page holds then, so no formula is left as
LaTeX because the page won the race. A notebook's text cells keep their
formulas between dollar signs, since nbconvert, not `markup.py`, turns them
into HTML; `app.js` finds them there, outside code, and puts each in the same
element a document's formula arrives in, so the one typesetting step draws
both.

### The links have to keep working

The documents link to each other the way GitHub expects: `docs/assumptions/A-07.md`,
or `../README.md`, or `#8-results-so-far`. Those are paths on disk, and they
mean nothing to a browser looking at a single-page website.

So every link is rewritten as it is rendered, relative to the document it was
found in. A link to `../README.md` inside `src/thesis_gnn_rul_xai/data/README.md`
becomes `#/doc/src/thesis_gnn_rul_xai/README.md`. Links to `.ipynb` files go to
the notebook viewer, links to `.py` files to the source viewer, and links
starting `http` are left exactly as they are.

A link to a `.pdf` — a cited paper in `docs/references/` — becomes
`/api/paper?path=docs/references/...` and opens in a new tab, in the browser's
own PDF viewer, beside the page that cites it. A fragment such as `#page=7` is
kept, and PDF viewers open the file at that page, so a sentence can link to the
exact page it cites:
`[Saxena et al.](../references/saxena_2008_damage_propagation_modeling.pdf#page=7)`
opens the dataset paper at its scoring function.

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

Ten of them read the **real NASA files**: `rul-curve`, `worked-example`,
`correlation`, `smoothing`, `windowing`, `notebook-map`, and the four the
C-MAPSS resource page draws, `lifetimes`, `sensor-trends`,
`sensor-correlations` and `noise-limit`. If those files are
not in `data/raw/`, the figure is replaced by a note saying so and pointing at
the download. `worked-example` also needs the saved training run it follows,
and its note names that run. A figure does **not** draw an invented signal that
happens to have the right shape. That is the same rule the rest of the project
follows, applied to pictures: show the thing, or say you cannot.

`notebook-map` is the one that answers "what does the preprocessing notebook
actually do?" in a single picture. It runs the pipeline while the page loads
and prints the count the data really has at every step, so it cannot drift away
from the notebook it describes the way a hand-drawn diagram would.

The C-MAPSS resource page's four figures are drawn from small tables that the
page's own Python examples print as well, so a chart and the numbers beside it
cannot disagree; `tests/test_examples.py` compares the two. Each table comes
from one function in `figures.py`, all reading the 100 engines of
`train_FD001.txt` and nothing else:

- `training_engines()` reads the file; `engine_lives()` gives each engine's
  last recorded cycle, the one at which it failed; `sensor_readings()` keeps
  the paper's 14 sensors, named by their symbols.
- `flight_to_flight_changes()` gives each reading minus the same engine's
  reading one flight earlier, and `drift_directions()` counts the engines on
  which each sensor's last 20 readings average above, below or level with its
  first 20.
- `sensor_noise_table()` gives, for each sensor, the spread of its readings,
  the spread of its scatter from one flight to the next, and the share of its
  variance (the square of the spread) left to the slow drift. The scatter's
  variance is taken as half the variance of the flight-to-flight change, which
  holds when the scatter of one flight is unrelated to the next; the page
  checks that it is.
- `sensor_correlations()` gives the two tables of correlations, of the
  readings and of their changes, and `noise_limit_table()` sets each pair's
  correlation beside its noise limit, the square root of the two sensors'
  shares multiplied: the largest size their scatter allows.

Why those quantities, and what they show about the paper's correlation
heatmap, is taught on the page itself (finding F-23).

Three figures need PyTorch, the library the network is written in.
`architecture` and `architecture-overview` build the default network to read
its layer sizes and its counts of learned numbers off it, and `worked-example`
loads a saved network and runs one real window through it. Without PyTorch
each of the three is replaced by a note saying so, and the other figures are
drawn as usual. The playground's training needs PyTorch too.

**Correction, 2026-10-02.** Until this date the paragraph on the NASA files
said that five figures read them, and the paragraph above said that only
`architecture` needed PyTorch, with nothing else on the site affected; the note
shown in place of a figure without PyTorch said the same.
`worked-example` reads the files and needs PyTorch, and `architecture-overview`
needs PyTorch. This was found when a review of the figure fixes compared these
paragraphs with what each figure in `FIGURES` says it needs.

`architecture-overview` uses the same default model for the Architecture and
training page. Each graph has the same sensor nodes and actual connections.
GCN names a graph convolution layer; FC names a fully connected layer. Each
layer's ReLU or Sigmoid is labelled. Output sizes and counts of learned numbers
come from the model. Flattening turns the final sensor values into one list:
14 rows of 32 numbers become
448 entries. The circles after that list stand for computed numbers in the
fully connected layers, where every output uses every input. Only some list
entries, circles and lines are drawn, with the full sizes written beneath.
The example in `tests/test_examples_torch.py` checks these labels against a
real FD001 window passed through the default model. On narrow screens the
drawing scrolls horizontally and can receive keyboard focus. The original
`architecture` figure remains available to its other callers.

Article figures, tables and code boxes end at the same right edge as the text.
Wide contents scroll inside that shared width; the figure panel no longer
extends farther into the margin. Paragraphs align to both margins on desktop;
on narrow screens they keep ordinary left alignment to avoid large word gaps.

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

## The playground

The playground page has a folder of its own,
[`playground/`](playground/README.md), and its README explains the page:
its tabs, how a setting is drawn, where each default comes from, the jobs,
the runs, the comparisons and the explanations. The half of the page that runs
in the browser is `assets/playground.js`.

---

## How every page fits the screen

The author's complaint was that the pages were "not responsive at all", and
measured on 2026-09-23 it was so: pages of every kind could be scrolled
sideways at the widths of phones, tablets and laptops, the notebooks' tables by
up to 637 pixels, one of them even at 1024. Two audits of every kind of page at
widths from 320 to 2560 followed, each finding checked by a second reader
before it was acted on. Afterwards none of the 44 pages the menu links to
scrolls sideways at any of 16 widths (320, 375, 414, 600, 768, 901, 940, 1024,
1152, 1199, 1200, 1280, 1424, 1440, 1920 and 2560), nor does the playground
with a run or an explanation open, measured on 2026-09-24 with a 15-pixel
scroll bar; nothing on them reaches past the window except inside a box that
scrolls on its own. What the pages now do, and
why:

- **The menu is a drawer below 1200 pixels**, for the reason given under
  "The menu can be closed" further down, and **the contents list shows only
  where it fits** beside a full line of text: from 1152 pixels with the menu
  closed or open as a drawer over the page (the page's margins 80, the text
  848, the list 224), and from 1424 with it open beside the page (the menu's
  272 more). Where it does not fit, the same links sit folded at the top of
  the page, under "On this page". A tablet keeps a 32-pixel margin, a phone
  16.
- **These widths, and the drawer's 1200, leave out the browser's scroll
  bar**, since a phone's and a Mac's float over the page and take no room.
  Where it does take room, 15 pixels on Windows, a line of text is up to 15
  pixels short of 848 in the first 15 pixels past each width, and never
  shorter than 833: measured, the line is 833 at 1152, 1200 and 1424 and 848
  at 1167, 1215 and 1439. The 848 itself is 74 characters of Sitka Text, the
  serif Windows picks from the page's list of fonts; a system that picks
  another font has another width for the same 74 characters. Every page
  keeps the scroll bar's room whether or not it is long enough to scroll, so
  a chart drawn before the page grew long is not 15 pixels too wide after.
- **On a wide screen the text keeps its line of 74 characters, 848 pixels**,
  and tables, code and figures may grow beside it to 1080; the page's column
  and the contents list stand together, centred in the space right of the
  menu, rather than at the screen's two edges with a band of nothing between.
  At 1920 a figure is 1080 pixels wide beside 848 of text.
- **A table's headings may wrap**, and a name in code inside a cell breaks
  after a dot, an underscore, a slash, a colon, a bracket, a comma or an
  equals sign, never between two letters. Kept on one line, the headings set
  each table's narrowest width and pushed it sideways on a phone; allowed to
  break anywhere, the names were squeezed into fragments.
- **Whatever still scrolls sideways shows a shadow at the edge that hides
  more**: tables, code, formulas, figures, a notebook's output and pictures,
  and the playground's sensor graph and Fig. 10 rings. A phone draws no
  scroll bar until the box is touched. A picture would cover a shadow drawn
  behind it, so a notebook's picture has its shadows drawn over it, dark in
  both themes because every picture stands on white.
- **A long word breaks** rather than push the page wide: a web address, a
  file path, the path at the foot of every page, which breaks after a slash.
- **A code block's language stands above its first line**, not over it; **a
  horizontal line (`---` in the Markdown) just before a section heading is
  not drawn**, since the heading draws its own; **a formula has no scroll bar
  of its own** (MathJax's hidden copy for screen readers reached below it);
  three formulas too wide for a phone were rewritten: one split into two
  shorter formulas, in `docs/learn/04_the_pipeline.md`, and two in
  `06_reading_results.md`, one whose label moved into the sentence before it
  and one with shorter labels.
- **A figure never shrinks below a width its words can be read at**, 820
  pixels: wherever its box is narrower, on a phone or a tablet, it scrolls
  sideways. Shrunk to a phone's screen, its labels fell to 2 to 7 pixels. A
  notebook's plot keeps 65% of the size it was saved at, but no more than
  820, for the same reason.
- **A notebook is held to 1080 pixels**, its text to the reading line, and
  every output and table in it scrolls on its own.
- **Every link to a section works**: the contents list's links, and the # a
  heading shows when the pointer rests on it, lead to the section without
  loading the page again, and each names its page as well, so it can be
  copied or opened in a new tab. Before, the part of `app.js` that reads the
  address after the # and chooses the page to draw read "#section" as the
  address of a page that did not exist. A section whose name holds a
  character beyond plain letters, the "²" of R² or a notebook's "—", is found
  too, and a page opened at a section goes straight to it, rather than
  gliding towards where the heading stood before the figures above it
  arrived. On a touch screen, which has no pointer to rest, the # always
  shows.
- **Search on a phone** opens across the screen under the top bar, not as
  wide as the search box, and closes once a result is chosen; its hint reads
  "Search the site", short enough not to be cut at 320. The site's name gives
  way only below 640, where it no longer fits beside the search box.
- **The text in every field is 16 pixels on a touch screen**, the
  playground's included, where a phone's browser would otherwise zoom the
  page in on it.

Four things the audit found were left: the settings box on the Design tab
keeps the height of its tallest step, as the author asked, although on the
Data tab that puts the Train button below the first screen of a laptop; the
top bar stays pinned on very short windows; the text sizes are in pixels, so
a reader's own default size in the browser does not change them (the page's
zoom still does); and the step names give way to numbers beside the pipeline
from 1232 to about 1417 with the menu open.

---

## What the server will and will not serve

It listens on `127.0.0.1`, which reaches your own machine and nothing else. It
is a reader for your own files, not a way to publish them, and it is worth
being clear about the difference because the project directory contains a
paper's worth of unpublished work.

Six rules keep it that way:

- **Every path is resolved and checked to be inside the project directory.**
  `../../../etc/passwd` is refused, and so is anything else that climbs out.
  There is a test for this.
- **The source viewer shows a fixed list of file types** — `.py`, `.yaml`,
  `.toml`, `.json`, `.txt` and two more. Anything else is refused, so the
  server cannot be turned into a way of reading the whole disk.
- **Files above 400 kB are refused** by the source viewer. Every source file in
  the project is far below that; the raw NASA files are far above it, and a
  3.5 MB table of numbers is not something to read in a browser.
- **The paper address, `/api/paper`, sends PDFs from `docs/references/` and
  nothing else**, whatever their size. A cited paper that is not on this
  computer — the copies are not committed — gets a short page saying that
  `scripts/fetch_references.py` downloads the free ones and
  `docs/REFERENCES.md` says where to obtain the rest. Both are tested.
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

**Every stored run has an address of its own.** Opening a run in Run history
changes the address to `#/explore/playground/runs/` followed by the run's
folder under `reports/`, for example
`#/explore/playground/runs/fd001_physics_row_cap_5seed_varsplit_net9f3068_cfgba82e3_19dceb5_20261002T225228Z`,
so a run can be bookmarked, sent as a link, reloaded, and reached again with
the browser's Back and Forward buttons. The list itself is
`#/explore/playground/runs` and the explaining tab `#/explore/playground/explain`.
Opening or leaving a run adds a step to the browser's history; moving between
the three tabs only corrects the address, so Back does not walk through every
tab clicked. When the address changes while the playground is on show,
`app.js` hands the part after `playground/` to `playground.js` (its `follow`
function) instead of building the page again, which would lose the form and
the list's place. A run's page is the one place with a link above its title,
"All runs", because it is the one page that sits below another page rather
than directly below the menu.

**A run's page shows one part at a time.** Under the run's name a row of four
tabs chooses between the overview (the four results and the notes that
qualify them), the charts, the explanations stored for the run, and the
settings it ran with. The author asked for this on 2026-11-14: the page first
showed all of it in one column 1280 pixels wide, which was too much to take in
at once and left a third of a wide screen empty. Every part now takes the
panel's whole width. The part on show follows the run in the address, for
example `…/runs/<folder>#charts`; the overview adds nothing, and changing part
replaces the address rather than adding a step, so Back still leaves the run.
An address written before the tabs existed, such as `…#pg-run-settings`, opens
the part that holds that section. The notes stay on the same part as the
results they qualify; the charts, which also show results, carry one line that
counts the notes and leads to them.

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
loaded when the server starts and never again. Edit a file in `playground/` and a
running server keeps answering with the old code while serving the new script,
which then asks for things the old code does not send. That happened here on
2026-09-16: a server started two days earlier served a page that expected the
paper's Table 12 and did not get it. The server now compares the modification
time of every Python file in the package with the moment it started, reports
the answer as `stale` in `/api/state`, and the page shows a banner under the
top bar telling the reader to restart it. The banner scrolls away with the
page: pinned under the bar, as it first was, it covered the top of the menu,
of the contents list and every heading a link leads to. A server too old to report the
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
button at the left of the top bar hides or shows it. On a wide screen the
playground spreads into the width the menu gave up, a document keeps its line
of text and moves to the middle of the wider space, and from 1200 to 1423
pixels the contents list comes back beside the text. The choice is remembered
in the browser,
so a reader who works in the playground with the menu shut finds it shut next
time. A narrow screen is one below 1200 pixels: the menu (272), the page's
margins (80) and a full line of text (848) need 1200 side by side, and below
that a menu beside the page took 272 pixels from the text (the section "How
every page fits the screen" says what the browser's scroll bar adds). There
the menu is a
drawer over the page: it starts closed,
opens over a dimmed page, and closes when the dimmed page is tapped, when
Escape is pressed, or when a page is chosen from it. The state is one
attribute on the document, `data-nav`, and a few lines in `shell.html` set it
before the first paint so a closed menu does not flash open while the page
loads.

**"About n minutes" is a word count, not a promise.** It divides the prose word
count by 220 words a minute. It is there to warn you that a page such as the
playground's README ("The playground" in the menu) is more than an hour's reading (73 minutes on 2026-10-03), not to
be accurate about you. (Corrected 2026-10-03: this named the assumptions log as
an hour's reading. As one file it came to 339 minutes by the same count; since
that day it is a folder of pages, one per entry.)

**The assumptions log is parsed, not just displayed.** Each assumption and
each finding is a file of its own, such as `docs/assumptions/A-07.md` or
`docs/findings/F-16.md`. `results.py` lists those files with
`log_entry_files()` from `utils/paths.py` and reads the first line of each,
picking out the entry's identifier and title, so that a link such as
`#/entry/A-07` opens that entry's own page. That parsing depends on the first
line keeping its style — `# A-01 — The label ceiling is 125 cycles`. A test
checks that the entries come out numbered consecutively with no gaps, so if a
future entry is written differently the test fails rather than the link
quietly going nowhere. The entry pages are not in the menu, which lists the
two folders' README pages, Assumptions and Findings, instead; each entry opens
from the table on its folder's page, and the site's search reads every one of
them. The Assumptions page also shows, under its title, the result of the last
run of `scripts/verify_claims.py`, which recomputes the numbers the log quotes,
the findings' included: how many it re-checked, how many disagree, and when.

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
text. The assumptions and findings are searched as well, although each is a
page of its own outside the menu; a match in one is named by the entry's own
heading, such as `A-14 — Weights are initialised with Glorot uniform sampling`
for the entry on the network's starting weights. The notebooks and the playground are not searched. It is a plain
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
a block of HTML explaining what is absent and how to get it. Python records, on
a failed import, the name of the module it was importing, and only when that
name is `torch`, or one of its parts such as `torch.nn`, does the page say
PyTorch is missing. Any other failed import is shown as the error it is, with
the module's name in it. Until 2026-10-02 every failed import that reached
`render_figure` got the PyTorch notice, so a module of this repository renamed
without the figure that imports it being updated would have sent the reader to
install a library that was not the problem. The worked example's failed imports
never reached it: `worked_example.py` caught them itself and reported each one,
a missing PyTorch included, as the FD001 files or its saved run being absent.
Since that date it leaves a failed import to `render_figure`, so the same rule
covers every figure. It never raises, because a missing figure should not take
the page down with it.

### `load_entries(root)` and `latest_verification(root)`

The first returns every assumption and finding in the log as an `Entry`, with
its identifier, title, status and the path of its own file, such as
`docs/assumptions/A-07.md`: every assumption in number order, then every
finding. The second returns the
most recent run of `scripts/verify_claims.py` stored in `reports/`, as a
`Verification` holding how many claims it re-checked and how many failed, or
`None` if it never ran.

### `render_notebook(path, root)` and `start_jupyter(port, root)`

The first returns a notebook's cells as HTML, with the outputs it was saved
with. The second starts JupyterLab and returns a `JupyterProcess`, the record of
a running server — where it is serving, how it was started, and a `stop()` that
shuts it down.
