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
| `playground.py` | Checks the playground's settings, describes each box of its pipeline as a flowchart of its own, builds the command that runs them, queues the jobs, and reads stored runs and explanations back. |
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

## The playground: design, results, explain

The page at `#/explore/playground` is where the parts of the method can be
swapped and the effect seen. It has three tabs, in the order the work happens:

1. **Design a run.** Choose the sub-dataset (one of the four parts of NASA's
   simulated turbofan engine data, FD001 to FD004), the preprocessing, the
   temporal stage (the part that turns each sensor's window of recent cycles
   into the few numbers per sensor the graph layers receive; the paper's is an
   LSTM, a kind of recurrent network, one for each sensor), the sensor graph
   and the widths of the graph layers (GCN, for graph convolutional network,
   the layers in which each sensor's numbers are mixed with its neighbours'
   in the graph), and the training settings. The graph layers themselves stay
   the three of the paper's Table 4; only their sizes change.
2. **Results.** Everything that has been run, under two tabs of its own:
   the training runs and the explanations, each with the jobs that have not
   finished among them. A run opens with its four measures beside the
   paper's, every seed's curve and predictions, and any two runs can be
   paired seed by seed.
3. **Explain.** Explain one stored run's predictions, with the paper's method
   or a cheaper one, and read back how much each sensor moved them.

The design tab has twenty-nine settings, and shown all at once they ran to five
screens. So the form is laid out as six numbered steps, with a map of them. The
**pipeline** draws the whole run as a flowchart, in the order the data passes
through it: the sub-dataset, the preprocessing, the temporal stage, the graph
layers, and the remaining life that comes out. The form shows one step at a
time, under six tabs: Data, then five numbered 1 to 5 — Preprocessing,
Temporal stage, Graph layers, Training and Repeats.

The data are what goes into the pipeline, not one of its steps, so the
pipeline draws them the way it draws the remaining life that comes out: a
picture and words, "FD001", above the first arrow, rather than a numbered box.
They still open their own flowchart, and their tab carries their picture where
the others carry a number. The steps are then numbered from 1, Preprocessing.
The author asked for this on 2026-09-26.

Each box of the pipeline shows its step's number in its top left corner, a
small picture of its step, and what the step is now, "LSTM" or "GCN", so a
step's name is said once, on its tab, and the number ties the two together.
The preprocessing box is the exception: its step has no one value to show, so
it carries its step's name, and a screen reader hears that name once. The
pictures are a store of records for the data, sliders for the preprocessing,
whose steps adjust the readings, a clock for the temporal stage, which reads
the cycles in the order they happened, and three joined circles for the graph
layers; an hourglass stands above "Remaining life". A funnel stood for the
preprocessing at first, and was changed because it reads as "filter", while
the preprocessing removes nothing. The step being worked on is marked by its
filled number alone. The same picture heads the step's own flowchart. The
pictures are drawn in the page itself, not fetched, and say nothing a screen
reader needs, so it skips them.

The six steps' settings are all drawn, one over another in the same place, and
only the chosen tab's are shown. The box that holds them is therefore as tall
as the tallest step at the window's width, and choosing another tab never moves
anything on the page: at a window 1440 pixels wide with the menu open it is
489 pixels tall whichever tab is chosen, and 451 with the menu closed, where
the wider form fits three settings in a row rather than two (measured on
2026-09-26; before the settings were redrawn that day, as the next section
describes, it was 593 and 517). The first
version had the six groups fold open one under another, which moved everything
below each time one opened. On a phone, where the settings stand one across,
the tallest step would leave the others a screen of empty box, so there each
step takes its own height; so it does on a screen less than 640 pixels tall,
such as a phone on its side, where the box and the top bar could not both show
at once anyway.

When the six names do not fit in one row, only the chosen tab keeps its name on
screen and the others show their numbers, each still named for a screen reader
and when the pointer rests on its number. On the narrowest phones, where even
that row does not fit, every tab shows only its number, the chosen one filled
in and underlined: at 320 pixels wide that happens with Preprocessing, Temporal
stage or Graph layers chosen, whose names are the longest. All of it is
measured, not guessed from the width: at a window 1440 pixels wide with the
menu open all six names fit, at 1280 they do not, and at 375 the chosen tab
always keeps its name. The arrow keys, Home and End move from tab to tab, as in any row of
tabs. The chosen step is remembered in the reader's own browser, and the page
opens on it.

A tab carries a small dot on its number while any of its step's settings
differs from its default. The chosen tab does not, because there every such
setting shows its own Reset. Reset all stands at the foot of the settings card,
under whichever step is shown; with no heading above the card, it starts level
with the pipeline's card beside it.

Where the Design tab's own area, the page to the right of the menu less its
margins, is at least 880 pixels wide, the pipeline stands top to bottom in a
column to the right of the form, and stays where it is while the page scrolls,
so the map is always in view. The form takes up to 960 pixels and the column
everything to its right, never less than 400: at a window 1920 pixels wide with
the menu closed, and no scroll bar taking room, the column is 841 pixels wide. The author asked for the column
to fill the page's width, where it had stopped at 480 and left a band of
nothing at the right. Inside it, the boxes keep the widths they have in a
narrow column, 320 pixels for a box of the overview and 480 for a step of a
flowchart, and stand in its middle.

The column is exactly as tall as the form beside it, or as the window less
the top bar and the page's bottom margin where that is less, whichever box is
open: measured at 1440 by 900 with the menu open, it is 669 pixels tall with
the overview showing and with each of the four flowcharts open, the form's
own height. It used to be as tall as what it showed, up to that limit, so
opening a box made it grow and going back made it shrink, which the author
asked to stop. Its heading, and with a flowchart open the way back, stay at
the top, and what is under them scrolls on its own when it is taller; the
overview's boxes stand in the middle of the height. The column never
lengthens the page and never slides under the top bar at the page's end. At
880 the form beside the column keeps 456 pixels, room for two cells of its
grid (next section): the card needs 418 for them, 384 for the two cells and
the space between them, plus its padding and border. On a narrower
area there is no room for both, and the pipeline stands above the form instead,
left to right when its boxes fit. With the menu open, the area reaches 880 in a
window 1232 pixels wide plus the width of the browser's scroll bar.

Choosing a box opens that step's own flowchart in the pipeline's place, and
opens the same step's tab in the form, so beside the form the settings are
next to the drawing of what they decide; above the form, on a narrower area,
the tab is further down the page, and each value in the flowchart leads to its
setting. "Pipeline" (or Escape) leads back; the section after next says what
each flowchart shows.

The sensor graph is built while the data is prepared, but its setting sits in
the Graph layers step, beside the layers that use it, because that is where
the pipeline draws it: as the first step of the graph layers' own flowchart,
not as a label on their box.

The chosen step is the current one. Its number is filled in, on its tab and on
its box in the pipeline, and every other number is an outline, which also keeps
the form's numbers from reading as the numbers of the Design, Results and
Explain tabs above them.

The form ends, below the settings, with the card that runs it: the network the
settings build, what stops or qualifies the run, and the button that starts it.
It stands last so the page reads from top to bottom in the order the work is
done, and so each caution sits directly above the button it is about.

### How a setting is drawn

Each setting's name stands directly above its control, followed by its i
(the small circled i beside a name, which opens a card saying what the setting
is for and where its default comes from; the section "Where each default comes
from", further down this part, describes it), and the settings of a step stand
side by side in a grid of cells: one across in a card less than 416 pixels
wide inside its border, two across from there, and three from 834, where each
cell is at least 256 pixels. Measured on 2026-09-26, in windows 1440 and 1920 pixels wide with
the menu open or closed, one name takes two lines, "Same held-back engines for
every seed", which wraps beside its switch; in a window 1024 wide with the menu
closed or 1280 with it open, where two cells of 228 and 220 pixels stand
across, five do. The first version put the name at the left of a row and
the control in the row's right half, so the eye crossed the page from one to
the other. Measured at a window 1440 pixels wide, from the end of each
setting's name to the start of its control, that crossing was 252 pixels for
the middle one of the 28 settings on show and 355 at the most; with the name
above, the middle one is 6 pixels. Rows of cells also take less height than a
row for every setting: with every group open one under another, the groups took
2814 pixels, and 1995 once their settings stood in cells, before the groups
became tabs.

The cells of one row share their two lines, the names' and the controls': a
name that needs two lines pushes its neighbours' controls down with its own,
so the controls of a row always stand on one line and every name sits on the
line just above its control. A name that wraps keeps its i on the line of its
last word. The browser will break a line where a name's element ends and the
i begins, whatever character stands between the two, so each name is drawn in
two parts, all but its last word and then its last word held on one line with
the i; both parts name the control, and Chrome's accessibility tree reads
"Passes over the training data" as the name of that field, not the first part
alone.

A switch is the one control whose name stands beside it, to its right, on the
line of the other controls, as a switch's name does in every other program;
a click on the name turns it, and a name too long for the room beside the
switch wraps there. Above the switch, its name stood alone on the names' line
over a control two thirds of a field's height. A number with a unit shows the
unit inside the field's box, "30 cycles", and every number field but the three
graph layer widths is the same width, 128 pixels, so the fields of a step line
up; before, the unit floated beside a field 96 pixels wide.

**The settings of a pipeline box follow its flowchart.** In the tabs of the
preprocessing, the temporal stage and the graph layers, the settings stand in
groups named after the steps of the box's own flowchart that they decide, in
the order those steps run, each group under the same small picture the step
has in the flowchart: Preprocessing reads "Cap the labels", with the label
ceiling and whether the test labels are capped, then "Smooth each sensor",
with its weight, then "Rescale each sensor" and "Cut into windows". A group
that holds a single list of options names it, so the list has no name of its
own and its i stands beside the group's name; a step that can be switched off,
the smoothing, has its switch in its group's top right corner. Switched off,
the group stays in view, faded, with its settings out of use, so the reader
sees what switching it back on would bring, and the step leaves the box's
flowchart, since it no longer happens. A group whose settings
are all hidden, the recurrent network's size when the stage is not recurrent,
is hidden with them, and a group's name that depends on the settings, "One GRU
per sensor", follows them. `SECTIONS` in `playground.js` holds the groups; the
author asked for them on 2026-09-26. The Data, Training and Repeats tabs have
no flowchart to follow and are grouped by subject the same way: the
sub-dataset; the procedure, the passes and the learning rate; the seeds and
the device. Each group stands on a panel of its own, in the panel grey, so it
is seen as one thing and told from the next, as the author asked; the panels
stand 12 pixels apart.

A list of options, such as the sensor scaling, takes a whole row and stands as
many options across as the settings above it stand, one, two or three, so the
radio buttons fall on the settings' columns: each option is as wide as a
column and the 16 pixels between two, 8 of them on either side. The words of an
option are its value, so they are set in the regular weight, and the setting's
name above them in the heavier one. The radio button is the control, so an
option has no frame of its own: the chosen one is filled with a tint of the
accent colour and the one under the pointer with a pale grey. With a frame
round every option, the Training step alone drew seven frames. The list reaches
8 pixels past its cell on each side, so its radio buttons stand in line with
the names and fields above them.

The one list of more than four options that shares a single source, how the
learning rate moves from its first value to its last, is a menu instead, in
the row of the two rates it joins; it took three rows of options on a row of
its own, and made Training the tallest step and every step that tall. A list
whose options come from different places keeps its options in view, each with
its own i. The graph layer widths share one cell, each of their three fields a
third of it up to 72 pixels. The sub-dataset takes a row of its own, its four
choices under its name like every other setting's control. Inside a step, the
lists about what the step is for come first: the temporal stage, the sensor graph and its rescaling. The
small settings follow, and a list about something else, the sensor scaling,
follows them on a row of its own; `FIELDS` in `playground.py` holds that order.

Reset, which appears once a setting differs from its default, is put where it
can appear without moving what the pointer is on: beside a small setting's
control, beside a list's name or its group's switch, and under a list that its
group names. Choosing
it leaves the keyboard's focus on the setting's control, and Reset all leaves
it on Reset all; neither draws the form again, so the page does not move. The
inputs only the chosen procedure shows stand directly under its options, on
the same columns as every other setting. They were once indented, with a line
down their left edge, which set them under the first option, the one they do
not belong to.

The Explain tab's form is drawn the same way, by the same function.

A number field's value is reported when the field loses the focus, which a
press of the mouse elsewhere causes. What depends on the settings is therefore
changed in place, never drawn again, unless what it shows has changed: a click
on a pipeline box, a flowchart's value or a Reset, made straight after typing a
number, would otherwise land on a button that had just been replaced, and be
lost. For the same reason a step's open flowchart only turns grey when the
settings move, and its steps are replaced when the new answer arrives.

Each number field's default lies on the steps its arrow keys take, counted
from its lowest allowed value; a test checks it. The browser marks any other
value invalid, and three fields once held their defaults between two steps,
so their arrow keys could never land on the paper's values again. Their lowest
allowed values were moved onto the steps: a first learning rate of at least
0.0001, a last one of at least 0.00001, and a smoothing weight of at least 0.05.

A switch turns wherever it is clicked. Its drawn track lies over the
checkbox, which is invisible, and lets the click through to it; before, only a
switch inside a label turned, since the label passed the click on, and the
switches of the Design tab, which sit beside their names rather than inside
them, could not be turned by a click at all.

The page tells the browser which of its two colour schemes is showing, so the
browser's own parts of a control, a radio button's circle and a number field's
arrows, are drawn to match. Before, the dark page left them light, and an
empty radio button read as a chosen one. A switch or a row of choices shows a
solid ring in the accent colour when the keyboard reaches it, as every other
control does, since the pale glow used before measured 1.5 to 1 against the
page.

### A box's own flowchart

A box of the pipeline stands for several steps of the code, and its flowchart
shows them, top to bottom, in the order the code runs them. Each step is a box
of centred words: its name, the values that decide it and how many numbers it
learns; what it does, in a sentence or two, and, under a thin line, where it
comes from are behind its i. Each value a setting decides is a link to that
setting, so a step decided by two settings, such as the window's length and its
spacing, carries two links.

Each step shows a small drawing of what it does above its name, so it can be
recognised and remembered by its shape as well as its words: a jagged line
made smooth for the smoothing, a window over a line of cycles for the cutting
into windows, a few sensors joined by lines for the sensor graph, a table for
its rescaling, one sensor gathering from four for a graph layer, an S-shaped
curve for the output. The server names each step's drawing (its `picture`),
and the groups of the form that decide a step carry the same drawing. The
words that open and close a flowchart, what comes in and what goes out, stand
with the picture of the box they come from or go to, as the overview draws
its input and output.

The sensor graph is not a step the windows pass through: it is a second input
of the first graph layer. So the graph layers' flowchart draws it as a branch
beside the main line: the arrow from the temporal stage carries `(B, 14, 1)`
straight down to graph layer 1, and beside it the sensor graph and its
rescaling end in their own arrow, `(14, 14)`, into the same layer; each of
the two writes its shape once, on the arrow under it. Drawn in a
single line, as it first was, the arrow into "Sensor graph" said the windows
went into the graph.

With the LSTM or GRU stage, the step that runs the networks shows one button
per sensor, fourteen in all, as the paper's Section 2.2 gives each sensor a
network of its own; one button when a single network is shared. A button
opens that network from inside in the pipeline's place, and the way back leads
to the stage's flowchart: a drawing of the network as it is usually drawn,
with the memory along the top for an LSTM, the gates fed by the cycle's
reading and the hidden state, and the new hidden state it hands on, then each
gate in words with its learned numbers, read from the built network, and how
many times the network runs, once per cycle of the window. The fourteen
networks have the same parts and learned numbers of their own. The author
asked for the drill-down on 2026-09-26.

Every arrow carries what passes along it. Before the windows are cut, what
passes from one preprocessing step to the next is the training file's table,
`(20 631, 27)` for FD001: one row per engine and cycle, 27 columns, and 28
once the operating condition has been added; the test file's table goes the
same way beside it, and each step's i says what happens to it. At first these
arrows carried nothing, since the steps do not change the table's shape, and
the author asked why some arrows had an output and some did not. A row of
steps side by side names what the row hands on, the Data flowchart's three
files "3 tables". The arrows are 48 pixels long, and 80 where they carry a
shape, so each step stands clear of the next.

The shape of what a step hands on, the size of the table of numbers passed to
the next step, is written on the arrow under the step, and so reads as the next
step's input: each step is seen as what comes in above it, what is done to it
and what goes out below it. In the graph layers' flowchart, for one, the
arrow into "Rescale the graph" carries `(14, 14)`, the table of which sensors
are joined, and the arrow out of it carries the same shape rescaled. Each shape
is written once: the words that open a flowchart put theirs on the arrow under
them, and the words that close it show their own only when it differs from the
one on the arrow above them, as "To the temporal stage, B windows at a time",
`(B, 14, 30)`, does under the arrow carrying all the training windows at once.
Steps that work side by side on the same input, such as the three files of a
sub-dataset, share one row, and each shows its own shape inside its box, since
no one arrow could carry three. In a shape, `B` is the number of windows the
network is handed at a time. Before, the shape stood at the right edge of each
step's box, in a column of its own, beside the step's name, where it read as a
property of the step rather than as the thing passed from one step to the next.

| Box | Its steps |
|---|---|
| 1 Data | the training engines, the test engines and the file of their remaining life, side by side; a remaining life for every row; and the 14 of 21 sensors kept, in Table 3's order |
| 2 Preprocessing | capping the labels, smoothing, holding engines back, finding the operating conditions, rescaling, cutting windows, and the two or three sets of windows that result |
| 3 Temporal stage | for the LSTM or the GRU: splitting the window by sensor, one network per sensor (or one shared by all), its last hidden state, and the one weighted sum, shared by all sensors, that turns it into the numbers the graph layers receive; for the other stages, what each does to a window instead |
| 4 Graph layers | the sensor graph, its rescaling, the three graph layers, the flattening, the fully connected layer, the output and its scaling to cycles |

**A tag never claims a value the paper did not print.** Where a step is an
operation the paper describes, its tag names that part of the paper whatever
the value: capping the labels is the paper's Eq. 19 at any ceiling, and the
ceiling of 125, read off the paper's Fig. 7, is this project's choice
(assumption A-01), said in words. Where a step is itself one setting's value,
such as the window length, the share of engines held back or the kind of
rescaling, it carries that setting's tag only while the value is the default.
Any value that is not the default is said to be so in words, naming the setting
and where its default came from; for a window of 40 cycles the step reads that
the window length "was changed from its default, 30 cycles (the paper's Table
6)", and carries no tag. This rule came from a review that found the first
version tagging a changed window with Table 6, and a shared LSTM with the
Section 2.2 it departs from.

Nothing in a flowchart is typed in by hand as a number. `stage_detail()` counts
the rows and engines from the NASA files, runs the real preprocessing for the
windows and the operating conditions, and builds the real network for the
shapes and the learned numbers, reading them from `layer_summary()`. Of a
preprocessing run it keeps only those counts and the sensor graph, never the
windows, for the last sixteen groups of settings, and it runs one at a time, so
a second request for the same settings finds the first one's answer; the
sensor graph and its rescaling are not part of what it is kept under, since
neither changes the preprocessing, unless the graph is built from the data.
FD004 takes several seconds; while a new answer is on its way the old one stays
on screen with its coloured parts, the links and the tags, turned grey, which
keeps every text at its contrast, and "Updating…" appears beside the step's
name, so nothing on the page moves. The grey is made by giving those colours
grey values, not by a filter over the whole flowchart. A filter, a visual
effect the browser applies to an element and everything inside it, would also
make the flowchart the frame an i's card is positioned against, in place of the
browser window, and the pipeline's column would then cut the card off at its
edge. The correlation
graph is built
from the training engines here, where the preview uses the paper's graph in its
place, so its box can say which sensors it leaves alone and which it makes
identical, and whether every seed of a run gets the same graph: only when
engines are held back, each seed drawing its own, do the graphs differ. On the
paper's own graph that check finds three groups of sensors whose rescaled rows
are identical, so the layers after it can read each group only as a sum (audit
item D-1), and four pairs of sensors further apart than three graph layers
reach, which only the fully connected layer combines. A missing file is never
remembered: once the NASA files are put in place, the next request counts them.

Above the form, on a narrower area, the overview stands in a row when its boxes
fit, and that is measured, not guessed from the width: when they do not fit,
the words "Remaining life" are dropped first, and then the boxes stand in a
column, where the words come back. The names differ in length, "LSTM" against
"Window summary", so no one width could decide it. Beside the form the boxes
always stand in a column, however wide the column has grown, since there they
share the form's height.

Inside the pipeline, while it scrolls on its own beside the form, an i's card
is positioned relative to the browser window rather than to the i: the
column's own scrolling would otherwise cut off whatever passed its edges. The
card opens under the i, or above it when the window has no room below,
follows the i when the column or the page scrolls, and closes once the i has
scrolled out of the column's view rather than point at nothing.

The pipeline comes first in the page's order, for the keyboard and for screen
readers, although beside the form it is drawn on the right. It is the short
map of the form: six stops of the Tab key pass over it, where the form takes
between six, on the Data tab, and nineteen, on Training, with the default
settings, PyTorch installed and the NASA files in place, so the map is reached
at once and left at once. The same order holds when it stands above the form.
Of the page's own four tabs, and of the six steps' tabs, only the chosen one is
a stop; the arrow keys, Home and End move along the row, and with Alt or Ctrl
held they are left to the browser, where Alt with an arrow goes back or
forward.

"Pipeline", or Escape pressed inside the pipeline or with nothing focused (no
control selected for the keyboard), leads back to the overview, with the
keyboard's focus on the box that was open, or on the + that would put it back
when that box has since been taken out. Escape first closes an i's words, if
any are showing; pressed in the form, it leaves the flowchart alone. A value
that links to one of the graph layer widths puts the focus on that layer's own
number field. When a new answer replaces a flowchart's steps, the focus stays
with the button it was on, or moves to the one that took its place: after the
sensor graph's cross, to the + that puts it back.

The words were written from an account of each stage read out of the code and
then checked against it by a second reader, not from the paper: where the code
and the paper's figures differ, the flowchart follows the code. Tests hold the
parts that can drift: the learned numbers of the temporal and graph boxes add
up to the network's own count, what leaves the temporal box is what enters the
graph box, every linked value names a setting the form has, a changed value
never carries its default's tag, a shared network is never tagged as the
paper's, nothing held back means no held-back windows are spoken of, the
rescaled graph's row sums are said as measured, a count of one takes a singular
noun and verb, the data and preprocessing counts equal what the loader and the
pipeline produce, and the sensor graph is the only step of the graph box whose
value links to the sensor graph setting, which is what puts the cross and the
+ there.

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
  no engines held back. Those two are here because the settings involved sit
  under tabs that may not be the one on show, and "Set these settings up again"
  on a stored run can bring either in without the reader opening a tab.

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
physics graph". It is not drawn on the overview, whose boxes are stages; it is
the first step of the graph layers' own flowchart, "Sensor graph", and the
cross on that step joins every sensor to every other instead, the w/o Physics
prior row's graph. Each cross leaves a dashed button in its place that puts
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

The list of runs gives that row's root mean squared error (RMSE: the square
root of the average squared difference between the predicted and the true
remaining life, in cycles) in its Paper column, and a run's own
page names the row and sets its figures beside the run's, or gives the reason
from the table's last column. The design tab names no row: the pipeline already
shows which parts are in, and the paper's number belongs beside a result, not
beside settings that are still being chosen.

### Where each default comes from

Every setting has an i beside its name. Pointing at the i or reaching it with
the keyboard shows a small card in two parts. First, in plain words, what the
setting is for: the problem it answers and what changing it does to the run,
written for a reader who has never met the method, so that they can tell
whether to touch it. Then, under a thin line, a tag saying which of four
things set the default, because that is the distinction the thesis depends
on, and a few words on why. The first seed and the device have only the first
part, since no source decides either. The author asked for the order on
2026-09-26: the card beside "Sensor scaling" had said only where the default
came from, and never what sensor scaling is for. Choosing the i keeps the card
open, and only then can the keyboard reach a link in it; the card closes once
the focus moves on, or with Escape. The form itself shows only names, so it
reads as a list of choices rather than a list of citations.

The plain words are held in `FIELDS` in `playground.py`, as each setting's
`about`, and for the options that have an i of their own, as the option's.
Each was checked against the code it describes, never against the paper,
which is the tag's business, so none of them carries a table, an equation or
an assumption's number. A flowchart step's i has the same two parts: what the
step does, then the tag.

**Accuracy review, 2026-09-26.** The cards were checked again against the
functions that prepare the data, build the network, train it and explain its
predictions. Source labels were checked against the original paper. The
review corrected several claims: capping answers cannot increase squared
prediction errors, but it does not guarantee that every reported measure
improves; a constant training target does not mean an engine stops wearing
out; and replacing one sensor at a time does not generally divide the
prediction into shares that add up. Keeping the same engines held back across
runs also excludes one source of variation, without guaranteeing a smaller
observed spread in every experiment.

The short explanation states what goes in, what the operation does and what
its output means. The source line separates published instructions from
project choices. Details and worked examples remain behind **Read more**.
For sensor explanations, the implementation in `explain/shap.py` and
`explain/occlusion.py` determines the precise meaning of a contribution.
The general distinction between explaining a prediction and establishing a
physical cause also follows the
[SHAP authors' explanation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html).

| Tag | Meaning | Example |
|---|---|---|
| Paper | the paper prints the value | 200 passes over the training data, Table 6 |
| Our choice | the paper is silent, so a numbered assumption chose | the label ceiling of 125 (assumption A-01) |
| Project rule | neither: a rule of this repository | five seeds |
| Not in the paper | an option offered for comparison | the GRU stage |

Every card ends with **Read more**, a link that opens, in a new tab, the part
of the documentation that explains the setting, the option or the step: the
section of a Learn page, of a folder's README or an entry of the assumptions
log. `_READ_MORE` in `playground.py` holds where each one leads, as a document
and a heading written as the document has it, and a test checks that every
one of those headings exists, so a heading renamed later fails the suite
rather than leaving a link that leads nowhere. The author asked for it after
meeting "Find operating conditions" in a flowchart with nothing to say where
it came from. A card is drawn over everything around it: each grid of settings is
drawn by the browser in a layer of its own, since the layout's width rules
measure it, so while a card shows, the setting, the grid and the group that
hold it are lifted above their neighbours; before, a switch and an i of the
next group showed through it.

Each fact is said once: when every option of a setting shares one source, the
tag is behind the setting's i; when the options differ, as for the sensor graph
and the temporal stage, each option has an i of its own. An option's i is not a
stop for the Tab key. Moving onto the option with the arrow keys opens its card,
and a screen reader reads the card as the option's description. The cards of
the Explain tab's options work the same way.

A pass over the whole training set is what PyTorch calls an **epoch**; the page
says "pass" throughout.

The training procedure is the first choice in the Training step: **the
paper's procedure** (nothing held back, every pass, the final weights) or
**held back, best pass** (a fifth of the training engines held back, and the
weights of the pass that scored best on them restored). It is the only control
for those two things; while the second is chosen, the share of engines held
back and how many passes may go by without improvement sit under it. It does
not set the number of passes, which is the next setting in the step. The two
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
Stop button, and nothing at all when nothing is live, so the tabs never move;
the line is hidden while the job's own row is on screen in Results. Results has
no separate list of jobs. A job is a run or an explanation that has not
finished, so every job that is waiting, running, failed or stopped is a row of
its own at the top of the Results tab it belongs to, training runs or
explanations, with its state, its bar and what is happening now, its last
printed lines when it failed, and Stop or Remove; a finished job is its stored
run or explanation, listed below. The Results tab's badge counts the jobs
running or waiting. While Results is open its job rows follow each poll: they
are drawn again only when a job comes, goes or changes state, and otherwise
only their bars and lines move, so a filter being used keeps its focus. The
author asked for the two lists to become one on 2026-09-26, having found no
difference to tell them apart by. Results holds
every job this server ran and every one an earlier server left on disk, read
back from its folder. A job whose server stopped before it finished is listed
as interrupted, since a run writes nothing until its last seed is done. One
that **another server is still running** is listed as running instead, and
refused for removal: every job writes down the number of the process running
it, and a server that finds that process still alive knows the command is
writing into that folder at this moment. Without that check the job would read
as ended, and removing all the ended ones would delete the folder from under a
running command. A job that has ended
can be removed; removing one deletes its folder under
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
announced by name unless its row is on screen, where it changes in front of
the reader. One trap here: a job is announced when a poll finds it ended
after an earlier poll saw it live, and a short explanation can finish between
the request that starts it and the next poll. So a job this page started is
also announced when the very first poll already finds it ended.

### A run, and two runs

Each list shows ten rows a page, with the way to the page before and the page
after and which rows are on show, "11–20 of 34". Its filters stand above it,
each a name over a menu that starts at "All": for the training runs, the
sub-dataset, the temporal stage and the sensor graph, from the values the
stored runs hold, where the weights were taken from, the best pass or the
final one, and the state, finished, running or waiting, or failed or stopped,
with two switches for five seeds or more and for runs that kept their weights;
for the explanations, the method, what was explained, the sub-dataset and the
state. A job's own settings are not kept with it, so a job row answers only
to the state. Changing a filter goes back to the first page and leaves the
focus on the filter. A row of the explanations opens that explanation in the
Explain tab.

The list of runs gives, after when each finished, its RMSE and the paper's,
then the settings that produced it, and last the commit; a narrow panel leaves
the commit out, the column least needed to read a result. The date is written
without its year ("09-14 23:39"), the column's heading says the zone, and the
full date shows when the pointer rests on it: with the result third, a phone
shows it without scrolling the table. How the weights were chosen stays at any
width, since it qualifies the RMSE beside it (finding F-16), and so does the
"files" mark beside it, the one sign that a run kept its networks and can be
explained. A run is ticked for
comparison anywhere in its first cell, a target 40 pixels square, and a click
there never opens the run.

Click a run and it opens on its own, in place of the list, with a button back
to every run; the list is long, and a run's curves and predictions drawn
underneath it were a screen's scroll away. Its charts stand in one column no
wider than 1080 pixels, and each is drawn at the width it is shown at, again
whenever that width changes: one unit of the drawing is then one pixel on the
screen, so the numbers on its axes are held to the site's smallest size, 11
pixels, and its axis names are 12, at every width. Drawn once at a fixed width
and stretched to the column, as they first were, they fell to 4 pixels on a
phone and grew to 36 at 2560. A chart drawn again replaces the old one where
it stands, with no loading line between, so a change of width, the menu
closing or a different seed chosen for Fig. 8 leaves the reader where they
were, and the focus on the control they used. Every tab's boxes take the page's whole width, as the author asked on
2026-09-26, and the words beside a chart keep a readable line length however wide the tab. A commit ending in `-dirty` had
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
is not something the paper's table shows. Each stage's rings stand five to a
row at no less than 150 pixels each, so the names inside them stay readable;
where a stage is narrower, on a phone, it scrolls sideways, with its heading
staying in view, and in the first stage the names of its two rows, the
paper's and this engine's. Every name in the rings and on the sensor graph is
drawn after all the circles, so no circle covers one, and resting the pointer
on a name shows the same words as its circle. The sensor graph never shrinks
below 540 pixels, where its names are 8.6: on a phone it scrolls sideways
instead of falling to 4. In the ranked row above it, each circle is 50
pixels across, room for the longest name, htBleed, at 11.

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
