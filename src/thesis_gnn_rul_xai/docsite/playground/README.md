# `playground` — design a run, train it and explain it, from the browser

This folder is the Python half of the playground, the page of the website at
`#/explore/playground` where a run of the network is designed, trained, set
beside the paper and explained. The other half, the part that runs in the
browser and draws the page, is `../assets/playground.js`. When the page needs
something — the choices for its form, the flowchart of one box, a stored run —
it asks the server, and the server answers by calling a function from this
folder; `../server.py` holds the list of those addresses.

Nothing here computes a result. A training run started here is
`scripts/train.py` and an explanation is `scripts/explain.py`; each stores what
it computed in `reports/`, and `runs.py` reads it back.

---

## The files

| File | What it holds |
|------|---------------|
| `settings.py` | Every setting the form offers (`FIELDS`), in the order it draws them, with where each default comes from; the two training procedures, and the variants of the paper's Table 11 written as settings; where each "Read more" leads; and `validate_settings`, which checks what the page sent before anything runs. |
| `commands.py` | The command a checked request becomes (`train_arguments`, `explain_arguments`), and `options()`, everything the page needs to draw its form before anything is chosen. |
| `design.py` | What the Design tab shows before anything runs: the facts about the chosen sub-dataset's files, the cautions (`caveats`), the network those settings build (`preview`), and each box of the pipeline drawn as a flowchart of its own (`stage_detail`). |
| `runs.py` | The runs and explanations stored in `reports/`, read back for the page, and two runs compared seed by seed. It also checks a request to explain a stored run, since that means finding the run. |
| `jobs.py` | The queue: it runs one command at a time, follows its progress through the file the script writes as it goes, and stops it when asked. |
| `__init__.py` | Nothing of its own. It imports the names the server and the tests use from the five files and offers them under the folder's name, so `playground.validate_settings(...)` works without saying which file it is in. |

The table is in the order the files depend on one another: each one imports
only from files above it.

```
settings.py   imports from none of the others
commands.py   imports from settings.py
design.py     imports from settings.py and commands.py
runs.py       imports from settings.py and design.py
jobs.py       imports from settings.py, commands.py and runs.py
```

That order is what lets Python load them at all. If two files each imported a
name from the other, whichever was loaded first would stop half-way to load
the second, and the second would then ask for a name the first had not reached
yet: Python's error is "cannot import name … from partially initialized
module".

**Why five files.** Until 2026-10-02 all of this was one file,
`playground.py`, of 2,713 lines, and it was split along the five jobs it did.
The split was worth making because those jobs are only loosely tied: counted
over every top-level name, 144 uses of one stay inside its file and 33 cross to
another, and 26 of those 33 are uses of `settings.py`. A few names were placed
by what they use rather than where they stood: `caveats` sits in `design.py`
because it reads the data files' facts, `validate_explain` in `runs.py` because
it looks for the stored run, and the page's names for the temporal stages and
the explanation methods in `settings.py`, which every other file may read.
Nothing else changed: every statement and every comment of the old file is in
exactly one of the new ones, checked by comparing the two programs' syntax
trees statement by statement.

---

## The playground: design, explain, history

The page at `#/explore/playground` is where the parts of the method can be
swapped and the effect seen. Its three main tabs have icons beside their
names: sliders for Design a run, a magnifying glass for Explain predictions,
and a clock with a backward arrow for Run history. They are sections that can
be revisited independently, so they carry no step numbers:

1. **Design a run.** Choose the sub-dataset (one of the four parts of NASA's
   simulated turbofan engine data, FD001 to FD004), the preprocessing, the
   temporal stage (the part that turns each sensor's window of recent cycles
   into the few numbers per sensor the graph layers receive; the paper's is an
   LSTM, a kind of recurrent network, one for each sensor), the sensor graph
   and the widths of the graph layers (GCN, for graph convolutional network,
   the layers in which each sensor's numbers are mixed with its neighbours'
   in the graph), and the training settings. The graph layers themselves stay
   the three of the paper's Table 4; only their sizes change.
2. **Explain predictions.** Select a stored trained run, apply an explanation
   method, and explore its output. The run can be changed here directly.
3. **Run history.** Everything that has been run, under two tabs of its own:
   the training runs and the explanations, each with the jobs that have not
   finished among them. A run opens with its four measures beside the
   paper's, every seed's curve and predictions, and any two runs can be
   paired seed by seed. The badge counts jobs running or waiting.

The design form shows one of five tabs at a time: Data, Preprocessing,
Temporal stage, Graph layers, and Training. Training contains the procedure,
passes, learning rate, repeats, device, and the run summary and Train button.
The **pipeline** draws the data, preprocessing, temporal stage, graph layers,
and the remaining life that comes out. Its numbers still match the inner
design tabs. A browser that remembered the former Repeats tab opens Training.

The data are what goes into the pipeline, not one of its steps, so the
pipeline draws them the way it draws the remaining life that comes out: a
picture and words, "FD001", above the first arrow, rather than a numbered box.
They still open their own flowchart, and their tab carries their picture where
the others carry a number. The steps are then numbered from 1, Preprocessing.
The author asked for this on 2026-09-26.

Below the four sub-dataset choices, the Data tab describes the selected data
instead of leaving the rest of the tab empty. Its first row carries the four
counts transcribed from the paper's Table 5: training engines, test engines,
operating conditions and fault modes. The files beneath are measured from the
three real files in `data/raw/`: their row and column counts, engine counts and
the shortest, middle and longest history. The sensor catalogue then lists all
21 sensors with their file column, measurement, unit, location and whether the
paper uses them. Under it, the row section explains the 26-column structure and
shows ten consecutive rows measured from the selected training file: engine,
cycle, all three operating settings and the first three sensor inputs. Changing
FD001–FD004 updates the facts and sample in place. When the files are not
present, the page keeps the published counts and shared column description but
says plainly which measurements it cannot make.

Each box of the pipeline shows its step's number in its top left corner, a
small picture of its step, and what the step is now, "LSTM" or "GCN", so a
step's name is said once, on its tab, and the number ties the two together.
The preprocessing box is the exception: its step has no one value to show, so
it carries its step's name, and a screen reader hears that name once. The
pictures are a store of records for the data, sliders for the preprocessing,
whose steps adjust the readings, a memory cell with a return arrow for the
temporal stage, and three joined circles for the graph
layers; an hourglass stands above "Remaining life". A funnel stood for the
preprocessing at first, and was changed because it reads as "filter", while
the preprocessing removes nothing. The step being worked on is marked by its
filled number alone. The same picture heads the step's own flowchart. The
pictures are drawn in the page itself, not fetched, and say nothing a screen
reader needs, so it skips them.

All five design tabs, including Training, fill the same available height
between the navigation and the bottom of the page. Their contents scroll
between the fixed tab strip and Reset all, so opening Training or expanding
its layer details does not resize the surrounding boxes. Hidden panels take
no space. On a narrow screen the diagram takes two fifths of the available
height and the settings take three fifths; both scroll independently. Very
short screens retain a minimum page height so both areas remain usable.

When the inner design tab names do not fit in one row, only the chosen tab
keeps its name on screen and the others show their numbers. Each remains
named for a screen reader and when the pointer rests on it. If that row also
does not fit, all names are visually hidden. The chosen step is remembered
in the reader's browser. The main tabs always keep both their icons and text;
their text wraps on a narrow screen.

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

The diagram and settings columns share the available page height. Their
headings stay in place while the content below scrolls. At a 1440 by 900
viewport with the navigation open, both measured 735.2 pixels tall on
2026-09-27, ending 16 pixels above the bottom of the viewport. The height
remained the same across all five design tabs. The overview's boxes stand
in the middle of the diagram column. At
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

The chosen design step's number is filled in on its tab and on its pipeline
box. The main navigation uses icons, so its sections are visually distinct
from these numbered steps.

Training ends with a Run summary inside the same settings panel. Its card has
the same background, rounded corners, padding, and icon heading as the settings
cards above it. It names
the selected data, temporal stage, graph and window length, followed by the
built network's size and expandable layer details. These describe the whole
configuration, including settings from the earlier tabs. Anything that stops
or qualifies training stands directly above the Train button. Changing the
number of repeats updates that button and the relevant cautions.

The Procedure section is visible even though its controls are provided by
the procedure renderer rather than its ordinary field list. An empty field
list alone must not hide that section.

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
author asked for them on 2026-09-26. The Data and Training tabs have
no flowchart to follow and are grouped by subject the same way: the
sub-dataset; and the procedure, passes, learning rate, repeats and device. Each group stands on a panel of its own, in the panel grey, so it
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
follows them on a row of its own; `FIELDS` in `settings.py` holds that order.

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

The Explain predictions tab's form is drawn the same way, by the same function.

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
its rescaling, arrows from connected sensors into a filled sensor for a graph
layer, an S-shaped curve for **Predicted life fraction**. That step produces
one fraction of the life ceiling per window; **Scale to cycles** converts it
to remaining flight cycles. Its visible caption explains that `(B,)` contains
B values in one dimension, and its information card explains that B counts
windows processed together. The last layer initially gives `(B, 1)`;
`squeeze(-1)` removes the one-value axis without removing any predictions.
The server names each step's drawing (its `picture`),
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

Every arrow carries what passes along it. **Preprocessing names the quantities
in words:** rows, columns, sensors and cycles. A row means one engine at one
flight cycle. Boxes name the operation and its selected setting; the arrows
carry the sizes. The selected sensors become model inputs when windows are
made; the identifiers and answers remain in the working table. The separate
test file is named at the entrance and stays separate for scoring.

**Correction, 2026-09-27:** the earlier arrows repeated `(20 631, 27)` with no
explanation of either number. The author could not tell what the first meant.
The counts were correct, but the diagram assumed the knowledge it needed to
teach. An initial replacement put real-row calculations in every box. That
made the flow hard to follow, so on 2026-09-28 the examples were moved out of
the diagram. The calculation details belong on **Follow one real example**.
The engine split and final window groups still show their counts with their
purpose. The exit defines `B` as the number of windows processed together
and names the chosen maximum batch size.

For FD001's default settings and seed 42, engine 1 at cycle 2 starts with
sensor T24 at 642.15 and 190 cycles left. Capping changes the answer to 125;
smoothing changes the reading to `0.5 × 642.15 + 0.5 × 641.82 = 641.985`,
using the first cycle's reading, 641.82. Rescaling gives −1.6039, displayed
rounded to four decimal places. These are different changes to different
columns. Its first windows span cycles 1–30, 2–31 and 3–32, with the remaining
life at each final cycle as their answer. The actual window endpoints and
targets returned by the pipeline remain available in its response
for checks; changing the length or spacing changes those examples. The worked
example can be checked with:

```bash
python -m pytest tests/test_examples.py -k preprocessing -v
```

The file rows, split counts, changed readings and window ranges are read or
computed afresh for the selected settings, never supplied as invented sample
data. Missing files leave the examples absent. The diagrams use concise labels;
the full examples are kept for the dedicated explanation page and checks.
The Data flowchart shows training and test readings. The test-answer file
remains in the file details table, where its use in checking predictions is
explained, rather than adding an evaluation branch to the preparation flow.

The shape of what a step hands on, the size of the table of numbers passed to
the next step, is written on the arrow under the step, and so reads as the next
step's input: each step is seen as what comes in above it, what is done to it
and what goes out below it. In the graph layers' flowchart, for one, the
arrow into "Rescale the graph" carries `(14, 14)`, the table of which sensors
are joined, and the arrow out of it carries the same shape rescaled. Each shape
is written once: the words that open a flowchart put theirs on the arrow under
them, and the words that close it show their own only when it differs from the
one on the arrow above them. Preprocessing also names the quantities at its
entrance and explains `B` at its exit.
Steps that work side by side on the same input, such as the three files of a
sub-dataset, share one row, and each shows its own shape inside its box, since
no one arrow could carry three. In a shape, `B` is the number of windows the
network is handed at a time. Before, the shape stood at the right edge of each
step's box, in a column of its own, beside the step's name, where it read as a
property of the step rather than as the thing passed from one step to the next.

| Box | Its steps |
|---|---|
| 1 Data | the training engines, the test engines and the file of their remaining life, side by side; a remaining life for every row; and the 14 of 21 sensors kept, in Table 3's order |
| 2 Preprocessing | capping the labels, smoothing, holding engines back, finding the operating conditions, rescaling and cutting windows; the groups appear immediately after holding engines back and persist through the remaining steps |
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

**Choosing a procedure and applying its split happen in different places.**
Training owns the procedure control because validation also decides which
saved model to keep. Preprocessing applies its engine split before fitting
the rescaling statistics. Each arrow keeps its compact summary of the training
file, with an **i** beside it for the full breakdown. That card shows training,
validation and test row counts immediately after **Hold engines back**, carries
those groups through rescaling, and shows their window counts after **Cut into
windows**. It opens on hover, keyboard focus or click. The diagram has no
additional two-set or three-set operation. The stored
windows are collected by engine later in the pipeline, using the same
membership chosen earlier. This is a conceptual view of group membership,
not a claim that the program creates separate table objects at that point.
The real-data diagram tests check every group's row and window counts against
the pipeline, including the procedure without validation.

Every displayed output shape in Data, Preprocessing, the temporal stage and
the graph layers has the same **i** beside it. Its card names the output and
explains the dimensions in order, including the meaning of `B`. This includes
the sensor graph on the side branch and the separate window-summary outputs.
For recurrent networks, the card distinguishes a separate history for each
sensor from the combined window–sensor dimension used by a shared network.
After flattening, the card describes one list per window; after prediction,
it describes one value per window. The compact shape stays on the arrow.
The renderer checks cover every temporal option, both recurrent sharing
choices, the graph branch and the data lanes, using the real stage descriptions.

Nothing in a flowchart is typed in by hand as a number. `stage_detail()` counts
the rows and engines from the NASA files, runs the real preprocessing for the
windows and the operating conditions, and builds the real network for the
shapes and the learned numbers, reading them from `layer_summary()`. Of a
preprocessing run it keeps only those counts, the sensor graph, three example
rows with their changed readings, and three window ranges, never the full
tables or windows, for the last sixteen groups of settings. It runs one at a time, so
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

Inside the diagram and settings scroll areas, an i's card
is positioned relative to the browser window rather than to the i: the
scroll area's boundary would otherwise cut off whatever passed its edges. The
card opens under the i, or above it when the window has no room below,
follows the i when its area or the page scrolls, and closes once the i has
scrolled out of view.

The pipeline comes first in the page's order for keyboard and screen reader
navigation, although beside the form it is drawn on the right. The same
order holds when it stands above the form. In each tab strip, only the
selected tab is a Tab-key stop. Arrow keys, Home and End follow the visible
order. With Alt or Ctrl held they are left to the browser, where Alt with
an arrow goes back or forward.

"Pipeline", or Escape pressed inside the pipeline or with nothing focused (no
control selected for the keyboard), leads back to the overview, with the
keyboard's focus on the box that was open, or on the + that would put it back
when that box has since been taken out. Escape first closes an i's words, if
any are showing; pressed in the form, it leaves the flowchart alone. A value
that links to one of the graph layer widths puts the focus on that layer's own
number field. When a new answer replaces a flowchart's steps, the focus stays
with the button it was on, or moves to the one that took its place: after the
sensor graph's cross, to the + that puts it back.

The detail heading is centred across the whole panel, independently of the
back button at its upper left. The preprocessing response retains the real
example rows, their source file and their remaining life before the ceiling
is applied. These records support the worked-example checks; they no longer
appear as an opening table in the diagram. The checks compare remaining life
before and after the ceiling against the file, including a changed ceiling.
Sensor readings are not capped by this operation.

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
  The rows are `ESTGNN.layer_summary()` as it comes, one per row of Table 4.
  The temporal stage has no row of its own there. What it hands to the first
  graph layer — for the paper's stage and for the w/o LSTM network, one number
  per sensor for each of the `B` windows in a batch — is what Table 4's Input
  row lists, so `layer_summary()` counts the stage's learned numbers on that
  row and, when the stage is the LSTM or the GRU, names it; the page shows
  that as "Input from LSTM" (`models/README.md` explains the row). Until
  2026-10-03 the summary listed the stage as a row of its own and this page
  joined the two rows itself; the rows it shows did not change. Settings the
  network refuses — thirty raw cycles handed straight to a graph layer, for
  one — are refused with the network's own message.
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
own page lists in full, and `"design"` for the two above. The form offers no
graph drawn at random, but a stored run trained on one from the command line
(`--graph random` or `--graph rewired`, assumption A-41) gets a caution saying
whether every seed drew its own graph or one graph served them all, and "Set
these settings up again" on it keeps the form's own graph and says so. The
same holds for any recorded choice the form does not offer. One graph serves
every seed only when the same engines are held back for all of them: such a
graph is drawn from the seed the data is prepared under, and with no engine
held back every seed is prepared under its own. `scripts/train.py` therefore
refuses "Same held-back engines for every seed" with these graphs when nothing
is held back. The playground's check of a request, `validate_settings`, refuses
the same combination in the same terms (`fixed_graph_refusal` in
`settings.py`), so that it would be stopped before a job starts should the form
ever offer these graphs; and `caveats()` gives that refusal, not "one graph
served them all", for those settings. Until 2026-10-03 the caution said one
graph served every seed whenever the box was ticked, with engines held back or
not; a review of the change found it the same day. Most cautions are
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
| anything | the 41 most correlated pairs | none: the paper describes no such graph for its network and never says what its w/o Physics prior row used, and this project sets only every sensor joined beside that row |
| anything | 41 pairs picked at random, or the paper's graph rewired | none: the paper describes no such graph and never says what its w/o Physics prior row used, and this project sets only every sensor joined beside that row |
| anything else | the paper's or every sensor joined | none: the paper has no row for the stage |
| not recorded, in an older stored run | the paper's or every sensor joined | none: no row can be chosen |

(Corrected 2026-10-03: the row for the 41 most correlated pairs said "the
paper builds no such graph". It describes none for its network, but it never
says what replaced its graph in the w/o Physics prior row, so whether that row
used one cannot be told from it. Found by a review that flagged the sentence as
unsupported.)

The list of runs gives that row's root mean squared error (RMSE: the square
root of the average squared difference between the predicted remaining life
and the test label, the remaining life capped at 125 cycles (assumption A-02),
in cycles; corrected 2026-10-03 from "the true remaining life", found by a
review of the capped-label wording) in its Paper column, and a run's own
page names the row and sets its figures beside the run's, or gives the reason
from the table's last column. The design tab names no row: the pipeline already
shows which parts are in, and the paper's number belongs beside a result, not
beside settings that are still being chosen.

### Where each default comes from

Each settings section has an i beside its title. Individual controls and
choices have no separate icons. Pointing at the title's i or reaching it with
the keyboard shows a small card in two parts. First, in plain words, what the
section's settings are for: the problem they answer and what changing them does to the run,
written for a reader who has never met the method, so that they can tell
whether to touch it. Then, under a thin line, a tag saying which of four
things set the default, because that is the distinction the thesis depends
on, and a few words on why. Sections without a source, such as **Device**,
have only the explanation. The author asked for the order on
2026-09-26: the card beside "Sensor scaling" had said only where the default
came from, and never what sensor scaling is for. Choosing the i keeps the card
open, and only then can the keyboard reach a link in it; the card closes once
the focus moves on, or with Escape. The form itself shows only names, so it
reads as a list of choices rather than a list of citations.

The plain words are held in `FIELDS` in `settings.py`, as each setting's
`about`, and as the option's `about` for descriptions of particular choices.
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

The follow-up on 2026-09-27 also corrected the description of learning rate:
it scales an update, rather than specifying its complete size. The training
algorithm uses how error changes with each learned number and the history
of those measurements too, as the [Adam documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
describes. Repeated seeds measure variation between runs, rather than the
uncertainty of a particular engine's prediction. The graph-route card now
states that its shares start at the temporal stage's output; it no longer
claims the network is never evaluated again. The smoothing example's
arithmetic uses the weight selected on screen, checked on real FD001 rows
at both the default weight and a changed one in `tests/test_examples.py`.

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
log. `_READ_MORE` in `settings.py` holds the settings and option destinations.
Diagram cards use `_STEP_DOCUMENTS`, which pairs each displayed operation name
with its taught page. `_STEP_HEADINGS` adds the stage number where the page
uses one; the operation's name stays the same. Each destination explains
**Input**, **Function**, **Output** and a worked **Example** from real FD001
readings. For example, **Calculate training labels** opens that section of
**Data**, where the subtraction from an engine's final recorded cycle is shown
before the separate ceiling operation. Temporal alternatives each have their
own section, so keeping a reading cannot accidentally open the explanation of
a calculated memory state merely because both use the same picture.

**Correction, 2026-09-29.** Diagram cards previously reused broad topic links,
and a card connected to a setting could replace its operation's explanation
and link with that setting's help. The diagram now keeps the operation's own
text and tutorial link; its clickable values still open the related settings.
The number of values produced by the temporal conversion remains in the
diagram's detail, while its name, **Turn into numbers per sensor**, stays the
same as its documentation heading when that setting changes. Tests render the
actual cards and check the links, headings and tutorial labels across the
available stages and changed settings.

The Data, Preprocessing, Temporal algorithm, GNN, Architecture and training,
XAI, Dynamic SHAP and GNN-LRP pages integrate each example into its numbered
stage: intuition and the figure first, then a Process with complete sample
input and output tables. Input, Function and Output teach the information and
the general transformation, without naming the implementation. An expandable
**Example in Python** follows the conceptual calculation and precedes the
output. Its code blocks run in page order, and the example
tests check every table cell against those calculations on FD001. Older section
bookmarks are redirected by `DATA_MOVED_SECTIONS` in `app.js`, so merging the
earlier duplicate sections does not leave their links pointing nowhere.
The temporal tutorial uses four real FD001 cycles and one value per memory
list, so all readings, gates and memory updates can be shown. Its recurrent
parameters come from a fixed initialization and are not trained. The architecture
tutorial likewise uses an untrained teaching network with graph widths of
3, 2 and 1 and a fully connected width of 3. These teaching widths keep every
input, multiplier and output visible; they do not change the project's defaults.
The saved-model walkthrough remains on the Worked examples page, separate from
these calculations. Shared and separate recurrent networks have distinct
examples, while duplicate last-reading and average sections now have one
destination each.
The XAI tutorials share one small untrained network and real FD001 test-engine
readings. The overview compares explanation units; Dynamic SHAP follows every
sensor combination and its contribution; GNN-LRP follows every route and the
checks made by changing which sensors remain connected. Each page shows all
values in its reduced example. These arithmetic demonstrations are separate
from conclusions about trained models. The earlier XAI overview's section
links go directly to their current subpage sections through `XAI_MOVED_SECTIONS`.
Document navigation waits for its figures before aligning the requested heading;
otherwise an image arriving later could push the linked section off screen.

An introduction marks the start of each page's algorithm, whose steps begin
at 1. The step number appears in a badge in the article; earlier numbered
addresses redirect to their new sections. Wider gaps and dividers separate
stages and their operations. Process has a distinct heading, followed by
Input, Function and Output headings. These details remain in the article;
the right-hand contents and the folded contents on smaller screens show only
sections and subsections. Plain step numbers, regular-weight text and an
unfilled active link keep the contents visually secondary to the site navigation.
These treatments use the site's existing colours and fonts in both themes.

A card is drawn over everything around it: each grid of settings is
drawn by the browser in a layer of its own, since the layout's width rules
measure it, so while a card shows, the setting, the grid and the group that
hold it are lifted above their neighbours; before, a switch and an i of the
next group showed through it.

Design a run keeps the source notes and tutorial links in the section-title
cards and the corresponding diagram operations. Its radio choices no longer
open individual cards when reached with the arrow keys. In **Explain
predictions**, options still have their own cards: the radio button opens its
description for keyboard users, and the icon is not an extra Tab stop.

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
the line is hidden while the job's own row is on screen in Run history. Run history has
no separate list of jobs. A job is a run or an explanation that has not
finished, so every job that is waiting, running, failed or stopped is a row of
its own at the top of the Run history tab it belongs to, training runs or
explanations, with its state, its bar and what is happening now, its last
printed lines when it failed, and Stop or Remove; a finished job is its stored
run or explanation, listed below. The Run history tab's badge counts the jobs
running or waiting. While Run history is open its job rows follow each poll: they
are drawn again only when a job comes, goes or changes state, and otherwise
only their bars and lines move, so a filter being used keeps its focus. The
author asked for the two lists to become one on 2026-09-26, having found no
difference to tell them apart by. Run history holds
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

An opened run shows **Settings used for this run** before its charts. Three
groups show data preparation, temporal and graph layers, and training. They
read the saved preprocessing configuration, saved model configuration and
each seed's training configuration. Training values shared by all seeds appear
once; differing values name each seed. A seed without a training file uses
the recorded summary where available, visibly labelled as such. Missing
settings say **Not recorded**, never the current form's defaults. Zero,
false and null retain their distinct meanings. A saved but unused recurrent
setting remains visible, with a note when the temporal route does not use it.

The run record shows the stored split policy, stopping policy, reported
weights, code version and completion time. The split is shown as it happened:
a run that held no engine back is shown as having no split, and when its
record says otherwise — every such run stored before 2026-10-03 recorded
`varies_with_seed` (audit item D-4) — the recorded value is shown beside it.
Comparing two runs reads the split by the same rule, so two runs that both
held nothing back never differ in it. The ASUE card adds how many
engines the run predicted early, on average over its seeds, when the run
recorded it (since 2026-10-03); with none, ASUE has no value (assumption
A-39). An expandable source record keeps
every saved field accessible, including notes, additional configuration keys,
the launch command and software environment when the run recorded them.
The API's `recorded_summary` and `recorded_model` preserve the saved mappings;
`per_seed[seed].training` holds that seed's complete training settings.
Grouped study runs retain their full path below `reports/`, so opening or
revisiting two runs named `reference` does not mix up their records.

**How to interpret this run** gives each comparison note a descriptive title
and a separate background link. Its columns use the available width, becoming
one column in a narrow panel. The notes qualify the results; they do not
stand in for the actual recorded settings. The test-label note appears only
when capping was recorded, and an unknown weight-selection policy is labelled
as unknown rather than being described as the final pass.

**Correction, 2026-10-01.** Run history previously showed only a few settings
in badges and never exposed the complete per-seed training files. Inspecting
those saved files showed that information such as the optimizer, device and
stopping threshold was available but missing from the UI. The settings groups
now show it; tests compare the API response with the files written by a real
FD001 training run and check missing, disabled and differing settings in the UI.

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
Explain predictions tab.

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
in order of unit number, or ordered by their label instead, which shows how the
error changes along the life. The label is each test engine's remaining life
from NASA's answer file, capped at the run's ceiling, 125 cycles by default,
when the run capped its test labels (assumption A-02); the chart then calls it
the "remaining life capped at 125", calls it the true remaining life only for
a run that left its test labels uncapped, and calls it the test label for a run
that recorded neither. In FD001, 11 of the 100 test engines have more than 125
cycles left, so a capped run draws those 11 at 125.

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

Explain predictions opens the run selected with "Explain this run" in Run
history, or the newest run with a saved network on first entry. The Trained
run selector lists only runs with a saved network. Changing it reloads the
available seeds and stored explanations, clears the previously displayed
explanation and engine choice, and keeps keyboard focus on the selector.
View run details opens that run in Run history. If no usable run exists, the
page points to Design a run.

The selected method starts `scripts/explain.py`: exact Shapley values,
occlusion, or walks through the sensor graph. `explain/README.md` explains
these methods.

Measured
on this machine's processor with an untrained network, one window of exact
Shapley values takes 0.07 s for the w/o LSTM network and 0.50 s with the LSTM
stage, so an engine's whole life is minutes, not hours.

A stored explanation holds one row per window: the label, which is the
remaining life capped at 125 cycles (assumption A-01; on a test engine, only
when the run caps its test labels, as the default does, assumption A-02), the
predicted remaining life, one value per sensor in cycles, and, since
2026-09-16, what
each sensor was reading at the window's last cycle, because the paper's Fig. 9
colours every value by whether the sensor read high or low and nothing else in
the file could say which. That last number is the reading **after** the noise
filter and the rescaling, not the raw one: it is what the model was given, and
it is the only version of it the stored windows hold. An explanation stored
before the readings were kept is coloured by the window's place in the sequence
instead, and its legend says so. (Corrected 2026-10-03: this called the first
column the true remaining life. It is the label the network is scored against,
taken from the dataset's targets, and where it is capped it is below the true
remaining life of any window with more than 125 cycles left. Found by searching for the wording that
a review had corrected elsewhere.)

An explanation names the paper's figure when a chart follows one, as in "As
the paper's Fig. 9, left." A method added by this project states that origin
once in its introduction instead of repeating "Not in the paper" on every
chart. The distinction remains visible without making each caption carry the
same warning (working rule 5).

The explanation charts share one row of tabs and one full-width viewing area.
Only the chosen chart is visible, so the engine, remaining-life curve, sensor
ranking and later views sit beside one another as choices instead of forming a
long column down the page. The engine opens first. Left and right arrow keys
move between the tabs; on a narrow screen the tab row scrolls sideways while
the chart keeps the page's available width.

One shared control sits above those tabs, outside the Engine card. For an
engine's history it is named **Engine cycle**: the number is the last cycle
of the readings used for that prediction, not a numbered block called a
“window”. For comparisons across test engines it is named **Test engine**,
and its menu shows the actual engine identifiers. Moving between test engines
does not represent time passing. The arrow buttons move to the previous or
next saved entry; the menu jumps directly to a cycle or engine. The slider
shows the endpoints and position among the saved entries, alongside the label
and the predicted remaining life in cycles. The label is the remaining life
capped at the ceiling for the cycles of one engine, which are training-file
windows (assumption A-01), and for test engines when the run capped their
labels too (assumption A-02); the control then calls it the capped remaining
life and names the ceiling, since wherever it reads the ceiling more cycles may
truly remain. Only for a run that left its test labels uncapped does it say
true remaining life, and for a run that recorded neither it says test label.

(Corrected 2026-10-03: the control, the Fig. 7 and Fig. 8 charts, the history
caption and the slider's spoken value called the capped label the true or
actual remaining life, and this section said the control showed actual
remaining life. For held-back engine 5 of seed 42 of the default configuration
the capped label is below the life truly left in 114 of its 240 windows. Found
by a review of the wording around the capped label.)

Dragging updates the visible engine or graph, the selected-cycle marker on
the remaining-life chart, the history readouts, and the selected test engine's
score breakdown without waiting for release. Only the changing content is
replaced, at most once per browser drawing frame: the slider itself, selected
sensor, graph layout and expanded calculation details remain in place. On
switching chart tabs the newly visible chart uses the current selection.
Overview charts still include all saved cycles or engines; the selector does
not filter their data. These controls read stored results, so moving between
them does not run the model again or interpolate unrecorded cycles.

The guide above a walk explanation names the numbers beneath the sensor labels
in the Engine tab, explains their signs and units, and tells the reader how to
select a sensor. Its paragraphs use the full available width. The calculation
and its checks sit under “How this explanation is calculated and checked”. That
section includes the selected engine and cycle, the sum of its saved sensor
contributions, the contribution from added constants, the internal score and
the prediction in cycles. It updates with the slider. The distinction between
explaining a calculation and diagnosing engine damage stays visible even when
the detailed section is closed.

**A walk explanation shows only the GNN-LRP paper's own views**, since
2026-10-01: the Engine tab, whose numbers are the walks added up by their
starting sensor, which that paper says is how its method gives sensor-level
explanations; the score split into its parts; the connections, the walks added
up per connection, which that paper also allows; the largest walks; and
**Authors' check**, the node-flipping curves that paper uses to test its own
explanations (its Section 4.1), for the selected cycle or engine, with the area
under each curve averaged over every saved one. Views borrowed from the Shapley
explanation or from the main paper — the remaining-life chart drawn after the
main paper's Fig. 7, the sensors ranked by average size, the per-sensor panels
over time and the heat map across engines — are not offered for walks, and the
paragraph comparing walks with Shapley values was removed. The calculation
section names the rule's setting γ the explanation was made with; one made at
γ 0, before the authors' own setting became the default, says that it is their
simpler baseline, GNN-GI, and has no stored check.

**The rule's setting can be chosen**, since 2026-10-02. A walk explanation
divides each graph layer's numbers among the sensors that sent them, and γ
(gamma) is the number that decides how much extra weight a positive
contribution gets in that division before the layer's shares are rescaled back
to the total it computed. Negative shares shrink; the total stays the same,
and so does the prediction. `explain/README.md` works one node through it by
hand, under "The setting gamma: the authors' own". When walks are the method,
a row named "How walk shares are divided (γ)" appears under Method, with four
options. Each has a label, which the form shows, and a short name, which is
what the page sends to the server, what `scripts/explain.py --gamma` takes and
what a stored explanation's folder name carries:

| Option | Short name | γ at the three graph layers | Where it comes from |
|---|---|---|---|
| The authors' setting, the default | `authors` | 0, 1.5, 0 | the default of the method's published code, 3 falling to 0, with the first replaced by 0 (assumption A-29) |
| No extra weight | `gnn-gi` | 0, 0, 0 | the method's simpler baseline, which its authors call GNN-GI |
| Halving extra weight | `falling` | 0, 0.5, 0.25 | the halving the GNN-LRP paper uses for VGG-16 (its Section 5.3), started here at the second graph layer; the numbers for this network are ours |
| Equal extra weight after the first layer | `flat` | 0, 1, 1 | ours |

The last two are the settings that the log's walk measurements set beside the
authors' to see how far a ranking of connections moves with γ, in the text
under their table of stored runs (assumption A-29). Explaining the same
windows under two options shows a reader what that comparison measures, one
window at a time.

The page sends the short name, never numbers, and the server refuses any name
it does not list (`validate_explain` in `runs.py`). `scripts/explain.py` turns
the name into one γ per graph layer, and every name gives the first graph
layer 0: that layer receives rescaled readings, which are negative below their
training average, and the rule's derivation assumes the numbers entering a
layer never are (assumption A-29). The row's i gives this reason once, rather
than each option repeating it; of the options, only the authors' names the
first layer, because its label would otherwise claim the authors' 3 there.
For a network with another number of graph layers, halving goes on halving and
the equal option stays at 1.
`test_no_positive_gamma_reaches_the_first_graph_layer` in
`tests/test_playground.py` records every γ the rule is run with, at the one
place it is applied, under all four options, and finds the first always 0.
`test_a_gamma_at_the_first_layer_would_change_the_walks_of_that_network` shows
that on the network it uses, which hands the graph three numbers per sensor, a
3 at the first layer would move the walks, so the guard protects something
that matters there. With one number per sensor it would cancel.

The stored explanation records the short name (`gamma_setting` in its
`summary.json`) and its numbers (`gamma`). Only when the option is not the
authors' does the explanation's row in the tables of stored explanations and
its job's name in the queue add the option's label, and its folder name the
short name (`_gamma-falling`, for example); the calculation section then gives
its numbers, not its name again. One made with no extra weight, from the form
or with `--gamma gnn-gi`, says that setting was chosen when it was made and
has the stored check, unlike the γ 0 explanations from before 2026-10-01.
**Authors' check** always compares the authors' setting with their baseline;
an option that is neither is checked beside them, drawn in gold and named as
the form names it. Not in orange, because orange already means a negative
share in this view: the guide above the explanation and the legend of the
connections say so. Not in the accent blue either, because it sits close to
the baseline's blue-green. Gold means no sign anywhere here; it names the
model-added part in the Score parts tab, whose own legend says so. Measured as
the straight-line distance between two colours in the CIE Lab colour space,
which is built so that equal distances look roughly equally different, the
accent blue is 32 from the baseline's colour in light mode and 23 in dark;
gold is 78 and 83 from it, and 69 and 63 from the ink colour that draws the
authors' setting. Against the figure's background a gold line has a contrast of 4.9 to
1 in light mode and 9.2 in dark, above the 3 to 1 a drawn line needs
(`test_a_chosen_setting_s_check_is_drawn_in_a_colour_far_from_the_baseline`
recomputes all of these from the colours `assets/app.css` defines).

**Wording correction, 2026-10-02.** From the commit that made the authors'
setting the default (f1adb1e, 2026-10-02) until the setting became
selectable, the paragraph headed “The rule.” in the calculation section said
that a walk explanation counts positive contributions more heavily, as the
method's authors do. That was false for the explanations made at γ 0 between
2026-09-26 and 2026-09-28, which are still listed, and the sentence after it,
which named their setting, contradicted it. It was found on 2026-10-02 while
the setting was being made selectable, and confirmed by the review of that
change. The paragraph now says that γ decides how much more heavily positive
contributions count, and the sentence after it gives the setting used.

**Wording correction, 2026-09-27.** The previous introduction began “Each value”
without saying which values it meant, and mixed chart instructions with
calculation checks. The revised text points to the visible sensor labels and
introduces the internal score before describing how the explanation divides it.

For a walk explanation over the test engines, engine number is not drawn as a
horizontal sequence: it is an identifier, so a line or left-to-right pattern
would have no meaning. Instead, one row per part of the score shows every
engine, the middle half of their values and their median, the middle value after
ordering them. Beneath it, one chosen engine is a waterfall: the contribution
from the added constants starts at zero, the walks that stay within a sensor's
group start from that subtotal, and the walks that move between groups start
from the next subtotal. The final outlined bar is the saved score, so the
adding-up claim is visible rather than left to the reader's arithmetic. These
are signed totals. A small bar can mean little travelled through the graph, or
that large positive and negative walks cancelled; the largest-walk table names
that distinction separately.

**Label correction, 2026-09-28.** The author could not interpret “constants”,
“stays” and “moves”. These were internal column names presented as explanations.
The chart now says **model-added part**, **routes within one group** and
**routes between groups**. Its caption first tells the reader what comparison
the chart supports: which part has the largest net effect, which parts reinforce
one another and which cancel. The detailed definitions follow the chart, so
they remain available without delaying the first view of the data.
The model-added part comes from learned additions inside the network, also
called biases. The additions are fixed after training, but their contribution
can vary because different inputs make different calculations pass them on.
It is not a reference-engine prediction or a constant sensor channel.
The groups collect sensor positions with matching model connections, not
physical engine components. A route between groups leaves its starting group
at least once, even if it later returns. The saved numeric columns keep their
original names; only their presentation changes. The selected-engine chart
shows the saved addition and the resulting prediction, explicitly distinguishing
score units from cycles.

### Locating the explanations on an engine

The Data documentation calls `ThesisEngineView.mountGuide`, which uses the
same cutaway and variable-location catalogue as Explain predictions. It shows
no invented explanation values. Selecting a marker identifies the component
and explains the measurement; Enter and Space provide the same action.

### Help cards and the separate worked example

Settings and diagram steps with the same title use the same `sectionHelp`
function: the explanation, source note and tutorial link are identical. For
example, both **Rescale each sensor** cards explain the mean-and-spread and
training-range choices. The selected choice remains visible under the diagram
title. The tutorial addresses come from the same operation-name catalogue, so
sharing a card cannot replace its named tutorial with a general settings page.
Related titles share the explanation too: **Graph layer 1**, **Graph layer 2**
and **Graph layer 3** use the **Graph layers** help, while each links to its own
tutorial section. Their selected widths and input/output dimensions remain on
the diagram. The fixed temporal operations (last reading, average, three
summaries or every cycle) share the corresponding option's explanation and
tutorial link. Matching depends on the setting an operation controls as well
as its title, so a singular name or an alternative temporal stage does not
silently fall back to a second version of the text. Data operations without a
settings counterpart keep their own explanation in the same card layout.
Clicking a section heading opens its diagram operation; clicking that operation
returns to the matching setting. Help cards explain the operation once, cite
its source and link to the relevant reading. They do not repeat the diagram
as an input, operation and output table. The longer calculation sequence lives
at `docs/learn/08_worked_examples.md`, reached through **Follow one real
example** in the menu.

`worked_example.py` recomputes one training-engine window through one saved
network. Its displayed run name identifies the producing commit and timestamp;
the selected seed is named too. It never substitutes generated data or an
untrained network when that run is unavailable. This illustrates arithmetic,
not a single-seed accuracy claim. Tests compare the separated operations with
the model's normal prediction. The raw and capped RUL curves use recorded
cycles starting at one and one common vertical scale; the old dashed line
incorrectly started at the ceiling and was not an uncapped curve.

### The explanation overlay

Explain predictions opens with an engine cutaway drawn by `engine-view.js`.
Its layout follows the engine modules in Figures 1 and 2 of
[NASA's dataset paper](../../../../docs/references/saxena_2008_damage_propagation_modeling.pdf).
The same paper's Table 2 gives the variables' meanings. The section anchors
are documented; exact probe positions, blade counts, dimensions and detailed
cooling routes are not. Our drawing choices are recorded as the engine-overlay
assumption (A-31). NASA's
[EngineSim interactive](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/enginesimu/)
provided a precedent for selecting a component on a cutaway; its code and artwork
are not embedded in this page.

The engine panel answers two different questions. Dashed leaders answer
"where does this variable belong?"; graph connections answer "which other
variables can it exchange information with in this model?" The two types of
line must never be read as the same relationship. Circles locate temperature
and pressure measurement regions. Rounded boxes identify shaft quantities,
calculated quantities such as corrected speed, and system quantities such as
cooling flow. Selecting a marker explains that distinction and highlights its
component and graph neighbours. Keyboard users can select markers with Enter
or Space. The surrounding chart keeps the page's full width, but the drawing
itself stops at 1,000 pixels and stays centred so the selected sensor's
explanation remains close below it. A narrow screen scrolls the drawing
horizontally while the controls and explanation wrap to the available width.

The engine and graph views show the same stored values, with the same colour
and width scales. Positive shares are blue and negative shares orange; the
strongest absolute value across all explained windows sets the marker scale.
The chosen cycle changes with the shared control above the chart tabs. For test
scope, this picks one test engine's last window; the separate average-connection view remains
under a disclosure. No model is retrained or sensor data generated to draw it.

For graph walks, connection columns decide which edges exist, even when their
shares are zero; both directions are added before setting a line's width.
The total is over all graph layers, and opposing shares can cancel. For
Shapley and occlusion, `_explanation_edges` reads the selected seed's saved
matrix of connections, with `weights_only=True`; if the file or PyTorch is
unavailable, the panel says why it shows no connections. It never substitutes
the paper's graph for a correlation graph. Equal line widths for these methods
describe connectivity only, not importance.

The saved-walk selector shows one position per graph layer. A slash denotes
a group whose members the model cannot distinguish. All compatible connections
are highlighted; choosing one group member would invent a distinction missing
from the saved result. The walk's displayed share is its saved average across
all explained windows, explicitly separate from the chosen window's node and
connection values. Repeated groups remain visible in the layer sequence even
when the drawing would place them on top of themselves. A strong share is an
explanation of this model's prediction, not a diagnosis of the component.

The real stored examples in `tests/test_examples.py` exercise both layouts,
signed values, fixed scales, zero-share connections and grouped walk paths.
The saved-matrix example in `tests/test_examples_torch.py` checks that a
correlation graph comes from the selected network rather than the paper.

**Beside the paper's Fig. 10.** For an engine explained with Shapley values,
the page redraws the paper's Fig. 10 from the numbers of its Table 12, above
the stored engine at the same graph numbers, where graph `g` is the engine's
`g`-th window, ending at its cycle `g` + 29 (our reading; the paper never says
which cycle a graph is, assumption A-35), and counts **how often a sensor joins the red four**, the four
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

## Things that will catch you out

**A test that swaps out a function has to swap it in the file that calls it.**
A test can replace a function for as long as the test runs; pytest's
`monkeypatch.setattr(place, "name", replacement)` does that by rebinding
`name` in `place`. A Python function looks up the names it uses in the file
where it was defined, never in `__init__.py`. `stage_detail` is defined in
`design.py` and calls `_files_present` there, so the test that makes it behave
as though the NASA files were missing replaces `design._files_present`
(`tests/test_examples.py`). That is also why `__init__.py` does not offer
`_files_present`: if it did, replacing `playground._files_present` would rebind
a second name for the same function, one that nothing reads, and the test
would quietly use the real files. Checked on a two-file copy of this layout
when the file was split, replacing the name in each place in turn:

```
patched the package : read the files
patched the module  : show the download link
```

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

---

## What each name means

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

Every finished run anywhere below `reports/`, newest first, each with the
published row it may be set beside and the cautions that apply.  An ordinary
run is named by its folder; a run grouped inside a sensitivity study is named
by its whole path below `reports/`, such as
`sensitivity_.../runs/FD001/reference`, so two sub-datasets' `reference`
folders remain distinct.  The other two functions return one run in full,
with every seed's per-pass record and test predictions and every stored
explanation, or two runs paired seed by seed, with the settings that differ
and a paired *t* and *p* for each measure.
