# `graph/` — the sensor graph, one notebook per graph

The network in notebook 02 reads 14 sensors, and its three graph layers mix
each sensor's numbers with those of the sensors it is joined to. Which sensors
are joined is decided before training, by the **sensor graph**: a 14-by-14
table holding a 1 where two sensors are joined and a 0 where they are not.
Training never changes it. The paper designs its graph by hand, from
engineering knowledge of the engine (its Section 2.3.1 and Table 3).

A graph cannot produce a result on its own: it needs a network trained on it.
So each notebook here shows one graph two ways — the graph itself, rebuilt
from the real data exactly as a trained network received it, and the stored
five-seed result of the networks trained on it — and the last notebook sets
the three results side by side.

Read `notebooks/README.md` first, for how every notebook is laid out, and
notebook 02's step 9 before any of these, for what a graph layer is.

## The notebooks

| Notebook | The graph | Where it comes from |
|---|---|---|
| `01_physics_fd001.ipynb` | the paper's own, 41 edges | copied from the paper's Table 3 |
| `02_complete_fd001.ipynb` | every sensor joined to every other, 91 edges | this project's reading of the paper's "without physics prior" experiment (assumption A-19) |
| `03_correlation_fd001.ipynb` | the 41 pairs of sensors whose readings move together most strongly | measured on the training engines (assumption A-27) |
| `04_comparison_fd001.ipynb` | all three | their stored results side by side, paired by seed, without the recurrent stage and, from another commit, with it |

Read 01 first. 02 and 03 each take its steps on another graph and can be read
in either order. 04 comes last; until 2026-10-02 it was section 5 of
notebook 02.

## The words used here

- A **graph** is a set of points, called **nodes**, and connections between
  pairs of them, called **edges**. Here a node is a sensor, and there are 14.
- The **adjacency table** — "adjacency matrix" in most texts — writes a graph
  down as a 14-by-14 table `A`, with `A[i, j] = 1` when sensors `i` and `j` are
  joined. If `i` is joined to `j`, `j` is joined to `i`, so the table equals its
  own mirror image across the diagonal; no sensor is joined to itself in it, so
  the diagonal is zero. The sensors a row marks with a 1 are that sensor's
  **neighbours**, and how many there are is its **degree**.
- The **rescaled table** `Â` ("A hat") is what the network actually multiplies
  by. It adds 1 down the diagonal, so that each sensor also counts itself
  among its neighbours — a self-loop, in the usual word, meaning an edge from a
  node to itself — and then divides each row by its own total, so the row adds
  up to 1 and multiplying by it takes an average. The paper says its table is
  normalised and includes those self-loops, but never prints the formula;
  dividing each row by its total is this project's choice (assumption A-21).
- A **graph layer** computes `ReLU(Â H W)`: the paper's Equations 8 to 10,
  which write the rescaled table `Ã` where these notebooks, like notebook 02,
  write `Â`. `H` holds
  what each sensor carries in, one row per sensor. `Â H` replaces each
  sensor's row with the average over itself and its neighbours: the
  **mixing**. `W` is a small table of learned numbers, the same for all 14
  sensors, that turns each averaged row into a new row of numbers: the
  **transforming**. ReLU replaces every negative number with zero.
- **Reach.** A sensor's numbers after one layer are built from its neighbours'
  numbers before it. After two layers they are built from numbers that
  already held the neighbours' neighbours', so two layers reach two edges, and
  three layers reach three. Two sensors more than three edges apart cannot
  affect each other anywhere in the graph layers. The dense layers after
  them read all 14 sensors at once, so both can still affect the prediction.
- **Identical rows.** When two rows of `Â` are the same, those two sensors
  take the same average of the same numbers, so they hold identical numbers
  after the first graph layer and after every later one. The network can no
  longer tell them apart.

## What every notebook here shares, and why

**One configuration, with only the graph changed.** Every run read here is
`config/fd001.yaml` with its `graph` setting swapped and nothing else: the
readings smoothed, then rescaled with averages and spreads measured on the
training engines alone; windows of 30 cycles, one cycle apart; labels capped
at 125, the test labels included; each row of the graph divided by its total;
20 % of the training engines held back, with the parameters kept from the pass
over the training data that did best on them; each sensor carrying only its
last reading into the graph layers, so no recurrent stage; the layer sizes of
the paper's Table 4; and the training settings of its Table 6, up to 200
passes over the training data in batches of 32 with the learning rate set to
fall from 0.001 to 0.0001, the fall placed at the halfway pass (assumption
A-17). No run read here reached the halfway pass — the longest made 75 — so
every pass used 0.001, and the comparison notebook checks that. Each setup
cell asks for exactly this, and the comparison notebook reads every setting
the three runs saved and checks that the graph is the only one that differs.

Having no recurrent stage decides which published figure a result here may be
set beside: the paper's "ESTGNN w/o LSTM" row, 14.93 on FD001 in its Table 11,
and never its headline 11.78.

The one exception is step 6 of the comparison notebook, which reads the same
three graphs trained with the paper's recurrent stage. The only stored set of
those is inside a study that changed one setting at a time around the paper's
network (finding F-19), at another commit, so that step pairs those three
runs with one another, never with the runs without the stage.

**Stored runs, never training.** None of these notebooks trains a network. A
network trained in a notebook is one seed, and records nothing about the code
that trained it (working rules 3 and 3a). A stored run of `scripts/train.py`
trains five seeds, saves each seed's network, and records its commit. The
notebooks read the runs through `find_runs` and `load_seed`, in
`src/thesis_gnn_rul_xai/training/stored.py`.

**The same three runs in every notebook, from one recorded commit.** Each
notebook's setup cell takes the newest commit at which all three graphs have a
stored five-seed run of this configuration that saved its networks, and the
newest run of each graph at that commit. So all four notebooks read the same
runs, and the three runs record the same commit. On 2026-10-02 that commit is
`f386291-dirty`, and its three runs are the ones `docs/assumptions/A-27.md` and
`docs/findings/F-04.md` quote for these graphs (assumption A-27 and finding F-04).

That is less than a promise that the three trained from identical code. The
`-dirty` means the working tree held uncommitted changes, and these runs did
not save which ones, so whether all three trained from the same tree cannot be
checked; they finished within six minutes of each other. What supports
identical code is indirect: trained again from other states of the code, two
of the three give the same results on every seed — Table 3's matches the run
at the committed state `f4f40ad`, and the every-sensor graph's the run at
`9856794-dirty` — and the correlation graph has no other run of this
configuration to check against. The comparison notebook stops if the newest
set moves to another commit, because its text quotes these runs' figures.

(Corrected 2026-10-02, before this folder was first committed: this paragraph
said that a difference between two graphs here was never also a difference
between two versions of the code, which the recorded commits cannot show. A
review the same day found it.)

If a graph has no such run, the setup cell stops and prints the commands that
make the three, to be run from one committed state of the code:

```bash
python scripts/train.py --config config/fd001.yaml --graph physics --seeds 5
python scripts/train.py --config config/fd001.yaml --graph complete --seeds 5
python scripts/train.py --config config/fd001.yaml --graph correlation --seeds 5
```

**One network for the worked examples: seed 42's.** It is loaded from the
`model.pt` saved beside its run, with its data rebuilt under the same seed,
because the seed decides which 20 engines were held back — and, for the
correlation graph, which engines the graph was measured on. The loading cell
scores the test engines again and requires the error the run recorded.

**One window to follow: the first test engine's**, whose 30 cycles end at its
last recorded cycle, 31. Notebook 02's step 9 averages the same window by
hand, so its numbers can be followed from there into these notebooks.
Following one window through a layer chooses nothing, so using a test engine
for it tells the network nothing about its test score.

**The same measurements of a graph**: its edges, the sensors it joins to
nothing, the edges it shares with Table 3, how many different rows its
rescaled table has, and how many different sensors leave its first graph
layer.

**Do not confuse** different rows with the independent rows that
`docs/assumptions/A-27.md` and `docs/findings/F-18.md` count for the same
graphs (assumption A-27 and finding F-18). A row is independent of the others when it cannot be built by adding
up multiples of them, so a table can have fewer independent rows than
different ones: Table 3's rescaled table has 9 different rows and 8
independent ones. The notebooks here count different rows, because two
sensors with identical rows hold identical numbers after every graph layer.

## What no notebook here establishes

- **How the engine works.** A graph here is a list of which sensors the
  network may combine. The paper's graph encodes its authors' engineering
  judgement; the correlation graph encodes which readings rose and fell
  together in 80 engines. Which one helps this network predict is a different
  question from which relationships are physically real.
- **Anything about the paper's full network.** Only the comparison notebook's
  step 6 reads runs with the recurrent stage, and those keep the parameters of
  the pass that did best on the held-back engines, which the paper never
  describes and which changes what that stage is worth (finding F-16). This
  project's comparisons of settings have reversed when the rest of the network
  changed (working rule 3), so each result here belongs to its own
  configuration only.
- **The paper's own figure for removing its graph**, 0.56 cycles on FD001
  (its Table 11, 12.34 against 11.78). It is for the full network, with a
  replacement graph the paper never describes, and nothing here reproduces
  it.

## Adding a graph

1. Write it as a function in `src/thesis_gnn_rul_xai/graph/adjacency.py`, with
   a section in that folder's README, a test, and a worked example in
   `tests/test_examples.py`, and make it a value of the configuration's
   `graph` setting.
2. Train it from the same commit as the other graphs:
   `python scripts/train.py --config config/fd001.yaml --graph NAME --seeds 5`.
3. Copy the layout of `03_correlation_fd001.ipynb`. Add the graph's name to
   the list each setup cell searches, a row to the table above, and a line to
   the documentation site's list in
   `src/thesis_gnn_rul_xai/docsite/catalog.py`.
