# Thesis GNN RUL XAI

Predicting how much life is left in a jet engine, built as the software
foundation for a master's thesis on **explainable** graph-based deep learning
for predictive maintenance. It re-implements a published method on the NASA
C-MAPSS benchmark and measures how far it can be taken from the paper alone.

**What is implemented.** The preprocessing of the paper's Section 3.2, the
graph network of its Table 4, the training loop of its Table 6, the four
evaluation measures of its Section 3.3, and the dynamic SHAP analysis of its
Section 2.4 — the last computed exactly rather than by sampling. The
per-sensor LSTM stage of its Section 2.2 is built too, as a stage that can be
switched in and out; it is not the default. See
[section 12](#12-known-limitations).

**Where that gets to.** On FD001, **RMSE 16.50 ± 0.42** across five seeds
under the current default settings, against the 14.93 the paper reports for
its own "ESTGNN w/o LSTM" ablation — on an architecture that matches the
paper's Table 4 exactly: its printed weight shapes give 61,600 learned
numbers, and the two offsets its Equations 12 and 13 print for the dense
layers add 129, for 61,729 in all. (Corrected 2026-10-02: this said "the
61,729 parameters its printed shapes imply"; the shapes alone give 61,600.)

**Read how the settings were chosen before quoting that number.** The settings
the paper leaves unspecified were deliberately chosen to land near its figure
(assumption A-22), so the configuration is not independent of the quantity it
is compared against. Under the previous, independently chosen settings the
same code gives 20.35 ± 0.75. The paper specifies neither, so the honest summary
is that this re-implementation **spans 16.50 to 20.35 cycles depending on
choices the source leaves open**. [Section 8](#8-results-so-far) gives all four
measures.

**Status.** Ruff is clean. The notebooks are committed without their outputs
since 2026-10-04, because every run gives new results; run one to see them.
Each of the sixteen was last run from top to bottom on 2026-10-02 or
2026-10-03 without an error. Section 10 gives the test counts. (Corrected 2026-10-03:
this said the notebooks were re-executed on 2026-09-14 and that their stored
outputs match the code. Every notebook has been run since, twelve of them
added after that date, and nothing checks the match. Found by a review of
2026-10-03 that flagged the date, and re-checked by reading the run times
stored in each notebook and the commits that added them.)

---

## Table of contents

1. [What this project is about](#1-what-this-project-is-about)
2. [Words you need before reading further](#2-words-you-need-before-reading-further)
3. [The paper being re-implemented](#3-the-paper-being-re-implemented)
4. [Installation](#4-installation)
5. [Getting the data](#5-getting-the-data)
6. [Running the pipeline](#6-running-the-pipeline)
7. [What the pipeline does, step by step](#7-what-the-pipeline-does-step-by-step)
8. [Results so far](#8-results-so-far)
9. [How the project is organised](#9-how-the-project-is-organised)
10. [Testing](#10-testing)
11. [Rules this project follows](#11-rules-this-project-follows)
12. [Known limitations](#12-known-limitations)
13. [References](#13-references)

---

## 1. What this project is about

Every jet engine has a limited life: its parts wear a little with every flight.
An engine that fails in service is dangerous and expensive, costing an airline
repairs, lost flights and, at worst, human lives. The way to prevent that is to
service the engine in time. Servicing it on a fixed schedule is safe only if
the schedule is set for the engines that wear fastest, so most engines are
taken off the wing with many flights still left in them, and that usable life
is wasted. It would be better to estimate, for each engine, how many more
flights it will survive, and service it just before that point. Estimating that
number from sensor readings is the task this project works on.

Two things make the task interesting for a thesis:

**Sensors are related to each other, not independent.** A turbofan engine is a
chain of components: air enters the fan, is compressed, is burned with fuel,
and drives the turbines on the way out. A change in one part shows up in the
sensors of the next. A model that treats the 21 sensors as 21 unrelated numbers
throws that structure away. A *graph neural network* keeps it.

**A number alone is not enough for a safety decision.** If a model says "this
engine has 40 flights left", a maintenance engineer needs to know why before
acting on it. Methods that answer the "why" question are called *explainable
artificial intelligence*. Combining the two — a graph-based model that is also
explainable — is the subject of the thesis.

---

## 2. Words you need before reading further

Short definitions of every technical term used in this repository. Each one
points to a source where it is defined properly.

> If you would rather have these built up in order, with figures, than
> defined in a list, read `docs/learn/` instead — twelve pages covering
> the same ground for a reader who has never met any of it. They are best read
> in a browser: `python scripts/docs_ui.py`. (Corrected 2026-10-19: this and
> the table in section 9 said "eight short pages"; there are twelve, and
> several run to hundreds of lines.)

**RUL — Remaining Useful Life.** The number of flight cycles an engine will
still run before it fails. This is the quantity being predicted. Each row of
NASA's files is "a snapshot of data taken during a single operational cycle"
(NASA Prognostics Center of Excellence, 2008), and the paper describing the
dataset takes "one measurement snapshot per flight" (Saxena et al., 2008,
p. 5). Read together, they mean that one *flight cycle* is one flight. Neither
source says it in those words; it is this project's reading of the two, as
[Data, What one cycle is](docs/learn/01_the_problem.md#what-one-cycle-is)
explains. The cycle is the unit of time used everywhere in this project.
(Corrected 2026-10-19: this said a flight cycle is "one complete take-off,
flight, and landing", which neither NASA's readme nor the dataset paper says.
Found when the documentation pages were given their sources.)

**C-MAPSS.** *Commercial Modular Aero-Propulsion System Simulation*, a NASA
program that simulates a large turbofan engine. NASA used it to create a public
benchmark dataset in which simulated engines run until failure while 21 sensors
record the engine state once per cycle (Saxena et al., 2008). It has become the
standard test problem for this task, which is what makes results comparable
between papers.

**Predictive maintenance.** Servicing equipment based on its measured
condition, rather than at fixed intervals or after it breaks.

**Graph.** A set of points (**nodes**) with connections between them
(**edges**). Here every sensor is a node, and an edge means "these two
measurements are physically related".

**Adjacency matrix.** A graph written as a table of numbers. For 14 sensors it
is a 14-by-14 table where entry `(i, j)` is 1 if sensors *i* and *j* are
connected and 0 if they are not.

**GNN — Graph Neural Network.** A neural network that operates on a graph. It
works by **message passing**: each node repeatedly collects the current values
of its neighbours, combines them, and updates its own value. Stacking three
such layers lets information travel up to three edges away. See Corso et al.
(2024) for an accessible introduction, or Kipf and Welling (2017) for the
specific layer used here.

**LSTM — Long Short-Term Memory.** A type of neural network designed for
sequences, which keeps an internal memory of what it has already seen
(Hochreiter & Schmidhuber, 1997). The paper being re-implemented uses one LSTM
per sensor. This project deliberately leaves that part out; see
[section 12](#12-known-limitations).

**XAI — Explainable Artificial Intelligence.** Methods that make the output of
a machine learning model understandable to a person, for example by saying
which inputs mattered most for a particular prediction.

**Cooperative game theory.** The branch of mathematics that studies how a group
of players who achieve something together should fairly divide the credit for
it. It is the theoretical basis of the next two terms.

**Shapley value.** A method from cooperative game theory for dividing credit
among players (Shapley, 1953). It asks: for each player, how much does the
group's result change when that player joins, averaged over every possible
order in which the group could have been assembled? That average is the
player's fair share. Lundberg and Lee (2017, page 4) state that it is the only
division meeting three conditions at once, two of them close to properties
SHAP is built on: that the shares add up exactly to the result, and that a
player whose contribution grows never receives a smaller share. They cite
Shapley (1953) for the value itself, but credit that proof of uniqueness to
Young (1985), their reference [10]. Neither Shapley's chapter nor Young's
article is held in `docs/references/`, so this is Lundberg and Lee's account,
not one checked against those originals.

**SHAP — SHapley Additive exPlanations.** The application of the Shapley value
to machine learning (Lundberg & Lee, 2017). The "players" become the model's
input features and the "result" becomes its prediction. Each feature receives a
number saying how much it pushed the prediction up or down, and those numbers
add up exactly to the difference between this prediction and a starting value.
Lundberg and Lee's starting value is the model's average prediction (their
Figure 1, page 5). This project's is the model's prediction for one average
training window: each sensor's reading at each of the window's 30 positions,
averaged over the windows of the engines the model trained on. That average
window is what stands in for a sensor left out of a group, so it is also where
the adding-up has to start (assumptions A-20 and A-24). The two starting values
are different numbers, because the average of a network's outputs is in general
not its output for the average input. In this project the players are the 14
sensors.

**Correction, 2026-10-02.** The first of these two entries stated the Shapley
value's uniqueness with Shapley (1953) as its only source, without naming the
conditions; the second said that SHAP values add up to the prediction minus the
average prediction. Lundberg and Lee credit the uniqueness they rely on to
Young (1985), and the average prediction is their starting value, not this
project's. Found when the cited papers were collected into `docs/references/`
and checked against the places that cite them.

**Data leakage.** When information that should be unavailable at prediction
time reaches the model during training. It usually makes results look better
than they are, though nothing guarantees that direction in a given case, and
is a leading cause of published findings that cannot be reproduced
(Kapoor & Narayanan, 2023). Two rules in [section 11](#11-rules-this-project-follows)
exist to prevent it. (Corrected 2026-10-02: this said leakage makes results
look better, without "usually"; the authors' preprint, the copy held in
`docs/references/`, says on PDF page 4 that it "usually leads to inflated
estimates". Found by the search for that claim after the review of the
playground's information cards on 2026-09-26.)

**Normalisation (rescaling).** Putting different measurements onto a comparable
numerical scale, so that a temperature near 1400 does not dominate a ratio near
8 simply because it is a larger number. This project uses the **z-score**,
which subtracts each sensor's mean and divides by its standard deviation, the
typical size of its spread, so that a rescaled reading says how many spreads it
lies from that sensor's average:

$$\text{scaled} = \frac{x - \bar{x}}{s}$$

with the mean $\bar{x}$ and the standard deviation $s$ taken from the training
engines only. Min-max scaling, which maps each sensor's smallest training value
to 0 and its largest to 1 separately for each operating condition, is available
as an alternative and was the default until 2026-08-29. The paper describes
neither, so either one is this project's choice (assumption A-03).

**Exponentially weighted moving average (EWMA).** The filter used to damp
measurement noise. Each smoothed value is a blend of the current reading and
the previous smoothed value:

$$S_t = \alpha X_t + (1 - \alpha) S_{t-1}, \qquad S_1 = X_1, \qquad \alpha = 0.5$$

Because it only ever looks backwards, it cannot leak a future reading into the
present. The strength 0.5 is this project's reading of a sentence the paper
prints ambiguously (assumption A-04). The paper's Equation 18 is the recursive
formula above; the software library used here can also compute a variant that
treats the first few readings differently, and the recursive one is used
(assumption A-09).

**Graph convolution.** The operation each of the three network layers performs:

$$H' = \mathrm{ReLU}\left(\hat{A} H W\right), \qquad \hat{A} = D^{-1} (A + I)$$

`H` holds the current node features, `W` is learned, and `Â` is the fixed
sensor graph with each sensor also joined to itself and each row divided by its
own total, so that every sensor receives the average of itself and its
neighbours. The paper never prints which rescaling of the graph it uses; this
one is this project's choice (assumption A-21).

**Piecewise-linear RUL.** The target. While an engine is healthy the label is
held at a ceiling; after that it falls by one per cycle:

$$\mathrm{RUL}(t) = \min\left(T_{\text{total}} - t,\; \mathrm{RUL}_{\max}\right),
\qquad \mathrm{RUL}_{\max} = 125$$

The paper prints the formula but not the value 125, which is this project's
choice (assumption A-01).

**Time window.** A short, fixed-length slice of an engine's history, for
example 30 consecutive cycles. Windows are what the model actually receives,
because histories have different lengths and a network needs a fixed input size.

**Ablation.** An experiment that removes one part of a method and measures
what the rest can do without it. If accuracy barely changes, that part was
not doing much. It is how a paper shows which of its ingredients matter.

**The four measures used to score a prediction.** All four are computed over
the $N$ test engines, from one prediction each. Throughout, $y$ is the true
remaining life of an engine and $\hat{y}$ the predicted one, both in flight
cycles, and the error is

$$d = \hat{y} - y \qquad \text{positive means the model was optimistic}$$

#### RMSE — root mean squared error

The typical size of the error, in flight cycles.

$$\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} d_i^{\,2}}$$

Squaring makes it symmetric: a prediction 10 cycles too high counts exactly
like one 10 cycles too low. This is the number most papers quote.

#### NASA score (SF)

The official score of the 2008 PHM competition this dataset comes from.

$$\mathrm{SF} = \sum_{i=1}^{N} s_i, \qquad
s_i = \begin{cases}
e^{-d_i/13} - 1 & d_i < 0 \quad \text{(early, safe)} \\
e^{\;d_i/10} - 1 & d_i \ge 0 \quad \text{(late, dangerous)}
\end{cases}$$

Unlike RMSE it is **asymmetric**, and deliberately so: the divisor 10 for a
late prediction is smaller than the 13 for an early one, so the same error
costs more when the model was optimistic. Predicting more life than an engine
has leaves a failing engine flying; predicting less only services it early.
Being exponential, one badly wrong engine can outweigh ninety-nine good ones.
Lower is better, 0 is perfect, and because it is a **sum** it grows with the
size of the test set — so it can only be compared between runs on the same
sub-dataset.

#### R² — coefficient of determination

The fraction of the variation in the true labels that the model accounts for.

$$R^2 = 1 - \frac{\sum_i (\hat{y}_i - y_i)^2}{\sum_i (y_i - \bar{y})^2},
\qquad \bar{y} = \text{mean of the true labels}$$

1 is perfect; 0 is no better than always guessing the average; negative is
worse than that. (The paper prints a different denominator; see A-16.)

#### ASUE — Average Safe Underestimation Error

The average shortfall, counted only over the engines the model was
*pessimistic* about.

$$\mathrm{ASUE} = \frac{1}{N_{\text{under}}}
\sum_{i\,:\,\hat{y}_i < y_i} \max\left(0,\; y_i - \hat{y}_i\right)$$

Underestimating is the safe direction, so this measures the cost of being too
cautious: servicing engines that still had life left in them. Engines the
model was optimistic about do not contribute at all. When no engine was
underestimated, `N_under` is 0 and the formula is 0 / 0, so the code reports
"not a number" rather than 0, and every results file a script writes since
2026-10-03 has the count of underestimated engines, `n_early`, beside each
ASUE; for a file stored before then, and for the ASUEs quoted in this README,
it is the number of test engines minus `n_late`, which every stored run's
`results.csv` holds, and the comparison with ordinary models' `per_seed.csv`
(assumption A-39; until 2026-10-03 the code returned 0 and called it the best
value, and the first version of this sentence said every ASUE had the count
beside it).

All four are implemented in `src/thesis_gnn_rul_xai/evaluation/metrics.py`,
on plain NumPy arrays, so a figure copied out of a published table can be
scored the same way a model's output is.

> The formulas above are LaTeX, written between `$` delimiters. GitHub and
> JupyterLab both render that syntax natively, so **no library or build step
> is involved** — nothing needs installing to read them. In a plain text
> editor they appear as their source, which is still readable.

---

## 3. The paper being re-implemented

> Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
> (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
> enhancement for aeroengine RUL prediction. *Aerospace Science and
> Technology*, 168, 110875. https://doi.org/10.1016/j.ast.2025.110875

Their method, called ESTGNN, has four parts: one LSTM per sensor; a
hand-designed sensor graph based on engine physics; three graph convolution
layers; and SHAP applied after training to show which sensors mattered at each
moment.

**No source code was published with the paper**, and its data availability
statement reads "Data will be made available on request". This project is
therefore a re-implementation from the text alone.

### An important limit on what this code can claim

The paper does not specify everything needed to rebuild it. The choices made
here rather than read from the paper include the value of the label ceiling,
the entire input rescaling procedure, and the size of the LSTM networks.
[`docs/assumptions/`](docs/assumptions/README.md) records 41 of them, each with its
justification and its measured or argued effect; one has since been withdrawn,
because the paper turned out to state it.

Nothing in this repository should be described as "reproducing ESTGNN".
**Which published row a result may be set beside depends on which temporal stage
produced it**, and every run records that. With the default stage there is no
recurrent network, so the comparison is the **"ESTGNN w/o LSTM"** row of the
paper's own ablation table (their Table 11). With `--temporal lstm` the paper's
headline row becomes the right comparison instead, but only under a size of
recurrent network the paper never states (assumptions A-11 and A-26), and what
the stage is worth depends on whether the best pass's weights are kept
(finding F-16).

---

## 4. Installation

Python 3.10 or newer is required.

### With conda (recommended)

```bash
conda env create -f environment.yml
conda activate thesis-gnn-rul-xai
```

### With pip

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux or macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### PyTorch

Only `src/thesis_gnn_rul_xai/models/` needs it; the preprocessing pipeline and
its tests run without it, and the model tests skip themselves when it is
absent. `requirements.txt` installs the CPU build. To train on an NVIDIA card,
install the CUDA build for your driver instead, using the command generated at
<https://pytorch.org/get-started/locally/>, for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

`pip install -e .` installs the project in "editable" mode: the package becomes
importable from anywhere, while still pointing at the source files in this
folder, so edits take effect immediately without reinstalling.

### Checking the installation

```bash
pytest
```

The tests run on the real FD001 files, so download the dataset first (see
[section 5](#5-getting-the-data)). Without it they skip with a message saying
what is missing and where to get it, rather than failing.

---

## 5. Getting the data

The NASA files are not included in this repository; where to download them,
how to check them and how to keep them elsewhere is in
[`data/raw/README.md`](data/raw/README.md), the folder they go into.

---

## 6. Running the pipeline

### From the documentation site (start here)

```bash
python scripts/docs_ui.py
```

This opens a browser on the playground, where a run is designed, trained,
compared with the paper and explained. Its menu holds eight taught pages that
build the method from nothing for a reader who has never seen it, the
notebooks, the folder READMEs, the assumptions, the findings and the audits.

It is not a copy. Every page is rendered from the file on disk at the moment
you ask for it, and every figure is **computed** rather than drawn: the graph
drawings are laid out from the transcribed Table 3 and measured with the same
`adjacency_report()` the tests use, and the network diagram is produced by
building a real `ESTGNN` and reading its layers back. The figures that need the
NASA files say so when the files are absent rather than drawing an invented
signal.

Every run the playground starts is `scripts/train.py` or `scripts/explain.py`,
stored in `reports/` with the commit that produced it. The site serves on
`127.0.0.1` only and publishes nothing. See
[`src/thesis_gnn_rul_xai/docsite/README.md`](src/thesis_gnn_rul_xai/docsite/README.md).

Notebooks can be **read** on the site with the outputs they were saved with,
which needs nothing extra. To **run** one, the Open in JupyterLab button on its
page starts JupyterLab for you, which needs `pip install -e .[notebook]`.

### From the notebooks

```bash
jupyter lab notebooks/
```

Read `notebooks/README.md` first: it holds what every notebook shares. Then:

| Notebook | Covers |
|----------|--------|
| `01_preprocessing_fd001.ipynb` | steps 1 to 8: the raw text files become model-ready arrays |
| `02_model_fd001.ipynb` | steps 9 to 11: build the network, train it and score it |
| `temporal/01_last_cycle_fd001.ipynb` | the temporal stage, the part of the network that decides what each sensor carries into the graph layers from its last 30 cycles, as the default builds it: the newest reading kept and the other 29 dropped, standing in for the paper's recurrent stage (assumptions A-07 and A-11); worked on one real window, with its cost in learned numbers and its stored five-seed results |
| `temporal/02_mean_window_fd001.ipynb` | the average of each sensor's 30 readings in place of the newest one, the other choice that keeps the paper's one number per sensor (assumption A-07) |
| `temporal/03_window_summary_fd001.ipynb` | three numbers per sensor, its newest reading, its average and its change across the window, a stage this project added (assumption A-28) |
| `temporal/04_lstm_fd001.ipynb` | the paper's own stage, one LSTM per sensor (its Section 2.2), at a size the paper does not give (assumption A-26), with one sensor's 30 readings followed through a trained network by hand |
| `temporal/05_gru_fd001.ipynb` | the same with a gated recurrent unit, a simpler relative of the LSTM, in place of each sensor's LSTM; not a stage the paper has (assumption A-36) |
| `temporal/06_raw_window_fd001.ipynb` | all 30 readings handed to the first graph layer untouched, the arrangement this project removed on 2026-08-28 and keeps behind a gate (assumption A-07); no stored run of it exists under these settings, so it shows no result |
| `temporal/07_comparison_fd001.ipynb` | the stages with stored five-seed runs side by side, on the test and held-back engines, paired by seed |
| `graph/01_physics_fd001.ipynb` | the paper's sensor graph on the real data: its table, the rescaled table, one graph layer worked by hand, how far three layers reach, and its stored five-seed result |
| `graph/02_complete_fd001.ipynb` | every sensor joined to every other, and why it leaves the network predicting from one number |
| `graph/03_correlation_fd001.ipynb` | the 41 sensor pairs whose readings move together most strongly, built step by step from the training engines |
| `graph/04_comparison_fd001.ipynb` | the three graphs' stored five-seed runs side by side, on the test and held-back engines, paired by seed |
| `xai/01_shap_fd001.ipynb` | the paper's explanation method, exact Shapley values, for one prediction and then at every cycle of one engine's life; then how often the four sensors whose contributions are largest in size, sign set aside, change from one window to the next, on five seeds of the default network and five of one with the recurrent stage of the paper's Section 2.2, at a size the paper does not give (assumption A-26), beside the paper's Fig. 10 |
| `xai/02_occlusion_fd001.ipynb` | a cheaper explanation method that is not in the paper, set beside the Shapley values |
| `xai/03_walks_fd001.ipynb` | a method not in the paper that splits a prediction into the routes it took through the sensor graph, beside what removing each connection does |

The first two are read in order. Notebooks 01 to 06 of `temporal/` each show
one temporal stage on one real window, and read the stored runs of that stage
where any exist; 07 works on no window and compares those runs.
`notebooks/temporal/README.md` says what they share and which runs they read. The `graph/` notebooks read the stored runs of
three sensor graphs from one commit; `notebooks/graph/README.md` says which.
The `xai/` notebooks each explain the same
stored trained network, one method per notebook; `notebooks/xai/README.md` says
why. One step departs from that on purpose: step 8 of the Shapley-value
notebook loads all five seeds of two other stored runs, alike except in their
temporal stage, because its question is whether that stage changes the answer.
Each notebook explains every step, shows the intermediate tables and
figures, and verifies the stage as it goes. Together they are the intended
entry point for a reader who wants to understand what happens and why.

> Every notebook is shipped with its outputs already filled in, so it can
> be read without running anything. Those outputs were produced on the **real
> FD001 files**, checked against the counts of the files NASA distributes
> (20 631 training rows, 100 training engines, 100 test engines; the paper's
> Table 5 prints the same engine counts, and no held source prints the row
> count). The data files themselves are not in this repository; see
> [section 5](#5-getting-the-data). (Corrected 2026-10-02: this said "verified
> against the published release", but the row count is published nowhere this
> project holds, and NASA's readme prints FD004's two engine counts swapped;
> found when the cited papers were collected into `docs/references/` and
> checked against the places that cite them.)

### From the command line

```bash
python scripts/prepare_data.py --run --config config/fd001.yaml
```

### From Python

```python
from thesis_gnn_rul_xai.config import load_config
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing

prepared = run_preprocessing(load_config("config/fd001.yaml"))
print(prepared.summary())
prepared.save()
```

### Training a model

```bash
python scripts/train.py --seeds 5
```

This preprocesses, builds the network of Table 4, trains it with the settings
of Table 6, and reports the four measures on the test engines, averaged over
five seeds. One seed is not evidence, so the script says so when given only
one. `--graph complete` runs the "without physics prior" ablation;
`--graph random` and `--graph rewired` give the network graphs with the
paper's 41 connections but none of its engineering knowledge, drawn at random
for each seed (assumption A-41, not yet trained); and
a configuration that holds no engine back (`validation_fraction: 0.0`) follows
the paper's procedure: with nothing to stop on, all 200 epochs run and the last
one's weights are kept. `--no-early-stopping` changes nothing then. It matters
only when engines are held back, where it runs all 200 epochs but still
restores the weights of the held-back engines' best pass at the end
(assumption A-18). (Corrected 2026-10-03: this said `--no-early-stopping`
alone reproduces the paper's fixed 200 epochs, noticed while adding the
paragraph below; the first correction then said the paper's procedure needs
the option together with that configuration, and a review the same day read
`training/run.py` and found the configuration alone does it.)

Each run is written to a new folder under `reports/` whose name records its
settings, among them whether training engines were held back and whether they
changed with the seed: `varsplit`, `fixedsplit`, or `nosplit` when none were
held back. (Added 2026-10-03: until that day a run that held none back was
still named `varsplit` and recorded as varying its split, and stored runs keep
those names.) `src/thesis_gnn_rul_xai/training/README.md` reads a name part by
part, under "Loading a stored run back".

### What you get

Five files in `data/processed/`:

| File | Contents |
|------|----------|
| `windows.npz` | The training, validation and test examples as arrays |
| `adjacency.npz` | The sensor graph, before and after normalisation |
| `normalization_statistics.csv` | The learned scaling values, one row per sensor |
| `config.yaml` | The exact settings used |
| `diagnostics.json` | What the pipeline observed while running |

The feature array has shape `(examples, 14, 30)`: for each example, 14 sensors,
each carrying its whole 30-cycle window. The model's temporal stage turns that
into the `[batch, 14, 1]` its first graph layer takes, which is the input shape
printed in the paper's Table 4. Under the default stage that one number is the
most recent reading of the window, a stand-in for the paper's recurrent stage
(assumption A-07).

---

## 7. What the pipeline does, step by step

Paths in the "Where" column are relative to `src/thesis_gnn_rul_xai/`, so
`data/loading/` means `src/thesis_gnn_rul_xai/data/loading/`, whose code is
its `__init__.py` and whose explanation is its `README.md`.

| # | Step | What happens | Where |
|---|------|--------------|-------|
| 1 | Load | Read the three text files; work out the Remaining Useful Life at every row | `data/loading/` |
| 2 | Diagnose | Find sensors that never change, and compare with the paper's selection | `data/sensors/` |
| 3 | Select | Keep the paper's 14 sensors, in the order its adjacency matrix expects | `data/sensors/` |
| 4 | Cap labels | Hold the label at 125 while the engine is healthy | `data/labeling/` |
| 5 | Smooth | Remove measurement noise with a backward-looking filter | `data/smoothing/` |
| 6 | Split | Hold out 20 % of engines for validation, **before any statistic is learned** | `data/windowing/` |
| 7 | Rescale | Put all sensors on a comparable scale, using training engines only | `data/normalization/` |
| 8 | Window | Cut histories into 30-cycle examples | `data/windowing/` |
| — | Graph | Build the 14-node sensor graph from the paper's Table 3, and describe it: 41 edges, diameter 4, `T24` its only articulation point | `graph/adjacency.py` |

Those eight steps turn the raw files into arrays. Four more stages turn the
arrays into a result:

| # | Stage | What happens | Where |
|---|-------|--------------|-------|
| 9 | Build the network | Three graph convolution layers of widths 32, 64 and 32, then two fully connected layers ending in a sigmoid — the paper's Table 4, transcribed layer by layer | `models/estgnn.py` |
| 10 | Train | Adam, 200 epochs, batch size 32, learning rate 1e-3 decaying to 1e-4, mean squared error — the paper's Table 6 | `training/run.py` |
| 11 | Score | RMSE, the NASA score, R² and ASUE, on the 100 test engines | `evaluation/metrics.py` |
| 12 | Explain | Exact Shapley values for the 14 sensors, recomputed at every cycle — the paper's Section 2.4 | `explain/shap.py` |

Every folder under `src/thesis_gnn_rul_xai/` has a `README.md` that explains
its step in full — the reasoning, the traps, and what each function gives you
back. Start with `data/README.md`, whose sections on rescaling and on windowing
contain the decisions that matter most for the validity of the results. For the
graph, `adjacency_report()` computes every structural figure quoted above, so
none of them has to be taken on trust.

---

## 8. Results so far

Every figure below is the mean over **five seeds**, with the standard deviation
across those seeds. One seed is not evidence, and the training script says so
when it is given only one.

### On FD001

| Configuration | RMSE | NASA score | R² | ASUE |
|---------------|------|-----------|-----|------|
| **Current default** — z-score scaling, row rescaling, 20 % of engines held back, the paper's graph | **16.50 ± 0.42** | 540 ± 45 | 0.830 ± 0.008 | 11.65 ± 1.00 |
| the same, every sensor joined to every other | 19.28 ± 0.43 | 796 ± 79 | 0.768 ± 0.010 | 13.76 ± 0.49 |
| the same, the 41 most strongly correlated sensor pairs | 16.59 ± 0.35 | 1033 ± 292 | 0.829 ± 0.007 | 9.87 ± 0.15 |
| the same graph as the default, each sensor carrying its last reading, average and change over the window — **not in the paper** | 13.59 ± 0.68 | 375 ± 71 | 0.885 ± 0.012 | 8.64 ± 0.56 |
| z-score, row, **nothing held back** — the default until 2026-09-08 | 16.82 ± 0.16 | 610 ± 25 | 0.824 ± 0.003 | 12.33 ± 0.60 |
| the same, every sensor joined to every other | 19.21 ± 0.06 | 782 ± 9 | 0.770 ± 0.001 | 13.94 ± 0.15 |
| **min-max** scaling, symmetric rescaling, 20 % held back — the default until 2026-08-29 | 20.35 ± 0.75 | 827 ± 85 | 0.742 ± 0.019 | 15.67 ± 2.05 |
| the same, every sensor joined to every other | 20.40 ± 0.40 | 896 ± 113 | 0.741 ± 0.010 | 14.96 ± 1.65 |

(Corrected 2026-10-02: the fifth row said the default held nothing back
until 2026-09-02, a day on which the held-back setting did not change.
Holding engines back returned on 2026-09-08; the log's entry on the held-back
engines gives the dates from the project's history (assumption A-06).)

**What the paper's sensor graph is worth depends on what it is replaced
with.** Joining every sensor to every other costs 2.78 cycles under the current
default — paired `p = 0.0005`, every one of the five seeds worse. A graph built
from the data instead, keeping the 41 sensor pairs whose readings move together
most strongly — the same number of edges as the paper's — costs 0.10 cycles at
`p = 0.66`, indistinguishable from nothing, although its predictions are later
than the paper's graph's, by 1.75 cycles on average and on every seed
(`p = 0.0009`). So most of what the every-sensor comparison costs is that
graph's averaging of every sensor together, not the loss of the engineering
knowledge. That is written up in `docs/assumptions/A-27.md` (assumption A-27).

(Corrected 2026-10-03: this said the correlation graph's predictions "lean
late". They do against the test label, the remaining life capped at 125 cycles,
but against the life the test engines truly had its average error is below
zero on one seed of five and its lean is not established; what holds either way
is that it is later than the paper's graph on every seed. Found by a review of
the wording around the capped label.)

**Most of what the default is missing is the direction the readings are moving
in.** The default hands each sensor node only its last reading. Handing it three
numbers from the same thirty readings instead — the last, the average, and how
far the best straight line through them rises across the window — lowers the
error by 2.91 cycles on the same code and the same seeds, on every seed, paired
`p = 0.00025`; resampling the 100 test engines keeps the whole difference below
zero. That arrangement is not the paper's, which gives each node one number from
its own LSTM, so no published row applies to it, and on the test engines it
cannot be told apart from this project's run of the paper's recurrent stage
(assumption A-28).

**The every-sensor comparison has also given opposite answers under settings
the paper does not specify**: 2.78 and 2.40 cycles with z-score scaling, 0.044
(`p = 0.913`) with min-max. The history, including a withdrawn 5.7-cycle figure
from the removed `window` mode, is in `docs/findings/F-04.md` (finding F-04). The
paper reports 0.56 cycles for its own version of this comparison, for its full
model, and never says what it replaced its graph with.

The min-max rows differ from the current default in two ways — the input
scaling and the rescaling of the graph — so these runs do not isolate which
caused the difference, and the log says so rather than guessing.

Reproduce the first three rows with:

```bash
python scripts/train.py --config config/fd001.yaml --seeds 5
python scripts/train.py --config config/fd001.yaml --graph complete --seeds 5
python scripts/train.py --config config/fd001.yaml --graph correlation --seeds 5
```

Two earlier rows have been withdrawn. They used a `window` node-feature mode in
which each node carried all 30 cycles, which made the first weight matrix
`(30, 32)` instead of the `(1, 32)` Table 4 prints — 928 parameters the paper's
architecture does not have, computing a bank of linear filters over time. That
is a temporal encoder, which the default network deliberately does not have
(assumption A-11). The mode was removed on 2026-08-28; its numbers
(13.45 ± 0.20 physics, 19.17 ± 0.45 complete) are reproducible at commit
`cdcc827` and are recorded in `docs/findings/F-06.md` (finding F-06), but they are
not this project's results.

### How that compares with the paper

| Paper's variant | RMSE | Where |
|-----------------|------|-------|
| ESTGNN, full model | 11.78 | Tables 7–10 |
| **ESTGNN w/o LSTM** | **14.93** | Table 11 |
| ESTGNN w/o GCN | 16.42 | Table 11 |
| ESTGNN w/o physics prior | 12.34 | Table 11 |

**Only one of those numbers is comparable with the current default.** It was
produced without the per-sensor LSTM (see [section 12](#12-known-limitations)),
so the matching row is "w/o LSTM", and the comparison is 16.50 ± 0.42 against
14.93 — a 1.57-cycle shortfall. The paper's ablation table reports RMSE only,
which is why the other three columns are blank for it.

**The test set is too small to establish that shortfall.** The ± is the
spread over five seeds, and all five are scored on the same 100 test engines.
Drawing 100 engines at random from those 100, repeats allowed, 20,000 times,
and recomputing the five-seed average each time, gives a 95 % interval of
13.98 to 18.91 cycles, and 14.93 lies inside it. So 1.57 cycles describes these
100 engines; it is not a difference this test set can show between this network
and the paper's (finding F-06; added 2026-10-02).

### What that comparison does and does not show

The architecture matches: input `[batch, 14, 1]`, `W(1) ∈ ℝ^(1×32)`, the same
layer widths, and **61,729 parameters — Table 4's printed weight shapes
(61,600 numbers) plus the two offsets of the dense layers that Equations 12 and
13 print (129)**, with none in the graph layers (assumption A-12), asserted by a
test. Table 4 prints no total; the arithmetic is in `docs/assumptions/A-07.md`
(assumption A-07). (Corrected 2026-10-02: this said "exactly what Table 4's
printed shapes imply"; the shapes alone give 61,600.) The only thing removed is the LSTM,
which is what "w/o LSTM" means. So the **1.57-cycle gap has no shape mismatch
to hide behind**, and what is left to explain it is the documented choices:
the input rescaling was picked for landing nearest the paper's own figure
rather than read from the paper (assumptions A-03 and A-22), holding engines
back and stopping early are additions (assumptions A-06 and A-18), the target
scaling, weight initialisation and learning-rate schedule are all unstated
(assumptions A-13, A-14 and A-17), and capping the test labels lowers the
current default's figure by a measured 1.508 cycles, from 18.01 uncapped
(assumption A-02), so the underlying gap is wider than it looks. (Updated
2026-10-02: this gave only the 1.311 cycles measured on the configuration of
2026-08-28.)

[`docs/findings/`](docs/findings/README.md) records this in full (finding
F-06), together with what the sensor graph is worth (finding F-04, and
assumption A-27 in `docs/assumptions/`) and what the Shapley values say about the sensor that carries
almost no information (finding F-05).

### Beside ordinary models with no sensor graph

Whether the graph network is worth its complexity needs a model with no graph
in it, fitted on the same training engines and scored on the same test engines.
`scripts/baselines.py` does that for stored runs: for each seed it fits a fixed
list of models from scikit-learn, a Python library of standard statistical
models, every one at the library's default settings apart from the small
network's width of 64 rather than the library's 100, taken from the review of
2026-09-08 that first asked for this comparison
(`docs/audits/AUDIT_2026-09-08_scientific_validity.md`), on
exactly the 80 engines that seed's network learned from, and scores them on the
same 100 test engines. Linear regression predicts a weighted sum of its inputs.
A random forest averages 100 decision trees, each a chain of yes-or-no
questions about the inputs ending in a predicted value. Gradient boosting adds
100 small trees one after another, each fitted to what the trees before it
still get wrong. The small network computes 64 weighted sums of its inputs,
replaces each negative one by zero, and predicts a weighted sum of the 64. Each
model is given either the fourteen numbers the default network receives, each
sensor's latest reading, or three numbers per sensor describing its 30-cycle
window: the latest reading, the average, and the slope, how fast the readings
rise or fall per cycle. Their predictions are held to between 0 and 125 cycles,
the range the network cannot leave (assumption A-38). The record run,
`baselines_fd001_physics_row_cap_5seed_varsplit_cfgba82e3_6733d77_20261003T100207Z`
from a clean `6733d77`, sets them beside the default re-trained from a clean
`f4f40ad` (`fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z`)
and beside the same network with the paper's recurrent stage, trained from a
clean `19dceb5` with its best held-back pass restored
(`fd001_physics_row_cap_5seed_varsplit_lstm32_cfgba82e3_19dceb5_20261002T210700Z`):

| Model | Given | RMSE | NASA score | R² | ASUE |
|---|---|---|---|---|---|
| **Current default** | the latest readings, through the paper's graph | **16.50 ± 0.42** | 540 ± 45 | 0.830 ± 0.008 | 11.65 ± 1.00 |
| the same with the paper's recurrent stage, best held-back pass restored | each sensor's whole window | 13.23 ± 0.72 | 316 ± 57 | 0.891 ± 0.012 | 8.15 ± 0.48 |
| linear regression | the latest readings | 20.46 ± 0.22 | 1064 ± 116 | 0.739 ± 0.006 | 14.32 ± 0.29 |
| network with one layer of 64 | the latest readings | 17.21 ± 0.22 | 809 ± 107 | 0.815 ± 0.005 | 10.63 ± 0.40 |
| random forest | the latest readings | 18.06 ± 0.42 | 1048 ± 156 | 0.797 ± 0.009 | 11.73 ± 0.79 |
| gradient boosting | the latest readings | 17.25 ± 0.33 | 871 ± 116 | 0.815 ± 0.007 | 9.58 ± 0.80 |
| gradient boosting | latest, average and slope | 12.14 ± 0.35 | 223 ± 16 | 0.908 ± 0.005 | 8.72 ± 0.52 |
| gradient boosting | latest and slope | 11.83 ± 0.39 | 215 ± 16 | 0.913 ± 0.006 | 8.50 ± 0.78 |
| linear regression | latest, average and slope | 14.52 ± 0.12 | 358 ± 12 | 0.869 ± 0.002 | 10.97 ± 0.18 |
| gradient boosting trained on shuffled labels | the latest readings | 40.85 ± 0.29 | 20954 ± 1301 | −0.039 ± 0.015 | 26.43 ± 0.66 |
| the average training label, for every engine | nothing | 40.57 ± 0.09 | 18579 ± 1002 | −0.025 ± 0.004 | 26.16 ± 0.58 |

**Given the same fourteen numbers, the default is ahead of every flexible
ordinary model on every seed, by less than the test engines can show.** Its
RMSE is lower than the small network's by 0.72 cycles, the forest's by 1.57 and
gradient boosting's by 0.76, on all five seeds each, and its NASA score, which
charges a late prediction more than an early one, is lower than all three on
every seed. But resampling the 100 test engines leaves zero inside every one of
the three intervals for the RMSE difference, and none of the three paired tests
clears the bar this project sets when several comparisons are read together: p
below 0.05 divided by their number, here 0.05 / 58 for the 58 comparisons
behind this table and the one beside the networks trained the paper's way below
(assumption A-37). (Updated 2026-10-03: this said 0.05 / 29, for this table's
29 alone, until the second table was added; nothing in this paragraph or the
next changes side.) Against linear regression the
lead, 3.96 cycles, is clear on both counts. The network with the recurrent
stage is ahead of all four by 3.98 to 7.22 cycles, clearly on both.

**Given each window's latest reading and slope, an untuned gradient booster
beats the default clearly, and the recurrent-stage network on every seed by an
amount these five seeds and 100 engines do not establish.** On the latest
reading, the average and the slope it beats the default by 4.36 cycles on every
seed (paired `p = 0.0001`, interval over the test engines +2.03 to +6.68), and
on the latest reading and slope by 4.66 (`p = 0.0002`, +2.32 to +6.97).
Against the recurrent-stage network it is ahead on every seed too, by 1.09
cycles on all three and 1.40 on latest and slope, but neither clears the bar
(`p = 0.0048` and `0.0037`). The first interval over engines, −0.10 to +2.30,
contains zero; the second, +0.26 to +2.55, excludes it, but not once it is
widened to the same strictness as the bar. The slope alone is not enough: the
booster given only the slope beats the default by 2.80 cycles on every seed,
and trails the recurrent-stage network on four seeds of five (`p = 0.35`).

Both what the models are given and how they fit it matter. Adding the window
summaries lowers linear regression's RMSE by 5.94 cycles (20.46 to 14.52) and
gradient boosting's by 5.11 (17.25 to 12.14), while gradient boosting given
only the default's fourteen numbers is 0.76 cycles worse than the default: the
summaries carry information the default is never given, the direction the
readings are moving in. But on the same 42 numbers gradient boosting is a
further 2.38 cycles better than linear regression, so the boosting counts too.
All of this is FD001, with the library's settings untouched.

**The shuffled-label row is not a check for leaks.** Trained on labels shuffled
among the training windows, gradient boosting lands where the constant does,
which shows only that the scores need the link between readings and remaining
life. The check that the answers cannot leak is a test,
`tests/test_baselines.py`: it changes every answer in NASA's answer file and
requires every input to every model to stay identical.

**Beside the networks trained the paper's way.** The paper describes no engines
held back and 200 passes over the training data, and this project reads its
reported network as the one left after the last pass (finding F-16). A second
record run,
`baselines_fd001_physics_row_cap_5seed_varsplit_noearlystop_cfg00be26_6733d77_20261003T150939Z`
from a clean `6733d77`, sets the same models beside both networks trained that
way, each from a clean `19dceb5`: the default
(`fd001_physics_row_cap_5seed_varsplit_noearlystop_cfg00be26_19dceb5_20261002T183904Z`)
and the network with the recurrent stage
(`fd001_physics_row_cap_5seed_varsplit_lstm32_noearlystop_cfg00be26_19dceb5_20261003T144539Z`).
With nothing held back there is no split for a seed to change, so every model
is fitted on all 100 training engines, the same ones on every seed. The
constant and linear regression have no random step in their fitting, so they
make the same predictions on every seed. The gradient boosters given the true
labels draw no random sample of windows; their one random step is the order in
which a tree tries the inputs when it chooses each yes-or-no question, which
the seed shuffles and which can only decide between two questions that fit
equally well. It moved their predictions by at most 3.3e-13 cycles between
seeds. So for all of these the ± 0.00 below means no spread at all, not a small
one rounded away. The shuffled labels, the small network's starting values and
the order it sees the windows in, and the windows each of the forest's trees is
fitted on change with the seed by much more: the largest gap between two seeds'
predictions for one engine is 4.9 cycles for the small network, 13 for the
forest and 14 for the booster on shuffled labels. A paired test against a
model with no spread therefore measures only how much retraining moves the
network. (Corrected 2026-10-03, later the same day: this said the boosters have
no random step in their fitting, and that only the shuffled labels, the small
network and the forest still change with the seed. A review found the shuffled
order of the inputs by fitting one booster with two seeds; the figures are
with the comparison in `docs/findings/F-22.md` (finding F-22).)

| Model | Given | RMSE | NASA score | R² | ASUE |
|---|---|---|---|---|---|
| **The default, the weights after 200 passes** | the latest readings, through the paper's graph | **16.82 ± 0.16** | 610 ± 25 | 0.824 ± 0.003 | 12.33 ± 0.60 |
| the same with the paper's recurrent stage, the weights after 200 passes | each sensor's whole window | 16.05 ± 0.98 | 553 ± 85 | 0.839 ± 0.020 | 9.87 ± 0.84 |
| linear regression | the latest readings | 20.39 ± 0.00 | 1040 ± 0 | 0.741 ± 0.000 | 14.25 ± 0.00 |
| network with one layer of 64 | the latest readings | 17.10 ± 0.09 | 799 ± 35 | 0.818 ± 0.002 | 10.61 ± 0.53 |
| random forest | the latest readings | 17.95 ± 0.43 | 1021 ± 127 | 0.799 ± 0.010 | 11.83 ± 0.24 |
| gradient boosting | the latest readings | 16.94 ± 0.00 | 860 ± 0 | 0.821 ± 0.000 | 9.32 ± 0.00 |
| gradient boosting | latest, average and slope | 12.47 ± 0.00 | 237 ± 0 | 0.903 ± 0.000 | 9.41 ± 0.00 |
| gradient boosting | latest and slope | 12.14 ± 0.00 | 227 ± 0 | 0.908 ± 0.000 | 8.85 ± 0.00 |
| linear regression | latest, average and slope | 14.44 ± 0.00 | 352 ± 0 | 0.870 ± 0.000 | 11.31 ± 0.00 |
| gradient boosting trained on shuffled labels | the latest readings | 40.55 ± 0.31 | 19366 ± 1937 | −0.024 ± 0.016 | 26.05 ± 0.55 |
| the average training label, for every engine | nothing | 40.55 ± 0.00 | 18365 ± 0 | −0.024 ± 0.000 | 26.27 ± 0.00 |

**Trained the paper's way, the network with the recurrent stage loses its clear
lead over the flexible ordinary models given the latest readings, and the
booster given each window's latest reading and slope beats it.** Given the same
fourteen numbers, the default is still ahead of the small network (by 0.28
cycles) and the forest (1.14) on every seed, and of gradient boosting (0.12) on
four seeds of five, again by less than the test engines can show: none of the
three intervals over the test engines excludes zero, and on the seeds only the
lead over the forest clears the bar, and only just (`p = 0.000861` against a
bar of 0.000862). Its NASA score is still lower than all three on every seed.
The review of 2026-09-08 measured this question outside the repository on
the default of its day, which held nothing back and gave the figures of this
table's first row, and its rows for gradient boosting and linear regression
match this table to every digit it printed. The network with the recurrent
stage, ahead of the three flexible models by 3.98 to 4.83 cycles with its best
pass restored, is ahead of them by 0.88 to 1.90 trained the paper's way, on
four or five seeds, none of it established. It is still ahead of linear
regression on the same numbers by 4.34 cycles on every seed (`p = 0.00057`,
clearing the bar), though the interval over the test engines no longer
excludes zero once widened to the bar's strictness. Given each window's latest
reading and slope, gradient boosting beats both networks on every seed: the
default by 4.68 cycles and the recurrent-stage network by 3.91 (paired
`p = 0.00085`, interval over the test engines +1.93 to +5.95). Unlike the 1.40
with the best pass restored, that clears the bar, though only just, and on a
row added after the first results were seen. The row drawn up before it, the
booster on all three summaries, beats the recurrent-stage network by 3.58 on
every seed but does not clear the bar (`p = 0.0012`). What moved between the
two procedures is the recurrent-stage network, from 13.23 to 16.05 cycles; the
default moved from 16.50 to 16.82, and no ordinary model by more than 0.71.
(Corrected 2026-10-03, later the same day: the first sentence said "the
booster given the slope", which also fits the booster given the slope alone,
whose lead over the recurrent-stage network does not clear the bar, and
"ordinary models", which took in linear regression; and this paragraph did not
say that the default's lead over the forest clears the bar on the seeds.)

**What it does not show**: that the graph is useless, since no row is this
network with its graph taken out (assumptions A-19 and A-27 measure that); what
either side can do at its best, since the ordinary models are untuned and the
networks' open settings were chosen to land near the paper's figure
(assumption A-22); anything beyond FD001; or which procedure the paper itself
followed, since it never says which weights it reported (finding F-16).
(Updated 2026-10-03: the last item said what the recurrent stage is worth under
the paper's own procedure, before the second table was added.) The full
tables, both kinds of uncertainty, and how the first run differs from a first
measurement made outside the repository on 2026-09-14 and 15 are in
`docs/findings/F-22.md` (finding F-22).

Reproduce the two tables with:

```bash
python scripts/baselines.py --run fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z --also fd001_physics_row_cap_5seed_varsplit_lstm32_cfgba82e3_19dceb5_20261002T210700Z
python scripts/baselines.py --run fd001_physics_row_cap_5seed_varsplit_noearlystop_cfg00be26_19dceb5_20261002T183904Z --also fd001_physics_row_cap_5seed_varsplit_lstm32_noearlystop_cfg00be26_19dceb5_20261003T144539Z
```

---

## 9. How the project is organised

### The directory tree

```
Thesis GNN RUL XAI/
├── README.md                     This file
├── pyproject.toml                Declares the package and the test settings
├── requirements.txt              Dependencies, for pip
├── environment.yml               Dependencies, for conda
│
├── config/
│   ├── fd001.yaml                Settings for the FD001 sub-dataset
│   └── fd001_uncapped.yaml       The same, with the test labels left uncapped
│
├── data/
│   ├── raw/                      The NASA files go here (not in git)
│   └── processed/                Pipeline output (not in git)
│
├── docs/
│   ├── assumptions/              Every choice not taken from the paper, one file each
│   ├── findings/                 What was found about the paper, its data and the results, one file each
│   ├── REFERENCES.md             Full citations, each linked to its copy
│   ├── audits/                   The paper checked against the code, and outside reviews
│   ├── learn/                    The taught pages, for a reader new to this
│   ├── resources/                The dataset, LSTM, GCN, SHAP and GNN-LRP,
│   │                             each explained from its original paper
│   └── references/               A copy of every cited paper (not in git;
│                                 scripts/fetch_references.py gets them)
│
├── notebooks/
│   ├── README.md                 What every notebook shares; read it first
│   ├── 01_preprocessing_fd001.ipynb
│   ├── 02_model_fd001.ipynb
│   ├── temporal/                 One notebook per temporal stage, and a comparison
│   │   ├── README.md
│   │   ├── 01_last_cycle_fd001.ipynb
│   │   ├── 02_mean_window_fd001.ipynb
│   │   ├── 03_window_summary_fd001.ipynb
│   │   ├── 04_lstm_fd001.ipynb
│   │   ├── 05_gru_fd001.ipynb
│   │   ├── 06_raw_window_fd001.ipynb
│   │   └── 07_comparison_fd001.ipynb
│   ├── graph/                    One notebook per sensor graph, and a comparison
│   │   ├── README.md
│   │   ├── 01_physics_fd001.ipynb
│   │   ├── 02_complete_fd001.ipynb
│   │   ├── 03_correlation_fd001.ipynb
│   │   └── 04_comparison_fd001.ipynb
│   └── xai/                      One notebook per explanation method
│       ├── README.md
│       ├── 01_shap_fd001.ipynb
│       ├── 02_occlusion_fd001.ipynb
│       └── 03_walks_fd001.ipynb
│
├── scripts/
│   ├── prepare_data.py           Check or preprocess the raw data
│   ├── train.py                  Train, score, and average over seeds
│   ├── explain.py                Explain a stored run's trained network
│   ├── measure_walks.py          Measure walk explanations over every seed of a stored run
│   ├── baselines.py              Score ordinary models beside a stored run's networks,
│   │                             on the same engines (explained in evaluation/README.md)
│   ├── verify_claims.py          Recompute the figures the assumptions and findings quote
│   ├── check_prose.py            Catch unexplained jargon and codes used as words
│   ├── fetch_references.py       Download the cited papers that are free
│   └── docs_ui.py                Open all of this in a browser
│
├── src/
│   └── thesis_gnn_rul_xai/       All the computation lives here
│       ├── config.py
│       ├── data/
│       ├── graph/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── explain/
│       ├── utils/
│       └── docsite/              The documentation website (computes nothing)
│
└── tests/                        711 tests, counted on 2026-10-04
```

(Corrected 2026-10-03: the last line said 268 tests, the size of the suite in
mid-September, and was found stale when the suite was counted again with
`python -m pytest --collect-only -q`. The count grows whenever a test is
added, so count again before quoting it.)

### Why there is a folder inside a folder: `src/thesis_gnn_rul_xai/`

This looks like one folder too many. It is three separate things stacked, and
only the middle one is about this project.

**`src/` is a safety convention.** Python normally finds code that sits next to
whatever you are running. That is convenient and it hides mistakes: the tests
would pass on your own machine using the loose files in the project folder, and
then fail for anybody who installs the project properly, because some file you
forgot to declare was being picked up silently. Moving the code one level down
into `src/` makes that impossible. Python cannot see it by accident, so
`pip install -e .` has to succeed before anything runs at all. If the tests
pass, it is the *installed* package that passed them.

**`thesis_gnn_rul_xai/` is the name you type in an `import` statement.** The
folder name and the import name are the same thing:

```python
from thesis_gnn_rul_xai.data.smoothing import apply_ewma
#    └─ the folder ───┘ └ folder ┘ └ file ┘   └ function ┘
```

That line is a path through the tree: the folder `src/thesis_gnn_rul_xai/`,
then the folder `data/`, then the folder `smoothing/` whose code is its
`__init__.py`, then the function `apply_ewma` defined inside it. The link between "the code is in `src/`" and
"you import it as `thesis_gnn_rul_xai`" is declared once, in `pyproject.toml`:

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

**Every folder inside is one job.** Most hold one `.py` file per task, plus a
file called `__init__.py`, which is what marks a folder as importable and lists
which names are public. Under `data/` each task has gone one step further: it
is its own folder, holding its code as that folder's `__init__.py` and its own
`README.md` beside it.

| Folder | The question it answers | An example file inside |
|--------|-------------------------|------------------------|
| `data/` | How do raw text files become model input? | `smoothing/`, the noise filter |
| `graph/` | Which sensors are connected to which? | `adjacency.py`, the paper's Table 3 |
| `models/` | What is the network? | `estgnn.py`, the paper's Table 4 |
| `training/` | How is the network fitted to the data? | `run.py`, the paper's Table 6 |
| `evaluation/` | How good is the result? | `metrics.py`, the four measures |
| `explain/` | Which sensor drove this prediction? | `shap.py`, exact Shapley values; `walks.py`, shares along routes through the sensor graph, and `connections.py`, what removing a connection does (neither in the paper) |
| `utils/` | Housekeeping the others share | `paths.py`, `seeding.py` |

`config.py` sits at the top level rather than in a folder because every one of
those needs the settings.

### The notebook and `src/`: the same step in two forms

They are not alternatives, and neither is a copy of the other. Take one step,
the smoothing filter, and look at it in both places.

**In `src/thesis_gnn_rul_xai/data/smoothing/__init__.py`** it is 78 lines: the formula,
the errors raised for impossible settings, and one line saying what each
function does. The reasoning that used to sit alongside it — why each engine
must be filtered separately from its neighbours, why the filter must never look
forwards, where its one setting comes from — is in
`data/smoothing/README.md`. Eight
tests in `tests/test_smoothing.py` check the code, including one that confirms
the filter never reads a value from later in the sequence.

**In the notebook** the same step is one line and a picture:

```python
train_smoothed = apply_ewma(train_frame, sensor_columns, alpha=config.smoothing_alpha)
```

followed by a plot of one engine before and after, and a table showing that the
cycle-to-cycle jitter fell by about 59 %.

So: **`src/` is the machine, and the notebook is a guided tour of the machine
running.** The reason for keeping them apart is that a notebook can be run out
of order. You can execute cell 20 before cell 5, get a plausible-looking wrong
answer, and never be told. Code in a module cannot be run out of order and can
be tested automatically, so a result does not depend on which cells somebody
happened to click. A thesis figure that depends on cell execution order is not
reproducible, and reproducibility is the whole point of the arrangement.

### How to read this code without reading all of it

The explanation and the code are kept in separate places. Every folder under
`src/thesis_gnn_rul_xai/` has a `README.md` holding the reasoning, the
vocabulary, the traps and the meaning of every argument. The `.py` files
alongside carry one line of description per function, plus the `ASSUMPTION`
comments that mark each judgement call at the point where it is made.

Counting a line as *code* when it is neither blank, nor a comment, nor part of
a docstring, across the package and the two scripts:

| | lines | share |
|---|-------|-------|
| executable code | 2 016 | 57 % |
| explanation (docstrings and comments) | 962 | 27 % |
| blank | 577 | 16 % |

So the thing to be read is about **two thousand lines**, and it is spread
thinly: the largest single file has 217 lines of code, and several of the most
important have almost none. `data/smoothing/` is 78 lines of which 44 are
code; `data/labeling/` is 41 lines of which 22 are code. The reasons those
operations are what they are — which is what a thesis has to defend — fill
5 374 further lines across the fifteen `README.md` files under `src/`: one per
folder, and one for each of the seven files under `data/`.

Those figures were measured with a token-based counter, not by eye, and they
moved when the explanation was pulled out of the code: the same package
previously measured 2 250 lines of explanation inside the `.py` files. The code
itself did not change.

**Do not read it in file order.** The package has three layers, and nothing in a
layer depends on anything below it, so this order never requires you to hold an
unexplained name in your head:

**Layer 1 — no internal dependencies.** Read these first; each stands alone.

| File | What it is |
|------|------------|
| `config.py` | every setting, with the reason for each value |
| `data/sensors/` | the 21 sensors, and which 14 the paper keeps |
| `data/labeling/` | the piecewise-linear target (8 lines of code) |
| `models/layers.py` | one graph convolution (43 lines of code) |
| `utils/` | paths, seeding, logging — housekeeping, read only if curious |

**Layer 2 — uses layer 1.**

| File | What it is |
|------|------------|
| `data/loading/` | the three text files become one table |
| `data/smoothing/` | the noise filter (30 lines of code) |
| `data/normalization/` | operating conditions and rescaling |
| `data/windowing/` | histories become fixed-size examples |
| `graph/adjacency.py` | the sensor graph and its normalisation |
| `models/estgnn.py` | the network of Table 4 |
| `evaluation/metrics.py` | the four measures |

**Layer 3 — the three that put it together.** Each is a sequence of calls to
the layers above, so read them last and they will contain no surprises.

| File | What it is |
|------|------------|
| `data/pipeline/` | the eight preprocessing steps in order |
| `training/run.py` | the training loop |
| `explain/shap.py` | the Shapley values |

**Four shortcuts that save more time than they cost.**

1. **Start with the notebooks, not with `src/`.** They run the whole chain on
   real data and show the numbers changing at every step. An hour there makes
   `src/` read like a reference rather than a puzzle.
2. **Read the folder READMEs first, and no code at all.** There are eight, one
   per folder under `src/thesis_gnn_rul_xai/`. Each says what its step does,
   why it is done that way, and what the paper says about it. Reading only
   those gives you the entire argument without a line of code.
3. **When you want to know what a function really does, read its test.** The
   tests are written as statements about behaviour:
   `test_the_filter_only_looks_backwards`,
   `test_three_layers_cannot_reach_four_hops`,
   `test_the_prediction_can_never_exceed_the_ceiling`. A test is usually
   shorter than the function and tells you what matters about it.
4. **Follow the shapes.** Almost every step is a change of shape, and section 7
   lists them all. If you can say what shape the data has after each step —
   `(20631, 26)` to `(17731, 14, 30)` to `(100,)` — you understand the pipeline,
   whatever the code looks like.

**If you have one hour**, read notebook 01 end to end. **If you have two**, add
notebook 02. **If you have three**, add `src/thesis_gnn_rul_xai/data/README.md`
and `models/README.md`, in that order. **If you have a day**, add
`docs/assumptions/`, which is where every judgement call this project makes
is defended.

### Which file should I open?

| If you want to... | Open |
|-------------------|------|
| Read all of this in a browser, with figures | `python scripts/docs_ui.py` |
| Understand the method from nothing, in twelve pages | `docs/learn/`, or the **Learn** section of that site |
| Understand what happens to the data, step by step | `notebooks/01_preprocessing_fd001.ipynb` |
| Understand the model and the training | `notebooks/02_model_fd001.ipynb` |
| See what each sensor's 30-cycle window is turned into before the graph layers, and what each way of doing it scores | `notebooks/temporal/` |
| See what the sensor graph does, and whether the paper's beats two others | `notebooks/graph/` |
| Understand which sensors a prediction came from, and how that changes over an engine's life | `notebooks/xai/01_shap_fd001.ipynb` |
| Know exactly how one step is computed | the matching file in `src/thesis_gnn_rul_xai/` |
| Know *why* a step is computed that way | the `README.md` in the same folder |
| Know why a value was chosen when the paper is silent | `docs/assumptions/` |
| Run the whole thing without reading anything | `scripts/prepare_data.py`, then `scripts/train.py` |
| Check that a step really does what it claims | the matching file in `tests/` |

---

## 10. Testing

```bash
pytest                     # everything
pytest -v                  # with the name of each test
pytest tests/test_normalization.py
pytest tests/test_examples.py  # one worked example of each function
ruff check .               # style, unused imports, dead code
```

The tests check that the code does what it claims. A separate script checks
that **the entries in `docs/assumptions/` and `docs/findings/` still describe
the code**, which is a different
kind of rot and the one this project has actually suffered from:

```bash
python scripts/verify_claims.py --fast
```

It recomputes the figures the assumptions and findings quote — the sensor graph's edge
counts, `P15`'s two levels, how many test labels the ceiling touches, the
preprocessing-order measurement on FD002 — prints each beside the recorded
value, and exits non-zero on a disagreement. On 2026-10-03 that was **76
claims** in seconds, including the tables of the graph built from correlations
(assumption A-27). Dropping `--fast` adds two checks that train a network to
regenerate the tables of Shapley values the log quotes (assumption A-20 and
finding F-05), which takes minutes rather than seconds. On 2026-09-14, when the
fast check had 62 claims, that brought it to **91**; the full check has not
been counted since. Every run writes its results and its commit into
`reports/`, so a verification is itself reproducible. (Corrected 2026-10-03:
this gave 62 as the fast check's size, without a date. 62 was its count on
2026-09-14, and checks added since bring it to 76. Found by a review that ran
the check while reading this file's Status line.)

**On 2026-09-14 all 91 pass**, the trained ones at `8b487ab-dirty`. They did
not that morning. The two trained checks read today's default settings. The
entries they check were measured on 2026-08-31, under that day's
learning-rate schedule and with nothing held back. Trained under the old
schedule, the code reproduced every figure exactly. Under the current one, four
figures moved. Both checks now train under the settings their entries name, and
check each entry's table under both schedules. Both entries carry a correction
saying what changed. This script exists because entries drift: before it was
written, five of the entries it checks had already drifted from the code. The eleven newest come
from the audit of 2026-08-31 — see
[`docs/audits/AUDIT_2026-08-31_paper_against_code.md`](docs/audits/AUDIT_2026-08-31_paper_against_code.md),
which checked every equation, table and figure of the paper against this code
and records the agreements as well as the six new entries it produced.

Every push also runs `ruff check` and `pytest` on GitHub Actions, against
Python 3.10 and 3.13 — see `.github/workflows/checks.yml`. The runner starts
from a fresh clone, so it has **neither** the NASA data **nor** any stored run
in `reports/` — both are deliberately outside git — and **172 tests pass there
while 87 skip themselves**. That is six more skips than a developer machine
without the data alone, which gets 178 and 81: the extra six are the tests
about stored runs, which skip when there are none. A green run means
nothing that can be checked *without* the dataset is broken; it does not mean
the numbers in [section 8](#8-results-so-far) still hold, which needs the data
and has to be checked locally.

With the data, the stored runs and the cited papers, 711 tests are collected
and 709 pass, measured on 2026-10-04; one is skipped because no stored
run is trained on a control graph yet, and the one failure is a stored run of the
sensitivity study whose folder name and recorded commit disagree, written up in
`docs/PENDING.md`. (Until 2026-10-03 this said all 268 pass, measured on
2026-09-15.)

The suite covers each preprocessing step, the sensor graph, the network, the
training loop, the four measures, the Shapley values, occlusion and the walk
shares, loading a stored run back, and the pipeline end to end. The model, training and
explanation tests need PyTorch; without it they are not collected at all rather
than failing. On 2026-09-14, when the suite had 260 tests, that left 195 of
them collected: 191 passed, three skipped and one, since fixed, failed. That
count has not been re-measured since, and the suite has grown by several
hundred tests. (Until 2026-10-03 this said "since eight tests were added".)

`tests/test_examples.py` and `tests/test_examples_torch.py` hold one small
worked example of each function — the kind of thing that would once have gone
in a docstring. They are the quickest way to see what a function actually takes
and returns.

Some of the tests check properties rather than outputs, because a wrong
answer in this domain still looks plausible. Seven deserve special mention:

*No information may reach the model that would not be available in practice:*

- `test_normalization.py::test_test_data_never_influences_the_scaling`
- `test_pipeline.py::test_validation_engines_never_influence_the_scaling`
- `test_smoothing.py::test_the_filter_only_looks_backwards`
- `test_pipeline.py::test_training_and_validation_engines_never_overlap`

*The architecture and the attribution must match their definitions:*

- `test_models.py::test_parameter_counts_match_table_4` — every weight matrix
  has the shape the paper prints.
- `test_models.py::test_three_layers_cannot_reach_four_hops` — message
  passing follows the sensor graph and nothing else.
- `test_shap.py::test_efficiency_holds_to_machine_precision` — the fourteen
  attributions plus the baseline reconstruct the prediction exactly, which is
  the property that makes them Shapley values rather than a plausible ranking.

If any of these ever fails, results from this code are not trustworthy,
regardless of how good they look.

---

## 11. Rules this project follows

1. **Statistics come from training data only.** Every minimum, maximum, mean,
   standard deviation and cluster centre is computed from the training engines
   and then applied unchanged to the validation and test engines. The
   validation engines are held out *before* those statistics are computed, so a
   held-out engine can never define the scale of a sensor.
2. **Splits are made by engine, never by window.** Two neighbouring windows
   from one engine share 29 of their 30 cycles. Splitting windows at random
   would place nearly identical examples on both sides.
3. **Filters only look backwards.** No step may use a reading from the future
   of the sequence it is processing.
4. **Every assumption is written down.** If a value is not in the paper, it is
   in [`docs/assumptions/`](docs/assumptions/README.md) with its justification, and
   marked `ASSUMPTION` at the point of use in the code.
5. **Randomness is seeded and reported.** A single run is repeatable. Because
   one run is not evidence, experiments are intended to be repeated across
   several seeds with the mean and standard deviation reported.
6. **Only real measurements are used.** The project generates no stand-in
   data. Every number it produces traces back to the NASA files, and when
   those files are absent the code says so instead of substituting something
   that merely has the right shape.

---

## 12. Known limitations

**The LSTM stage is not the default.** The paper places one LSTM per sensor
before the graph layers, but never states its size, its depth, or how it
reduces a 30-cycle window to the single number its Table 4 requires. Those three
are now choices this project made (assumption A-26) rather than a reason not to
build it: `models/temporal/` holds the stage, with the fourteen dedicated
networks the paper does state, and `--temporal lstm` trains it. What has **not**
changed is that a run without it compares against the "w/o LSTM" row, and that
is still what the default does (assumption A-11). The paper publishes this exact
configuration itself, as the "ESTGNN w/o LSTM" row of its ablation table.
Comparisons must be made against that row.

**Only FD001 is handled so far.** The per-condition *scaler* is written and
tested, but the pipeline is not yet ready for FD002 or FD004. The paper's own
step order (Fig. 1) smooths in raw physical units before the per-condition
scaling removes the jumps between flight conditions, which blends readings from
different conditions together. On FD001 there is one condition and the order is
harmless; on FD002 and FD004 it collapses the wear signal. That order was once
logged as a choice of this project and withdrawn when Fig. 1 turned out to state
it; the measurement on FD002 is kept in the same entry (assumption A-08). FD003
has a single operating condition and should work as-is, but has not been run.

**The published numbers have not been reproduced, and exact agreement is not
a realistic target.** The paper's preprocessing is not fully described, its
results come from single runs with no seeds or error bars, and its reported
software environment lists no deep learning framework at all. This code now
produces its own numbers and compares them with the published ones in
[section 8](#8-results-so-far); the comparison differs in both directions and
the reasons are documented as assumptions rather than left as a gap.
Understanding and recording the difference is the achievable goal.

**Each choice has been measured under one configuration at most.** Many of the
choices in [`docs/assumptions/`](docs/assumptions/README.md) have now been varied
over five seeds: the input scaling, the smoothing, the label ceiling, the
learning-rate schedule and early stopping (finding F-10), the held-back engines
(assumption A-06), the rescaling of the graph (assumption A-21), and what
replaces the paper's graph (assumptions A-19 and A-27). But each was varied
under the configuration of its day, and this project's comparisons have
reversed when that configuration changed, so none of them is settled for every
configuration. The stand-in for a removed sensor (assumption A-20) changes no
trained network and no test error, only what an explanation reports. Three
stand-ins have been compared on five networks under one configuration, the
default with no training engine held back, and there they agree on which four
sensors lead and on `P15` coming last. (Corrected 2026-10-03: from 2026-09-14
this paragraph listed the stand-in among the choices varied over five seeds,
when it had been compared on one network; the five were measured on
2026-10-03. Found by a review that day that set this paragraph beside the
five-network results then added to the record of that choice.) The bias in the
graph layers (assumption A-12) has been varied once, in a study that changed one
choice at a time around the network with the paper's recurrent stage rather than
the default, five seeds each: adding the bias raised the test error by 0.282
cycles, which five seeds cannot distinguish from zero (p = 0.648; finding F-19).
Three choices have not been varied at all: the window stride (assumption A-05),
the starting values of the graph and dense layers (assumption A-14), and the
starting values of the recurrent stage (assumption A-34). The stride would take
one five-seed run of `scripts/train.py` with a copy of `config/fd001.yaml` that
changes `window_stride`; the starting values first need an option that sets
them, which the code does not have. (Corrected 2026-10-02: this said the bias
in the graph layers had not been varied, which stopped being true when that
study ran, and that each choice it listed would take one run of
`scripts/train.py --seeds 5`, which has no option for starting values. Found
while the recurrent stage's starting values were written up.)

---

## 13. References

Full citations, with a note on what each is used for and a link to its copy
in `docs/references/`, are in [`docs/REFERENCES.md`](docs/REFERENCES.md). The
copies are not committed; `python scripts/fetch_references.py` downloads the
ones that are free. The works cited above are:

- Corso et al. (2024), *Nature Reviews Methods Primers* — graph neural networks
- Hochreiter & Schmidhuber (1997), *Neural Computation* — LSTM
- Kapoor & Narayanan (2023), *Patterns* — data leakage
- Kipf & Welling (2017), *ICLR* — graph convolutional networks
- Li et al. (2018), *Reliability Engineering & System Safety* — CNN baseline
- Lundberg & Lee (2017), *NIPS* — SHAP
- NASA Prognostics Center of Excellence (2008), `readme.txt`: the description that comes with the C-MAPSS files
- Saxena et al. (2008), *PHM* — the C-MAPSS dataset
- Shapley (1953), *Contributions to the Theory of Games* — the Shapley value
- Wen et al. (2026), *Aerospace Science and Technology* — the paper re-implemented

---

*Author: Amin Darian. Master's thesis, Automation and Control Engineering,
RPTU Kaiserslautern-Landau, in collaboration with Fraunhofer Karlsruhe.*
