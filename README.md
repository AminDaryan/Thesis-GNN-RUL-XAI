# Thesis GNN RUL XAI

Preprocessing pipeline for predicting how much life is left in a jet engine,
built as the software foundation for a master's thesis on **explainable**
graph-based deep learning for predictive maintenance.

**Current stage:** preprocessing, the graph network of the paper's Table 4, the
training loop of its Table 6, and the four evaluation measures of its
Section 3.3. On FD001 this reaches **RMSE 13.45 +/- 0.20** across five seeds,
against the 14.93 the paper reports for its own "ESTGNN w/o LSTM" ablation
(see finding F-06 for what that comparison does and does not establish).
The dynamic SHAP analysis of its Section 2.4 is implemented, exactly rather
than by sampling.

**Status:** 157 automated tests pass (107 without PyTorch installed). The notebook has been executed end to end
on the real FD001 files and all 14 of its internal checks pass.

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
13. [What comes next](#13-what-comes-next)
14. [References](#14-references)

---

## 1. What this project is about

A jet engine wears out gradually. If an airline can estimate how many more
flights an engine will survive, it can plan maintenance before a failure
happens, instead of reacting afterwards. Estimating that number from sensor
readings is the task this project works on.

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

**RUL — Remaining Useful Life.** The number of flight cycles an engine will
still run before it fails. This is the quantity being predicted. One *flight
cycle* is one complete take-off, flight, and landing; it is the unit of time
used everywhere in this project.

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
player's fair share. It is the only way of dividing credit that satisfies a
small set of natural fairness conditions at the same time.

**SHAP — SHapley Additive exPlanations.** The application of the Shapley value
to machine learning (Lundberg & Lee, 2017). The "players" become the model's
input features and the "result" becomes its prediction. Each feature receives a
number saying how much it pushed the prediction up or down, and those numbers
add up exactly to the difference between this prediction and the average
prediction. In this project the players will be the 14 sensors.

**Data leakage.** When information that should be unavailable at prediction
time reaches the model during training. It makes results look better than they
are and is a leading cause of published findings that cannot be reproduced
(Kapoor & Narayanan, 2023). Two rules in [section 11](#11-rules-this-project-follows)
exist to prevent it.

**Normalisation (rescaling).** Putting different measurements onto a comparable
numerical scale, so that a temperature near 1400 does not dominate a ratio near
8 simply because it is a larger number.

**Time window.** A short, fixed-length slice of an engine's history, for
example 30 consecutive cycles. Windows are what the model actually receives,
because histories have different lengths and a network needs a fixed input size.

**Ablation.** An experiment that removes one part of a method and measures
what the rest can do without it. If accuracy barely changes, that part was
not doing much. It is how a paper shows which of its ingredients matter.

**The four measures used to score a prediction.** All four are computed for
each of the 100 test engines and then combined.

* **RMSE — root mean squared error.** The typical size of the error, in
  flight cycles. It treats a prediction 10 cycles too high exactly like one
  10 cycles too low. This is the number most papers quote.
* **NASA score.** The official score of the 2008 competition this dataset
  comes from. Unlike RMSE it is **asymmetric**: predicting more life than an
  engine has is punished far harder than predicting less, because the first
  mistake leaves a failing engine flying and the second only services it
  early. The penalty grows exponentially, so one badly wrong engine can
  outweigh ninety-nine good ones. Lower is better, and it is a sum rather
  than an average, so it can only be compared between runs on the same
  sub-dataset.
* **R² — coefficient of determination.** The fraction of the variation in
  the true labels that the model accounts for. 1 is perfect; 0 is no better
  than always guessing the average.
* **ASUE — Average Safe Underestimation Error.** The average size of the
  error, counted only over the engines the model was *pessimistic* about.
  Underestimating is the safe direction, so this measures the cost of being
  too cautious: servicing engines that still had life left in them.

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

The paper does not specify everything needed to rebuild it. Eleven separate
choices had to be made here rather than read from the paper, including the
value of the label ceiling, the entire input rescaling procedure, and the
configuration of the LSTM networks. Every one is documented, with its
justification and its likely effect, in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).

Nothing in this repository should be described as "reproducing ESTGNN".
Results from this code are comparable with the **"ESTGNN w/o LSTM"** row of the
paper's own ablation table (their Table 11), because the LSTM stage is
deliberately omitted here. See assumption A-11.

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

The NASA files are **not** included in this repository. NASA distributes them
under its own terms, and a research project should cite the original source
rather than ship a copy that may fall out of date.

1. Open the NASA Prognostics Center of Excellence data repository:
   <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
2. Download **Turbofan Engine Degradation Simulation Data Set**.
3. Copy these three files into `data/raw/`:
   ```
   data/raw/train_FD001.txt
   data/raw/test_FD001.txt
   data/raw/RUL_FD001.txt
   ```
4. Verify them:
   ```bash
   python scripts/prepare_data.py --check
   ```
   This reports whether each file is present and whether it has the row and
   engine counts published for the official release (20 631 training rows over
   100 engines, and 100 test engines, for FD001).

### Keeping the dataset somewhere else

If you already have the files outside this project, point the script at them
instead of copying:

```bash
python scripts/prepare_data.py --run --data-dir /path/to/CMAPSSData
```

The same applies to `run_preprocessing(config, data_dir=...)` in Python.

> **There is no substitute for the real files.** This project does not
> generate stand-in data, on purpose. A pipeline that appears to work on
> invented signals tells you nothing about whether it works on engines, and a
> number computed from invented signals is not a result. If the files are
> missing, every command and every test says so and points at the download.

---

## 6. Running the pipeline

### From the notebook (start here)

```bash
jupyter lab notebooks/01_preprocessing_fd001.ipynb
```

The notebook explains every step, shows the intermediate tables and figures,
and verifies each stage as it goes. It is the intended entry point for a reader
who wants to understand what happens and why.

> The notebook is shipped with its outputs already filled in, so it can be read
> without running anything. Those outputs were produced on the **real FD001
> files**, which were verified against the published release (20 631 training
> rows, 100 training engines, 100 test engines). The data files themselves are
> not in this repository; see [section 5](#5-getting-the-data).

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
one. `--graph complete` runs the "without physics prior" ablation, and
`--no-early-stopping` reproduces the paper's fixed 200 epochs.

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
each carrying its own 30-cycle history.

---

## 7. What the pipeline does, step by step

| # | Step | What happens | Where |
|---|------|--------------|-------|
| 1 | Load | Read the three text files; work out the Remaining Useful Life at every row | `data/loading.py` |
| 2 | Diagnose | Find sensors that never change, and compare with the paper's selection | `data/sensors.py` |
| 3 | Select | Keep the paper's 14 sensors, in the order its adjacency matrix expects | `data/sensors.py` |
| 4 | Cap labels | Hold the label at 125 while the engine is healthy | `data/labeling.py` |
| 5 | Smooth | Remove measurement noise with a backward-looking filter | `data/smoothing.py` |
| 6 | Split | Hold out 20 % of engines for validation, **before any statistic is learned** | `data/windowing.py` |
| 7 | Rescale | Put all sensors on a comparable scale, using training engines only | `data/normalization.py` |
| 8 | Window | Cut histories into 30-cycle examples | `data/windowing.py` |
| — | Graph | Build the 14-node sensor graph from the paper's Table 3 | `graph/adjacency.py` |

Those eight steps turn the raw files into arrays. Four more stages turn the
arrays into a result:

| # | Stage | What happens | Where |
|---|-------|--------------|-------|
| 9 | Build the network | Three graph convolution layers of widths 32, 64 and 32, then two fully connected layers ending in a sigmoid — the paper's Table 4, transcribed layer by layer | `models/estgnn.py` |
| 10 | Train | Adam, 200 epochs, batch size 32, learning rate 1e-3 decaying to 1e-4, mean squared error — the paper's Table 6 | `training/loop.py` |
| 11 | Score | RMSE, the NASA score, R² and ASUE, on the 100 test engines | `evaluation/metrics.py` |
| 12 | Explain | Exact Shapley values for the 14 sensors, recomputed at every cycle — the paper's Section 2.4 | `explain/shap.py` |

Each module's docstring explains the reasoning behind its step in full. Start
with `data/normalization.py` and `data/windowing.py`; those two contain the
decisions that matter most for the validity of the results.

---

## 8. Results so far

Every figure below is the mean over **five seeds**, with the standard deviation
across those seeds. One seed is not evidence, and the training script says so
when it is given only one.

### On FD001

| Configuration | RMSE | NASA score | R² | ASUE |
|---------------|------|-----------|-----|------|
| Physics graph, whole 30-cycle window (the default) | **13.45 ± 0.20** | 312 ± 11 | 0.887 ± 0.003 | 8.85 ± 0.36 |
| Physics graph, last reading only | 19.76 ± 0.57 | 750 ± 64 | 0.757 ± 0.014 | 15.19 ± 0.97 |
| Fully connected graph, whole window | 19.17 ± 0.45 | 743 ± 53 | 0.771 ± 0.011 | 13.44 ± 0.27 |

Reproduce the first row with:

```bash
python scripts/train.py --seeds 5
```

### How that compares with the paper

| Paper's variant | RMSE | Where |
|-----------------|------|-------|
| ESTGNN, full model | 11.78 | Tables 7–10 |
| **ESTGNN w/o LSTM** | **14.93** | Table 11 |
| ESTGNN w/o GCN | 16.42 | Table 11 |
| ESTGNN w/o physics prior | 12.34 | Table 11 |

**Only one of those numbers is comparable with this code.** The per-sensor LSTM
is deliberately not implemented (see [section 12](#12-known-limitations) and
assumption A-11), so the matching row is "w/o LSTM", and the comparison is
13.45 ± 0.20 against 14.93. The paper's ablation table reports RMSE only, which
is why the other three columns are blank for it.

### What that comparison does and does not show

This code comes out **1.5 cycles better** than the published figure. That is not
a reproduction, and it should not be presented as one. Being better than the
reference is itself evidence that something differs, and the differences are all
written down rather than mysterious: the input rescaling is substituted from a
different paper (A-03), early stopping is an addition (A-18), capping the test
labels can only help (A-02), and the target scaling, weight initialisation and
learning-rate schedule are all unstated by the paper (A-13, A-14, A-17).

There is also a sharper result. Assumption A-07 records that with the LSTM
removed there are exactly two defensible things to feed the graph layers, and
both have now been run. **The paper's 14.93 falls between them:** 13.45 with the
whole window, 19.76 with the last reading only. Both gaps are far larger than
the seed-to-seed spread, so whatever the paper's ablation fed its graph layers,
it was neither of the two things its own text supports.

Finding F-06 in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) records this in
full, together with F-04 on the physics graph and F-05 on what SHAP says.

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
│   └── fd001.yaml                Settings for the FD001 sub-dataset
│
├── data/
│   ├── raw/                      The NASA files go here (not in git)
│   └── processed/                Pipeline output (not in git)
│
├── docs/
│   ├── ASSUMPTIONS.md            Every choice not taken from the paper
│   └── REFERENCES.md             Full citations
│
├── notebooks/
│   └── 01_preprocessing_fd001.ipynb
│
├── scripts/
│   ├── prepare_data.py           Check or preprocess the raw data
│   └── train.py                  Train, score, and average over seeds
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
│       └── utils/
│
└── tests/                        157 tests
```

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
then the folder `data/`, then the file `smoothing.py`, then the function
`apply_ewma` defined inside it. The link between "the code is in `src/`" and
"you import it as `thesis_gnn_rul_xai`" is declared once, in `pyproject.toml`:

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

**Every folder inside is one job**, and holds one `.py` file per task. Each
also holds a file called `__init__.py`, which is what marks a folder as
importable and lists which names are public.

| Folder | The question it answers | An example file inside |
|--------|-------------------------|------------------------|
| `data/` | How do raw text files become model input? | `smoothing.py`, the noise filter |
| `graph/` | Which sensors are connected to which? | `adjacency.py`, the paper's Table 3 |
| `models/` | What is the network? | `estgnn.py`, the paper's Table 4 |
| `training/` | How is the network fitted to the data? | `loop.py`, the paper's Table 6 |
| `evaluation/` | How good is the result? | `metrics.py`, the four measures |
| `explain/` | Which sensor drove this prediction? | `shap.py`, exact Shapley values |
| `utils/` | Housekeeping the others share | `paths.py`, `seeding.py` |

`config.py` sits at the top level rather than in a folder because every one of
those needs the settings.

### The notebook and `src/`: the same step in two forms

They are not alternatives, and neither is a copy of the other. Take one step,
the smoothing filter, and look at it in both places.

**In `src/thesis_gnn_rul_xai/data/smoothing.py`** it is about 170 lines: the
formula, an explanation of why each engine must be filtered separately from its
neighbours, the errors raised for impossible settings, and a docstring giving a
worked example. Eight tests in `tests/test_smoothing.py` check it, including
one that confirms the filter never reads a value from later in the sequence.

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

There is a lot of text in `src/`, and most of it is not code. Measured across
the package and the two scripts:

| | lines | share |
|---|-------|-------|
| executable code | 1 970 | 36 % |
| explanation (docstrings and comments) | 2 690 | 49 % |
| blank | 839 | 15 % |

So the thing to be read is about **two thousand lines**, and it is spread thinly:
the largest single file has 219 lines of code, and several of the most important
have almost none. `data/smoothing.py` is 176 lines of which **30** are code.
`data/labeling.py` is 132 lines of which **8** are code. The explanation is long
because the *reasons* are what a thesis has to defend; the operations themselves
are small.

**Do not read it in file order.** The package has three layers, and nothing in a
layer depends on anything below it, so this order never requires you to hold an
unexplained name in your head:

**Layer 1 — no internal dependencies.** Read these first; each stands alone.

| File | What it is |
|------|------------|
| `config.py` | every setting, with the reason for each value |
| `data/sensors.py` | the 21 sensors, and which 14 the paper keeps |
| `data/labeling.py` | the piecewise-linear target (8 lines of code) |
| `models/layers.py` | one graph convolution (43 lines of code) |
| `utils/` | paths, seeding, logging — housekeeping, read only if curious |

**Layer 2 — uses layer 1.**

| File | What it is |
|------|------------|
| `data/loading.py` | the three text files become one table |
| `data/smoothing.py` | the noise filter (30 lines of code) |
| `data/normalization.py` | operating conditions and rescaling |
| `data/windowing.py` | histories become fixed-size examples |
| `graph/adjacency.py` | the sensor graph and its normalisation |
| `models/estgnn.py` | the network of Table 4 |
| `evaluation/metrics.py` | the four measures |

**Layer 3 — the three that put it together.** Each is a sequence of calls to
the layers above, so read them last and they will contain no surprises.

| File | What it is |
|------|------------|
| `data/pipeline.py` | the eight preprocessing steps in order |
| `training/loop.py` | the training loop |
| `explain/shap.py` | the Shapley values |

**Four shortcuts that save more time than they cost.**

1. **Start with the notebook, not with `src/`.** It runs the whole preprocessing
   chain on real data and shows the numbers changing at every step. An hour
   there makes `src/` read like a reference rather than a puzzle.
2. **Read every module's docstring first, and nothing else.** The docstring at
   the top of each file says what the step does, why it is done that way, and
   what the paper says about it. Reading only those — about forty minutes for
   the whole package — gives you the entire argument without a line of code.
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

**If you have one hour**, read the notebook end to end. **If you have three**,
add the module docstrings of `data/normalization.py`, `data/windowing.py`,
`graph/adjacency.py` and `models/estgnn.py`, in that order. **If you have a
day**, add `docs/ASSUMPTIONS.md`, which is where every judgement call this
project makes is defended.

### Which file should I open?

| If you want to... | Open |
|-------------------|------|
| Understand what happens to the data, step by step | `notebooks/01_preprocessing_fd001.ipynb` |
| Know exactly how one step is computed | the matching file in `src/thesis_gnn_rul_xai/` |
| Know why a value was chosen when the paper is silent | `docs/ASSUMPTIONS.md` |
| Run the whole thing without reading anything | `scripts/prepare_data.py`, then `scripts/train.py` |
| Check that a step really does what it claims | the matching file in `tests/` |

---

## 10. Testing

```bash
pytest                     # everything
pytest -v                  # with the name of each test
pytest tests/test_normalization.py
pytest --doctest-modules src   # only the examples inside docstrings
```

The suite covers each preprocessing step, the sensor graph, the network, the
training loop, the four measures, the Shapley values, and the pipeline end to
end. The model, training and explanation tests need PyTorch; without it they
are skipped rather than failed, and the remaining 107 still run.

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
   in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) with its justification, and
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

**The LSTM stage is not implemented.** The paper places one LSTM per sensor
before the graph layers, but never states its size, its depth, or how it
reduces a 30-cycle window to the single number its Table 4 requires. Rather
than invent those details, this project omits the stage. The paper publishes
this exact configuration itself, as the "ESTGNN w/o LSTM" row of its ablation
table. Comparisons must be made against that row. See assumption A-11.

**Only FD001 is handled so far.** The per-condition *scaler* is written and
tested, but the pipeline is not yet ready for FD002 or FD004. The paper's own
step order (Fig. 1) smooths in raw physical units before the per-condition
scaling removes the jumps between flight conditions, which blends readings from
different conditions together. On FD001 there is one condition and the order is
harmless; on FD002 and FD004 it collapses the wear signal. See A-08. FD003 has a
single operating condition and should work as-is, but has not been run.

**The published numbers have not been reproduced, and exact agreement is not
a realistic target.** The paper's preprocessing is not fully described, its
results come from single runs with no seeds or error bars, and its reported
software environment lists no deep learning framework at all. This code now
produces its own numbers and compares them with the published ones in
[section 8](#8-results-so-far); the comparison differs in both directions and
the reasons are documented as assumptions rather than left as a gap.
Understanding and recording the difference is the achievable goal.

**Most of the assumptions have not been tested for sensitivity.** Each entry
in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) states how much it is
expected to matter, but for most that expectation is an argument rather
than a measurement. Four have actually been measured — A-07 and A-19 in
finding F-06, A-20 and A-21 in their own entries. Nine more are settings that
could be varied and have not been: A-01 (the ceiling), A-03 (the scaling method),
A-04 (the smoothing strength), A-05 (the stride), A-06 (the validation
fraction), A-12 (the bias), A-14 (the initialisation), A-17 (the schedule)
and A-18 (early stopping). Each would take one run of
`scripts/train.py --seeds 5` to settle.


---

## 13. What comes next

1. ~~Run this pipeline on the real FD001 files and compare the sensor
   diagnosis with the paper's stated selection.~~ **Done.** The comparison
   produced assumption A-10 and findings F-01 to F-03.
2. ~~Implement the graph convolution model from Table 4 of the paper: three
   layers of sizes 32, 64 and 32, then two fully connected layers.~~ **Done**,
   in `src/thesis_gnn_rul_xai/models/`. `ESTGNN.layer_summary()` prints the
   architecture in the same form as Table 4 for checking against the paper.
   The training loop of Table 6 is in `src/thesis_gnn_rul_xai/training/`.
3. Reproduce a simple published baseline, as an independent check that the
   pipeline itself is sound. Li et al. (2018) report an error of about 12.6
   on FD001; a pipeline that cannot approach a known baseline has a problem
   no amount of model tuning will fix. **Still outstanding.** The graph
   model's own result is in [section 8](#8-results-so-far).
4. ~~Add the evaluation measures from the paper: the NASA scoring function, root
   mean squared error, the coefficient of determination, and average safe
   underestimation error.~~ **Done**, in `src/thesis_gnn_rul_xai/evaluation/`.
   Two of the paper's printed formulas could not be used as written: its
   Equations 20 and 21 disagree with each other about which direction of error
   is dangerous, and its Equation 23 is not the coefficient of determination.
   Both departures are recorded as A-15 and A-16.
5. Run the remaining ablations. "Without physics prior" is done and is
   reported as finding F-04: removing the physics graph costs 5.7 cycles of
   RMSE here, against the 0.56 the paper reports, most likely because the
   omitted LSTM would otherwise supply the same information by another route.
   "Without GCN" still needs a plain network on the same features.
6. ~~Add SHAP~~ **Done**, in `src/thesis_gnn_rul_xai/explain/`. With only 14
   sensors the Shapley values of Equation 15 can be enumerated exactly, so no
   sampling approximation is used and a disagreement with the paper cannot be
   blamed on estimation noise. The first run is recorded as finding F-05.
7. Move beyond node attribution to graph-native explanations that describe
   edges and subgraphs rather than single sensors — the actual research gap
   this thesis addresses. SHAP divides credit among the 14 sensors and cannot,
   by construction, express a fact like this one: every path from the fan
   sensors to the core sensors runs through `T24`, the graph's only
   articulation point.

---

## 14. References

Full citations, with a note on what each is used for, are in
[`docs/REFERENCES.md`](docs/REFERENCES.md). The works cited above are:

- Corso et al. (2024), *Nature Reviews Methods Primers* — graph neural networks
- Hochreiter & Schmidhuber (1997), *Neural Computation* — LSTM
- Kapoor & Narayanan (2023), *Patterns* — data leakage
- Kipf & Welling (2017), *ICLR* — graph convolutional networks
- Li et al. (2018), *Reliability Engineering & System Safety* — CNN baseline
- Lundberg & Lee (2017), *NeurIPS* — SHAP
- Saxena et al. (2008), *PHM* — the C-MAPSS dataset
- Shapley (1953), *Contributions to the Theory of Games* — the Shapley value
- Wen et al. (2026), *Aerospace Science and Technology* — the paper re-implemented

---

*Author: Amin Darian. Master's thesis, Automation and Control Engineering,
RPTU Kaiserslautern-Landau, in collaboration with Fraunhofer Karlsruhe.*
