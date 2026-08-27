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

**Status:** 155 automated tests pass (107 without PyTorch installed). The notebook has been executed end to end
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
8. [How the project is organised](#8-how-the-project-is-organised)
9. [Testing](#9-testing)
10. [Rules this project follows](#10-rules-this-project-follows)
11. [Known limitations](#11-known-limitations)
12. [What comes next](#12-what-comes-next)
13. [References](#13-references)

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
[section 11](#11-known-limitations).

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
(Kapoor & Narayanan, 2023). Two rules in [section 10](#10-rules-this-project-follows)
exist to prevent it.

**Normalisation (rescaling).** Putting different measurements onto a comparable
numerical scale, so that a temperature near 1400 does not dominate a ratio near
8 simply because it is a larger number.

**Time window.** A short, fixed-length slice of an engine's history, for
example 30 consecutive cycles. Windows are what the model actually receives,
because histories have different lengths and a network needs a fixed input size.

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

Each module's docstring explains the reasoning behind its step in full. Start
with `data/normalization.py` and `data/windowing.py`; those two contain the
decisions that matter most for the validity of the results.

---

## 8. How the project is organised

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
└── tests/                        155 tests
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

### Which file should I open?

| If you want to... | Open |
|-------------------|------|
| Understand what happens to the data, step by step | `notebooks/01_preprocessing_fd001.ipynb` |
| Know exactly how one step is computed | the matching file in `src/thesis_gnn_rul_xai/` |
| Know why a value was chosen when the paper is silent | `docs/ASSUMPTIONS.md` |
| Run the whole thing without reading anything | `scripts/prepare_data.py`, then `scripts/train.py` |
| Check that a step really does what it claims | the matching file in `tests/` |

---

## 9. Testing

```bash
pytest                     # everything
pytest -v                  # with the name of each test
pytest tests/test_normalization.py
pytest --doctest-modules src   # only the examples inside docstrings
```

The suite covers each preprocessing step, the sensor graph, and the pipeline
end to end. Four tests deserve special mention, because they check the
properties that make the experiment valid rather than merely correct:

- `test_normalization.py::test_test_data_never_influences_the_scaling`
- `test_pipeline.py::test_validation_engines_never_influence_the_scaling`
- `test_smoothing.py::test_the_filter_only_looks_backwards`
- `test_pipeline.py::test_training_and_validation_engines_never_overlap`

If any of these ever fails, results from this code are not trustworthy,
regardless of how good they look.

---

## 10. Rules this project follows

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

## 11. Known limitations

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

**The published numbers have not been reproduced, and may not be reachable.**
The paper's preprocessing is not fully described, its results come from a
single run with no seeds or error bars, and its reported software environment
lists no deep learning framework at all. Matching its figures exactly is not a
realistic target; understanding and documenting the difference is.


---

## 12. What comes next

1. Run this pipeline on the real FD001 files and compare the sensor diagnosis
   with the paper's stated selection.
2. ~~Implement the graph convolution model from Table 4 of the paper: three
   layers of sizes 32, 64 and 32, then two fully connected layers.~~ **Done**,
   in `src/thesis_gnn_rul_xai/models/`. `ESTGNN.layer_summary()` prints the
   architecture in the same form as Table 4 for checking against the paper.
   Still to write: the training loop (Table 6 gives Adam, 200 epochs, batch 32,
   learning rate 1e-3 decaying to 1e-4, MSE loss).
3. Reproduce a simple published baseline first, before trusting any graph
   result. Li et al. (2018) report an error of about 12.6 on FD001 and is a
   good target: a pipeline that cannot approach a known baseline has a problem
   that no amount of model tuning will fix. The figure to compare the graph
   model against is **14.93**, the FD001 entry of the paper's own
   "ESTGNN w/o LSTM" ablation (their Table 11), not the 19.68 four-dataset
   average quoted in assumption A-11.
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

## 13. References

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
