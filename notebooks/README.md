# The notebooks

The notebooks run this project's code on the real NASA C-MAPSS files and show
every intermediate table, shape and picture on the way. They are meant to be
read as well as run. This page holds what all of them share, so that none of
them has to repeat it: how they are laid out, who they are written for, and
where the code they call actually lives.

## Which notebook, in which order

```
01_preprocessing_fd001.ipynb    steps 1 to 8    the raw text files become examples
        │                                       and a sensor graph
        ▼
02_model_fd001.ipynb            steps 9 to 11   build the network, train it, score it,
        │                                       and put the sensor graph to the test
        ▼
xai/                            one notebook per explanation method, each explaining
  ├── 01_shap_fd001.ipynb       the same trained network; read xai/README.md first
  ├── 02_occlusion_fd001.ipynb
  └── 03_walks_fd001.ipynb
```

Read 01 and 02 in order: 02 starts from exactly what 01 produces. The step
numbers carry on from one to the other, so step 9 is the first step of notebook
02.

The notebooks in `xai/` are not a sequence. Each is one way of explaining a
trained network's predictions, and each can be read on its own once notebook 02
is familiar. They number their steps from 1 again, because they are
alternatives to one another rather than stages after one another. `xai/README.md`
says what they have in common and why they load a stored network instead of
training their own. The Shapley-value notebook, the paper's own method, is the
one to read first.

Two more folders are planned and not yet written: `temporal/`, one notebook per
way of turning a sensor's 30-cycle window into what the graph layers read, and
`graph/`, one per sensor graph. `docs/PENDING.md` describes them.

## How every notebook is laid out

Every notebook opens the same way — what it is for, what it does *not* do, and
a list of its steps — so a reader who has understood one can read the next.
Then:

1. **Background**, the ideas the notebook depends on, taught before they are
   used.
2. **Setup**, the imports and the settings.
3. **The data or the model**, loaded and checked, with one picture.
4. **The steps.** Each step is written in three parts:
   - **What** the step does.
   - **Why** it is necessary.
   - **A check**, a cell that fails loudly if the step did not do what the text
     says.
5. Further sections particular to the notebook.
6. **Final verification**, every check of the notebook gathered into one table,
   so a reader who trusts nothing else can run one cell.
7. **Summary**, with the decisions the notebook relies on that were not read
   from the paper.

## Who they are written for

Someone who knows some Python and some statistics, but who has not worked with
this dataset, has never trained a neural network with PyTorch — the library
this project builds its networks with — and has not read the paper. Every
technical term is defined the first time a notebook uses it. Where a notebook
depends on an earlier one, it says which idea it is borrowing and where that
idea was explained.

## Paper, choice or measurement

Every sentence in a notebook says what the paper states, what this project
chose where the paper is silent, or what this code measured, kept apart the way
[Paper, choice or measurement](../docs/learn/07_evidence.md) describes. In a
notebook a measurement is a number printed by a cell in the same notebook, or a
stored result named with the commit of the code that produced it.

**A notebook trains or explains one network**, fixed by one random seed, and a
different seed gives different numbers, so a number it prints is not a result
on its own. Where a notebook shows a stored result instead, it says which run
it read.

## Where the code actually is

The instructions that do the work are **not** in the notebooks. They are in
the Python package under `src/thesis_gnn_rul_xai/`, and a notebook calls them.
Smoothing the sensor readings, for example, is one line in notebook 01:

```python
train_smoothed = apply_ewma(train_frame, sensor_columns, alpha=0.5)
```

while `apply_ewma` itself lives in `src/thesis_gnn_rul_xai/data/smoothing/`,
where automated tests check it.

The reason for the split: a notebook's cells can be run in any order. Run cell
20 before cell 5 and you may get a wrong answer that looks fine, and nothing
tells you. A function kept in a file cannot be run out of order, and the test
suite runs it the same way every time. So the files hold the computation, and
the notebooks show what that computation does and why.

Two ways to find out more about anything a notebook calls:

- `help(apply_ewma)`, typed into an empty cell, prints the one line that says
  what a function does.
- The `README.md` in the folder beside the code explains the rest: the
  reasoning, the traps, and what every argument means. Every folder under
  `src/thesis_gnn_rul_xai/` has one.

## Before running one

- **Install the project** with `pip install -e ".[model,notebook]"` from the
  top of the repository. The first cell of each notebook also adds `src/` to
  Python's search path, so a clone that has not been installed still imports.
- **Put the NASA files in `data/raw/`.** The notebooks run on the real
  measurements and nothing else; without them the loading cell stops with a
  link to the download. `data/raw/README.md` says how the files are checked.
- **Notebook 02 and the `xai/` notebooks need PyTorch**, included in the
  `model` extra above. Notebook 01 does not.
- **The `xai/` notebooks need a stored five-seed run** of `config/fd001.yaml`
  that saved its trained networks. Make one with
  `python scripts/train.py --config config/fd001.yaml --seeds 5`; the notebooks
  stop with that command if none exists.

Every notebook is saved with its outputs filled in, so all of them can be read
without running anything. After changing one, run it again from the top so the
saved outputs match the code:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/02_model_fd001.ipynb
```
