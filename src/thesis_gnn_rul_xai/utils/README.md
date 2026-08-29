# `utils` — small shared helpers

Three things every other part of the project needs: where the files live, how
random numbers are fixed, and how progress messages are printed. Nothing here
knows anything about engines, graphs or the paper.

> **The `.py` files here carry one line of description each, and nothing more.**
> All of the explanation lives in this file.

## Files

- `paths.py` — locates the project root and returns the standard data, report
  and figure directories.
- `seeding.py` — sets random seeds and creates random number generators.
- `logging_setup.py` — configures where log messages go and how they look.
- `__init__.py` — marks the folder as a package. It contains no code.

## Paths

Scripts and notebooks in this project can be started from any directory. To
make sure they all read and write the same places, `project_root()` walks up
the directory tree from `paths.py` until it finds a folder containing
`pyproject.toml`, and returns that folder. That is the usual convention in
Python projects, and it means the code behaves identically from a notebook,
from a terminal, or from an automated test. If no parent folder has that file,
`project_root()` raises `FileNotFoundError` — which normally means the package
was copied somewhere without its project files.

Everything else here is built on that:

- `raw_data_dir()` → `data/raw`, the original C-MAPSS text files.
- `processed_data_dir()` → `data/processed`, arrays produced by preprocessing.
- `reports_dir()` → `reports`, generated tables, logs and summaries.
- `figures_dir()` → `reports/figures`.
- `ensure_dir(path)` creates a directory and any missing parents, then returns
  it. Calling it on a directory that already exists does nothing and raises
  nothing.

**`data/raw` is read-only.** Nothing in this project ever writes into it. That
is what guarantees the original measurements can always be recovered.
`raw_data_dir()` deliberately does not create the directory; the other three
accessors do.

## Seeding

A **random seed** is a fixed starting number for a random number generator.
Given the same seed, the generator produces the same sequence of numbers every
run. Several steps of the pipeline draw random numbers — shuffling, splitting
data into training and validation parts, initialising network weights — so
without a fixed seed the results differ every run and nobody can repeat the
experiment.

`set_global_seed(seed)` sets the seed of Python's built-in `random` module,
NumPy's legacy global generator, and the `PYTHONHASHSEED` environment variable
(which affects the iteration order of some Python data structures). It returns
the seed it set, so the caller can log or store it without repeating the value.

`make_rng(seed)` returns a NumPy `Generator` that is independent of the global
state. Prefer it over the global generator: a local generator cannot be
disturbed by unrelated library code that also draws random numbers, and passing
the generator object into a function makes the source of randomness visible in
that function's signature, which is easier to audit.

`DEFAULT_SEED` is 42. The particular value means nothing; it only has to stay
constant.

### The warning that matters

Fixing the seed makes a single run *repeatable*. It does **not** make the
result *reliable*. A number from one seed can be noticeably better or worse
than the average across seeds, purely by chance. Experiments in this project
are meant to be run with several seeds, reporting the mean and the standard
deviation. Quoting only the best seed is a well-known source of
over-optimistic results in the machine learning literature, and is avoided
here.

### A note for anyone extending this

Deep learning frameworks keep their own generators, which `set_global_seed`
does not touch. If PyTorch's seeding call were added to that function, one call
would cover everything. As the code stands it is seeded elsewhere instead — see
the "Known loose end" note below.

## Logging

The pipeline prints short progress messages so a reader can see what the code
did and in what order. It uses Python's `logging` module rather than `print`,
because logging can be switched off, redirected to a file, or filtered by
severity without editing any code.

Three severity levels are used:

- `INFO` — normal progress, such as "loaded 20631 rows" or "kept 14 of 21
  sensors".
- `WARNING` — something unexpected that the code can still handle, such as an
  engine unit shorter than the chosen time window.
- `ERROR` — a condition that stops the pipeline.

`configure_logging(level)` sets up the root logger to write to standard output
with a timestamped format. It **removes any existing handlers first**. That is
deliberate: without it, re-running a notebook cell adds a second handler and
every log line appears twice. Because of this, calling `configure_logging`
repeatedly is harmless.

`get_logger(name)` returns the logger for one module. Pass `__name__`, so log
lines show which module produced them.

## Judgement calls

None. This folder makes no choices that depart from the paper, so it has no
`A-xx` entries in `docs/ASSUMPTIONS.md` — that file records every such
judgement call made elsewhere in the project.

## Known loose end

PyTorch is seeded directly by the training code rather than inside
`set_global_seed`. `set_global_seed` is called by the preprocessing notebook,
but not by the training path. This is recorded here as an observation, not
fixed — changing it would change behaviour, and any stored result produced
before the change would no longer be reproducible.
