# `utils` — small shared helpers

Four things every other part of the project needs: where the files live, how
random numbers are fixed, how progress messages are printed, and how a stored
result records the code that produced it. Nothing here knows anything about
engines, graphs or the paper.

> **The `.py` files here carry one line of description each, and nothing more.**
> All of the explanation lives in this file.

## Files

- `paths.py` — locates the project root and returns the standard data, report
  and figure directories.
- `seeding.py` — sets random seeds and creates random number generators.
- `logging_setup.py` — configures where log messages go and how they look.
- `provenance.py` — names the commit a result was produced at, and the moment
  it finished, so the result can be traced back to its code.
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
- `log_entry_files()` → the project's record of its choices and findings, one
  file per entry: every file in `docs/assumptions/` named like `A-07.md`, in
  number order, then every file in `docs/findings/` named like `F-16.md`, in
  number order. On 2026-10-03 that is 63 paths, `A-01.md` to `A-41.md` and
  then `F-01.md` to `F-22.md`. Each folder's `README.md` is left out, and so is
  any file named another way, such as `A-7.md`, so nothing that is not an
  entry is ever read as one. The website and `tests/test_paper_to_code.py`
  list the entries through this one function; `tests/test_log_layout.py`
  lists the folders its own way on purpose, to check it.

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

## Provenance: which code produced a result

### The problem

A number in `reports/` is only evidence if someone can find the exact code that
produced it and run it again. This repository once held a results file that
disagreed with its own documentation and could not be reproduced from any
committed state of the code (finding F-09, in `docs/findings/F-09.md`). So every
stored result carries a stamp saying which code it came from and when, and
`provenance.py` makes the two halves of that stamp. `scripts/train.py` and
`scripts/explain.py` both use it, and put both halves into the name of the
folder they write and into its `summary.json`.

### Prerequisite: what a commit is

The project's history is kept with git. A **commit** is a saved snapshot of
every tracked file, and git names each one by a long code of hexadecimal
characters (the digits 0 to 9 and the letters a to f). The first seven
characters are enough to find it again, so that short form is what gets
recorded: `git checkout 563c390` puts every file back exactly as it was in that
snapshot.

A snapshot only describes the files as they were when it was saved. If a file
has been edited since and not yet saved in a new commit, the code that ran is
not the code the commit holds, and checking the commit out would not reproduce
the result.

### `current_commit(cwd=None)`

Returns the short commit, with `-dirty` added when the files on disk differ
from it. It asks git two questions:

1. `git rev-parse --short HEAD` — which commit is checked out? This gives the
   short name, such as `563c390`.
2. `git status --porcelain` — is anything different from that commit? This
   prints one line per changed, added or untracked file, and nothing at all
   when the files match the commit exactly.

If the second answer is not empty, `-dirty` is appended. Run in this repository
on 2026-09-14, it returned:

```text
563c390-dirty
```

No tracked file had been modified at that moment. The mark came entirely from
**untracked** files — ones git has never been told to keep: `AGENTS.md`, and
two scratch folders, `.claude/` and `tmp/`. Those two were added to
`.gitignore`, the list of files git is told to overlook, in the very next
commit, and `git status` does not report a file that list covers, so they no
longer set the mark. Counting untracked files is deliberate and conservative. An
untracked file under `src/` would be a module the commit does not contain, so
the code that ran could not be rebuilt from the commit; counting every untracked
file catches that case, at the cost of a false alarm for a stray file anywhere
else. A result marked `-dirty` because of a scratch folder is still traceable,
but only by someone who can confirm that nothing under `src/`, `scripts/` or
`config/` was among the differences, so clear the stray files before a run
whose numbers you mean to quote.

If git cannot be run at all — it is not installed, or the folder is a copy
rather than a clone — the answer is `"unknown"`, and a result stamped
`unknown` cannot be quoted under the project's rule that a stored result must
name its code.

`cwd` is the folder git is asked about. Left out, it is the folder
`provenance.py` itself sits in, so the answer describes the repository the
package lives in, whichever directory the script was started from.

### `utc_stamp()`

Returns the current time in UTC (Coordinated Universal Time, the world's reference clock, which never changes for summer), written with no separators:

```text
20260914T170300Z
```

That reads as 2026-09-14, then `T` for time, then 17:03:00, then `Z`, the
standard mark for UTC. UTC rather than local time, because local time jumps an
hour twice a year and differs between machines, so two runs could otherwise
sort in the wrong order. No colons, because Windows does not allow `:` in a
file or folder name, and this stamp goes into folder names.

The stamp is also the only thing that separates two runs of identical settings
at the same commit, and it only counts whole seconds. Two such runs finishing
within the same second would be given the same folder name, and because both
scripts create their folder with `exist_ok=True` — carry on if it already
exists — the second would write its files over the first rather than stop. One
run at a time, which is how the playground runs its jobs, never meets this.

### What else a run records, and why the commit is not enough

The commit says which code a run came from. Three more things decide the
number, and none of them is in the commit, so `scripts/train.py` records all
three beside every run, in `summary.json` and in the run's folder.

**`environment()`** — the machine and the libraries. It returns the Python
version, the operating system, the processor, and the installed version of
NumPy, pandas, scikit-learn, SciPy, PyTorch and PyYAML, with how many processor
threads PyTorch uses. A trained network is not bit-for-bit the same on another
PyTorch version or another processor: the independent audit of 2026-09-08
re-trained this project's default on a different machine and got the same
answer only to within the seed-to-seed spread, not to the last digit. A run
written on 2026-09-15 recorded:

```text
python 3.11.9   platform Windows-10-10.0.26200-SP0   processor AMD64 Family 25 Model 117
numpy 2.4.6   pandas 3.0.3   scikit-learn 1.9.0   scipy 1.17.1   torch 2.13.0+cpu
torch_threads 8   cuda_available False
```

**`file_fingerprints(paths)`** — which data files were read. A fingerprint here
is the SHA-256 digest of a file: a 64-character code computed from every byte of
it. The same bytes always give the same code, and changing a single byte
changes the code completely, so two files with the same fingerprint are, for
every practical purpose, the same file. The known answer, from the standard that
defines SHA-256, is that the three bytes `abc` give
`ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`, and
`tests/test_run_records.py` checks the function against it. The FD001 files in
`data/raw/` on 2026-09-15:

```text
train_FD001.txt  963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8
test_FD001.txt   3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851
RUL_FD001.txt    a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca
```

A re-downloaded file, a file converted to other line endings, or one edited by
hand would each show a different code, and the reload test in
`tests/test_run_records.py` stops with "the NASA files differ" instead of
reporting a confusing change in the predictions.

**`capture_code_state(cwd=None)` and `write_code_state(directory, state)`** —
what `-dirty` actually meant. `-dirty` says the code differed from the commit,
not how. `capture_code_state` asks git for every file under `src/`, `scripts/`,
`config/` and `pyproject.toml` that differs from the commit (the list is
`CODE_PATHS`), for the differences themselves as a **patch** — a text file
listing, for each changed file, the lines removed and the lines added — and for
the full content of any file there that git has never seen, since a patch cannot
hold one. Changes to documents are left out on purpose: they cannot change a
number.

It is called when the run **starts**, while the code on disk is the code the
process has just loaded; editing a file during a long run would otherwise be
recorded as though the run had used it. `write_code_state` writes what was
captured into the run's `code_state/` folder at the end — `status.txt`, the list
of changed files; `uncommitted.patch`; and `untracked/` for new files — and
returns the list, which also goes into `summary.json` as `uncommitted_code`.
When nothing under those paths differs, it writes nothing.

**`write_code_state_archive(directory, state)`** writes the same three things
into one file, `code_state.zip`, instead of a folder. Results written inside a
run's own folder — an explanation, a walk measurement — sit several folders
deeper than the run itself, and under the longest run names a path such as
`…/code_state/untracked/src/thesis_gnn_rul_xai/explain/connections.py` passes
the 260-character limit Windows puts on a path, so the folder cannot be made.
Inside a zip file the names are not paths on disk and have no such limit.

**A trap that hid until 2026-09-16.** Git prints the patch as the bytes of the
files themselves, which in this repository are UTF-8. The call that reads it
used to let Python pick the decoding, and on Windows that is the old
single-byte code page, which has no character for a handful of byte values.
Every earlier patch happened to avoid them; the first "←" in a source file did
not, the read failed inside Python's own reader thread, and the run recorded
the word `None` as its patch. Every git call here now decodes as UTF-8, and a
byte that still cannot be decoded becomes a replacement character rather than
a failure, since a patch with one odd character beats no patch at all.

To rebuild the code of such a run: check out its commit, then
`git apply code_state/uncommitted.patch`, then copy anything in `untracked/`
back into place.

## Judgement calls

None. This folder makes no choices that depart from the paper, so it has no
`A-xx` entries in `docs/assumptions/` — that folder records every such
judgement call made elsewhere in the project.

## Known loose end

PyTorch is seeded directly by the training code rather than inside
`set_global_seed`. `set_global_seed` is called by the preprocessing notebook,
but not by the training path. This is recorded here as an observation, not
fixed — changing it would change behaviour, and any stored result produced
before the change would no longer be reproducible.
