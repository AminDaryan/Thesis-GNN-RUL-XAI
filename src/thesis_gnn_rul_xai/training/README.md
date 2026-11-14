# `training` — fitting the network

This folder fits **ESTGNN**, the neural network this project re-implements from
Wen et al. (2026), to the sliding windows produced by the preprocessing half of
the project. It also holds the settings object that records how a run was
configured, so a result can be traced back to the settings that produced it.

> **The `.py` files here carry one line of description each, and nothing more.**
> All of the explanation lives in this file. The section
> [What each name means](#what-each-name-means) at the end lists every setting
> and every field of the run record.

PyTorch is an optional dependency of this project. Only this folder, `models`
and `explain` need it:

```
pip install -e ".[model]"
```

## The files

| File | What it is |
|------|------------|
| `config.py` | `TrainingConfig`, the dataclass holding every setting for one run, plus `load_training_config` for reading one back from YAML. |
| `run.py` | Everything involved in running the model: the training loop (`train`), the prediction helper (`predict`), the scoring wrapper (`evaluate_model`), the device chooser (`resolve_device`), and `TrainingHistory`, the record of a finished run. |
| `stored.py` | Finding a finished run in `reports/` and loading one of its trained networks back: `find_runs`, `read_run`, `load_seed`, and `StoredRun`, what a run recorded about itself. |
| `__init__.py` | Re-exports the names above. |

## What one run does

`train(model, train_set, validation_set, config)` takes a network, the training
windows, optionally some held-out validation windows, and a `TrainingConfig`.
Each epoch it shuffles the training windows, takes gradient steps in batches
(32 by default), and — only if there is a validation set — scores the held-out
windows without updating anything. It returns a `TrainingHistory`: a table with
one row per epoch, the epoch whose weights the model ended up holding, whether
it stopped early, how long it took, and the settings used.

The model is modified in place. On return it holds the weights of whichever
epoch the loop selected, not necessarily the last one — see the next section.

## Validation is optional, and what changes without it

This is the part most likely to trip someone up.

The project's preprocessing settings have a `validation_fraction`, and its
default value is `0.2`: a fifth of the training engines are held out. Setting it
to `0.0` — which is the paper's arrangement — means **no engines are held out at
all**, and `train` is called with no validation set. When that happens:

- nothing is scored between epochs;
- nothing can stop early — every epoch in `epochs` runs;
- the weights you get back are the **last epoch's**;
- the recorded best validation error is `nan`, because there was none, and
  `summary()` says so in words rather than printing `nan` at you.

The loop logs a warning when it is in this state, because with no held-out data
there is no honest way to choose between two settings.

When a validation split *does* exist, the loop scores the held-out engines
after every epoch and uses that number for three things:

1. it keeps a copy of the weights from the best-scoring epoch, and restores
   those weights at the end of the run;
2. it stops early if the score has not improved for `early_stopping_patience`
   epochs (when that is not `None`);
3. it drives the learning-rate schedule, but only when that schedule is
   `"plateau"`, which reacts to the score rather than to the epoch number.

Point 1 happens whether or not early stopping is switched on. So turning early
stopping off while a validation split is present does **not** give you the
paper's procedure: you still end up with the best-validation checkpoint rather
than the final-epoch model. Only removing the validation split reaches the
branch that keeps the final weights.

## Never report validation error as test error

The validation set here is not a small stand-in for the test set, and its error
must never be published as though it were. The two weight their engines quite
differently:

- **Validation** holds *every* window of the held-out engines. On the FD001
  sub-dataset its 20 engines contribute between 108 and 240 windows each, so a
  single unusual engine can move the figure.
- **The test set** holds *one* window per engine. Each of its 100 engines is
  worth exactly one per cent.

Neither is reliably the harder of the two, and the gap between them says
nothing about the model. Validation error is the right signal for deciding when
to stop training and for comparing settings, because both are relative
judgements made against a fixed set. It is the wrong number to publish.

## Targets are scaled to 0–1 for training; results come back in cycles

The network ends in a sigmoid, so its output can only lie between 0 and 1. A
label measured in flight cycles is out of reach for such a layer: no setting of
the weights makes a sigmoid emit 40. Two conversions therefore bracket the
model. On the way in, the loop divides every target by `rul_max`, the ceiling
the labels are capped at, before the loss ever sees it. On the way out, the
model's `predict_cycles` multiplies the sigmoid back by the same constant, so
`predict` and `evaluate_model` return flight cycles.

For a single window whose true remaining life is 40 cycles, with the default
`rul_max` of 125:

| Step | Value | Unit |
|------|-------|------|
| label from the preprocessor | 40 | cycles |
| target given to the loss | 40 / 125 = 0.32 | normalised |
| what the sigmoid emits | 0.36 | normalised |
| what the loss sees | (0.36 − 0.32)² = 0.0016 | normalised |
| prediction handed back | 0.36 × 125 = 45 | cycles |
| error a reader is shown | 45 − 40 = 5 | cycles |

The constant is the same for every window, so the change of unit cannot change
any comparison: an error in cycles is exactly `rul_max` times the normalised
error. Each epoch of the history records both numbers — `validation_loss` (mean
squared error, normalised) and `validation_rmse_cycles` (root mean squared
error, in cycles) — and `sqrt(validation_loss) * rul_max` reproduces the second
from the first. They are kept apart for readability, not arithmetic: 0.0016 is
not a quantity anyone can judge, and "five cycles of remaining life" is.

## Getting the examples into PyTorch: `_as_loader`

This is the small private function that sits between the preprocessing half of
the project and the training loop. It is worth a section because it does three
unrelated jobs in six lines, and because one of them is a judgement call.

The preprocessing gives you a `WindowedDataset`, which is plain NumPy arrays.
The training loop needs batches of PyTorch tensors, with labels the network can
actually produce. `_as_loader` bridges the two.

**One — convert the arrays into tensors.**

```python
features = torch.from_numpy(np.ascontiguousarray(dataset.features)).float()
```

`torch.from_numpy` wraps a NumPy array as a tensor, sharing the same memory
rather than copying it where it can. It can only do that when the array is one
unbroken run of memory, and slicing an array does not guarantee that:
`subset_by_units`, which is how the held-back engines are separated, leaves
behind a view whose rows are scattered. `np.ascontiguousarray` compacts such an
array into a fresh contiguous one, and does nothing at all when the array is
already contiguous. `.float()` then pins the type to 32-bit floating point,
which is what the network's weights are.

**Two — shrink the labels to fit the last layer.**

```python
targets = targets / float(rul_max)
```

The labels arrive in flight cycles, from 0 to 125. The network ends in a
sigmoid, which can only ever emit a number between 0 and 1, so no setting of
the weights could make it produce 40. Dividing by the ceiling puts the labels
where the output can reach them. `ESTGNN.predict_cycles` multiplies by the same
constant on the way back out, so everything a reader sees is in cycles again.

The constant is the same for every window, so this cannot change any
comparison. It is nonetheless a decision, because the paper never states it
(assumption A-13, in `docs/assumptions/A-13.md`).

**Three — pair each window with its label, and hand out batches.**

```python
DataLoader(TensorDataset(features, targets), batch_size=..., shuffle=..., ...)
```

`TensorDataset` zips the two together so that item *i* is window *i* with label
*i*, and they can never drift apart. `DataLoader` is what the training loop
iterates over; it slices that pairing into batches. Three of its settings
matter here:

- **`shuffle`** is `True` for training, so the examples arrive in a different
  order every epoch and the network cannot learn anything from the order
  itself. It is `False` everywhere else, where the order is wanted intact.
- **`generator`** is the seeded source of randomness used for that shuffling.
  Passing it is what makes the batch order repeatable from one run to the next.
- **`drop_last=False`** keeps the final short batch. With 1 000 examples and a
  batch size of 32 the last batch holds 8, not 32; the usual alternative,
  `drop_last=True`, would throw those 8 away every epoch. Discarding real
  training data to keep the batches a tidy shape is not a trade this project
  makes.

## The learning-rate schedule

The paper gives a starting rate of 1e-3 and a final rate of 1e-4 but never says
how the rate travels between them. `lr_schedule` offers four ways:

- `"step"` (the default) — hold the starting rate, then drop straight to
  `final_learning_rate` in one move, halfway through the run unless
  `lr_step_epoch` says otherwise. The run therefore only ever uses the two
  rates Table 6 actually prints, which happen to be exactly one factor of ten
  apart. This is the closest thing to a published practice: Li, Zhao, Sun, Yan
  and Chen (2021) decay "by multiplying 0.1 in epochs 120 and 160", and no
  C-MAPSS paper found decays exponentially (assumption A-17, and
  `docs/LITERATURE_BASIS.md`).
- `"exponential"` — multiply by a constant factor after every epoch, chosen so
  the final epoch lands exactly on `final_learning_rate`. This was the default
  until 2026-09-01, when the literature search found nothing supporting it.
- `"cosine"` — follow a cosine curve between the two values: slow at first,
  fastest in the middle, slow again at the end.
- `"plateau"` — drop the rate by a factor of ten only when the held-out error
  stops improving, never going below `final_learning_rate`. Unlike the other
  three it reacts to a measurement rather than to the epoch number, so it needs
  something to measure. **With nothing held back it never fires at all**, and
  the run finishes at its starting rate of 1e-3 instead of the 1e-4 Table 6
  gives. The default configuration held nothing back from 2026-08-29 until
  2026-09-08, and the loop still warns when you ask for this combination
  rather than letting the run look normal. (Corrected 2026-10-02: this said
  until 2026-09-02, a day on which the held-back setting did not change; the
  log's entry on holding engines back has the history (assumption A-06).)
- `"constant"` — never change it. Included so the effect of the schedule can be
  measured.

## Smaller things worth knowing

- **Unknown keys in a settings file are an error.** `load_training_config`
  refuses a YAML file containing any key that is not a field of
  `TrainingConfig`, rather than ignoring it. A silently ignored typo in a
  settings file is a common and hard-to-find source of wrong results.
- **`device="auto"` says which device it picked.** A run that silently falls
  back to the CPU and takes forty times as long is a confusing way to spend an
  afternoon, so `resolve_device` logs its choice. Asking for `"cuda"` when
  PyTorch cannot see a CUDA device raises instead of falling back quietly.
- **One seed covers everything random.** `seed` fixes both the weight
  initialisation and the batch order, and it is stored in the history, so a run
  can be repeated exactly.
- **`batch_size` in `predict` changes nothing but speed and memory.** It does
  not affect the numbers.
- **Only Adam and mean squared error are implemented**, because those are what
  the paper specifies. Asking for anything else raises rather than silently
  substituting.

## The judgement calls made in this folder

Each of these is a place where the paper does not say what to do and we had to
choose. All are written up in `docs/assumptions/`, each in the file named after
the code given in parentheses.

- **Scaling the targets.** The paper's final layer is a sigmoid, but it never
  states the arithmetic that turns that number back into flight cycles or what
  the training targets look like. We divide targets by `rul_max` and multiply
  predictions back by it (assumption A-13).

- **How the learning rate decays.** The paper gives both endpoints and calls it
  an adaptive schedule, but never the mechanism. The default holds the first
  rate and drops once, straight to the final one, halfway through the run, so
  only the two rates the paper prints are ever used. Exponential decay was the
  default until 2026-09-01; it and the other options exist so the sensitivity to
  that choice can be measured (assumption A-17).

- **Early stopping.** The paper trains for a fixed number of epochs with no
  validation set and no early stopping described. Stopping when the validation
  error has stopped improving, and restoring the best epoch's weights, is a
  deliberate departure from the paper rather than a reconstruction of it, and
  has to be described that way (assumption A-18).

- **Holding engines out at all.** Whether to have a validation set is itself a
  departure, since the paper has none (assumption A-06), and it is the reason
  validation error and test error must be kept apart, as described above.

- **Nothing added to keep the weights small.** A network trained long enough
  can fit the noise of its own training engines and do worse on engines it has
  not seen; *regularisation* is anything added to training to discourage that.
  The paper names none, so none is added: Adam keeps PyTorch's weight decay of
  0 (a penalty that shrinks every weight a little at each step), the network
  has no dropout (setting a random share of a layer's numbers to zero while
  training), and `gradient_clip` (a cap on the size of one step) is off. Early
  stopping on the held-back engines is the only guard. Over 200 passes with a
  fifth of the engines held back, the default network's held-back error rises
  by 0.411 ± 0.398 cycles after its best pass and the recurrent stage's by
  3.574 ± 0.721, so the choice is unlikely to matter much for the first and may
  matter a great deal for the second, which is an argument, not a measurement
  (assumption A-40).

## Reference

Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
168, 110875. https://doi.org/10.1016/j.ast.2025.110875

## What each name means

### `TrainingConfig` — the settings for one run

The first six come from the paper's Table 6. The next four are ours, because
the paper does not state them.

| Setting | What it is |
|---|---|
| `epochs` | How many times to go through the whole training set. Table 6 gives 200. |
| `batch_size` | How many examples to look at before adjusting the weights. Table 6 gives 32. |
| `learning_rate` | How big those adjustments start out. Table 6 gives 0.001. |
| `final_learning_rate` | How big they are at the last epoch. Table 6 gives 0.0001. It may not be larger than `learning_rate`; the paper describes a rate that shrinks. |
| `optimizer` | Only `"adam"` is implemented, which is what Table 6 specifies. Anything else raises rather than quietly substituting. |
| `loss` | Only `"mse"`, likewise. |
| `lr_schedule` | How the rate travels between the first value and the last, one of the options above (assumption A-17). |
| `early_stopping_patience` | Stop when the held-out error has not improved for this many epochs, and go back to the best epoch's weights. `None` runs every epoch (assumption A-18). |
| `min_improvement` | How far the held-out error has to fall, in cycles, before it counts as an improvement worth restarting the count for. |
| `gradient_clip` | A limit on how large one adjustment may be, or `None` to leave them alone. The paper never mentions it, so it is `None` by default and the run matches (assumption A-40, which covers everything else that could hold the weights back). |
| `seed` | Fixes the starting weights and the order the examples are shuffled into. |
| `device` | `"auto"` uses a graphics card if there is one, otherwise the processor. `"cpu"` and `"cuda"` force it. |
| `notes` | Free text saved with the settings. |

`save(path)` writes the settings to a YAML file. `load_training_config(path)`
reads one back, and refuses a file containing any key that is not a setting
above — a silently ignored typo in a settings file is a common and very
hard-to-find cause of wrong results.

### `TrainingHistory` — the record of one finished run

| Field | What it holds |
|---|---|
| `epochs` | One row per epoch: the training loss, the held-out loss, the held-out error in cycles, and the learning rate that was in force. |
| `best_epoch` | Which epoch's weights the model now holds, counting from 1. |
| `best_validation_rmse` | The held-out error at that epoch, in cycles. `nan` when nothing was held out. |
| `stopped_early` | Whether the run ended because patience ran out rather than because it reached `epochs`. |
| `seconds` | How long it took. |
| `config` | The settings used, so a result can be traced back to them. |

`summary()` returns one paragraph describing the run, worded differently
depending on whether any engines were held back — with none, there is no best
epoch to report, because no epoch was chosen over any other. `save(directory)`
writes
`training_history.csv` and `training_config.yaml` into that folder.

### The four functions in `run.py`

| Name | What it does |
|---|---|
| `train(model, train_set, validation_set=None, config=None)` | Trains the network and returns a `TrainingHistory`. The model is changed in place and ends holding the weights of whichever epoch the loop chose. |
| `predict(model, dataset, batch_size=256, device=None)` | One prediction per window, in flight cycles, in the order the windows are already in. |
| `evaluate_model(model, dataset, device=None)` | Predicts, then scores. Returns the same dictionary as `evaluate` in `evaluation/metrics.py`. |
| `resolve_device(preference="auto")` | Decides between the graphics card and the processor, and logs which. Asking for `"cuda"` when PyTorch cannot see one raises rather than falling back quietly. |

### Loading a stored run back: `stored.py`

`scripts/train.py` writes one folder per run into `reports/`, and inside it one
folder per seed holding that seed's trained weights (`model.pt`), its per-pass
record and its test predictions. Anything that needs a trained network after
the fact — the explanation notebooks in `notebooks/xai/`, `scripts/explain.py` —
loads it from there rather than training its own, so what it works on is a
network whose commit and results are on record.

Loading weights back is the easy half. The trap is the data. **The seed decides
which engines were held back**, so the data a seed's network was trained on can
only be rebuilt with that seed. `load_seed` does that: it reads the settings the
run saved beside itself, replaces the seed unless the run held the split fixed,
runs the pipeline, builds the network those settings describe on the graph that
pipeline produced, and loads the weights into it. A network trained without
graph layers is rebuilt without them the same way, from the `spatial: none` in
its `model_config.yaml`, and its saved weights still hold the sensor graph, so
the check of that graph below runs on it too. Rebuilding with any other seed
would hand back engines the network learned from under the name "held back".
A run that held nothing back is rebuilt under each seed too, unless it recorded
`"fixed"`, because the seed still matters to its data;
[The split part of a run folder's name](#the-split-part-of-a-run-folders-name-and-its-change-of-2026-10-03)
below explains how.

| Name | What it does |
|---|---|
| `find_runs(preprocessing=None, seeds=None, with_weights=False, root=None, **summary_fields)` | Every run in `reports/` (or `root`) whose saved settings match, oldest first. `preprocessing` compares every setting except `notes`; `seeds` the number of seeds; any other keyword a field of the run's `summary.json`, such as `temporal_stage="last_cycle"`; `with_weights=True` keeps only runs that saved a network for every seed. A folder that is not a finished run, or whose settings file no longer loads, is skipped. |
| `read_run(path)` | One run, as a `StoredRun`: its folder, its `summary.json`, its preprocessing and model settings, and its per-seed results table. Refuses a folder missing any of those files. |
| `load_seed(run, seed, data_dir=None)` | The trained network of one seed, switched to evaluation, and the `PreparedData` rebuilt as that network saw it. Refuses a seed the run never trained, a run from before weights were saved, and, since 2026-10-03, a rebuilt sensor graph that is not the one the network was trained on ([below](#the-graph-a-reload-rebuilds-must-be-the-one-the-network-was-trained-on)). |
| `StoredRun` | What `read_run` returns. `seeds` lists the seeds in training order, `weights(seed)` is where one seed's weights live, and `has_weights()` says whether every seed saved them. |

#### Everything a run folder holds

Written by `scripts/train.py`. The last group has been written since 2026-09-15;
older runs lack it, and nothing reads it as required.

| File | What it records |
|---|---|
| `summary.json` | The commit, the settings' digest, when it finished, the mean and spread of the four measures, and every setting that changes the number. Since 2026-10-03 also the mean and spread of `n_early`, and `null` for a mean or spread that is undefined, such as an ASUE when a seed predicted no engine early (assumption A-39). |
| `results.csv` | One row per seed: the four measures, the best pass, the held-back error, how many passes ran, how many test engines were predicted late (`n_late`) and, since 2026-10-03, early (`n_early`, the count ASUE averages over; for an older run it is `n_engines` minus `n_late`). |
| `preprocessing_config.yaml`, `model_config.yaml` | The data and network settings in full. A network without graph layers has the line `spatial: none` in `model_config.yaml`, and `"spatial": "none"` under `summary.json`'s `model`; for a network with them the line is left out, as it was before the setting existed on 2026-11-13, and a missing line reads as the graph layers (`models/README.md`, under "The eight settings"). |
| `seed_N/model.pt` | That seed's trained weights. |
| `seed_N/training_history.csv`, `seed_N/training_config.yaml` | The error after every pass, and the training settings. |
| `seed_N/test_predictions.csv` | One row per test engine: its number, last cycle, label and prediction. |
| `summary.json` → `command` | The command line the run was started with. |
| `summary.json` → `environment` | Python, the operating system, the processor, and every library version a number depends on (`utils/README.md`). |
| `summary.json` → `data_files` | The fingerprint of each NASA file read, so a changed file is caught. |
| `summary.json` → `uncommitted_code`, and `code_state/` | Which code files differed from the commit, and the differences themselves as a patch. |
| `seed_N/split.json` | The engine numbers this seed trained on, held back and was scored on. |
| `seed_N/scaling.csv` | The average, spread, lowest and highest value the scaling measured for each sensor on this seed's training engines. |
| `seed_N/graph.json` | Since 2026-10-03: the sensor graph this seed was given, as its kind, the seed it was drawn from (`null` unless it is a random or rewired graph, assumption A-41) and every connection as a pair of sensor names. |
| `summary.json` → `graph_seeds` | Since 2026-10-03: for a random or rewired graph, the seed each seed folder's graph was drawn from, in the order of `seed_folders`; `null` for the other graphs. |

#### The `cfg` code in a run folder's name, and its change of 2026-10-03

A run folder's name reads like
`fd001_physics_row_cap_5seed_varsplit_cfg818b40_<commit>_<time>`: the
sub-dataset, the sensor graph, how the graph is rescaled, whether the test
labels are capped, the number of seeds, whether the held-back engines change
with the seed (`varsplit`, `fixedsplit` or, since 2026-10-03, `nosplit`; the
next section), a mark for each network or training choice away from its
default (such as `_lstm32` for the recurrent stage of hidden size 32, or,
since 2026-11-13, `_nogcn` after it for a network with its graph layers removed,
assumption A-42), then
`cfg` and six characters, then the commit and the time it finished. The six
characters are the digest of the preprocessing settings, a short code that
changes whenever any setting does; the package's own `README.md`, one folder
up, works one out by hand. The same code is written into `summary.json` as
`config_digest`.

**On 2026-10-03 the code stopped counting the `notes`**, the sentence of free
prose saved with the settings. Before that, a run's code changed whenever the
sentence did, so runs with identical settings were filed under different codes.
Stored runs keep their names, since a stored result is never renamed, so the
same settings now appear under an old code in folders written by the code from
before the change and under the new one in folders written by the code since.
A run trained from a checkout of an older commit still gets an old code.
Measured on 2026-10-03 over the 50 runs stored in `reports/` whose names carry
a code: they used 16 old codes for 10 different sets of settings, and none yet
carried a new one.

| New code | Old codes in stored folder names (how many runs) | How the settings differ from `config/fd001.yaml` |
|---|---|---|
| `818b40` | `ba82e3` (15), `72f506` (4) | none; `PreprocessingConfig()` gave `e3a331` |
| `2c9b5b` | `00be26` (6), `722cd2` (4), `ee5482` (1) | nothing held back (`validation_fraction` 0.0) |
| `5ebd59` | `5d7c10` (6), `31648f` (1) | graph rescaled `symmetric` |
| `c9dec2` | `7fdfce` (2), `c7e4ad` (1) | `correlation` graph |
| `9ea488` | `b017ba` (2), `2b37a4` (1) | `symmetric`, and nothing held back |
| `25dd1e` | `d20b12` (2) | `complete` graph, and nothing held back |
| `7b9a7b` | `f21547` (2) | `complete` graph |
| `a82754` | `8ff2ea` (1) | `minmax_per_condition` scaling |
| `d53000` | `d3d83e` (1) | `minmax_per_condition`, and nothing held back |
| `7c2c29` | `e49671` (1) | test labels not capped (`config/fd001_uncapped.yaml`) |

So two runs belong together when their saved settings agree, not when their
names do. `find_runs` compares the saved settings, never the name, and is the
way to collect every run of one configuration across the change.
`tests/test_run_records.py` checks that every stored name carries the code of
the settings saved beside it, under one rule or the other.

#### The split part of a run folder's name, and its change of 2026-10-03

The *split* is which of the training engines are held back — a fifth of them
by default — to decide when training stops and which pass is kept. Each seed
draws its own unless `--fixed-split` is given. A run records what happened in
two places, its folder name and the `split` field of its `summary.json`:

| In the name | `split` in `summary.json` | What happened |
|---|---|---|
| `varsplit` | `"varies_with_seed"` | Engines were held back, and each seed drew its own (the default). |
| `fixedsplit` | `"fixed"` | Engines were held back, the same ones for every seed (`--fixed-split`): the data was prepared once, under the first seed, and `load_seed` rebuilds it that way. |
| `nosplit` | `"none"` | No engine was held back (`validation_fraction: 0.0`, the paper's own arrangement), so there was no split to vary or to fix. Every seed's data was prepared under that seed, even when `--fixed-split` was given, and `load_seed` rebuilds it that way. |

**Before 2026-10-03 both came from the `--fixed-split` option alone**, so a run
that held nothing back was still named `varsplit` and recorded
`"varies_with_seed"`, describing a split that never happened (the audit of
2026-09-08, its item D-4). Since then `split_record` in `scripts/train.py`
derives both from the engines each seed actually held back. Stored runs keep
what they recorded, because a results file is never edited. Counted on
2026-10-03, 17 of the 136 stored runs held nothing back, every one of them
recorded as `"varies_with_seed"`, the latest
`fd001_physics_row_cap_5seed_varsplit_noearlystop_cfg00be26_19dceb5_20261002T183904Z`;
no stored run recorded `"fixed"`.

So for a run of either age, `validation_fraction` in its
`preprocessing_config.yaml` is the field to trust: 0.0 means nothing was held
back, whatever `split` says. The readers of the field handle both ages. The
playground's run list shows such a run's split as none held back, with the
recorded value beside it on the run's own page; `scripts/baselines.py` compares
the `split` of two runs only when they held engines back; and
`find_runs(split=...)` matches the field as written, so
`find_runs(split="none")` finds only runs made since the change — to collect
both ages, select on `validation_fraction` through the `preprocessing`
argument instead.

**How a run that held nothing back is prepared, and rebuilt.** `load_seed`
goes by the recorded `split`: `"fixed"` is rebuilt once, under the first seed,
and anything else under each seed. That is how each was prepared.
`scripts/train.py` prepares every seed of a run that holds nothing back under
that seed, even when given `--fixed-split`, since there is no split to hold;
such a run records `"none"`. Before 2026-10-03 such a run recorded
`"varies_with_seed"` and was prepared under each seed too, or `"fixed"` under
`--fixed-split` and was prepared once; no stored run is of the last kind.

How a seed was prepared and how it is rebuilt have to agree even with nothing
held back, because the seed still decides two things. A graph drawn at random, or the paper's graph rewired at
random, is drawn from it (assumption A-41), which is why `scripts/train.py`
refuses `--fixed-split` for those graphs when nothing is held back. And under
min-max scaling per operating condition — the flight conditions of altitude,
speed and throttle, six of them in FD002 and FD004 — the conditions are found
by a clustering method that groups the rows by their three settings starting
from positions picked with the seed, so each seed can number the same six
groups differently. On 2026-10-03, with the code at `6733d77`, nothing held
back and seeds 43 to 46 each set against seed 42, the training windows, labels
and test windows came out the same to the last digit on all four
sub-datasets, under each of the three ways of rescaling the readings
(subtracting each sensor's average and dividing by its spread; one range per
sensor over all conditions; one range per sensor and condition) and, under the
first, with the graph built from correlations; so did the graph and the scaling
table, except
the table under the per-condition ranges on FD002 and FD004. There
the condition numbered 0 covers 13,458 training rows under seed 42 and 8,122
under seed 43 on FD002, and 15,395 and 9,224 on FD004: the same scaled numbers,
filed under different condition numbers. A table saved under one seed is
therefore not the one a reload under another seed rebuilds, and the checks
that compare the two would fail.

(Corrected 2026-10-03: the first version of this paragraph said that with
nothing held back the seed decides no number the preprocessing hands on, and
`scripts/train.py` then still prepared the data once under `--fixed-split`
while recording `"none"`. An independent review of the change found the
scaling tables above the same day.)

`tests/test_examples_torch.py` works both cases through: the FD001 one, where
seeds 42 and 43 agree to the scaling table, and the FD002 one, where they do
not; and checks that `--fixed-split` shares one preparation only when engines
are held back. `tests/test_run_records.py` reloads one stored run of each age
and a copy that records `"none"`, and checks that each gives back its saved
predictions.

#### The graph a reload rebuilds must be the one the network was trained on

The *sensor graph* says which sensors pass numbers to which in the graph
layers. A trained network keeps its own copy of it: the graph, rescaled — each
connection, and each sensor's own reading, turned into the share of that
sensor's incoming numbers it carries, shares which by default add up to one for
every sensor — is saved in `model.pt`
together with the learned numbers, as a fixed table named `adjacency` that
training never changes. `load_seed` builds the graph a second time, with the
data, and hands that one back inside the `PreparedData`, the rebuilt data.

The two copies can disagree without any error, and the network's predictions
would not show it. Loading the weights copies the saved table into the network
over whatever was rebuilt, so the network runs on the graph it was trained on,
while the data handed back beside it carries the other one. Anything that reads
the graph from the data then describes connections the network never used:
`scripts/measure_walks.py`, for one, names every connection from the rebuilt
graph and builds its untrained comparison network on it.

They can disagree because a graph can depend on more than the settings a run
saves. A graph drawn at random, or the paper's graph rewired at random
(assumption A-41), depends on the seed it is drawn from and on settings that
live in the code, such as how many swaps the rewiring proposes, 4,000. Measured
on 2026-10-03:

```
rewire from seed 42 with 4,000 proposals, then with 5,000   →  22 of the 41 connections in common
build the network on the 5,000 graph, load weights saved on the 4,000 one
                                                            →  no error; the network now holds the
                                                               4,000 graph, the rebuilt data the 5,000 one
```

The graph built from correlations depends on the training engines and so on
the NASA files as well.

So since 2026-10-03 `load_seed` compares the two before it loads the weights,
and refuses to go on when they differ:

1. **Against `seed_N/graph.json`**, where the seed folder has one (runs made
   since 2026-10-03): the rebuilt graph must be of the same kind, drawn from
   the same seed, and join the same pairs of sensors, compared by name.
2. **Against the table saved in `model.pt`**, for every run, with or without
   `graph.json`: the rebuilt graph must join the same pairs of sensors, and its
   rescaled numbers must agree with the saved ones to within 0.000001
   (`GRAPH_TOLERANCE` in `stored.py`), since both are the same arithmetic on
   the same graph.

The refusal says how many connections differ and names up to three of each
kind. With the rewiring above changed to 5,000 proposals after a run was
trained with 4,000, the 41 − 22 = 19 connections lost are counted, and so are
the 19 that replaced them:

> ..., seed 42: the sensor graph rebuilt for this seed is not the one recorded
> in seed_42/graph.json: 19 of its 41 connections are missing from the rebuild
> (T24-Ps30, T24-htBleed, T24-W32, ...), which joins 19 other pair(s) instead
> (T24-NRf, T24-BPR, T24-W31, ...). Something that decides this graph has
> changed since the run was made: the code, the run's settings file or the
> NASA files. The data handed back would not be the data the network was
> trained on, so rebuild it from a checkout of the run's commit, ..., or train
> the configuration again.

`tests/test_stored.py` writes such a run from the real FD001 files, with and
without its `graph.json`, and checks both refusals; it also checks that a
`graph.json` naming another kind of graph, or another seed, is refused. It
then checks the two refusals only the second comparison can make, since
`graph.json` names pairs and nothing else. One is the same pairs rescaled by
the other formula the settings allow (assumption A-21), which does not make
every sensor's shares add up to one: on that run's graph it moves some of the
rescaled numbers by up to 0.0509, and the refusal names that gap. The other is
a table saved for 13 sensors when the rebuild has 14. Last, a `graph.json` one
connection short is refused with the pair the rebuild adds named.

Every stored run passes. Measured on 2026-10-03 over every network then in
`reports/`, 469 networks in 95 run folders (counting the runs kept inside the
sensitivity studies' folders), on the paper's graph, the complete graph and the
graph built from correlations: the comparison `load_seed` makes refused none
of them, each rebuilt graph joined exactly the pairs saved with its network,
and its rescaled numbers differed from the saved ones by 0.0 at most. None of
them has a `graph.json` yet, since `scripts/train.py` began writing one that
day, so for now the second comparison is the one that runs.

(Corrected on 2026-10-03, the day it was written: this paragraph first said
464 networks in 94 run folders. That count was taken just before one more
five-network run was written to `reports/`, so it left out a run the count of
25 runs in the next section includes. A reviewer found the difference by
counting the folders again; the comparison was then run again over all 469.)

#### A stored run must still reproduce

A number in the documents is quoted from a stored run, and a stored run is a
network plus the data rebuilt for it. If the preprocessing or the network code
later changes what it computes, the stored number silently stops describing the
program. `tests/test_run_records.py` checks, for every run in `reports/` that
saved its networks, that today's code rebuilds each seed's data, reloads its
weights, and gives the saved test predictions back to within 0.001 cycles —
the file keeps a handful of decimals, so an exact match is not expected — and,
where the run recorded them, the same engine split, the same scaling statistics
and the same data files.

On 2026-09-15 it passed for all eight stored runs with saved networks, 34
networks in all. On 2026-10-03, reloading through the graph comparison of the
previous section, it passed for all 25 runs then at the top level of
`reports/` that list their seed folders and saved each seed's network and
predictions, 119 networks in all; the test does not look inside the
sensitivity studies' folders.

When it fails, the run no longer describes the code. Either the change was a
mistake, or it was intended and the run has to be trained again before anything
from it is quoted. With no stored runs, as on the automated checks on GitHub, it
skips.
