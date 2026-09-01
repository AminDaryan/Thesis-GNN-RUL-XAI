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

## Validation is optional, and it is off by default

This is the part most likely to trip someone up.

The project's preprocessing settings have a `validation_fraction`, and its
default value is `0.0`. With that default **no engines are held out at all**,
and `train` is called with no validation set. When that happens:

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
comparison. It is nonetheless a decision, because the paper never states it:
**A-13** in `docs/ASSUMPTIONS.md`.

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
  C-MAPSS paper found decays exponentially. See **A-17** and
  `docs/LITERATURE_BASIS.md`.
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
  gives. Since the default configuration holds nothing back, the loop warns
  when you ask for this combination rather than letting the run look normal.
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
choose. All are written up in `docs/ASSUMPTIONS.md` under the codes given.

- **Scaling the targets.** The paper's final layer is a sigmoid, but it never
  states the arithmetic that turns that number back into flight cycles or what
  the training targets look like. We divide targets by `rul_max` and multiply
  predictions back by it. That choice is recorded as **A-13**.

- **How the learning rate decays.** The paper gives both endpoints and calls it
  an adaptive schedule, but never the mechanism. We picked exponential decay,
  with the factor chosen so the last epoch lands exactly on the final rate; the
  other three options exist so the sensitivity to that choice can be measured.
  That choice is recorded as **A-17**.

- **Early stopping.** The paper trains for a fixed number of epochs with no
  validation set and no early stopping described. Stopping when the validation
  error has stopped improving, and restoring the best epoch's weights, is a
  deliberate departure from the paper rather than a reconstruction of it, and
  has to be described that way. That choice is recorded as **A-18**.

- **Holding engines out at all.** Whether to have a validation set is itself a
  departure, since the paper has none. That choice is recorded as **A-06**, and
  it is the reason validation error and test error must be kept apart, as
  described above.

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
| `lr_schedule` | How the rate travels between the first value and the last, one of the four above. **A-17.** |
| `early_stopping_patience` | Stop when the held-out error has not improved for this many epochs, and go back to the best epoch's weights. `None` runs every epoch. **A-18.** |
| `min_improvement` | How far the held-out error has to fall, in cycles, before it counts as an improvement worth restarting the count for. |
| `gradient_clip` | A limit on how large one adjustment may be, or `None` to leave them alone. The paper never mentions it, so it is `None` by default and the run matches. |
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

### The four functions

| Name | What it does |
|---|---|
| `train(model, train_set, validation_set=None, config=None)` | Trains the network and returns a `TrainingHistory`. The model is changed in place and ends holding the weights of whichever epoch the loop chose. |
| `predict(model, dataset, batch_size=256, device=None)` | One prediction per window, in flight cycles, in the order the windows are already in. |
| `evaluate_model(model, dataset, device=None)` | Predicts, then scores. Returns the same dictionary as `evaluate` in `evaluation/metrics.py`. |
| `resolve_device(preference="auto")` | Decides between the graphics card and the processor, and logs which. Asking for `"cuda"` when PyTorch cannot see one raises rather than falling back quietly. |
