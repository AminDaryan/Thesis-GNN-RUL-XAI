# Planned experiment — does A-07's choice of node feature matter?

**Status:** not started. Written 2026-08-31.

**Instruction to a future session:** carry out exactly what is below. Do not
improvise a different comparison, and do not change the default.

---

## 1. The question, in one sentence

Each sensor hands the first graph layer **one number**. Today that number is
the **last cycle** of the window. Would the **mean of the window** do better,
worse, or the same?

## 2. Why it is worth eight minutes

**A-07 argues for the last cycle; it has never measured it against the mean.**
The figures A-07 quotes (20.35 against 13.45) compare the last cycle against
the removed `window` mode, which is a different question — that mode fed the
graph all thirty cycles and so re-introduced the temporal encoder A-11
declines to invent. The mean does not. It keeps `W(1)` at `1 x 32`, keeps the
parameter count at 61 729, and keeps Table 4 of the paper intact. It is
therefore the only rival A-07 has that costs nothing to adopt.

**F-12 makes the gap conspicuous.** None of the twelve neighbouring papers
uses either the last cycle or the mean, so the literature cannot settle it.
Only a measurement can.

**Either outcome is worth having.** If the two land inside the seed-to-seed
spread, A-07 can stop resting on an argument and say "measured, and it does
not matter" — a stronger entry than it has now. If they differ, that is a real
result about how much this reproduction depends on a detail the paper never
specifies.

## 3. What must not change

- **The default stays `last`.** Whatever the measurement says, changing the
  default is a separate decision for the author, under rule 8.
- **No new learned parameters.** If the change adds a single weight, it is the
  wrong change — see the trap in section 7.
- **`window_size` stays 30**, the graph stays `physics`, the scaling stays
  `zscore`, the adjacency normalisation stays `row`. This experiment varies
  exactly one thing.

## 4. The code change, file by file

Five files, roughly thirty lines. The work is threading one setting from the
configuration down to the single line that currently takes the last cycle.

### 4.1 `src/thesis_gnn_rul_xai/config.py`

Add a type beside the others near the top of the file:

```python
NodeFeature = Literal["last", "mean"]
```

Add a field to `PreprocessingConfig`, directly after `window_stride`, carrying
the assumption comment rule 4 requires:

```python
    # ASSUMPTION A-07: give each sensor one number, the most recent reading of
    # its window. The paper's Table 4 prints one number per sensor going into
    # the first layer; in the paper that number leaves a stage this project
    # does not implement (A-11). Set to "mean" to average the window instead,
    # which keeps the same shape and the same parameter count.
    # Written up in docs/ASSUMPTIONS.md.
    node_feature: NodeFeature = "last"
```

Add validation inside `__post_init__`, in the style of the checks already
there:

```python
        if self.node_feature not in ("last", "mean"):
            raise ValueError(
                f"node_feature must be 'last' or 'mean', got {self.node_feature!r}."
            )
```

### 4.2 `src/thesis_gnn_rul_xai/data/windowing.py`

Replace the single line inside `_extract_one_unit` that takes the last cycle.
It currently reads:

```python
    windows = windows[:, :, -1:]
```

It becomes:

```python
    if node_feature == "mean":
        windows = windows.mean(axis=2, keepdims=True)
    else:
        windows = windows[:, :, -1:]
```

Keep the existing `ASSUMPTION A-07` comment above it and extend it to name
both options in words.

Then add `node_feature: str` as a parameter to `_extract_one_unit`, to
`_build`, and as a keyword argument on both `make_windows` and
`make_last_windows`, passing it down at each step. **The default must be
`"last"` at every level**, so that every existing caller and every existing
test keeps its current behaviour untouched.

### 4.3 `src/thesis_gnn_rul_xai/data/pipeline.py`

In step 8, pass the setting to **both** calls:

```python
    train_windows = make_windows(
        train, columns,
        window_size=config.window_size,
        stride=config.window_stride,
        node_feature=config.node_feature,
    )
    test_windows = make_last_windows(
        test, columns,
        window_size=config.window_size,
        node_feature=config.node_feature,
    )
```

Both must get it. Training on means while testing on last cycles would be a
silent bug that still produces plausible-looking numbers.

### 4.4 `scripts/train.py`

Add the flag beside the other override flags:

```python
    parser.add_argument("--node-feature", choices=("last", "mean"), default=None,
                        help="what one number each sensor carries: the last "
                             "cycle of its window (default) or its mean.")
```

and apply it where `--graph` and `--adjacency-norm` are applied, **before** the
`preprocessing.__post_init__()` re-validation call:

```python
    if arguments.node_feature is not None:
        preprocessing.node_feature = arguments.node_feature
```

### 4.5 `src/thesis_gnn_rul_xai/data/README.md`

Describe the new setting in prose, per rule 9: what each option means, and
that the choice behind it is A-07.

### 4.6 Tests

Add to `tests/test_windowing.py`:

- the mean option returns the same array shape as the last option;
- on a small hand-built frame with known values, the mean option returns the
  arithmetic mean of each sensor's window;
- the default is `"last"`, so existing behaviour is unchanged.

Add one worked example to `tests/test_examples.py`, per rule 9.

## 5. The commands

Baseline and variant, five seeds each. A baseline at the current commit may
already exist in `reports/` — check before spending eight minutes regenerating
it.

```bash
python scripts/train.py --seeds 5
```

```bash
python scripts/train.py --seeds 5 --node-feature mean
```

Then the checks CLAUDE.md requires:

```bash
ruff check .
```

```bash
pytest
```

```bash
python scripts/verify_claims.py --fast
```

**Rule 3a is handled for you.** The run directory name contains a digest of
the whole preprocessing configuration, so the two runs cannot collide, and
each records the commit that produced it. Nothing needs renaming by hand.

## 6. How to decide what the answer is

Take the mean test RMSE and its standard deviation over the five seeds from
each run's `summary.json`.

- **If the two means differ by less than the larger of the two standard
  deviations**, the honest verdict is **"no measurable difference"**. Say
  exactly that. Do not report the smaller number as the better one.
- **If they differ by more**, run the same significance test used for A-21,
  A-02 and F-04, quote the p-value, and name the configuration the result
  belongs to.

Rule 3 applies: a comparison is a fact about one architecture, not about the
two options in general. Whatever comes out is true of this code at this commit
and may reverse if the architecture moves again — as F-04 has twice.

## 7. The trap to avoid

Do **not** implement this as a learned reduction. A layer that maps thirty
cycles to one number using trainable weights is a temporal encoder, which is
exactly what A-11 refuses to invent and exactly what got the `window` mode
removed on 2026-08-28. `numpy.mean` has no parameters, and that is the whole
point: the parameter count must still be exactly **61 729** afterwards.
Assert it.

Do not silently widen the experiment to median, maximum, or a first-and-last
pair. Those are separate questions and each would need its own entry.

## 8. What to write when it is finished

1. **Update A-07** in `docs/ASSUMPTIONS.md`. Its "How much it matters" section
   currently answers a different question; give it the mean-versus-last number,
   with both standard deviations and the command that produced it. Delete the
   sentence saying the mean has never been measured.
2. **If the difference is significant**, add a finding recording it, naming the
   configuration it belongs to.
3. **Leave the default alone** and tell the author what the number was.
   Under rule 8, changing it is their decision.
4. **Do not commit.** Report what changed, then stop.
