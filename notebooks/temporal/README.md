# `temporal/` — the temporal stage, one notebook per stage

The network in notebook 02 receives, for each of the 14 sensors, the readings
of the last 30 flight cycles. Before its graph layers mix each sensor with the
sensors it is joined to, something has to decide what each sensor's node
carries into them. That is the **temporal stage**: it receives 30 readings per
sensor and hands on `k` numbers per sensor. The graph layers after it never see
a cycle again, so whatever the stage drops is gone for the rest of the network.

```
(batch, 14 sensors, 30 cycles)  →  temporal stage  →  (batch, 14 sensors, k)  →  graph layers
```

The paper's stage is a small recurrent network for each sensor, one that reads
the sensor's 30 readings one cycle at a time (its Section 2.2). This project's
default keeps only the newest reading instead (assumption A-07), and four more
stages are built. A stage cannot produce a result on its own: it needs the
graph layers after it and training. So each notebook here shows one stage two
ways — what it does to one sensor's 30 real readings, worked through by hand,
with what it costs in learned numbers; and the stored five-seed results of the
networks trained with it — and the last notebook sets those results side by
side.

Read `notebooks/README.md` first, for how every notebook is laid out, and
notebook 02 before any of these, for what the network is. The stages
themselves are in `src/thesis_gnn_rul_xai/models/temporal/`, whose README
explains each one in code.

## The notebooks

| Notebook | The stage | What each sensor's node receives | `k` | Where it comes from |
|---|---|---|---|---|
| `01_last_cycle_fd001.ipynb` | `last_cycle` | its newest reading | 1 | this project's default, standing in for the paper's recurrent stage (assumptions A-07 and A-11) |
| `02_mean_window_fd001.ipynb` | `mean_window` | the average of its 30 readings | 1 | the other choice that keeps the paper's one number per sensor (assumption A-07) |
| `03_window_summary_fd001.ipynb` | `window_summary` | its newest reading, its average, and its change across the window | 3 | this project's (assumption A-28) |
| `04_lstm_fd001.ipynb` | `lstm` | one number from a trained LSTM, a long short-term memory network, of its own | 1 | the paper's Section 2.2; its size is this project's (assumption A-26) |
| `05_gru_fd001.ipynb` | `gru` | one number from a trained GRU, a gated recurrent unit, of its own | 1 | not in the paper; the LSTM stage's size with a simpler recurrent network (assumption A-36) |
| `06_raw_window_fd001.ipynb` | `raw_window` | all 30 readings, untouched | 30 | the arrangement removed on 2026-08-28, kept behind a gate (assumption A-07) |
| `07_comparison_fd001.ipynb` | every stage with a stored run | — | — | their stored results side by side, paired by seed |

Read 01 first: it explains what a stage receives and returns. 02 and 03 build
on it, and 03 on 02. 05 follows 04 step for step and uses its words. 06 uses
03's three numbers. 07 comes last.

## The words used here

- A **reading** is one sensor's value at one cycle as the network receives
  it. The raw measurement has been smoothed, so that each reading is half the
  cycle's raw value plus half the previous smoothed reading (assumptions A-04
  and A-09), and then rescaled: the sensor's average is subtracted and the
  result divided by its standard deviation, the typical size of its departures
  from that average (assumption A-03). Both are measured on the 80 engines the
  networks train on under the run's seed, never on the 20 held back or on the
  test engines (assumption A-25), so they change a little from seed to seed.
  So a reading of 1.0 means one standard deviation above those 80 engines'
  average, not a temperature or a pressure.
- A **window** is the 14 sensors' readings over 30 consecutive cycles, ending
  at the cycle a prediction is made for: an array of shape `(14, 30)`, one row
  per sensor, oldest cycle first. A reading's **position** is its column, 1 to
  30. Training and prediction hand the network several windows at once, a
  **batch**, of shape `(batch, 14, 30)`.
- A **learned number** is one that training adjusts: a multiplier or an
  offset. Most texts say "parameter".
- A **pass** is one round of training through every training window; most
  texts say "epoch".
- The **held-back engines** are the fifth of the 100 training engines that no
  network trains on, kept to judge it during training. Which 20 they are is
  drawn from the run's **seed**, the number that fixes every random choice in
  a run (assumption A-06).
- A **commit** is one saved state of the project's code. A run records the
  commit it was made at, with `-dirty` added when the code had changes not yet
  saved into a commit (working rule 3a).
- A **stored five-seed run** is one run of `scripts/train.py --seeds 5`: five
  networks, seeds 42 to 46, each trained and scored, with the commit of the
  code that trained them and every setting written beside the results in
  `reports/`.
- The **sensitivity study** is one set of such runs, made on 2026-09-29 at
  commit `fd65a3b-dirty`: eighteen configurations of FD001, five seeds each —
  a reference with the paper's recurrent stage, and seventeen that each change
  one of its settings (finding F-19). Its runs are in
  `reports/sensitivity_fd65a3b-dirty_20260929T101059Z/runs/FD001/`.

## What every notebook here shares, and why

**One window and one sensor.** Every notebook but 07, which reads stored
results only, rebuilds the data from `config/fd001.yaml` with seed 42 and takes
the held-back engine with the most windows — engine 5, with 240 windows, the
engine the `xai/` notebooks follow too — and its last window, cycles 240 to
269, which ends at the cycle the engine failed. A held-back engine, because no network of a seed-42 run trained
on it; its last window, because there the readings change fastest, which is
where the stages differ most; not a test engine, because the test records stop
at a random point before failure (the paper's Section 3.2.1). The sensor
followed is T50, the temperature at the outlet of the low-pressure turbine. In
this window it climbs from 1.29 to 2.74, more than one standard deviation, so
the stages hand its node visibly different numbers. Using the same window
everywhere means a difference between two notebooks is a difference between
two stages.

**One configuration, with only the stage changed.** Every stored run read
here saved the settings of `config/fd001.yaml` with its temporal stage
swapped: FD001; the readings smoothed and rescaled as above; windows of 30 cycles, one cycle
apart; labels capped at 125 cycles, the test labels included (assumptions A-01
and A-02); the paper's sensor graph, each row divided by its total (assumption
A-21); a fifth of the training engines held back, with the parameters of the
pass that did best on them kept and training stopped after 20 passes without
improvement (assumptions A-06 and A-18); up to 200 passes in batches of 32,
the learning rate dropping from 0.001 to 0.0001 at the halfway pass (assumption
A-17); and the graph and dense layer sizes of the paper's Table 4. Runs of
different commits differ in more than those settings: in the device that
trained them, the processor or a graphics card; in the version of PyTorch; and
in the code, committed and not yet committed. Four runs, made on 2026-09-08 and
2026-09-09, saved nothing about their training but the learning-rate path, and
no network sizes; each repeats, seed by seed and to every stored digit, a later
run that saved everything. Each notebook prints these settings beside its
stored runs, and `07_comparison_fd001.ipynb` checks, from the files each run
saved, that runs compared directly differ in the stage among their saved
settings, and that the uncommitted code they recorded agrees outside the
documentation site's folder.

**Stored runs, never training.** None of these notebooks trains a network. A
network trained in a notebook is one seed and records nothing about the code
that trained it (working rules 3 and 3a). The notebooks find stored runs with
`find_runs` and load a trained network with `load_seed`, both in
`src/thesis_gnn_rul_xai/training/stored.py`. Where a stage has no stored
five-seed run, the notebook says so and prints the command that makes one.

**The trained networks the worked examples use.** Notebooks 04 and 05 follow
T50's readings through a trained recurrent network, and both load seed 42 of
the sensitivity study's runs, the only place where both recurrent stages were
trained at one commit (finding F-19). Notebook 06 rebuilds the first graph
layer of seed 42's network from the stored window-summary run. Each loading
cell scores the test engines again and requires the error the run recorded for
that seed.

| Notebook | Stored run, under `reports/` | Commit | Trained on |
|---|---|---|---|
| 04 | `sensitivity_fd65a3b-dirty_20260929T101059Z/runs/FD001/reference` | `fd65a3b-dirty` | a graphics card, PyTorch 2.5.1 |
| 05 | `sensitivity_fd65a3b-dirty_20260929T101059Z/runs/FD001/temporal_gru` | `fd65a3b-dirty` | a graphics card, PyTorch 2.5.1 |
| 06 | `fd001_physics_row_cap_5seed_varsplit_window_summary_cfgba82e3_5602773-dirty_20260914T233435Z` | `5602773-dirty` | the processor, PyTorch 2.13.0 |

Run here on the processor, the two networks trained on a graphics card score
the test engines within a few millionths of a cycle of the error they recorded
— 1.5 millionths for notebook 04's, 0.6 for notebook 05's — and the check
allows 0.0001. Notebook 06's network, trained on the processor with the
PyTorch build used here, scores exactly the error its run recorded.

**How the cost is counted.** A stage costs two things. One is what it drops,
which each notebook shows on the window. The other is learned numbers, counted
part by part under the names of the paper's Table 4: the stage itself, the
three graph layers GCN-1 to GCN-3, and the two dense layers FC-1 and FC-2.
With one number per sensor the network holds 61,729: the five weight matrices
Table 4 prints, 61,600 numbers, plus the 129 offsets of the dense layers that
its Equations 12 and 13 print (`01_last_cycle_fd001.ipynb` adds them up). A
stage that hands on `k` numbers adds `k − 1` rows of 32 to GCN-1's weight
matrix.

| Stage | In the stage | Added to GCN-1 | Whole network |
|---|---|---|---|
| `last_cycle` | 0 | 0 | 61,729 |
| `mean_window` | 0 | 0 | 61,729 |
| `window_summary` | 0 | 64 | 61,793 |
| `lstm` | 62,753 | 0 | 124,482 |
| `gru` | 47,073 | 0 | 108,802 |
| `raw_window` | 0 | 928 | 62,657 |

## What no notebook here establishes

- **Which stage suits an engine.** Every result belongs to the configuration
  it was measured under. When the network changed on 2026-08-28, two of this
  project's three comparisons of settings reversed direction (working rule 3).
- **What the paper's training procedure gives.** Every stored run here keeps
  the pass that did best on the held-back engines, which the paper never
  describes; under the procedure it does describe, the recurrent stage's
  advantage over the newest reading was much smaller and did not survive the
  seed-to-seed spread (finding F-16).
- **What a trained recurrent stage has learned.** Notebooks 04 and 05 show
  what one trained network computes on one window. Why training made it so is
  a different question, and no explanation method has been run on these stages.
- **Anything about another sensor, engine or seed.** The worked examples
  follow one sensor on one window of one engine, through networks of one seed.

## Adding a stage

1. Write it in `src/thesis_gnn_rul_xai/models/temporal/`, as that folder's
   README describes, with its worked example in `tests/test_examples_torch.py`.
2. Train it from one committed state of the code, together with the stages it
   will be compared with:
   `python scripts/train.py --config config/fd001.yaml --temporal NAME --seeds 5`.
3. Copy the layout of `02_mean_window_fd001.ipynb` for a stage with nothing to
   learn, or of `05_gru_fd001.ipynb` for a trained one. Add the stage to the
   list `07_comparison_fd001.ipynb` searches, a row to the tables above, and a
   line to the documentation site's list in
   `src/thesis_gnn_rul_xai/docsite/catalog.py`.
