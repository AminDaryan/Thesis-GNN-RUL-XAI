# Swappable temporal, graph and explanation stages

**Status:** partly built. Written 2026-09-09, and steps 1, 2, 3 and 7 of the
migration in section 6 were carried out the same day: `models/temporal/` with
five stages, `ESTGNN` taking one and reading its width from it, the pipeline
handing over the whole window, and the paper's per-sensor LSTM. The step-3
gate held — the five-seed run reproduced bit-identically, `best_epoch`
included. Everything below is as first written, so read section 6 as the plan
it was rather than as a record of what happened.

*Note added 2026-10-03: what was built, what was deferred, and one fix.* This
paragraph said until today that steps 4, 5, 6, 8 and 9 were not built, and
pointed at an entry of the pending-work list. Step 4 had been built on
2026-09-14 (commit `c8ba1f1`), in a narrower form than section 6 describes:
`models/config.py` records the network's shape, a run whose network differs
from Table 4's sizes carries `_net` and a digest in its folder name (not
`mdl`), and every seed's weights, per-pass record and test predictions are
saved; there is no `config/model/table4.yaml`, and `matches_table_4()` and
`comparable_paper_row()` were never written. The error was found on
2026-10-03 by setting this paragraph beside the pending-work list in
`docs/PENDING.md`, which recorded step 4 as done, and confirmed in the
repository's history: `models/config.py` first appears in commit `c8ba1f1`,
dated 2026-09-14. On 2026-10-02 the author decided
to build none of the remaining steps for now. Each is deferred, not dropped,
and building it would take the following:

- **Step 5, the graph layers as a named stage.** Moving the three graph layers,
  and the check that refuses a graph with a zero on its diagonal, out of
  `ESTGNN` into a `models/spatial/` package, then showing with a test of the
  starting weights and a second five-seed run that no number moved; it adds no
  new capability on its own.
- **Step 6, the stage that hands every cycle to the graph layers, measured
  again.** The stage and its gate already exist as `--temporal raw_window`,
  marked `_raw_window` in the folder name; what is missing is a five-seed run of
  it under today's configuration and an entry under the next free finding
  number recording the result, because the measurements of it recorded under
  assumption A-07 were made under settings that have since changed.
- **Step 8, a graph-attention stage and a stage with no graph layers.** Two new
  classes, an assumption entry under the next free number for each (how many
  attention heads, and which graph decides who attends to whom; what feeds the
  fully connected layers once the graph layers are gone), and five-seed runs of
  both, after which the paper's "ESTGNN w/o GCN" row of Table 11 (16.42 RMSE
  on FD001) could be set beside a measured number for the first time.
- **Step 9, explanations of connections and of parts of the graph.** A common
  record saying what each explanation is about, a sensor, a connection or a
  walk, so that values of different kinds are never compared, and Shapley
  values over the paper graph's 41 connections; those cannot be computed
  exactly, since 41 connections have 2 to the power 41 (2,199,023,255,552)
  subsets, so they need sampling, a standard error reported beside each value,
  and an assumption entry of their own. Three pieces of it were built since
  this design was written: `scripts/explain.py`, which explains the
  predictions of one seed of a stored run, sensor by sensor or walk by walk,
  and saves the result in that run's folder (first added in commit `ffdaad6`,
  2026-09-14); explaining walks through the
  graph (assumption A-29); and removing one connection by deleting it from the
  graph before rescaling, the choice section 5 proposed as a provisional A-31
  (assumption A-30), in `explain/connections.py` rather than the
  `explain/edges.py` section 6 names. What remains is the common record
  (`Attribution` and `compare_attributions` in section 6, neither of which
  exists) and the sampled Shapley values over connections. The provisional
  A-32, the sampling over connections, takes the next free number if step 9 is
  ever built.

The fix: the defect in section 7, `PreprocessingConfig.notes` inside the run
digest, was fixed on 2026-10-03 at the author's decision of 2026-10-02. The
digest now leaves the notes out, so `PreprocessingConfig()` and
`config/fd001.yaml` both give `818b40`; the `e3a331` and `ba82e3` quoted in
sections 0, 4, 6 and 7 come from the rule before then. The log's entry on how
runs are named (finding F-09 in `docs/ASSUMPTIONS.md`) records the change.

## What this is

The author asked how to add the paper's LSTM stage while keeping the current
no-LSTM model as a first-class option, and beyond that how to arrange the code
so the temporal stage, the graph stage and the explanation method can each be
swapped and compared. This is the answer, and it is a proposal: **none of it has
been built.**

Three designs were worked up independently against this repository -- one aiming
at the smallest safe diff, one at running a matrix of combinations without
losing track of which produced which number, one at what this repository's own
conventions already imply -- each was judged against the code it claimed to
change, and this is the synthesis. The spine is the third; the other two
contributed the regression gate, the per-seed saved model, and the attribution
type that refuses to compare fourteen sensor values with forty-one edge values.

## What has been checked, and what has not

Re-measured by hand on 2026-09-09 before this document was written:

| Claim | Verified |
|---|---|
| `ESTGNN` at today's defaults holds 61,729 parameters | yes |
| `nn.LSTM(1, 32)` costs 4,480 parameters, `nn.GRU(1, 32)` costs 3,360 | yes |
| `PreprocessingConfig()` and `load_config("config/fd001.yaml")` agree on every setting **except the `notes` prose**, and still digest differently | yes -- the defect in section 7 is real |
| `figures.py` lays out on `_WIDTH = 860.0` | yes |
| Five stored runs already carry `node_feature_mode` in their summaries | yes |
| `"14, 1)"` appears 73 times; `in_features=` has 25 Python call sites | yes, re-counted |

Everything else below -- the layout, the interfaces, the nine migration steps --
is design, not measurement. **Read it as a proposal to argue with**, and check
any number it quotes before repeating it, per rule 2.

## The one thing to take away if you read nothing else

The reduction from thirty cycles to a node feature must happen **inside the
model**, in a named stage that says it is doing it, and not in `data/windowing/`
where it currently happens and looks like a property of the data. That single
move is what makes the LSTM addable, and it is what closes the hole that
produced the 2026-08-28 mistake: with `in_features` read off the data, there is
a configuration path from a preprocessing setting to a thirty-wide first graph
matrix, and nothing in between is named.

---

I read the files, then verified every number I use below by running it.

---

# RECOMMENDATION

**Spine: `codebase-fit`.** It is the only one of the three whose shape is the shape this repository has already proven three times (`normalize_adjacency(method=)`, `config.graph`, `TrainingConfig.lr_schedule`), and it is the only one that closes A-07 structurally by deleting `in_features` as an argument. Grafted on: `minimal-risk`'s weight-hash regression gate and its `comparable_paper_row` mechanism; `research-throughput`'s saved `model.pt` per seed, its `Attribution` type with a `target` field, and its measured finding that exact Shapley values do not survive the move to edges. Discarded: `research-throughput`'s `experiments/` folder and sweep machinery (too large for one migration, and its frozen-dataclass and `notes`-in-digest problems are real), and `minimal-risk`'s decision to keep the reduction inside `data/windowing/`.

## 0. What I verified before writing (so you can re-check me)

| Claim | Result |
|---|---|
| `ESTGNN` default parameters, 7 `layer_summary()` rows | **61,729**, **7 rows**, torch 2.13.0+cpu |
| Weight digest, `torch.manual_seed(42)`, physics graph, row normalisation | `ceff1a8c04f9edd1…` (first 16 hex of sha256 over concatenated parameters) |
| `nn.LSTM(input_size=1, hidden_size=32)` / `nn.GRU(…)` | **4,480** / **3,360** parameters |
| Physics graph | **41 undirected edges**, degrees 4–8; T24 has **7**, T30 has **8** |
| `normalize_adjacency` adds `np.eye` **before** summing degrees (line 127 then 129) | true — deleting all of a node's edges still leaves degree 1 |
| `PreprocessingConfig()` digest today | **`e3a331`** — no stored run carries it |
| `load_config("config/fd001.yaml")` digest today | **`ba82e3`** — matches stored runs |
| Why they differ | the `notes` **prose string** is inside `to_dict()` and therefore inside the digest |
| `reports/…cfgba82e3_ae59bc1-dirty_20260908T223607Z` | mean RMSE **16.4965 ± 0.4153**; per-seed 16.6949, 16.1311, … |
| README's headline **16.82 ± 0.16** | belongs to `cfg00be26`, which sets `validation_fraction: 0.0` — **not** today's default |
| `reports/` | 38 training runs, 24 `verification_*`; **four** `_window_` runs and one `_last_step_` run |
| Stored vocabulary in those five summaries | `"node_feature_mode": "window"` and `"last_step"` — **not** absent, as one design assumed |
| `docsite/results.py` glob | `reports.glob("*/summary.json")`, one level deep, at lines 110 and 160 |
| `figures.py` | `_WIDTH = 860.0`; 7 boxes × 96 + 6 gaps × 24 = 816, so `left = 22`. **Eight boxes need 936 and `left` goes to −38** |
| `check_prose.py:140` | `rf"\b{re.escape(word)}"` — no closing boundary, so adding `head` would fire on "header"/"headline" |
| `in_features=` call sites | 25 in Python files, plus the notebooks; `tests/test_models.py` alone has 11 uses of `in_features=30`, two of which (`:146`, `:173`) call `model.graph_layers[0]` directly with `(1, 14, 30)` |
| `"14, 1)"` in prose and code | **73 occurrences** (re-counted 2026-09-09, including the notebooks) |
| Highest entries in `docs/ASSUMPTIONS.md` | **A-25** and **F-15** (`tests/test_docsite.py:185` asserts they are contiguous from 01) |
| `docs/EXPERIMENT_A-07_mean_window.md` | exists, 234 lines, "**Status:** not started", and specifies `PreprocessingConfig.node_feature` — this design overrules it and must say so |

---

## 1. THE SHAPE

The network becomes three named stages instead of two anonymous ones, and each stage is chosen the same way this repository already chooses its adjacency formula: a `Literal` of allowed names, a `__post_init__` that rejects anything else, and one builder function with an `if`-chain. **The temporal stage owns the time axis and nothing else may touch it** — `data/windowing/` stops throwing 29 of the 30 cycles away and hands the model the whole window, so the shape of the data is a fact about the data; the model then says what it does with those 30 cycles, and the width of the first graph weight matrix `W(1)` is read off the *stage* (`temporal.out_features`) rather than off the *data* (`features.shape[2]`), which is the line that made the 2026-08-28 mistake possible. A stage that widens a node beyond one number **without holding any parameters** is, by construction, handing raw cycles to the graph layer, so `ESTGNN` computes that fact from the object it was given, refuses to build unless the caller passed `allow_time_in_the_graph_layer=True`, and stamps `foldedtime` into the run directory name when it does. A new `ModelConfig` sits beside the existing `PreprocessingConfig` and `TrainingConfig`, is digested into the run name as `mdl<six hex>`, and answers one question per run in `summary.json`: is this the architecture Table 4 prints, and if not, which published row (if any) it may be set beside. The explanation axis is deliberately **not** a symmetric fourth swap: an `Attribution` object carries what it attributes to — a sensor, an edge, a subgraph — and `compare_attributions` refuses to put 14 sensor values beside 41 edge values rather than computing a correlation about nothing.

---

## 2. THE LAYOUT

```
Thesis GNN RUL XAI/
├── config/
│   ├── fd001.yaml                                UNCHANGED
│   ├── fd001_uncapped.yaml                       UNCHANGED
│   │   (fd001_validated.yaml stood here; deleted 2026-09-14, identical to fd001.yaml)
│   └── model/                                    NEW  (one file per architecture)
│       ├── table4.yaml                           NEW  ModelConfig() written out
│       ├── lstm32.yaml                           NEW  step 5
│       └── gat.yaml                              NEW  step 6
│
├── scripts/
│   ├── train.py                                  CHANGED — build_model() deleted
│   ├── verify_claims.py                          CHANGED — _trained_model() at :306
│   ├── compare.py                                NEW  step 5 (promotes A-21's hand-run test)
│   ├── explain.py                                NEW  step 7 (loads a stored seed's model.pt)
│   ├── prepare_data.py                           UNCHANGED
│   ├── check_prose.py                            CHANGED — WORDS gains entries (see §7)
│   └── docs_ui.py                                UNCHANGED
│
├── src/thesis_gnn_rul_xai/
│   ├── config.py                                 CHANGED — one comment on window_size only
│   │
│   ├── data/
│   │   ├── windowing/__init__.py                 CHANGED — the A-07 slice leaves
│   │   ├── windowing/README.md                   CHANGED — substantial
│   │   ├── pipeline/__init__.py                  CHANGED — diagnostics gains one key
│   │   ├── pipeline/README.md                    CHANGED — 8 shape mentions
│   │   └── (loading, sensors, labeling,
│   │        smoothing, normalization)            UNCHANGED
│   │
│   ├── graph/
│   │   ├── adjacency.py                          UNCHANGED
│   │   └── README.md                             CHANGED — one section: graph vs spatial_stage
│   │
│   ├── models/
│   │   ├── README.md                             CHANGED — rewritten as an overview
│   │   ├── __init__.py                           CHANGED — re-exports
│   │   ├── config.py                             NEW   ModelConfig, load_model_config
│   │   ├── estgnn.py                             CHANGED — assembly only
│   │   ├── layers.py                             UNCHANGED  (Layer-1 reading-list file)
│   │   ├── temporal/                             NEW   step 1a
│   │   │   ├── README.md                         NEW   the longest new document
│   │   │   ├── __init__.py                       NEW   TemporalStage, build_temporal_stage,
│   │   │   │                                           LastCycle, MeanWindow, RawWindow
│   │   │   └── recurrent.py                      NEW   step 5: LSTMStage, GRUStage
│   │   └── spatial/                              NEW   step 3
│   │       ├── README.md                         NEW
│   │       ├── __init__.py                       NEW   SpatialStage, build_spatial_stage,
│   │       │                                           GCNStage, NoSpatialStage
│   │       └── attention.py                      NEW   step 6: GATStage
│   │
│   ├── explain/
│   │   ├── README.md                             CHANGED — rule 10.13's seven questions per method
│   │   ├── __init__.py                           CHANGED — re-exports
│   │   ├── shap.py                               UNCHANGED as a file; NOT moved (see below)
│   │   ├── config.py                             NEW   ExplanationConfig
│   │   ├── attribution.py                        NEW   Attribution, compare_attributions, EXPLAINERS
│   │   └── edges.py                              NEW   step 7
│   │
│   ├── training/
│   │   ├── run.py                                CHANGED — TrainingHistory.save() finally called
│   │   ├── config.py                             UNCHANGED
│   │   └── README.md                             CHANGED — one paragraph on what is stored
│   │
│   ├── evaluation/                               UNCHANGED
│   ├── utils/                                    UNCHANGED
│   └── docsite/
│       ├── catalog.py                            CHANGED — _MODEL_MODULE_PAGES
│       ├── results.py                            CHANGED — Run gains three fields
│       ├── figures.py                            CHANGED — _WIDTH, and a new box
│       └── server.py                             CHANGED — one ESTGNN(...) call
│
├── docs/
│   ├── ASSUMPTIONS.md                            CHANGED — A-07, A-11 amended; A-26…A-32 added
│   ├── EXPERIMENT_A-07_mean_window.md            CHANGED — §4.1/4.2 superseded, §5–8 stand
│   └── learn/04_the_pipeline.md, 05_the_model.md CHANGED — shapes
│
├── notebooks/01_…, 02_…                          CHANGED — re-execute both
└── tests/
    ├── test_models.py                            CHANGED — 11 in_features=30 sites
    ├── test_temporal.py                          NEW
    ├── test_spatial.py                           NEW   step 3
    ├── test_windowing.py, test_pipeline.py,
    │   test_shap.py, test_training.py,
    │   test_docsite.py                           CHANGED
    └── test_examples.py, test_examples_torch.py  CHANGED — one worked example per new function
```

**`explain/shap.py` is not moved into `explain/shap/`.** A-20, A-24, F-05 and F-08 all cite the import path `thesis_gnn_rul_xai.explain.shap`, and `docsite/catalog.py:258` hard-codes `if folder == "data":` for the folder-per-file arrangement. Splitting `explain/` into folders is the right move when it holds four methods; it is not the right move in the same change as adding the first one. `explain/README.md` stays one document with sections, exactly as `models/`, `graph/` and `training/` do today.

**`models/layers.py` is not moved.** `GraphConvolution` is 43 lines and is named in README section 9's Layer-1 reading list ("read these first; each stands alone"). `models/spatial/__init__.py` imports it.

---

## 3. THE SEAMS

### 3.1 The temporal stage

`src/thesis_gnn_rul_xai/models/temporal/__init__.py`:

```python
class TemporalStage(nn.Module):
    """Turns one sensor's window of cycles into the numbers its node carries."""

    # Both are bare annotations with no default on purpose: a stage that forgets
    # to set one raises AttributeError the first time it is used, rather than
    # inheriting a value that happens to be safe here and wrong later.
    window_size: int          # how many cycles it consumes
    out_features: int         # how many numbers it puts on each sensor node

    def forward(self, windows: Tensor) -> Tensor:
        """(batch, nodes, window_size) -> (batch, nodes, out_features)."""
        raise NotImplementedError


class LastCycle(TemporalStage):
    """Keep the most recent cycle of each window and drop the rest. No parameters."""

    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.out_features = 1

    # ASSUMPTION A-07: each sensor node carries one number, the last cycle of
    # its 30-cycle window. Table 4 prints one number per sensor entering the
    # first graph layer; in the paper that number leaves a per-sensor recurrent
    # stage this project does not build (A-11), so the latest reading stands in
    # for it. This class is where that choice is made. It used to be made in
    # data/windowing/, where it looked like a property of the data rather than
    # a choice about the model. Written up in docs/ASSUMPTIONS.md.
    def forward(self, windows: Tensor) -> Tensor:
        return windows[:, :, -1:]


class MeanWindow(TemporalStage):
    """Average each sensor's window into one number. No parameters."""
    # ASSUMPTION A-07: the other option that keeps Table 4's printed shape and
    # its 61,729 parameters. Never measured; docs/EXPERIMENT_A-07_mean_window.md
    # is the standing instruction for measuring it.


class RawWindow(TemporalStage):
    """Hand the graph layer all 30 cycles. This is NOT a temporal encoder."""
    # See §5. ESTGNN refuses to build with this unless the caller says so in
    # words, and stamps the run when it does.


class LSTMStage(TemporalStage):          # step 5, in recurrent.py
    def __init__(self, *, window_size: int, hidden_size: int = 32,
                 num_layers: int = 1, out_features: int = 1,
                 n_nodes: int = 14, shared_across_sensors: bool = False) -> None:
        ...
        self.window_size = int(window_size)
        self.out_features = int(out_features)
        count = 1 if shared_across_sensors else n_nodes
        self.cells = nn.ModuleList(
            nn.LSTM(input_size=1, hidden_size=hidden_size,
                    num_layers=num_layers, batch_first=True)
            for _ in range(count)
        )
        self.project = nn.Linear(hidden_size, out_features)

    def forward(self, windows: Tensor) -> Tensor:
        batch, nodes, width = windows.shape
        if self.shared_across_sensors:
            flat = windows.reshape(batch * nodes, width, 1)
            last = self.cells[0](flat)[0][:, -1, :]              # (batch*nodes, hidden)
            return self.project(last).reshape(batch, nodes, self.out_features)
        parts = [self.cells[i](windows[:, i, :, None])[0][:, -1, :] for i in range(nodes)]
        return self.project(torch.stack(parts, dim=1))
```

Shapes, with the real numbers: in `(B, 14, 30)`, out `(B, 14, out_features)`. `LastCycle` and `MeanWindow` give `out_features = 1`, so `W(1)` stays `1 × 32` and the model stays at 61,729 parameters. An `LSTMStage` at hidden size 32 costs **4,480** parameters shared across the fourteen sensors, or **62,720** with one per sensor, plus **33** for the projection down to one number — that is, either 4,513 or 62,753 on top of a 61,729-parameter model, so the per-sensor reading **roughly doubles it**. Both figures come from `sum(p.numel() ...)` run above, not from arithmetic on paper.

> ### Settled from the paper, 2026-09-09 — the LSTM is one per sensor
>
> Read from the PDF, Section 2.2 "Parallel long short-term memory networks",
> page 4. The paper is explicit and says it four times:
>
> - "**Each sensor s∈{1, 2, …, n} is associated with a dedicated LSTM network**
>   that processes the temporal sequence to extract sensor-specific temporal
>   features", with Equation (7) reading
>   `h_t^(s) = LSTM([x_{t−w+1,s}, x_{t−w+2,s}, ..., x_{t,s}])` — indexed by
>   sensor `s`.
> - "parallel LSTM networks are employed to process individual sensor channels
>   **independently**, thereby preserving the physical significance of each
>   measurement type".
> - "This parallel architecture processes different sensor types through
>   **separate specialized pathways**".
> - Among the architectural strengths: "(1) Modularity: **Independent parallel
>   LSTM networks for each sensor** preserve physical significance".
>
> So `shared_across_sensors=False` is the paper's architecture. This is a
> **rule 5 "the paper says X" statement**, not a choice, and it needs a citation
> rather than an assumption entry.
>
> **What the paper still does not say, and what therefore stays an assumption.**
> Table 4 has **no LSTM row at all** — it begins at "Input | Node features |
> [batch size,14,1]", so the recurrent stage sits entirely outside the only
> architecture table the paper prints. Table 6's hyperparameters are the
> learning rate, 200 epochs, batch size 32, 30 time steps, Adam and MSE loss —
> **no hidden size, no layer count, no output projection.** The word "hidden"
> appears three times in the whole paper: once in the literature review and
> twice inside the gate equations, meaning `h_{t-1}`, the previous hidden state.
>
> That leaves exactly the gap A-11 records, and it is narrower than "the LSTM is
> unspecified": the paper specifies **how many** networks (fourteen, one each)
> and **what goes in** (a 30-cycle window of one sensor), and never specifies the
> width of the hidden state, the depth, or how that hidden state becomes the one
> number Table 4 feeds to `W(1) ∈ ℝ(1×32)`. Those three are what A-26 and A-28
> have to cover.

The builder, in the shape `normalize_adjacency` already uses:

```python
def build_temporal_stage(name: str, *, window_size: int,
                         config: ModelConfig) -> TemporalStage:
    """Build the stage named. Every option returns the same kind of object."""
    if name == "last_cycle":
        return LastCycle(window_size)
    if name == "mean_window":
        return MeanWindow(window_size)
    if name == "raw_window":
        return RawWindow(window_size)
    if name in ("lstm", "gru"):
        from .recurrent import build_recurrent
        return build_recurrent(name, window_size=window_size, config=config)
    raise ValueError(
        f"temporal_stage must be one of {list(get_args(TemporalStageName))}, "
        f"got {name!r}."
    )
```

### 3.2 The graph stage

`src/thesis_gnn_rul_xai/models/spatial/__init__.py`:

```python
class SpatialStage(nn.Module):
    """Mixes each sensor's numbers with its neighbours'. Carries its own graph."""

    out_features: int
    n_layers: int

    def forward(self, nodes: Tensor) -> Tensor:
        """(batch, nodes, in_features) -> (batch, nodes, out_features)."""
        raise NotImplementedError


def build_spatial_stage(name, *, adjacency_raw, adjacency_normalized,
                        in_features, config) -> SpatialStage:
    """Build the graph stage named. Every option is handed both forms of the graph.

    Both go in because the two stages need different ones, and letting each keep
    what it needs is safer than making the caller remember which. GCNStage keeps
    the rescaled matrix; GATStage keeps the raw 0/1 one.
    """
```

`GCNStage` carries the zero-diagonal refusal, **moved verbatim** out of `ESTGNN.__init__` — it is a requirement of the convolution, not of the model. `GATStage` carries the mirror of it:

```python
        if not np.array_equal(mask, mask.astype(bool)):
            raise ValueError(
                "A graph attention layer was handed a matrix whose entries are "
                "not 0 or 1, so it has already been through "
                "normalize_adjacency(). Attention works out its own weight for "
                "each neighbour and then makes those weights add up to one "
                "across the neighbourhood. Given weights that have already been "
                "averaged, it would average them a second time, and the "
                "attention weight it reports would no longer be the share it "
                "actually gave that neighbour. Pass PreparedData.adjacency_raw."
            )
```

`NoSpatialStage` returns its input unchanged and reaches the paper's own **"ESTGNN w/o GCN"** row (16.42 on FD001, Table 11), which this repository has never been able to run. With `last_cycle` the flatten becomes `(B, 14)` and FC-1 becomes `14 × 128`, so the parameter count changes a great deal — that is A-30.

`ESTGNN` after the change:

```python
    def __init__(self, adjacency_normalized, *, temporal: TemporalStage,
                 adjacency_raw=None, rul_max: int = 125,
                 gcn_hidden=GCN_HIDDEN_SIZES, fc_hidden=FC_HIDDEN_SIZE,
                 gcn_bias: bool = False, spatial_stage: str = "gcn",
                 allow_time_in_the_graph_layer: bool = False) -> None:
        super().__init__()
        self.temporal = temporal
        # The width entering the first graph layer is the stage's business.
        # It is never read off the data. Reading it off the data is how the
        # mode removed on 2026-08-28 was reachable without anyone naming it.
        self.in_features = int(temporal.out_features)
        ...

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[2] != self.temporal.window_size:
            raise ValueError(
                f"this network's temporal stage reads a "
                f"{self.temporal.window_size}-cycle window, but the data gives "
                f"{features.shape[2]} number(s) per sensor. The window length "
                "is a preprocessing setting (PreprocessingConfig.window_size) "
                "and the stage that consumes it is a model setting "
                "(ModelConfig.temporal_stage). They have to agree, and neither "
                "is allowed to work the other out."
            )
        nodes = self.temporal(features)            # (B, 14, 30) -> (B, 14, F)
        hidden = self.spatial(nodes)               # (B, 14, F)  -> (B, 14, 32)
        flat = torch.flatten(hidden, start_dim=1)  # (B, 448)
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(flat)))).squeeze(-1)
```

That `forward` check is the one that catches the genuinely silent failure: a recurrent network run over a one-step sequence trains happily and is not a temporal encoder.

### 3.3 The explanation method

`src/thesis_gnn_rul_xai/explain/attribution.py`:

```python
@dataclass(frozen=True, eq=False)
class Attribution:
    """One explanation, and what it says it is about.

    eq=False because values is a NumPy array: the generated __eq__ would raise
    "truth value of an array is ambiguous", and the generated __hash__ would
    raise "unhashable type". Identity comparison is what this object needs.
    """

    method: str                      # "shap_node", "edge_occlusion", ...
    target: Literal["node", "edge", "subgraph", "node_time"]
    labels: tuple[str, ...]          # "T24" for a sensor; "T24-T30" for an edge
    values: np.ndarray               # one number per label, same order
    units: str                       # "flight cycles" for the SHAP family
    is_exact: bool                   # False the moment anything is sampled
    baseline: str                    # what absence meant, naming its A-xx entry
    model_fingerprint: str           # "<temporal>-<spatial>+mdl<digest>"


def compare_attributions(items: Sequence[Attribution]) -> pd.DataFrame:
    """Put several explanations side by side. Refuses to mix what they explain."""
    targets = {a.target for a in items}
    if len(targets) > 1:
        raise ValueError(
            f"These explanations attribute to different things ({sorted(targets)}). "
            "On the paper's graph that is 14 sensors against 41 connections: the "
            "two lists describe different objects, there is no row-by-row "
            "correspondence between them, and a correlation across them would be "
            "a number about nothing. Compare within one target, or compare them "
            "on a property both have -- how far the prediction moves when the top "
            "k are removed -- and say in writing which you did."
        )
    fingerprints = {a.model_fingerprint for a in items}
    if len(fingerprints) > 1:
        raise ValueError(
            f"These explanations came from different networks ({sorted(fingerprints)}). "
            "An explanation is a statement about one trained model, so comparing "
            "two of them would measure the difference between the models."
        )
```

`explain/shap.py` gains one thin adapter that wraps `shap_values` into an `Attribution` with `target="node"`, `is_exact=True`, `baseline="A-20: the average training window"`. Nothing else in that file changes: its `_masked_batch` already does `keep = masks[:, :, None]`, which works at any window width, and its own comment already says why one decision covers a sensor's whole window.

`explain/edges.py` is the first graph-native method and the one that carries the thesis's actual contribution. Two things about it, both verified: `normalize_adjacency` adds the self-connections before it sums the degrees, so removing every edge of a sensor still leaves degree 1 and never divides by zero; and removing the T24–T30 edge changes T24's degree from 8 to 7 (7 neighbours plus itself, down to 6 plus itself), so under row normalisation its row goes from **eight entries of 0.125 to seven entries of 0.142857** — seven entries grow, and every one of them is a number that was not asked about. That is why the honest measurement deletes the edge from the raw graph and renormalises, rather than zeroing one entry of the already-rescaled matrix.

---

## 4. HOW A COMBINATION IS CHOSEN AND RECORDED

### `src/thesis_gnn_rul_xai/models/config.py`

Mirrors `training/config.py` line for line in style: a `Literal`, a `@dataclass` with `ASSUMPTION` comments at each judgement call, `__post_init__` that names the valid values, `to_dict`, `save`, and a module-level `load_model_config` where an unknown key is an error.

```python
TemporalStageName = Literal["last_cycle", "mean_window", "raw_window", "lstm", "gru"]
SpatialStageName  = Literal["gcn", "gat", "none"]

@dataclass
class ModelConfig:
    """Which stage does which job. README.md in this folder explains each field."""

    temporal_stage: TemporalStageName = "last_cycle"        # A-07
    temporal_hidden: int = 32                               # A-26
    temporal_layers: int = 1                                # A-26
    temporal_out_features: int = 1                          # A-28
    # The paper's Section 2.2 gives each sensor its own network, so False is
    # what it describes, not a choice. True is offered as a cheaper
    # alternative and is a departure that must be declared wherever a
    # number produced with it is quoted.
    temporal_shared_across_sensors: bool = False
    allow_time_in_the_graph_layer: bool = False             # not an assumption, a gate

    spatial_stage: SpatialStageName = "gcn"
    gcn_hidden: tuple[int, ...] = (32, 64, 32)              # Table 4
    gcn_bias: bool = False                                  # A-12
    gat_heads: int = 1                                      # A-29

    fc_hidden: int = 128                                    # Table 4
```

Two methods carry the weight:

```python
    def effective_dict(self) -> dict[str, Any]:
        """The settings that actually reached the network, for the run digest.

        A run with temporal_stage="last_cycle" never reads temporal_hidden, so
        digesting the whole dataclass would give two such runs different names
        over a number neither of them used, and the digest would then mean
        "different YAML file" rather than "different experiment". `notes` is
        dropped for the same reason: editing a comment must not rename a run.
        """

    def matches_table_4(self) -> tuple[bool, list[str]]:
        """Whether this build is the architecture Table 4 prints, and if not, why.

        Returns (True, []) for the default. Otherwise every reason is a sentence
        with the real numbers in it, printed by the code, not written here.
        """
```

`comparable_paper_row()` returns `"ESTGNN w/o LSTM, Table 11, FD001 RMSE 14.93"` when `matches_table_4()` is true and `temporal_stage` holds no parameters, `"ESTGNN w/o GCN, Table 11, FD001 RMSE 16.42"` when `spatial_stage == "none"` under the same condition, and **`None`** otherwise.

### The run directory name

```
fd001_lastcycle-gcn_physics_row_cap_5seed_varsplit_cfge3a331_mdl4f1e2a_9856794-dirty_20260909T142233Z
      └architecture┘                                          └model digest┘
```

with `_foldedtime` inserted after the architecture pair when `raw_window` was used. Longest realistic name is about 103 characters; project root (47) plus `reports\` (8) plus the name plus `\seed_42\training_history.csv` (28) comes to about 186, inside Windows' 260-character limit.

**The architecture is printed even when it is the default**, which departs from that file's existing "defaults stay silent so ordinary names stay short" convention. The reason goes in the comment beside it: the *presence* of the field is what lets a directory without one be read unambiguously as "made before this change, when there was only one architecture", so no stored run has to be renamed or edited — which rule 3a forbids anyway.

### `summary.json`

Every key `docsite/results.py::Run` reads today keeps its name and its meaning, including `config_digest` staying the six-character preprocessing-only digest. New keys beside them:

```json
  "temporal_stage": "last_cycle",
  "spatial_stage": "gcn",
  "temporal_out_features": 1,
  "node_feature_width": 1,
  "model_digest": "4f1e2a",
  "matches_table_4": true,
  "table_4_departures": [],
  "time_folded_into_graph_layer": false,
  "comparable_paper_row": "ESTGNN w/o LSTM, Table 11, FD001 RMSE 14.93",
  "parameters": {"temporal": 0, "graph": 4128, "dense": 57601, "total": 61729},
  "seeds_used": [42, 43, 44, 45, 46]
```

The parameter split is checkable by hand and the total must not move: `1×32 + 32×64 + 64×32 = 32 + 2048 + 2048 = 4128` for the graph layers, and `448×128 + 128 + 128×1 + 1 = 57344 + 128 + 128 + 1 = 57601` for the dense ones, giving 61,729.

### What a run directory holds

```
reports/fd001_lastcycle-gcn_.../
├── summary.json
├── results.csv                  unchanged name, unchanged columns
├── preprocessing_config.yaml    unchanged
├── model_config.yaml            NEW
├── seed_42/
│   ├── model.pt                 NEW — ~250 KB
│   └── training_history.csv     NEW
│   └── training_config.yaml     NEW
└── seed_43/ …
```

`TrainingHistory.save()` already exists at `training/run.py:87` and writes both `training_history.csv` and `training_config.yaml` — and **nothing calls it**, so `epochs`, `batch_size` and `learning_rate` are recorded in no run directory today. That is two lines while the surrounding lines are open. Saving the weights matters for a different reason: `scripts/train.py` currently throws the trained model away, so any explanation pass has to retrain and therefore explains a *different* model from the one whose RMSE was quoted. At 61,729 parameters that is about 250 KB per seed and 1.25 MB per five-seed run; `reports/*` is already outside git.

### The epilogue must become conditional

`scripts/train.py` currently ends every run with "The LSTM stage is not implemented here (A-11), so the row to compare against is 'ESTGNN w/o LSTM'." That becomes a false statement the first time anyone runs `--temporal lstm`. It becomes:

```python
    row = model_config.comparable_paper_row()
    if row is not None:
        print(f"\nThe row to compare against is: {row}.")
    else:
        matches, why = model_config.matches_table_4()
        print("\nThis is NOT the paper's architecture. "
              f"Temporal stage {model_config.temporal_stage!r}, graph stage "
              f"{model_config.spatial_stage!r}.")
        for reason in why:
            print(f"  - {reason}")
        print("The paper publishes no row for this. Nothing above may be set "
              "beside Table 8 or Table 11. See A-26 to A-28.")
```

### Reading old runs back

`docsite/results.py` gains `temporal_stage`, `spatial_stage` and `architecture_recorded` on the frozen `Run`, and a **translation from the vocabulary already in the files** — not an inference from the directory name, which is the weaker source:

```python
# Five runs at cdcc827 (2026-08-27) predate the architecture fields and record
# the choice under its old name. The two vocabularies are translated here
# rather than in the files: rule 3a forbids editing a stored result, and the
# directory name is a worse source than the summary that sits inside it.
_OLD_NODE_FEATURE_MODE = {
    "window": "raw_window (the mode removed on 2026-08-28; see A-07)",
    "last_step": "last_cycle",
}
```

Both globs at lines 110 and 160 stay one level deep, so nothing about where runs live changes.

---

## 5. THE A-07 / A-11 PROBLEM

Four guards. The first is the one that matters; the rest are what make it visible.

### Guard 1 — the width comes from the stage, and the data is checked against it

Deleted outright:

```python
# scripts/train.py:112 and scripts/verify_claims.py:312
        in_features=prepared.train.features.shape[2],
```

Replaced by `self.in_features = int(temporal.out_features)` inside `ESTGNN.__init__`, and by the `forward` check in §3.2 that runs the comparison the other way round. There is now **no configuration path** from a preprocessing setting to a wide first graph matrix, because preprocessing no longer knows what the model does with a window.

### Guard 2 — "hands the graph layer raw cycles" is derived from the object, not declared

```python
        learns_something = any(p.numel() for p in temporal.parameters())
        self.time_folded_into_graph_layer = self.in_features > 1 and not learns_something
        if self.time_folded_into_graph_layer and not allow_time_in_the_graph_layer:
            raise ValueError(
                f"{type(temporal).__name__} puts {self.in_features} numbers on "
                "each sensor without learning anything across the cycles, so "
                f"the first graph weight matrix becomes ({self.in_features}, "
                f"{gcn_hidden[0]}). For each of the {gcn_hidden[0]} output "
                f"channels that is {self.in_features} multipliers learned over "
                "time and shared across sensors -- adding up a sensor's "
                "readings in learned proportions, which IS a temporal encoder, "
                "sitting inside a layer that is not named one. That is the "
                "arrangement removed on 2026-08-28; see A-07. To build it on "
                "purpose pass allow_time_in_the_graph_layer=True, which writes "
                "'foldedtime' into the run directory name and "
                "'time_folded_into_graph_layer': true into summary.json."
            )
```

This is the piece worth arguing for. A stage cannot lie about it, because nothing is asked of the stage: `LastCycle` and `MeanWindow` have `out_features == 1`, so the property is false; `LSTMStage` at `out_features=32` holds parameters, so the property is false and correctly so — the graph layer's `32 × 32` matrix is then mixing *encoder outputs*, not raw cycles; `RawWindow` at 30 with no parameters is the only thing that trips it. It also catches something a registry cannot: `nn.Identity()` has no `out_features` and fails on the attribute; a hand-rolled passthrough that sets `out_features = 30` trips the refusal.

**Its one honest limit**, which the README must state: a stage that holds parameters and still passes cycles through — a learned per-cycle scaling, say — would evade it. The mitigation is that `build_temporal_stage` is the only path from a configuration and `models/temporal/README.md` lists every option in words; the guard closes the accident, the README closes the deliberate case.

### Guard 3 — `raw_window` exists, is named, and stamps every run it produces

Where `minimal-risk` and the other two genuinely conflict, I choose to **keep the removed mode as a named, gated stage** rather than remove it from the registry entirely: A-07's own comparison (13.45 against 20.35) is otherwise re-runnable only at the dead commit `cdcc827`, and a mode reachable only by writing a new class is *less* safe than one that names itself in the run directory, because the new class would carry no label at all.

The gate is on the configuration and the command line, not on the class. `tests/test_models.py:146` and `:173` need a 30-wide first graph layer to probe `model.graph_layers[0]` directly, and they will write `ESTGNN(adjacency, temporal=RawWindow(30), allow_time_in_the_graph_layer=True)` — which reads as exactly what it is, in a file where a reviewer will see it.

### Guard 4 — the run says whether it is Table 4

`matches_table_4()` and `comparable_paper_row()` from §4 turn rule 5 into something a file records rather than something a person remembers. `layer_summary()` gains a first row for the temporal stage, so notebook 02's side-by-side against Table 4 keeps working and starts showing the extra row explicitly.

*Reversed by the author on 2026-10-02, carried out 2026-10-03.* The separate row was built, and it described the same `(B, 14, 1)` array as Table 4's Input row beneath it: what the stage hands on is that row, which learns nothing of its own. The author pointed this out on the playground on 2026-09-14; the playground and the site's figures then each joined the two rows themselves, and only notebook 02 still showed both. The author decided that `layer_summary()` itself should fold the stage into the Input row, so that every place showing it shows the same thing without joining rows of its own: the Input row now runs from the window the stage receives to the `(B, 14, 1)` it hands on, counts what the stage learns, and names the stage when it learns anything. The default network's summary has seven rows again, the count §0 measured before the stage existed (re-measured 2026-10-03, torch 2.13.0+cpu: 7 rows, 61,729 parameters). `models/README.md` explains the row.

### Assumption entries to write or amend

**Amended, keeping their numbers (rule 6 — say what changed, do not overwrite):**

- **A-07.** Its `**Where:**` line becomes `models/temporal/`; `models/estgnn.py (in_features)` is deleted because that argument no longer exists. Its measured comparison, its F-12 paragraph, its `window_size` note and its open question about the mean all stay. A new paragraph records that the choice is now one of several named stages rather than a property of the data, and that the removed mode is reachable again only behind a flag whose name is a sentence.
- **A-11.** Gains one appended, dated line in the style A-06 and A-18 already use, so its heading stays unmarked and `_status_of` at `docsite/results.py:179` continues to show it as active — which is correct, because it still governs the default:

  > **Extended 2026-09-XX by A-26 to A-28.** The stage can now be switched on. It is off by default and every result in this repository to date was produced with it off, so everything above still describes the default and still governs every stored run. What A-26 to A-28 add is that a run with the stage on is an *extension* of the paper, not a reproduction of it, and compares against nothing published — not 11.78, and not 14.93 either, since 14.93 is the row for a model with no recurrent stage at all.

- **A-12.** Its `**Where:**` gains the `ModelConfig.gcn_bias` path.
- **`docs/EXPERIMENT_A-07_mean_window.md`.** Its §4.1 and §4.2 put `node_feature` in `PreprocessingConfig`; this design overrules that and puts it in `ModelConfig` as a temporal stage. The document must say so in the open rather than be quietly ignored — its §5 (the commands), §6 (how to decide), §7 (the trap: no learned reduction, the parameter count must still be 61,729) and §8 (what to write) all stand unchanged. **The mean window stays A-07's own open question and does not get a new number**; the repository already owns that gap.

**New, A-26 onward** (A-25 and F-15 are the current maxima, and `tests/test_docsite.py:185` asserts contiguity):

*Note added 2026-09-14.* The numbers from A-27 down in this table were
provisional and none of them reached the log. The log's contiguity test gave
the next free number, A-27, to the correlation-based sensor graph (the 41 most
strongly correlated pairs), so every row below A-26 takes the next free number
when it is written.

| id | the gap | the choice |
|---|---|---|
| A-26 | the recurrent stage's hidden size and depth, which the paper never prints | 32 and 1, and the run is not a reproduction |
| ~~A-27~~ | whether the fourteen sensors share one recurrent network or get one each | **Not an assumption. The paper states it** — see the box below. One dedicated network per sensor: 62,720 parameters at hidden size 32, against a 61,729-parameter model. The shared option stays in the code as a named alternative, never as the default. |
| A-28 | how the recurrent stage's output becomes what the graph layer reads | a learned projection to `temporal_out_features`; 1 keeps Table 4's printed shape |
| A-29 | the attention stage's head count, and which matrix defines the neighbourhood — the raw `A` or the rescaled `Â` | the raw one; attention decides weights *within* a neighbourhood, the adjacency decides the neighbourhood |
| A-30 | what stands in for the graph layers when they are removed, so "ESTGNN w/o GCN" (16.42) can be reached | flatten straight from the temporal stage; FC-1 becomes `14 × 128` |
| A-31 | what "removing an edge" means | delete it from the raw graph and renormalise, so the degree change is honest — not zero one entry of `Â` |
| A-32 | 2^41 coalitions is out of reach, so the exactness this project guarantees for sensors does not survive the move to edges | exact over one sensor's incident edges (2^8 = 256 at most), sampled globally with a standard error reported beside every value |

**Findings to expect, recorded once measured and not before:** F-16, the refactor changed nothing (the §6 gate); F-17, what `raw_window` scores under *today's* preprocessing — because A-07's 13.45 was measured at `cdcc827` under min-max scaling, symmetric normalisation and the exponential schedule, and is not a number about the current configuration.

*Note added 2026-09-14.* These two finding numbers were provisional too, and
both have been taken by other entries: F-16 records what the paper's recurrent
stage is worth under two training procedures, and F-17 records that the paper's
Fig. 10 is drawn from its Table 12 except in two graphs. The `raw_window`
re-measurement, and every other "F-17" below, takes the next free finding
number when it is written.

---

## 6. MIGRATION

Nine commits. Each leaves `ruff check .`, `pytest`, `python scripts/check_prose.py` and `python scripts/verify_claims.py --fast` green.

**A note on the baseline before you start.** The regression baseline is `reports/fd001_physics_row_cap_5seed_varsplit_cfgba82e3_ae59bc1-dirty_20260908T223607Z`: mean RMSE **16.4965 ± 0.4153**, per-seed 16.6949 / 16.1311 / 16.8432 / 15.9704 / 16.8427. Reproduce it with `python scripts/train.py --config config/fd001.yaml --seeds 5`, **not** with bare `python scripts/train.py` — the bare default now carries a different `notes` string and therefore digests to `e3a331`, so the directory name differs while every number should match. Do not use the run `minimal-risk` nominated for this and do not use the README's headline 16.82, which belongs to `cfg00be26` (`validation_fraction: 0.0`) and is a different configuration.

1. **`models/temporal/` exists and nothing uses it.** `TemporalStage`, `LastCycle`, `MeanWindow`, `RawWindow`, `build_temporal_stage`, `models/temporal/README.md`, `tests/test_temporal.py`, worked examples in `tests/test_examples_torch.py`. One line in `docsite/catalog.py`. Green; no behaviour change anywhere.

2. **`ESTGNN` accepts `temporal`.** `in_features` still accepted; given without `temporal` it builds `LastCycle(window_size=in_features)`, which on today's width-1 data is the identity, so every existing call site and test passes untouched. Add `tests/test_models.py::test_the_default_weights_start_where_they_started_before_the_temporal_stage`, asserting the digest `ceff1a8c04f9edd1…` measured above, with a docstring saying it is a fact about one PyTorch version (2.13.0+cpu here). This is the test that catches the one thing that would silently break bit-identity: a new submodule drawing from the random number generator before the graph layers do.

3. **The shape change.** `data/windowing/` stops truncating; `_extract_one_unit`'s empty return becomes `np.empty((0, len(sensor_columns), window_size))`; `in_features` is removed from `ESTGNN`; every one of the 30 call sites across 11 files is updated; the 68 `(14, 1)` mentions across 15 files are corrected; `docsite/figures.py` gains a temporal box **and `_WIDTH` goes from 860 to 960** (eight boxes need 936); both notebooks are re-executed. Gate: the weight digest is unchanged, and the five-seed run reproduces the baseline above **to six decimals including `best_epoch`**. Record the outcome as **F-16**, in the form F-06 uses. If the means differ past the sixth decimal, the refactor changed the network and must not be committed. **This is the riskiest step.**

4. **`ModelConfig`, run naming, `summary.json`, storage.** `models/config.py`, `config/model/table4.yaml`, the `mdl` digest, the six new summary keys, `model.pt` and `training_config.yaml` per seed, `docsite/results.py`'s three fields and the `_OLD_NODE_FEATURE_MODE` translation. No new architecture: this commit is only about being able to say which architecture a run used.

5. **`models/spatial/` with `gcn` only.** Moves the zero-diagonal refusal out of `ESTGNN`. Same weight-digest gate as step 2, plus a second five-seed reproduction. No new capability.

6. **`raw_window` behind the gate**, `--allow-time-in-the-graph-layer`, `foldedtime` in the name. Re-measure it at today's configuration and record **F-17**; do not carry 13.45 forward.

7. **`LSTMStage` and `GRUStage`**, A-26 and A-28, the conditional epilogue, `scripts/compare.py` (which promotes the paired test currently living as a fenced code block inside A-21). A-27 is **withdrawn before it was written**: the PDF was read on 2026-09-09 and settles it, so `shared_across_sensors` defaults to `False` and needs no assumption entry — only a citation.

8. **`GATStage` and `NoSpatialStage`**, A-29 and A-30. The second of these turns a published ablation row (16.42) that this project cannot currently address into a one-word config change.

9. **`Attribution`, `compare_attributions`, `explain/edges.py`, `scripts/explain.py`**, A-31 and A-32.

You can stop after step 5 with everything green, the blur closed, every run naming its architecture, and no new capability at all — which is the correct place for a natural pause.

---

## 7. WHAT IT COSTS

### What gets harder

- **Every documented shape moves.** `WindowedDataset.features` becomes `(17731, 14, 30)`. `"14, 1)"` appears **73 times** across Markdown, Python and the notebooks, including whole worked-example walkthroughs in `data/pipeline/README.md` (8 occurrences, some of them pasted logger output — `Built 17731 window(s) of shape (14, 1)`) and `data/windowing/README.md`. That is the single largest chunk of work in the change and it is prose, not code.
- **`data/processed/windows.npz` must be regenerated.** 17731 × 14 × 30 × 4 bytes is 29.8 MB uncompressed for the training features alone, roughly thirty times what it holds now. Not in git. Add `"node_feature_axis": "cycles"` to `diagnostics.json` so an old archive cannot be mistaken for a new one.
- **Both notebooks must be re-executed** (CLAUDE.md requires it). `notebooks/02_model_fd001.ipynb` builds `ESTGNN` in four places.
- **`scripts/verify_claims.py` is not a bystander.** `_trained_model()` at line 306 builds the model with `in_features=prepared.train.features.shape[2]`, and `check_baselines` (A-20) and `check_p15` (F-05) both depend on it. Pin it to `ModelConfig()` on purpose: every F-05 number in the log was measured on that architecture, and letting the drift-guard follow a flag would silently redefine them.
- **`check_prose.py`'s `WORDS` list** will need `hidden state`, `cell state`, `attention head`, `occlusion`, `faithfulness`, `edge mask`, `perturbation`. **Do not add `head` or `gate` on their own**: the pattern at line 140 is `rf"\b{re.escape(word)}"` with no closing boundary, so `head` fires on "header", "headline" and "header=None" across files that pass today, and `gate` fires on `check_prose.py`'s own docstring. Add only multi-word forms, or fix the pattern first.
- **`pyproject.toml`'s `[tool.ruff.lint.per-file-ignores]`** needs an entry for each new test module that calls `pytest.importorskip("torch")` before importing, matching the four already listed.
- **Test counts move**, in CLAUDE.md and README section 10. Recount rather than adjust.
- **Tests that need real edits, not just a dropped argument:** `test_a_wider_first_layer_only_changes_w1` (`:81`) asserts `wider.count_parameters() - table4.count_parameters() == (30 - 1) * 32`; a wider first layer now requires a stage that produces the width, and a learned stage brings its own parameters, so the difference must be taken over `model.spatial.parameters()` and the test renamed. `test_activations_and_shapes_match_table_4` (`:62`) asserts a positional list of seven rows and gains one. `test_parameter_counts_match_table_4` indexes rows by name and survives untouched — which is the point: the test guarding 61,729 does not move.

### What must be re-measured

- **Nothing, if step 3's gate passes.** That is the whole design of the migration order. Every stored run keeps parsing, every number in `docs/ASSUMPTIONS.md` keeps meaning what it meant, and the four `_window_` runs come out **better** labelled than they are now.
- **A-07's 13.45 must not be carried forward.** It was measured at `cdcc827` under min-max scaling, symmetric normalisation and the exponential schedule. If step 6 reinstates `raw_window`, measure it again at today's configuration and record it as F-17.
- **A-21, A-02, F-04 and F-10's tables** stay valid and stay interpretable; each gains one clause on its heading — every row is now "under `last_cycle` + `gcn`". No re-running is required unless the default architecture moves, at which point rule 3 already predicts what will happen: two of the three comparisons reversed direction after the A-07 correction, and they can reverse again.
- **The README's headline is already stale and this change will expose it.** 16.82 ± 0.16 is labelled "no validation split (A-22)" and belongs to `validation_fraction: 0.0`; today's default holds 20% back after A-06's reinstatement of 2026-09-08 and measured 16.4965 ± 0.4153. That is a pre-existing drift, not caused by this design, and it should be fixed before step 3 so the regression gate has an uncontested baseline.

### Two pre-existing defects this change would otherwise repeat

- **`PreprocessingConfig.notes` is inside the digest.** `to_dict()` is `dataclasses.asdict`, so the prose field is hashed: today's default digests to `e3a331` while `config/fd001.yaml` gives `ba82e3`, and the *only* difference between them is a sentence of English. Editing a comment renames a run. `ModelConfig.effective_dict()` excludes `notes` from day one; whether to make the same fix to the preprocessing digest is your call, because it would give future runs a different `cfg` value than stored runs with identical settings — a one-time discontinuity, worth a line in the log if you take it.
- **`TrainingHistory.save()` is never called**, so `epochs`, `batch_size` and `learning_rate` appear in no run directory. Step 4 fixes it in two lines.

### What must be written into `docs/ASSUMPTIONS.md`

A-07 amended (the `**Where:**` line and one new paragraph), A-11 given one dated appended line, A-12's `**Where:**` extended, and **A-26 through A-32** written in the file's own gap / what-we-chose / why / how-much-it-matters shape, each with its identifier quoted in the code at the point of use, in words. `docs/EXPERIMENT_A-07_mean_window.md` amended in the open to say its §4.1 and §4.2 are superseded and why. **F-16** written after step 3's gate, **F-17** after step 6. Nothing in `reports/` is edited, renamed or deleted.