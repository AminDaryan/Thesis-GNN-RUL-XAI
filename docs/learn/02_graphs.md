# GNN: connections between sensors

Wen et al. (2026, p. 2) argue that predicting an engine's remaining life needs
the relationships between its sensors as well as how each sensor changes over
time. A **graph neural network**, or **GNN**, lets a sensor's calculated values
depend on other sensors as well as itself. The graph specifies which sensors may
contribute (Corso et al., 2024, p. 2).

A **node** is one sensor position. An **edge** is a permitted connection
between two positions (Wen et al., 2026, Section 2.1). An edge permits exchange
of information inside the model; it does not prove that one measurement causes
another.

Keep three objects separate:

| Object | What its rows contain | Where it comes from |
|---|---|---|
| Sensor input table | Readings or temporal summaries for each sensor | The engine data and the selected temporal algorithm |
| Graph table | Which sensors may send to each receiving sensor | The chosen connection rule |
| Calculated sensor values | Results of combining inputs and applying learned multipliers | The graph layers |

The temporal algorithm follows cycles **within** a sensor. The graph layers
combine information **between** sensors. A column produced by a graph layer
is a calculated value, not another time or another physical sensor.

## Example setup

The examples keep all fourteen sensor positions. The input sample shows
three complete FD001 rows (NASA Prognostics Center of Excellence [NASA PCoE],
2008) with all selected sensor columns. We then build
the implemented Physics graph and follow the same graph through rescaling.
The graph is not inferred from those three rows: its connections are supplied
by the project's copy of Wen et al.'s (2026) Table 3.

The tables stand on their own. The optional code uses the real FD001 files;
run the setup and then the examples in page order to reproduce every table.

<details class="process-example">
<summary>Set up the Python examples</summary>

```python
import numpy as np
import pandas as pd
from thesis_gnn_rul_xai.data.loading import load_subset
from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_COLUMNS
from thesis_gnn_rul_xai.graph.adjacency import build_physics_adjacency, normalize_adjacency
from thesis_gnn_rul_xai.utils.paths import raw_data_dir

raw = load_subset("FD001", data_dir=raw_data_dir())
matrix, names = build_physics_adjacency()
names = list(names)
sample = raw.train.loc[:2, list(PAPER_SENSOR_COLUMNS)].copy()
sample.columns = names
sample.insert(0, "Cycle", raw.train.loc[:2, "cycle"].to_numpy())
relationships = pd.DataFrame({"Sensor": names,
    "Permitted other senders": [", ".join(names[j] for j in np.flatnonzero(row)) for row in matrix]})

def graph_table(values):
    """Show all receiving rows and sending columns in the fixed sensor order."""
    frame = pd.DataFrame(values, columns=names)
    frame.insert(0, "Receiver", names)
    return frame

raw_graph = graph_table(matrix.astype(int))
```

</details>

<div class="algorithm-intro">
<p class="algorithm-label">Sensor-graph preparation algorithm</p>
<p>Follow steps 1 and 2 to turn a connection rule into the fixed multipliers
used by every graph layer. Sensor readings remain a separate input. The
Architecture and training page then follows those readings through the layers.</p>
</div>

---

## 1 — Sensor graph

First decide which sensor positions may exchange information. The **adjacency matrix** is that decision written as a table: a 1 permits a connection and a 0 excludes it (Wen et al., 2026, p. 5). Rows name receiving sensors; columns name sending sensors.

The left table shows the permitted connections. Its diagonal is zero, as in Wen et al.'s (2026) Table 3, because including the receiving sensor itself is the next step's job. The right table previews the combining shares calculated in that next step.

{{figure:adjacency}}

### Process

#### Input

Three actual rows of sensor readings establish the sensor columns and their order. These readings do not choose the Physics graph's connections:

<!-- sample: sample -->
| Cycle | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 641.820000 | 1589.700000 | 1400.600000 | 21.610000 | 554.360000 | 2388.060000 | 9046.190000 | 47.470000 | 521.660000 | 2388.020000 | 8.419500 | 392 | 39.060000 | 23.419000 |
| 2 | 642.150000 | 1591.820000 | 1403.140000 | 21.610000 | 553.750000 | 2388.040000 | 9044.070000 | 47.490000 | 522.280000 | 2388.070000 | 8.431800 | 392 | 39.000000 | 23.423600 |
| 3 | 642.350000 | 1587.990000 | 1404.200000 | 21.610000 | 554.260000 | 2388.080000 | 9052.940000 | 47.270000 | 522.420000 | 2388.030000 | 8.417800 | 390 | 38.950000 | 23.344200 |

#### Function

Use the supplied relationships below, rather than calculating connections from the sample readings. This list includes every sensor and every permitted other sender in the implemented Physics graph (Wen et al., 2026, Table 3):

<!-- sample: relationships -->
| Sensor | Permitted other senders |
|---|---|
| T24 | T30, P15, P30, Nf, Ps30, NRf, BPR |
| T30 | T24, T50, P30, Nc, Ps30, phi, htBleed, W31 |
| T50 | T30, Nc, W31, W32 |
| P15 | T24, Nf, NRf, BPR |
| P30 | T24, T30, Nc, Ps30, phi, htBleed, W31 |
| Nf | T24, P15, NRf, BPR |
| Nc | T30, T50, P30, Ps30, phi, htBleed, W31, W32 |
| Ps30 | T24, T30, P30, Nc, phi, htBleed, W31 |
| phi | T30, P30, Nc, Ps30, htBleed, W31 |
| NRf | T24, P15, Nf, BPR |
| BPR | T24, P15, Nf, NRf |
| htBleed | T30, P30, Nc, Ps30, phi, W31, W32 |
| W31 | T30, T50, P30, Nc, Ps30, phi, htBleed, W32 |
| W32 | T50, Nc, htBleed, W31 |

For each receiving row, mark 1 under each listed sender and 0 elsewhere. For example, P15 permits T24, Nf, NRf and BPR. Its T24 entry becomes 1, its T30 entry 0, and its own entry stays 0. Apply the same rule to all rows; a permitted pair is recorded in both directions, as it is in the source table (Wen et al., 2026, Table 3).

Design a run also offers two different rules. **Every sensor joined to every other** permits every distinct pair; this project uses it for the variant Wen et al. (2026, p. 11) train "without physics prior knowledge", whose replacement graph the paper does not describe (assumption A-19). **The most correlated sensor pairs** ranks how strongly the training readings vary together, either in the same or opposite directions, and keeps the requested pairs (assumption A-27). That rule uses the training engines after the split; held-back and test engines do not choose its connections. The Physics rule is the one worked through here.

<details class="process-example">
<summary>Example in Python</summary>

```python
rebuilt = np.zeros_like(matrix)
for receiver, senders in enumerate(relationships["Permitted other senders"]):
    for sender in senders.split(", "):
        if sender:
            rebuilt[receiver, names.index(sender)] = 1
np.testing.assert_array_equal(rebuilt, matrix)
np.testing.assert_array_equal(rebuilt, rebuilt.T)
assert not np.diag(rebuilt).any()
graph_output = graph_table(rebuilt.astype(int))
graph_counts = pd.DataFrame({"Sensors": [len(names)], "Undirected connections": [int(matrix.sum() // 2)]})
```

</details>

#### Output

The complete connection table. Every row and column uses the sensor order of the input. The sensor measurements themselves have not changed.

<!-- sample: graph_output -->
| Receiver | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T24 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| T30 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 |
| T50 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| P15 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| P30 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 |
| Nf | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| Nc | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| Ps30 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 0 |
| phi | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 |
| NRf | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| BPR | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| htBleed | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
| W31 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 1 |
| W32 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |

Counted from this table (a pair is counted once, rather than once in each direction):

<!-- sample: graph_counts -->
| Sensors | Undirected connections |
|---|---|
| 14 | 41 |

---

## 2 — Rescale the graph

More neighbours would produce a larger sum even if all their values were alike. Rescaling controls how much each sender contributes, while including the receiving sensor's own value so that it is not discarded (Kipf & Welling, 2017, Eq. 2; Wen et al., 2026, p. 6).

**Do not confuse this with rescaling sensor readings.** Here we change the numbers describing connections. The earlier preprocessing step changes measurements.

### Process

#### Input

The same complete raw connection table from step 1:

<!-- sample: raw_graph -->
| Receiver | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T24 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| T30 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 |
| T50 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| P15 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| P30 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 |
| Nf | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| Nc | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| Ps30 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 0 |
| phi | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 |
| NRf | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| BPR | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| htBleed | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
| W31 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 1 |
| W32 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |

#### Function

For the default **row** method, put 1 on the diagonal to include each sensor itself. Count the ones in each receiving row and divide every entry by that count. Allowed senders get equal shares whose total is 1, so the next graph layer takes their average. This particular rescaling is a project choice (assumption A-21); the paper's text beneath its Equation 8 specifies rescaling with self-connections without giving its formula (Wen et al., 2026, p. 6).

P15 has four other senders. Including itself makes five, so each permitted entry becomes `1 / 5 = 0.2`. T24 has seven other senders; including itself makes eight, giving `1 / 8 = 0.125`. The complete counts are:

<!-- sample: counts_table -->
| Receiver | Other senders | Including itself | Each allowed share |
|---|---|---|---|
| T24 | 7 | 8 | 0.125000 |
| T30 | 8 | 9 | 0.111111 |
| T50 | 4 | 5 | 0.200000 |
| P15 | 4 | 5 | 0.200000 |
| P30 | 7 | 8 | 0.125000 |
| Nf | 4 | 5 | 0.200000 |
| Nc | 8 | 9 | 0.111111 |
| Ps30 | 7 | 8 | 0.125000 |
| phi | 6 | 7 | 0.142857 |
| NRf | 4 | 5 | 0.200000 |
| BPR | 4 | 5 | 0.200000 |
| htBleed | 7 | 8 | 0.125000 |
| W31 | 8 | 9 | 0.111111 |
| W32 | 4 | 5 | 0.200000 |

The **symmetric** option uses both endpoints' counts: divide a connection's 1 by the square root of the two counts multiplied together (Kipf & Welling, 2017, Eq. 2). P15 receiving from T24 gets `1 / sqrt(5 × 8) = 0.158114`. The reverse connection gets the same multiplier. A row need not sum to 1, so this is not generally an average.

<!-- sample: method_comparison -->
| Method | P15 receives from T24 | P15 row total |
|---|---|---|
| Row | 0.200000 | 1.000000 |
| Symmetric | 0.158114 | 0.958114 |

<details class="process-example">
<summary>Example in Python</summary>

```python
with_self = matrix + np.eye(len(names), dtype=matrix.dtype)
counts = with_self.sum(axis=1)
scaled = normalize_adjacency(matrix, method="row")
symmetric = normalize_adjacency(matrix, method="symmetric")
np.testing.assert_allclose(scaled, with_self / counts[:, None])
counts_table = pd.DataFrame({"Receiver": names, "Other senders": matrix.sum(axis=1).astype(int),
                             "Including itself": counts.astype(int), "Each allowed share": 1 / counts})
p15, t24 = names.index("P15"), names.index("T24")
method_comparison = pd.DataFrame({"Method": ["Row", "Symmetric"],
    "P15 receives from T24": [scaled[p15, t24], symmetric[p15, t24]],
    "P15 row total": [scaled[p15].sum(), symmetric[p15].sum()]})
scaled_output = graph_table(scaled)
```

</details>

#### Output

The complete table of combining multipliers for the **row** method. Each graph layer reuses it, and training adjusts another table of multipliers for transforming the combined values (Kipf & Welling, 2017, Eq. 2; Wen et al., 2026, Eqs. 8–10); it does not adjust these connections or their combining shares.

<!-- sample: scaled_output -->
| Receiver | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T24 | 0.125000 | 0.125000 | 0.000000 | 0.125000 | 0.125000 | 0.125000 | 0.000000 | 0.125000 | 0.000000 | 0.125000 | 0.125000 | 0.000000 | 0.000000 | 0.000000 |
| T30 | 0.111111 | 0.111111 | 0.111111 | 0.000000 | 0.111111 | 0.000000 | 0.111111 | 0.111111 | 0.111111 | 0.000000 | 0.000000 | 0.111111 | 0.111111 | 0.000000 |
| T50 | 0.000000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 | 0.000000 | 0.200000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.200000 | 0.200000 |
| P15 | 0.200000 | 0.000000 | 0.000000 | 0.200000 | 0.000000 | 0.200000 | 0.000000 | 0.000000 | 0.000000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 | 0.000000 |
| P30 | 0.125000 | 0.125000 | 0.000000 | 0.000000 | 0.125000 | 0.000000 | 0.125000 | 0.125000 | 0.125000 | 0.000000 | 0.000000 | 0.125000 | 0.125000 | 0.000000 |
| Nf | 0.200000 | 0.000000 | 0.000000 | 0.200000 | 0.000000 | 0.200000 | 0.000000 | 0.000000 | 0.000000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 | 0.000000 |
| Nc | 0.000000 | 0.111111 | 0.111111 | 0.000000 | 0.111111 | 0.000000 | 0.111111 | 0.111111 | 0.111111 | 0.000000 | 0.000000 | 0.111111 | 0.111111 | 0.111111 |
| Ps30 | 0.125000 | 0.125000 | 0.000000 | 0.000000 | 0.125000 | 0.000000 | 0.125000 | 0.125000 | 0.125000 | 0.000000 | 0.000000 | 0.125000 | 0.125000 | 0.000000 |
| phi | 0.000000 | 0.142857 | 0.000000 | 0.000000 | 0.142857 | 0.000000 | 0.142857 | 0.142857 | 0.142857 | 0.000000 | 0.000000 | 0.142857 | 0.142857 | 0.000000 |
| NRf | 0.200000 | 0.000000 | 0.000000 | 0.200000 | 0.000000 | 0.200000 | 0.000000 | 0.000000 | 0.000000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 | 0.000000 |
| BPR | 0.200000 | 0.000000 | 0.000000 | 0.200000 | 0.000000 | 0.200000 | 0.000000 | 0.000000 | 0.000000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 | 0.000000 |
| htBleed | 0.000000 | 0.125000 | 0.000000 | 0.000000 | 0.125000 | 0.000000 | 0.125000 | 0.125000 | 0.125000 | 0.000000 | 0.000000 | 0.125000 | 0.125000 | 0.125000 |
| W31 | 0.000000 | 0.111111 | 0.111111 | 0.000000 | 0.111111 | 0.000000 | 0.111111 | 0.111111 | 0.111111 | 0.000000 | 0.000000 | 0.111111 | 0.111111 | 0.111111 |
| W32 | 0.000000 | 0.000000 | 0.200000 | 0.000000 | 0.000000 | 0.000000 | 0.200000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.200000 | 0.200000 | 0.200000 |

---

## How information travels through layers

A graph layer first combines each receiving sensor's permitted senders,
using the shares above. It does that separately for every input column.
Learned multipliers then turn each combined row into new calculated values.
Negative results become zero (Kipf & Welling, 2017, Eq. 2; Wen et al., 2026,
Eqs. 8–10). These updated rows are the next layer's inputs.
This collect–combine–update operation is called **message passing**
(Corso et al., 2024, p. 2).

{{figure:message-passing}}

In the drawing, T24 receives P15's information in one round. A sensor that
receives T24's updated value in the next round can then depend on P15 too.
This is why one layer can reach immediate neighbours, two layers can reach
two connections away, and three can reach three (Corso et al., 2024, Fig. 1;
Kipf & Welling, 2017, p. 3). All sensors use the previous
layer's values; updates within a layer do not cascade through the whole graph.

{{figure:reach}}

Three layers do not necessarily connect every pair. The figure counts the
distances in the implemented graph. Later fully connected layers combine all
sensor positions (Wen et al., 2026, Eqs. 11–12), including those that cannot
reach each other within the graph layers. [Graph layer 1](05_the_model.md#1--graph-layer-1) starts a complete
numerical walkthrough of that process.

## Why use a graph?

The Physics graph expresses a hypothesis about useful sensor relationships,
proposed from engine structure by Wen et al. (2026, Section 2.3.1). Whether that
hypothesis helps prediction must be measured; the presence of a connection
alone is not evidence of benefit or causation.

Joining every sensor to every other is not a neutral replacement: after row
rescaling, every sensor receives the same average. Different sensor positions
then lose their distinct inputs to that layer. Some rows also coincide in the
Physics graph. The [graph implementation guide](../../src/thesis_gnn_rul_xai/graph/README.md)
explains these effects and links to the measured comparisons, with the
configuration and training procedure each comparison used.

The next page, [Architecture and training](05_the_model.md), follows actual
FD001 readings through the graph layers and onwards to a prediction.

## References

- Corso, G., Stark, H., Jegelka, S., Jaakkola, T., & Barzilay, R. (2024).
  Graph neural networks. *Nature Reviews Methods Primers*, *4*, Article 17.
  https://doi.org/10.1038/s43586-024-00294-7
  [Copy](../references/corso_2024_graph_neural_networks.pdf)
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with
  graph convolutional networks. *International Conference on Learning
  Representations (ICLR)*. https://arxiv.org/abs/1609.02907v4
  [Copy](../references/kipf_2017_graph_convolutional_networks.pdf). Page
  numbers on this page are those of arXiv version 4, the copy held.
- NASA Prognostics Center of Excellence. (2008). readme.txt, the description of
  the C-MAPSS data set distributed inside its zip file. NASA Prognostics Data
  Repository. [Copy](../references/nasa_2008_cmapss_readme.txt)
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and
  Technology*, *168*, 110875. https://doi.org/10.1016/j.ast.2025.110875
  [Copy](../references/wen_2026_estgnn_dynamic_shap.pdf)
