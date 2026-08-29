# `graph` — the sensor graph and its adjacency matrix

This folder builds the graph that the neural network passes messages over. The
graph has 14 nodes, one per retained sensor, and its edges say which pairs of
sensors the model may exchange information between directly.

> **The `.py` files here carry one line of description each, and nothing more.**
> All of the explanation lives in this file. The section
> [What each name means](#what-each-name-means) at the end lists every function
> and value the folder offers, and what it gives you back.

## Files

- `__init__.py` — makes the folder a package. No logic.
- `adjacency.py` — everything else: the transcribed matrix from the paper, two
  alternative graphs to compare it against, the normalisation the network
  applies before using it, and a report describing the graph's shape.

## The vocabulary, in one place

- A **graph** is a set of points, called **nodes**, plus a set of connections
  between them, called **edges**. Here each of the 14 sensors is one node. An
  edge between two sensors means "these two measurements are physically
  related, so the model may pass information directly between them".
- An **adjacency matrix** writes a graph down as a table of numbers. For 14
  nodes it is a 14-by-14 table `A` where `A[i, j] = 1` if nodes `i` and `j` are
  joined and `0` otherwise. "Sensor i is related to sensor j" means the same
  thing as "sensor j is related to sensor i", so the matrix is **symmetric** —
  it equals its own mirror image across the diagonal.
- A **Graph Neural Network** (GNN) works by **message passing**: every node
  collects its neighbours' current feature vectors, combines them, and updates
  its own. The adjacency matrix is what defines "neighbour". Stacking three
  such layers, as the paper does, lets information travel up to three edges:
  the first layer reaches direct neighbours, the second reaches neighbours of
  neighbours, the third reaches three steps out.

## Where this graph comes from

Most published work on this dataset builds the graph from the data, usually by
connecting sensors whose readings are strongly correlated. Wen et al. (2026)
instead design the graph by hand from engineering knowledge of how a turbofan
engine works, and print the finished matrix as their Table 3. Their five stated
rules are:

1. Sensors at the same physical location in the engine are connected, because
   they measure the same gas at the same point. Example: `P30` and `Ps30`, the
   total and static pressure at the high-pressure compressor outlet.
2. A raw speed is connected to its corrected counterpart, because one is
   computed from the other. Example: `Nf` and `NRf`.
3. Sensors are connected along the path the gas actually takes through the
   engine. Example: the temperature chain `T24` to `T30` to `T50`, following the
   flow from the low-pressure compressor through the high-pressure compressor to
   the low-pressure turbine.
4. Fuel and control quantities are connected to what they influence. Example:
   `phi`, the ratio of fuel flow to static pressure, is connected to the
   temperatures and pressures that burning more fuel changes.
5. Cooling quantities are connected to each other, because together they manage
   the temperature of the turbine section.

`PHYSICS_ADJACENCY_ROWS` in `adjacency.py` is that table transcribed row by row,
exactly as printed. The paper renders the matrix twice — as numbers in Table 3
and as a colour grid in Fig. 3 — and both renderings were checked against the
transcription. All 196 cells agree.

### Two things the paper leaves open

- **The paper's own rule 4 requires an edge its matrix does not contain.** The
  text says `phi` influences `htBleed`, `W31` *and* `W32`, but the `phi`-`W32`
  entry is 0 in Table 3 and in Fig. 3 alike. This code follows the matrix,
  because the matrix is what the model consumes. The conflict is recorded as
  finding F-07 in `docs/ASSUMPTIONS.md`. (Findings, numbered `F-xx`, are facts
  about the paper established from data; assumptions, numbered `A-xx`, are
  choices this project had to make.)
- **The paper never prints the normalisation formula.** See the assumptions
  section at the end of this file.

## Normalising the matrix before use

A graph convolution layer does not use the raw matrix `A`. It uses

```
A_hat = D^(-1/2) (A + I) D^(-1/2)
```

Two separate things happen in that formula, both from Kipf and Welling (2017):

- Adding the identity matrix `I` puts a **self-loop** on every node. Without it
  a node would be updated only from its neighbours and would forget its own
  current value.
- Dividing by the square roots of the **degrees** — the number of edges at each
  node, stored on the diagonal of `D` — stops well-connected nodes from
  dominating. Without that scaling, repeatedly summing over neighbours makes
  feature values grow layer after layer and training becomes unstable.

`normalize_adjacency` implements this. It accepts two forms, listed in the
`AdjacencyNormalization` type:

- `"symmetric"` — `D^(-1/2) (A + I) D^(-1/2)`, the form Kipf and Welling define.
  Each edge is divided by the geometric mean of the two degrees it joins, so a
  message between two well-connected sensors is damped more than one between two
  sparsely connected sensors.
- `"row"` — `D^(-1) (A + I)`, which simply averages over each node's own
  neighbourhood. Rows sum to exactly 1, so every node receives the mean of its
  neighbours regardless of how many it has.

Both add self-loops, and both "balance feature aggregation across different node
degrees", which is all the paper says about the matter.

## What the report tells you, and why it matters

`adjacency_report` describes a graph and checks that it is usable. Several of
the things it reports bound what the model can express *before any training
happens*:

- `is_connected` — whether every node can be reached from every other one. A
  graph that is not connected splits the sensors into groups that can never
  exchange information, however many layers the network has.
- `diameter` — the largest number of edges between any two sensors. A network
  with fewer graph convolution layers than the diameter cannot let every pair of
  sensors influence each other, whatever it learns.
- `articulation_points` — sensors whose removal would split the graph in two.
  Every message between the two halves passes through one of them, which makes
  them the natural target of a graph-native explanation.
- `algebraic_connectivity` — the second-smallest eigenvalue of the Laplacian.
  Near zero means the graph nearly falls into two pieces; larger means it is
  well-knit. It is reported as `None` for a disconnected graph, where it is zero
  by definition and says nothing.
- `distance_histogram`, `degrees`, `density`, `n_nodes`, `n_edges`,
  `isolated_nodes`, `is_symmetric`, `has_self_loops` — plain structure.

On the paper's own graph the report gives 41 edges, a diameter of 4, and one
articulation point, `T24`. Those three values are pinned by
`test_example_adjacency_report` in `tests/test_examples.py`, so a change to the
transcribed matrix cannot pass unnoticed.

`pairs_beyond_reach` turns the same idea into a list. One graph convolution
layer moves information one edge, so two sensors further apart than the number
of layers cannot influence each other at all. With the three layers of the
paper's Table 4, on the paper's own graph, four pairs are out of reach.

These numbers come from `networkx`, not from code written here. A
breadth-first search written for one project is tested by that project alone,
and these numbers appear in the thesis. `to_networkx` does the conversion;
it ignores self-loops, because a node is never its own neighbour for the
purpose of counting distance.

## Traps

- **The correlation graph must be built from training data only.**
  `build_correlation_adjacency` connects two sensors when the absolute value of
  their Pearson correlation coefficient reaches a threshold. (The Pearson
  correlation coefficient measures how closely two quantities follow a straight
  line: -1 means one rises exactly as the other falls, 0 means no straight-line
  relationship, +1 means they rise and fall together. The absolute value is used
  because a strong inverse relationship is as informative as a strong direct
  one.) Feeding it test data leaks test information into the model, for the same
  reason that scaling statistics are computed on training data only. It exists
  so the hand-designed graph can be compared directly against the data-driven
  approach everyone else uses.
- **The transcription is checked when the module is imported.** The bottom of
  `adjacency.py` asserts that the matrix is 14-by-14, symmetric, zero on the
  diagonal, and connected. A silent transcription error would corrupt every
  downstream result while leaving the code apparently working, so it is caught
  at import rather than left to a test that might not be run.
- **`normalize_adjacency` rejects an asymmetric matrix.** An asymmetric matrix
  would mean sensor i is a neighbour of j but not the other way round, which the
  symmetric normalisation formula does not support.

## Judgement calls made in this folder

Both are recorded in full in `docs/ASSUMPTIONS.md`.

- **What the "without physics prior" ablation replaces the graph with.** The
  paper names that ablation — it removes the engineering knowledge while keeping
  the rest of the architecture identical, so any change in accuracy is
  attributable to the graph alone — but never says what matrix takes the hand
  designed one's place. A graph with no edges would ablate message passing
  itself, which the paper already reports separately as "without GCN", so
  `build_fully_connected_adjacency` uses a complete graph instead: every sensor
  joined to every other. That choice is recorded as **A-19**.
- **Which normalisation formula to use.** The paper's Equation 8 says only that
  "A denotes the normalized adjacency matrix incorporating self-loops" and that
  the normalisation "ensures stable gradient propagation and balanced feature
  aggregation across different node degrees". The formula itself is never
  printed, and both the symmetric and the row form fit that description. The
  default of `normalize_adjacency` is the symmetric form, because it is the one
  defined by the layer the paper cites. That choice is recorded as **A-21**;
  read the entry there before quoting either form, because it has been
  re-measured since and the project's own configuration default lives in
  `config.py`, not here.

## References

- Corso, G., Stark, H., Jegelka, S., Jaakkola, T., & Barzilay, R. (2024). Graph
  neural networks. *Nature Reviews Methods Primers*, 4, 17.
  https://doi.org/10.1038/s43586-024-00294-7
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph
  convolutional networks. *International Conference on Learning Representations
  (ICLR)*. https://arxiv.org/abs/1609.02907
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
  168, 110875. https://doi.org/10.1016/j.ast.2025.110875

## What each name means

Everything `adjacency.py` offers.

| Name | What it is |
|---|---|
| `build_physics_adjacency()` | Returns the paper's own graph and the sensor name of each row: a 14-by-14 grid of zeros and ones with zeros down the diagonal, and the 14 names in order. |
| `build_fully_connected_adjacency(n_nodes=14)` | Returns a graph where every sensor is joined to every other, for the comparison described above. Same pair of return values. |
| `build_correlation_adjacency(frame, sensor_columns, threshold=0.5)` | Builds a graph from data instead: joins two sensors when their correlation reaches `threshold`. **Training data only.** Same pair of return values. |
| `normalize_adjacency(matrix, add_self_loops=True, method="symmetric")` | Rescales a graph for the network, in one of the two forms above. Rejects a matrix that is not square or not symmetric, and one where a node has nothing joined to it. Set `add_self_loops=False` only to demonstrate what the self-loops do; the network needs them. |
| `adjacency_report(matrix, symbols)` | Measures a graph. Returns the dictionary described above. |
| `pairs_beyond_reach(matrix, symbols, n_layers)` | The sensor pairs too far apart for that many layers to connect, as `(name, name, distance)`, sorted by distance and then by name. Empty when every pair is within reach. |
| `adjacency_to_frame(matrix, symbols)` | The grid as a labelled table, for looking at in a notebook. |
| `to_networkx(matrix, symbols)` | Hands the graph to the `networkx` library, with the sensors named. Self-loops are ignored. |
| `PHYSICS_ADJACENCY_ROWS` | Table 3 of the paper, copied out row by row. The row order matches `PAPER_SENSOR_SYMBOLS` and must not change. |
| `AdjacencyNormalization` | The two allowed values of `method`: `"symmetric"` and `"row"`. |

The first three all return the same pair — the grid of numbers, then the sensor
names in row order — so they can be swapped for one another anywhere.
