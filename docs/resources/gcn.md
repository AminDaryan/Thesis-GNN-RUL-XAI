# GCN (2017)

The graph layers in this project have the form that Thomas Kipf and Max
Welling introduced in 2017 under the name **graph convolutional network**, or
**GCN** (Kipf & Welling, 2017, [Section 2](../references/kipf_2017_graph_convolutional_networks.pdf#page=2)). This page explains that layer from their paper: the problem they set
out to solve, how they arrived at the layer, what they claimed for it and on
what evidence, and what this project takes from it.

How this project runs the layer, step by step on real engine data, is taught
elsewhere. [GNN: connections between sensors](../learn/02_graphs.md) builds
and rescales the sensor graph, and [Architecture and training](../learn/05_the_model.md)
follows one window's readings through the layers. This page does not repeat
them; it is the paper's account.

Equation, section, table and figure numbers below are those of each paper.
Links open the PDF at the page where the passage is.

## The problem the paper set out to solve

### One large graph, a few labels

Kipf and Welling (2017, [Section 1](../references/kipf_2017_graph_convolutional_networks.pdf#page=1))
set out to classify the nodes of one large graph when the class of only a few
nodes is known.
Their main example is a **citation network**: a node is a document, an edge
joins two documents when one cites the other, and each document has a class
label. Each node also carries a **feature vector**, a list of numbers
describing that node on its own. For the citation networks it is a sparse
bag-of-words vector, with one entry per word of a fixed vocabulary; sparse
means most entries are zero
([Section 5.1](../references/kipf_2017_graph_convolutional_networks.pdf#page=6)).

The Cora network, for instance, has 2,708 documents, 5,429 citation links,
7 classes and 1,433 entries in each feature vector
(Kipf & Welling, 2017, [Table 1](../references/kipf_2017_graph_convolutional_networks.pdf#page=5)).
Training is given the class of only 20 documents per class (Section 5.1), so
20 × 7 = 140 of the 2,708, and 140 / 2,708 rounds to the label rate of 0.052
that Table 1 lists for Cora. The task is called **semi-supervised** because
both kinds of node take part: the 140 labelled documents supply the errors to
correct, and training sees the features of every document, labelled or not
(Section 5.1).

### Putting the graph inside the model

A common earlier approach used the graph only in the training error: it added
a penalty that grows whenever two connected nodes receive different outputs
(Kipf & Welling, 2017, Section 1, Eq. 1). That penalty assumes that connected nodes probably share a class.
Kipf and Welling point out that an edge need not mean similarity and can carry
other information, so the assumption can limit what the model learns
(Section 1).

Their alternative is to hand the graph to the network itself. The network is a
function $f(X, A)$ of the **node feature matrix** $X$, one row of features per
node, and the **adjacency matrix** $A$, an $N \times N$ table with a 1 where two
of the $N$ nodes are connected and 0 elsewhere; the paper allows an edge weight
in place of the 1, and its citation graphs use 1 (Kipf & Welling, 2017,
Sections 1 and 5.1).
The training error is computed on the labelled nodes only (Section 1).

What makes the unlabelled nodes matter is that a labelled node's output is
computed from its neighbours' features too, labelled or not. Those features
therefore shape the correction that training makes to the layer's learned
multipliers. The same multipliers process every node, so the correction reaches
the unlabelled nodes as well. No penalty term is needed for that. Kipf and
Welling (2017, Section 1, p. 1) put it in one sentence: conditioning the
model on the adjacency matrix will "distribute gradient information from the
supervised loss" and let it "learn representations of nodes both with and
without labels". The step-by-step reasoning above is this page's.

**Do not confuse the adjacency matrix with the node feature matrix.** $A$ says
which nodes are connected and is the same in every layer; $X$ says what was
measured at each node and is what the layers transform.

The paper states two contributions (Kipf & Welling, 2017, Section 1): a layer-by-layer rule for
passing information along a graph, motivated by an approximation of spectral
graph convolutions (explained in the next section), and the use of that rule
for fast semi-supervised classification of nodes.

### How this differs from this project's use

| | Kipf and Welling (2017) | This project |
|---|---|---|
| Graph | One graph of thousands of nodes (Table 1) | Many small graphs: every window is its own copy of the same 14-node sensor graph, carrying that window's readings |
| What a node carries | One document's word vector | One sensor's value or values for one window |
| What is predicted | A class for every node (Eq. 9) | One remaining-life number per window, that is, per graph |
| From node outputs to the answer | Nothing further: each node's own output is its answer | The 14 node rows are placed end to end and passed through ordinary layers (Wen et al., 2026, [Eqs. 11–13](../references/wen_2026_estgnn_dynamic_shap.pdf#page=6); the [Flatten step](../learn/05_the_model.md#4--flatten)) |
| Known answers | A few labelled nodes (Section 5.1) | A label for every training window |
| Error measure | Cross-entropy over the labelled nodes (Eq. 10) | Mean squared error (Wen et al., 2026, [Table 6](../references/wen_2026_estgnn_dynamic_shap.pdf#page=9)) |
| One training step uses | The whole graph (Section 3.1) | A batch of 32 windows (Wen et al., 2026, Table 6): 32 graphs of 14 nodes, shape [32, 14, values per node]; Wen et al.'s (2026, [Table 4](../references/wen_2026_estgnn_dynamic_shap.pdf#page=7)) network has one value per node, as this project's default does (assumption A-07) |

**Do not confuse the paper's test nodes with this project's test engines.** In
Kipf and Welling's setting the test documents sit in the training graph: their
features flow through the layers during training and only their classes are
hidden (Kipf & Welling, 2017, Section 5.1). The paper calls this **transductive** classification
([Section 4.2](../references/kipf_2017_graph_convolutional_networks.pdf#page=5)),
meaning the cases to be classified are present while the model is trained. In
this project the test engines are never shown to the model during training,
not even through the rescaling statistics (assumption A-25). Letting their readings in would be **leakage**: information about the cases
being scored reaching the model before it is scored.

## From spectral filters to one matrix multiplication

Section 2 does not derive the layer; it motivates it (Kipf & Welling, 2017,
[p. 2](../references/kipf_2017_graph_convolutional_networks.pdf#page=2)). It starts from an exact
but expensive operation, simplifies it in three steps, and then replaces the
result with a better-behaved matrix in a fourth. Each step below is followed by
its arithmetic on this project's sensor graph and one real engine reading.

### The graph and the reading used below

The graph is this project's Physics graph: the 41 connections among the
fourteen sensors that Wen et al. (2026, [Table 3](../references/wen_2026_estgnn_dynamic_shap.pdf#page=6))
print. The reading is FD001 training engine 1 at cycle 192, its last recorded
cycle; in the training files every engine runs until it fails (NASA
Prognostics Center of Excellence [NASA PCoE], 2008; see
[C-MAPSS dataset](cmapss.md#the-four-released-subsets)).

The fourteen values have been through the project's default preprocessing.
They were smoothed over time as Wen et al. (2026, [Eq. 18](../references/wen_2026_estgnn_dynamic_shap.pdf#page=8))
describe, with strength 0.5 (assumption A-04). They were then put on a common scale: for each
sensor, subtract its mean and divide by its **standard deviation**, the typical
distance of its values from their mean (assumption A-03). The mean and standard
deviation come from the 80 engines kept for training once a fifth of FD001's
100 training engines (NASA PCoE, 2008) are held back (assumptions A-06 and
A-25). The
[preprocessing page](../learn/04_the_pipeline.md#4--put-readings-on-a-comparable-scale)
explains these steps. Nine of the fourteen values lie between two and three
standard deviations from the mean, in one direction or the other. P15 barely
varies in FD001, so its value is the same at every cycle of this engine
(finding F-01).

<!-- sample: reading -->
| Engine | Cycle | T24 | T30 | T50 | P15 | P30 | Nf | Nc | Ps30 | phi | NRf | BPR | htBleed | W31 | W32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 192 | 2.027002 | 2.040299 | 2.149321 | 0.227969 | -2.714381 | 3.139698 | -1.327671 | 2.696151 | -2.215010 | 3.310080 | 2.207709 | 1.885497 | -2.166420 | -2.717634 |

<details class="process-example">
<summary>Set up the Python examples</summary>

```python
import numpy as np
import pandas as pd
from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.graph.adjacency import (
    build_physics_adjacency, normalize_adjacency, pairs_beyond_reach)

raw, names = build_physics_adjacency()
names = list(names)
A = raw.astype(np.float64)
assert A.sum() / 2 == 41                # 41 connections among the 14 sensors
identity = np.eye(len(names))
degree = A.sum(axis=1)                  # d: how many other sensors each one is connected to
A_tilde = A + identity                  # every sensor also connected to itself
degree_tilde = A_tilde.sum(axis=1)      # the same count, including itself

def divide_by_both_ends(matrix, counts):
    """Divide entry (i, j) by the square root of counts[i] * counts[j]."""
    root = 1 / np.sqrt(counts)
    return root[:, None] * matrix * root[None, :]

laplacian = identity - divide_by_both_ends(A, degree)        # L of Section 2.1
before = identity + divide_by_both_ends(A, degree)           # the matrix of Eq. 7
after = divide_by_both_ends(A_tilde, degree_tilde)           # after the renormalisation trick
row_form = A_tilde / degree_tilde[:, None]                   # the project's default rescaling

# The project's own function builds the same matrices. It stores them with about
# seven significant digits, so the page recomputes them at full precision.
np.testing.assert_allclose(before, identity + normalize_adjacency(raw, add_self_loops=False), atol=1e-6)
np.testing.assert_allclose(after, normalize_adjacency(raw, method="symmetric"), atol=1e-6)
np.testing.assert_allclose(row_form, normalize_adjacency(raw, method="row"), atol=1e-6)
np.testing.assert_allclose(before, 2 * identity - laplacian)

prepared = run_preprocessing(PreprocessingConfig(seed=42))
windows = prepared.train                                     # the engines kept for training
assert len(set(windows.units.tolist())) == 80                # FD001's 100, less the fifth held back
engine_1 = np.flatnonzero(windows.units == 1)
last = int(engine_1[np.argmax(windows.end_cycles[engine_1])])
assert windows.end_cycles[last] == 192
x = windows.features[last, :, -1].astype(np.float64)        # the window's newest cycle
p15, t24 = names.index("P15"), names.index("T24")
assert np.unique(windows.features[engine_1, p15, :]).size == 1   # P15 never changes for engine 1
assert np.sum((np.abs(x) >= 2) & (np.abs(x) <= 3)) == 9
reading = pd.DataFrame([x], columns=names)
reading.insert(0, "Cycle", [192])
reading.insert(0, "Engine", [1])
```

</details>

### Prerequisites: degrees, the Laplacian, eigenvectors

A node's **degree** $d_i$ is how many edges it has: the sum of row $i$ of $A$.
The paper writes $D$ for the $N \times N$ matrix with the degrees on its
diagonal and zeros elsewhere (Kipf & Welling, 2017, Section 1, below Eq. 1).
Section 2.1 then uses
the **normalised graph Laplacian**

$$L = I_N - D^{-1/2} A D^{-1/2}$$

- $I_N$ is the identity matrix: ones on the diagonal, zeros elsewhere.
- $D^{-1/2} A D^{-1/2}$ divides each edge's 1 by the square root of the
  degrees at its two ends, so its entry $(i, j)$ is $1/\sqrt{d_i d_j}$ when
  $i$ and $j$ are connected and 0 otherwise.
- Multiplying $L$ by a list $x$ that holds one number per node gives, at
  node $i$, its own value minus a degree-weighted sum of its neighbours'
  values: how much node $i$ differs from its surroundings.

On the reading above, at P15: P15 has degree 4 and is connected to T24
(degree 7) and to Nf, NRf and BPR (degree 4 each; the
[GNN page](../learn/02_graphs.md#2--rescale-the-graph) lists every sensor's
count). So

$$(Lx)_{\mathrm{P15}} = 0.227969 - \left(\frac{2.027002}{\sqrt{4 \times 7}} + \frac{3.139698 + 3.310080 + 2.207709}{\sqrt{4 \times 4}}\right) = 0.227969 - 2.547439 = -2.319470$$

P15 reads far below the weighted sum of its neighbours, so the result is
strongly negative. The derivation needs $L$ because its eigenvectors are what
the first step below filters.

An **eigenvector** of a square matrix $M$ is a list of numbers $v$ that $M$
only stretches, $Mv = \lambda v$; the single number $\lambda$ is its
**eigenvalue**, the stretch factor. A **symmetric** matrix, one whose entry
$(i, j)$ equals its entry $(j, i)$ as $L$'s does, has $N$ eigenvectors at right
angles to one another. So any list of $N$ node values can be split into $N$
pieces, one along each eigenvector. Multiplying by $M$ multiplies each piece by
its own eigenvalue, and multiplying $k$ times multiplies it by $\lambda^k$. A
piece with eigenvalue 2 doubles every time; one with eigenvalue 0.5 halves.

**What an eigenvalue of $L$ measures.** For any list $x$,

$$x^\top L x = \sum_{(i,j)} \left(\frac{x_i}{\sqrt{d_i}} - \frac{x_j}{\sqrt{d_j}}\right)^2$$

where the sum runs over the connections, each counted once (41 here): divide
each node's value by the square root of its degree, then add up the squared
differences across every connection. For an eigenvector $u$ of length 1, the
left side is its eigenvalue. So an eigenvector with a small eigenvalue hardly
changes from one end of a connection to the other, and one with a large
eigenvalue changes sharply between neighbours. The paper does not print this
identity; the Python example below checks it for every eigenvector of this
graph. On this graph the two extremes are:

<!-- sample: smooth_and_sharp -->
| Sensor | Eigenvalue 0, entry divided by square root of d | Eigenvalue 1.406711, entry divided by square root of d |
|---|---|---|
| T24 | 0.110432 | 0.132647 |
| T30 | 0.110432 | -0.141780 |
| T50 | 0.110432 | 0.207613 |
| P15 | 0.110432 | -0.028669 |
| P30 | 0.110432 | -0.060593 |
| Nf | 0.110432 | -0.028669 |
| Nc | 0.110432 | 0.035846 |
| Ps30 | 0.110432 | -0.060593 |
| phi | 0.110432 | 0.014395 |
| NRf | 0.110432 | -0.028669 |
| BPR | 0.110432 | -0.028669 |
| htBleed | 0.110432 | 0.156147 |
| W31 | 0.110432 | 0.035846 |
| W32 | 0.110432 | -0.267667 |

The eigenvector with eigenvalue 0 is the same number at every sensor once
divided by the square root of the degree: nothing changes across any
connection. The one with the largest eigenvalue, 1.406711, puts T50 at 0.207613
and its neighbour W32 at −0.267667, and T30, also T50's neighbour, at
−0.141780.

<details class="process-example">
<summary>Example in Python</summary>

```python
nf, nrf, bpr = names.index("Nf"), names.index("NRf"), names.index("BPR")
assert degree[p15] == 4 and degree[t24] == 7 and degree[nf] == degree[nrf] == degree[bpr] == 4
neighbour_sum = x[t24] / np.sqrt(4 * 7) + (x[nf] + x[nrf] + x[bpr]) / np.sqrt(4 * 4)
assert np.isclose((laplacian @ x)[p15], x[p15] - neighbour_sum)
assert round(float(neighbour_sum), 6) == 2.547439
assert round(float((laplacian @ x)[p15]), 6) == -2.319470

lam, U = np.linalg.eigh(laplacian)          # eigenvalues from smallest to largest, eigenvectors as columns
first, second = np.nonzero(np.triu(A))      # the 41 connections, each listed once

def change_across_connections(u):
    """Sum over connections of the squared difference of u / sqrt(d) at the two ends."""
    scaled = u / np.sqrt(degree)
    return float(np.sum((scaled[first] - scaled[second]) ** 2))

for k in range(len(names)):
    assert np.isclose(change_across_connections(U[:, k]), lam[k])
assert np.isclose(lam[0], 0) and round(float(lam[-1]), 6) == 1.406711
smoothest = U[:, 0] * np.sign(U[t24, 0])    # an eigenvector's sign is arbitrary: make T24's entry positive
sharpest = U[:, -1] * np.sign(U[t24, -1])
smooth_and_sharp = pd.DataFrame({"Sensor": names,
    "Eigenvalue 0, entry divided by square root of d": smoothest / np.sqrt(degree),
    "Eigenvalue 1.406711, entry divided by square root of d": sharpest / np.sqrt(degree)})
```

</details>

### Step 0: a filter defined on the eigenvectors (Eq. 3)

Kipf and Welling (2017, [Section 2.1](../references/kipf_2017_graph_convolutional_networks.pdf#page=2))
define a **spectral convolution**, an operation defined through the
eigenvectors and eigenvalues of $L$, of a signal $x \in \mathbb{R}^N$, meaning
one number per node:

$$g_\theta \star x = U g_\theta U^\top x$$

- $U$ is the $N \times N$ matrix whose columns are the eigenvectors of $L$,
  so that $L = U \Lambda U^\top$, with $\Lambda$ the diagonal matrix of its
  eigenvalues.
- $U^\top x$ lists how much of each eigenvector $x$ contains, one number per
  eigenvector. The paper calls it the graph Fourier transform of $x$, a name
  from signal processing, where the Fourier transform splits a sound into
  components that change slowly or quickly; on a graph the eigenvectors of $L$,
  ordered by eigenvalue from smooth to sharply changing, play that part.
- $g_\theta = \mathrm{diag}(\theta)$ multiplies each of those amounts by its
  own number $\theta_i$. These $N$ numbers are the filter, the thing a network
  would learn; the paper reads them as a function of the eigenvalues,
  $g_\theta(\Lambda)$.
- The final multiplication by $U$ adds the rescaled eigenvectors back into one
  number per node.

In words: measure how much of each eigenvector the node values contain, turn
each amount up or down, and reassemble. Because the eigenvectors are at right
angles and of length 1, the squares of the amounts add up to the squared length
of $x$: 75.571499 for the reading. Grouped by eigenvalue (the eigenvalue 1.25
has three eigenvectors, and only the total over the three is fixed by the
matrix, not how it splits among them):

<!-- sample: pieces -->
| Eigenvalue of L | Eigenvectors | Squared amount of the reading | Squared amount after Eq. 7 with θ = 1 |
|---|---|---|---|
| 0.000000 | 1 | 3.692179 | 14.768715 |
| 0.117871 | 1 | 22.347261 | 79.163141 |
| 0.686028 | 1 | 0.057908 | 0.099979 |
| 0.968596 | 1 | 5.183250 | 5.513907 |
| 1.078893 | 1 | 0.524361 | 0.444888 |
| 1.125000 | 1 | 0.351749 | 0.269308 |
| 1.133409 | 1 | 10.311889 | 7.744022 |
| 1.142857 | 1 | 14.636928 | 10.753662 |
| 1.250000 | 3 | 6.002448 | 3.376377 |
| 1.260930 | 1 | 3.967316 | 2.167045 |
| 1.329705 | 1 | 4.167646 | 1.872506 |
| 1.406711 | 1 | 4.328563 | 1.523619 |

<details class="process-example">
<summary>Example in Python</summary>

```python
amounts = U.T @ x                           # how much of each eigenvector the reading contains
assert np.isclose(np.sum(amounts ** 2), np.sum(x ** 2))
assert round(float(np.sum(x ** 2)), 6) == 75.571499
groups = pd.DataFrame({"Eigenvalue of L": np.round(lam, 9) + 0.0,
                       "before": amounts ** 2,
                       "after": (amounts * (2 - lam)) ** 2})
pieces = (groups.groupby("Eigenvalue of L", as_index=False)
          .agg(**{"Eigenvectors": ("before", "size"),
                  "Squared amount of the reading": ("before", "sum"),
                  "Squared amount after Eq. 7 with θ = 1": ("after", "sum")}))
```

</details>

The last column belongs to Step 3 below. The obstacle to Eq. 3 is cost:
multiplying by $U$ takes on the order of $N^2$ operations, and finding $U$ at
all can be prohibitively expensive for a large graph (Kipf & Welling, 2017,
Section 2.1). For Cora's
2,708 nodes, $N^2$ is over seven million. The next step removes $U$.

### Step 1: a polynomial instead of free numbers (Eq. 4 and 5)

The paper takes from Hammond et al. (2011, as cited in Kipf & Welling, 2017,
Section 2.1) the idea of replacing the $N$ free numbers with a polynomial in the eigenvalues, built from
**Chebyshev polynomials** (Eq. 4). These are a fixed family defined by
$T_0(x) = 1$, $T_1(x) = x$ and $T_k(x) = 2x\,T_{k-1}(x) - T_{k-2}(x)$, so that
$T_2(x) = 2x^2 - 1$. The eigenvalues are first rescaled to
$\tilde\Lambda = \frac{2}{\lambda_{\max}}\Lambda - I_N$, where $\lambda_{\max}$
is the largest eigenvalue of $L$; this maps the smallest possible eigenvalue,
0, to −1 and the largest, $\lambda_{\max}$, to 1.

Why rescale at all? The paper refers the reader to Hammond et al. (2011) for
the details (Kipf & Welling, 2017, Section 2.1); the reason, supplied here, is that Chebyshev polynomials stay between
−1 and 1 for inputs between −1 and 1 and grow fast outside that interval:
$T_2(0.5) = 2(0.25) - 1 = -0.5$, but $T_2(3) = 2(9) - 1 = 17$. With every
eigenvalue moved into $[-1, 1]$, no term of the sum can blow up.

A polynomial in the eigenvalues is the same polynomial in the matrix, because
$(U\Lambda U^\top)^k = U\Lambda^k U^\top$ (the paper's justification: Kipf &
Welling, 2017, Section 2.1, below Eq. 5), which
holds because each inner $U^\top U$ is the identity. So the eigenvectors are no
longer needed:

$$g_{\theta'} \star x \approx \sum_{k=0}^{K} \theta'_k \, T_k(\tilde L)\, x, \qquad \tilde L = \frac{2}{\lambda_{\max}} L - I_N$$

- $K$ is the degree of the polynomial, chosen before training.
- $\theta'_k$ are the learned coefficients, $K + 1$ of them. (The paper writes
  $\theta' \in \mathbb{R}^K$, but the sum from $k = 0$ to $K$ has $K + 1$ terms.)
- $T_k(\tilde L)\,x$ needs only $k$ multiplications of a list by $\tilde L$.

This is Eq. 5. With $K = 1$, $\theta'_0 = 0$ and $\theta'_1 = 1$, it is
$\tilde L x$; at P15, with this graph's $\lambda_{\max} = 1.406711$,

$$(\tilde L x)_{\mathrm{P15}} = \frac{2}{1.406711} \times (-2.319470) - 0.227969 = -3.525690$$

$\tilde L$ is zero wherever two different nodes are not connected, so each
multiplication by it moves information one edge further. The paper draws two
consequences (Kipf & Welling, 2017, Section 2.1): the filter is **$K$-localised**, depending only on
nodes at most $K$ edges away, and its cost grows in proportion to the number of
edges instead of $N^2$. On the Physics graph, for every pair of sensors more
than $k$ connections apart, the entry of $L^k$ for that pair is exactly zero,
for $k$ = 1, 2 and 3. How many pairs each depth reaches is drawn on the
[GNN page](../learn/02_graphs.md#how-information-travels-through-layers).

<details class="process-example">
<summary>Example in Python</summary>

```python
def chebyshev_2(value):
    """The second Chebyshev polynomial, 2 x^2 - 1."""
    return 2 * value ** 2 - 1

assert chebyshev_2(0.5) == -0.5 and chebyshev_2(3) == 17
lambda_max = lam[-1]
L_rescaled = 2 / lambda_max * laplacian - identity
assert np.abs(np.linalg.eigvalsh(L_rescaled)).max() <= 1 + 1e-12
assert round(float((L_rescaled @ x)[p15]), 6) == -3.525690

for hops in (1, 2, 3):
    power = np.linalg.matrix_power(laplacian, hops)
    for one, other, distance in pairs_beyond_reach(raw, names, hops):
        assert power[names.index(one), names.index(other)] == 0
```

</details>

### Step 2: first order, and the largest eigenvalue taken as 2 (Eq. 6)

Section 2.2 keeps only $K = 1$
(Kipf & Welling, 2017, [pp. 2–3](../references/kipf_2017_graph_convolutional_networks.pdf#page=2)), so
one layer reaches only direct neighbours, and relies on stacking layers to reach
further: $k$ stacked layers cover a node's neighbourhood up to $k$ edges away.
Its reasons ([p. 3](../references/kipf_2017_graph_convolutional_networks.pdf#page=3))
are that stacking several such layers can still express many different filters
without being tied to the Chebyshev form; that it expects less
**overfitting**, fitting the training examples so closely, noise included,
that accuracy on new ones drops, to the local structure of graphs whose nodes
have very different numbers of neighbours; and that, for a fixed amount of
computation, it allows deeper models.

It also approximates $\lambda_{\max} \approx 2$, expecting the learned
parameters to adapt to the change of scale during training (Kipf & Welling,
2017, Section 2.2, p. 3). Why 2 in
particular is not spelled out; it is the largest value $\lambda_{\max}$ can take
on any graph, as the section
[Eigenvalues before and after the trick](#eigenvalues-before-and-after-the-trick)
shows. With $\lambda_{\max} = 2$, $\tilde L = L - I_N = -D^{-1/2} A D^{-1/2}$,
and Eq. 5 with $K = 1$ becomes Eq. 6:

$$g_{\theta'} \star x \approx \theta'_0\, x - \theta'_1\, D^{-1/2} A D^{-1/2} x$$

Two learned numbers: $\theta'_0$ multiplies the node's own value and
$\theta'_1$ the degree-weighted sum of its neighbours' values. The
approximation is loose on this project's graph, whose $\lambda_{\max}$ is
1.406711, not 2.

### Step 3: one parameter (Eq. 7)

To cut the parameters further, against overfitting, and the operations per
layer, the paper ties the two numbers together, $\theta = \theta'_0 = -\theta'_1$
(Kipf & Welling, 2017, Section 2.2). This is Eq. 7:

$$g_\theta \star x \approx \theta \left(I_N + D^{-1/2} A D^{-1/2}\right) x$$

A node's own value and its neighbours' weighted sum now enter with the same
multiplier $\theta$. At P15 with $\theta = 1$ that is
$0.227969 + 2.547439 = 2.775408$.

Eq. 7 is still a filter of the kind Eq. 3 defines: since
$I_N + D^{-1/2}AD^{-1/2} = 2I_N - L$, it multiplies the amount of $x$ along
each eigenvector of $L$ by $\theta(2 - \lambda)$, so each squared amount by
$\theta^2(2 - \lambda)^2$. That is the last column of the table in Step 0: the
eigenvalue-0 part grows fourfold, from 3.692179 to 14.768715, and the
1.406711 part shrinks to about a third, from 4.328563 to 1.523619. Reassembling the rescaled eigenvectors
gives 2.775408 at P15 again, the same as the direct multiplication.

<details class="process-example">
<summary>Example in Python</summary>

```python
filtered = U @ np.diag(2 - lam) @ amounts   # Eq. 3 with each amount multiplied by 2 - lambda
np.testing.assert_allclose(filtered, before @ x)
assert np.isclose(filtered[p15], x[p15] + neighbour_sum)
assert round(float(filtered[p15]), 6) == 2.775408
assert np.isclose(pieces["Squared amount after Eq. 7 with θ = 1"].sum(), np.sum((before @ x) ** 2))
```

</details>

### Step 4: the renormalisation trick

Section 2.2 then notes that the matrix in Eq. 7 has eigenvalues between 0 and
2, and concludes that applying it repeatedly in a deep network can cause
**numerical instabilities**, numbers too large or too small for the computer to
hold accurately, and exploding or vanishing gradients (Kipf & Welling, 2017,
Section 2.2, p. 3). A **gradient** here is
the rate at which the training error changes when one learned multiplier
changes; training moves each multiplier a small step against its gradient, so
gradients that grow without bound or shrink to nothing make training fail. The
paper does not spell out why the eigenvalue range leads there; the eigenvector
picture above supplies the reason: every application multiplies each piece by
its eigenvalue, so pieces with an eigenvalue above 1 grow every time and pieces
below 1 shrink.

The fix, which the paper calls the **renormalisation trick**, is stated in the
text of Section 2.2, in the sentence before the one that introduces Eq. 8
(Kipf & Welling, 2017, p. 3):

$$I_N + D^{-1/2} A D^{-1/2} \;\rightarrow\; \tilde D^{-1/2} \tilde A \tilde D^{-1/2}, \qquad \tilde A = A + I_N, \qquad \tilde D_{ii} = \sum_j \tilde A_{ij}$$

- $\tilde A$ puts a 1 on the diagonal: every node becomes connected to itself,
  which the paper calls adding self-connections (below Eq. 2).
- $\tilde D_{ii}$ is the degree counted with that extra connection, $d_i + 1$.
- Each entry of $\tilde A$ is divided by $\sqrt{\tilde D_{ii} \tilde D_{jj}}$.

Before the trick, a node keeps its own value at full weight and adds its
neighbours' weighted sum on top. After it, the node enters that weighted sum as
one more neighbour, divided like the others. The trick is a substitution, not a
further step of the approximation. The new matrix is still a filter of Eq. 3's
kind, but on a different graph: it equals $I_N$ minus the normalised Laplacian
of the graph in which every node is joined to itself, so it multiplies each
eigenvector of that Laplacian by $1 - \lambda$. The section
[The stability argument on this project's graph](#the-stability-argument-on-this-projects-graph)
shows what this does to the eigenvalues.

**Do not confuse the Laplacian with the matrix a layer multiplies by.** $L$
measures how a node differs from its neighbours; the layer's matrix, before or
after the trick, mixes a node with its neighbours. Before the trick the two are
tied by $I_N + D^{-1/2}AD^{-1/2} = 2I_N - L$.

### Many values per node, and the layer rule (Eq. 8 and Eq. 2)

So far each node held one number. With $C$ numbers per node and $F$ filters,
Eq. 8 reads (Kipf & Welling, 2017, Section 2.2, p. 3)

$$Z = \tilde D^{-1/2} \tilde A \tilde D^{-1/2} X \Theta$$

- $X \in \mathbb{R}^{N \times C}$: $N$ rows, one per node, and $C$ columns,
  the values each node carries (the paper's input channels).
- $\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$, $N \times N$: fixed by the graph,
  it combines rows, so each node receives a weighted sum of itself and its
  neighbours, separately in every column.
- $\Theta \in \mathbb{R}^{C \times F}$: the learned multipliers. The single
  $\theta$ of Eq. 7 has become a table with one column per output value; it
  combines columns.
- $Z \in \mathbb{R}^{N \times F}$: $F$ new values for every node.

Shapes: $[N, C]$ stays $[N, C]$ through the graph matrix and becomes $[N, F]$
through $\Theta$. Because the graph matrix is mostly zeros, the paper gives the
cost as proportional to the number of edges times $F$ times $C$.

Section 2.2 opens by building a network from stacked layers of the form of
Eq. 5, each followed by a nonlinearity applied entry by entry (Kipf & Welling,
2017, p. 2). Stacking the layer of Eq. 8 in the same way, written layer by
layer, gives the rule Section 2 opens with, Eq. 2, whose form the section sets
out to motivate (p. 2):

$$H^{(l+1)} = \sigma\!\left(\tilde D^{-1/2} \tilde A \tilde D^{-1/2} H^{(l)} W^{(l)}\right)$$

- $H^{(l)} \in \mathbb{R}^{N \times D}$: the values every node holds entering
  layer $l$, which the paper calls the layer's activations; $H^{(0)} = X$. Here
  $D$ counts columns; it is not the matrix of degrees.
- $W^{(l)}$: layer $l$'s learned multipliers, the $\Theta$ of Eq. 8, a
  different table in every layer.
- $\sigma$: an activation function, a fixed function applied to every entry,
  such as ReLU, $\max(0, \cdot)$, which keeps a positive number and replaces a
  negative one with 0.
- No bias term, a learned constant added to every output, appears in Eq. 2 or
  Eq. 8.

Kipf and Welling (2017, [Appendix A](../references/kipf_2017_graph_convolutional_networks.pdf#page=11))
write the same rule node by node (Eq. 12): node $i$'s new values are $\sigma$
of the sum, over the nodes $j$ in its neighbourhood $\mathcal{N}_i$, of $j$'s
values divided by $c_{ij} = \sqrt{d_i d_j}$ and multiplied by $W^{(l)}$. Footnote
5 assumes every node has already been joined to itself, so $\mathcal{N}_i$
contains $i$ and Appendix A's $d_i = |\mathcal{N}_i|$ counts the
self-connection: it is this page's $d_i + 1$, that is $\tilde D_{ii}$, and the
divisor is $\sqrt{\tilde D_{ii} \tilde D_{jj}}$. The Architecture page works
one such layer through term by term with the project's default rescaling, in
[Graph layer 1](../learn/05_the_model.md#1--graph-layer-1).

The key idea is: Kipf and Welling start from a filter that turns each
eigenvector's share of the node values up or down freely, restrict it to a
first-degree polynomial so that it needs no eigenvectors and touches only
direct neighbours, tie its two numbers into one, and then swap the resulting
matrix for one in which every node counts as its own neighbour, because the
first version's eigenvalues reach 2.

## The stability argument on this project's graph

Section 2.2 argues from the range of the eigenvalues and shows no numbers
(Kipf & Welling, 2017, p. 3). This
section applies the argument to the Physics graph and the reading above.
Nothing is trained here. The point is what repeated multiplication by the graph
matrix alone does, which is what the paper's argument is about. In a network,
learned multipliers and ReLU sit between those multiplications, so this is not
a measurement of a network.

Three matrices are compared. **Before the trick** is the matrix of Eq. 7,
$I_N + D^{-1/2}AD^{-1/2}$. **After the trick** is
$\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$, which the project's
`normalize_adjacency` calls `"symmetric"`. **Row form** is
$\tilde D^{-1}\tilde A$, each row of $\tilde A$ divided by its own total, which
the project uses by default. Wen et al. (2026, Eq. 8, p. 6) describe the
rescaling without printing a formula, and both forms fit their words; the row form was chosen because it
came out slightly closer to their reported error when no engines were held
back, an advantage that disappears once engines are held back as the default
now does (assumptions A-21 and A-22).

### Row totals: why the matrix of Eq. 7 roughly doubles

A row total is what a matrix does to a list in which every sensor has the
same value: each sensor's result is that value times its row total. Before the
trick, P15 keeps itself at weight 1, takes T24 at
$1/\sqrt{4 \times 7} = 0.188982$ and each of Nf, NRf and BPR at
$1/\sqrt{4 \times 4} = 0.25$. Total: $1 + 0.188982 + 3 \times 0.25 = 1.938982$.
After the trick, P15's total is 0.958114, which the GNN page works out as its
symmetric option in [Rescale the graph](../learn/02_graphs.md#2--rescale-the-graph).
Every sensor follows the same pattern:

<!-- sample: row_totals -->
| Sensor | Row total before the trick | Row total after the trick |
|---|---|---|
| T24 | 2.175274 | 1.125307 |
| T30 | 2.105637 | 1.079797 |
| T50 | 1.780330 | 0.847214 |
| P15 | 1.938982 | 0.958114 |
| P30 | 1.983767 | 0.987184 |
| Nf | 1.938982 | 0.958114 |
| Nc | 2.148783 | 1.111017 |
| Ps30 | 1.983767 | 0.987184 |
| phi | 1.895923 | 0.921713 |
| NRf | 1.938982 | 0.958114 |
| BPR | 1.938982 | 0.958114 |
| htBleed | 2.029892 | 1.020298 |
| W31 | 2.148783 | 1.111017 |
| W32 | 1.792536 | 0.856256 |

Before the trick every total lies near 2: the sensor's own value at full
weight, plus neighbour shares that add up to about 1. After it, every total
lies near 1. Row totals describe only lists of equal values, though; the
eigenvalues describe every list.

<details class="process-example">
<summary>Example in Python</summary>

```python
assert before[p15, p15] == 1 and np.isclose(before[p15, t24], 1 / np.sqrt(4 * 7))
assert np.isclose(before[p15].sum(), 1 + 1 / np.sqrt(28) + 3 / 4)
row_totals = pd.DataFrame({"Sensor": names,
                           "Row total before the trick": before.sum(axis=1),
                           "Row total after the trick": after.sum(axis=1)})
```

</details>

### Eigenvalues before and after the trick

**One eigenvector by hand.** Before the trick, give every sensor the square
root of its count of other senders: $v_i = \sqrt{d_i}$, so $v_{P15} = 2$ and
$v_{T24} = \sqrt 7$. Multiply P15's row of the matrix by $v$ and add:

$$1 \times 2 + \frac{1}{\sqrt{4 \times 7}} \times \sqrt 7 + 3 \times \frac{1}{\sqrt{4 \times 4}} \times 2 = 2 + 0.5 + 1.5 = 4 = 2 \times v_{P15}$$

Every other row gives the same doubling, for the same reason: the neighbours
contribute $\sum_j \sqrt{d_j}/\sqrt{d_i d_j} = d_i/\sqrt{d_i} = \sqrt{d_i}$, and
the sensor's own weight 1 adds another $\sqrt{d_i}$. So $v$ is an eigenvector
with eigenvalue exactly 2, the top of the paper's range.

After the trick, the same construction with the counts that include the sensor
itself, $\tilde v_i = \sqrt{d_i + 1}$, gives for P15

$$\frac{1}{5}\sqrt 5 + \frac{1}{\sqrt{5 \times 8}} \times \sqrt 8 + 3 \times \frac{1}{5}\sqrt 5 = 5 \times \frac{1}{\sqrt 5} = \sqrt 5 = 1 \times \tilde v_{P15}$$

an eigenvector with eigenvalue exactly 1.

**All fourteen.** Each column below is one matrix's eigenvalues, sorted from
largest to smallest on its own; a row does not pair an eigenvector of one
matrix with one of another.

<!-- sample: eigenvalues -->
| Rank | Before the trick | After the trick |
|---|---|---|
| 1 | 2.000000 | 1.000000 |
| 2 | 1.882129 | 0.903079 |
| 3 | 1.313972 | 0.431716 |
| 4 | 1.031404 | 0.174821 |
| 5 | 0.921107 | 0.078768 |
| 6 | 0.875000 | 0.000000 |
| 7 | 0.866591 | 0.000000 |
| 8 | 0.857143 | 0.000000 |
| 9 | 0.750000 | 0.000000 |
| 10 | 0.750000 | 0.000000 |
| 11 | 0.750000 | 0.000000 |
| 12 | 0.739070 | -0.077354 |
| 13 | 0.670295 | -0.136510 |
| 14 | 0.593289 | -0.198330 |

What the table shows:

- **Before the trick**, all fourteen lie between 0 and 2, as the paper states
  (Kipf & Welling, 2017, Section 2.2), and four of them exceed 1. The pieces of a reading along those four
  eigenvectors grow at every application.
- **After the trick**, all lie between −1 and 1, and none exceeds 1, so no
  piece can grow.
- **The row form has the same eigenvalues** as the matrix after the trick,
  identical to six decimals. It is that matrix with every row multiplied by
  $1/\sqrt{\tilde D_{ii}}$ and every column by $\sqrt{\tilde D_{jj}}$, and such
  a rescaling never changes eigenvalues: if $\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$
  stretches $u$ by $\lambda$, the row form stretches $\tilde D^{-1/2} u$ by the
  same $\lambda$. Its eigenvectors differ, and it is not symmetric, which
  matters in the next section.
- **Six eigenvalues after the trick are zero.** $\tilde A$ has only eight
  independent rows: every one of its fourteen rows is a sum of multiples of
  eight of them. Each of the six dependencies gives a list that $\tilde A$
  sends to all zeros, an eigenvector with eigenvalue 0. Take +1 at P15, −1 at
  Nf and 0 elsewhere: P15 and Nf have exactly the same senders including
  themselves (T24, P15, Nf, NRf and BPR), and connections work both ways, so
  for every receiving sensor the share taken from P15 equals the share taken
  from Nf, and the +1 and −1 cancel. Within the groups P15, Nf, NRf and BPR; P30 and Ps30; and Nc and
  W31, whose rows are identical, this gives 3 + 1 + 1 = 5 such lists. The sixth
  comes from four different rows: P30's row plus Nc's row equals T30's row
  plus htBleed's row, so +1 at P30 and Nc with −1 at T30 and htBleed is sent to
  zero too. (For the matrix after the trick, multiply each entry of such a list
  by the square root of that sensor's count; for P15 and Nf, both 5, that
  only rescales the list.) The same groups make three sets of sensors
  indistinguishable to the network (finding F-18): after one multiplication,
  P15, Nf, NRf and BPR carry the same number, which the
  [Architecture page's first layer](../learn/05_the_model.md#1--graph-layer-1)
  shows in its combined table.
- **The largest eigenvalue of $L$ is 1.406711.** Since the matrix before the
  trick is $2I_N - L$, its smallest eigenvalue, 0.593289, is 2 minus the
  largest eigenvalue of $L$. That is the $\lambda_{\max}$ the paper approximates
  by 2 in Step 2.

**Why these ranges hold on every graph.** The paper states the range $[0, 2]$
without proof (Kipf & Welling, 2017, Section 2.2) and gives none after the trick; the argument below is this
page's. Take $D^{-1}A$, each row of $A$ divided by its degree: it replaces
every node's value with the plain average of its neighbours' values. An average
can never exceed the largest of the values it averages, in size. Now suppose
$D^{-1}A$ stretched some list $v$ by $\mu$, and look at the entry of $v$ that
is largest in size: it becomes $\mu$ times itself, yet it is an average of
entries no larger, so $|\mu| \le 1$. $D^{-1/2}AD^{-1/2}$ has the same
eigenvalues as $D^{-1}A$ by the rescaling argument above, so its eigenvalues lie
in $[-1, 1]$. Adding $I_N$ moves them to $[0, 2]$, the paper's range for Eq. 7,
and subtracting them from 1 shows that $L$'s eigenvalues also lie in $[0, 2]$:
that is why $\lambda_{\max}$ can never exceed 2. After the trick the same
argument applies to the row form, which averages each node with its
neighbours, so its eigenvalues, and those of the matrix after the trick, lie in
$[-1, 1]$. They cannot even reach −1: the largest entry would have to equal
minus an average of entries no larger that includes the entry itself, which
forces it to be zero. So after the trick the range is $(-1, 1]$, with 1 always
reached by the list proportional to $\sqrt{d_i + 1}$. The argument needs every
node to have at least one connection, which holds here.

<details class="process-example">
<summary>Example in Python</summary>

```python
np.testing.assert_allclose(before @ np.sqrt(degree), 2 * np.sqrt(degree))
np.testing.assert_allclose(after @ np.sqrt(degree_tilde), np.sqrt(degree_tilde))

def largest_first(values):
    """Sort eigenvalues from largest to smallest, writing a rounding-level zero as 0."""
    return np.round(np.sort(values)[::-1], 12) + 0.0

eigenvalues = pd.DataFrame({"Rank": range(1, len(names) + 1),
                            "Before the trick": largest_first(np.linalg.eigvalsh(before)),
                            "After the trick": largest_first(np.linalg.eigvalsh(after))})

# The neighbour average D^-1 A and the row form both have rows adding up to 1, and each
# shares its eigenvalues with the symmetric matrix of the same graph.
average = A / degree[:, None]
np.testing.assert_allclose(average.sum(axis=1), 1)
np.testing.assert_allclose(row_form.sum(axis=1), 1)
np.testing.assert_allclose(largest_first(np.linalg.eigvals(identity + average).real),
                           eigenvalues["Before the trick"], atol=1e-9)
row_values = np.linalg.eigvals(row_form)
assert np.abs(row_values.imag).max() < 1e-9
np.testing.assert_allclose(largest_first(row_values.real), eigenvalues["After the trick"], atol=1e-9)
np.testing.assert_allclose(
    row_form, np.diag(degree_tilde ** -0.5) @ after @ np.diag(degree_tilde ** 0.5))

# Eight independent rows, the three groups of identical rows, and the sixth dependency.
assert np.linalg.matrix_rank(A_tilde) == 8
same_row = {}
for i, name in enumerate(names):
    same_row.setdefault(tuple(A_tilde[i]), []).append(name)
assert [group for group in same_row.values() if len(group) > 1] == [
    ["P15", "Nf", "NRf", "BPR"], ["P30", "Ps30"], ["Nc", "W31"]]
to_zero = np.zeros(len(names))
to_zero[[p15, nf]] = 1, -1
assert np.abs(A_tilde @ to_zero).max() == 0 and np.abs(after @ to_zero).max() < 1e-15
pairing = np.zeros(len(names))
pairing[[names.index("P30"), names.index("Nc")]] = 1
pairing[[names.index("T30"), names.index("htBleed")]] = -1
assert np.abs(A_tilde @ pairing).max() == 0
assert np.abs(after @ (np.sqrt(degree_tilde) * pairing)).max() < 1e-12

assert np.isclose(lambda_max, 2 - eigenvalues["Before the trick"].min())
```

</details>

### Repeated application to a real reading

The **size** of a list of fourteen numbers is measured as its length, the
square root of the sum of their squares. For the reading:
$\sqrt{2.027002^2 + 2.040299^2 + \dots + (-2.717634)^2} = \sqrt{75.571499} = 8.693187$.
The table gives that size after multiplying the reading by each matrix 1, 2, 3
and 10 times in succession:

<!-- sample: sizes -->
| Applications | Before the trick | After the trick | Row form |
|---|---|---|---|
| 0 | 8.693187 | 8.693187 | 8.693187 |
| 1 | 11.300317 | 4.714147 | 4.755577 |
| 2 | 19.086691 | 4.308839 | 4.388300 |
| 3 | 35.343465 | 3.981995 | 4.055655 |
| 10 | 3290.164359 | 2.604172 | 2.573271 |

- **Before the trick** the size grows with every application and approaches
  doubling each time, driven by the pieces with eigenvalues 2 and 1.882129.
  After ten applications it is 3290.164359. This is the explosion Kipf and
  Welling (2017, Section 2.2) warn of, on this project's graph.
- **After the trick** the size shrinks instead, from 8.693187 to 2.604172
  after ten applications. Most of the first drop comes from the six zero
  eigenvalues: 0.422848 of the reading's squared length, 42.3 %, lies along
  those six eigenvectors, and one application removes that part entirely,
  more than half of the fall in squared length from 75.571499 to
  $4.714147^2$. The
  trick removes the growth; it does not stop the shrinking. Every piece with an
  eigenvalue between −1 and 1 fades, and only the piece along the top
  eigenvector, eigenvalue 1, keeps its size. After many applications the
  result is that piece alone: values proportional to $\sqrt{d_i + 1}$ for every
  sensor.
- **Under the row form** this reading also shrinks, and after many
  applications every sensor carries the same number, a weighted mean of the
  reading. Either way the final pattern is fixed by the graph, and the reading
  decides only its overall scale.

**Equal eigenvalues are not equal stretching.** The row form shares the
eigenvalues of the matrix after the trick, but it is not symmetric, so its
eigenvectors are not at right angles. A list can then be split into pieces
that partly cancel; when one multiplication shrinks those pieces by different
amounts, the cancellation weakens and the list can come out slightly longer.
The eigenvalues still stop growth from building up over many applications. The
table gives the most that any list of fourteen numbers can be lengthened, for
each matrix applied 1, 2, 3 and 10 times:

<!-- sample: stretch -->
| Applications | Before the trick | After the trick | Row form |
|---|---|---|---|
| 1 | 2.000000 | 1.000000 | 1.016562 |
| 2 | 4.000000 | 1.000000 | 1.016708 |
| 3 | 8.000000 | 1.000000 | 1.019157 |
| 10 | 1024.000000 | 1.000000 | 1.023915 |

Before the trick the worst list doubles at every application. After the trick
no list ever gets longer. Under the row form, some list gets up to 1.7 % longer
in one application and up to 2.4 % longer over ten, and the growth stays
bounded however many applications follow. None of the
project's training readings is such a list: over the newest cycle of all 14,459
training windows, the largest ratio of length after one row-form application
to length before is 0.828599. What does hold for the row form, for every list,
is the averaging argument above: each output is an average of inputs, so no
sensor's value can exceed the largest input in size.

**The same holds for gradients.** Training differentiates the loss, the error
measure, with respect to the learned multipliers. On the way back from the
loss to an early layer's multipliers, the calculation passes through every
multiplication by the graph matrix. If $y = Mx$ and the rates
$g_i = \partial\mathcal{L}/\partial y_i$ are already known, then $x_j$ moves
every $y_i$ by $M_{ij}$ per unit, so

$$\frac{\partial\mathcal{L}}{\partial x_j} = \sum_i g_i\, M_{ij} = (M^\top g)_j$$

For P15 before the trick, $x_{\mathrm{P15}}$ enters P15's own output with
weight 1, T24's with 0.188982 and those of Nf, NRf and BPR with 0.25 each, so
$\partial\mathcal{L}/\partial x_{\mathrm{P15}} = g_{\mathrm{P15}} + 0.188982\,g_{\mathrm{T24}} + 0.25\,(g_{\mathrm{Nf}} + g_{\mathrm{NRf}} + g_{\mathrm{BPR}})$.
Going backwards therefore multiplies by the transpose $M^\top$, the matrix
flipped about its diagonal. For the two symmetric matrices that is the matrix
itself; the row form's transpose has the same eigenvalues as the row form and
the same largest stretch, 1.016562. So before the trick a gradient can double
at each graph matrix; after the trick it cannot grow, and under the row form
it can grow only by that small stretch.

<details class="process-example">
<summary>Example in Python</summary>

```python
def size(vector):
    """Length of a list of numbers: the square root of the sum of squares."""
    return float(np.sqrt(np.sum(vector ** 2)))

assert round(size(x), 6) == 8.693187
applications = [0, 1, 2, 3, 10]
sizes = pd.DataFrame({
    "Applications": applications,
    "Before the trick": [size(np.linalg.matrix_power(before, k) @ x) for k in applications],
    "After the trick": [size(np.linalg.matrix_power(after, k) @ x) for k in applications],
    "Row form": [size(np.linalg.matrix_power(row_form, k) @ x) for k in applications]})

# The part of the reading along the six eigenvectors with eigenvalue 0 after the trick.
after_values, after_vectors = np.linalg.eigh(after)
zero_part = after_vectors[:, np.abs(after_values) < 1e-9].T @ x
assert len(zero_part) == 6
assert round(float(np.sum(zero_part ** 2) / np.sum(x ** 2)), 6) == 0.422848
assert np.sum(zero_part ** 2) > (np.sum(x ** 2) - size(after @ x) ** 2) / 2

# After many applications only the piece along the top eigenvector is left.
top = np.sqrt(degree_tilde) / size(np.sqrt(degree_tilde))
np.testing.assert_allclose(np.linalg.matrix_power(after, 200) @ x, (top @ x) * top, atol=1e-9)
weighted_mean = degree_tilde @ x / degree_tilde.sum()
np.testing.assert_allclose(np.linalg.matrix_power(row_form, 200) @ x,
                           np.full(len(names), weighted_mean), atol=1e-9)

def largest_stretch(matrix):
    """The most any list of numbers can be lengthened by one multiplication."""
    return float(np.linalg.norm(matrix, 2))

stretch = pd.DataFrame({
    "Applications": applications[1:],
    "Before the trick": [largest_stretch(np.linalg.matrix_power(before, k)) for k in applications[1:]],
    "After the trick": [largest_stretch(np.linalg.matrix_power(after, k)) for k in applications[1:]],
    "Row form": [largest_stretch(np.linalg.matrix_power(row_form, k)) for k in applications[1:]]})
assert np.isclose(largest_stretch(row_form.T), largest_stretch(row_form))

newest = windows.features[:, :, -1].astype(np.float64)      # every training window's newest cycle
growth = np.linalg.norm(newest @ row_form.T, axis=1) / np.linalg.norm(newest, axis=1)
assert len(newest) == 14459 and round(float(growth.max()), 6) == 0.828599
assert np.all(np.abs(newest @ row_form.T).max(axis=1) <= np.abs(newest).max(axis=1) + 1e-12)
```

</details>

The key idea is: a matrix applied again and again multiplies each eigenvector
piece of a reading by its eigenvalue every time, so the matrix of Eq. 7, whose
top eigenvalue is exactly 2, makes this engine's reading 3290.164359 long
after ten applications; the renormalised matrix has eigenvalues between −1 and
1 and eigenvectors at right angles, so it can never lengthen any list; the row
form shares those eigenvalues, so nothing grows without limit under it, but
because it is not symmetric one application can lengthen some lists slightly.

## The two-layer model and its training (Sections 3 and 5)

### The model (Eq. 9)

Kipf and Welling (2017, [Section 3.1](../references/kipf_2017_graph_convolutional_networks.pdf#page=3))
compute $\hat A = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ once, before
training, and use it in two layers:

$$Z = f(X, A) = \mathrm{softmax}\!\left(\hat A\; \mathrm{ReLU}\!\left(\hat A X W^{(0)}\right) W^{(1)}\right)$$

- $X$: the node features, $N \times C$. For Cora, $2{,}708 \times 1{,}433$.
- $W^{(0)}$, $C \times H$: input to hidden layer, with $H$ hidden values per
  node. $W^{(1)}$, $H \times F$: hidden to output, one output per class
  ([Section 3.1](../references/kipf_2017_graph_convolutional_networks.pdf#page=4)).
- **Softmax**, applied to each node's row, turns $F$ scores into $F$
  probabilities: each score is exponentiated and divided by the sum of the
  exponentials, so the results are positive and add up to 1. Seven equal
  scores, for Cora's seven classes, give $1/7$ each.
- $Z$, $N \times F$: for every node, a probability for every class.

For Cora with the 16 hidden values the paper used (Kipf & Welling, 2017,
Section 6.1), the shapes run
$[2708, 1433] \to [2708, 16] \to [2708, 7]$. Because $\hat A$ is applied twice,
each node's output depends on nodes up to two edges away.

### The loss (Eq. 10)

$$\mathcal{L} = -\sum_{l \in \mathcal{Y}_L} \sum_{f=1}^{F} Y_{lf} \ln Z_{lf}$$

- $\mathcal{Y}_L$: the set of labelled nodes; every other node is left out of
  the sum. Here $l$ is a labelled node, not a layer number as in Eq. 2.
- $Y_{lf}$: 1 if node $l$ belongs to class $f$, otherwise 0. Only the true
  class's term survives the inner sum.
- $Z_{lf}$: the model's probability for class $f$ at node $l$.

This **cross-entropy** charges each labelled node $-\ln$ of the probability
the model gave its true class. A model that gives all seven Cora classes $1/7$
pays $-\ln(1/7) = \ln 7 = 1.945910$ per labelled node; a model that gives the
true class probability 1 pays 0. Gradient descent, which moves each multiplier
a small step against its gradient, then adjusts $W^{(0)}$ and $W^{(1)}$ to
lower $\mathcal{L}$ (Kipf & Welling, 2017, Section 3.1).

### How the paper trained it (Sections 3.1, 5.2 and 6.1)

- **The whole graph in every step.** Each training iteration uses the full
  dataset. Within a run, the randomness in training comes from **dropout**
  (Kipf & Welling, 2017, Section 3.1), which sets a random share of the values
  to zero during training; separate runs also start from different random
  multipliers (Section 6.1).
- **Two layers**, evaluated on 1,000 labelled test nodes, with a separate
  500 labelled validation nodes used to choose the dropout rate, the strength
  of a penalty on the first layer's multipliers and the number of hidden
  values; the validation labels are not used for training
  ([Section 5.2](../references/kipf_2017_graph_convolutional_networks.pdf#page=6)).
  Those settings were chosen on Cora only and reused for Citeseer and Pubmed.
  The penalty is **L2 regularisation**: a multiple of the sum of the squared
  multipliers added to the loss, which pulls them towards zero.
- **At most 200 epochs** — an epoch is one pass over the training data, here
  one step, since every step sees the whole graph. The optimiser is Adam, a
  rule that turns gradients into changes of the multipliers (Kingma & Ba,
  2015, p. 1), with step size 0.01, and **early stopping** ends training once
  the validation loss has not fallen for 10 consecutive epochs (Kipf &
  Welling, 2017, Section 5.2).
- **Starting values** are drawn with the rule of Glorot and Bengio (2010),
  named after its first author, Xavier Glorot, and the input feature vectors
  are row-normalised (Kipf & Welling, 2017, Section 5.2). The section
  [Starting values](#starting-values-glorot-initialisation-and-the-relu-caveat)
  below returns to the rule.
- **The chosen settings** were dropout 0.5, penalty factor $5 \cdot 10^{-4}$
  and 16 hidden values for the three citation networks
  ([Section 6.1](../references/kipf_2017_graph_convolutional_networks.pdf#page=7)).

**Do not confuse a parameter with a setting chosen before training.** $W^{(0)}$
and $W^{(1)}$ are parameters, learned by gradient descent. The dropout rate,
the penalty factor and the 16 hidden values are settings, chosen on the 500
validation nodes (Kipf & Welling, 2017, Section 5.2).

**Do not confuse the validation nodes with the test nodes.** The 500
validation nodes chose the settings and decided when training stopped; the
1,000 test nodes only scored the finished model (Kipf & Welling, 2017,
Section 5.2).

**Do not confuse the paper's row-normalised features with this project's row
form.** Kipf and Welling (2017, Section 5.2) rescale each node's input
feature row; they do not give the formula. This project's `"row"` option rescales the rows of the graph
matrix (assumption A-21), not the readings.

## What the paper claims, and on what evidence

**Accuracy against other methods**
(Kipf & Welling, 2017, [Table 2](../references/kipf_2017_graph_convolutional_networks.pdf#page=7)).
On each of the four datasets the GCN row has the highest accuracy: 70.3 %
on Citeseer, 81.5 % on Cora, 79.0 % on Pubmed and 66.0 % on NELL, a graph
built from a knowledge graph, a set of entities joined by labelled relations
(Section 5.1). The best variant of Planetoid, the method of Yang et al. (2016,
as cited in Kipf & Welling, 2017, Section 5.3), reaches 75.7 % on Cora. The GCN figures are means of
100 runs with random starting multipliers on the fixed data splits of Yang et
al. All baseline accuracies except those of ICA, an iterative classification
method the authors re-ran, are taken from the Planetoid paper; Planetoid's
training times, in brackets in Table 2, were measured by Kipf and Welling with
its authors' code (Section 6.1). On 10 random splits of the same size the paper
also reports a mean and its standard error, the uncertainty of a mean over 10
runs: for example 80.1 ± 0.5 % on Cora.

**The renormalisation trick against the alternatives**
(Kipf & Welling, 2017, [Table 3](../references/kipf_2017_graph_convolutional_networks.pdf#page=7)).
The same two-layer model is run with each propagation rule of Section 2, plus
two reference variants: the neighbour term $D^{-1/2}AD^{-1/2}X\Theta$ alone,
and a **multi-layer perceptron**, ordinary layers that ignore the graph (the
row $X\Theta$). The renormalised rule has the highest mean on all three
citation networks. On Cora, for example: renormalisation trick 81.5 %,
Chebyshev with $K = 2$ 81.2 %, Eq. 6 80.0 %, the single-parameter Eq. 7
79.2 %, and the multi-layer perceptron 55.1 %. Section 7.1 concludes that the
renormalised model is both cheaper and more accurate than Eq. 6 or the
Chebyshev filters.

**What the evidence does not show.** Table 3 gives means of 100 runs without
their spread (Kipf & Welling, 2017, Section 6.2), so whether a gap such as 81.5 against 81.2 exceeds the
run-to-run variation cannot be read from it. And the instability that motivated
the trick is argued, not measured: Table 3 records accuracy only.

**Limitations the paper names**
(Kipf & Welling, 2017, [Section 7.2](../references/kipf_2017_graph_convolutional_networks.pdf#page=8)):
memory that grows with the dataset under full-batch training; no direct
support for edge features, and only **undirected** graphs, in which every
connection works both ways, weighted or not; and two built-in assumptions,
that a $K$-layer model depends only on nodes within $K$ edges, and that a
node's own value and its neighbours' count equally. For the last, they suggest
a learnable weight $\lambda$ on the self-connections, $\tilde A = A + \lambda I_N$
(Eq. 11); this $\lambda$ is a weight, not an eigenvalue.

## Depth (Appendix B)

Kipf and Welling (2017, [Appendix B](../references/kipf_2017_graph_convolutional_networks.pdf#page=14))
train models with 1 to 10 layers on Cora, Citeseer and Pubmed. They use 5-fold
cross-validation, in which the labelled nodes are split into five parts and
each part is held out once for testing while the model trains on the other
four, with all labels available (unlike the main experiments), 400 epochs
without early stopping, and 16 hidden values per layer. They compare the plain
rule of Eq. 2 with a variant that has **residual connections** (Eq. 14): each
layer's input $H^{(l)}$ is added to its output, so information can pass a layer
unchanged.

Two or three layers scored best on these datasets (Kipf & Welling, 2017,
Appendix B, Fig. 5 and the text under it). Past seven layers, models without residual connections became hard to
train; the authors connect this to each extra layer widening the neighbourhood
every node draws on, and note that more layers also mean more parameters to
overfit with.

Wen et al. (2026, [Table 4](../references/wen_2026_estgnn_dynamic_shap.pdf#page=7))
use three graph layers, inside the range Appendix B found best. Appendix B measured citation graphs and
node classes, not a 14-sensor graph and remaining life, so it is context for
that choice, not evidence for it. How far three layers reach on the sensor
graph is drawn on the [GNN page](../learn/02_graphs.md#how-information-travels-through-layers).

## Starting values: Glorot initialisation, and the ReLU caveat

Training has to start from some multipliers, and they are drawn at random.
How widely they are spread matters: values passing through many layers are
multiplied again and again, so a spread slightly too wide or too narrow
compounds (Glorot & Bengio, 2010, Section 4.2.1, p. 253; He et al., 2015,
Section 2.2, p. 1030). Kipf and Welling (2017) use Glorot initialisation
(Section 5.2, and again for the untrained model of Eq. 13 in Appendix A.1). PyTorch, the library this
project uses for its networks, calls it `xavier_uniform_`.

### What Glorot and Bengio derived

Glorot and Bengio (2010, [Section 4.2.1](../references/glorot_2010_understanding_deep_feedforward.pdf#page=4),
pp. 252–253) ask what spread of starting multipliers keeps the
**variance**, the average squared distance from the mean, of the values
steady from layer to layer. The derivation rests on stated premises
([Eq. 4 and the sentences before it, p. 253](../references/glorot_2010_understanding_deep_feedforward.pdf#page=5)):
an activation function symmetric about zero with slope 1 there, a network that
starts out in that nearly linear region, multipliers drawn independently, and
inputs of equal variance. The activations the paper studies are the sigmoid,
the hyperbolic tangent and the softsign
([Section 2.3, p. 251](../references/glorot_2010_understanding_deep_feedforward.pdf#page=3)),
not ReLU.

Why the number of inputs matters: before its activation, each value a layer
computes is a sum of $n_i$ products, one per input, of a multiplier and an input
value. When multipliers and inputs are independent and average zero, one
product has variance $\mathrm{Var}[w]\,\mathrm{Var}[x]$, and the variance of a
sum of independent terms is the sum of their variances, so the whole sum has
variance $n_i\,\mathrm{Var}[w]\,\mathrm{Var}[x]$; Glorot and Bengio (2010,
Eq. 5, p. 253) chain this over the layers. An activation with slope 1 near zero passes that variance on
unchanged. So the forward values keep their variance if
$n_i \mathrm{Var}[W^i] = 1$ (Eq. 10). The same reasoning backwards, where each
gradient is a sum over the $n_{i+1}$ values the layer feeds, keeps the
gradients steady if $n_{i+1} \mathrm{Var}[W^i] = 1$ (Eq. 11). Both can hold
only when the two counts are equal, so the paper takes a compromise, Eq. 12:

$$\mathrm{Var}[W^i] = \frac{2}{n_i + n_{i+1}}$$

A draw spread evenly over $[-a, a]$ has variance $a^2/3$, so Eq. 12 is met by
drawing every multiplier evenly from

$$W \sim U\!\left[-\frac{\sqrt 6}{\sqrt{n_j + n_{j+1}}},\; \frac{\sqrt 6}{\sqrt{n_j + n_{j+1}}}\right]$$

which Glorot and Bengio (2010, Eq. 16, p. 253) call the normalized
initialization. With 32 values
in and 64 out, $a = \sqrt{6/96} = 0.25$ and the variance is
$0.25^2/3 = 0.020833 = 2/96$. Sums of 32 inputs of variance 1 then have
variance $32 \times 2/96 = 0.666667$, not 1: the compromise of Eq. 12 at work.

### Why ReLU changes the arithmetic

He et al. (2015) redo the derivation for ReLU, forwards and backwards; the
forward case is followed here. Of Glorot and Bengio's
linear premise they write, "This assumption is invalid for ReLU and PReLU"
([Section 2.2, p. 1029](../references/he_2015_delving_deep_rectifiers.pdf#page=4)),
PReLU being a ReLU whose slope for negative inputs is learned (Section 2.1,
p. 1027). Their Eq. 9 keeps
a step the linear premise hides: the variance of a layer's sum is
$n_l\,\mathrm{Var}[w_l]\,E[x_l^2]$, where $E[x_l^2]$ is the average of the
squared inputs. That equals the inputs' variance only if the inputs average
zero, and ReLU outputs never do, because every negative becomes 0. If the
previous layer's sums $y_{l-1}$ are spread symmetrically around zero, ReLU
keeps the positive half and zeros the other, so the average square of what
enters the next layer is half the variance of $y_{l-1}$. The variance then
passes from layer to layer as

$$\mathrm{Var}[y_l] = \tfrac{1}{2}\, n_l\, \mathrm{Var}[w_l]\, \mathrm{Var}[y_{l-1}]$$

with $n_l$ the number of values entering layer $l$ (Eq. 10). Keeping it steady
needs $\tfrac{1}{2} n_l \mathrm{Var}[w_l] = 1$ (Eq. 12): multipliers drawn from
a normal (bell-shaped) distribution around zero with standard deviation $\sqrt{2/n_l}$, that
is variance $2/n_l$, twice the $1/n_l$ of Glorot and Bengio's forward condition
(Glorot & Bengio, 2010, Eq. 10). For the first layer, whose input has passed through no ReLU,
He et al. note that the factor $\tfrac{1}{2}$ does not apply, but that one such
factor does not matter.

Their comparison (He et al., 2015, Section 2.2, [p. 1030](../references/he_2015_delving_deep_rectifiers.pdf#page=5))
was against a version of Glorot's rule that differs from the one Glorot and
Bengio (2010, Eq. 16) print: multipliers
drawn from a normal distribution with variance $1/n_l$, the forward condition
alone; a footnote records that Glorot and Bengio average the forward and
backward cases and draw evenly. With ReLU, a 22-layer model converged, its
error falling and settling, under both rules, and theirs began reducing its
error sooner; on another model the final error rates were close (33.82 %
against 33.90 % of images misclassified); a 30-layer model stalled, its error
no longer falling, under the forward-only Glorot rule and converged under
theirs.

### The arithmetic for the paper's widths

Wen et al.'s three graph layers take 1, 32 and 64 values per sensor and give
32, 64 and 32 (Wen et al., 2026, Table 4), and each is followed by ReLU (their
Eqs. 8–10). Below,
*Factor if linear* is $n \mathrm{Var}[w]$, the multiplier on the variance from
one layer's outputs to the next under Glorot and Bengio's (2010, Eq. 4) premise; *Factor with
ReLU* is He et al.'s (2015, Eq. 10) $\tfrac{1}{2} n \mathrm{Var}[w]$, with no halving for
layer 1, whose input passes through no ReLU. Glorot's rule supplies the
variance in every row.

<!-- sample: glorot -->
| Layer | Values in | Values out | Range limit | Weight variance | Factor if linear | Factor with ReLU |
|---|---|---|---|---|---|---|
| 1 | 1 | 32 | 0.426401 | 0.060606 | 0.060606 | 0.060606 |
| 2 | 32 | 64 | 0.250000 | 0.020833 | 0.666667 | 0.333333 |
| 3 | 64 | 32 | 0.250000 | 0.020833 | 1.333333 | 0.666667 |

None of the factors is 1, even under the linear premise, because the counts in
and out differ in every layer; Eq. 12 is a compromise exactly there. Multiplied
over the three layers they give 0.053872 under the linear premise and 0.013468
with ReLU. Both derivations also treat a layer as multipliers plus a constant
that starts at zero (Glorot & Bengio, 2010, pp. 251, 253; He et al., 2015,
Eq. 7, p. 1029): in a graph
layer the fixed graph matrix mixes the sensors first, which changes the
variance by an amount neither paper accounts for.

<details class="process-example">
<summary>Example in Python</summary>

```python
widths = [1, 32, 64, 32]   # values per sensor entering and leaving each graph layer, Wen et al. Table 4
layers = []
for layer, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:]), start=1):
    limit = np.sqrt(6 / (n_in + n_out))            # Glorot and Bengio, Eq. 16
    variance = limit ** 2 / 3                       # variance of an even draw from [-limit, limit]
    assert np.isclose(variance, 2 / (n_in + n_out))  # Glorot and Bengio, Eq. 12
    halving = 1.0 if layer == 1 else 0.5            # He et al., Eq. 10: ReLU before the layer halves
    layers.append({"Layer": layer, "Values in": n_in, "Values out": n_out,
                   "Range limit": limit, "Weight variance": variance,
                   "Factor if linear": n_in * variance,
                   "Factor with ReLU": halving * n_in * variance})
glorot = pd.DataFrame(layers)
assert round(32 * 2 / 96, 6) == 0.666667
assert round(float(np.prod(glorot["Factor if linear"])), 6) == 0.053872
assert round(float(np.prod(glorot["Factor with ReLU"])), 6) == 0.013468
```

</details>

So Glorot's rule, taken into this project with Kipf and Welling's layer
(assumption A-14), does not by its own derivation keep the size of the values steady through these
three ReLU layers. Whether that affects training here has not been measured.

The key idea is: Glorot's rule sets the multipliers' variance so that a sum of
$n$ inputs keeps the variance of its inputs when the activation passes values
through unchanged; ReLU zeroes the negative half, so under the same rule the
variance falls by about half at each ReLU layer, which is why He et al. double
it.

## What this project takes from the paper

- **The layer's form.** Wen et al. (2026, [Eqs. 8–10](../references/wen_2026_estgnn_dynamic_shap.pdf#page=6))
  print their graph layers as
  ReLU of a rescaled adjacency matrix times the values times a learned table,
  the form of Kipf and Welling's (2017, Eq. 2) rule. Wen et al. describe the rescaled
  matrix only in words, as normalised and including self-connections (p. 6),
  and their reference list does not include Kipf and Welling's paper
  (pp. 15–16); tracing the layer to it is
  this project's reading. The project's `GraphConvolution` multiplies the graph
  matrix by the values and then by the learned table, in the order Wen et
  al.'s Eq. 8 writes it.
- **Self-connections and rescaling.** `normalize_adjacency` adds the
  self-connections of the trick, $\tilde A = A + I_N$, unless told not to
  (`add_self_loops=False`, which the setup above uses to rebuild the matrix
  before the trick). Its `"symmetric"` option is the trick exactly; the
  project's default is the row form, $\tilde D^{-1}\tilde A$ (assumption A-21),
  which shares the trick's eigenvalues but not its guarantee that no list
  gets longer.
- **No bias in the graph layers.** Neither Kipf and Welling (2017, Eqs. 2 and
  8) nor Wen et al. (2026, Eqs. 8–10) print one, and the project adds none
  (assumption A-12).
- **Glorot starting values** in the graph layers, as Kipf and Welling (2017,
  Section 5.2) use them (assumption A-14), with the caveat of the previous
  section.
- **Undirected graphs only.** `normalize_adjacency` refuses a table that is
  not symmetric, matching the limitation that Kipf and Welling (2017,
  Section 7.2) name.
- **Not taken:** the semi-supervised setting, the softmax output, the
  cross-entropy loss, full-batch training and the training settings of
  Section 5.2. The project predicts one number per window. It follows Wen et
  al. (2026, Table 6) for the loss, the optimiser, the batch size and the
  window length, fills in the learning-rate schedule Table 6 leaves open (assumption
  A-17), and adds held-back engines with early stopping (assumptions A-06 and
  A-18).

## References

- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training
  deep feedforward neural networks. In Y. W. Teh & M. Titterington (Eds.),
  *Proceedings of the Thirteenth International Conference on Artificial
  Intelligence and Statistics* (Vol. 9, pp. 249–256). PMLR.
  https://proceedings.mlr.press/v9/glorot10a.html. The copy is the PMLR
  proceedings paper; page numbers on this page are the proceedings page
  numbers it prints.
  [Copy](../references/glorot_2010_understanding_deep_feedforward.pdf)
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers:
  Surpassing human-level performance on ImageNet classification. In
  *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*
  (pp. 1026–1034). https://doi.org/10.1109/ICCV.2015.123. The copy is the
  Computer Vision Foundation's open-access version of the ICCV paper; page
  numbers on this page are the proceedings page numbers it prints.
  [Copy](../references/he_2015_delving_deep_rectifiers.pdf)
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic
  optimization. *3rd International Conference on Learning Representations
  (ICLR)*. https://arxiv.org/abs/1412.6980v9. The copy is arXiv version 9;
  page numbers on this page are the page numbers it prints.
  [Copy](../references/kingma_2015_adam.pdf)
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with
  graph convolutional networks. *International Conference on Learning
  Representations (ICLR)*. https://arxiv.org/abs/1609.02907v4. The copy is
  arXiv version 4, the ICLR 2017 paper; equation, section, table, figure and
  page numbers on this page refer to it.
  [Copy](../references/kipf_2017_graph_convolutional_networks.pdf)
- NASA Prognostics Center of Excellence. (2008). readme.txt, the description
  of the C-MAPSS data set distributed inside its zip file. NASA Prognostics
  Data Repository. [Copy](../references/nasa_2008_cmapss_readme.txt)
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and
  Technology*, *168*, 110875. https://doi.org/10.1016/j.ast.2025.110875
  [Copy](../references/wen_2026_estgnn_dynamic_shap.pdf)
