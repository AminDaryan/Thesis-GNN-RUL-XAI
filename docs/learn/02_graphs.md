# Graphs, and what a graph network computes

## A graph is points and connections

A **graph** is a set of points, called **nodes**, with connections between
them, called **edges**. A road map is a graph: junctions are nodes, roads are
edges.

Here, **every sensor is a node**, and an edge between two sensors means "these
two measurements are physically related". There are 14 sensors, so 14 nodes.

## The adjacency matrix is the graph written as a table

For 14 sensors the **adjacency matrix** is a 14-by-14 table: the entry in row
*i*, column *j* is 1 if sensors *i* and *j* are connected and 0 if not. It holds
the same information as a drawing of the graph, in a form a computer can
multiply.

{{figure:adjacency}}

The matrix is **symmetric**: if `T24` is a neighbour of `P30`, then `P30` is a
neighbour of `T24`. Its diagonal is zero, because a sensor is not listed as its
own neighbour. The code checks both when it is imported.

---

## The graph this project uses

Counted from this project's copy of the paper's Table 3, the graph has 41 edges
between its 14 sensors. It is drawn further down, carrying information from one
sensor to the rest.

**The graph is a physical claim, not a statistical one.** The edges were
hand-designed by the paper's authors from engine physics, not learned from
correlations in the data. That makes it a **prior**: knowledge put into the
model from outside, before it sees any measurements.

---

## What a graph neural network does: message passing

A **graph neural network** (GNN) is a neural network that operates on a graph.
The mechanism is called **message passing**, and one round of it is three
steps:

1. **Collect.** Every node looks at its neighbours and gathers their current
   values.
2. **Combine.** It mixes those values together with its own.
3. **Update.** It transforms the mixture — the same transformation everywhere
   in the graph — to produce its new value.

Every node does this at the same time. One round is one **layer**. Stack three
layers and information travels up to three edges, or hops, because in each
round a node takes in values that already hold what its neighbours collected in
the round before.

{{figure:message-passing}}

The drawing is the graph of Table 3 itself: each circle is a sensor, a larger
circle has more neighbours, and each line is an edge. **`T24` is the only
sensor whose removal splits the graph**, which is what its dashed ring marks:
the four fan and bypass sensors, `P15`, `Nf`, `NRf` and `BPR`, reach the other
nine only through `T24`, so everything `P15` sends to them passes `T24` first.

However long the network trains, three graph layers reach no further. Two
sensors four edges apart, such as `P15` and the one the drawing marks "never",
do not affect each other inside the graph layers at all; they meet only in the
layers after them, which take all fourteen sensors' results together. The next
figure counts every pair of sensors by how far apart the two are.

{{figure:reach}}

---

## One graph layer, as a formula

$$H' = \mathrm{ReLU}\left(\hat{A} H W\right)$$

`H` is the current table of node values: one row per sensor, holding that
sensor's current numbers. `Â` is the adjacency matrix after the normalisation
below, and multiplying by it is the collect-and-combine step: row *i* of `ÂH` is
a weighted sum of the rows of sensor *i* and its neighbours. `W` is a matrix of
numbers the network learns; multiplying by it is the update step, turning each
row into a new row with as many numbers as `W` has columns. It is the *same*
`W` for every node, which is why the layer works on a graph of any size. `ReLU`
replaces every negative number with zero, and the result, `H′`, is the new
table of node values that the next layer starts from.

The hat on `Â` is the normalisation this project uses, drawn in the right panel
of the adjacency matrix figure:

$$\hat{A} = D^{-1} (A + I)$$

Adding the identity matrix `I`, which has ones on its diagonal and zeros
elsewhere, gives every node an edge to itself, called a **self-loop**, so a
sensor's own value survives the mixing instead of being replaced by its
neighbours'. `D` holds each row total of `A + I` on its diagonal and zeros
elsewhere, so multiplying by $D^{-1}$ divides every row by its total. `P15` has
four neighbours, so its row of `A + I` holds five ones and becomes five entries
of 0.2: row `P15` of `ÂH` is the average of the rows of `P15`, `T24`, `Nf`,
`NRf` and `BPR`. `T24`, with seven neighbours, gets eight entries of 0.125, the
average of eight rows. Without the division `T24` would receive a sum of eight
rows and `P15` a sum of five, so a sensor's value would grow with its number of
neighbours.

> **A choice hides here.** The paper says its matrix is "normalized ...
> incorporating self-loops" and never prints the formula. The symmetric form
> $D^{-1/2}(A+I)D^{-1/2}$ fits that description as well as the row form above:
> it divides the entry joining two sensors by the square root of the product of
> their two row totals, so the `P15`–`T24` entry becomes 1/√(5 × 8) ≈ 0.158 in
> both their rows. Over five training runs on all 100 training engines, the row
> form's error on the test engines was 0.25 cycles lower, a difference five runs
> cannot tell from zero (*p* = 0.068). The default keeps a fifth of the training
> engines out of training, and no difference between the two forms has been
> shown there either (assumption A-21).

---

## Why a graph at all

It is a hypothesis, not a certainty: telling the model which sensors are
physically related should help it more than leaving it to work that out from
the data. The paper's evidence is one experiment: without the physics graph,
its model's error rises by 0.56 cycles, from 11.78 to 12.34 (its Table 11).

This project has repeated that comparison, replacing the physics graph with one
joining every sensor to every other, on its default network, which leaves out
one stage of the paper's model (assumption A-11). Under three configurations it
measured the error rising by 0.044, 2.396 and, under the current default
settings, 2.784 cycles. The physics graph was the same each time. What moved
were settings the paper does not state, among them the input scaling, the
normalisation of the graph and how many training engines were kept out of
training, so a reader cannot tell whether the paper's 0.56 would survive a
different choice of them (finding F-04).

Joining every sensor to every other removes more than the knowledge. Every row
of `A + I` becomes fourteen ones, so every row of `Â` is fourteen entries of
1/14, and every sensor receives the same average. The paper's graph repeats
some rows too: `P15`, `Nf`, `NRf` and `BPR` are each joined to `T24` and to the
other three, so all four get the row of five entries of 0.2 worked out above,
and the paper's 14 rows hold only 9 different ones. A graph built from the data
instead, joining the 41 sensor pairs whose readings move together most
strongly, has as many edges as the paper's; rebuilt from each training run's
own engines, it had 10 to 12 different rows over five runs. Under the current
default its error is only 0.096 cycles higher than the physics graph's, a
difference five runs cannot tell from zero (*p* = 0.655). So, in that
configuration, most of the 2.784 measures the averaging into one row rather
than the missing knowledge (assumption A-27).
