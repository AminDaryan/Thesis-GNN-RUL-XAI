# GNN-LRP

## Walks: the routes a prediction took through the graph

Shapley values say *which sensors'* readings mattered. They cannot say *how*
the readings reached the prediction: whether a sensor mattered at its own place
in the graph, or because the graph layers mixed its reading into its neighbours
and it was used there. The paper says its graph matters, so that second
question matters too.

**What a graph layer does.** Each of the network's three graph layers gives
every sensor a new number made from its neighbours' numbers: a weighted average
over the sensors it is joined to, itself included, multiplied by a learned
weight, with anything negative set to zero. After three layers, a sensor's
number carries readings from up to three steps away.

**What a walk is.** The route one piece of a reading takes, one sensor per
layer. `T50 → T30 → T30 → Ps30` is T50's reading, carried into T30 by the first
layer, kept at T30 by the second, and carried into Ps30 by the third.

**Why a prediction splits into walks.** Just before the network turns its output
into cycles of remaining life, that output is one number, the **score**. For
one window, every step between the readings and the score is a multiplication,
an addition, or a switch that is either on or off. Multiplied out, the score
becomes a sum of pieces, each travelling one walk, plus a part that belongs to
no sensor, which the network's last layers add. This split is a method called
GNN-LRP (Schnake et al., 2022), used here with its simplest setting (assumption
A-29).

A small real case: test engine 5's last readings of T24, T30 and T50 are
−0.3734, 0.1146 and 0.8176 (each counted in standard deviations from its
training average). In a toy network with one graph layer, the walk T50 → T30
carries 0.8176 × 1/3 = 0.2725 of the score: T50's reading, times T30's weight
for T50 in the graph, times what one unit arriving at T30 is worth. Adding the
five walks the graph allows and the network's constant of 0.5 gives exactly the
score, 0.2202 (the code's worked example).

**What it does not show.**

- The shares are parts of the score, **not cycles**: the last step bends the
  score into cycles, so its parts cannot be turned into cycles that still add
  up. Their ranking may be set beside a Shapley ranking; their sizes may not.
- What flowed along a connection is **not what removing it would do**. On the
  default network's five stored seeds, the order in which the two rank the
  connections is unrelated (assumption A-30).
- It describes one trained network, not the engine, and it cannot say whether
  the paper's graph is better than another; only networks trained on other
  graphs can (assumption A-27).
- Three groups of sensors are joined to exactly the same neighbours in the
  paper's graph, so the network cannot tell them apart; where a walk ends among
  them is set by the network's random starting weights, and the playground adds
  such walks together (finding F-18).
