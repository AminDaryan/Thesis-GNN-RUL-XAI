# The model, layer by layer

{{figure:architecture}}

The diagram above is not a drawing of what the network is supposed to be. It is
built by constructing the real model in memory and reading its layers back, so
every shape and every parameter count on it is what the code actually has.

---

## The shape of it

Six layers, and the whole thing has **61 729 trainable parameters** — the exact
count the paper's Table 4 prints, which a test asserts on every run.

| Layer | In | Out | Parameters | Activation |
|---|---|---|---|---|
| Input | (14, 1) | (14, 1) | 0 | — |
| GCN-1 | (14, 1) | (14, 32) | 32 | ReLU |
| GCN-2 | (14, 32) | (14, 64) | 2 048 | ReLU |
| GCN-3 | (14, 64) | (14, 32) | 2 048 | ReLU |
| Flatten | (14, 32) | (448) | 0 | — |
| FC-1 | (448) | (128) | 57 472 | ReLU |
| FC-2 | (128) | (1) | 129 | Sigmoid |

Read it as a story. Fourteen sensors arrive carrying one number each. Three
rounds of message passing let each sensor's value be shaped by its neighbours,
growing from 1 number per sensor to 32, then 64, then back down to 32. The
14 × 32 grid is then flattened into one long vector of 448 numbers, and two
ordinary fully connected layers reduce that to a single output.

Two things in that table are worth noticing.

**The graph layers are tiny.** All three together hold 4 128 parameters — under
7 % of the model. The dense layer that follows holds 57 472, over 93 %. The
graph part of this graph network is a very small fraction of it, which is worth
keeping in mind when reading any claim about what the graph contributes.

**The graph layers have no bias term.** The paper's Equations 8 to 10 print
only a weight matrix, while its dense layers do print a bias, so this project
follows what is printed. That is assumption A-12, and it is one line to switch
if someone wants to measure what a bias would change.

---

## The last layer, and what it makes impossible

The final activation is a **sigmoid**, which produces a number between 0 and 1.
That number is then multiplied by the label ceiling of 125 to give an answer in
flight cycles.

This is not a detail. It means the network is **structurally incapable of
predicting more than 125 cycles**, no matter what it is trained on or how long
for. There is no weight setting that would let it.

Three consequences follow, and all three are documented choices:

- The **training targets are divided by the ceiling** so they lie in 0 to 1,
  matching what the sigmoid can output. The paper calls its output a
  "normalized RUL value" without saying by what, so the division is assumption
  A-13.
- The **test labels are capped** at the same ceiling. Leaving one at 140 would
  charge the model for an error it cannot avoid. That is A-02, it has been
  measured, and because it can only move the reported numbers favourably it
  must be stated whenever they are compared with published ones.
- A test called `test_the_prediction_can_never_exceed_the_ceiling` enforces the
  property, so it cannot be broken by accident.

---

## What is deliberately missing

The paper's method has four parts: one small recurrent network per sensor, the
hand-designed graph, three graph convolution layers, and SHAP afterwards.

**This project implements three of the four.** The per-sensor recurrent stage —
a **long short-term memory** network, or **LSTM**, a kind of network built for
sequences that keeps an internal memory of what it has already seen — is not
implemented.

The reason is that the paper never states its size, its depth, or how it turns
a 30-cycle window into the single number Table 4 requires. Any version of it
would be invention rather than reproduction, and inventing the missing half of
someone else's method and then reporting the result as a reproduction is
exactly the failure mode this project is organised to avoid. It is assumption
A-11.

This has a clean consequence for how results may be compared. The paper
publishes an ablation table containing a row for its own method **without** the
LSTM stage, which is precisely this configuration. So:

> Results from this code compare against the paper's **"ESTGNN w/o LSTM"** row.
> They are never comparable with its headline figures, and nothing in this
> repository should be described as "reproducing ESTGNN".

The gap between the two rows in the paper's own table is large, so this is not
a small omission being waved away — it is a substantial part of the method,
left out on purpose, with the comparison adjusted to match.

---

## Why the shapes are checked so aggressively

Every layer in this code validates the shape of what it is given, and refuses
rather than reshaping. The sensor graph is rejected if it has a zero anywhere
down its diagonal. A zero there means each sensor is not joined to itself, and
a sensor not joined to itself is dropped from its own new value — the mixing
step would replace T24 entirely by its neighbours. Joining each sensor to
itself is one of the two things `normalize_adjacency()` does, so a zero on the
diagonal is proof the graph never went through it.

That strictness is deliberate, and the reason is domain-specific: **a wrong
answer here still looks plausible.** A model that predicts 60 cycles when the
truth is 40 produces no error message, no obviously broken output, nothing that
looks wrong on a plot. The only defence is to make the impossible states
impossible to reach, and to check the properties that must hold rather than
just the numbers that come out.

The three checks worth knowing about:

- `test_parameter_counts_match_table_4` — every weight matrix has the shape the
  paper prints, so a result cannot come from a differently-shaped network
  wearing the paper's name.
- `test_three_layers_cannot_reach_four_hops` — message passing follows the
  sensor graph and nothing else.
- `test_efficiency_holds_to_machine_precision` — the fourteen attributions plus
  the baseline reconstruct the prediction exactly.

If any of those fails, results from this code are not trustworthy regardless of
how good they look.

---

## Training

The training settings come from the paper's Table 6: the Adam optimiser, 200
epochs, batches of 32, mean squared error as the loss, and a learning rate
starting at 1e-3 and ending at 1e-4.

The paper gives both ends of that rate and never says how it gets between them.
This project shrinks it by the same factor every epoch — exponential decay
reaching 1e-4 at the last epoch — which is assumption A-17. Two further silences
are filled here too: the starting values of the weights (A-14, Glorot uniform,
following the paper that introduced this layer type) and whether to stop early
(A-18, no — the paper states a fixed 200 epochs).

---

## What you now know

- The network is **three graph layers, a flatten and two dense layers**, with
  **61 729 parameters** matching Table 4 exactly.
- Over **93 %** of those parameters are in the first dense layer, not in the
  graph part.
- The **sigmoid output makes predictions above 125 impossible**, which forces
  the target scaling (A-13) and the test-label cap (A-02).
- The **LSTM stage is deliberately absent** (A-11), so results compare against
  the paper's "ESTGNN w/o LSTM" row and never its headline numbers.
- Shapes and properties are checked aggressively because **a wrong answer in
  this domain still looks plausible**.

Next: how to read a result, and when a difference between two of them is real.
