# Dynamic SHAP

## Dividing credit fairly

The paper's explanation method comes from **cooperative game theory**, the
branch of mathematics that studies how a group of players who achieve something
together should divide the credit for it.

The problem is hard. Imagine three people building something. You cannot just
measure what each contributes alone, because contributions interact: two of
them may be useless separately and excellent together. You cannot measure what
is lost when each leaves, either: each of those two would be credited with
everything they achieve together, so their joint result would be counted twice.

In 1953 Lloyd Shapley proposed the answer now called the **Shapley value**. It
asks: for each player, **how much does the result change when that player
joins, averaged over every possible order in which the group could have been
assembled?**

{{figure:shapley}}

The order matters because B adds a different amount depending on who is already
there when it joins.

The Shapley value is the only way of dividing credit that meets a small set of
fairness conditions at once, among them that the shares add up to the total,
and that two players who always contribute identically get identical shares.

---

## SHAP: the same idea, applied to a model

**SHAP**, *SHapley Additive exPlanations*, applies the Shapley value to machine
learning:

| Game theory | Machine learning |
|---|---|
| Players | The model's input features |
| The result the group achieves | The model's prediction |
| A player's fair share | How much that feature pushed the prediction up or down |

In this project **the players are the 14 sensors**. Each receives an
**attribution**: a number of flight cycles saying how far that sensor moved the
prediction away from the one the model makes when every sensor is replaced by
its average training reading.

The shares-add-up condition makes this checkable: the 14 attributions plus that
average-reading prediction must equal the prediction itself, and
`test_efficiency_holds_to_machine_precision` checks that they do to within
0.0001 cycles. Numbers that failed it could not be Shapley values, however
plausible their ranking looked.

---

## Exact, not sampled

With 14 sensors there are 14 factorial (14 × 13 × … × 1), or
87 178 291 200, orders to average over. That is not a computation anyone runs.

One way out is to sample: average over a random selection of orders, which
gives an estimate rather than the exact value.

This project does not need to. What a sensor adds depends only on which sensors
are already **present**, not on the order they arrived in, and since each
sensor is either in or out, there are only $2^{14} = 16\,384$ distinct groups
that can be drawn from 14 sensors, counting the empty group and the full one.
Every one can be evaluated. The formula weights each group by how many orders
have exactly that group present when the sensor joins. In the three-player
figure, B comes first in two of the six orders, after A alone in one, after C
alone in one, and after both in two, so B's value is (2 × what B adds on its
own + 1 × what it adds to A + 1 × what it adds to C + 2 × what it adds to A and
C) ÷ 6. Counting orders this way for 14 sensors is where the factorials in the
paper's Equation 15 come from, and the result is the exact Shapley value, with
no sampling error at all.

The cost is 16 384 runs of the network per prediction explained. Each extra
sensor doubles that, to 1 048 576 runs at 20 sensors, and the code refuses more
than 20 rather than silently switching to sampling.

---

## What "removing a sensor" has to mean

The Shapley value needs to know what the model predicts from a *group* of
sensors — say, 13 of the 14. But the network takes exactly 14 inputs. It cannot
be handed 13. Something has to occupy the fourteenth slot.

The paper's Equation 15 requires this quantity and never says what it puts
there. **This project replaces an absent sensor with its average reading across
the training data**, so the model is shown "this sensor is at its typical value"
(assumption A-20).

It is not the only defensible option; substituting zero is another common
choice, and the two have been compared by explaining the same predictions with
each.

Under the project's **current** input scaling, which subtracts each sensor's
training mean so that its typical reading becomes zero, "substitute zero" and
"substitute the mean" are very nearly the same instruction: on one network
trained on all 100 training engines, the sensor rankings they produce agree with
a rank correlation of 0.965, where 1 would mean the same order. Under the
**previous** min-max scaling, which put each sensor's smallest reading at zero
instead, the two rankings were unrelated, at −0.16.

So how much the stand-in matters depends on the scaling. The mean is kept
because it stands for a typical reading under any scaling, where zero does so
only under some, and because it is computed from the training engines alone, so
nothing from the test engines enters it. An attribution is only as meaningful
as the value it is measured against.

---

## Dynamic: the same calculation at every cycle

A single explanation tells you which sensors drove one prediction. **Dynamic
SHAP** repeats the calculation cycle after cycle along one engine's life,
producing a trace of how the attributions change as the engine wears.

That shows when in an engine's life the model relies on each sensor: one may
dominate only in the last twenty cycles, another throughout. It describes the
model, not the engine: a sensor the model leans on near failure is not thereby
shown to be where the failure is.

The playground's Explain tab draws these attributions for a stored run in the
layouts the paper uses, and sets one engine's beside the paper's own (its
Fig. 10). The separate [GNN-LRP page](11_gnn_lrp.md) follows routes through the
model instead of replacing sensor inputs.

---

## What SHAP says about the two-value sensor

`P15` takes only two distinct values across the whole training set, yet the
paper keeps it in the graph. On the network behind the 0.965 above, exact
Shapley values over every prediction along the lives of three of its training
engines rank it last of the fourteen sensors on each engine, and the paper's
Fig. 9 also puts it last. Its attribution is not always near zero, though. On engine 5, whose
reading switches between the two values 34 times, one prediction gives it an
attribution 6.8 cycles in size; on engine 25, whose reading never switches, none
is larger than 1.7 (finding F-05).

Because attributions can vary this much from one prediction to the next, a
claim about them has to be computed over every prediction. The first version of
this result used 14 of engine 5's 240 predictions, and got both the size of the
effect and its cause wrong.
