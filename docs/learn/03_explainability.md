# Explainability: which sensor drove this prediction?

## Why a number is not enough

Suppose the model says an engine has 40 cycles left. A maintenance engineer has
to decide whether to act on that. The first question they will ask is *why* —
what in the readings makes this engine look like a 40-cycle engine?

Without an answer, the number is unusable for a safety decision. You cannot
distinguish a model that has learned genuine degradation from one that has
latched onto an artefact of how the data was collected, and you cannot tell an
engineer what to inspect.

**Explainable artificial intelligence**, or **XAI**, is the family of methods
that answer the "why" question — typically by saying which inputs mattered most
for a particular prediction. Combining a graph-based model with an explanation
method is the subject of the thesis this repository supports.

---

## Dividing credit fairly

The explanation method used here comes from an unexpected place:
**cooperative game theory**, the branch of mathematics that studies how a group
of players who achieve something together should divide the credit for it.

The problem is genuinely hard. Imagine three people building something. You
cannot just measure what each contributes alone, because contributions
interact: two of them may be useless separately and excellent together. You
cannot measure what is lost when each leaves, either, because that
double-counts overlapping contributions.

In 1953 Lloyd Shapley proposed the answer now called the **Shapley value**. It
asks: for each player, **how much does the result change when that player
joins, averaged over every possible order in which the group could have been
assembled?**

{{figure:shapley}}

With three players there are six orders. In some of them B joins an empty
group; in others it joins after A, or after C, or after both. B's contribution
is different in each case, and the Shapley value is the average across all of
them.

What makes this the *right* answer rather than one plausible answer is that it
is the only way of dividing credit that satisfies a small set of natural
fairness conditions simultaneously — among them that the shares add up to the
total, and that two players who always contribute identically get identical
shares.

---

## SHAP: the same idea, applied to a model

**SHAP** — *SHapley Additive exPlanations* — applies the Shapley value to
machine learning. The translation is direct:

| Game theory | Machine learning |
|---|---|
| Players | The model's input features |
| The result the group achieves | The model's prediction |
| A player's fair share | How much that feature pushed the prediction up or down |

In this project **the players are the 14 sensors**, and each one receives a
number in flight cycles saying how far it moved the prediction away from the
average.

The "adds up to the total" property becomes concrete and checkable here: the 14
attributions plus a baseline must reconstruct the prediction **exactly**. This
project has a test for precisely that,
`test_efficiency_holds_to_machine_precision`. It matters more than it sounds.
That property is what makes the numbers Shapley values rather than merely a
plausible-looking ranking, and it is the single check that would catch an
attribution method that had quietly stopped being one.

---

## Exact, not sampled

Here is the practical problem. With 14 sensors there are 14 factorial —
87 178 291 200 — orders to average over. That is not a computation anyone runs.

The standard escape is to sample: try a few thousand random orders and average
those. Most SHAP implementations do this, and the answer is an estimate with
error bars.

This project does not need to. What actually matters is not the order but which
sensors are **present**, and there are only $2^{14} = 16\,384$ distinct groups
of 14 sensors. Every one can be evaluated. The formula weights each group by
how many orders produce it — which is where the factorials in the paper's
Equation 15 come from — and the result is the exact Shapley value, with no
sampling error at all.

The cost is 16 384 runs of the network per window explained, which for a
network of this size is seconds. The code refuses outright above 20 sensors,
where the approach stops being reasonable, rather than silently switching to an
approximation.

---

## What "removing a sensor" has to mean

There is a subtlety the definition forces on us, and it is a good example of
the kind of gap this project documents.

The Shapley value needs to know what the model predicts from a *group* of
sensors — say, 13 of the 14. But the network takes exactly 14 inputs. It cannot
be handed 13. Something has to occupy the fourteenth slot.

The paper's Equation 15 requires this quantity and never says what it puts
there. **This project replaces an absent sensor with its average reading across
the training data** — the intuition being "this sensor is at its typical value,
so it tells us nothing in particular". That is assumption A-20.

It is not the only defensible option; substituting zero is another common
choice. The two have been compared on the same trained model and the same
windows, and the result is a small lesson in itself.

Under the project's **current** scaling, which centres every sensor on its
training mean, "substitute zero" and "substitute the mean" turn out to be very
nearly the same instruction written two ways — the two rankings agree with a
correlation of 0.996. Under the **previous** min-max scaling, where zero is the
smallest value a sensor ever took rather than its typical one, the same
comparison said the opposite.

So the choice of baseline matters exactly as much as the scaling makes it
matter, and the original argument for the mean window — that zero would be
actively misleading — turned out to be true only of the scaling in force when
it was written. The entry now rests on a different reason that survives the
change: the training mean keeps the explanation honest in the same sense the
scaling is, because nothing from the test engines enters it.

The general point is worth carrying away. An explanation method has its own
assumptions, and an attribution is only as meaningful as the baseline it is
measured against.

---

## Dynamic: the same calculation at every cycle

A single explanation tells you which sensors drove one prediction. **Dynamic
SHAP** repeats the calculation at every cycle of an engine's life, producing a
trace of how the reasoning changes as the engine degrades.

This is the part of the method with the most obvious practical value. A sensor
that dominates the explanation only in the last twenty cycles is describing a
late-stage failure mode; one that matters throughout is describing gradual
wear. That distinction is what a maintenance engineer can actually act on.

---

## A finding worth meeting

Recall from page 1 that `P15` takes only two distinct values across the entire
training set, yet the paper includes it in the graph.

What does SHAP say about a sensor that carries almost no information? The
answer is recorded as finding F-05, and it has a history worth knowing: **the
first version of that finding was wrong.** It was written from 14 sampled
windows out of 240, and got both the size of the effect and the mechanism
behind it wrong. It was corrected after being recomputed over the full set.

The correction is still in the log, marked as one. That is the house style, and
it is also a warning about attribution methods in general: they produce
confident-looking numbers, and a confident-looking number computed from a
non-representative sample looks exactly like a confident-looking number
computed properly.

---

## What you now know

- **XAI** answers "why did the model say that", which a safety decision needs.
- A **Shapley value** divides credit by averaging a player's contribution over
  every joining order, and it is the unique method satisfying a set of natural
  fairness conditions.
- **SHAP** makes the sensors the players and the prediction the result; the
  attributions plus a baseline must reconstruct the prediction exactly, and a
  test enforces that.
- This implementation is **exact** — all 16 384 groups — rather than sampled.
- Removing a sensor means **substituting its training average**, which is a
  choice the paper leaves open (A-20) — and how much that choice matters
  depends on the input scaling.
- **Dynamic SHAP** repeats the calculation every cycle, showing how the
  reasoning changes as the engine wears.

Next: the eight preprocessing steps, and the three that decide whether any of
this is valid.
