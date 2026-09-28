# XAI

## Why a number is not enough

Suppose the model says an engine has 40 cycles left. A maintenance engineer has
to decide whether to act on that, and will first ask *why*: what in the readings
makes this engine look like a 40-cycle engine?

Without an answer, the engineer cannot tell a model that has learned genuine
wear from one that has latched onto an artefact of how the data was collected,
and does not know what to inspect.

**Explainable artificial intelligence**, or **XAI**, is the family of methods
that answer the "why" question, typically by saying which inputs mattered most
for a particular prediction.


## Choose the question you want to answer

Both methods start with a trained model and one sensor window. Neither trains
the model again or identifies a physical cause of engine damage.

| Question | Method | What you read |
|---|---|---|
| Which sensors moved this prediction away from a typical-input prediction? | [Dynamic SHAP](10_dynamic_shap.md) | One contribution per sensor, in remaining-life cycles, repeated along the engine's life |
| Through which sensor connections did the model's score form? | [GNN-LRP](11_gnn_lrp.md) | Contributions assigned to routes through the calculation layers, in internal-score units |

SHAP replaces selected sensor inputs with training-derived reference readings
and runs the model on those combinations. GNN-LRP instead reads the model's
internal calculations and divides its score among the routes that produced it.
Their output units differ: a share of an internal score is not a number of
remaining cycles, so do not compare their contribution sizes directly.

## Read an explanation in context

First identify the saved run, engine and window being explained. Then inspect
the prediction and what it is being compared with. On a SHAP chart, that
comparison is the prediction for the reference input. On a route chart, the
categories add to the model's score before its conversion to cycles.

A contribution can be positive or negative, and contributions can cancel.
A large contribution describes this model's calculation; it does not prove
that the corresponding engine component is damaged. To see where that
calculation starts, [follow one real example](08_worked_examples.md).

