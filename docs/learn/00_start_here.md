# Start here

A jet engine wears out gradually. If an airline can estimate how many more
flights an engine will survive, it can service the engine before it fails
rather than after. This project estimates that number from sensor readings,
using a neural network that treats the engine's sensors as **connected to each
other** rather than as a list of unrelated numbers, and then explains each
estimate by saying which sensors drove it.

The method, called ESTGNN, comes from one paper published in 2026, with no
source code and with its data offered only on request. This thesis rebuilds it
from the paper's text alone, runs it on NASA's public engine data, and measures
how far the results land from the paper's own. That distance is the result: a
measure of how much a published description leaves unsaid.

You need no background in machine learning, aircraft engines or graph theory.
Every term is spelled out the first time it appears.

---

## From NASA's files to an explained prediction

The work runs in five stages. NASA's text files are prepared into examples a
network can learn from; the network is built; it is trained on those examples;
it is scored on engines it never saw during training; and each of its
predictions is explained by how far every sensor moved it.

Follow the pages in the order the information moves:

1. [Data](01_the_problem.md): what the files, rows and sensor columns mean.
2. [Preprocessing](04_the_pipeline.md): turn readings into windows, with separate answers.
3. [Temporal algorithm](09_temporal.md): choose what each sensor keeps from its window.
4. [GNN](02_graphs.md): combine information from connected sensors to make a prediction.
5. [XAI](03_explainability.md): explain that prediction using
   [Dynamic SHAP](10_dynamic_shap.md) or [GNN-LRP](11_gnn_lrp.md).

[Follow one real example](08_worked_examples.md) traces the same window through
these calculations with values loaded from the dataset and a saved network.

---

## Paper, choice or measurement

Every claim on these pages is what the paper says, what this project chose
where the paper is silent, or what this code measured, and
[Paper, choice or measurement](07_evidence.md) shows how the three are kept
apart. A score this code measured is never one training run's; why is
explained under [One seed is not a result](06_reading_results.md#one-seed-is-not-a-result).
