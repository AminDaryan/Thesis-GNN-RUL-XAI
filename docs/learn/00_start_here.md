# Start here

Every jet engine has a limited life: its parts wear a little with every flight
(Saxena et al., 2008, pp. 4–5). An engine that fails in service is dangerous
and expensive: the failure can bring large economic losses and, at worst, loss
of life (Li & Dong, 2026, p. 2). The way to prevent that is to service the
engine in time (Wen et al., 2026, p. 9). Engines do not all wear at the same
rate (Saxena et al., 2008, p. 5), so servicing them on a fixed schedule is safe
only if the schedule is set for the engines that wear fastest. Most engines are
then taken off the wing with many flights still left in them, and that usable
life is wasted. This argument about fixed schedules is this project's own
reasoning, not a published finding; Wen et al. (2026) make the related point
that estimating far too little remaining life "results in unnecessary
maintenance costs" (p. 9).

It would be better to estimate, for each engine, how many more flights it will
survive, and service it just before that point. Saxena et al. (2008) say that
reliable estimates of remaining life promise cost savings, by avoiding
unscheduled maintenance and by using equipment for longer, as well as better
operational safety (p. 2), and that, because the key aim is to avoid failures,
an early estimate is preferred to a late one (p. 7). This project estimates
that remaining life from sensor readings, using a neural network that treats
the engine's sensors as **connected to each other** rather than as a list of
unrelated numbers, and then explains each estimate by saying which sensors
drove it.

The method, called ESTGNN, comes from one paper, Wen et al. (2026, p. 1),
published with no source code and with its data offered only on request
(p. 15). This thesis rebuilds it from the paper's text alone, runs it on NASA's
public engine data, and measures how far the results land from the paper's
own. That distance is the result: a measure of how much a published
description leaves unsaid.

NASA's data come from a computer simulation of engines whose wear grows until
they fail (Saxena et al., 2008, p. 1). They count an engine's life in operating
cycles (NASA Prognostics Center of Excellence [NASA PCoE], 2008), and this
project reads one cycle as one flight; the [Data](01_the_problem.md) page gives
the sources behind that reading.

You need no background in machine learning, aircraft engines or graph theory.
Every term is spelled out the first time it appears.

---

## From NASA's files to an explained prediction

The work runs in five stages. NASA's text files (NASA PCoE, 2008) are prepared
into examples a network can learn from; the network is built; it is trained on those examples;
it is scored on engines it never saw during training; and each of its
predictions is explained by how far every sensor moved it.

Follow the pages in the order the information moves:

1. [Data](01_the_problem.md): what the files, rows and sensor columns mean.
2. [Preprocessing](04_the_pipeline.md): turn readings into windows, with separate answers.
3. [Temporal algorithm](09_temporal.md): choose what each sensor keeps from its window.
4. [GNN](02_graphs.md): combine information from connected sensors to make a prediction.
5. [XAI](03_explainability.md): explain that prediction using
   [Dynamic SHAP](10_dynamic_shap.md) (Wen et al., 2026, Section 2.4) or
   [GNN-LRP](11_gnn_lrp.md) (Schnake et al., 2022, Section 3.2).

[Follow one real example](08_worked_examples.md) traces the same window through
these calculations with values loaded from the dataset and a saved network.

---

## Paper, choice or measurement

Every claim on these pages is what the paper or another published work says,
what this project chose where the paper is silent, or what this code measured,
and [Paper, choice or measurement](07_evidence.md) shows how the three are kept
apart. A claim taken from a published work names its source the way a journal
paper does, by authors, year and page, for example (Saxena et al., 2008, p. 2),
and the list of references at the end of the page gives the full entry. A
score this code measured is never one training run's; why is explained under
[One seed is not a result](06_reading_results.md#one-seed-is-not-a-result).

## References

- Li, T., & Dong, J. (2026). Multi-level attention graph feature fusion smooth
  prognostics approach for aircraft engines remaining useful life prediction.
  *Journal of Control and Decision*, *13*(2), 399–415 (published online 2024).
  https://doi.org/10.1080/23307706.2024.2398536
  [Copy](../references/li_2026_multilevel_attention_graph.pdf). Page numbers
  on this page are those of the PDF file, which opens with a cover sheet, so
  its PDF page 2 is the journal's page 399.
- NASA Prognostics Center of Excellence. (2008). readme.txt, the description of
  the C-MAPSS data set distributed inside its zip file. NASA Prognostics Data
  Repository. [Copy](../references/nasa_2008_cmapss_readme.txt)
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation
  modeling for aircraft engine run-to-failure simulation. In *2008
  International Conference on Prognostics and Health Management* (pp. 1–9).
  IEEE. https://doi.org/10.1109/PHM.2008.4711414
  [Copy](../references/saxena_2008_damage_propagation_modeling.pdf)
- Schnake, T., Eberle, O., Lederer, J., Nakajima, S., Schütt, K. T., Müller,
  K.-R., & Montavon, G. (2022). Higher-order explanations of graph neural
  networks via relevant walks. *IEEE Transactions on Pattern Analysis and
  Machine Intelligence*, *44*(11), 7581–7596.
  https://doi.org/10.1109/TPAMI.2021.3115452
  [Copy](../references/schnake_2022_gnn_lrp_relevant_walks.pdf). Section
  numbers on this page are those of arXiv version 3, the copy held.
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and
  Technology*, *168*, 110875. https://doi.org/10.1016/j.ast.2025.110875
  [Copy](../references/wen_2026_estgnn_dynamic_shap.pdf)
