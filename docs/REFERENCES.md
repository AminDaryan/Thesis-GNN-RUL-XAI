# References

Every source cited in the code, the notebook, or the assumptions log, with a
short note on what it is used for. Citations follow APA style.

---

## The paper being re-implemented

**Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026).** Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
*168*, 110875. https://doi.org/10.1016/j.ast.2025.110875

> The method this project re-implements. Received 5 July 2025, accepted
> 1 September 2025. The word "neutral" in the title is a typographical error in
> the published version; the paper is about a graph *neural* network. The
> authors publish no source code, and the data availability statement reads
> "Data will be made available on request".
>
> Used for: the 14-sensor selection, the hand-designed adjacency matrix
> (their Table 3), the network layer sizes (their Table 4), and the training
> hyper-parameters (their Table 6).

---

## The dataset

**Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).** Damage propagation
modeling for aircraft engine run-to-failure simulation. *2008 International
Conference on Prognostics and Health Management*, Denver, CO, pp. 1-9. IEEE.
https://doi.org/10.1109/PHM.2008.4711414

> The paper that introduced the C-MAPSS benchmark dataset and explains how the
> simulated engine degradation was produced. This is the correct citation for
> the data itself.

**NASA Prognostics Center of Excellence Data Set Repository.**
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

> The official distribution point for the C-MAPSS files.

---

## The label design

**Heimes, F. O. (2008).** Recurrent neural networks for remaining useful life
estimation. *2008 International Conference on Prognostics and Health
Management*, Denver, CO, pp. 1-6. IEEE.
https://doi.org/10.1109/PHM.2008.4711422

> Introduced the piecewise-linear Remaining Useful Life target: hold the label
> at a ceiling while the engine is healthy, then let it fall linearly. Reported
> that a ceiling between roughly 120 and 130 cycles works well. This is the
> origin of the value 125 used almost universally since.

---

## Graph neural networks

**Kipf, T. N., & Welling, M. (2017).** Semi-supervised classification with
graph convolutional networks. *International Conference on Learning
Representations (ICLR)*. https://arxiv.org/abs/1609.02907

> The Graph Convolutional Network layer used by Wen et al. and by this project.
> The source of the symmetric normalisation formula
> `D^(-1/2) (A + I) D^(-1/2)` implemented in `graph/adjacency.py`.

**Corso, G., Stark, H., Jegelka, S., Jaakkola, T., & Barzilay, R. (2024).**
Graph neural networks. *Nature Reviews Methods Primers*, *4*, 17.
https://doi.org/10.1038/s43586-024-00294-7

> A clear, recent introduction to message passing on graphs. Good background
> reading for the thesis chapter that introduces graph neural networks.

---

**Hochreiter, S., & Schmidhuber, J. (1997).** Long short-term memory.
*Neural Computation*, *9*(8), 1735-1780.
https://doi.org/10.1162/neco.1997.9.8.1735

> The Long Short-Term Memory network. Wen et al. place one of these before each
> sensor node. This project omits that stage (assumption A-11), but the
> reference is needed when the thesis describes their architecture.

---

## Explainability

**Shapley, L. S. (1953).** A value for n-person games. In H. W. Kuhn &
A. W. Tucker (Eds.), *Contributions to the Theory of Games, Volume II*
(pp. 307-317). Princeton University Press.
https://doi.org/10.1515/9781400881970-018

> The origin of the Shapley value in cooperative game theory. *Cooperative game
> theory* is the mathematical study of how a group that achieves something
> together should divide the credit for it. The Shapley value is the unique way
> of dividing that credit which satisfies a small set of fairness conditions.

**Lundberg, S. M., & Lee, S.-I. (2017).** A unified approach to interpreting
model predictions. *Advances in Neural Information Processing Systems 30
(NeurIPS)*, 4765-4774. https://arxiv.org/abs/1705.07874

> Introduced SHAP (SHapley Additive exPlanations), which applies the Shapley
> value to machine learning. The "players" become the input features and the
> "payout" becomes the model's prediction, so each feature receives a number
> saying how much it pushed the prediction up or down. This is the method Wen
> et al. apply after training, and the starting point for the explainability
> half of this thesis.

---

## Research practice

**Kapoor, S., & Narayanan, A. (2023).** Leakage and the reproducibility crisis
in machine-learning-based science. *Patterns*, *4*(9), 100804.
https://doi.org/10.1016/j.patter.2023.100804

> A survey of how data leakage produces published results that cannot be
> reproduced. The justification for two rules enforced throughout this
> pipeline: all statistics are learned from training data only, and the
> validation split is made by engine rather than by window.

---

## Comparable work on the same dataset

**Jin, Y., Meng, Z., Hu, Q., & Chen, D. (2025).** Heterogeneous dynamic-aware
GNN for RUL prediction of aeroengine. *IEEE Transactions on Instrumentation and
Measurement*, *74*, 3538014. https://doi.org/10.1109/TIM.2025.3569885

> A graph-based method on the same dataset that documents its preprocessing
> explicitly: k-means clustering of the operating conditions, per-condition
> min-max scaling to (0, 1), and a label ceiling of 125. This is the source of
> the substituted preprocessing described in assumption A-03, and the
> supporting evidence for A-01.

**Li, X., Ding, Q., & Sun, J.-Q. (2018).** Remaining useful life estimation in
prognostics using deep convolution neural networks. *Reliability Engineering &
System Safety*, *172*, 1-11. https://doi.org/10.1016/j.ress.2017.11.021

> A widely cited convolutional baseline on C-MAPSS, reported in the comparison
> table of Wen et al. Recommended as the first baseline to reproduce, because
> a pipeline that cannot reach its published error is a pipeline with a
> problem.

---

## Software

**Harris, C. R., et al. (2020).** Array programming with NumPy. *Nature*,
*585*, 357-362. https://doi.org/10.1038/s41586-020-2649-2

**The pandas development team (2024).** pandas-dev/pandas: Pandas. Zenodo.
https://doi.org/10.5281/zenodo.3509134

**Pedregosa, F., et al. (2011).** Scikit-learn: Machine learning in Python.
*Journal of Machine Learning Research*, *12*, 2825-2830.
https://www.jmlr.org/papers/v12/pedregosa11a.html

> Cite these in the thesis if a software section is required. The k-means
> implementation used for operating-condition detection comes from scikit-learn.
