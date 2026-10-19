# Follow one real example

This page follows the calculations step by step using the real FD001 files.
The model example loads one named saved
network; it is an illustration of arithmetic, not a measurement of accuracy.
It does not change when you edit a new run in the playground.

## Read the files and attach answers

A **history** is one engine's rows in cycle order. A row contains its engine
number, cycle number, three operating settings and 21 sensor readings (NASA
Prognostics Center of Excellence [NASA PCoE], 2008; Saxena et al., 2008,
Table 2). Engine numbers restart in the test file: training engine 1 and test
engine 1 are different engines, because each of NASA's time series "is from a
different engine" (NASA PCoE, 2008).

In `train_FD001.txt`, engine 1 has its last recorded cycle at 192. Its first
row has cycle 1 and T24 = 641.82. T24 measures temperature leaving the
low-pressure compressor, in degrees Rankine (Saxena et al., 2008, Table 2).

| Input | Operation | Output |
|---|---|---|
| Training engine 1, cycle 1, final cycle 192 | Subtract the current cycle from the final cycle | Raw remaining life: 192 − 1 = 191 cycles |
| Raw remaining life 191; ceiling 125 | Keep the smaller number | Training label: 125 cycles |
| Test engine 1 stops at cycle 31; answer-file value 112 | At its last recorded row: 31 − 31 + 112 | Test answer: 112 cycles, separate from sensor inputs |
| The same test engine, earlier row at cycle 1 | 31 − 1 + 112 | Raw answer at the earlier row: 142 cycles |

The answer-file value is "the number of operational cycles after the last
cycle that the engine will continue to operate" (NASA PCoE, 2008). For an
earlier test row, add the cycles between that row and the last recorded
row to the answer-file value. This is label preparation, not information the
model is allowed to read. Changing an answer must not change its sensor input.

The [label chart](04_the_pipeline.md#1--cap-the-labels) shows
the entire training engine, before and after capping. The raw remaining life
may exceed the ceiling; the capped training label cannot. Keeping the smaller
number is the paper's capping rule (Wen et al., 2026, Section 3.2.3.1,
Eq. 19), which names the ceiling only as a symbol; 125 is this project's
choice (assumption A-01).

## Select, smooth, rescale and make windows

Selection names the sensor columns that will enter the model: the 14 sensors
of the paper's sensor graph (Wen et al., 2026, Table 3). It does not
discard the identifiers and answers needed to organise the working table.

For engine 1, T24 begins with 641.82, 642.15 and 642.35. With the default
smoothing weight, the first value is unchanged (assumptions A-04 and A-09).
The next output is
0.5 × 642.15 + 0.5 × 641.82 = 641.985. The third is
0.5 × 642.35 + 0.5 × 641.985 = 642.1675. Each step reads the previous
**smoothed output**, not the previous raw reading (Wen et al., 2026,
Section 3.2.3, Eq. 18). No future row is used.

Whole engines are then assigned to learning or validation (assumption A-06).
Validation means
checking progress without using those engines to adjust the model's learned
numbers. The test file is a separate final check, never a source of training
statistics (assumption A-25). Rescaling subtracts each sensor's training
average and divides by its training spread (assumption A-03). Alternative
range-based scaling is explained in
[Preprocessing](04_the_pipeline.md).

A window keeps consecutive cycles from just one engine. With a length of 30,
as in the paper (Wen et al., 2026, Section 3.4, Table 6), and a step of one
cycle (assumption A-05), engine 1 supplies windows at cycles 1–30, 2–31 and
3–32. Each window contains 14 sensor rows and 30 cycle columns. Its answer
belongs to its **last** cycle: the paper predicts the remaining life "at the
current time step" from a window that ends at that step (Wen et al., 2026,
p. 4, Eq. 7). The first window has raw life 192 − 30 = 162, capped
to 125. Engine number, cycle, operating settings and that answer are not
extra sensor rows.

The playground's Preprocessing diagram shows the steps and sizes for your
selected settings. The detailed arithmetic below uses its named saved run's
settings throughout.

## Through the saved model

A **shape** lists the size of each dimension. At entry, `[1, 14, 30]` means
one window, fourteen sensors and thirty cycles. After the temporal operation,
`[1, 14, 1]` means one computed number per sensor, the input shape Wen et al.
(2026, Table 4) print for the first graph layer. Inside a graph layer, the
last dimension counts computed numbers, not cycles. After flattening,
`[1, 448]` is one row of numbers (Table 4), with each sensor's block placed
after the previous one. The small samples below are the first three entries
in that order, not three different engines.

Each graph layer first combines the numbers of the sensors connected to a
given sensor. The graph supplies the mixing multipliers. Wen et al. (2026)
take them from their table of connections (Table 3), with each sensor also
connected to itself and the whole rescaled (Eq. 8); the rescaling formula is
this project's choice (assumption A-21). The layer's learned
multipliers then turn that mixture into a new list of numbers. Negative
results become zero, an operation called ReLU (Kipf & Welling, 2017, p. 2;
Wen et al., 2026, Eqs. 8–10). The next layer reads these
new lists; it does not read the raw file again.

{{figure:worked-example}}

The live trace reports both the hand-separated calculation and, in its test,
checks it against the model's ordinary prediction function. Expand an
addition to inspect every product, including products that are zero.
The first expandable table follows T24, T30 and T50 through smoothing and
rescaling, using the saved seed's actual training averages and spreads. The
next list shows all sensor values leaving the temporal stage.
Each graph layer then exposes two additions: combine T24's neighbours, and
apply the learned multipliers to that mixture. Fully connected shows every
term forming its first output; Output shows every term forming the final
internal score. Each addition ends with an offset and its total. Only the two
fully connected layers, Fully connected and Output, learn an offset
(Wen et al., 2026, Eqs. 12–13). The graph layers have none: the paper prints
none in their equations (Wen et al., 2026, Eqs. 8–10), and this project
follows them (assumption A-12), so their tables show an offset of 0. None of
these tables is a ranking of sensors.

## Other temporal algorithms

The example above deliberately names its Last cycle operation (assumption
A-07). Mean instead
averages a sensor's values in the window. Window summary keeps its last value,
average and fitted change (assumption A-28). Full window keeps all the values.
LSTM and GRU read them in order and update internal lists of computed numbers
(Hochreiter & Schmidhuber, 1997, Section 4; Cho et al., 2014, Section 2.3).
The paper's model has an LSTM for each sensor (Wen et al., 2026, Section 2.2),
which this project leaves out by default (assumption A-11); the GRU is this
project's addition (assumption A-36). An LSTM or GRU
network's learned weights are different from the example's Last cycle network.
Do not interpret that saved prediction as a prediction of those alternatives.
The [temporal explanation](09_temporal.md#the-stage-left-out-by-default)
shows the memory operations and the distinction between stored memory and
the output passed to the graph.

## Training and explaining are different operations

Training takes sensor windows **and separate answers**. It calculates a
prediction, compares it with the answer, then adjusts learned multipliers to
reduce that mismatch. A batch is the group of windows used for one update;
a pass goes through all training windows. The learning rate controls the
update size: the paper trains with Adam (Wen et al., 2026, Table 6), and
Kingma and Ba (2015) write that Adam's steps are "approximately bounded by
the stepsize setting α" (p. 3), a stepsize they also call the learning rate
(p. 4). A seed selects
repeatable random choices; changing it starts a
different training repetition. A device changes which processor performs
these calculations, not what a row means.

Prediction takes only the sensor input and a trained model. Evaluation then
compares predictions with the separate answers. Explanation starts with an
already trained model and asks how its input contributed to a prediction
(Lundberg & Lee, 2017, p. 2).
Read the [SHAP worked addition](10_dynamic_shap.md) for the separate
input → replacement → prediction-difference calculation; what stands in for
a replaced sensor is this project's choice (assumption A-20). A SHAP
explanation assigns each input an effect on the model's output (Lundberg &
Lee, 2017, p. 2). Those effects describe what the network computes, not what
happens in the engine, so in this project's reading an explanation does not
establish that a sensor caused physical damage.

## References

- Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
  Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN
  encoder–decoder for statistical machine translation. In A. Moschitti, B.
  Pang, & W. Daelemans (Eds.), *Proceedings of the 2014 Conference on
  Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1724–1734).
  Association for Computational Linguistics. https://doi.org/10.3115/v1/D14-1179
  [Copy](../references/cho_2014_rnn_encoder_decoder.pdf)
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural
  Computation*, *9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
  [Copy](../references/hochreiter_1997_long_short_term_memory.pdf). The copy
  held is the authors' 32-page manuscript, whose pages are not numbered as in
  the journal, so it is cited here by section.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization.
  *3rd International Conference on Learning Representations (ICLR)*.
  https://arxiv.org/abs/1412.6980v9
  [Copy](../references/kingma_2015_adam.pdf). The copy held is arXiv version
  9, and page numbers here are its printed pages.
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with
  graph convolutional networks. *International Conference on Learning
  Representations (ICLR)*. https://arxiv.org/abs/1609.02907v4
  [Copy](../references/kipf_2017_graph_convolutional_networks.pdf). Page
  numbers on this page are those of arXiv version 4, the copy held.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. In I. Guyon, U. von Luxburg, S. Bengio, H. Wallach,
  R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), *Advances in Neural
  Information Processing Systems 30 (NIPS 2017)* (pp. 4765–4774).
  https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
  [Copy](../references/lundberg_2017_unified_shap.pdf). Page numbers on this
  page are those printed on the copy held, which numbers its pages 1 to 10.
- NASA Prognostics Center of Excellence. (2008). readme.txt, the description of
  the C-MAPSS data set distributed inside its zip file. NASA Prognostics Data
  Repository. [Copy](../references/nasa_2008_cmapss_readme.txt)
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation
  modeling for aircraft engine run-to-failure simulation. In *2008
  International Conference on Prognostics and Health Management* (pp. 1–9).
  IEEE. https://doi.org/10.1109/PHM.2008.4711414
  [Copy](../references/saxena_2008_damage_propagation_modeling.pdf)
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and
  Technology*, *168*, 110875. https://doi.org/10.1016/j.ast.2025.110875
  [Copy](../references/wen_2026_estgnn_dynamic_shap.pdf)
