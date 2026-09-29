# Follow one real example

This page follows the calculations step by step using the real FD001 files.
The model example loads one named saved
network; it is an illustration of arithmetic, not a measurement of accuracy.
It does not change when you edit a new run in the playground.

## Read the files and attach answers

A **history** is one engine's rows in cycle order. A row contains its engine
number, cycle number, three operating settings and 21 sensor readings.
Engine numbers restart in the test file: training engine 1 and test engine 1
are different engines.

In `train_FD001.txt`, engine 1 has its last recorded cycle at 192. Its first
row has cycle 1 and T24 = 641.82. T24 measures temperature leaving the
low-pressure compressor, in degrees Rankine.

| Input | Operation | Output |
|---|---|---|
| Training engine 1, cycle 1, final cycle 192 | Subtract the current cycle from the final cycle | Raw remaining life: 192 − 1 = 191 cycles |
| Raw remaining life 191; ceiling 125 | Keep the smaller number | Training label: 125 cycles |
| Test engine 1 stops at cycle 31; answer-file value 112 | At its last recorded row: 31 − 31 + 112 | Test answer: 112 cycles, separate from sensor inputs |
| The same test engine, earlier row at cycle 1 | 31 − 1 + 112 | Raw answer at the earlier row: 142 cycles |

For an earlier test row, add the cycles between that row and the last recorded
row to the answer-file value. This is label preparation, not information the
model is allowed to read. Changing an answer must not change its sensor input.

The [Data chart](01_the_problem.md#what-the-model-is-asked-to-predict) shows
the entire training engine, before and after capping. The raw remaining life
may exceed the ceiling; the capped training label cannot.

## Select, smooth, rescale and make windows

Selection names the sensor columns that will enter the model. It does not
discard the identifiers and answers needed to organise the working table.

For engine 1, T24 begins with 641.82, 642.15 and 642.35. With the default
smoothing weight, the first value is unchanged. The next output is
0.5 × 642.15 + 0.5 × 641.82 = 641.985. The third is
0.5 × 642.35 + 0.5 × 641.985 = 642.1675. Each step reads the previous
**smoothed output**, not the previous raw reading. No future row is used.

Whole engines are then assigned to learning or validation. Validation means
checking progress without using those engines to adjust the model's learned
numbers. The test file is a separate final check, never a source of training
statistics. Rescaling subtracts each sensor's training average and divides by
its training spread. Alternative range-based scaling is explained in
[Preprocessing](04_the_pipeline.md).

A window keeps consecutive cycles from just one engine. With a length of 30
and a step of one cycle, engine 1 supplies windows at cycles 1–30, 2–31 and
3–32. Each window contains 14 sensor rows and 30 cycle columns. Its answer
belongs to its **last** cycle: the first has raw life 192 − 30 = 162, capped
to 125. Engine number, cycle, operating settings and that answer are not
extra sensor rows.

The playground's Preprocessing diagram shows the steps and sizes for your
selected settings. The detailed arithmetic below uses its named saved run's
settings throughout.

## Through the saved model

A **shape** lists the size of each dimension. At entry, `[1, 14, 30]` means
one window, fourteen sensors and thirty cycles. After the temporal operation,
`[1, 14, 1]` means one computed number per sensor. Inside a graph layer, the
last dimension counts computed numbers, not cycles. After flattening,
`[1, 448]` is one row of numbers, with each sensor's block placed after the
previous one. The small samples below are the first three entries in that
order, not three different engines.

Each graph layer first combines the numbers of the sensors connected to a
given sensor. The graph supplies the mixing multipliers. The layer's learned
multipliers then turn that mixture into a new list of numbers. Negative
results become zero, an operation called ReLU. The next layer reads these
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
internal score. Each addition includes its learned offset and total. None of
these tables is a ranking of sensors.

## Other temporal algorithms

The example above deliberately names its Last cycle operation. Mean instead
averages a sensor's values in the window. Window summary keeps its last value,
average and fitted rate of change. Full window keeps all the values.
LSTM and GRU read them in order and update internal lists of computed numbers;
their learned weights are different from the example's Last cycle network.
Do not interpret that saved prediction as a prediction of those alternatives.
The [temporal explanation](09_temporal.md#the-stage-left-out-by-default)
shows the memory operations and the distinction between stored memory and
the output passed to the graph.

## Training and explaining are different operations

Training takes sensor windows **and separate answers**. It calculates a
prediction, compares it with the answer, then adjusts learned multipliers to
reduce that mismatch. A batch is the group of windows used for one update;
a pass goes through all training windows. The learning rate controls the
update size. A seed selects repeatable random choices; changing it starts a
different training repetition. A device changes which processor performs
these calculations, not what a row means.

Prediction takes only the sensor input and a trained model. Evaluation then
compares predictions with the separate answers. Explanation starts with an
already trained model and asks how its input contributed to a prediction.
Read the [SHAP worked addition](10_dynamic_shap.md) for the separate
input → replacement → prediction-difference calculation. An explanation
does not establish that a sensor caused physical damage.
