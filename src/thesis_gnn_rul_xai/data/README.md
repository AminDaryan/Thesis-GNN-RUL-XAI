# `data` — from raw NASA text files to model input

This folder does all of the preprocessing. It reads the C-MAPSS text files,
picks the sensors, builds the labels, smooths and rescales the signals, cuts
each engine history into fixed-length windows, and hands back plain NumPy
arrays.

If you only want the end result, call `run_preprocessing()` from
[`pipeline`](pipeline/README.md). Everything else here is one step of what that
function does.

> **Each file in this folder has its own folder, holding its code and its own
> page.** This page carries only what they share: what the dataset is, what the
> chain looks like end to end, and the two rules that span more than one file.
> The step-by-step detail — every function, its real input, what happens to that
> input line by line, and its real output — lives on the seven pages linked
> below.

---

## The seven files, and where to read about each

| Folder | What it does | Page |
|---|---|---|
| `loading/` | Reads the three text files, checks their shape, and works out a label for every row. | [read](loading/README.md) |
| `sensors/` | Names all 21 sensors, fixes the 14 the paper keeps and their order, and offers a diagnostic that finds sensors carrying no information. | [read](sensors/README.md) |
| `labeling/` | Stops the label counting above a ceiling. A healthy engine is labelled "125" rather than "191 flights left", because nothing in the sensors can tell those apart. | [read](labeling/README.md) |
| `smoothing/` | Damps the flight-to-flight jitter in each engine's sensor signals. | [read](smoothing/README.md) |
| `normalization/` | Groups rows into operating conditions, then rescales the sensor columns — measuring on training engines only. | [read](normalization/README.md) |
| `windowing/` | Cuts histories into fixed-length windows, and splits engines into a training and a validation set. | [read](windowing/README.md) |
| `pipeline/` | Runs all of the above in order and returns arrays plus a record of what it observed. | [read](pipeline/README.md) |

`__init__.py` at this level is a package marker and imports nothing.

Each of those folders holds its code as `__init__.py`, so the import paths are
unchanged: `from thesis_gnn_rul_xai.data.loading import load_subset` works
exactly as before.

---

## Background you need before reading any of it

**C-MAPSS** is a NASA simulator of a large turbofan aircraft engine. NASA used
it to generate a public benchmark dataset in which simulated engines run until
they fail while 21 virtual sensors record the engine state once per flight
cycle. Each row of NASA's files is "a snapshot of data taken during a single
operational cycle", in the words of the readme NASA ships with them, and the
paper describing the dataset takes "one measurement snapshot per flight"
(Saxena et al., 2008, p. 5). Read together, they mean that one **flight cycle**
is one flight. Neither source says it in those words; it is this project's
reading of the two, as
[Data, What one cycle is](../../../docs/learn/01_the_problem.md#what-one-cycle-is)
explains. The cycle is the unit of time everywhere in this project: there are
no seconds or hours anywhere. (Corrected 2026-10-19: this said a flight cycle
is "one take-off, flight and landing", which neither the readme nor the
dataset paper says. Found when the documentation pages were given their
sources.)

**RUL** stands for Remaining Useful Life: the number of flight cycles an engine
will still run before it fails. It is the number the model has to predict. A
label of 47 means "this engine fails 47 flight cycles from now".

Each sub-dataset (`FD001` … `FD004`) comes as three plain text files:

| File | What it holds |
|---|---|
| `train_FD001.txt` | Complete run-to-failure histories. Every engine is recorded from new until it fails. |
| `test_FD001.txt` | Partial histories. Recording stops at some point *before* failure. |
| `RUL_FD001.txt` | One number per test engine: the true remaining life at the last row that engine has in the test file. |

The two history files have 26 columns separated by spaces, no header row, and a
trailing space at the end of every line. The RUL file has a single column.
Column 1 is the engine number, column 2 is the cycle count, columns 3–5 are
three operational settings (altitude, Mach number, throttle resolver angle), and
columns 6–26 are the 21 sensor measurements.

The dataset is described in Saxena, Goebel, Simon & Eklund (2008), *Damage
propagation modeling for aircraft engine run-to-failure simulation*, IEEE PHM
2008, pp. 1–9. The paper this project re-implements is Wen, Liang, Choy, Fei,
Li & Keshtegar (2026), *Explainable spatiotemporal graph neutral network with
dynamic SHAP enhancement for aeroengine RUL prediction*, Aerospace Science and
Technology 168, 110875.

---

## The chain, end to end

```
  three .txt files
        |
        v
  loading         read, check the shape, attach a label to every row
        |
        v
  sensors         diagnose the 21; select the paper's 14, in Table 3 order
        |
        v
  labeling        stop the label counting above the ceiling
        |
        v
  smoothing       damp the jitter, one engine at a time
        |
        v
  SPLIT           decide which engines are held back  (before anything is measured)
        |
        v
  normalization   operating conditions, then rescaling, fitted on training engines only
        |
        v
  windowing       cut into fixed-length examples; apply the split
        |
        v
  PreparedData    arrays, the sensor graph, the fitted scaling, and a diagnostics record
```

[`pipeline`](pipeline/README.md) walks every one of those steps on a real run,
with the shapes at each stage and the log the run actually produced.

**Where the training and test paths differ.** Both go through the same code.
There are exactly two deliberate differences:

- Every measured quantity — cluster centres, minimums, maximums, means, spreads
  — is learned from the training engines and applied to the validation and test
  engines unchanged.
- Training engines produce **every** window; test engines produce only their
  **final** window, because that is the single moment the official labels
  describe.

---

## The two rules that span more than one file

Everything else is local to a step. These two are not, and both exist to stop a
failure that produces plausible wrong numbers rather than an error.

### 1. Nothing is measured on data the model is not entitled to see

Averages, spreads, minimums, maximums and cluster centres are all computed on
the **training engines only**, then applied unchanged to the held-back and test
engines. The split into training and held-back engines therefore happens
*before* any of them is computed.

This is why [`normalization`](normalization/README.md) splits `fit` from
`transform` rather than offering one function, and why the split lives in
[`pipeline`](pipeline/README.md) at step 6 rather than step 9. The measured
consequences of getting it wrong are on the normalization page.

The related rule inside [`windowing`](windowing/README.md) is that the split is
made **by engine, never by window**: with a stride of one flight, two
neighbouring windows share 29 of their 30 cycles, so splitting windows at random
would put near-copies on both sides.

### 2. The order of the 14 sensors is load-bearing

The sensor graph is a plain 14-by-14 grid of numbers indexed by **position**.
Nothing in it records which sensor is which — that is
`PAPER_SENSOR_SYMBOLS` in [`sensors`](sensors/README.md), and reordering it
would silently wire the model to the wrong sensors with no error of any kind.
The sensors page demonstrates that failure on the real matrix.

Two things defend against it: assertions that run when the module is imported,
and `WindowedDataset` carrying the sensor names alongside the array so the order
travels with the data.

---

## The judgement calls made in this folder

Each of these is a place where the paper does not say what it did, so we had to
choose. Every one has a numbered entry in
[`docs/assumptions/`](../../../docs/assumptions/README.md) with the full reasoning,
and each is explained in plain terms on the page of the file it lives in.

| ID | The choice | Where it lives |
|---|---|---|
| **A-01** | The label ceiling is 125 cycles. Equation 19 gives the formula; the number is never printed. | [labeling](labeling/README.md) |
| **A-02** | The ceiling is applied to the test labels too. This can never raise the reported root mean squared error (the square root of the average squared miss) or NASA score (a sum of penalties that grow exponentially with each miss), though R-squared (the share of the labels' spread the predictions account for) and ASUE (the average shortfall over the engines predicted too low) carry no such guarantee, and it must be stated whenever a number from here is compared with a published one. (Corrected 2026-10-02 from "can only move reported error favourably"; found by the search for that claim after the review of the playground's information cards on 2026-09-26.) | [labeling](labeling/README.md) |
| **A-03** | How the sensors are rescaled. The paper describes no scaling at all. **The largest measured effect in the project.** | [normalization](normalization/README.md) |
| **A-04** | The smoothing strength is 0.5. Section 3.2.3 contradicts itself. | [smoothing](smoothing/README.md) |
| **A-05** | Windows are taken one cycle apart. Table 6 gives the size, never a stride. | [windowing](windowing/README.md) |
| **A-06** | A validation set is held out at all. The paper describes none. Validation error and test error are **not** comparable here. | [windowing](windowing/README.md) |
| **A-07** | This folder hands over each sensor's whole window; the model's default stage then keeps its last cycle, standing in for the paper's recurrent stage, which is built but not the default (assumption A-11). | [windowing](windowing/README.md) |
| **A-08** | The order of the steps — which turned out **not** to be a choice: Figure 1 prints the chain. | [pipeline](pipeline/README.md) |
| **A-09** | The recursive form of the smoothing filter, matching Equation 18. | [smoothing](smoothing/README.md) |
| **A-10** | The seventh dropped sensor is NRc. Section 3.2.2 says seven and names six. | [sensors](sensors/README.md) |
| **A-23** | An engine shorter than the window is left out, not padded. | [windowing](windowing/README.md) |
| **A-25** | The split is decided before the scaling is fitted. | [pipeline](pipeline/README.md) |

Two observations about the paper, established from data rather than chosen, are
findings rather than assumptions: that the shortest FD001 test history is 31
cycles, so no FD001 engine is affected by leaving out engines shorter than the
window (finding F-03), and the fact that
**P15** is one of the paper's 14 while carrying only two distinct values in the
whole FD001 training set — reported on every run rather than quietly corrected.

---

## Getting the data

The NASA files are not in this repository, because they are redistributed under
NASA's own terms. Put `train_FD00X.txt`, `test_FD00X.txt` and `RUL_FD00X.txt`
into `data/raw/`. When a file is missing, the error quotes the download link, so
the fix arrives with the problem — see
[`loading`](loading/README.md).

`load_subset` also compares the row and engine counts against those of the
files NASA distributes and **warns** on a mismatch rather than stopping,
because a legitimate variant should not halt the work while a silent mismatch
would be far worse. The paper's Table 5 prints the same engine counts, while
NASA's own readme swaps FD004's two (corrected 2026-10-02: this paragraph
called them "the published figures"; [`loading`](loading/README.md) has the
details).
