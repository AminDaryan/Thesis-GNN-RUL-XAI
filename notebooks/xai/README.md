# `xai/` — explaining a trained network, one method per notebook

A trained network takes one example — 14 sensors, each with its last 30 flight
cycles of readings — and returns one number: the flight cycles of life it
predicts the engine has left. Nothing among the network's parameters says which
sensors that number came from. An **explanation method** is a procedure for
dividing the prediction up among the sensors, so a reader can see what the
network leaned on.

There is more than one such procedure, they can disagree, and each rests on
assumptions of its own. So each gets a notebook here, and all of them explain
the *same* network in the *same* way, so that a difference between two
notebooks is a difference between two methods and nothing else.

Read `notebooks/README.md` first, for how every notebook is laid out, and
notebook 02 before any of these, for what the network is.

The notebooks so far, and what sets their methods apart, are in one table near
the end of this page, after the words it uses have been explained.

## What every notebook here shares, and why

**The same trained network.** Every notebook loads seed 42 of the most recent
stored five-seed run of `config/fd001.yaml` — the paper's sensor graph, the
last-cycle temporal stage, 20 % of engines held back — from the `model.pt` that
`scripts/train.py` saved beside the results. None of them trains a network of
its own, for three reasons:

- Two methods can be compared only if they explain the same network. Two
  networks trained separately, even with the same seed, are equal only if
  nothing about the code, the library or the processor differs between the two
  runs, and a notebook cannot promise that.
- A stored run records the commit of the code that trained it. A network
  trained inside a notebook records nothing (working rule 3a).
- It is a network whose test error is part of this project's reported result,
  so what is explained is what was measured.

The loading cell checks the network it loaded against the run: it scores the
test engines again and requires the same error the run recorded for that seed.

**The same data**, rebuilt from the settings saved beside the run with that
seed. The seed decides which 20 engines were held back from training, so
rebuilding with a different seed would hand the network engines it learned
from and call them unseen.

**The same engine for the view over time.** The held-back engine with the most
windows, which the network never learned from and which shows the most of an
engine's life. For seed 42 that is engine 5, with 240 windows.

**The same stand-in for a removed sensor.** A network built for fourteen inputs
cannot simply be given thirteen, so every method here "removes" a sensor by
replacing its 30 readings with that sensor's average window over the training
engines (assumption A-20). The prediction for a window made entirely of those
averages is the reference every contribution is measured from.

**The same unit.** Every contribution is in flight cycles of predicted
remaining life. A value of +4 for a sensor means its actual readings moved the
prediction four cycles higher than a typical reading would have.

**The same picture.** `plot_contributions_over_time`, in
`src/thesis_gnn_rul_xai/explain/plots.py`, draws one small panel per sensor
showing its contribution at every cycle of the engine's life, largest average
contribution first, all panels on one vertical scale. A notebook comparing two
methods draws both lines in each panel.

## The seven questions each notebook answers

An explanation method is not explained by its definition. Each notebook here
answers, in its own section, these seven:

1. **What object is being explained?** Here always one prediction: one window
   in, one number of cycles out.
2. **What does the method have access to?** Only inputs and outputs, or the
   network's inner workings too?
3. **What does it change?** Which part of the input it varies, and to what.
4. **What does it measure?** The change in the prediction, and under which
   circumstances.
5. **What does the result look like?** Here, fourteen numbers per window, in
   cycles.
6. **What does it assume?**
7. **What does it not show?**

## What no method here establishes

- **How an engine wears.** Every number here describes how *this network* used
  its inputs. A large contribution from `Nc` means the network's prediction
  moved with `Nc`'s readings. It does not mean the core speed causes wear, or
  even that it is the best indicator of wear: a network that leaned on a
  different sensor could have predicted as well.
- **What would happen to a real engine.** Replacing a sensor's readings with an
  average makes a combination of readings no engine ever produced. The network
  still returns a number for it, but nothing checks that the number means
  anything.
- **Anything about another network.** A second seed trains a different
  network, and its explanation can differ. The Shapley-value notebook compares
  all five of the stored run's networks before calling a ranking stable.

## The methods so far, side by side

| | Shapley values | Occlusion |
|---|---|---|
| Notebook | `01_shap_fd001.ipynb` | `02_occlusion_fd001.ipynb` |
| What it tries | every combination of sensors | one sensor at a time swapped for its stand-in |
| In the paper | Yes, its Equations 14 to 16 | No |
| Network evaluations per window, for 14 sensors | 16 384, one per combination | 15, the window and one per sensor |
| Adds up to the prediction minus the reference prediction | Always, to rounding | No |
| What "the sensor's effect" is measured against | every combination of the other sensors, averaged | only the one where every other sensor is present |

## Adding a method

1. Write it as a function in `src/thesis_gnn_rul_xai/explain/`, with a section in
   that folder's `README.md`, a test, and a worked example in
   `tests/test_examples_torch.py`.
2. Copy the layout of `02_occlusion_fd001.ipynb`: load the network with
   `load_seed`, explain the same engine, draw it with
   `plot_contributions_over_time` beside the Shapley values, answer the seven
   questions, and end with a final verification.
3. Add a column for it to the table above and a line to the documentation
   site's list in `src/thesis_gnn_rul_xai/docsite/catalog.py`.
