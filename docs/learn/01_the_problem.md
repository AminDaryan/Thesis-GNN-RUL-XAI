# The problem: how long has this engine got?

## Remaining useful life

**Remaining Useful Life**, almost always shortened to **RUL**, is the number of
flight cycles an engine will still run before it fails, and it is what this
project predicts. A **flight cycle** is one complete take-off, flight and
landing, and it is the unit of time used everywhere in this project.

The reason anyone wants the number is **predictive maintenance**: servicing
equipment according to its measured condition rather than on a fixed calendar
or after it breaks. Servicing too early wastes an engine's remaining life;
servicing too late means a failure in service.

---

## What the engine's sensors measure

{{figure:engine}}

Air enters at the fan, is compressed by a low-pressure compressor (LPC) and
then a high-pressure one (HPC), is burned with fuel in the combustor, and
drives a high-pressure turbine (HPT) and then a low-pressure one (LPT) on its
way out. The names in the diagram are the ones used throughout this project:
`T24` is the temperature at the LPC outlet, `P30` the pressure at the HPC
outlet, and so on.

**The sensors are not independent.** The engine is a chain: air that has been
heated by one stage arrives at the next stage already hot, so a change in one
component shows up in the readings of the component after it. A model that
treats the sensors as unrelated numbers throws that structure away, and keeping
it is the whole point of the method this project rebuilds.

---

## NASA's benchmark data: C-MAPSS

**C-MAPSS** stands for *Commercial Modular Aero-Propulsion System Simulation*,
a NASA program that simulates a large jet engine. NASA used it to build a
public dataset in which simulated engines are run until they fail, with 21
sensors recorded once per cycle. It has been the standard test problem for this
task since 2008, which is what makes results comparable between papers at all.

This project works on the sub-dataset called **FD001**. Measured from the files
in `data/raw/`:

| | |
|---|---|
| Training engines | 100 |
| Rows of training data (one per engine per cycle) | 20 631 |
| Test engines | 100 |
| Rows of test data | 13 096 |
| Shortest training life | 128 cycles |
| Median training life | 199 cycles |
| Longest training life | 362 cycles |

Every training engine is run to failure, so its last row is the moment it
broke. Every test engine is cut off partway through, and a separate file gives
the true remaining life at that cut-off point. That is the number a model has
to guess.

The engines are nominally identical, yet they last anywhere from 128 to 362
cycles. Whatever separates a 128-cycle engine from a 362-cycle one has to be
read out of the sensors.

---

## Only 14 of the 21 sensors

Running this project's own diagnostic over the 100 training engines finds that
**six sensors never change at all**: `T2`, `P2`, `epr`, `farB`, `Nf_dmd` and
`PCNfR_dmd` hold a single value for all 20 631 rows. A sensor that never moves
cannot carry information about wear.

The paper keeps 14 sensors and drops 7. It names six of the dropped ones, and
they are exactly these six. It does not name the seventh, but it states its
rule: drop a sensor that is constant, or that rises and falls almost in step
with another sensor (a correlation above 0.95, on a scale where 1 means
perfectly in step). On this data only one pair passes that bar: `Nc` and
`NRc`, two measures of the same shaft's speed, at 0.96. The paper keeps `Nc`, so the seventh is `NRc`
(assumption A-10).

The same diagnostic flags one more sensor, `P15`, which takes only two values,
21.60 and 21.61, across all 20 631 training rows, so its reading can only jump
between two levels, never drift gradually as the engine wears. The paper keeps
`P15` anyway, and so does this project: the paper's map of which sensors are
connected is drawn over exactly its 14, so removing one would change that map.
The weakness is recorded as a fact about the paper (finding F-01).

---

## What the model is asked to predict

While it trains, a model is shown sensor readings together with the right
answer, called the **label**. The obvious label is the number of cycles the
engine really had left, counting down by one each cycle. That is not the label
used here, and the reason is physical.

A new engine is healthy. Its sensors barely move, and nothing in the readings
distinguishes an engine with 300 cycles left from one with 250 left, because
neither has begun to degrade. Asking a model to tell those apart is asking it
to learn noise.

So the label is **piecewise linear**, made of two straight pieces: it is the
smaller of two numbers, the cycles really left and a ceiling of 125. Far from
failure the ceiling is the smaller, so the label stays flat; in the last 125
cycles the real count is the smaller, so the label falls by one per cycle.
Engine 1 of the training data ran 192 cycles:

| Cycle | Cycles really left (192 − cycle) | Label (smaller of that and 125) |
|---|---|---|
| 50 | 142 | 125 |
| 67 | 125 | 125 |
| 68 | 124 | 124 |
| 152 | 40 | 40 |

{{figure:rul-curve}}

The paper's Equation 19 uses a ceiling but **never prints its value**, so 125
is this project's choice (assumption A-01). It is read off the paper's Fig. 7,
whose FD001 panel draws the paper's own label for one engine, so the height of
its flat part is the ceiling. Measured against that panel's axis ticks, the
flat part sits at 125.2, which is 125 to within the width of the printed line.
It also agrees with the paper's description of predictions holding "around
125" while an engine is healthy, and with related work on the same data.
