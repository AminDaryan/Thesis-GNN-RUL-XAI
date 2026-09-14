# The problem: how long has this engine got?

## Remaining useful life

**Remaining Useful Life**, almost always shortened to **RUL**, is the number of
flight cycles a machine will still run before it fails. A **flight cycle** is
one complete take-off, flight and landing. It is the unit of time used
everywhere in this project — not hours, not kilometres, but whole flights.

RUL is the quantity being predicted. If a model says an engine has 40 cycles
left, that is a claim that the engine will survive roughly 40 more flights.

The reason anyone wants the number is **predictive maintenance**: servicing
equipment according to its measured condition rather than on a fixed calendar
or after it breaks. Servicing too early wastes an engine's remaining life.
Servicing too late means a failure in service. A good RUL estimate lets an
airline plan.

---

## What a turbofan measures

{{figure:engine}}

Air enters at the fan, is compressed by a low-pressure compressor and then a
high-pressure one, is burned with fuel in the combustor, and drives two
turbines on its way out. Each stage has sensors, and the names in the diagram
are the ones used throughout this project — `T24` is the temperature at the
low-pressure compressor outlet, `P30` the pressure at the high-pressure
compressor outlet, and so on.

The important structural fact is that **the sensors are not independent**. The
engine is a chain: air that has been heated by one stage arrives at the next
stage already hot. A change in one component shows up in the readings of the
component after it. A model that treats the sensors as unrelated numbers throws
that structure away — and keeping it is the whole point of the method this
project rebuilds.

---

## The C-MAPSS benchmark

**C-MAPSS** stands for *Commercial Modular Aero-Propulsion System Simulation*,
a NASA program that simulates a large turbofan. NASA used it to build a public
dataset in which simulated engines are run until they fail, with 21 sensors
recorded once per cycle. It has been the standard test problem for this task
since 2008, which is what makes results comparable between papers at all.

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

The spread of lifetimes is what makes the task hard. The engines are nominally
identical, and they last anywhere from 128 to 362 cycles. Whatever separates a
128-cycle engine from a 362-cycle one has to be read out of the sensors.

> **Where these numbers come from.** They were computed from the real NASA
> files by this project's own loading code, not copied from a paper. The
> project deliberately contains no synthetic data: a pipeline that appears to
> work on invented signals tells you nothing about whether it works on engines.
> If the files are missing, the code says so and points at the download rather
> than substituting something with the right shape.

---

## Only 14 of the 21 sensors

Not every sensor is worth using. Running this project's own diagnostic over the
100 training engines finds that **six sensors never change at all** — `T2`,
`P2`, `epr`, `farB`, `Nf_dmd` and `PCNfR_dmd` hold a single value for all
20 631 rows. A sensor that never moves cannot carry information about wear.

The paper being re-implemented keeps 14 sensors and drops 7. Six of its seven
are exactly the six flat ones above, which is a strong sign the selection is
sound. The seventh it names only implicitly; working out which sensor it must
be is one of this project's documented choices (A-10).

There is a twist worth meeting early, because it is a good example of what this
kind of study produces. The diagnostic flags a **seventh** uninformative
sensor, `P15`, which takes only two distinct values across the whole training
set. The paper *keeps* `P15`. This project keeps it too — the sensor graph is
defined over the paper's fixed set of 14, and quietly changing the selection
would make the reproduction something else — but the observation is recorded as
a finding rather than swept up (finding F-01).

---

## The label: what the model is asked to predict

You might expect the training label to be simply "cycles until this engine
failed", counting down one per cycle. It is not, and the reason is physical.

A new engine is healthy. Its sensors barely move, and nothing in the readings
distinguishes an engine with 300 cycles left from one with 250 left, because
neither has begun to degrade. Asking a model to tell those apart is asking it
to learn noise.

So the label is **piecewise linear**: it is held at a ceiling while the engine
is healthy, and only starts falling once degradation is under way.

$$\mathrm{RUL}(t) = \min\left(T_{\text{total}} - t,\; \mathrm{RUL}_{\max}\right)$$

{{figure:rul-curve}}

The solid line is the label this project uses. The dashed line is the
alternative — counting down from the start — which would demand that the model
distinguish states that are genuinely indistinguishable.

The ceiling value is 125 cycles. **The paper never prints that number.** It
defines the ceiling symbolically in its Equation 19 and leaves the value
unstated. 125 was arrived at three ways: by reading it off the true-RUL curve
in one of the paper's own figures at high magnification, by matching the
paper's description of predictions holding "around 125" while an engine is
healthy, and by noting that closely related work on the same dataset uses the
same value. The choice is recorded with that reasoning (assumption A-01).

That is the first judgement call of the path, and it is a fair example of the
rest: the paper is specific about the shape of the thing and silent about a
number the shape needs.

---

## What you now know

- **RUL** is remaining flight cycles, and predicting it enables predictive
  maintenance.
- The engine's sensors are **physically related** through the gas path, which
  is the structure the method exploits.
- **FD001** holds 100 training engines run to failure and 100 test engines cut
  off partway, with lifetimes spread from 128 to 362 cycles.
- **14 of 21 sensors** are used; six of the seven dropped ones are provably
  flat in the data, and a seventh flat sensor is kept anyway (finding F-01).
- The label is **held at a ceiling of 125** while the engine is healthy, and
  that ceiling is a choice this project had to make (A-01).

Next: what a graph is, and what a network built on one actually computes.
