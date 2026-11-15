# C-MAPSS dataset (2008)

NASA's C-MAPSS engine data come as four subsets, FD001 to FD004, each made of
three text files: simulated jet engines recorded once per flight, the training
engines until they fail and the test engines cut off before they do, and a
file holding each test engine's true remaining life (NASA Prognostics Center
of Excellence [NASA PCoE], 2008; Saxena et al., 2008, pp. 1, 5). That each row
is one flight is this project's reading of those sources, explained on
[Data, What one cycle is](../learn/01_the_problem.md#what-one-cycle-is).
This project trains and tests on one of the four subsets, FD001. This page explains where
the files come from, as their three sources tell it:

- the **dataset paper**, [Saxena et al. (2008)](../references/saxena_2008_damage_propagation_modeling.pdf),
  which explains how engine damage was simulated and how the challenge data
  built from it were split and scored;
- the **simulator's user guide**, [Frederick et al. (2007)](../references/frederick_2007_cmapss_users_guide.pdf),
  which describes the engine model the paper ran and is the paper's
  reference [11] for it (Saxena et al., 2008, pp. 2, 9);
- the **readme** NASA ships inside the data download,
  [`readme.txt`](../references/nasa_2008_cmapss_readme.txt) (NASA PCoE, 2008),
  the only one of the three that names the four subsets.

How this project reads the files, works out a label for every row and chooses
its 14 sensors is taught step by step on the Learn page
[Data: from engine histories to known answers](../learn/01_the_problem.md).
This page is the sources' account; that one is the project's. One section here is
the project's own as well:
[What the FD001 training engines look like](#what-the-fd001-training-engines-look-like)
draws the training engines' lives, their sensors' paths and the sensors'
correlations from the files themselves.

The worked examples below run on the real files in `data/raw/`. Every table
computed from the files is reproduced by the code in the example above it.

<details class="process-example">
<summary>Set up the Python examples</summary>

Run the blocks on this page in order from the repository root. This one reads
all twelve files, which takes a few seconds:

```python
import numpy as np
import pandas as pd
from thesis_gnn_rul_xai.data.loading import read_cmapss_frame, read_rul_file
from thesis_gnn_rul_xai.data.sensors import SENSORS
from thesis_gnn_rul_xai.evaluation.metrics import nasa_score

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
train = {name: read_cmapss_frame(f"data/raw/train_{name}.txt") for name in SUBSETS}
test = {name: read_cmapss_frame(f"data/raw/test_{name}.txt") for name in SUBSETS}
answers = {name: read_rul_file(f"data/raw/RUL_{name}.txt") for name in SUBSETS}
symbol = {sensor.column: sensor.symbol for sensor in SENSORS}


def rounded_settings(frame):
    """Settings 1 and 3 rounded to whole numbers, setting 2 to two decimals; adding 0.0 turns -0.0 into 0.0."""
    return pd.DataFrame({
        "setting 1": frame.op_setting_1.round(0) + 0.0,
        "setting 2": frame.op_setting_2.round(2) + 0.0,
        "setting 3": frame.op_setting_3.round(0) + 0.0,
    })


def condition_count(frame):
    """How many distinct rounded combinations of the three settings the rows contain."""
    return len(rounded_settings(frame).drop_duplicates())
```

</details>

---

## The problem the paper set out to solve

**Prognostics**, as Saxena et al. (2008, Section II, p. 1) define it, is
estimating the **remaining useful life (RUL)** of a component, measured in
hours, cycles or another unit of time. The end of that life is a user's
choice: the moment a measured quantity passes the limit the user sets for safe
operation (Section II, pp. 1–2). What remaining useful life means for one row of
these files is on
[Data, Remaining useful life](../learn/01_the_problem.md#remaining-useful-life).

To build a method that estimates remaining life from data, you need examples
where the answer is known: machines recorded all the way to failure.
Saxena et al. (2008, Section I) explain why those are scarce: records that
follow a real machine to failure are rare and seldom made public, and few
shared run-to-failure datasets existed, which made it hard for research groups
to compare their methods on the same machines (p. 1).

The authors' answer was to **simulate** the engines instead of waiting for real
ones to fail. They took an existing engine simulator, injected damage that grows
over time, ran each simulated engine until a failure criterion was met, and
recorded the sensor readings once per flight (Saxena et al., 2008, pp. 1, 5).
The result became the data for the first data challenge of the Prognostics and
Health Management conference, PHM'08, whose task was to estimate remaining life
from historical data alone (Saxena et al., 2008, Section I, p. 1).

---

## The simulator: C-MAPSS

### The engine and its flow path

**C-MAPSS**, the Commercial Modular Aero-Propulsion System Simulation, is a
NASA program (Frederick et al., 2007, p. 45), written in MATLAB and Simulink
(two numerical computing environments), that simulates a large commercial
**turbofan**: a jet engine whose large front fan drives air through a bypass
duct around the engine core as well as into the core itself (Saxena et al.,
2008, Section III.A and Figs. 1–2, p. 2). The simulated engine is in the
90,000 lb thrust class, and both sources give its operating range as sea level
to 40,000 ft, Mach numbers (speed as a fraction of the speed of sound) from 0
to 0.90, and sea-level temperatures from −60 to 103 °F
(Frederick et al., 2007, Section 1.1,
[p. 8](../references/frederick_2007_cmapss_users_guide.pdf#page=8); Saxena et al., 2008,
Section III.A, [p. 2](../references/saxena_2008_damage_propagation_modeling.pdf#page=2)).

Air passes through the engine's modules in this order, which is the content of
Figure 2 of Saxena et al. (2008, p. 2) redrawn as text:

```text
Ambient air → Inlet → Fan → Splitter ─┬→ Bypass path → Bypass nozzle
                                      │
                                      └→ LPC → HPC → Burner → HPT → LPT → Core nozzle
                                                       ↑
                                                      Fuel
```

The fan, the low-pressure compressor (LPC) and the high-pressure compressor
(HPC) squeeze the air; fuel burns in the burner, which Figure 1 calls the
combustor (Saxena et al., 2008, Figs. 1–2, p. 2); the
high-pressure and low-pressure turbines (HPT and LPT) take energy out of the
hot gas. A **spool** is one shaft with compressor and turbine stages fixed to
it, so that the turbine turns the compressor. Figure 1 of the paper draws two:
the HPC and the HPT, in yellow, share the outer shaft, and the LPC and the
LPT, in dark blue, share the inner shaft, which also carries the fan (this
page's reading of the drawing, which colours each spool's parts alike). The
user guide's list of symbols names N1 the fan spool speed and N2 the core
spool speed (Frederick et al., 2007,
[p. 7](../references/frederick_2007_cmapss_users_guide.pdf#page=7)), which is
why the files hold two physical speeds, Nf and Nc (Saxena et al., 2008,
Table 2). The drawing's arrows say the reverse: in Figure 1, and in the
guide's own Figure 1.1, the arrow from N1 ends on the outer shaft and the
arrow from N2 on the inner one (Saxena et al., 2008, Fig. 1,
[p. 2](../references/saxena_2008_damage_propagation_modeling.pdf#page=2);
Frederick et al., 2007, Fig. 1.1,
[p. 9](../references/frederick_2007_cmapss_users_guide.pdf#page=9); this
page's reading of both figures rendered at high resolution). This page
follows the list of symbols.

The engine does not fly itself. C-MAPSS includes a **control system**, the
software that sets the fuel flow (Frederick et al., 2007, pp. 29–30). Its
parts, as Saxena et al. (2008, Section III.A, p. 2) and Frederick et al.
(2007, Section 1.1, p. 8) list them:

| Part | What it does |
|---|---|
| Fan-speed controller | Sets the fuel flow so that the fan turns at the speed the throttle demands; the throttle setting is the throttle resolver angle (TRA), given in degrees (Frederick et al., 2007, [p. 8](../references/frederick_2007_cmapss_users_guide.pdf#page=8), [p. 11](../references/frederick_2007_cmapss_users_guide.pdf#page=11) and [pp. 28–29](../references/frederick_2007_cmapss_users_guide.pdf#page=28)) |
| Three upper-limit regulators | Keep the core speed, the engine pressure ratio and the HPT exit temperature below their design limits |
| One lower-limit regulator | Keeps the static pressure at the HPC exit from falling too low |
| Acceleration and deceleration limiters | Bound how fast the core speed may change |

The paper ran the engine **closed loop**, with this control system active, and
in batch mode, without the program's graphical screens (Saxena et al., 2008,
Section III.A, [pp. 2–3](../references/saxena_2008_damage_propagation_modeling.pdf#page=2)).

The closed loop matters for reading the data. In the guide's example of a
sudden fan fault, the fan speed first jumps and then returns to its demanded
value as the controller cuts the fuel flow, while fuel flow, engine pressure
ratio, thrust and other variables settle at lower values
(Frederick et al., 2007, Section 3.1, p. 29,
[Fig. 3.2](../references/frederick_2007_cmapss_users_guide.pdf#page=29) and
[Fig. 3.3](../references/frederick_2007_cmapss_users_guide.pdf#page=30)).
That example is a sudden fault in the fan, and the guide's example of slow
wear degrades all five rotating modules at once with the limit regulators
switched off (Frederick et al., 2007, Section 3.2,
[p. 30](../references/frederick_2007_cmapss_users_guide.pdf#page=30)),
so neither says which way slow HPC wear moves each sensor. What they show is
the controller holding the fan at its demand. How far the fan speed of the
FD001 engines still moves is measured below, under
[One training engine's record](#one-training-engines-record).

### What goes in: fourteen inputs

Table 1 of Saxena et al. (2008) lists the simulator's 14 inputs. One is the
fuel flow. The other 13 are **health-parameter modifiers**: numbers that shift
a rotating module's efficiency, its flow capacity (how much air it passes) and,
for the fan, the LPC and the HPC, its pressure ratio, away from their healthy
values (Frederick et al., 2007, p. 27; Saxena et al., 2008, Table 1).
In standard engineering usage, which neither source spells out, a
compressor's **pressure ratio** is its outlet pressure divided by its inlet
pressure, and its **efficiency** is the share of the work put into it that
goes into raising the pressure rather than into heating the air: a less
efficient compressor heats the air more for the same rise in pressure. The
fan, LPC and HPC have three modifiers each; the two turbines have two,
efficiency and flow (Frederick et al., 2007, p. 28; Saxena et al., 2008,
Table 1). The user guide shows what a modifier's value means: a step of a
modifier from 0 to −0.02 is a 2 percent degradation of that quantity
(Frederick et al., 2007, Section 3.0,
[p. 28](../references/frederick_2007_cmapss_users_guide.pdf#page=28)).

To simulate a worn HPC, the paper drove two of these: the **HPC efficiency
modifier** and the **HPC flow modifier**, the rows Table 1 shades in grey
(Saxena et al., 2008, Table 1, [p. 3](../references/saxena_2008_damage_propagation_modeling.pdf#page=3)).

The last row of Table 1 is named as an HPT flow modifier but carries the symbol
`LPT_flow_mod`; the user guide's Table 1.1 has the same slip (Frederick et
al., 2007, Table 1.1, p. 10; Saxena et al., 2008, Table 1). From the symbol and
from the guide's statement that each turbine has an efficiency and a flow
modifier (Frederick et al., 2007, p. 28), the row is the LPT's flow modifier.

**Do not confuse inputs with sensors.** The modifiers are what the authors
changed to create damage. None of them is in the released files; the files hold
the flight condition and what the engine reported back (NASA PCoE, 2008).

### What comes out: 21 sensors and four hidden margins

C-MAPSS can report 58 output variables, and the paper kept 21 of them as the
sensor data given to participants (Saxena et al., 2008, Section III.A, p. 3).
Table 2 lists them with their units (Saxena et al., 2008, Table 2,
[p. 3](../references/saxena_2008_damage_propagation_modeling.pdf#page=3)).
Most of the names are engineering terms. In plain words, by standard usage
(Table 2 itself gives a formula only for epr and phi):

| Symbols | What they are |
|---|---|
| T2, T24, T30, T50 | Total temperature at the fan inlet, the LPC outlet, the HPC outlet and the LPT outlet, in °R |
| P2, P15, P30 | Pressure at the fan inlet; total pressure in the bypass duct and at the HPC outlet; in psia, pounds per square inch counted from vacuum |
| Ps30 | Static pressure at the HPC outlet |
| Nf, Nc | Physical speeds of the fan spool and the core spool, in rpm (revolutions per minute) |
| NRf, NRc | Corrected speeds, derived from the physical speeds and the inlet temperature (Wen et al., 2026, p. 5): the physical speeds rescaled to what they would be on a standard day, so that readings from hot and cold days compare |
| epr | Engine pressure ratio, P50/P2: the LPT outlet pressure over the fan inlet pressure (Saxena et al., 2008, Table 2; for P50, Frederick et al., 2007, Table 1.2, p. 10) |
| phi | Fuel flow divided by Ps30 (Saxena et al., 2008, Table 2) |
| BPR | Bypass ratio: the mass of air passing around the core per unit mass passing through it |
| farB | Burner fuel-air ratio: the mass of fuel burned per unit mass of air in the burner |
| htBleed | Bleed enthalpy: the heat content of air drawn off the compressor (bleed air); Table 2 gives it no unit |
| Nf_dmd, PCNfR_dmd | The demanded fan speed, and the same demand as a corrected speed |
| W31, W32 | Coolant bleed: flows of compressor air sent to cool the turbines, in lbm/s (pounds of air per second) |

**°R**, degrees Rankine, are Fahrenheit-sized degrees counted from absolute
zero, so a temperature in °F is the one in °R minus 459.67. The fan-inlet
temperature T2 checks this against the guide:

<details class="process-example">
<summary>Example in Python</summary>

```python
fd001_rows = pd.concat([train["FD001"], test["FD001"]])
inlet = fd001_rows.sensor_1.unique()
assert symbol["sensor_1"] == "T2" and list(inlet) == [518.67]
assert round(inlet[0] - 459.67, 2) == 59.0
```

</details>

T2 reads 518.67 °R in every FD001 row, and 518.67 − 459.67 = 59 °F, the
standard-day sea-level temperature of the guide's Table 1.4
(Frederick et al., 2007, Table 1.4,
[p. 11](../references/frederick_2007_cmapss_users_guide.pdf#page=11)).

**Do not confuse total pressure with static pressure.** P30 and Ps30 are both
measured at the HPC outlet (Saxena et al., 2008, Table 2). Static pressure is
what the moving air presses on a wall it flows along; total pressure is what
the air would reach if it were brought to rest, so it also counts the part due
to the air's speed. The two
are different quantities at the same place, and both are among this project's
14 sensors.

Table 2 has a second part. Four more outputs were used **only** to compute the
engine's health index and were not given to participants: the HPT outlet
temperature T48, also called the exhaust gas temperature (EGT), and the stall
margins of the fan, the LPC and the HPC (Saxena et al., 2008, Table 2). A
**stall margin** is how far a compressor is operating from its stall limit
(Saxena et al., 2008, Section V.D, p. 6), which by standard usage is the point
where its airflow breaks down; the smaller the margin, the closer it is to
stalling.

The user guide stores these variables in two places. Twelve of the 21 sensors
are among its 27 model outputs (Frederick et al., 2007,
[Table 1.2, p. 10](../references/frederick_2007_cmapss_users_guide.pdf#page=10));
the other nine — T2, P2, BPR, farB, htBleed, Nf_dmd, PCNfR_dmd, W31 and W32 —
are internal variables it lists separately (Frederick et al., 2007,
[Table 1.3, p. 11](../references/frederick_2007_cmapss_users_guide.pdf#page=11)).
Its Table 1.3 calls both W31 and W32 an HPT coolant bleed, where the paper's
Table 2 calls W32 the LPT's (Frederick et al., 2007, Table 1.3, p. 11; Saxena
et al., 2008, Table 2). The project's engine drawing follows the paper's
Table 2 here (assumption A-31).

**Do not confuse the four fan speeds.** Four columns hold a fan speed, and the
evidence for the column order further down depends on telling them apart. The
first row of the FD001 training file, read in the column order the project
uses:

<details class="process-example">
<summary>Example in Python</summary>

```python
first_row = train["FD001"].iloc[0]
fan_speeds = pd.DataFrame([
    {"sensor": symbol[column], "column": column,
     "engine 1, cycle 1": first_row[column], "what it is": meaning}
    for column, meaning in [
        ("sensor_8", "measured physical fan speed"),
        ("sensor_13", "measured fan speed, corrected to a standard day"),
        ("sensor_18", "the fan speed the controller is told to hold"),
        ("sensor_19", "that demand as a corrected speed"),
    ]
])
```

</details>

<!-- sample: fan_speeds -->
| sensor | column | engine 1, cycle 1 | what it is |
| --- | --- | --- | --- |
| Nf | sensor_8 | 2388.06 | measured physical fan speed |
| NRf | sensor_13 | 2388.02 | measured fan speed, corrected to a standard day |
| Nf_dmd | sensor_18 | 2388 | the fan speed the controller is told to hold |
| PCNfR_dmd | sensor_19 | 100 | that demand as a corrected speed |

FD001's inlet is at the standard-day temperature, as the T2 check showed, so
the corrected fan speed nearly equals the physical one, 2388.02 against
2388.06, as the standard definition of a corrected speed, from the physical
speed and the inlet temperature (Wen et al., 2026, p. 5), leads one to expect.

---

## How damage was modelled

### The idea: degrade one module's efficiency and flow

The paper builds on earlier work that tracks an engine's wear through the
efficiency and flow of its compressors and turbines — Kurosaki et al.,
Chatterjee and Litt, and Goebel et al., its references [8] to [10]
(Saxena et al., 2008, Section III, [p. 2](../references/saxena_2008_damage_propagation_modeling.pdf#page=2)) —
and states the premise outright: the health of a compressor or turbine module
is described by both its efficiency and its flow (Saxena et al., 2008,
Section IV.D, [p. 5](../references/saxena_2008_damage_propagation_modeling.pdf#page=5)).

Before simulating damage, the authors checked that C-MAPSS reacts to such
changes the way a real engine does. For the HPC, HPT and LPT they lowered
efficiency and flow step by step, ran the simulator at randomly chosen cruise
conditions, and recorded each sensor and margin. A **response surface** is the
result: a map of what one output reads for every combination of efficiency
loss and flow loss. Figures 3 and 4 show the maps of the HPC stall margin and
of the exhaust gas temperature for a degrading HPC, and the authors compared
their shape with maps published for a real engine by Goebel et al., their
reference [10] (Saxena et al., 2008, Section III.B,
[pp. 3–4](../references/saxena_2008_damage_propagation_modeling.pdf#page=3)).

Read another way — this page's reading, not the paper's — the maps also show
why two numbers per module can stand for its damage: if every output is a
function of a module's efficiency and flow, then making those two fall over
time makes every output drift.

### Why an exponential

Section IV reviews three common models of time to failure: the Arrhenius model
for chemical and diffusion processes, the Coffin–Manson model for fatigue
cracking under repeated heating, and the Eyring model for several stresses
together (Saxena et al., 2008, Section IV, Eqs. 1–3,
[p. 4](../references/saxena_2008_damage_propagation_modeling.pdf#page=4)).
All three give the time, or the number of cycles, to failure from a
temperature or a stress; none of them describes damage as a function of time.
The authors nevertheless state that exponential growth of the fault is common
to all degradation models, cite similar trends seen in practice (their
reference [10]), and on that basis impose an exponential term on the health
parameters instead of modelling any one physical mechanism (Saxena et al.,
2008, Section IV.D, p. 4).

### One module's health over time: Equations 4 to 6

The paper builds the formula in three moves. Wear is written as
$w = A e^{B(t)}$, growing with time $t$. An upper wear threshold $th_w$ marks the
point beyond which the module cannot be used. A **health index** $h(t)$ is then
defined as the distance still left to that threshold, as a fraction of it:
$h(t) = 1 - A e^{B(t)}/th_w$, which is 1 for a module with no wear and 0 at the
threshold. Renaming $A/th_w = e^{a}$ and choosing $B(t) = t^{b}$, the paper
writes (Saxena et al., 2008, Section IV.D, p. 4):

$$h(t) = 1 - \exp\{a\,t^{b}\} \tag{4}$$

and, adding a non-zero initial degradation $d$ so that the generation can start
at an arbitrary point of wear rather than with a brand-new engine
(Saxena et al., 2008, p. 5):

$$h(t) = 1 - d - \exp\{a\,t^{b}\} \tag{5}$$

- $t$ is time in cycles, as the paper's abstract and its Figure 5 count it
  (Saxena et al., 2008, p. 1, and Fig. 5, p. 5); one cycle is one
  flight, in this project's reading of its sources
  ([Data, What one cycle is](../learn/01_the_problem.md#what-one-cycle-is)).
- $h(t)$ is the health of one quantity at time $t$: 1 healthy, 0 at the limit.
- $d$ is the degradation the module already has at the start.
- $a$ sets how fast the damage term grows, and $b$ how sharply it bends upward
  with time.

**What it is for.** It turns the idea of exponentially growing damage into a
curve that starts near 1 and falls ever faster to 0, so that each simulated engine wears
out at its own pace, set by its own $a$, $b$ and $d$.

**What the printed formula does, with numbers.** Two things do not fit, and
both can be checked by arithmetic. First, substituting the stated renamings
into $h(t) = 1 - A e^{B(t)}/th_w$ gives $1 - e^{a}\,e^{t^{b}} = 1 - \exp\{a + t^{b}\}$,
a sum in the exponent, not the product $a\,t^{b}$ that Equation 4 prints.
Second, at $t = 0$ the printed Equation 5 gives $1 - d - e^{0} = -d$: a health
already at or below zero before the engine has flown, whereas Section V.B says
each health trajectory starts between 1 and 0.99 (Saxena et al., 2008, p. 6).
The table evaluates Equation 5 exactly as printed, at the smallest and largest $a$ and $b$ that
Equation 10 allows and at a starting degradation of 0 and of 1 percent:

<details class="process-example">
<summary>Example in Python</summary>

```python
def printed_health(t, a, b, d):
    """Eq. (5) exactly as printed: h(t) = 1 - d - exp(a t^b)."""
    return 1 - d - np.exp(a * t**b)


eq5_values = pd.DataFrame([
    {"a": a, "b": b, "d": d,
     "h at t = 0": round(printed_health(0, a, b, d), 6),
     "h at t = 100": round(printed_health(100, a, b, d), 6)}
    for a, b, d in [(0.001, 1.4, 0.0), (0.003, 1.6, 0.01)]
])
```

</details>

<!-- sample: eq5_values -->
| a | b | d | h at t = 0 | h at t = 100 |
| --- | --- | --- | --- | --- |
| 0.001 | 1.4 | 0 | 0 | -0.879409 |
| 0.003 | 1.6 | 0.01 | -0.01 | -115.136438 |

For the gentlest setting, $h(100) = 1 - 0 - e^{0.001 \times 100^{1.4}}$. The
exponent is $0.001 \times 100^{1.4} = 0.630957$, so $h(100) = 1 - e^{0.630957}$,
which is $1 - 1.879409 = -0.879409$.
A health index meant to run from 1 down to 0 is negative from the first cycle.
The paper does not give a form that satisfies both its renamings and its stated
starting range, so the exact curve used to generate the data cannot be
reconstructed from it. This project never regenerates data, so nothing in it
depends on the answer.

**Equation 6** applies the same form separately to a module's **efficiency**
$e(t)$ and **flow** $f(t)$, each with its own initial degradation ($d_e$, $d_f$)
and its own growth parameters, because different faults move the two
differently (Saxena et al., 2008, p. 5):

$$e(t) = 1 - d_e - \exp\{a_e(t)\,t^{b_e(t)}\}, \qquad f(t) = 1 - d_f - \exp\{a_f(t)\,t^{b_f(t)}\} \tag{6}$$

Equation 6 writes the growth parameters as functions of time, $a_e(t)$ and
$b_e(t)$, while Equation 10 below treats them as constants drawn once per
engine (Saxena et al., 2008, pp. 5–6); the paper does not say which is
meant. What comes next is that $e(t)$
and $f(t)$ are fed into the simulator as the HPC's efficiency and flow, and the
engine's response decides when it has failed (Saxena et al., 2008,
pp. 5–6).

**Do not confuse the three kinds of health.** They have different ranges and
different roles:

| Quantity | Where it appears | Healthy value | Falls towards | Role |
|---|---|---|---|---|
| A health-parameter modifier, such as `HPC_eff_mod` | Table 1; guide Section 3 | 0 | negative values: −0.02 is a 2 percent loss | an input the simulator receives |
| $h(t)$, $e(t)$, $f(t)$ | Equations 4 to 6 | 1 | 0 | the paper's description of how one quantity wears over time |
| $H(t)$ | Equations 7 and 8, next | 1 | 0, the failure criterion | the engine's health, computed from its margins; never released |

The paper never says how a value of $e(t)$, which runs down from 1, becomes a
modifier value, which runs down from 0. Nor does it relate the starting values
$e_0$ and $f_0$ of Equation 9 to $d_e$ and $d_f$ of Equation 6: read naively,
$e_0 = 1 - d_e$, but the printed Equation 6 gives $e(0) = -d_e$, as the table
above shows for Equation 5.

### From module health to engine health: Equations 7 and 8

The engine fails when any one of its safety margins runs out, not when one
module's efficiency reaches some value (Saxena et al., 2008, p. 1).
Section V.D defines the **operability
margins** as how far the engine is operating from its limits, such as stall
and temperature limits. A few correspond to quantities a sensor reads — the
paper's examples are the limits on core speed and the upper limits on exhaust
gas temperature (EGT) — while the rest exist only inside the simulation
(Saxena et al., 2008, Section V.D,
[p. 6](../references/saxena_2008_damage_propagation_modeling.pdf#page=6)).
Equations 7 and 8 combine them into the engine's overall health index $H(t)$
(Saxena et al., 2008, p. 5):

$$H(t) = g\big(e(t), f(t)\big) \tag{7}$$

$$g\big(e(t), f(t)\big) = \min\left(m_{Fan},\, m_{HPC},\, m_{HPT},\, m_{EGT}\right) \tag{8}$$

- Each $m$ is one margin, computed by the simulator from the current $e(t)$ and
  $f(t)$, and rescaled to run from 1, perfectly healthy, to 0, where the margin
  has shrunk by its allowed limit (Saxena et al., 2008, pp. 5–6).
- $\min$ takes the smallest of them, so the margin closest to its limit sets the
  engine's health.

Section V.D gives the limits: a margin reaches 0 when the fan, LPC or HPC stall
margin has shrunk by 15 percent, or the EGT margin by about 2 percent
(Saxena et al., 2008, Section V.D, p. 6). If a
pairing of efficiency and flow would breach any margin at any flight condition
the engine might meet — a hot-day take-off, a maximum climb, a cruise — its
health index is 0; otherwise it is the lowest rescaled margin. The margins
also depend on how the engine is being flown: Figure 6 plots the three stall
margins and the EGT against the throttle setting, and the paper concludes that
the health index must be adjusted for the operating condition as well
(Saxena et al., 2008, Fig. 6, p. 6). The paper does not give the rescaling as
a formula, so no number can be worked through it. Its own example is in Figures 7 and 8: three degradation
trajectories drawn over the maps of HPC stall margin and EGT, where the EGT
margin reaches its limit first in all three, so EGT sets the health index
(Saxena et al., 2008, Figs. 7–8,
[p. 7](../references/saxena_2008_damage_propagation_modeling.pdf#page=7)).

Equation 8 names the fan, HPC, HPT and EGT margins (Saxena et al., 2008,
p. 5). Three other passages name a different set — fan, LPC and HPC stall
margins and EGT: Table 2's list of health-index parameters, the limits in
Section V.D, and item 1 of Section VI (Saxena et al., 2008, pp. 3, 6–7).
Table 2 contains no HPT margin at all. Three passages against one favour the
LPC reading, but the paper does not resolve it.

**Do not confuse the health index with remaining useful life.** $H(t)$ is a
number between 0 and 1 that the simulator computed and never released
(Saxena et al., 2008, Section VI, p. 7). Remaining useful life is a count of
cycles; in a training file it runs out where the record ends, which the
readme calls system failure (NASA PCoE, 2008) and which, if the files were
made by the method the paper describes, is where $H$ reached 0 (Saxena et al.,
2008, pp. 6–7). The model in this
project is shown only sensor readings and must infer the second without ever
seeing the first.

The key idea is that the engine's health is not one module's efficiency but
its weakest margin: the simulator turns the degraded efficiency and flow into
four margins, each rescaled to run from 1 to 0, and the smallest of them is
$H(t)$.

### Initial wear and manufacturing variation

No two real engines leave the factory identical, and an engine in service is
already slightly worn. The paper treats both as normal, not as faults
(Saxena et al., 2008, Section V.A,
[p. 5](../references/saxena_2008_damage_propagation_modeling.pdf#page=5)).
Its Table 3 recites reference values, which it attributes to Chatterjee and
Litt (its reference [9]), for how much a module's efficiency and flow change
with use, at the start and after 3000 and 6000 cycles; the largest initial
value is a 1.01 percent loss of LPC flow (Saxena et al., 2008, Table 3,
p. 5). The table has rows for the fan, LPC,
HPT and LPT, but none for the HPC, the module the challenge data degrade.

For the data, the starting efficiency and flow of each engine were drawn at
random within 1 percent of healthy (Saxena et al., 2008, Section V.B and
Eq. 9, p. 6):

$$e_0 \in [0.99,\, 1], \qquad f_0 \in [0.99,\, 1] \tag{9}$$

$e_0 = 0.995$, for example, is an engine that starts with half a percent less
efficiency than a perfect one. The readme repeats the consequence for users:
each engine starts with a different, unknown amount of initial wear and
manufacturing variation, which counts as normal and not as a fault
(NASA PCoE, 2008).

### Three kinds of noise

Section V.B separates three sources of noise and adds each at a different
stage (Saxena et al., 2008, Section V.B,
[pp. 5–6](../references/saxena_2008_damage_propagation_modeling.pdf#page=6)):

1. **Manufacturing and assembly variation** — the random starting values of
   Equation 9.
2. **Process noise** — variation in the degradation itself. Each engine's
   trajectory parameters $a$ and $b$ were drawn at random (the text says from
   a normal distribution), which with $e_0$ and $f_0$ fixes one smooth path
   for its efficiency and flow. That path was then disturbed by random noise
   drawn from a **mixture** of two distributions whose variances differ a
   little (the variance is the square of a distribution's typical spread): each
   disturbance is drawn from one of the two distributions, picked at random
   each time. The disturbed efficiency and flow were fed to the simulator, and
   the outputs were recorded once the engine had settled. Because the
   disturbance enters as a change of efficiency and flow, the simulator turns
   it into matching changes of all the outputs together: temperatures,
   pressures and speeds move as the engine model dictates. Section V adds that
   maintenance between flights is not modelled on its own but folded into this
   noise, so efficiency and flow are allowed to recover a little, within set
   bounds, at any time, and their loss is not steady from one flight to the
   next (Saxena et al., 2008, Section V, Fig. 5,
   [p. 5](../references/saxena_2008_damage_propagation_modeling.pdf#page=5)).
3. **Measurement noise** — random noise added to every output channel after
   the simulation, on top of what the engine model produced.

The paper's reason for the layering is that participants should face
de-noising tasks like those real data pose (Saxena et al., 2008, Section V.B,
p. 6). It gives no sizes for any of the
three: no variances, no amplitudes. Figure 5, one engine's efficiency loss over
50 cycles, is the only picture of how large the process noise is, so the noise
cannot be reproduced from the paper.

### The four-step recipe and its limits

Section V.C condenses the generation into four steps (Saxena et al., 2008,
p. 6):

1. Choose the starting efficiency and flow (Equation 9).
2. Let efficiency and flow fall exponentially, which makes the health index
   $H(t) = g(e(t), f(t))$ fall with time. The paper says the randomly chosen
   direction and evolution of the faults are bounded by Equation 10:

   $$f_i,\, e_i \le 1\%, \qquad a_k \in [0.001,\, 0.003], \qquad b_k \in [1.4,\, 1.6], \quad k = 1, 2 \tag{10}$$

3. Stop when $H = 0$. This is the failure criterion.
4. Add measurement noise to the outputs.

The **direction** of a fault is best seen in Figures 7 and 8, which draw each
trajectory as a path across a map whose two axes are efficiency loss and flow
loss: an engine that loses flow faster than efficiency heads one way across the
map, one that loses efficiency faster heads another, and the paper notes that
whether a given margin's threshold is crossed depends on that direction
(Saxena et al., 2008, Section V.D, p. 7). Different $a$ and $b$ for efficiency
and for flow are what set it (this page's reading of Equations 6 and 10). The rest of Equation 10 is left
unexplained: it gives bounded intervals while the text calls the distribution
of $a_k$ and $b_k$ normal (Saxena et al., 2008, Section V.B and Eq. 10,
p. 6); the subscript $k$ names the engine in the text but
takes the values 1 and 2 in Equation 10 (probably efficiency and flow); and
the index $i$ in $f_i, e_i \le 1\%$ is defined nowhere. Most plausibly that
bound repeats Equation 9's 1 percent of initial wear, but that too is a
reading.

The output is a time series of sensor readings, which the paper calls cruise
snapshots, produced by degrading **only the HPC's** efficiency and flow; the
paper states that degradation of other modules was left out of the challenge
data on purpose (Saxena et al., 2008, Section V.C, p. 6). FD001's files,
though, hold a single condition, the one the user's guide calls sea level,
static (Frederick et al., 2007, Fig. 4.6, p. 37), with the engine not moving
(see [Which flight conditions the settings are](#which-flight-conditions-the-settings-are)).
Section V gives the scenario
behind the snapshots (Saxena et al., 2008, Section V,
[p. 5](../references/saxena_2008_damage_propagation_modeling.pdf#page=5)):
each engine may fly a different flight condition from one flight to the next;
the damage done in one flight cannot be worked out from its duration and its
conditions alone, so it has to be read from the sensors; and because the
scenario models gradual wear rather than sudden faults, health changes little
within one flight, so one reading taken during or right after each flight
characterises it. How this project groups rows by flight condition is on
[Preprocessing, Find operating conditions](../learn/04_the_pipeline.md#find-operating-conditions).

The key idea is that each engine's HPC efficiency and flow fall along their
own randomly drawn exponential path from a randomly worn start, disturbed by
noise; the simulator turns them into sensor readings and four margins, and the
record stops at the flight where the smallest margin reaches zero.

How those readings look across FD001's 100 training engines, and how far the
sensors move together, is drawn under
[What the FD001 training engines look like](#what-the-fd001-training-engines-look-like).

---

## The challenge data the paper describes

Section VI lists the properties of the data generated for PHM'08
(Saxena et al., 2008, Section VI,
[p. 7](../references/saxena_2008_damage_propagation_modeling.pdf#page=7)):

1. HPC degradation, simulated under six combinations of altitude, throttle
   resolver angle and Mach number, with the fan, HPC and LPC stall margins and
   EGT used to decide when to stop.
2. Time series of the operational settings and the sensed variables, running
   from an undefined initial condition to the failure threshold. Participants
   never saw the health index and had to infer it.
3. A split into a **training** set, whose trajectories end at the failure
   threshold, and **test** and **validation** sets, cut off some time before it.
4. The true remaining lives at the cut-off ranged from 10 to 150 cycles in the
   test set and from 6 to 190 in the validation set. The wider validation range
   was deliberate and not announced, to test robustness. The sentence goes on
   to say that initial wear, noise levels and the spread of degradation
   parameters were kept as before, but as printed it lacks the negation its
   context needs and says they "remained changed" (Saxena et al., 2008,
   p. 7).

During the challenge, participants submitted predictions for the test set and
received a score back automatically, so they could improve their methods. So
that no method could be over-fitted to the test set — tuned to the feedback on
that one set — the validation set was held back, published later, and scored
without feedback until the competition had closed (Saxena et al., 2008,
Section VI, p. 7).

**Do not confuse the paper's validation set with this project's.** The paper's
validation set is a second, held-back set of cut-off engines with known answers
(Saxena et al., 2008, Section VI, p. 7).
This project's validation engines are training engines it holds back from
fitting, to decide when to stop training (assumptions A-06 and A-18;
[Preprocessing, step 3](../learn/04_the_pipeline.md#3--hold-engines-back)).
Neither is the test set, and validation error is not comparable with test error
in this project (assumption A-06).

---

## The four released subsets

The paper never names FD001, FD002, FD003 or FD004. Of the dataset's own
sources, only the readme does: it lists for each subset its training and test
trajectories, its operating conditions and its fault modes, and cites the
dataset paper as its reference
([NASA PCoE, 2008](../references/nasa_2008_cmapss_readme.txt)). Its description agrees
with the paper's method: a fleet of engines of one type, each with unknown
initial wear, three operational settings that change performance
substantially, sensor noise, and a fault that appears partway through each
record and grows. In the training files it grows until failure; in the test
files the record ends some time before, and a separate file holds the true
remaining life after each test engine's last recorded cycle (NASA PCoE, 2008).
Wen et al. (2026), the paper this project re-implements, also describe the
four subsets, in their Table 5
([p. 7](../references/wen_2026_estgnn_dynamic_shap.pdf#page=7)).

The row layout is shown on
[Data, Training engines](../learn/01_the_problem.md#training-engines). The
readme lists the 26 columns and names the last one sensor measurement 26
(NASA PCoE, 2008),
although columns 6 to 26 hold 21 sensors, so it is sensor 21.

### What the files contain, set beside the readme

<details class="process-example">
<summary>Example in Python</summary>

The readme's numbers are typed in as it prints them. Fault modes are shortened:
"HPC" for HPC degradation, "HPC, fan" for HPC and fan degradation.

```python
readme = {
    "FD001": (100, 100, 1, "HPC"),
    "FD002": (260, 259, 6, "HPC"),
    "FD003": (100, 100, 1, "HPC, fan"),
    "FD004": (248, 249, 6, "HPC, fan"),
}
everything = pd.concat([
    frame.assign(subset=name)
    for collection in (train, test) for name, frame in collection.items()
])
subset_counts = pd.DataFrame([
    {
        "subset": name,
        "training engines (readme)": readme[name][0],
        "training engines (files)": train[name].unit.nunique(),
        "test engines (readme)": readme[name][1],
        "test engines (files)": test[name].unit.nunique(),
        "conditions (readme)": readme[name][2],
        "conditions found": condition_count(everything[everything.subset == name]),
        "fault modes (readme)": readme[name][3],
    }
    for name in SUBSETS
])
assert all(len(answers[name]) == test[name].unit.nunique() for name in SUBSETS)
for name in SUBSETS:
    rows_of_subset = everything[everything.subset == name]
    altitude_alone = rounded_settings(rows_of_subset)["setting 1"].nunique()
    assert altitude_alone == condition_count(rows_of_subset)
largest_wobble = train["FD001"][["op_setting_1", "op_setting_2"]].abs().max()
assert largest_wobble.op_setting_1 == 0.0087 and largest_wobble.op_setting_2 == 0.0006
mach_wobble = (everything.op_setting_2 - everything.op_setting_2.round(2)).abs().max()
assert round(mach_wobble, 4) == 0.002
```

</details>

<!-- sample: subset_counts -->
| subset | training engines (readme) | training engines (files) | test engines (readme) | test engines (files) | conditions (readme) | conditions found | fault modes (readme) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FD001 | 100 | 100 | 100 | 100 | 1 | 1 | HPC |
| FD002 | 260 | 260 | 259 | 259 | 6 | 6 | HPC |
| FD003 | 100 | 100 | 100 | 100 | 1 | 1 | HPC, fan |
| FD004 | 248 | 249 | 249 | 248 | 6 | 6 | HPC, fan |

"Conditions found" counts distinct combinations of the three operational
settings after rounding; why rounding is needed is on
[Preprocessing, Find operating conditions](../learn/04_the_pipeline.md#find-operating-conditions).
The wobble is small: in the FD001 training file, setting 1 never strays more
than 0.0087 from zero and setting 2 never more than 0.0006. Settings 1 and 3
are rounded to whole numbers, which keeps distinct conditions apart because
they differ in setting 1 by at least 5. Setting 2 is too finely spaced for
that — its values 0.62, 0.70 and 0.84 would all round to 1 — so it is rounded
to two decimals, which is safe because in all eight files it never lies more
than 0.002 from a two-decimal value. Setting 1 alone already separates every
condition, and the code checks that it gives the same counts.

Every count agrees except FD004, where the readme swaps the two (NASA PCoE,
2008): the files hold
249 training engines and 248 test engines, and `RUL_FD004.txt` has 248 lines,
one per test engine. Wen et al. (2026, Table 5, p. 7) give 249 and 248, as the
files do.
The project's record of the expected sizes, `EXPECTED_SHAPES` in
`data/loading/`, also follows the files.

In what follows, an engine's **life** means the number of cycles from its
first record to failure. It is not the engine's age since it was built: the
paper starts each record at an arbitrary, already worn point (Saxena et al.,
2008, p. 5, and Section VI, item 2, p. 7).

<details class="process-example">
<summary>Example in Python</summary>

```python
subset_lengths = pd.DataFrame([
    {
        "subset": name,
        "training rows": len(train[name]),
        "test rows": len(test[name]),
        "shortest training life": train[name].groupby("unit").cycle.max().min(),
        "longest training life": train[name].groupby("unit").cycle.max().max(),
        "shortest test record": test[name].groupby("unit").cycle.max().min(),
        "longest test record": test[name].groupby("unit").cycle.max().max(),
        "smallest true RUL": answers[name].min(),
        "largest true RUL": answers[name].max(),
    }
    for name in SUBSETS
])
```

</details>

<!-- sample: subset_lengths -->
| subset | training rows | test rows | shortest training life | longest training life | shortest test record | longest test record | smallest true RUL | largest true RUL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FD001 | 20631 | 13096 | 128 | 362 | 31 | 303 | 7 | 145 |
| FD002 | 53759 | 33991 | 128 | 378 | 21 | 367 | 6 | 194 |
| FD003 | 24720 | 16596 | 145 | 525 | 38 | 475 | 6 | 145 |
| FD004 | 61249 | 41214 | 128 | 543 | 19 | 486 | 6 | 195 |

A training engine's life is its last recorded cycle, since its record runs to
failure (NASA PCoE, 2008). A test engine's record is how many cycles were kept
before the cut.

**These files are not the challenge data the paper describes.** Three
differences say so:

- The paper's challenge data degrade only the HPC and leave other modules out
  on purpose (Saxena et al., 2008, Section V.C, p. 6); the readme gives FD003
  and FD004 a second fault mode, fan degradation, which the paper never
  describes (NASA PCoE, 2008).
- The paper's test answers ranged from 10 to 150 cycles (Saxena et al., 2008,
  Section VI, p. 7). No released RUL file
  stays inside that range: all four reach below 10, and FD002 and FD004 reach
  194 and 195, above even the validation set's 190.
- FD001 and FD003 have a single flight condition (next section), while the
  paper's challenge data used six (Saxena et al., 2008, Section VI, p. 7).

The paper therefore documents the simulator and the method that produced data
of this kind. For what the four files themselves contain, the readme and the
files are the only direct sources.

### Which flight conditions the settings are

Neither the paper nor the readme says which setting is which. The paper says
the challenge's six conditions combine altitude (0 to 42,000 ft), Mach number
(0 to 0.84) and TRA (20 to 100) (Saxena et al., 2008, Section V.D, p. 6), and
the user guide's
Table 1.4 lists 14 ready-made flight conditions with their altitude, Mach
number, TRA and the fan speed the engine settles at (Frederick et al., 2007,
[Table 1.4, p. 11](../references/frederick_2007_cmapss_users_guide.pdf#page=11)).
Finding the conditions in the files and looking them up in that table tests
both the settings and the sensor order at once:

<details class="process-example">
<summary>Example in Python</summary>

The guide's values are typed in from its Table 1.4, with altitude in thousands
of feet.

```python
guide = {  # (altitude, Mach, TRA) -> (menu name, fan speed in rpm)
    (0, 0.0, 100): ("FC01", 2388),
    (10, 0.25, 100): ("FC05", 2319),
    (20, 0.70, 100): ("FC06", 2324),
    (25, 0.62, 60): ("FC07", 1915),
    (35, 0.84, 100): ("FC08", 2223),
    (42, 0.84, 100): ("FC09", 2212),
}
keys = rounded_settings(everything)
rows = []
for (altitude, mach, tra), group in everything.groupby(
    [keys["setting 1"], keys["setting 2"], keys["setting 3"]]
):
    demanded = group.sensor_18.unique()
    assert len(demanded) == 1
    name, fan_speed = guide[(altitude, mach, tra)]
    rows.append({
        "setting 1": altitude, "setting 2": mach, "setting 3": tra,
        "subsets": ", ".join(sorted(group.subset.unique())),
        "sensor 18 in every row": demanded[0],
        "guide condition": name, "guide fan speed": fan_speed,
    })
flight_conditions = pd.DataFrame(rows)
assert symbol["sensor_18"] == "Nf_dmd"
assert (flight_conditions["sensor 18 in every row"] == flight_conditions["guide fan speed"]).all()
assert sorted(everything.op_setting_3.unique()) == [60.0, 100.0]
assert sorted(everything.sensor_19.unique()) == [84.93, 100.0]
```

</details>

<!-- sample: flight_conditions -->
| setting 1 | setting 2 | setting 3 | subsets | sensor 18 in every row | guide condition | guide fan speed |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 100 | FD001, FD002, FD003, FD004 | 2388 | FC01 | 2388 |
| 10 | 0.25 | 100 | FD002, FD004 | 2319 | FC05 | 2319 |
| 20 | 0.7 | 100 | FD002, FD004 | 2324 | FC06 | 2324 |
| 25 | 0.62 | 60 | FD002, FD004 | 1915 | FC07 | 1915 |
| 35 | 0.84 | 100 | FD002, FD004 | 2223 | FC08 | 2223 |
| 42 | 0.84 | 100 | FD002, FD004 | 2212 | FC09 | 2212 |

Each of the six combinations in the files is one of the guide's menu
conditions, read as altitude in thousands of feet, Mach number and TRA in
degrees. Within each condition, sensor 18 holds one value across every row of
all eight files, and that value is exactly the guide's fan speed for the
condition. Sensor 18 is the 18th row of the paper's Table 2, the **demanded fan
speed** (Saxena et al., 2008, Table 2), and the fan-speed controller holds the
real fan speed at that demand (Frederick et al., 2007, Section 3.1, p. 29). A
column that never moves within a condition and equals the guide's settled fan speed is what the demanded fan speed should
look like, so the check supports reading the 21 sensor columns in Table 2's
order. It is evidence, not a statement from either source.

The key idea is that one check supports two readings at once: taking the
settings as altitude, Mach number and TRA puts every row in one of the guide's
menu conditions, and in each of those conditions the 18th sensor column equals
the fan speed the guide gives, which is what Table 2's 18th sensor, the
demanded fan speed, should do.

Four more observations come out of the same rows:

- FD001's only condition is the guide's FC01: altitude 0, Mach 0 and TRA 100,
  the highest setting in the guide's menu (Frederick et al., 2007, Table 1.4,
  p. 11). The guide's own menu labels FC01 as sea level, static (the engine is
  not moving), standard day (Frederick et al., 2007, Fig. 4.6,
  [p. 37](../references/frederick_2007_cmapss_users_guide.pdf#page=37)),
  and the readme describes the single condition of FD001 and FD003 as sea
  level (NASA PCoE, 2008). Section V.C of the paper calls the outputs cruise
  snapshots, although its own ranges in Section V.D start at altitude 0 and
  Mach 0 (Saxena et al., 2008, p. 6); the paper does not reconcile the two.
- The files' TRA takes only the values 60 and 100, not the range 20 to 100 the
  paper gives (Saxena et al., 2008, Section V.D, p. 6).
- The highest condition, FC09, is at 42,000 ft, above the 40,000 ft both
  sources give as the simulator's range (Frederick et al., 2007, p. 8; Saxena
  et al., 2008, p. 2), yet it is the guide's own menu condition (Frederick et
  al., 2007, Table 1.4, p. 11) and inside the paper's 0 to 42,000 ft (Saxena et
  al., 2008, p. 6); neither source comments.
- The paper's Table 2 gives PCNfR_dmd (sensor 19) in rpm, the guide's
  Table 1.3 in percent (Frederick et al., 2007, Table 1.3, p. 11; Saxena et
  al., 2008, Table 2), and the files hold only 100 and 84.93 for it — values
  that read as percentages.

### One training engine's record

A training record runs from an engine's first recorded cycle, already worn by
the paper's design, to system failure (NASA PCoE, 2008), which in the method
the paper describes is the cycle where the health index reached zero
(Saxena et al., 2008, pp. 6–7). The
shortest-lived and the longest-lived training engines of FD001 show what the
method of the previous section produces. The table averages five sensors over
each engine's first 20 and last 20 recorded cycles; averaging 20 readings
damps the measurement noise enough to see the drift. T30, P30 and Ps30 are
measured at the HPC outlet, the module being degraded, T50 at the LPT outlet,
and Nf is the fan speed the controller holds (Frederick et al., 2007, p. 29;
Saxena et al., 2008, Table 2). The last two columns measure the
longer-lived engine's change in two ways: as a percentage of where it
started, and against the sensor's own scatter, its standard deviation (the
typical distance from the average) over all FD001 training rows.

<details class="process-example">
<summary>Example in Python</summary>

```python
fd001 = train["FD001"]
lives = fd001.groupby("unit").cycle.max()
short_engine, long_engine = lives.idxmin(), lives.idxmax()
watched = ["sensor_3", "sensor_7", "sensor_11", "sensor_4", "sensor_8"]


def ends(unit, column):
    """Average of one sensor over an engine's first 20 and last 20 recorded cycles."""
    history = fd001[fd001.unit == unit][column]
    return history.head(20).mean(), history.tail(20).mean()


rows = []
for column in watched:
    short_start, short_finish = ends(short_engine, column)
    long_start, long_finish = ends(long_engine, column)
    change = long_finish - long_start
    rows.append({
        "sensor": symbol[column],
        f"engine {short_engine}, first 20": round(short_start, 4),
        f"engine {short_engine}, last 20": round(short_finish, 4),
        f"engine {long_engine}, first 20": round(long_start, 4),
        f"engine {long_engine}, last 20": round(long_finish, 4),
        f"engine {long_engine}, change as % of start": round(100 * change / long_start, 6),
        f"engine {long_engine}, change ÷ standard deviation": round(change / fd001[column].std(), 6),
    })
two_lives = pd.DataFrame(rows)
assert (short_engine, long_engine, lives.min(), lives.max()) == (39, 69, 128, 362)
t50_39, t50_69 = ends(39, "sensor_4"), ends(69, "sensor_4")
assert round(t50_39[0] - t50_69[0], 3) == 4.739
assert 0 < t50_69[1] - t50_39[1] < 1
```

</details>

Engine 39's record runs 128 cycles to failure and engine 69's runs 362:

<!-- sample: two_lives -->
| sensor | engine 39, first 20 | engine 39, last 20 | engine 69, first 20 | engine 69, last 20 | engine 69, change as % of start | engine 69, change ÷ standard deviation |
| --- | --- | --- | --- | --- | --- | --- |
| T30 | 1589.9785 | 1600.1485 | 1586.6575 | 1600.627 | 0.880436 | 2.278447 |
| P30 | 553.3085 | 551.734 | 553.9655 | 551.4055 | -0.462123 | -2.892354 |
| Ps30 | 47.515 | 48.0675 | 47.3545 | 48.103 | 1.580631 | 2.802453 |
| T50 | 1407.1295 | 1426.6425 | 1402.3905 | 1427.5935 | 1.797146 | 2.800145 |
| Nf | 2388.119 | 2388.2735 | 2388.0455 | 2388.2575 | 0.008878 | 2.986526 |

Read across one row. T50 on engine 39 rises from 1407.1295 to 1426.6425 °R, an
increase of 19.513; on engine 69 from 1402.3905 to 1427.5935, an increase of
25.203. The two engines **start** 4.739 °R apart, which is consistent with the
differing initial wear the paper builds in, though two engines cannot separate
it from leftover measurement noise or from wear accrued within those first 20
cycles. They **finish** less than 1 °R apart, although one took almost three
times as long to get there. That is the shape the recipe predicts: random
starting points, random rates, one common stopping rule.

The fan speed shows the closed loop at work. In percent, engine 69's Nf moves
by under a hundredth of a percent while its T50 moves by almost two: the
controller keeps the fan within a fraction of an rpm of its demand. Measured
against its own scatter, though, the small drift of Nf is as large as T50's,
about three standard deviations, because the fan speed scatters as little as
it drifts. So
damage barely moves the level of the controlled fan speed, but it does not
leave it untouched.

Two engines are an anecdote, so the next table measures the same thing over all
100 training engines: the standard deviation of the first-20-cycle averages,
and of the last-20-cycle averages.

<details class="process-example">
<summary>Example in Python</summary>

```python
starts = fd001.groupby("unit")[watched].apply(lambda h: h.head(20).mean())
finishes = fd001.groupby("unit")[watched].apply(lambda h: h.tail(20).mean())
spread = pd.DataFrame({
    "sensor": [symbol[column] for column in watched],
    "spread of first-20 averages": starts.std().round(6).to_numpy(),
    "spread of last-20 averages": finishes.std().round(6).to_numpy(),
})
```

</details>

<!-- sample: spread -->
| sensor | spread of first-20 averages | spread of last-20 averages |
| --- | --- | --- |
| T30 | 2.468276 | 1.404379 |
| P30 | 0.483843 | 0.269373 |
| Ps30 | 0.149518 | 0.05591 |
| T50 | 4.500587 | 1.691117 |
| Nf | 0.046462 | 0.049977 |

For the four sensors on the gas path (the air's route through the engine) the
engines disagree more at the start than at the end, as random starts and one common stopping rule would produce; with one
subset and a handful of sensors this is consistent with the paper's design,
not a proof of it. The fan speed, held by the controller, does not show the
pattern. Nothing in a training file marks the cycle where the fault begins:
the readme says only that it appears partway through each record
(NASA PCoE, 2008).

### One test engine's truncation

How the answer file completes a test record is on
[Data, Test engines](../learn/01_the_problem.md#test-engines). Adding an
engine's recorded cycles and its true remaining life gives the life it would
have had. Three FD001 test engines span the range — the one closest to
failure, the one furthest from it, and the one with the longest record:

<details class="process-example">
<summary>Example in Python</summary>

```python
fd001_test = test["FD001"]
recorded = fd001_test.groupby("unit").cycle.max()
remaining = answers["FD001"].loc[recorded.index]
implied = recorded + remaining
chosen = [remaining.idxmin(), remaining.idxmax(), recorded.idxmax()]
truncation = pd.DataFrame({
    "test engine": chosen,
    "cycles recorded": recorded.loc[chosen].to_numpy(),
    "true RUL after the last record": remaining.loc[chosen].to_numpy(),
    "implied total life": implied.loc[chosen].to_numpy(),
    "share of life recorded": (recorded / implied).loc[chosen].round(6).to_numpy(),
})
life_comparison = pd.DataFrame([
    {"engines": "training: run to failure", "count": len(lives), "shortest life": lives.min(),
     "longest life": lives.max(), "average life": round(lives.mean(), 6)},
    {"engines": "test: recorded + true RUL", "count": len(implied), "shortest life": implied.min(),
     "longest life": implied.max(), "average life": round(implied.mean(), 6)},
])
share = recorded / implied
assert (round(share.min(), 6), round(share.max(), 6)) == (0.206704, 0.966667)
```

</details>

<!-- sample: truncation -->
| test engine | cycles recorded | true RUL after the last record | implied total life | share of life recorded |
| --- | --- | --- | --- | --- |
| 34 | 203 | 7 | 210 | 0.966667 |
| 25 | 48 | 145 | 193 | 0.248705 |
| 49 | 303 | 21 | 324 | 0.935185 |

For engine 34: 203 cycles recorded, 7 left, so it would have failed at cycle
203 + 7 = 210, and the record shows 203 / 210 = 0.966667 of its life. Over all
100 FD001 test engines the share recorded runs from 0.206704 to 0.966667, so the
cut falls anywhere from a fifth of the way through a life to its last few
cycles. The readme says no more about where the test records stop than that
each one ends some time before failure (NASA PCoE, 2008).

Set beside the training engines, the test engines' implied lives look like
lives drawn from the same fleet:

<!-- sample: life_comparison -->
| engines | count | shortest life | longest life | average life |
| --- | --- | --- | --- | --- |
| training: run to failure | 100 | 128 | 362 | 206.31 |
| test: recorded + true RUL | 100 | 141 | 341 | 206.48 |

---

## What the FD001 training engines look like

The tables above count the engines and give their shortest, longest and
average lives. They do not show three things that matter for any model built
on these files: how long each engine lived, how each sensor moves on the way
to failure, and how far the sensors move together. The third matters most for
this project. The paper it re-implements joins its 14 sensors in a graph whose
entries are set to 1 "for strong physical relationships between sensor pairs
and 0 otherwise" (Wen et al., 2026, Section 2.3.1,
[p. 5](../references/wen_2026_estgnn_dynamic_shap.pdf#page=5)), and it reads
its heatmap of the sensors' correlations as showing such relationships (Wen et
al., 2026, Section 3.2.2,
[p. 8](../references/wen_2026_estgnn_dynamic_shap.pdf#page=8)).

This section draws all three from the 100 engines of `train_FD001.txt`; it
reads no test file and no answer file. Unlike the rest of this page, it is not
the sources' account: the charts, the measurements and how they are read are
this project's, and each source is cited where it is used.

### How long the engines lived

{{figure:lifetimes}}

Each dot is one training engine, placed at its life: the number of cycles from
its first record to its last, the cycle at which it failed (NASA PCoE, 2008).
Dots that would overlap are stacked. Point at a dot to see which engine it is.

<details class="process-example">
<summary>Example in Python</summary>

`lives` holds each engine's life, from
[One training engine's record](#one-training-engines-record) above.

```python
life_bands = (
    lives.groupby(pd.cut(lives, bins=range(100, 401, 50), right=False), observed=False)
    .size()
    .rename_axis("life in cycles")
    .reset_index(name="engines")
)
life_bands["life in cycles"] = [f"{band.left} to {band.right - 1}" for band in life_bands["life in cycles"]]
assert life_bands.engines.sum() == 100 and lives.median() == 199
assert (lives.min(), lives.max(), (lives <= 199).sum()) == (128, 362, 52)
```

</details>

<!-- sample: life_bands -->
| life in cycles | engines |
| --- | --- |
| 100 to 149 | 6 |
| 150 to 199 | 46 |
| 200 to 249 | 31 |
| 250 to 299 | 13 |
| 300 to 349 | 3 |
| 350 to 399 | 1 |

Lives run from 128 to 362 cycles, and half the engines had failed by cycle
199. The longest life is 2.8 times the shortest, in engines of one type
recorded under one operating condition (NASA PCoE, 2008). The dataset paper's
scenario expects a spread like this: the amount and rate of damage differ from
engine to engine (Saxena et al., 2008, Section V, p. 5), and for the data it
generated it drew each engine's starting efficiency and flow, and the
parameters $a$ and $b$ of its path of wear, at random (Saxena et al., 2008,
Section V.B, p. 6; see
[Initial wear and manufacturing variation](#initial-wear-and-manufacturing-variation)
and [Three kinds of noise](#three-kinds-of-noise)). That FD001 was made the
same way is this page's reading: the readme names that paper as its reference
but describes only the starting wear (NASA PCoE, 2008).

The shortest life, 128 cycles, also sets the span of the next figure: every
training engine has at least 128 recorded cycles.

### How each sensor moves on the way to failure

{{figure:sensor-trends}}

Each panel is one of the 14 sensors the paper keeps, in the order of its
Table 3 (Wen et al., 2026, Table 3,
[p. 6](../references/wen_2026_estgnn_dynamic_shap.pdf#page=6)), drawn over the
last 128 cycles of every training engine. The engines are lined up at their
failure, so the horizontal axis counts cycles before failure, from 127 at the
left to 0, the last recorded cycle, at the right. At each of those 128 places
the 100 engines give 100 readings. The line is their median, the value with
half the readings above it and half below. The shaded band runs from the 10th
to the 90th percentile, the values below which a tenth and nine tenths of the
readings lie, so it holds the middle 80 or so engines. The thin grey line is
engine 1 alone, one flight after another. Point at a panel for its numbers.

The table counts, for each sensor, the engines on which its last 20 readings
average higher than its first 20, lower, or exactly the same: the comparison
[One training engine's record](#one-training-engines-record) makes for two
engines, here made for all 100.

<details class="process-example">
<summary>Example in Python</summary>

`readings` keeps the 14 sensors in the paper's order, named by their symbols;
the later examples in this section use it too. `cycles_left` counts each row's
cycles before its engine's last record.

```python
from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_SYMBOLS, symbol_to_column

readings = fd001[[symbol_to_column(name) for name in PAPER_SENSOR_SYMBOLS]].set_axis(
    list(PAPER_SENSOR_SYMBOLS), axis=1)
by_engine = readings.assign(unit=fd001.unit).groupby("unit")
drift = by_engine.tail(20).groupby("unit").mean() - by_engine.head(20).groupby("unit").mean()
directions = pd.DataFrame({
    "sensor": readings.columns,
    "engines where it rises": (drift > 0).sum().to_numpy(),
    "engines where it falls": (drift < 0).sum().to_numpy(),
    "engines where it stays level": (drift == 0).sum().to_numpy(),
})
cycles_left = fd001.groupby("unit").cycle.transform("max") - fd001.cycle
median_path = readings[cycles_left <= 127].groupby(cycles_left).median()
earlier = median_path.loc[64] - median_path.loc[127]
later = median_path.loc[0] - median_path.loc[64]
steady = [name for name in readings.columns if name not in ("P15", "Nc")]
assert (later[steady].abs() > 2 * earlier[steady].abs()).all()
assert (round(earlier.T50, 3), round(later.T50, 3)) == (4.56, 21.575)
assert sorted(readings.P15.unique()) == [21.6, 21.61] and (readings.P15 == 21.6).sum() == 406
never_change = sorted(symbol[column] for column in symbol if fd001[column].nunique() == 1)
assert never_change == sorted(["T2", "P2", "epr", "farB", "Nf_dmd", "PCNfR_dmd"])
nc_with_nrc = fd001[symbol_to_column("Nc")].corr(fd001[symbol_to_column("NRc")])
assert fd001[symbol_to_column("NRc")].nunique() == 6078 and round(nc_with_nrc, 6) == 0.963157
```

</details>

<!-- sample: directions -->
| sensor | engines where it rises | engines where it falls | engines where it stays level |
| --- | --- | --- | --- |
| T24 | 100 | 0 | 0 |
| T30 | 100 | 0 | 0 |
| T50 | 100 | 0 | 0 |
| P15 | 40 | 0 | 60 |
| P30 | 0 | 100 | 0 |
| Nf | 100 | 0 | 0 |
| Nc | 71 | 29 | 0 |
| Ps30 | 100 | 0 | 0 |
| phi | 0 | 100 | 0 |
| NRf | 100 | 0 | 0 |
| BPR | 100 | 0 | 0 |
| htBleed | 100 | 0 | 0 |
| W31 | 0 | 100 | 0 |
| W32 | 0 | 100 | 0 |

Twelve of the 14 sensors drift the same way on every engine: T24, T30, T50,
Nf, Ps30, NRf, BPR and htBleed rise, and P30, phi, W31 and W32 fall. The drift
is slow at first and quickens towards failure: for each of the twelve, the
median moves more than twice as far over the last 64 cycles as over the 64
before them. T50, for example, rises 4.56 °R and then 21.575 °R. A loss of
efficiency and flow that grows exponentially, as the dataset paper models it,
would bend this way ([Why an exponential](#why-an-exponential)), but the files
record neither efficiency nor flow, so joining the two is this page's reading.
Around the drift, every reading scatters from one flight to the next, as
engine 1's grey line shows; how large that scatter is beside the drift is
measured below.

Two sensors break the pattern. **P15** is recorded at two values only, 21.60
and 21.61 psia, and 406 of the 20,631 rows read 21.60, so on 60 engines its
first and last 20 readings average exactly the same; a sensor that can take
two values carries almost no information about wear (finding F-01). **Nc**
rises on 71 engines and falls on 29, which is why its band widens towards
failure.

The other seven of the 21 sensors are not drawn. Six of them, T2, P2, epr,
farB, Nf_dmd and PCNfR_dmd, never change in this file. The seventh, NRc, takes
6,078 different values, but it moves almost exactly with Nc: the two correlate
at 0.963157 over the file (how a correlation is measured is the next part's
subject). The paper says seven sensors are removed, names six, and gives its
rule as the removal of "features with high correlational redundancy (above
0.95) and constant values" (Wen et al., 2026, Section 3.2.2, p. 8), so this
project reads NRc as the seventh (assumption A-10). How it chooses its 14 is
taught on
[Data, Select 14 sensor columns](../learn/01_the_problem.md#3--select-14-sensor-columns-for-model-inputs).

### How far the sensors move together

**What a correlation is.** Wen et al. measure how two variables move together
with the Pearson correlation coefficient (Wen et al., 2026, Section 3.2.2,
Eq. 17, p. 8):

$$r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\,\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}} \tag{17}$$

They define $x_i$ and $y_i$ as the $i$-th observed values of the two
variables, $\bar{x}$ and $\bar{y}$ as their means, and $n$ as the sample size
(Wen et al., 2026, p. 8). Here the two variables are two sensors and one
observation is one row of the file, so $n$ is 20,631. How to read the formula
is this page's explanation, not the paper's:

- The top multiplies, row by row, how far one reading lies from its average by
  how far the other does. A product is positive when both readings lie on the
  same side of their averages and negative when they lie on opposite sides,
  so the sum leans positive when the two sensors tend to be high together, and
  negative when one tends to be high while the other is low.
- The bottom divides by each sensor's own spread, which removes the units and
  keeps $r$ between −1 and +1.
- At +1 every row lies on one rising straight line, at −1 on one falling line.
  At 0 the straight line that best fits the points is flat: knowing one
  reading does not help to predict the other along a straight line.

Three real flights show the arithmetic: T50 and Ps30 over engine 1's first
three cycles.

<details class="process-example">
<summary>Example in Python</summary>

```python
three = readings.loc[fd001.unit == 1, ["T50", "Ps30"]].head(3)
from_mean = three - three.mean()
by_hand = pd.DataFrame({
    "cycle": [1, 2, 3],
    "T50": three.T50.to_numpy(),
    "Ps30": three.Ps30.to_numpy(),
    "T50 minus its average": from_mean.T50.round(6).to_numpy(),
    "Ps30 minus its average": from_mean.Ps30.round(6).to_numpy(),
    "product": (from_mean.T50 * from_mean.Ps30).round(6).to_numpy(),
})
r_three = (from_mean.T50 * from_mean.Ps30).sum() / np.sqrt(
    (from_mean.T50 ** 2).sum() * (from_mean.Ps30 ** 2).sum())
assert round(r_three, 6) == round(three.T50.corr(three.Ps30), 6) == -0.668256
assert round(readings.T50.corr(readings.Ps30), 6) == 0.830136
```

</details>

<!-- sample: by_hand -->
| cycle | T50 | Ps30 | T50 minus its average | Ps30 minus its average | product |
| --- | --- | --- | --- | --- | --- |
| 1 | 1400.6 | 47.47 | -2.046667 | 0.06 | -0.1228 |
| 2 | 1403.14 | 47.49 | 0.493333 | 0.08 | 0.039467 |
| 3 | 1404.2 | 47.27 | 1.553333 | -0.14 | -0.217467 |

The averages are 1402.646667 °R and 47.41 psia. The three products add up to
−0.3008. The squared distances from the averages add up to 6.845067 for T50
and 0.0296 for Ps30, so

$$r = \frac{-0.3008}{\sqrt{6.845067 \times 0.0296}} = \frac{-0.3008}{0.450127} = -0.6683$$

which the code gives as −0.668256 before rounding. Over these three flights
the two sensors moved in opposite directions. Over all 20,631 rows of the file
the same two sensors correlate at +0.830136. Which of the two numbers says
something about T50 and Ps30, and what it says, is the question the rest of
this part answers.

{{figure:sensor-correlations}}

The left table holds the correlation of every pair of the 14 sensors over all
20,631 rows of the training file, the 100 engines pooled. Wen et al.'s Fig. 5
draws the same table, with four more rows and columns for the engine number,
the cycle and two of the operating settings (Wen et al., 2026, Fig. 5, p. 8),
and all 91 of its pairs of sensors agree with the left table to the two
decimals it prints (finding F-23). The
right table correlates something else: for every sensor, the change in its
reading since the same engine's previous flight. Red means that two sensors
move in the same direction, blue in opposite directions, and the grey middle
that they have no straight-line relation; the stronger the colour, the larger
the correlation. Each sensor correlates perfectly with itself, so the diagonal
is +1 in both. Point at a square for its value.

<details class="process-example">
<summary>Example in Python</summary>

`changes` holds, for every row but each engine's first, the change in each
reading since that engine's previous cycle.

```python
changes = readings.groupby(fd001.unit).diff().dropna()
upper = np.triu_indices(14, 1)
together = pd.DataFrame([
    {
        "what is correlated": what,
        "rows": len(table),
        "average size over the 91 pairs": round(np.abs(table.corr().to_numpy()[upper]).mean(), 6),
        "largest size": round(np.abs(table.corr().to_numpy()[upper]).max(), 6),
    }
    for what, table in [("the readings", readings), ("the change since the previous flight", changes)]
])
```

</details>

<!-- sample: together -->
| what is correlated | rows | average size over the 91 pairs | largest size |
| --- | --- | --- | --- |
| the readings | 20631 | 0.569119 | 0.846884 |
| the change since the previous flight | 20531 | 0.011354 | 0.047887 |

The size of a correlation is its value without its sign. Over whole lives it
averages 0.569119 over the 91 pairs. From one flight to the next it averages
0.011354, and the largest, Nf against NRf, is 0.047887: from one flight to the
next, every sensor moves on its own.

**Why the two tables differ.** Think of each reading as two parts added
together. This is this page's model of the data, not the paper's: the dataset
paper names three kinds of noise but gives the size of none of them
([Three kinds of noise](#three-kinds-of-noise)).

- A **slow part**, the drift of the figure above, which wear moves a little at
  each flight and a long way over a life.
- A **fast part**, the scatter, drawn afresh at every flight.

From one flight to the next the slow part barely moves, so a change since the
previous flight is almost all fast part, and the right table says that the
fast parts of different sensors are unrelated. Over a whole life the slow
parts move a long way, and they move together, because one fault drives them
all: FD001 has a single fault mode, HPC degradation (NASA PCoE, 2008). The
left table sees both parts at once. The shared slow parts pull every
correlation towards +1 or −1, and each sensor's own scatter pulls it back
towards 0.

The right table also says something about the dataset paper's noise. Of its
three kinds, the process noise disturbs the engine's efficiency and flow, so
the simulator passes it on to all the outputs together, while the measurement
noise is added to each output channel on its own (Saxena et al., 2008,
Section V.B, p. 6). If the process noise moved the readings much from one
flight to the next, the changes of different sensors would correlate. They do
not, so the scatter seen here behaves like separate noise on each sensor. That
is this page's reading of the right table; neither source says how large
either kind of noise is.

**How large the scatter is.** Suppose the scatter of one flight is unrelated
to the scatter of the next. Then the change from one flight to the next is,
apart from the slow part's small step, the difference of two unrelated draws.
The variance of a quantity is the square of its typical spread, and the
variance of a sum or a difference of two unrelated draws is the sum of their
variances. So the variance of the change is twice the variance of the scatter,
and the scatter's variance is half the change's.

The supposition can be checked. Two neighbouring changes share one draw with
opposite signs: the change into a flight ends on that flight's draw, and the
change out of it starts from the same draw. If nothing else links them, one
change correlates with the next at exactly −0.5. For all 14 sensors it lies
between −0.510 and −0.487.

The **share of the slow drift** is the fraction of a sensor's variance that is
left once the scatter's is taken away:

$$s = 1 - \frac{\sigma_{\text{scatter}}^2}{\sigma_{\text{reading}}^2}$$

Here $\sigma_{\text{reading}}$ is the standard deviation of the sensor's
readings over the whole file, their typical distance from their average, and
$\sigma_{\text{scatter}}$ is the standard deviation of its scatter. For T50
the readings spread 9.000605 °R and the scatter 4.030060 °R, so
$s = 1 - (4.030060 / 9.000605)^2$, which is $0.799516$: four fifths of T50's
variance over the file is drift, and one fifth is flight-to-flight scatter.

<details class="process-example">
<summary>Example in Python</summary>

`previous_change` holds, beside each change, the change one cycle earlier on
the same engine.

```python
previous_change = changes.groupby(fd001.unit.loc[changes.index]).shift(1)
noise_shares = pd.DataFrame({
    "sensor": readings.columns,
    "spread of the readings": readings.std().round(6).to_numpy(),
    "spread of the scatter": np.sqrt(changes.var() / 2).round(6).to_numpy(),
    "share of the slow drift": (1 - changes.var() / 2 / readings.var()).round(6).to_numpy(),
    "correlation of one change with the next": changes.corrwith(previous_change).round(6).to_numpy(),
})
step = noise_shares["correlation of one change with the next"]
assert step.min() > -0.511 and step.max() < -0.486
```

</details>

<!-- sample: noise_shares -->
| sensor | spread of the readings | spread of the scatter | share of the slow drift | correlation of one change with the next |
| --- | --- | --- | --- | --- |
| T24 | 0.500053 | 0.301939 | 0.63541 | -0.496507 |
| T30 | 6.13115 | 4.000488 | 0.574262 | -0.496037 |
| T50 | 9.000605 | 4.03006 | 0.799516 | -0.501567 |
| P15 | 0.001389 | 0.001334 | 0.07721 | -0.486944 |
| P30 | 0.885092 | 0.408235 | 0.787263 | -0.50991 |
| Nf | 0.070985 | 0.030266 | 0.818213 | -0.488189 |
| Nc | 22.08288 | 4.174574 | 0.964263 | -0.494223 |
| Ps30 | 0.267087 | 0.101538 | 0.855471 | -0.501256 |
| phi | 0.737553 | 0.300201 | 0.834332 | -0.498547 |
| NRf | 0.071919 | 0.030417 | 0.821122 | -0.486982 |
| BPR | 0.037505 | 0.020175 | 0.710637 | -0.506946 |
| htBleed | 1.548763 | 0.946681 | 0.626374 | -0.495401 |
| W31 | 0.180746 | 0.100997 | 0.68777 | -0.501082 |
| W32 | 0.108251 | 0.060272 | 0.689992 | -0.502489 |

**What the scatter does to a correlation.** Write one sensor's reading as
$x = a + e$, with $a$ its slow part and $e$ its scatter, and another's as
$y = b + f$. Put these into Equation 17.

- In the top, each row's product becomes four products: the slow parts'
  product, and three that contain a scatter term. Because $e$ and $f$ are
  unrelated to each other and to both slow parts, those three add up to about
  zero over many rows, so the top is the slow parts' own top.
- In the bottom, each sum of squares becomes the slow part's plus the
  scatter's, for the same reason. By the definition of the share of slow
  drift, the slow part's sum of squares is the fraction $s$ of the whole. So
  the whole is the slow part's divided by $s$, and the bottom is the slow
  parts' own bottom divided by $\sqrt{s_x s_y}$.

Dividing the top by the bottom therefore gives

$$r_{xy} = r_{ab}\,\sqrt{s_x\,s_y}$$

where $r_{ab}$ is the correlation of the two slow parts, and $s_x$ and $s_y$
are the two sensors' shares of slow drift. When the slow parts move in perfect
step, $r_{ab}$ is +1 or −1, and the size of the measured correlation is
$\sqrt{s_x s_y}$, called here the **noise limit**: the largest size the two
sensors' correlation can reach, given how much scatter each carries. Scatter
can only shrink a correlation, and more rows do not undo the shrinking: it
does not depend on $n$. Spearman (1904, pp. 89–90) called this weakening by
"accidental" errors of measurement "attenuation", and wrote that it "cannot be
in the least eliminated by any prolongation of the series". The derivation
above is this page's own.

Nf and NRf show it at work. NRf is the fan speed corrected for the inlet
temperature (Wen et al., 2026, p. 5), and FD001's inlet sits at the standard
day's temperature, so the two are nearly the same quantity (see the four fan
speeds under
[What comes out](#what-comes-out-21-sensors-and-four-hidden-margins)). Their
shares of slow drift are 0.818213 and 0.821122, so their noise limit is
$\sqrt{0.818213 \times 0.821122} = 0.819666$, and their readings correlate at
0.826084: at the limit, within how precisely the shares can be measured from
one file. From one flight to the next they correlate at only 0.047887, because
each carries its own scatter, as separate measurement noise on every output
channel would give it (Saxena et al., 2008, Section V.B, p. 6). Two readings of
one fan speed correlate at 0.83 rather than 1, because about a fifth of each
one's variance is its own scatter.

<details class="process-example">
<summary>Example in Python</summary>

```python
share = noise_shares.set_index("sensor")["share of the slow drift"]
limit_examples = pd.DataFrame([
    {
        "pair": f"{first} and {second}",
        "share of the first": share[first],
        "share of the second": share[second],
        "noise limit": round(np.sqrt(share[first] * share[second]), 6),
        "correlation of the readings": round(readings[first].corr(readings[second]), 6),
        "correlation of the changes": round(changes[first].corr(changes[second]), 6),
    }
    for first, second in [("Nf", "NRf"), ("T50", "Ps30"), ("T24", "T30"), ("Nc", "T50")]
])
```

</details>

<!-- sample: limit_examples -->
| pair | share of the first | share of the second | noise limit | correlation of the readings | correlation of the changes |
| --- | --- | --- | --- | --- | --- |
| Nf and NRf | 0.818213 | 0.821122 | 0.819666 | 0.826084 | 0.047887 |
| T50 and Ps30 | 0.799516 | 0.855471 | 0.82702 | 0.830136 | 0.009879 |
| T24 and T30 | 0.63541 | 0.574262 | 0.604063 | 0.60261 | -0.000447 |
| Nc and T50 | 0.964263 | 0.799516 | 0.878034 | 0.297429 | -0.000507 |

T50 and Ps30, and T24 and T30, sit at their limits too. Nc and T50 fall far
below theirs, for a reason given under the next figure.

{{figure:noise-limit}}

Each mark is one of the 91 pairs. Across: its noise limit, from the two shares
in the table of shares above. Up: the size of its correlation in the left
table of the previous figure. A mark on the dashed line is a pair whose slow
parts move in perfect step, so that the scatter alone decides how strong the
correlation looks. Point at a mark for its pair.

<details class="process-example">
<summary>Example in Python</summary>

```python
import itertools

pair_limits = pd.DataFrame([
    {
        "first": first, "second": second,
        "size": abs(readings[first].corr(readings[second])),
        "limit": np.sqrt(share[first] * share[second]),
    }
    for first, second in itertools.combinations(readings.columns, 2)
])
with_nc = (pair_limits["first"] == "Nc") | (pair_limits["second"] == "Nc")
with_p15 = (pair_limits["first"] == "P15") | (pair_limits["second"] == "P15")
pair_limits["group"] = np.where(with_nc, "pairs with Nc",
                                np.where(with_p15, "pairs with P15", "the other pairs"))
ratio = pair_limits["size"] / pair_limits["limit"]
limit_groups = (
    ratio.groupby(pair_limits["group"]).agg(["count", "min", "max"]).round(6)
    .reset_index()
    .set_axis(["group", "pairs", "smallest size ÷ limit", "largest size ÷ limit"], axis=1)
)
plain = pair_limits["group"] == "the other pairs"
fan = pair_limits["first"].isin(["Nf", "NRf"]) ^ pair_limits["second"].isin(["Nf", "NRf"])
assert (plain & fan).sum() == 20 and ratio[plain & fan].max() < 0.97 < ratio[plain & ~fan].min()
nc_with_t50 = readings.assign(unit=fd001.unit).groupby("unit").apply(
    lambda engine: engine.Nc.corr(engine.T50))
assert ((nc_with_t50 > 0).sum(), (nc_with_t50 < 0).sum()) == (71, 29)
assert set(nc_with_t50.index[nc_with_t50 > 0]) == set(drift.index[drift.Nc > 0])
```

</details>

<!-- sample: limit_groups -->
| group | pairs | smallest size ÷ limit | largest size ÷ limit |
| --- | --- | --- | --- |
| pairs with Nc | 13 | 0.036129 | 0.434012 |
| pairs with P15 | 12 | 0.551079 | 0.636279 |
| the other pairs | 66 | 0.874844 | 1.012246 |

For the 66 pairs that involve neither Nc nor P15, the size divided by the
limit runs from 0.874844 to 1.012246. A ratio above 1 is impossible for the
true values, so the 1.012246 shows how precisely the shares can be estimated
from one file. These pairs' slow parts move in step, and how strongly any two
of them appear to be linked is set by how much scatter each one carries. The
lowest of them are the 20 pairs that join Nf or NRf to one of the other ten
sensors, all below 0.97: the drift of the two fan speeds follows the others' a
little less closely than theirs follow each other.

The two groups below the line are the two exceptions of the previous figure.
Within one engine, Nc rises and falls with T50 on the 71 engines where Nc
rises, and against T50 on the 29 where it falls: its correlation with T50 over
each engine's own record is positive on exactly those 71 engines and negative
on the other 29. Pooling the engines mixes the two, and Nc's correlations fall
to between 0.036129 and 0.434012 of their limits. P15's two recorded values do
not fit the split into a slow part and a scatter, so its limits mean little.

**What this says about the paper's reading.** Section 3.2.2 describes three
patterns in the correlation table (Wen et al., 2026, Section 3.2.2, p. 8):
temperature sensors (T24, T30, T50) "demonstrating strong intercorrelations
due to thermodynamic coupling throughout the engine cycle"; pressure
measurements (P15, P30, Ps30) that "exhibit similar correlation patterns
reflecting the progressive compression process"; and rotational speed
parameters (Nf, Nc, NRf, NRc) that "show high correlation coefficients,
indicating mechanical coupling through shared shaft systems". The pairs those
three sentences name are below, with the value the paper's Fig. 5 prints for
each (Wen et al., 2026, Fig. 5, p. 8), this file's value and the noise limit.

<details class="process-example">
<summary>Example in Python</summary>

The first column is typed in from Fig. 5, read off the page drawn at eight
times its size.

```python
fig5_printed = {
    ("T24", "T30"): 0.60, ("T24", "T50"): 0.71, ("T30", "T50"): 0.68,
    ("P15", "P30"): -0.16, ("P15", "Ps30"): 0.16, ("P30", "Ps30"): -0.82,
    ("Nf", "Nc"): -0.03, ("Nf", "NRf"): 0.83, ("Nc", "NRf"): -0.03,
}
paper_groups = pd.DataFrame([
    {
        "pair": f"{first} and {second}",
        "printed in Fig. 5": printed,
        "this file": round(readings[first].corr(readings[second]), 6),
        "noise limit": round(np.sqrt(share[first] * share[second]), 6),
    }
    for (first, second), printed in fig5_printed.items()
])
assert (paper_groups["this file"].round(2) == paper_groups["printed in Fig. 5"]).all()
assert round(readings.T50.corr(readings.phi), 2) == -0.82
```

</details>

<!-- sample: paper_groups -->
| pair | printed in Fig. 5 | this file | noise limit |
| --- | --- | --- | --- |
| T24 and T30 | 0.6 | 0.60261 | 0.604063 |
| T24 and T50 | 0.71 | 0.714949 | 0.712756 |
| T30 and T50 | 0.68 | 0.678413 | 0.677593 |
| P15 and P30 | -0.16 | -0.15572 | 0.246545 |
| P15 and Ps30 | 0.16 | 0.160014 | 0.257004 |
| P30 and Ps30 | -0.82 | -0.822805 | 0.820659 |
| Nf and Nc | -0.03 | -0.032091 | 0.888241 |
| Nf and NRf | 0.83 | 0.826084 | 0.819666 |
| Nc and NRf | -0.03 | -0.034763 | 0.889819 |

- The three temperature pairs sit at their noise limits, and T50 correlates
  more strongly with Ps30 (0.83) and with phi (−0.82) than with either of the
  other two temperatures. Nothing in the table sets the temperatures apart as
  a group.
- P15 correlates with P30 at −0.16 and with Ps30 at +0.16, so it shares no
  pattern with the other two pressures. P30 and Ps30, both measured at the HPC
  outlet (Wen et al., 2026, Table 2, p. 5), correlate at −0.82: one falls
  while the other rises.
- The two pairs of speeds that join the fan to the core, Nf with Nc and Nc
  with NRf, print −0.03. The fan and the core turn on different shafts
  ([The engine and its flow path](#the-engine-and-its-flow-path)), and the
  only strong correlation among the speeds, Nf with NRf at 0.83, joins two
  readings of the same fan speed. NRc is not in the figure: it is the seventh
  sensor the paper removes (assumption A-10).

So the table does not show the coupling the text reads into it. Where two
sensors correlate strongly, both drift with the one fault and carry little
scatter of their own; where the text expects coupling through shared shafts,
the figure prints almost none. The links the text names may exist in the
engine, but this table is not evidence of them (finding F-23).

**Do not confuse a strong correlation with a strong physical link.** In FD001
a large correlation between two sensors says that both respond to the one
simulated fault and that neither carries much scatter of its own. It does not
say that one sensor drives the other, or that the two sit close together in
the engine.

**Smoothing shows the same thing from the other side.** The paper smooths the
readings with an exponentially weighted moving average (Wen et al., 2026,
Section 3.2.3, Eq. 18, p. 8), whose weight this project reads from a
self-contradictory sentence as 0.5 (assumption A-04), and this project
smooths each engine's record on its own
([Preprocessing, Smooth each sensor](../learn/04_the_pipeline.md#2--smooth-each-sensor)).
Smoothing takes away part of the scatter and changes nothing about how any
engine was built. Yet it raises the correlation of T24 and T30 from 0.60 to
0.81, and the average size over the 91 pairs from 0.57 to 0.69:

<details class="process-example">
<summary>Example in Python</summary>

```python
from thesis_gnn_rul_xai.data.smoothing import apply_ewma
from thesis_gnn_rul_xai.graph.adjacency import build_physics_adjacency

columns = [symbol_to_column(name) for name in readings.columns]
smoothed = apply_ewma(fd001, columns, alpha=0.5)[columns].set_axis(list(readings.columns), axis=1)
smoothing_effect = pd.DataFrame([
    {
        "readings": what,
        "T24 and T30": round(table.T24.corr(table.T30), 6),
        "average size over the 91 pairs": round(np.abs(table.corr().to_numpy()[upper]).mean(), 6),
    }
    for what, table in [("as recorded", readings), ("smoothed, weight 0.5", smoothed)]
])
table3, order = build_physics_adjacency()
assert list(order) == list(readings.columns)
sizes = readings.corr().abs().to_numpy()[upper]
strongest = np.argsort(-sizes, kind="stable")[:41]
in_strongest = {readings.columns[upper[0][k]] for k in strongest} | {
    readings.columns[upper[1][k]] for k in strongest}
correlation_graph = pd.DataFrame([{
    "pairs kept": len(strongest),
    "weakest size kept": round(sizes[strongest].min(), 6),
    "also joined in Table 3": int((table3[upper][strongest] > 0).sum()),
    "pairs Table 3 joins": int((table3[upper] > 0).sum()),
    "sensors joined to nothing": ", ".join(sorted(set(readings.columns) - in_strongest)),
}])
```

</details>

<!-- sample: smoothing_effect -->
| readings | T24 and T30 | average size over the 91 pairs |
| --- | --- | --- |
| as recorded | 0.60261 | 0.569119 |
| smoothed, weight 0.5 | 0.812857 | 0.69373 |

**What it means for a graph built from correlations.** This project tests the
paper's hand-built graph against one that joins the 41 sensor pairs with the
strongest correlations, as many pairs as Table 3 joins (assumption A-27). The
analysis above says what those 41 are: pairs whose slow parts move in step and
whose readings carry little scatter, wherever the two sensors sit in the
engine. On the raw readings of all 100 engines, the 41 strongest pairs are
those of size 0.69 and above; Nc and P15 are in none of them, and 18 of the 41
are pairs Table 3 also joins:

<!-- sample: correlation_graph -->
| pairs kept | weakest size kept | also joined in Table 3 | pairs Table 3 joins | sensors joined to nothing |
| --- | --- | --- | --- | --- |
| 41 | 0.692893 | 18 | 41 | Nc, P15 |

The graph the project trains with is built separately for every seed, the
random number that decides, among other things, which training engines are
held back to judge when to stop training. It is built from the smoothed and
rescaled rows of that seed's training engines alone. There the weakest pair
kept correlates at between 0.8484 and 0.8550 rather than 0.69, 17 or 18 of its
41 pairs are also joined in Table 3, and Nc and P15 are again joined to
nothing (assumption A-27).

On the test engines the two graphs could not be told apart. The measure is the
root mean squared error: each test engine's error in cycles, squared, averaged
over the 100 test engines and square-rooted, a typical error in cycles. In the
project's default network, which passes each sensor's latest smoothed reading
to the graph layers, five training runs with different seeds gave 16.496 ±
0.415 cycles with Table 3 and 16.592 ± 0.354 with the correlations, where ±
gives the spread between runs. Comparing the two seed by seed gives p = 0.655:
if the graph made no difference at all, a gap at least this large between two
sets of five runs would still turn up about two times in three. With the
recurrent stage of the paper's Section 2.2 in front of the graph layers, at
the size this project chose (assumption A-26), and keeping the training pass
that did best on the held-back engines rather than the paper's 200 passes
(finding F-16), it was 13.234 ± 0.723 against 13.237 ± 0.967 (p = 0.995). In
the default network the paper's graph did make late predictions rarer and
smaller (assumption A-27). By this measure, on FD001, choosing by engineering
judgement which sensors belong together did not make the network more
accurate than the 41 pairs the correlations rank highest.

The key idea is that in FD001 one fault drives every sensor, so twelve of the
fourteen drift in step, and how strongly two of them then appear to correlate
is set by how much flight-to-flight scatter each one carries, not by how the
two are linked inside the engine.

---


## The scoring function

### What Section VII argues

Section VII starts from the purpose of prognostics: the point is to avoid
failures, so predicting failure early is generally preferable to predicting it
late. It notes that the preference could reverse where a failure does not
threaten life and early maintenance is expensive, so a scoring rule should be
set for the application. For the engine scenario, early is preferred, and the
paper states that the challenge's scoring was asymmetric so that "late
predictions were more heavily penalized than early predictions" (Saxena et
al., 2008, Section VII, p. 7), with a penalty that grows exponentially with
the error. What the score rewards in a model, and how it compares with the other measures this
project reports, is on
[Reading a result, The NASA score](../learn/06_reading_results.md#the-nasa-score).

### Equation 11

The paper prints one equation for the whole score (Saxena et al., 2008,
Eq. 11, [p. 7](../references/saxena_2008_damage_propagation_modeling.pdf#page=7)).
As printed, each of its two branches is a sum over the engines from $i = 1$ to
$n$ of an exponential, followed by $-1$ and by the condition on $d$, and $d$
carries no index $i$. Read literally, the $-1$ would be subtracted once and one
branch would be chosen for all engines together. Since the error differs from
engine to engine, this page reads it as one penalty per engine, with the $-1$
inside, summed; the index $i$ on $d$ and the placement of the $-1$ are this
page's reading, not the paper's print. Figure 9, which plots the score of a
single error and is 0 at an error of 0 (Saxena et al., 2008, Fig. 9, p. 8),
matches it. Eq. (11), read per engine:

$$s = \sum_{i=1}^{n} s_i, \qquad s_i = \begin{cases} e^{-d_i/a_1} - 1 & d_i < 0 \\ e^{\,d_i/a_2} - 1 & d_i \ge 0 \end{cases}$$

- $s$ is the computed score, the sum of every engine's penalty $s_i$.
- $n$ is the number of units under test — here, test engines.
- $d_i = \hat{t}_{RUL} - t_{RUL}$ is engine $i$'s error: estimated remaining
  life minus true remaining life. Negative $d_i$ is an early prediction,
  positive $d_i$ a late one.
- $a_1$ and $a_2$ set how fast the penalty grows on each side: a smaller
  divisor makes the penalty climb faster.

It was the number participants received for their test-set predictions
(Saxena et al., 2008, Section VI, p. 7), and it is one of the four measures
this project reports.

### The printed constants contradict the sentence above them

Under the equation the paper prints $a_1 = 10$ and $a_2 = 13$ (Saxena et al.,
2008, Eq. 11, p. 7). Put those into
Equation 11: an early error ($d < 0$) is divided by 10, a late error ($d \ge 0$)
by 13. A larger divisor grows more slowly, so the printed constants penalise
**late** predictions **less** — the opposite of the sentence quoted above
(finding F-20).

Figure 9 supports the sentence (Saxena et al., 2008, Fig. 9,
[p. 8](../references/saxena_2008_damage_propagation_modeling.pdf#page=8)).
Rendered at eight times its printed size, its tick marks at 0 and at 150 lie
651 pixels apart, 4.34 pixels per unit of score; measured that way, its curve
ends at about 45 for an error of −50 and at about 147 for an error of +50, so
the steep side is the late side (finding F-20, which measured to the centre of
the label 150, one pixel above its tick, and so records 4.35 pixels per unit
and curve ends of 45.5 and 147.2). The figure labels its axis only as the error,
without a sign convention, so it supports rather than settles the reading. The
two readings give, at those two errors:

<details class="process-example">
<summary>Example in Python</summary>

```python
def penalty(d, late_divisor, early_divisor):
    """One engine's penalty: exp(d / late) - 1 when late (d >= 0), exp(-d / early) - 1 when early."""
    d = np.asarray(d, dtype=float)
    return np.where(d >= 0, np.expm1(d / late_divisor), np.expm1(-d / early_divisor))


at_fifty = pd.DataFrame({
    "error d": [-50, 50],
    "late ÷ 10, early ÷ 13": penalty([-50, 50], 10, 13).round(6),
    "Eq. (11) as printed: late ÷ 13, early ÷ 10": penalty([-50, 50], 13, 10).round(6),
})
```

</details>

<!-- sample: at_fifty -->
| error d | late ÷ 10, early ÷ 13 | Eq. (11) as printed: late ÷ 13, early ÷ 10 |
| --- | --- | --- |
| -50 | 45.812668 | 147.413159 |
| 50 | 147.413159 | 45.812668 |

For example, $e^{50/10} - 1 = e^{5} - 1 = 147.413159$. Figure 9 matches the
first column and not the second. The sentence gives the direction, late
punished harder, and the figure matches divisors of 10 for late and 13 for
early; only the two printed constants disagree. They appear to have been
swapped. The project's `nasa_score` divides late errors by 10 and early ones
by 13 (assumption A-15).

Wen et al. (2026), the paper this project re-implements, print the same pair
of divisors in their Equation 21. Read through their Equation 20, which
defines the error as true minus predicted, it gives the 13 to late
predictions, as this paper's printed constants do; their own text calls a
positive error an over-prediction and punishes it harder, which fits
Equation 21 only with the error taken as predicted minus true. So in Wen et
al. it is Equation 20's sign, not the constants, that disagrees with the text
(Wen et al., 2026, Eqs. 20–21,
[p. 9](../references/wen_2026_estgnn_dynamic_shap.pdf#page=9); assumption A-15).

### The asymmetry on real test-engine errors

Seeing the asymmetry needs real errors of both signs. A trained network's
predictions would do, but they depend on a stored training run, so this example
uses the simplest prediction the files support without one: for a test engine
that has flown $c$ cycles, take every training engine that lived longer than $c$
and average how many more cycles each of them lived. It ignores the sensors
completely; it is a yardstick for the score, not a model of this project, and
its numbers are comparable with nothing else on this site.

Test engines 78 and 96 have flown 72 and 97 cycles. Every FD001 training engine
lived at least 128, so all 100 count, and the prediction is the average life
minus the cycles flown: 206.31 − 72 = 134.31 and 206.31 − 97 = 109.31.

<details class="process-example">
<summary>Example in Python</summary>

```python
prediction = pd.Series(
    [(lives[lives > flown] - flown).mean() for flown in recorded], index=recorded.index,
)
error = prediction - remaining
pair = [78, 96]
assert (lives > recorded.loc[pair].max()).all()
score_pair = pd.DataFrame({
    "test engine": pair,
    "cycles recorded": recorded.loc[pair].to_numpy(),
    "prediction": prediction.loc[pair].round(6).to_numpy(),
    "true RUL": remaining.loc[pair].to_numpy(),
    "error d": error.loc[pair].round(6).to_numpy(),
    "late ÷ 10, early ÷ 13": penalty(error.loc[pair], 10, 13).round(6),
    "Eq. (11) as printed": penalty(error.loc[pair], 13, 10).round(6),
})
for engine in pair:
    single = nasa_score([remaining.loc[engine]], [prediction.loc[engine]])
    assert abs(single - penalty(error.loc[engine], 10, 13)) < 1e-9
```

</details>

<!-- sample: score_pair -->
| test engine | cycles recorded | prediction | true RUL | error d | late ÷ 10, early ÷ 13 | Eq. (11) as printed |
| --- | --- | --- | --- | --- | --- | --- |
| 78 | 72 | 134.31 | 107 | 27.31 | 14.348228 | 7.172454 |
| 96 | 97 | 109.31 | 137 | -27.69 | 7.414867 | 14.942683 |

Engine 78 is predicted 27.31 cycles late: $e^{27.31/10} - 1 = e^{2.731} - 1 = 14.348228$.
Engine 96 is predicted 27.69 cycles early: $e^{27.69/13} - 1 = e^{2.13} - 1 = 7.414867$.
Two misses of almost the same size, and the late one costs nearly twice as
much. Under the printed constants the costs trade places. The code checks that
the project's `nasa_score`, given one engine at a time, returns exactly the
first penalty column.

Over all 100 test engines the choice moves the total a long way, in a direction
that depends on which misses are largest:

<details class="process-example">
<summary>Example in Python</summary>

```python
late_count = int((error >= 0).sum())
assert (late_count, len(error) - late_count) == (67, 33)
worst = error.idxmin()
assert (worst, recorded[worst], remaining[worst]) == (89, 177, 136)
assert (round(prediction[worst], 2), round(error[worst], 2)) == (46.16, -89.84)
others = error.drop(worst)
score_totals = pd.DataFrame([
    {"engines": "all 100",
     "this project (nasa_score)": round(nasa_score(remaining, prediction), 6),
     "Eq. (11) as printed": round(float(penalty(error, 13, 10).sum()), 6)},
    {"engines": f"engine {worst} alone",
     "this project (nasa_score)": round(float(penalty(error[worst], 10, 13)), 6),
     "Eq. (11) as printed": round(float(penalty(error[worst], 13, 10)), 6)},
    {"engines": f"the other {len(others)}",
     "this project (nasa_score)": round(float(penalty(others, 10, 13).sum()), 6),
     "Eq. (11) as printed": round(float(penalty(others, 13, 10).sum()), 6)},
])
```

</details>

<!-- sample: score_totals -->
| engines | this project (nasa_score) | Eq. (11) as printed |
| --- | --- | --- |
| all 100 | 7778.6788 | 12997.155918 |
| engine 89 alone | 1002.018498 | 7973.46627 |
| the other 99 | 6776.660301 | 5023.689648 |

67 of this yardstick's 100 errors are late and 33 early, yet the printed
constants give the larger total. One engine decides it. Test engine 89 has
flown 177 cycles and has 136 left; the yardstick predicts 46.16, which is
89.84 cycles early. Its penalty is $e^{89.84/13} - 1 = 1002.018498$ under the
project's reading and $e^{89.84/10} - 1 = 7973.46627$ under the printed
constants. Without it, the printed constants would give the smaller total,
5023.689648 against 6776.660301. A score is a sum of exponentials, so a few
large misses decide it, and which side carries the smaller divisor decides
which misses those are.

The key idea is that the score is one exponential penalty per engine, summed,
and whichever side gets the smaller divisor is the side whose large misses
dominate the total; the printed constants and the paper's own sentence and
figure put that side in opposite places.

### What else Section VII proposes

After the challenge the authors suggested two refinements, which the challenge
did not use (Saxena et al., 2008, Section VII,
[p. 8](../references/saxena_2008_damage_propagation_modeling.pdf#page=8)).
Errors close to the end of life matter more and are easier to make small, so
engines with short true remaining life could be weighted more heavily. And
because a sum over engines can hide whether a method is consistently good or
good on some engines and bad on others, a correlation measure across the
engines could be reported beside it. Figure 10 is a simplified asymmetric
score made for the illustration; Figure 11 applies it to six sets of
predictions for four engines, each set's scores totalling 6, with printed
correlations ranging from 0.00 to 1.00. Each panel plots the estimated beside
the true remaining lives, but the paper does not define the correlation
further. (One sentence of the text swaps the two figure numbers; the captions
are followed here.)

---

## What this project takes from these sources

- **The files.** The project trains and tests on FD001, read as on
  [Data](../learn/01_the_problem.md); the loader's expected sizes follow the
  files, as described under
  [What the files contain](#what-the-files-contain-set-beside-the-readme).
- **The sensor names.** The names attached to columns 6 to 26 come from Wen et
  al.'s Table 2, which numbers the 21 sensors 1 to 21 in the same order as this
  paper's Table 2 (Saxena et al., 2008, Table 2; Wen et al., 2026, Table 2,
  p. 5). The evidence that the file columns follow that order is the
  check under
  [Which flight conditions the settings are](#which-flight-conditions-the-settings-are).
- **The labels.** Training labels count the cycles left before the record ends,
  which the readme calls system failure (NASA PCoE, 2008) and which, if the
  files were made by the method the paper describes, is where the simulated
  health index reached zero (Saxena et al., 2008, p. 7); test labels come from
  the RUL files ([Data](../learn/01_the_problem.md)).
  Wen et al. (2026) never state the value of their label ceiling (Eq. 19,
  p. 9); 125 cycles is the project's reading of their Fig. 7 (p. 11;
  assumption A-01), and applying the ceiling to the test labels
  too is the project's own choice (assumption A-02). Neither comes from this
  paper.
- **The score.** `nasa_score` in `evaluation/metrics.py` follows the reading
  set out under
  [The printed constants contradict the sentence above them](#the-printed-constants-contradict-the-sentence-above-them)
  (assumption A-15).
- **What it does not take.** The health index, the margins and the health
  modifiers were never released (NASA PCoE, 2008; Saxena et al., 2008, Table 2,
  p. 3, and Section VI, p. 7). The paper's validation set is not in these
  files, and the project's own validation engines are a different thing
  (assumption A-06). The
  generation equations are not used either: the project contains no simulated
  data of its own, only NASA's files, so the inconsistencies in Equations 4, 5,
  6, 8 and 10 affect the account of where the data came from, not any result.

---

## References

- Frederick, D. K., DeCastro, J. A., & Litt, J. S. (2007). *User's guide for
  the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)*
  (NASA/TM-2007-215026). NASA Glenn Research Center.
  https://ntrs.nasa.gov/citations/20070034949. Page numbers on this page are
  those of the PDF file, not the report's printed page numbers.
  [Copy](../references/frederick_2007_cmapss_users_guide.pdf)
- NASA Prognostics Center of Excellence. (2008). readme.txt, the description
  of the C-MAPSS data set distributed inside its zip file. NASA Prognostics
  Data Repository. The file itself carries no date.
  [Copy](../references/nasa_2008_cmapss_readme.txt)
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation
  modeling for aircraft engine run-to-failure simulation. In *2008
  International Conference on Prognostics and Health Management* (pp. 1–9).
  IEEE. https://doi.org/10.1109/PHM.2008.4711414. The copy is the authors'
  9-page manuscript as NASA distributes it on DASHlink; equation, table,
  figure and page numbers on this page refer to it.
  [Copy](../references/saxena_2008_damage_propagation_modeling.pdf)
- Spearman, C. (1904). The proof and measurement of association between two
  things. *The American Journal of Psychology*, *15*(1), 72–101.
  https://doi.org/10.2307/1412159. No copy is held; the passages cited were
  read in the scan the Internet Archive keeps,
  https://archive.org/details/jstor-1412159.
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and
  Technology*, *168*, 110875. https://doi.org/10.1016/j.ast.2025.110875
  [Copy](../references/wen_2026_estgnn_dynamic_shap.pdf)
