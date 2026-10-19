# SHAP (2017)

Lundberg and Lee's (2017) paper, *A unified approach to interpreting model
predictions*, starts from a crowded field: several methods for explaining one
prediction of a complex model, and no account of how they relate or when to
prefer one. It shows that six of them produce the same kind of explanation,
states three requirements that only one set of numbers of that kind can meet,
and names those numbers **SHAP values**, short for SHapley Additive exPlanations
([pp. 1–2](../references/lundberg_2017_unified_shap.pdf#page=1); Section 3, p. 4;
Section 4, pp. 4–5).

This page follows the paper: its mechanism in its own equation numbers, what it
claims and on what evidence, and what this project takes from it. How this
project computes the values for its own network, sensor group by sensor group
along an engine's life, is taught on the
[Dynamic SHAP](../learn/10_dynamic_shap.md) page and is not repeated here.

Every equation, section, figure and page number below refers to the
[proceedings version of the paper](../references/lundberg_2017_unified_shap.pdf#page=1)
kept in this repository, unless another paper is named.

## The paper's three steps

Lundberg and Lee (2017, Section 1, pp. 1–2) list three results:

1. Treat any explanation of a prediction as a model in its own right, the
   **explanation model**. This defines a class of methods, *additive feature
   attribution methods*, and six existing methods turn out to belong to it
   (Section 2).
2. Results from cooperative game theory, the mathematics of dividing a jointly
   produced payout among the players who produced it, apply to that whole
   class and guarantee a unique solution. The paper proposes it, the SHAP
   values, as a single measure that the various methods approximate
   (Sections 3 and 4).
3. New ways of estimating SHAP values, which the paper reports agree better
   with human explanations in user studies and separate a model's output
   classes more effectively than several existing methods (Section 5).

## The example used on this page

Every equation on this page is run on one real prediction from **FD001**, the
first of the four subsets of NASA's simulated run-to-failure records of
turbofan engines, C-MAPSS (NASA Prognostics Center of Excellence
[NASA PCoE], 2008; Saxena et al., 2008, pp. 1–2), described on the
[C-MAPSS](cmapss.md) page. Each engine is recorded once per **cycle**
(NASA PCoE, 2008), one flight in this project's reading of its sources
([What one cycle is](../learn/01_the_problem.md#what-one-cycle-is)). A
training engine is recorded until it fails (NASA PCoE, 2008), so at every
cycle its **remaining-life label** is the number of cycles it had left; any
label above 125 is set to 125, as everywhere in this project (assumption
A-01). A **test engine** is a separate engine whose record stops before
failure, kept aside for evaluation; a vector of the test engines' true
remaining lives is provided with them (NASA PCoE, 2008), and in the download it
is a separate file, `RUL_FD001.txt` for FD001.

The model being explained is a **straight-line model** fitted by least squares:
it chooses four multipliers and an intercept that make the sum of squared
differences between its prediction and the label, over every row of the 100
training engines (NASA PCoE, 2008), as small as possible. Its inputs are four
raw sensor readings taken at a single cycle: T24, T30 and T50, total
temperatures at the outlets of the low-pressure compressor, the high-pressure
compressor and the low-pressure turbine, recorded in °R, and P30, total pressure
at the high-pressure compressor outlet, recorded in psia (Saxena et al., 2008,
Table 2, [p. 3](../references/saxena_2008_damage_propagation_modeling.pdf#page=3);
Frederick et al., 2007, Nomenclature, p. 7, for the names of the three
components). Neither source spells out the two units. In their standard
meaning, °R (`degR` in this project's code) is degrees Rankine, an absolute
temperature scale whose degree is the size of a Fahrenheit degree, which the
[C-MAPSS](cmapss.md#what-comes-out-21-sensors-and-four-hidden-margins) page
checks against the user's guide, and psia is pounds per square inch absolute.

Three reasons for a straight-line model rather than the project's network. One
of the paper's results, Linear SHAP, applies only to straight-line models
(Lundberg & Lee, 2017, Section 4.2, p. 6). Every number it produces can be
checked by hand. And it needs no PyTorch. It is **this page's demonstration
model**: it is not the project's model, its accuracy is not being assessed, and
its readings are the raw values from the NASA files, neither smoothed nor
rescaled.

The prediction explained is that of **FD001 test engine 5 at its last recorded
cycle**, the engine the [Learn pages](../learn/03_explainability.md) also use.
Tables round to six decimal places; the calculations use full precision. The
optional Python examples run in page order from the repository root with the
real FD001 files.

<details class="process-example">
<summary>Set up the Python examples</summary>

```python
from itertools import combinations
from math import comb, factorial
import numpy as np
import pandas as pd
from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.labeling import apply_piecewise_rul
from thesis_gnn_rul_xai.data.loading import load_subset
from thesis_gnn_rul_xai.data.sensors import SENSORS, symbol_to_column
from thesis_gnn_rul_xai.utils.paths import raw_data_dir

raw = load_subset("FD001", data_dir=raw_data_dir())
ceiling = PreprocessingConfig().rul_max
train = apply_piecewise_rul(raw.train, rul_max=ceiling)
names = ["T24", "T30", "T50", "P30"]
M = len(names)
columns = [symbol_to_column(name) for name in names]
units = {sensor.symbol: sensor.unit for sensor in SENSORS}

# A NumPy array of shape (20631, 4): one row per training cycle, one column per sensor.
readings = train[columns].to_numpy(dtype=np.float64)
# A NumPy array of shape (20631,): remaining life at each row, capped at the ceiling.
labels = train["rul"].to_numpy(dtype=np.float64)

# Ordinary least squares: four multipliers and one intercept.
design = np.column_stack([readings, np.ones(len(readings))])
solution, *_ = np.linalg.lstsq(design, labels, rcond=None)
w, b = solution[:M], solution[M]

def f(rows):
    """The fitted straight-line model: predicted remaining life, in cycles."""
    return rows @ w + b

mean = readings.mean(axis=0)
engine = raw.test[raw.test["unit"] == 5]
x = engine[columns].to_numpy(dtype=np.float64)[-1]
last_cycle = int(engine["cycle"].iloc[-1])
answer = int(engine["rul"].iloc[-1])

fit_table = pd.DataFrame({
    "Quantity": [
        "Training rows", "Intercept b (cycles)", "Mean label (cycles)",
        "Mean prediction over the training rows (cycles)",
        "Prediction at the training means (cycles)",
        f"Prediction for engine 5 at cycle {last_cycle} (cycles)",
        "Recorded remaining life of engine 5 (cycles)",
    ],
    "Value": pd.Series([len(readings), b, labels.mean(), f(readings).mean(),
                        f(mean), f(x), answer], dtype=object),
})
instance_table = pd.DataFrame({
    "Sensor": names,
    "Unit": [units[name] for name in names],
    "Multiplier w (cycles per unit)": w,
    "Training mean E[x]": mean,
    f"Engine 5 at cycle {last_cycle}": x,
    "Difference from mean": x - mean,
})
```

</details>

<!-- sample: fit_table -->
| Quantity | Value |
|---|---|
| Training rows | 20631 |
| Intercept b (cycles) | 7274.336749 |
| Mean label (cycles) | 86.829286 |
| Mean prediction over the training rows (cycles) | 86.829286 |
| Prediction at the training means (cycles) | 86.829286 |
| Prediction for engine 5 at cycle 98 (cycles) | 76.655142 |
| Recorded remaining life of engine 5 (cycles) | 91 |

<!-- sample: instance_table -->
| Sensor | Unit | Multiplier w (cycles per unit) | Training mean E[x] | Engine 5 at cycle 98 | Difference from mean |
|---|---|---|---|---|---|
| T24 | degR | -14.563452 | 642.680934 | 642.270000 | -0.410934 |
| T30 | degR | -1.145302 | 1590.523119 | 1589.940000 | -0.583119 |
| T50 | degR | -1.531026 | 1408.933782 | 1419.360000 | 10.426218 |
| P30 | psia | 11.115376 | 553.367711 | 553.290000 | -0.077711 |

A multiplier is the change in predicted remaining life for one unit more of
that reading with the other three held fixed. The intercept is the prediction
when all four readings are zero, a point far outside anything an engine
records. Reaching it means extending the fitted line far beyond the data it was
fitted on, an **extrapolation** (an answer outside the range of the fitting
data), as opposed to an **interpolation** (an answer among them); its size has
no physical meaning. It matters again under Linear SHAP below.

The mean prediction over the training rows equals the mean label exactly. That
is a property of least squares with an intercept (its errors on the training
rows add to zero), not of SHAP. The prediction at the training means equals
them too, because the average of a straight line's outputs is its output at the
average input. Both coincidences are special to this model, and the section on
missing features shows why that matters.

---

## An explanation is itself a model

Lundberg and Lee (2017, [Section 2,
p. 2](../references/lundberg_2017_unified_shap.pdf#page=2)) start from an
observation: a simple model, such as a straight line, is its own best
explanation, but a complex one is not, so a complex model $f$ is explained by a
simpler **explanation model** $g$. The paper concerns **local** methods, which
explain one prediction $f(x)$ for one input $x$. It credits that framing to
LIME, the method of Ribeiro and colleagues (the paper's reference [5], p. 10),
which, as Section 2.1 describes it, explains one prediction by fitting a linear
model to the original model's outputs around that input.

### Simplified inputs and the mapping

An explanation model does not read the raw input. It reads a **simplified
input**: one switch per feature, where $z'_i = 1$ means feature $i$ is present
at its actual value and $z'_i = 0$ means it is missing (Lundberg & Lee, 2017,
pp. 2, 4). The simplified version of the explained input itself is written $x'$;
for engine 5 every reading was observed, so $x' = (1, 1, 1, 1)$.

A **mapping** $h_x$ turns a vector of switches back into a complete input the
original model can accept, $h_x(z')$. It is specific to the input $x$ being
explained: a switch that is on takes $x$'s own reading, so $h_x(x') = x$
(Lundberg & Lee, 2017, p. 2). What a switch that is off becomes is the mapping's
choice, and each method in Section 2 makes it differently (table below). SHAP's
own answer, and its alternatives, are the subject of a later section. Until then
this page uses the simplest answer: **a missing reading is replaced by its
training mean**.

For example, $z' = (1, 0, 1, 0)$ keeps T24 and T50 and switches off T30 and
P30. Its mapping gives the input (642.27, 1590.523119, 1419.36, 553.367711):
engine 5's T24 and T50, the training means of T30 and P30. The model's output
on it is the row "T24 + T50" of this table, which runs every one of the
$2^4 = 16$ switch settings through the mapping and the model:

<details class="process-example">
<summary>Example in Python</summary>

```python
# Every group of present sensors, smallest first, as tuples of positions.
groups = [group for size in range(M + 1) for group in combinations(range(M), size)]

def label(group):
    """Name a group by its present sensors."""
    return " + ".join(names[i] for i in group) or "None"

def mean_fill(group):
    """The mapping h_x: present sensors keep engine 5's reading, missing ones get the mean."""
    row = mean.copy()
    row[list(group)] = x[list(group)]
    return row

def background_average(group):
    """Equation 11: average the model over every training row, present sensors fixed."""
    rows = readings.copy()
    rows[:, list(group)] = x[list(group)]
    return f(rows).mean()

value_mean = {group: f(mean_fill(group)) for group in groups}
coalitions = pd.DataFrame([mean_fill(group) for group in groups], columns=names)
coalitions.insert(0, "Present", [label(group) for group in groups])
coalitions["Prediction, missing set to mean"] = [value_mean[group] for group in groups]
coalitions["Average over training rows"] = [background_average(group) for group in groups]
```

</details>

<!-- sample: coalitions -->
| Present | T24 | T30 | T50 | P30 | Prediction, missing set to mean | Average over training rows |
|---|---|---|---|---|---|---|
| None | 642.680934 | 1590.523119 | 1408.933782 | 553.367711 | 86.829286 | 86.829286 |
| T24 | 642.270000 | 1590.523119 | 1408.933782 | 553.367711 | 92.813897 | 92.813897 |
| T30 | 642.680934 | 1589.940000 | 1408.933782 | 553.367711 | 87.497133 | 87.497133 |
| T50 | 642.680934 | 1590.523119 | 1419.360000 | 553.367711 | 70.866473 | 70.866473 |
| P30 | 642.680934 | 1590.523119 | 1408.933782 | 553.290000 | 85.965497 | 85.965497 |
| T24 + T30 | 642.270000 | 1589.940000 | 1408.933782 | 553.367711 | 93.481744 | 93.481744 |
| T24 + T50 | 642.270000 | 1590.523119 | 1419.360000 | 553.367711 | 76.851084 | 76.851084 |
| T24 + P30 | 642.270000 | 1590.523119 | 1408.933782 | 553.290000 | 91.950108 | 91.950108 |
| T30 + T50 | 642.680934 | 1589.940000 | 1419.360000 | 553.367711 | 71.534320 | 71.534320 |
| T30 + P30 | 642.680934 | 1589.940000 | 1408.933782 | 553.290000 | 86.633344 | 86.633344 |
| T50 + P30 | 642.680934 | 1590.523119 | 1419.360000 | 553.290000 | 70.002684 | 70.002684 |
| T24 + T30 + T50 | 642.270000 | 1589.940000 | 1419.360000 | 553.367711 | 77.518931 | 77.518931 |
| T24 + T30 + P30 | 642.270000 | 1589.940000 | 1408.933782 | 553.290000 | 92.617955 | 92.617955 |
| T24 + T50 + P30 | 642.270000 | 1590.523119 | 1419.360000 | 553.290000 | 75.987295 | 75.987295 |
| T30 + T50 + P30 | 642.680934 | 1589.940000 | 1419.360000 | 553.290000 | 70.670531 | 70.670531 |
| T24 + T30 + T50 + P30 | 642.270000 | 1589.940000 | 1419.360000 | 553.290000 | 76.655142 | 76.655142 |

The last column belongs to the section on missing features and can be ignored
until then.

### Definition 1: the additive form

Definition 1 of Lundberg and Lee (2017, p. 2) says an **additive feature
attribution method** is one whose explanation model adds up one number per
switch, plus a constant:

$$g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i \qquad \text{(Eq. 1)}$$

- $M$ is the number of simplified features, here 4.
- $z' \in \{0,1\}^M$ is one setting of the switches.
- $\phi_i$ is a real number attributed to feature $i$, in the unit of the
  model's output, here cycles of remaining life.
- $\phi_0$ is the value of $g$ with every switch off.

The equation starts from $\phi_0$ and adds $\phi_i$ for every switch that is
on. It is needed because a sum of one number per feature is a form a person can
read: each feature gets a single signed amount, and the amounts add up. With
the values computed further down this page, setting every switch on gives
$g(x') = 86.829286 + 5.984611 + 0.667847 - 15.962813 - 0.863789 = 76.655142$
cycles, which is the model's prediction for engine 5. Definition 1 only fixes
the form; the next section's properties are what fix the numbers.

**Do not confuse the simplified input $z'$ with the input $x$.** $x$ holds
readings in degrees Rankine and psia. $z'$ holds only on/off switches. A
$\phi_i$ is attached to a switch, so it describes the effect of *having engine
5's reading instead of the replacement*, never the reading itself.

### Six existing methods share this form

Lundberg and Lee (2017, Section 2, pp. 2–3) show that six published methods all
fit Definition 1. They differ in what a switch set to 0 does:

| Method | Section | What a switch set to 0 does |
|---|---|---|
| LIME | 2.1 | Text: the word's count becomes zero. Images: a patch of pixels is replaced by the average of neighbouring pixels |
| DeepLIFT | 2.2 | The input takes a reference value chosen by the user, meant as a typical uninformative background value |
| Layer-wise relevance propagation | 2.3 | The input takes the value 0; the paper notes, citing Shrikumar et al. without a reference number, that this equals DeepLIFT with the reference of every value inside the network fixed at zero |
| Shapley regression values | 2.4 | The feature is left out of a model retrained without it |
| Shapley sampling values | 2.4 | The model is not retrained; leaving a feature out is approximated by averaging over samples from the training data |
| Quantitative Input Influence | 2.4 | A sampling approximation the paper calls nearly identical to Shapley sampling values |

DeepLIFT, as Lundberg and Lee (2017, Section 2.2) describe it, is a method for
deep networks that gives each input the effect of its being set to the reference
value instead of its actual value, worked out recursively through the network.
Its adding-up rule (Eq. 3, p. 3) already has the shape of Definition 1: its
attributions sum to $f(x) - f(r)$, where $r$ is the reference input. The paper
states the starting value $\phi_0$ for only two of the six methods: DeepLIFT's
is the output at the reference, $f(r)$, and that of Shapley regression values is
the output of a model trained on no features, $f_\varnothing(\varnothing)$. For
the other four it gives none.

Layer-wise relevance propagation in this table is the original method for
ordinary networks, that of Bach et al. (2015, Abstract, p. 1), which is
Lundberg and Lee's (2017, p. 10) reference [1]; the project's route
explanation for its graph layers (assumption A-29) is a later extension of it
to graph networks (Schnake et al., 2022, Abstract), presented on the
[GNN-LRP resource page](gnn_lrp.md).

The key idea is that once an explanation is written as "a starting value plus
one number per present feature", methods that looked unrelated become
different ways of choosing the same numbers, and the question becomes which
choice is right.

---

## Three properties that leave one answer

Lundberg and Lee (2017, [Section 3,
p. 4](../references/lundberg_2017_unified_shap.pdf#page=4)) state three properties
and a theorem that only one set of numbers satisfies all of them.

### Property 1: local accuracy

$$f(x) = g(x') = \phi_0 + \sum_{i=1}^{M} \phi_i x'_i \qquad \text{(Eq. 5)}$$

with $\phi_0 = f(h_x(\mathbf{0}))$, the model's output with every switch off.
The explanation model must reproduce the original model's output at the input
being explained (Lundberg & Lee, 2017, p. 4). For engine 5, every switch off
means every reading at its training mean, so $\phi_0$ is the row "None" of the
table above, 86.829286 cycles, and the four $\phi_i$ must add to $76.655142 - 86.829286 = -10.174144$
cycles. Without this property the attributions could add to anything, and "T50
lowered the prediction by so many cycles" would have no fixed meaning.

### Property 2: missingness

$$x'_i = 0 \implies \phi_i = 0 \qquad \text{(Eq. 6)}$$

A feature that is switched off in the explained input itself receives nothing.
It constrains the values only when $x'$ contains zeros, such as a word absent
from the document LIME is explaining. For engine 5 all four readings exist,
$x' = (1,1,1,1)$, so the condition never applies. The paper notes that every
method in Section 2 already obeys it (Lundberg & Lee, 2017, p. 4).

### Property 3: consistency

Write $f_x(z') = f(h_x(z'))$ for the model's output at a switch setting, and
$z' \setminus i$ for the same setting with switch $i$ turned off. Then
$f_x(z') - f_x(z' \setminus i)$ is **what feature $i$ adds** at that setting;
where switch $i$ is already off the two terms are equal and it adds 0. For any
two models $f$ and $f'$, if

$$f'_x(z') - f'_x(z' \setminus i) \;\ge\; f_x(z') - f_x(z' \setminus i) \quad \text{for all } z' \in \{0,1\}^M \qquad \text{(Eq. 7)}$$

then $\phi_i(f', x) \ge \phi_i(f, x)$. If a model changes so that a feature adds
at least as much as before in every situation, its attribution must not fall
(Lundberg & Lee, 2017, p. 4).
For example, below it turns out that T50 adds $-15.962813$ cycles whatever
group it joins in this model. A second model, identical except that T50's
multiplier is zero, would add 0 whenever T50 joins a group, which is at least
as much, so consistency requires that model's T50 value to be no lower than
this one's. A case where a method fails this test is built further down, under
Kernel SHAP.

### Theorem 1 and the weight it uses

Theorem 1 (Lundberg & Lee, 2017, p. 4) says only one explanation model satisfies
Definition 1 and Properties 1 to 3:

$$\phi_i(f, x) = \sum_{z' \subseteq x'} \frac{|z'|!\,(M - |z'| - 1)!}{M!}\,\big[f_x(z') - f_x(z' \setminus i)\big] \qquad \text{(Eq. 8)}$$

Here $z' \subseteq x'$ runs over every switch setting whose switched-on
features are among those of $x'$, and $|z'|$ counts the switches that are on.
The same sum appears earlier, in the paper's account of Shapley regression
values (Lundberg & Lee, 2017, Eq. 4, p. 3), written with groups of features
instead of switches:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(|F| - |S| - 1)!}{|F|!}\,\big[f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_S(x_S)\big] \qquad \text{(Eq. 4)}$$

- $F$ is the set of all features, so $|F| = M$.
- $S$ is a group of the *other* features, those that are present before
  feature $i$ is added; it runs over all $2^{M-1}$ such groups.
- $f_S(x_S)$ is the model's output with only the features in $S$ present. In
  Shapley regression values that is a model retrained on $S$; in SHAP it is the
  one model, fed through the mapping.
- The bracket is what feature $i$ adds when it joins $S$.

**A note on reading Eq. 8.** In Eq. 8 the bracket is zero whenever switch $i$ is
off, so only settings with $i$ on contribute, and in those $|z'|$ includes $i$
itself. Read that way, the setting $z' = x'$ with all $M$ switches on would need
$(M - M - 1)! = (-1)!$, which is undefined. The weight agrees with Eq. 4, and
with the counting argument below, when $|z'|$ is read as the number of switches
on *besides* $i$, which is the size of the group that $i$ joins. This page, the
project's code and Eq. 15 of Wen et al. (2026, Section 2.4, p. 7) all use
Eq. 4's form.

**Where the weight comes from.** Let the $M$ features arrive one at a time, in
one of the $M!$ possible orders, each order equally likely. The orders in which
exactly the features of $S$ are already present when $i$ arrives number
$|S|!\,(M - |S| - 1)!$: the features of $S$ in any order, then $i$, then the
rest in any order. So the weight is the probability that $i$ finds exactly $S$
when it arrives, and Eq. 4 is **the average, over all arrival orders, of what
feature $i$ adds when it arrives**. The [Dynamic SHAP](../learn/10_dynamic_shap.md)
page counts this out for three sensors and says why every order is averaged.
For four:

<details class="process-example">
<summary>Example in Python</summary>

```python
def order_weight(size):
    """Equation 4's weight: the share of arrival orders that put exactly this group first."""
    return factorial(size) * factorial(M - size - 1) / factorial(M)

order_weights = pd.DataFrame({"Size of S": list(range(M))})
order_weights["Groups of that size"] = [comb(M - 1, size) for size in range(M)]
order_weights["Weight of each group"] = [order_weight(size) for size in range(M)]
order_weights["Weight of all groups of that size"] = (
    order_weights["Groups of that size"] * order_weights["Weight of each group"])
assert np.isclose(order_weights["Weight of all groups of that size"].sum(), 1.0)
```

</details>

<!-- sample: order_weights -->
| Size of S | Groups of that size | Weight of each group | Weight of all groups of that size |
|---|---|---|---|
| 0 | 1 | 0.250000 | 0.250000 |
| 1 | 3 | 0.083333 | 0.250000 |
| 2 | 3 | 0.083333 | 0.250000 |
| 3 | 1 | 0.250000 | 0.250000 |

Each size of group receives a total of one quarter, because feature $i$ is
equally likely to arrive first, second, third or fourth. Within a size, the
quarter is shared equally among the groups of that size. The weights add up to
1, so Eq. 4 is a weighted average.

Here are the eight terms of Eq. 4 for T50 under this page's mapping:

<details class="process-example">
<summary>Example in Python</summary>

```python
def joined(group, i):
    """The group with feature i added, in the same sorted form as the keys."""
    return tuple(sorted(group + (i,)))

t50 = names.index("T50")
before_t50 = [group for group in groups if t50 not in group]
t50_changes = pd.DataFrame({
    "Group before T50": [label(group) for group in before_t50],
    "Weight": [order_weight(len(group)) for group in before_t50],
    "What T50 adds (cycles)": [value_mean[joined(group, t50)] - value_mean[group]
                               for group in before_t50],
})
t50_changes["Weighted"] = t50_changes["Weight"] * t50_changes["What T50 adds (cycles)"]

def shapley(value):
    """Equation 4 over every group: the exact Shapley value of each feature."""
    phi = np.zeros(M)
    for i in range(M):
        for group in groups:
            if i not in group:
                phi[i] += order_weight(len(group)) * (value[joined(group, i)] - value[group])
    return phi

phi_mean = shapley(value_mean)
start = value_mean[()]
exact_values = pd.DataFrame({"Sensor": names, "Shapley value (cycles)": phi_mean})
accuracy_check = pd.DataFrame({
    "Starting value (cycles)": [start],
    "Sum of the four values": [phi_mean.sum()],
    "Starting value plus sum": [start + phi_mean.sum()],
    "Model prediction f(x)": [f(x)],
})
assert np.isclose(t50_changes["Weighted"].sum(), phi_mean[t50])
assert np.isclose(start + phi_mean.sum(), f(x))
```

</details>

<!-- sample: t50_changes -->
| Group before T50 | Weight | What T50 adds (cycles) | Weighted |
|---|---|---|---|
| None | 0.250000 | -15.962813 | -3.990703 |
| T24 | 0.083333 | -15.962813 | -1.330234 |
| T30 | 0.083333 | -15.962813 | -1.330234 |
| P30 | 0.083333 | -15.962813 | -1.330234 |
| T24 + T30 | 0.083333 | -15.962813 | -1.330234 |
| T24 + P30 | 0.083333 | -15.962813 | -1.330234 |
| T30 + P30 | 0.083333 | -15.962813 | -1.330234 |
| T24 + T30 + P30 | 0.250000 | -15.962813 | -3.990703 |

T50 adds the same amount whatever group it joins. That is a property of this
model and this mapping, not of Shapley values in general: in a straight-line
model with missing readings set to their means, the switch for T50 changes the
input in one position only, by $x_{T50} - E[x_{T50}] = 10.426218$ degrees
Rankine, and the output by the multiplier $-1.531026$ times that difference,
−15.962813 cycles (the product of the unrounded numbers). The weighted sum of
eight equal numbers whose weights add to 1 is that number.

Running Eq. 4 for every sensor gives the four Shapley values, and Property 1
can be checked directly:

<!-- sample: exact_values -->
| Sensor | Shapley value (cycles) |
|---|---|
| T24 | 5.984611 |
| T30 | 0.667847 |
| T50 | -15.962813 |
| P30 | -0.863789 |

<!-- sample: accuracy_check -->
| Starting value (cycles) | Sum of the four values | Starting value plus sum | Model prediction f(x) |
|---|---|---|---|
| 86.829286 | -10.174144 | 76.655142 | 76.655142 |

### Where the theorem comes from

Lundberg and Lee (2017, p. 4) attribute Theorem 1 to combined results of
cooperative game theory, where the $\phi_i$ are known as **Shapley values**,
citing for them their reference [6] (p. 10), *A value for n-person games*, a
1953 work by Lloyd Shapley. They credit a 1985 paper by H. Peyton Young, their
reference [10], with showing that Shapley values are the only values satisfying
three axioms: two similar to Properties 1 and 3, and a third that the authors
show to be redundant in this setting. Property 2 is required to adapt those
proofs to the whole class of additive methods. The argument that the third axiom is
redundant is left to the Supplementary Material, which is not part of the
10-page file kept here, and the uniqueness result itself is Young's, so this
page can show what the theorem says but not why it holds. Neither Shapley's nor
Young's original is in this project's library, so this page says nothing about
them beyond what Lundberg and Lee attribute to them.

The paper then draws the consequence (Lundberg & Lee, 2017, p. 4): once the
simplified input mapping $h_x$ is fixed, only one additive feature attribution
method has all three properties, so any method not based on Shapley values
breaks local accuracy, consistency or both (the methods of Section 2 already respect
missingness, it adds). That condition, a mapping fixed in advance, matters
later.

The key idea is that "start from the all-missing output and add one number per
feature, so that the numbers add up exactly and never reward a feature less
when it does more" has exactly one solution: average what each feature adds
over every order in which the features could arrive.

**Do not confuse a Shapley value with a SHAP value.** A Shapley value is the
game-theory quantity of Eq. 4, defined for any rule that gives each group of
features a value. A SHAP value is the Shapley value of one particular rule,
which the next section defines (Lundberg & Lee, 2017, Section 4, p. 4). Numbers
computed with a different rule for missing features are Shapley values, but they
are SHAP values only under the paper's definition or one of its stated
approximations.

---

## What a missing feature becomes

### The definition: a conditional expectation

Lundberg and Lee (2017, [Section 4,
pp. 4–5](../references/lundberg_2017_unified_shap.pdf#page=4)) define **SHAP values
as the Shapley values of a conditional expectation of the model**: the value of
a group $S$ of present features is

$$f_x(z') = f(h_x(z')) = E[f(z) \mid z_S]$$

where $S$ is the set of switches that are on. Most models cannot accept an
input with arbitrary readings missing, so the paper replaces "the model with
only $S$ known" by **the model's average output over inputs that agree with
$x$ on the features in $S$**.

Figure 1 of Lundberg and Lee (2017, p. 5) draws this for one order of four
features. Its axis starts at 0, and an arrow labelled $\phi_0$ carries it to the base value $E[f(z)]$, the
prediction expected when no feature is known. Each further arrow $\phi_i$ is
the change in the expected prediction when one more feature becomes known, and
the last ends at $f(x)$; in the drawn order the fourth arrow points backwards, a
negative contribution. The caption adds that when the model is not linear, or
the features are not independent, the order matters, and the SHAP values are
the average over all orders.

### Four lines, two simplifications

Section 4 of Lundberg and Lee (2017, p. 5) calls the exact computation
challenging and offers two optional assumptions that simplify the expected
values, written as a chain. (In
this page's words, the difficulty is that Eq. 9 needs the joint distribution of
the features, how often each combination of readings occurs together, which is
seldom known.)

$$\begin{aligned}
f(h_x(z')) &= E[f(z) \mid z_S] && \text{(Eq. 9)}\\
&= E_{z_{\bar S} \mid z_S}[f(z)] && \text{(Eq. 10)}\\
&\approx E_{z_{\bar S}}[f(z)] && \text{(Eq. 11)}\\
&\approx f([z_S, E[z_{\bar S}]]) && \text{(Eq. 12)}
\end{aligned}$$

- $z$ is an input drawn from the data the model is used on.
- $S$ is the set of present features and $\bar S$ the set of missing ones.
- $z_S$ is fixed at the explained input's values, $x_S$.
- $E[\,\cdot\,]$ is an average; its subscript says what is averaged over.

Line by line:

- **Eq. 9** is the definition above.
- **Eq. 10** says the same thing as an average over the missing features drawn
  from their distribution *given* the present ones: only inputs that look like
  $x$ on $S$ count.
- **Eq. 11** assumes the features are independent: the missing features are
  drawn from their own overall distribution, ignoring the present values. The
  paper notes that Shapley sampling values, Quantitative Input Influence, LIME
  and DeepLIFT make this assumption (Lundberg & Lee, 2017, p. 5). (This page's
  count of its cost: one model evaluation for every row of the data averaged over, the **background
  data**, for each group.)
- **Eq. 12** further assumes the model is linear: the average of the outputs is
  replaced by the output at the average of the missing features. (By the same
  count, one model evaluation per group.)

The rule used on this page so far, "missing reading set to its training mean",
is Eq. 12. The coalitions table above also carries Eq. 11, computed by
averaging the model over all 20,631 training rows with the present readings
fixed at engine 5's. The two columns agree to every printed decimal, because the
model is a straight line, and the average of a straight line's outputs is its
output at the average. For a network that is not linear the two can differ.

Whichever line is used, it supplies the value of every group, and Eq. 4 then
averages what each feature adds to those values. The choice of line therefore
reaches the final numbers only through the group values, which the next
example makes visible.

### Why the choice matters: correlated readings

Eq. 10 and Eq. 11 differ when the features are related to one another, and
these four readings are strongly related across the training rows:

<details class="process-example">
<summary>Example in Python</summary>

```python
correlations = pd.DataFrame(np.corrcoef(readings.T), columns=names)
correlations.insert(0, "Sensor", names)

high = readings[:, t50] >= x[t50]
against = high & (readings[:, 0] <= mean[0]) & (readings[:, 1] <= mean[1]) & (readings[:, 3] >= mean[3])
sparse_corner = pd.DataFrame({
    "Training rows": [
        "All",
        "With T50 at least engine 5's",
        "Of those, with T24 and T30 at or below average and P30 at or above",
    ],
    "Count": [len(readings), int(high.sum()), int(against.sum())],
})
```

</details>

<!-- sample: correlations -->
| Sensor | T24 | T30 | T50 | P30 |
|---|---|---|---|---|
| T24 | 1.000000 | 0.602610 | 0.714949 | -0.702136 |
| T30 | 0.602610 | 1.000000 | 0.678413 | -0.664595 |
| T50 | 0.714949 | 0.678413 | 1.000000 | -0.793130 |
| P30 | -0.702136 | -0.664595 | -0.793130 | 1.000000 |

Each number is the Pearson correlation between two sensors over the training
rows: +1 for readings that rise together along a straight line, −1 for one
that falls along a straight line as the other rises, 0 for no straight-line
relation. When T50 is high, T24 and T30 tend to be high and P30 low.

Under Eq. 12 the group "T50" alone pairs engine 5's high T50 with the average
T24, T30 and P30. Under Eq. 11 it pairs that T50 with every training row's own
T24, T30 and P30, whatever that row's T50 was, and averages the model's
outputs. Both ignore the correlations. The Eq. 12 input sits in a corner the
training rows hardly reach:

<!-- sample: sparse_corner -->
| Training rows | Count |
|---|---|
| All | 20631 |
| With T50 at least engine 5's | 2864 |
| Of those, with T24 and T30 at or below average and P30 at or above | 10 |

So the model is asked about a combination its fitting data barely contain, and
its answer there comes from extending the fitted line into that corner, closer
to extrapolation than to interpolation.

To see what Eq. 10 would do instead, this page estimates the conditional
expectation with a straight line: for each missing reading, fit a
least-squares line that predicts it from the present readings over the training
rows, and feed the line's prediction at engine 5's present readings. **This is
this page's illustration only**; the paper does not prescribe it, and the
project does not use it. Because the model being explained is itself a
straight line, $E[f(z) \mid z_S] = f([x_S, E[z_{\bar S} \mid z_S]])$ holds
exactly, so the only approximation is the straight-line estimate of the
missing readings.

<details class="process-example">
<summary>Example in Python</summary>

```python
def line_fill(group):
    """Fill each missing reading with a least-squares line fitted on the present ones."""
    row = x.copy()
    present = list(group)
    inputs = np.column_stack([readings[:, present], np.ones(len(readings))])
    for j in range(M):
        if j not in group:
            coefficients, *_ = np.linalg.lstsq(inputs, readings[:, j], rcond=None)
            row[j] = np.append(x[present], 1.0) @ coefficients
    return row

fill_example = pd.DataFrame({
    "Sensor": names,
    "Engine 5": x,
    "Missing set to mean": mean_fill((t50,)),
    "Missing filled from T50 by a line": line_fill((t50,)),
})
value_line = {group: f(line_fill(group)) for group in groups}
phi_line = shapley(value_line)
dependence = pd.DataFrame({
    "Sensor": names,
    "Missing set to mean, Eqs. 11-12": phi_mean,
    "Missing filled by lines, Eq. 10 estimate": phi_line,
})
t50_line_changes = pd.DataFrame({
    "Group before T50": [label(group) for group in before_t50],
    "Weight": [order_weight(len(group)) for group in before_t50],
    "What T50 adds (cycles)": [value_line[joined(group, t50)] - value_line[group]
                               for group in before_t50],
})
assert np.isclose(phi_line.sum(), phi_mean.sum())
```

</details>

For the group "T50" alone, the two rules feed the model these inputs:

<!-- sample: fill_example -->
| Sensor | Engine 5 | Missing set to mean | Missing filled from T50 by a line |
|---|---|---|---|
| T24 | 642.270000 | 642.680934 | 643.095073 |
| T30 | 1589.940000 | 1590.523119 | 1595.341387 |
| T50 | 1419.360000 | 1419.360000 | 1419.360000 |
| P30 | 553.290000 | 553.367711 | 552.554529 |

The resulting Shapley values:

<!-- sample: dependence -->
| Sensor | Missing set to mean, Eqs. 11-12 | Missing filled by lines, Eq. 10 estimate |
|---|---|---|
| T24 | 5.984611 | 14.756501 |
| T30 | 0.667847 | 2.427071 |
| T50 | -15.962813 | -27.273242 |
| P30 | -0.863789 | -0.084474 |

Both columns add to the same total, because both rules give the same output
with every switch off (the training means) and with every switch on (engine 5's
readings). In between they disagree: under the line fill, knowing T50 also
implies a hotter T24 and T30 and a lower P30, so T50's arrival carries their
expected effect with it, and the split among the four sensors changes.

Under the line fill, what T50 adds also depends on the group it joins, so the
value of a group is no longer a sum of one amount per feature:

<!-- sample: t50_line_changes -->
| Group before T50 | Weight | What T50 adds (cycles) |
|---|---|---|
| None | 0.250000 | -36.551312 |
| T24 | 0.083333 | -40.497804 |
| T30 | 0.083333 | -29.532019 |
| P30 | 0.083333 | -21.500151 |
| T24 + T30 | 0.083333 | -29.777237 |
| T24 + P30 | 0.083333 | -22.041889 |
| T30 + P30 | 0.083333 | -18.600481 |
| T24 + T30 + P30 | 0.250000 | -18.558462 |

The key idea is that "missing" has to be given a meaning before any Shapley
value exists. The paper's meaning is the conditional expectation, the
estimators it describes rely on the independence of Eq. 11 or on the output at
the average reading of Eq. 12, and with related features those choices move
credit between them.

---

## Kernel SHAP: Shapley values from a weighted regression

Exact Shapley values need the value of every group, $2^M$ of them. Lundberg and
Lee (2017, [Section 4.1,
p. 5](../references/lundberg_2017_unified_shap.pdf#page=5)) describe two ways to
estimate them that treat the model as a closed box, using only its inputs and
outputs. Shapley sampling values estimate Eq. 8 by sampling orders of the
features, separately for each feature. The paper's new method, **Kernel SHAP**,
estimates all of them at once by regression, and it reports that this needs
fewer model evaluations for similar accuracy (Section 5).

### LIME's regression, Eq. 2

As Lundberg and Lee (2017, Section 2.1, p. 2) describe it, LIME fits its
explanation model by solving

$$\xi = \underset{g \in \mathcal{G}}{\arg\min}\; L(f, g, \pi_{x'}) + \Omega(g) \qquad \text{(Eq. 2)}$$

- $\xi$ is the explanation model the procedure returns.
- $\arg\min_{g \in \mathcal{G}}$ means: of all the allowed explanation models
  $g$, the one that makes the expression after it smallest.
- $\mathcal{G}$ is the family of allowed explanation models, here those of
  Eq. 1.
- $L$ is a loss: how far $g(z')$ is from $f(h_x(z'))$, summed over a set of
  sampled switch settings.
- $\pi_{x'}(z')$ is a **weighting kernel**: how much each sampled setting counts
  in that sum.
- $\Omega(g)$ is a penalty on the complexity of $g$.

With $g$ of the form of Eq. 1 and a squared loss, Eq. 2 is a penalised
weighted linear regression: the switches are the inputs, the model's outputs
are the targets, and the fitted coefficients are the $\phi_i$.

What makes LIME **local** is the kernel. Lundberg and Lee (2017, p. 5) say only
that LIME measures closeness in the space of switches. Figure 2A (p. 7), rendered at
eight times its printed size, labels two LIME kernels by the distance each
uses, cosine distance (based on the angle between two vectors of switches) and
L2 distance (the straight-line distance). Reading its axis, settings ordered by
how many switches are on, as running from few to many, both LIME kernels give
the least weight to settings with few switches on and the most to settings with
nearly all on, the settings nearest $x' = (1, \dots, 1)$. The paper points out
(p. 5) that LIME
chooses $L$, $\pi_{x'}$ and $\Omega$ by rule of thumb, and that with those
choices the solution is not the Shapley values, so by Theorem 1 it breaks local
accuracy or consistency.

### Theorem 2: the Shapley kernel

Theorem 2 (Lundberg & Lee, 2017, p. 6) gives the choices that make the solution
of Eq. 2 the Shapley values:

$$\Omega(g) = 0, \qquad
\pi_{x'}(z') = \frac{M - 1}{\binom{M}{|z'|}\,|z'|\,(M - |z'|)}, \qquad
L(f, g, \pi_{x'}) = \sum_{z' \in Z} \big[f(h_x(z')) - g(z')\big]^2\,\pi_{x'}(z')$$

- $|z'|$ is the number of switches on, and $\binom{M}{|z'|}$ the number of
  settings with that many on.
- $Z$ is the set of switch settings used in the fit.
- $\Omega(g) = 0$: no penalty.

The kernel gives each setting a weight that depends only on how many switches
are on. With four features and two switches on it is
$(4 - 1)/(\binom{4}{2} \cdot 2 \cdot 2) = 3/24 = 0.125$; with one switch on,
$3/(4 \cdot 1 \cdot 3) = 0.25$. For every count:

<details class="process-example">
<summary>Example in Python</summary>

```python
def shapley_kernel(size):
    """Theorem 2's weight for a setting with this many switches on, 0 < size < M."""
    return (M - 1) / (comb(M, size) * size * (M - size))

kernel_weights = pd.DataFrame({
    "Switches on": list(range(M + 1)),
    "Settings": [comb(M, size) for size in range(M + 1)],
    "Shapley kernel per setting": pd.Series(
        ["infinite"] + [shapley_kernel(size) for size in range(1, M)] + ["infinite"],
        dtype=object),
})
```

</details>

<!-- sample: kernel_weights -->
| Switches on | Settings | Shapley kernel per setting |
|---|---|---|
| 0 | 1 | infinite |
| 1 | 4 | 0.250000 |
| 2 | 6 | 0.125000 |
| 3 | 4 | 0.250000 |
| 4 | 1 | infinite |

The weights are symmetric, the same for one switch on as for one switch off,
which is the shape Figure 2A plots for an example with $2^{15}$ settings: high
at both ends and low in the middle, unlike LIME's kernels, which rise towards
the all-on end. Here the settings with one or three switches on count twice as
much, each, as those with two.

Lundberg and Lee (2017, p. 6) note that the weight is infinite with no switch
on or all switches on, which forces $\phi_0 = f_x(\varnothing)$ and $f(x) = \sum_{i=0}^{M}
\phi_i$: Property 1. It also notes that in practice the two infinite weights
are avoided by eliminating two unknowns with these two constraints. One way to
do that, used here: fix $\phi_0$ at the all-off output, write the last
feature's value as the total minus the others,
$\phi_M = \Delta - \sum_{i<M} \phi_i$ with $\Delta = f(x) - \phi_0$, and
substitute into Eq. 1. Each remaining setting $z'$ then gives one equation in
$M - 1$ unknowns,

$$f_x(z') - \phi_0 - z'_M\,\Delta = \sum_{i=1}^{M-1} \phi_i\,(z'_i - z'_M),$$

and the 14 settings with between one and three switches on are fitted by
least squares with the kernel weights. Here are all 14 equations, using the
line fill of the previous section, where (as shown below) the weights decide
the answer:

<details class="process-example">
<summary>Example in Python</summary>

```python
middle = [group for group in groups if 0 < len(group) < M]
phi0_line = value_line[()]
delta_line = value_line[tuple(range(M))] - phi0_line
design_rows = []
for group in middle:
    z = [1 if j in group else 0 for j in range(M)]
    design_rows.append([" ".join(str(v) for v in z), label(group),
                        *[z[j] - z[M - 1] for j in range(M - 1)],
                        value_line[group] - phi0_line - z[M - 1] * delta_line,
                        shapley_kernel(len(group))])
regression_design = pd.DataFrame(design_rows, columns=[
    "Setting z'", "Present", "T24 - P30", "T30 - P30", "T50 - P30",
    "Target (cycles)", "Kernel weight"])
```

</details>

<!-- sample: regression_design -->
| Setting z' | Present | T24 - P30 | T30 - P30 | T50 - P30 | Target (cycles) | Kernel weight |
|---|---|---|---|---|---|---|
| 1 0 0 0 | T24 | 1 | 0 | 0 | 23.234885 | 0.250000 |
| 0 1 0 0 | T30 | 0 | 1 | 0 | 2.596202 | 0.250000 |
| 0 0 1 0 | T50 | 0 | 0 | 1 | -36.551312 | 0.250000 |
| 0 0 0 1 | P30 | -1 | -1 | -1 | 7.492052 | 0.250000 |
| 1 1 0 0 | T24 + T30 | 1 | 1 | 0 | 16.789453 | 0.125000 |
| 1 0 1 0 | T24 + T50 | 1 | 0 | 1 | -17.262919 | 0.125000 |
| 1 0 0 1 | T24 + P30 | 0 | -1 | -1 | 19.384612 | 0.125000 |
| 0 1 1 0 | T30 + T50 | 0 | 1 | 1 | -26.935817 | 0.125000 |
| 0 1 0 1 | T30 + P30 | -1 | 0 | -1 | 9.414900 | 0.125000 |
| 0 0 1 1 | T50 + P30 | -1 | -1 | 0 | -14.008099 | 0.125000 |
| 1 1 1 0 | T24 + T30 + T50 | 1 | 1 | 1 | -12.987784 | 0.250000 |
| 1 1 0 1 | T24 + T30 + P30 | 0 | 0 | -1 | 18.558462 | 0.250000 |
| 1 0 1 1 | T24 + T50 + P30 | 0 | -1 | 0 | -2.657277 | 0.250000 |
| 0 1 1 1 | T30 + T50 + P30 | -1 | 0 | 0 | -9.185580 | 0.250000 |

Each row reads "the sum of the $\phi$ values of its three columns, each times
its entry, should equal the target". The row "T24", $z' = (1, 0, 0, 0)$, asks
for $\phi_{T24}$ to equal what T24 adds on its own, $f_x(z') - \phi_0$. The row
"P30", $z' = (0, 0, 0, 1)$, has $-1$ in every column and subtracts $\Delta$
from its target: it asks for $-\phi_{T24} - \phi_{T30} - \phi_{T50}$ to equal
$f_x(z') - \phi_0 - \Delta$, which, since the four values add to $\Delta$, is
the same as asking $\phi_{P30}$ to equal what P30 adds on its own. No set of
three numbers satisfies all 14 rows at once when the group values are not a
sum of one amount per feature, so least squares finds the compromise that
makes the weighted sum of squared misses smallest, and the kernel weight of
each row sets how much its miss counts.

### Run on engine 5

The regression is run with the Shapley kernel and, for comparison, with every
setting weighted equally, on both rules for missing readings:

<details class="process-example">
<summary>Example in Python</summary>

```python
def kernel_shap(value, weight):
    """Theorem 2: weighted least squares over the settings, the two infinite weights eliminated."""
    middle = [group for group in groups if 0 < len(group) < M]
    switches = np.array([[1.0 if j in group else 0.0 for j in range(M)] for group in middle])
    phi0 = value[()]
    total = value[tuple(range(M))] - phi0
    targets = np.array([value[group] for group in middle]) - phi0 - switches[:, -1] * total
    columns_left = switches[:, :-1] - switches[:, [-1]]
    # Multiplying a row by the square root of its weight multiplies its squared miss by the weight.
    root = np.sqrt([weight(len(group)) for group in middle])
    first, *_ = np.linalg.lstsq(columns_left * root[:, None], targets * root, rcond=None)
    return np.append(first, total - first.sum())

def equal_weight(size):
    """The comparison: every setting counts the same."""
    return 1.0

kernel_mean = pd.DataFrame({
    "Sensor": names,
    "Exact, Eq. 4": phi_mean,
    "Regression, Shapley kernel": kernel_shap(value_mean, shapley_kernel),
    "Regression, equal weights": kernel_shap(value_mean, equal_weight),
})
kernel_line = pd.DataFrame({
    "Sensor": names,
    "Exact, Eq. 4": phi_line,
    "Regression, Shapley kernel": kernel_shap(value_line, shapley_kernel),
    "Regression, equal weights": kernel_shap(value_line, equal_weight),
})
assert np.allclose(kernel_line["Regression, Shapley kernel"], phi_line)
```

</details>

With missing readings set to their means:

<!-- sample: kernel_mean -->
| Sensor | Exact, Eq. 4 | Regression, Shapley kernel | Regression, equal weights |
|---|---|---|---|
| T24 | 5.984611 | 5.984611 | 5.984611 |
| T30 | 0.667847 | 0.667847 | 0.667847 |
| T50 | -15.962813 | -15.962813 | -15.962813 |
| P30 | -0.863789 | -0.863789 | -0.863789 |

With missing readings filled by lines:

<!-- sample: kernel_line -->
| Sensor | Exact, Eq. 4 | Regression, Shapley kernel | Regression, equal weights |
|---|---|---|---|
| T24 | 14.756501 | 14.756501 | 14.219886 |
| T30 | 2.427071 | 2.427071 | 2.517487 |
| T50 | -27.273242 | -27.273242 | -26.942169 |
| P30 | -0.084474 | -0.084474 | 0.030652 |

With the mean fill, every weighting gives the exact values. That is not
Theorem 2 at work: there every group's value is exactly the all-off output plus
one fixed amount per present feature, so the regression fits all 14 settings
with no error whatever the weights. The theorem shows itself in the second
table. With the line fill the values are not additive, the regression cannot
fit every setting, and the weights decide the compromise: the Shapley kernel
lands on the exact values, equal weighting does not.

### Why a weighted regression can return Shapley values

Lundberg and Lee's (2017, p. 6) intuition is that Eq. 8 is a difference of
means, that the mean is the best least-squares estimate for a set of points, and that it is
therefore natural to look for a weighting under which least squares reproduces
Eq. 8. The proof that the Shapley kernel does so is in the Supplementary
Material, which this project does not hold. What follows makes the intuition
concrete; it is this page's illustration, not the paper's proof.

Eq. 4 for T50 is $\sum_S w(S)\,[v(S \cup \{T50\}) - v(S)]$, where $v$ is the
value of a group and $w(S)$ the weight of the order-weights table. It splits
into $\sum_S w(S)\,v(S \cup \{T50\}) - \sum_S w(S)\,v(S)$, and because the
eight weights add to 1, each of the two sums is a weighted average: of the
eight group values with T50 present, and of the eight without.

A least-squares line fitted to points whose only input is a 0/1 switch can set
its height at 0 and its height at 1 independently, and the single number
closest, in summed squared difference, to a set of points is their average. So
the fitted line passes through the average of the points at 0 and the average
of the points at 1, and its slope is the difference of the two averages; with
weights, the weighted averages. Under the line fill:

<details class="process-example">
<summary>Example in Python</summary>

```python
on_t50 = sum(order_weight(len(group)) * value_line[joined(group, t50)] for group in before_t50)
off_t50 = sum(order_weight(len(group)) * value_line[group] for group in before_t50)

# Sixteen points: each group value placed at 0 or 1 by T50's switch, weighted as in Eq. 4.
switch = np.array([0.0] * len(before_t50) + [1.0] * len(before_t50))
points = np.array([value_line[group] for group in before_t50]
                  + [value_line[joined(group, t50)] for group in before_t50])
root_weights = np.sqrt([order_weight(len(group)) for group in before_t50] * 2)
one_switch, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(switch)), switch]) * root_weights[:, None],
                                 points * root_weights, rcond=None)
difference_of_means = pd.DataFrame({
    "Quantity": [
        "Weighted average of the 8 group values with T50 present",
        "Weighted average of the 8 group values without T50",
        "Difference",
        "Slope of a weighted least-squares line on T50's switch alone",
        "T50's Shapley value, Eq. 4",
    ],
    "Value (cycles)": [on_t50, off_t50, on_t50 - off_t50, one_switch[1], phi_line[t50]],
})
assert np.isclose(one_switch[1], phi_line[t50])
```

</details>

<!-- sample: difference_of_means -->
| Quantity | Value (cycles) |
|---|---|
| Weighted average of the 8 group values with T50 present | 65.684596 |
| Weighted average of the 8 group values without T50 | 92.957838 |
| Difference | -27.273242 |
| Slope of a weighted least-squares line on T50's switch alone | -27.273242 |
| T50's Shapley value, Eq. 4 | -27.273242 |

For one feature at a time, then, a weighted least-squares fit with Eq. 4's
weights returns its Shapley value. Kernel SHAP fits all $M$ switches in one
regression, where each setting gets a single weight. Under Eq. 4 a setting with
$k$ switches on is weighted $w(k - 1)$ when it serves a feature that is on in
it and $w(k)$ when it serves one that is off (with four features, 1/4 and 1/12
for $k = 1$), so Eq. 4's weights cannot in general be carried over. Theorem 2
states that the Shapley kernel is the weighting under which the joint fit
still returns every Shapley value, which the second engine-5 table above
confirms on these numbers.

A footnote of Lundberg and Lee (2017, p. 6, footnote 2) points out that
econometrics, the statistics of economic data, had already stated Shapley
values in an equivalent way, as the answer to a minimisation under constraints
(its reference [2], p. 10, a 1988 work by Charnes and colleagues): the numbers
that make an expression built from squares and products of the unknowns as
small as possible under side conditions.
The paper also says (p. 6) that LIME's mapping is equivalent to the
approximation of Eq. 12, which is what lets a LIME-style regression estimate
SHAP values without reference to the model's insides.

### What other weightings get wrong

Every column of both engine-5 tables adds to −10.174144 cycles, because the two
constraints were imposed, so local accuracy alone does not single out the
Shapley values. By Theorem 1 the equal-weight numbers must then break
consistency for some pair of models (missingness cannot fail here, since every
switch of $x'$ is on). Here is such a pair. Take the line fill as
the mapping $h_x$; every group then feeds the model a different input row.
Build a second model that agrees with the straight line everywhere except at
the two rows used for the groups "T24 + T30" and "T24 + T30 + T50", where it
answers one cycle more. T50 joining "T24 + T30" adds exactly what it added
before (both values rose by 1), and every other amount T50 adds is untouched.
Eq. 7 therefore holds in both directions, and consistency requires T50's value
to stay the same:

<details class="process-example">
<summary>Example in Python</summary>

```python
rows_used = np.array([line_fill(group) for group in groups])
assert len(np.unique(rows_used, axis=0)) == len(groups)  # one distinct input row per group
raised = [(names.index("T24"), names.index("T30")),
          (names.index("T24"), names.index("T30"), t50)]
second_model = dict(value_line)
for group in raised:
    second_model[group] += 1.0
assert all(np.isclose(second_model[joined(group, t50)] - second_model[group],
                      value_line[joined(group, t50)] - value_line[group]) for group in before_t50)
consistency = pd.DataFrame({
    "Method": ["Exact, Eq. 4", "Regression, Shapley kernel", "Regression, equal weights"],
    "T50, straight line": [phi_line[t50], kernel_shap(value_line, shapley_kernel)[t50],
                           kernel_shap(value_line, equal_weight)[t50]],
    "T50, second model": [shapley(second_model)[t50], kernel_shap(second_model, shapley_kernel)[t50],
                          kernel_shap(second_model, equal_weight)[t50]],
})
consistency["Change"] = consistency["T50, second model"] - consistency["T50, straight line"]

full = tuple(range(M))
leave_one_out = [value_line[full] - value_line[tuple(j for j in range(M) if j != i)]
                 for i in range(M)]
local_accuracy = pd.DataFrame({
    "Sensor": [*names, "Sum"],
    "Leave one out (cycles)": [*leave_one_out, sum(leave_one_out)],
    "Shapley value (cycles)": [*phi_line, phi_line.sum()],
})
```

</details>

<!-- sample: consistency -->
| Method | T50, straight line | T50, second model | Change |
|---|---|---|---|
| Exact, Eq. 4 | -27.273242 | -27.273242 | 0.000000 |
| Regression, Shapley kernel | -27.273242 | -27.273242 | 0.000000 |
| Regression, equal weights | -26.942169 | -27.004669 | -0.062500 |

The equal-weight regression lowers T50's value by 0.0625 cycles although
nothing T50 adds has changed; the second model satisfies Eq. 7 against the
first, so Property 3 forbids the drop. The Shapley values do not move.

The opposite failure comes from a method that keeps consistency and gives up
local accuracy: **leave one out**, which gives each sensor what is lost when it
alone is switched off from the full input, $f(x) - f_x(x' \setminus i)$. (It
keeps consistency, in this page's reasoning, because that amount is one of the
amounts Eq. 7 compares.) The [Dynamic SHAP](../learn/10_dynamic_shap.md#3--average-what-each-sensor-adds)
page calls the same idea occlusion. Under the line fill:

<!-- sample: local_accuracy -->
| Sensor | Leave one out (cycles) | Shapley value (cycles) |
|---|---|---|
| T24 | 9.185580 | 14.756501 |
| T30 | 2.657277 | 2.427071 |
| T50 | -18.558462 | -27.273242 |
| P30 | 2.813640 | -0.084474 |
| Sum | -3.901964 | -10.174144 |

Leave-one-out values add to −3.901964 cycles, not to the −10.174144 between the
starting value and the prediction, so they break Property 1.

### Sampled settings: what the ten pages do not give

The paper gives the cost of the regression over all settings as $O(2^M + M^3)$
(Lundberg & Lee, 2017, p. 6, under Low-Order SHAP), meaning the work grows no
faster than $2^M + M^3$ does; that is practical only for small $M$. For larger
$M$, Kernel SHAP fits a sample of settings, but the main text does not say how
the settings are sampled, or how sampled settings are weighted, so the sampled
estimator cannot be rebuilt from these ten pages. The Figure 3 runs also add a
penalty on the coefficients (Figure 3's caption, p. 8, names a "debiased
lasso", and Section 5.1, p. 8, says Kernel SHAP's advantage holds particularly
when such a penalty is added), whereas Theorem 2 sets $\Omega = 0$. The
theorem's guarantee covers the full enumeration demonstrated here, not the sampled and penalised estimator
whose accuracy Section 5 measures.

**Do not confuse the Shapley kernel with the weight of Eq. 4.** Eq. 4's weight
multiplies what one feature adds to a group, and for four features is 1/4, 1/12,
1/12, 1/4 for groups of size 0 to 3 before that feature. The kernel multiplies
a squared miss of the whole regression, and for four features is 0.25, 0.125,
0.25 for settings with one, two or three switches on. They lead to the same
values by different routes.

**Do not confuse model-agnostic with model-specific estimation.** Shapley
sampling values and Kernel SHAP are **model-agnostic** (Lundberg & Lee, 2017,
p. 5): they use only inputs and outputs, so they apply to any model but learn only
what varying the input reveals. Section 4.2 (pp. 6–7) lists four methods for
specific model types. Three of them, Linear SHAP, Max SHAP and Deep SHAP, read
the model's form, its multipliers or its layers, and the paper presents them as
faster for that reason. The fourth, Low-Order SHAP, is the full regression of
Theorem 2; it needs only inputs and outputs, and what sets it apart is that $M$
is small.

The key idea is that a linear regression of the model's outputs on the on/off
switches, weighted by the Shapley kernel and forced through the all-off and
all-on outputs, returns the Shapley values as its coefficients.

---

## Model-specific shortcuts

### Linear SHAP

Corollary 1 (Lundberg & Lee, 2017, Section 4.2, p. 6): for a straight-line model
$f(x) = \sum_{j=1}^{M} w_j x_j + b$, assuming independent features (Eq. 11),

$$\phi_i(f, x) = w_i\,\big(x_i - E[x_i]\big).$$

The paper prints the right-hand side as $w_j(x_j - E[x_j])$, mixing the index
$i$ with $j$; it is read with $j = i$. It says the result follows from Theorem 2
and Eq. 11 and had been noted before by Štrumbelj and Kononenko (its
reference [9]).

- $w_i$ is feature $i$'s multiplier; $x_i$ its value in the explained input;
  $E[x_i]$ its average over the data.
- The product is the change in output caused by moving feature $i$ alone from
  its average to its actual value.

The reason it holds is the observation made under Eq. 4: in a straight-line
model with independent missing features, feature $i$ adds the same
$w_i(x_i - E[x_i])$ to every group, and the weighted average of a constant is
that constant. On engine 5:

<details class="process-example">
<summary>Example in Python</summary>

```python
linear_shap = pd.DataFrame({
    "Sensor": names,
    "Multiplier w": w,
    "x - E[x]": x - mean,
    "w (x - E[x])": w * (x - mean),
    "Exact, Eq. 4": phi_mean,
})
starting_values = pd.DataFrame({
    "Starting value": ["E[f(X)], from Property 1", "b, as Corollary 1 prints it"],
    "Value (cycles)": [f(readings).mean(), b],
    "Plus the four values": [f(readings).mean() + phi_mean.sum(), b + phi_mean.sum()],
    "Model prediction f(x)": [f(x), f(x)],
})
assert np.allclose(linear_shap["w (x - E[x])"], phi_mean)
```

</details>

<!-- sample: linear_shap -->
| Sensor | Multiplier w | x - E[x] | w (x - E[x]) | Exact, Eq. 4 |
|---|---|---|---|---|
| T24 | -14.563452 | -0.410934 | 5.984611 | 5.984611 |
| T30 | -1.145302 | -0.583119 | 0.667847 | 0.667847 |
| T50 | -1.531026 | 10.426218 | -15.962813 | -15.962813 |
| P30 | 11.115376 | -0.077711 | -0.863789 | -0.863789 |

**The starting value in Corollary 1.** The corollary also prints
$\phi_0(f, x) = b$, the intercept. Property 1 defines $\phi_0$ as the output
with every switch off, which under Eq. 11 is
$E[f(z)] = b + \sum_j w_j E[x_j]$. The two agree only when
$\sum_j w_j E[x_j] = 0$, for example when every feature has average zero, which
the corollary does not state. With raw readings they are far apart:

<!-- sample: starting_values -->
| Starting value | Value (cycles) | Plus the four values | Model prediction f(x) |
|---|---|---|---|
| E[f(X)], from Property 1 | 86.829286 | 76.655142 | 76.655142 |
| b, as Corollary 1 prints it | 7274.336749 | 7264.162604 | 76.655142 |

Only Property 1's starting value reconstructs the prediction. This page uses it,
and treats the printed $\phi_0 = b$ as holding for features centred on zero.

### Max SHAP and Deep SHAP

Two further shortcuts are described in words here and not demonstrated, because
this project uses neither.

**Max SHAP** (Lundberg & Lee, 2017, Section 4.2, p. 7) computes the Shapley
values of a function that returns the largest of its $M$ inputs, by working
through the inputs in sorted order and computing the probability that each one
raises the maximum. The paper gives its cost as $O(M^2)$ instead of $O(M 2^M)$,
and leaves the full algorithm to the Supplementary Material. Figure 4B, under
the paper's evidence below, is a function of this kind, worked by hand.

**Deep SHAP** (Lundberg & Lee, 2017, Section 4.2, p. 7) starts from a
connection with DeepLIFT: if DeepLIFT's reference value is read as the average input $E[x]$ of Eq. 12, DeepLIFT
approximates SHAP values under the assumptions of independent features and a
linear model. Deep SHAP computes SHAP values exactly for small pieces of a
network whose SHAP values can be written down by a formula: a linear layer,
max pooling (a layer that keeps the largest of a few neighbouring values), or a
function of one input. It turns each into a **multiplier**, the piece's SHAP value divided by
its input's distance from that input's average (Eqs. 13–14). It then combines
the multipliers along the paths through the network by the chain rule of
calculus (Eq. 15): the rate of change along a path of computations is the
product of the rates of its links, and the products of several paths are
added. Finally it multiplies by the input's distance from its average
(Eq. 16). Figure 2B draws this for a network of three pieces. The result is an
approximation for the whole network, built from exact values for its pieces.

---

## The paper's evidence

Lundberg and Lee (2017, [Section 5,
pp. 8–9](../references/lundberg_2017_unified_shap.pdf#page=8)) support three claims.

**Kernel SHAP estimates are better for a given number of model evaluations.**
Figure 3 of Lundberg and Lee (2017, p. 8) compares Kernel SHAP, fitted with
what the paper calls a debiased lasso, Shapley sampling values, and the
open-source implementation of LIME. (A lasso is a regression whose penalty on the sizes of the coefficients
sets small ones to exactly zero; the paper does not say how it removes the
penalty's bias.) Each method estimates the value of one feature in two
decision-tree models (models that predict by a sequence of yes/no threshold
tests), one using all 10 of its input features and one using 3 of 100, as the
number of model evaluations grows; the figure shows the 10th and 90th
percentiles of 200 repeated estimates at each size (the values below which
10% and 90% of the estimates fall). The paper reads it as
Kernel SHAP reaching accurate estimates with fewer evaluations, and LIME's
values differing substantially from the Shapley values.

**SHAP values agree better with human explanations.** Figure 4 of Lundberg and
Lee (2017, p. 9) reports a study with two small models, run on Amazon
Mechanical Turk (Section 5.2, p. 8), a paid online service on which people
complete small tasks. In the first, a sickness
score is 2 when a patient has both fever and cough, 5 when only one of them,
and 0 otherwise; a third input, congestion, plays no part. In the second, three
men are paid according to the best score any of them achieved, with 5, 4 and 0
questions right and a profit of 5 dollars. The figure sets the most common
explanation given by 30 and 52 participants respectively beside SHAP and LIME
in both panels, and beside the original DeepLIFT in the second, and the paper
reports much stronger agreement between the people and SHAP. The study takes
agreement with people who understand the model as its measure of a good
explanation, an assumption the paper states (Section 5.2, p. 8).

Both models are small enough to compute exactly with Eq. 4. Taking "no input
present" as the reference, a score of 0 and a profit of 0 (this page's choice;
the paper does not state one), every group of inputs has a value, and with
three inputs the weights for groups of size 0, 1 and 2 are 1/3, 1/6 and 1/3:

<details class="process-example">
<summary>Example in Python</summary>

```python
def game_shapley(players, value):
    """Equation 4 for a small game given as a value for every group of players."""
    n = len(players)
    result = []
    for player in players:
        others = [other for other in players if other != player]
        total = 0.0
        for size in range(n):
            for group in combinations(others, size):
                weight = factorial(size) * factorial(n - size - 1) / factorial(n)
                total += weight * (value(set(group) | {player}) - value(set(group)))
        result.append(total)
    return result

def sickness(present):
    """Figure 4A's model: 2 with fever and cough, 5 with exactly one of them, 0 otherwise."""
    return {0: 0, 1: 5, 2: 2}[len(present & {"Fever", "Cough"})]

def profit(present):
    """Figure 4B's model: the best score among the men present, 0 if none is."""
    scores = {"Man 1": 5, "Man 2": 4, "Man 3": 0}
    return max([scores[man] for man in present], default=0)

figure_4 = pd.DataFrame({
    "Panel": ["A", "A", "A", "B", "B", "B"],
    "Input": ["Fever", "Cough", "Congestion", "Man 1", "Man 2", "Man 3"],
    "Shapley value": game_shapley(["Fever", "Cough", "Congestion"], sickness)
                     + game_shapley(["Man 1", "Man 2", "Man 3"], profit),
})
```

</details>

<!-- sample: figure_4 -->
| Panel | Input | Shapley value |
|---|---|---|
| A | Fever | 1.000000 |
| A | Cough | 1.000000 |
| A | Congestion | 0.000000 |
| B | Man 1 | 3.000000 |
| B | Man 2 | 2.000000 |
| B | Man 3 | 0.000000 |

Fever's value, term by term over the groups it can join, is
$\tfrac13(5 - 0) + \tfrac16(2 - 5) + \tfrac16(5 - 0) + \tfrac13(2 - 5) = 1$, for
the groups none, cough, congestion, and cough with congestion. Congestion never
changes the score, so it receives 0 and leaves the other two values as they
would be without it: with fever and cough alone, fever's value is
$\tfrac12(5 - 0) + \tfrac12(2 - 5) = 1$ as well. Man 1's is
$\tfrac13(5 - 0) + \tfrac16(5 - 4) + \tfrac16(5 - 0) + \tfrac13(5 - 4) = 3$.
Rendered at eight times its printed size, Figure 4 shows the people's bars and
SHAP's at these values: 1.0 for fever and cough and zero for congestion, and 3,
2 and zero for the three men. LIME's bars for fever and cough are below zero,
and the original DeepLIFT gives all 5 dollars to the first man.

**Better SHAP estimates explain class differences better.** Figure 5 of
Lundberg and Lee (2017, p. 9) explains a pre-trained network that recognises
handwritten digits (the MNIST dataset; Bach et al., 2015, Abstract, p. 1), the
same as in the paper's reference [7]: two convolution layers,
which slide small learned filters across the image, then two fully connected
layers (every input joined to every output) and ten outputs, one per digit.
SHAP, computed with Kernel SHAP, and LIME each use 50,000 samples. Each method
picks 20% of the pixels to mask so as to turn an 8 into a 3, and the change in
log-odds (the logarithm of a class's probability divided by one minus it) over
20 random images compares an original and an updated DeepLIFT, SHAP and LIME.
Rendered at eight times its printed size, Figure 5B shows SHAP's box and its
coloured middle bar highest, the original DeepLIFT's lowest, and the updated
DeepLIFT's and LIME's between them. The boxes overlap, the paper does not say
what the boxes and bars mark, and it reports no statistical test; the caption
reads the result as support for better estimates of SHAP values.

The three comparisons use two decision trees whose training the main text does
not describe, two hand-built functions, and one pre-trained digit-recognition
network evaluated on 20 images. None of them concerns a regression on sensor
time series.

---

## What this project takes from it

**The equation.** Wen et al. (2026, [Section 2.4,
p. 7](../references/wen_2026_estgnn_dynamic_shap.pdf#page=7)) define their
explanation. Wen et al.'s Eq. 14 is Lundberg and Lee's Definition 1 written for
the sensor nodes, Wen et al.'s Eq. 15 is Lundberg and Lee's Eq. 4 with $n = 14$
sensors and $f(S)$ the network's prediction when only the sensors in $S$ are
considered, and Wen et al.'s Eq. 16 repeats it at each time step, which is what
makes it dynamic. The project computes Wen et al.'s Eq. 15 with Eq. 4's weight,
`coalition_weights` in `explain/shap.py`; for four sensors that function returns
1/4, 1/12, 1/12 and 1/4, the "Weight of each group" column of the order-weights
table above.

**Do not confuse Lundberg and Lee's Eq. 15 with Wen et al.'s Eq. 15.** The first
is Deep SHAP's chain rule, which the project does not use. The second is the
Shapley sum the project computes.

**Where the project sits in Lundberg and Lee's terms.** It evaluates all $2^{14}
= 16{,}384$ groups of sensors for each window, so it uses none of Section 4's
estimators: no Shapley sampling, no Kernel SHAP, no Linear, Max or Deep SHAP.
Wen et al. (2026, Section 2.4) do not say what replaces a sensor outside $S$;
the project replaces its whole window of readings by the sensor's average
training window (assumption A-20). In Lundberg and Lee's terms that is Eq. 12,
independence plus the output at the average reading, applied to a network that
is not linear. The project's numbers are therefore exact Shapley values of that
replacement rule, not the conditional SHAP values of Eqs. 9–10 and not the
background averages of Eq. 11. The mechanics are on the Dynamic SHAP page,
[steps 1 to 3](../learn/10_dynamic_shap.md#1--choose-a-reference-window).

**The starting value.** Three different quantities carry that name. Lundberg and
Lee (2017, Fig. 1, p. 5) call the base value $E[f(z)]$, the average prediction.
Wen et al. (2026, Eq. 14, p. 7) describe $\phi_0$ as the average of the target
variable. The project uses the prediction at the average training window, which
Property 1 forces under its replacement rule (assumption A-24; see [step
4](../learn/10_dynamic_shap.md#4--check-that-the-contributions-add-up) of the
Dynamic SHAP page). For this page's straight line all three coincide, as the
first table shows; for the project's network they do not, because the average of
a non-linear network's outputs is in general not its output at the average
input, and [the record of that choice](../assumptions/A-24.md) holds the gap
measured on one trained network.

---

## What SHAP values do not establish

### They depend on the reference

The same model, the same engine and the same rule, with a different set of
rows supplying the averages: below, only the training rows recorded 125 or
more cycles before their engine failed, whose label therefore sits at the
125-cycle ceiling. With the straight-line model, Linear SHAP gives the values
directly.

<details class="process-example">
<summary>Example in Python</summary>

```python
plateau = readings[labels == ceiling]
plateau_mean = plateau.mean(axis=0)

def reference_column(rows, average):
    """Linear SHAP against one reference, then its starting value and the sum check."""
    values = w * (x - average)
    return pd.Series([len(rows), *values, f(average), f(average) + values.sum()],
                     dtype=object)

reference_change = pd.DataFrame({
    "Row": ["Training rows averaged", *names, "Starting value",
            "Starting value plus the four values"],
    "Reference: all training rows": reference_column(readings, mean),
    "Reference: rows at the ceiling": reference_column(plateau, plateau_mean),
})
```

</details>

<!-- sample: reference_change -->
| Row | Reference: all training rows | Reference: rows at the ceiling |
|---|---|---|
| Training rows averaged | 20631 | 8131 |
| T24 | 5.984611 | 1.883631 |
| T30 | 0.667847 | -3.150836 |
| T50 | -15.962813 | -24.780595 |
| P30 | -0.863789 | -6.969030 |
| Starting value | 86.829286 | 109.671973 |
| Starting value plus the four values | 76.655142 | 76.655142 |

Both columns satisfy local accuracy, yet T30 changes sign and T50's share grows
by more than half. Theorem 1's uniqueness holds once the mapping is fixed
(Lundberg & Lee, 2017, Section 3, p. 4); choosing the mapping, and the data
behind it, is outside the theorem. A SHAP value answers "compared with what?",
and the reference is part of the answer.

### They depend on what "missing" means when features are related

The dependence table above showed the same model and engine giving T50 a value
of −15.962813 cycles under Eqs. 11–12 and −27.273242 under the line estimate of
Eq. 10. Neither breaks the three properties. They answer different questions:
the first asks what the model does when one reading alone is moved to typical;
the second what the model expects once a reading is known, given what usually
accompanies it.

### They are neither causes nor a check on accuracy

The general point is made on the [XAI](../learn/03_explainability.md) and
[Dynamic SHAP](../learn/10_dynamic_shap.md#4--check-that-the-contributions-add-up)
pages; this example adds two specifics. T24 receives +5.984611 cycles because
engine 5's T24 is 0.410934 degrees Rankine below the training mean and the
fitted multiplier, −14.563452 cycles per degree, is negative; that multiplier
is a least-squares fit over readings correlated with one another, an
association across training rows, so it does not say that cooling T24 in a real
engine would add life. And the explanation reproduces the prediction of
76.655142 cycles exactly, although engine 5's recorded remaining life is 91
cycles, 14.344858 more: local accuracy means matching the model, not the truth.

### They are local

Each set of values explains one prediction (Lundberg & Lee, 2017, Section 2,
p. 2). An average of values over many predictions, such as a sensor ranking, is a
different object, a **global** summary, and inherits every choice above.

### SHAP in seven answers

| Question | Answer (this page's reading of the paper) |
|---|---|
| What is explained? | One prediction $f(x)$ of one model |
| What does the method see? | Kernel SHAP, Shapley sampling and Low-Order SHAP: only inputs and outputs. Linear, Max and Deep SHAP: the model's form, multipliers or layers |
| What does it change? | Which features are present; missing ones are replaced as the mapping says (Eqs. 9–12) |
| What does it measure? | What each feature adds to the (expected) prediction when it joins a group, averaged over arrival orders |
| What comes out? | One signed number per feature, in the output's unit, plus a starting value; they add to $f(x)$ |
| What does it assume? | A mapping and the data behind it; for Eqs. 11–12, independent features and, for Eq. 12, a linear model; and that Properties 1–3 are the right requirements, which the paper calls desirable (Lundberg & Lee, 2017, p. 4) |
| What does it not prove? | This page's conclusion; the paper does not discuss it: that a feature causes the outcome, that the model is accurate, or that the split is the only reasonable one once the mapping is open to choice |

---

## References

- Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., & Samek, W.
  (2015). On pixel-wise explanations for non-linear classifier decisions by
  layer-wise relevance propagation. *PLOS ONE*, *10*(7), Article e0130140.
  https://doi.org/10.1371/journal.pone.0130140
  [Copy](../references/bach_2015_layerwise_relevance_propagation.pdf)
- Frederick, D. K., DeCastro, J. A., & Litt, J. S. (2007). *User's guide for
  the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)*
  (NASA/TM-2007-215026). NASA Glenn Research Center.
  https://ntrs.nasa.gov/citations/20070034949. Page numbers on this page are
  those of the PDF file, not the report's printed page numbers.
  [Copy](../references/frederick_2007_cmapss_users_guide.pdf)
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. In I. Guyon, U. von Luxburg, S. Bengio, H. Wallach,
  R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), *Advances in Neural
  Information Processing Systems 30 (NIPS 2017)* (pp. 4765–4774).
  https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
  [Copy](../references/lundberg_2017_unified_shap.pdf). The copy held is the
  10-page proceedings paper, numbered 1 to 10; equation, section, figure and
  page numbers on this page refer to it. Its Supplementary Material is not held.
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
  [Copy](../references/schnake_2022_gnn_lrp_relevant_walks.pdf). The copy held
  is arXiv version 3.
- Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
  (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
  enhancement for aeroengine RUL prediction. *Aerospace Science and
  Technology*, *168*, 110875. https://doi.org/10.1016/j.ast.2025.110875
  [Copy](../references/wen_2026_estgnn_dynamic_shap.pdf)
