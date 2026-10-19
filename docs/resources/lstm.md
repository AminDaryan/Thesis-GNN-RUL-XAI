# LSTM (1997)

Wen et al. (2026, [Section 2](../references/wen_2026_estgnn_dynamic_shap.pdf#page=4))
put a **long short-term memory network**, or **LSTM**, in front of
their graph layers. An LSTM is a **recurrent network**: it reads a sequence one
step at a time and carries numbers from each step to the next, and it is built
so that what it carries can survive for many steps. This page explains it from
the papers that introduced it.
Hochreiter and Schmidhuber (1997, [Abstract](../references/hochreiter_1997_long_short_term_memory.pdf#page=1))
reviewed Hochreiter's 1991 analysis of why ordinary recurrent networks could
not learn to carry information far, and built the memory cell to get round it.
Gers et al. (2000, [Abstract](../references/gers_2000_learning_to_forget.pdf#page=2))
added the forget gate, which the 1997 cell did not have and every LSTM cell in
this project has. A short note near the end separates the LSTM from the unit
proposed by Cho et al. (2014, [Section 2.3](../references/cho_2014_rnn_encoder_decoder.pdf#page=3)),
which PyTorch calls a gated recurrent unit and which the project also offers
(assumption A-36).

How this project runs the stage, cycle by cycle on real engine data, is taught
in [Temporal algorithm](../learn/09_temporal.md#2--read-cycles-with-memory).
This page does not repeat it. It is the papers' account, and its last section
says which parts of each paper the project's code inherits.

Section, equation, table and figure numbers below are those of each paper. The
1997 and 2000 files are the authors' own manuscripts rather than the journal's
typeset pages, so they are cited by section and equation, and the one
quotation from the 2000 paper by the page of the PDF file; links open the PDF
at the page where the passage is.

## The problem the 1997 paper set out to solve

### Memory in a recurrent network

A recurrent network is made of **units**. Each unit holds one number, its
**activation**, and recomputes it at every time step from the activations of
the step before.
Hochreiter and Schmidhuber (1997, [Section 3.1](../references/hochreiter_1997_long_short_term_memory.pdf#page=3))
write the rule as

$$\mathrm{net}_i(t) = \sum_j w_{ij}\, y^j(t-1), \qquad y^i(t) = f_i\big(\mathrm{net}_i(t)\big)$$

- $y^j(t-1)$ is the activation of unit $j$ at the previous step. The sum runs
  over every unit $j$ with a connection into unit $i$, and that can include
  unit $i$ itself.
- $w_{ij}$ is the **weight** on the connection from $j$ to $i$: a multiplier
  that training adjusts.
- $\mathrm{net}_i(t)$, which the paper calls the **net input**, is the weighted
  sum of everything arriving at unit $i$.
- $f_i$ is the unit's **activation function**, a fixed curve that turns the net
  input into the new activation.

With invented numbers: a unit with weight 0.5 from itself and 2.0 from an input
unit, whose own activation was 0.4 and whose input was 0.3, gets
$\mathrm{net} = 0.5 \times 0.4 + 2.0 \times 0.3 = 0.8$, and its new activation
is $f(0.8)$. That new activation is then part of the next step's sum. This is
how a number can be carried forward: each step's activations are inputs to the
next. Hochreiter and Schmidhuber (1997, Section 1) call the activations the
network's **short-term memory**, as opposed to the long-term memory held in
the slowly changing weights. The
question the paper asks is not whether such a network *can* hold a number for
a long time, but whether training can *find* weights that make it hold the
right number for as long as it is needed.

### What training differentiates

Training compares the network's output with a target and changes the weights
to shrink the difference. The quantity being made smaller is the **error**,
here the squared difference between targets and outputs (Hochreiter &
Schmidhuber, 1997, [Appendix A.1, Eq. 16](../references/hochreiter_1997_long_short_term_memory.pdf#page=26)). During one
calculation the input sequence and its targets are held fixed, and what can
change is the weights. The **derivative** of the error with respect to one
weight says how much the error would change if that weight alone were nudged
up, and its sign says in which direction. Such a derivative exists because
every step from a weight to the error, the multiplications, the sums and the
smooth activation curves, has a slope. The list of these derivatives, one per
weight, is the **gradient**, and **gradient descent** moves each weight a small
step against its own derivative (Gers et al., 2000, [Section 3.2, Eq. 9](../references/gers_2000_learning_to_forget.pdf#page=7)).

To get those derivatives, Hochreiter and Schmidhuber (1997, Section 3.1) work
with an **error signal** $\vartheta_j(t)$ for each unit $j$ and time $t$. It is
*minus* the derivative of the error with respect to that unit's net input at
that time: how much the error would fall per unit increase of the net input.
The appendix states this definition outright for its own symbol $e_l$
([Appendix A.1, Eq. 18](../references/hochreiter_1997_long_short_term_memory.pdf#page=26)).
For an output unit $k$ with target $d_k(t)$:

$$\vartheta_k(t) = f_k'\big(\mathrm{net}_k(t)\big)\,\big(d_k(t) - y^k(t)\big)$$

and for any other unit $j$:

$$\vartheta_j(t) = f_j'\big(\mathrm{net}_j(t)\big) \sum_i w_{ij}\, \vartheta_i(t+1)$$

- $f'$ is the **slope** of the activation function at the unit's net input.
- $d_k(t) - y^k(t)$ is how far output $k$ is from its target.
- The sum in the second formula runs over the units $i$ that unit $j$ feeds,
  one step later, each error signal weighted by the connection it came through.

The first formula is minus a derivative. Write the error of output $k$ as half
its squared difference, $\tfrac12 (d - y)^2$, as Gers et al. (2000,
[Eq. 8](../references/gers_2000_learning_to_forget.pdf#page=7)) do. Its derivative
with respect to the output $y$ is $-(d - y)$, and the chain rule of calculus
multiplies that by the slope $f'$ to reach the net input; the minus sign of the
definition removes the one in front. The 1997 appendix writes its squared error
without the half, which would double every error signal, yet gives the formula
above for output units (Hochreiter & Schmidhuber, 1997, Appendix A.1, Eqs. 16
and 19), so this page uses the halved form that makes the two agree. With
invented numbers: a target of 1, an output of 0.8, and the logistic curve
introduced below, whose slope at an output $y$ is $y\,(1 - y) = 0.8 \times 0.2 = 0.16$, give
$\vartheta = 0.16 \times (1 - 0.8) = 0.032$. It is positive: raising this
unit's net input would lower the error.

The second formula is the chain rule applied once: a change in unit $j$'s net
input changes its activation by the slope $f_j'$, that activation reaches unit
$i$ through the weight $w_{ij}$, and unit $i$'s error signal says what that
does to the error. With invented numbers, a slope of 0.2, one weight of 0.5 and
an error signal of 1.0 one step later give
$\vartheta_j(t) = 0.2 \times 0.5 \times 1.0 = 0.1$.

Hochreiter and Schmidhuber (1997, Section 3.1) then give each weight its
share. Time $t$'s contribution to the
update of the weight $w_{jl}$, from a unit $l$ into unit $j$, is
$\alpha\, \vartheta_j(t)\, y^l(t-1)$, where $\alpha$ is the **learning rate**,
the size of the step, and the contributions of all time steps are added
together (Appendix A.1 says so after its Eq. 28). Because $\vartheta$ already
carries the minus sign, adding $\alpha\,\vartheta\,y$ moves the weight against
its derivative, as gradient descent requires. With $\alpha = 0.1$, the error
signal 0.1 above and $y^l(t-1) = 0.5$, time $t$ adds
$0.1 \times 0.1 \times 0.5 = 0.005$ to the weight.

Applying the second formula over and over carries the error signal back
through the steps of the sequence. That is **back-propagation through time**
(BPTT). Hochreiter and Schmidhuber (1997, Section 1) name it together with
**real-time recurrent learning** (RTRL), whose memory does not grow with the
length of the sequence (Section 4, footnote 3). RTRL reaches the same
derivatives working forwards: at every step it updates, for every weight, how
much the units' activations depend on that weight. The paper does not spell out
how RTRL in general works, so that description is this page's summary. Eq. 27 further down is a
derivative carried forwards in this way. Section 5 notes that the two give
exactly the same gradient when BPTT is not cut short and RTRL changes the
weights only at the end of each sequence. Section 1 says both suffer from the
problem below.

**Do not confuse the error signal with the error.** The error is one number
measuring how wrong the outputs are. An error signal is minus a derivative of
it, one per unit and time step, and it is what travels backwards.

### How far an error signal travels (Eqs. 1 and 2)

The rest of Section 3.1 is the paper's outline of the analysis in Hochreiter's
1991 thesis, which the paper cites and this project does not hold. Its question
is how much of an error signal at one unit, at the last step, still arrives at
a unit several steps earlier. Eq. 1 expresses the answer for $q$ steps back
through the answer for $q - 1$ steps, and Eq. 2, which the paper proves from it
by induction, writes it out in full (Hochreiter & Schmidhuber, 1997,
[Section 3.1, Eq. 2](../references/hochreiter_1997_long_short_term_memory.pdf#page=4)):

$$\frac{\partial \vartheta_v(t-q)}{\partial \vartheta_u(t)} = \sum_{l_1=1}^{n} \cdots \sum_{l_{q-1}=1}^{n} \;\prod_{m=1}^{q} f'_{l_m}\big(\mathrm{net}_{l_m}(t-m)\big)\, w_{l_m l_{m-1}}, \qquad l_0 = u,\; l_q = v$$

- $u$ is the unit where the error signal starts, at time $t$; $v$ is the unit
  it is traced back to, $q$ steps earlier.
- $n$ is the number of units that are not inputs.
- $l_1, \dots, l_{q-1}$ are the units the signal passes through on the way, one
  per step. Each choice of them is one **path** from $u$ back to $v$, and the
  sums add up all $n^{q-1}$ paths.
- Along one path, step $m$ contributes one **factor**: the slope of the unit
  passed through at time $t-m$, times the weight crossed.
- The left side is the **scaling factor**: by how much the error signal at
  $v$, $q$ steps back, changes per unit of error signal at $u$ now.

In words: the signal that arrives $q$ steps back is the sum, over all paths,
of the product of the factors along the path. The equation is needed because it
makes the dependence on $q$ visible: $q$ appears as the number of factors
multiplied together.

Eq. 2 is copied above as the manuscript prints it, and its weight index is the
wrong way round. By the paper's own convention $w_{ij}$ is the weight from $j$
to $i$, and Eq. 1's one-step case is $f'_v\,w_{uv}$, the weight from $v$ to
$u$ (Hochreiter & Schmidhuber, 1997, Section 3.1, Eq. 1). Eq. 2 with $q = 1$
gives $w_{l_1 l_0} = w_{vu}$ instead. The weight an error actually crosses at
step $m$ is $w_{l_{m-1} l_m}$, the forward connection from the earlier unit $l_m$ to the later unit $l_{m-1}$. Nothing below depends
on this: for a single unit connected to itself, as in the example, both
readings name the same weight.

For a single unit connected to itself there is one path, and the formula
reduces to a product of $q$ factors $f'(\mathrm{net}(t-m))\,w$. If every factor
is larger than 1 in size, the product grows exponentially with $q$ and the
signal **blows up**, which, Hochreiter and Schmidhuber (1997, Section 3.1)
warn, may make the weights swing back and forth and the learning unstable. If
every factor is smaller than 1, the product shrinks
exponentially and the signal **vanishes**: nothing learned at the last step
reaches the earlier ones in useful size (Hochreiter & Schmidhuber, 1997,
Section 3.1). With invented numbers, ten factors of
0.25 multiply to $0.25^{10} = 1/1{,}048{,}576$, less than one millionth.

Which case happens depends on the activation function. The paper uses the
**logistic sigmoid**, $f(x) = 1/(1 + e^{-x})$ (Hochreiter & Schmidhuber, 1997,
[Appendix A.1, Eq. 3](../references/hochreiter_1997_long_short_term_memory.pdf#page=24)),
an S-shaped curve that goes from 0 to 1. Its slope is $f(x)\,(1 - f(x))$,
largest at $x = 0$, where it is $0.5 \times 0.5 = 0.25$. Section 3.1 draws three
conclusions from this:

- **Below a weight of 4.0, the signal tends to vanish.** The size of a factor
  $f'\,w$ stays below 1.0 whenever the weight's size is below 4.0, because
  $0.25 \times 4.0 = 1$.
- **Larger weights do not help.** As the weight grows without limit, the slope
  falls to zero faster than the weight grows: a large weight pushes the net
  input into the flat ends of the curve.
- **A larger learning rate does not help either.** It scales the long-range
  and the short-range error signals alike, so the ratio between them stays the
  same.

For a whole network the section also bounds the scaling factor (Hochreiter &
Schmidhuber, 1997, [Section 3.1](../references/hochreiter_1997_long_short_term_memory.pdf#page=5)):
if every weight is smaller in size than $4.0/n$, it is at most $n\,\tau^q$ with
$\tau = n\,w_{\max}/4.0 < 1$, where $w_{\max}$ is the largest weight size. The
bound shrinks exponentially in $q$. The paper calls it a weak bound, since it
is reached only when every slope is at its largest and every path has the same
sign.

Wen et al. (2026, [Section 2.2](../references/wen_2026_estgnn_dynamic_shap.pdf#page=4))
motivate their recurrent stage by this problem, which they call the vanishing
gradient problem.

### Worked example: one unit reading T50

The examples on this page use real readings of one sensor, **T50**, a
temperature measured at the outlet of the engine's low-pressure turbine
(Saxena et al., 2008, [Table 2](../references/saxena_2008_damage_propagation_modeling.pdf#page=3)).
They come from training engine 1 of **FD001** (NASA Prognostics Center of
Excellence [NASA PCoE], 2008), the part of NASA's simulated-engine data,
C-MAPSS (Saxena et al., 2008, Section III.A, p. 2), that this project uses
([C-MAPSS](cmapss.md)). The data hold one row
per engine per **cycle** (NASA PCoE, 2008), which this project reads as one
flight ([What one cycle is](../learn/01_the_problem.md#what-one-cycle-is)), and
engine 1 ran for 192 cycles. The readings are prepared the way the project
prepares every reading:
[smoothed](../learn/04_the_pipeline.md#2--smooth-each-sensor) (Wen et al.,
2026, Section 3.2.3, Eq. 18; assumption A-04), then
[rescaled](../learn/04_the_pipeline.md#rescale-each-sensor) by subtracting the
sensor's average over the training engines and dividing by its **standard
deviation** there, the typical distance of a reading from that average
(assumptions A-03 and A-25). A value of 2 is two standard deviations above the
training average.

The demonstration unit is one logistic unit connected to itself with weight
$w$. At each cycle its net input is $w$ times its previous activation plus 1.0
times the cycle's reading, and its activation starts at 0 before the first
cycle read. It reads engine 1's last 30 cycles, 163 to 192. This is an
**untrained demonstration**: its two multipliers are set by hand, and it
predicts nothing. At each cycle the factor is $f'(\mathrm{net})\,w$, and the
product for $q$ steps back multiplies the factors at cycles 191, 190, …, down
to $192 - q$: Eq. 2 with one unit.

The code computes the products twice. Once from the formula, and once with
PyTorch's **automatic differentiation**, which records every arithmetic
operation as it runs and then applies the chain rule backwards through all of
them to get exact derivatives. The two agree.

Every Python block on this page needs PyTorch, because the setup block imports
it; the first two worked examples do their own arithmetic in NumPy. During
preprocessing the project logs a warning that one sensor, P15, carries no
usable variation: it takes only two distinct values in the FD001 training
data, although how often it sits at the lower one still changes weakly with
wear (finding F-01). The warning does not affect this page.

<details class="process-example">
<summary>Set up the Python examples</summary>

```python
import numpy as np
import pandas as pd
import torch
from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.data.sensors import symbol_to_column
from thesis_gnn_rul_xai.models.temporal import LSTMStage

# Smoothed and rescaled FD001 training table; the rescaling is fitted on the
# training engines that seed 42 selects.
prepared = run_preprocessing(PreprocessingConfig(seed=42), keep_frames=True)
scaled = prepared.frames["train"]
t50 = symbol_to_column("T50")
engine_1 = scaled[scaled["unit"] == 1]
# NumPy arrays of shape (192,): engine 1's cycle numbers and its T50 readings.
cycles_1 = engine_1["cycle"].to_numpy()
t50_1 = engine_1[t50].to_numpy(dtype=np.float64)
assert len(t50_1) == 192 and 1 in prepared.train.units

def sigmoid(z):
    """The logistic function 1 / (1 + e^-z): any number in, a number between 0 and 1 out."""
    return 1.0 / (1.0 + np.exp(-z))
```

</details>

<details class="process-example">
<summary>Example in Python</summary>

```python
cycles = cycles_1[-30:]      # NumPy array of shape (30,): cycles 163 to 192
readings = t50_1[-30:]       # the T50 reading at each of those cycles
steps = [1, 2, 3, 5, 10, 20, 29]

def factors(self_weight):
    """Eq. 2's factor f'(net) * w at each of the 30 cycles, for one logistic unit fed back to itself."""
    output, result = 0.0, []
    for reading in readings:
        output = sigmoid(self_weight * output + 1.0 * reading)
        result.append(output * (1.0 - output) * self_weight)
    return np.array(result)

def products(factor):
    """For each q in steps, the product of the q factors before the last cycle."""
    return np.array([np.prod(factor[-1 - q:-1]) for q in steps])

def autograd_ratios(self_weight, identity=False):
    """The same products, read off PyTorch's automatic differentiation."""
    output, nets = torch.zeros((), dtype=torch.float64), []
    for reading in torch.tensor(readings, requires_grad=True):
        net = self_weight * output + 1.0 * reading
        net.retain_grad()
        output = net if identity else torch.sigmoid(net)
        nets.append(net)
    output.backward()
    # d(output at 192)/d(net at 192 - q), divided by d(output at 192)/d(net at 192).
    return np.array([(nets[-1 - q].grad / nets[-1].grad).item() for q in steps])

weak, strong = factors(1.0), factors(3.0)
np.testing.assert_allclose(weak[:2], [0.154596, 0.104151], atol=5e-7)   # written out below
assert (strong[1:] < weak[1:]).all()   # the 29 cycles after the first, 163
np.testing.assert_allclose(autograd_ratios(1.0), products(weak), rtol=1e-12)
np.testing.assert_allclose(autograd_ratios(3.0), products(strong), rtol=1e-12)
carousel = autograd_ratios(1.0, identity=True)
vanishing = pd.DataFrame({
    "Steps back q": steps,
    "Cycle": [int(cycles[-1 - q]) for q in steps],
    "Reading": [readings[-1 - q] for q in steps],
    "Factor, w = 1.0": [weak[-1 - q] for q in steps],
    "Product, w = 1.0": [f"{v:.3e}" for v in products(weak)],
    "Factor, w = 3.0": [strong[-1 - q] for q in steps],
    "Product, w = 3.0": [f"{v:.3e}" for v in products(strong)],
    "Product, carousel": [f"{v:.3e}" for v in carousel],
})
```

</details>

"Cycle" is the cycle whose factor is the last one multiplied in, and
"Reading" is T50 at that cycle. Products are written in scientific notation:
`4.399e-02` means $4.399 \times 10^{-2}$.

<!-- sample: vanishing -->
| Steps back q | Cycle | Reading | Factor, w = 1.0 | Product, w = 1.0 | Factor, w = 3.0 | Product, w = 3.0 | Product, carousel |
|---|---|---|---|---|---|---|---|
| 1 | 191 | 2.074416 | 0.043991 | 4.399e-02 | 0.018874 | 1.887e-02 | 1.000e+00 |
| 2 | 190 | 2.097681 | 0.043044 | 1.894e-03 | 0.018439 | 3.480e-04 | 1.000e+00 |
| 3 | 189 | 2.122425 | 0.042422 | 8.033e-05 | 0.018063 | 6.286e-06 | 1.000e+00 |
| 5 | 187 | 2.155909 | 0.040878 | 1.653e-07 | 0.017421 | 2.398e-09 | 1.000e+00 |
| 10 | 182 | 1.830532 | 0.056480 | 5.361e-14 | 0.024365 | 1.167e-17 | 1.000e+00 |
| 20 | 172 | 0.737611 | 0.140747 | 1.683e-25 | 0.072434 | 1.189e-32 | 1.000e+00 |
| 29 | 163 | 1.442722 | 0.154596 | 1.289e-33 | 0.463787 | 1.460e-42 | 1.000e+00 |

The loop in `factors`, written out for its first two cycles with $w = 1.0$, and
the first two products:

```text
cycle 163   previous activation 0
            net    = 1.0 × 0 + 1.0 × 1.442722          = 1.442722
            f(net) = 0.808876
            factor = 0.808876 × (1 − 0.808876) × 1.0   = 0.154596
cycle 164   net    = 1.0 × 0.808876 + 1.0 × 1.201690   = 2.010566
            f(net) = 0.881902
            factor = 0.881902 × (1 − 0.881902) × 1.0   = 0.104151
   …
q = 1       product = factor at 191                    = 0.043991
q = 2       product = 0.043991 × 0.043044              = 0.001894
```

Near the end of engine 1's life the readings are high, between 2.074416 and
2.155909 in the rows for cycles 187 to 191, and a high reading puts the net
input on the flat upper part of the sigmoid, where the slope is small. With
$w = 1.0$ the factor at cycle 191 is therefore 0.043991, well under the 0.25
ceiling, and 29 steps back the product has fallen to $1.289 \times 10^{-33}$.
With $w = 3.0$ the factor is smaller still at every cycle after the first (the
code checks all 29), and the 29-step product is $1.460 \times 10^{-42}$: the
larger weight added three times the unit's own previous activation to the net
input and pushed it further up the flat part, which is the paper's second
conclusion seen on real readings. Cycle 163 is the exception, 0.463787 against
0.154596: the unit has just started from 0, so both nets equal the reading,
both slopes are the same, and the larger weight multiplies it by 3. The last
column belongs to the next section.

The key idea is that an error signal sent $q$ steps back is multiplied by one
factor per step, the activation function's slope times the weight crossed;
with a logistic unit every factor stays below 1 unless the weight exceeds 4,
and large weights flatten the slope, so the product shrinks exponentially and
the network cannot learn from events far in the past.

## The constant error carousel (Section 3.2)

Hochreiter and Schmidhuber (1997, [Section 3.2](../references/hochreiter_1997_long_short_term_memory.pdf#page=5))
ask what a single unit $j$ connected to itself would need for the error signal
to pass through it unchanged. From the backward formula, one step back
multiplies its error signal by $f_j'(\mathrm{net}_j(t))\,w_{jj}$, where
$w_{jj}$ is the weight of its connection to itself. Constant flow therefore
requires

$$f_j'\big(\mathrm{net}_j(t)\big)\, w_{jj} = 1.0$$

at every step. A slope that is the same constant at every net input belongs to
a straight line, so $f_j(x) = x / w_{jj}$: the activation function has to be
linear. The unit's activation then stays constant,
$y^j(t+1) = f_j(w_{jj}\, y^j(t)) = y^j(t)$. The paper takes the simplest such
choice, the identity function $f_j(x) = x$ with $w_{jj} = 1.0$, and calls the
result the **constant error carousel** (the paper spells it with a double r,
carrousel). It is the centre of the LSTM.

With invented numbers: a unit holding 0.7 computes $1.0 \times 0.7 = 0.7$ at
the next step, and passes it through the identity unchanged. Going backwards,
an error signal of 1.0 is multiplied by slope 1 and weight 1.0 and arrives one
step earlier as 1.0. The last column of the table above runs the same unit on
the 30 real readings, with automatic differentiation: every product is exactly
1.

The carousel cannot be used alone, because it is also connected to other
units, and Hochreiter and Schmidhuber (1997, Section 3.2) name two conflicts
that this causes:

- **Input weight conflict.** One incoming weight has to do two jobs at
  different times: let a relevant input switch the unit on, and keep later,
  irrelevant inputs from switching it off again. During training the same
  weight receives updates pulling it towards both jobs.
- **Output weight conflict.** One outgoing weight has to let another unit read
  the stored value when it is needed, and keep the stored value from disturbing
  that unit the rest of the time.

The paper notes that both conflicts exist for short time lags too, but that
they grow with the lag, because stored information must be protected for
longer and more already-correct outputs must be protected from disturbance
(Hochreiter & Schmidhuber, 1997, Section 3.2).

The key idea is that an error signal passes back through a unit unchanged only
if the unit's slope times its self-weight is exactly 1, which forces a linear
unit with weight 1.0 that holds its value; such a unit stores and remembers,
but on its own it cannot choose what to store or when to be read.

## The memory cell (Section 4)

### Two gates around the carousel

Hochreiter and Schmidhuber (1997, [Section 4](../references/hochreiter_1997_long_short_term_memory.pdf#page=6))
add two **gates** to the carousel. A gate is a unit of its own whose
activation, a number between 0 and 1, multiplies some other quantity: at 0 it
blocks it, at 1 it lets it through whole. The **input gate** multiplies what
enters the carousel and the **output gate** multiplies what leaves it. The
carousel, the two squashing curves $g$ and $h$ below, and the two
multiplications make up a **memory cell**. The gates are separate units that
feed it, drawn outside the cell's box in
[Figure 1](../references/hochreiter_1997_long_short_term_memory.pdf#page=7),
which is why several cells can share one pair of them (the cell blocks below).
The forward computation, given in Section 4 and repeated in
[Appendix A.1](../references/hochreiter_1997_long_short_term_memory.pdf#page=24)
as Eqs. 7 to 9, is:

$$\mathrm{net}_{\mathrm{in}_j}(t) = \sum_u w_{\mathrm{in}_j u}\, y^u(t-1), \qquad y^{\mathrm{in}_j}(t) = f\big(\mathrm{net}_{\mathrm{in}_j}(t)\big)$$

$$\mathrm{net}_{\mathrm{out}_j}(t) = \sum_u w_{\mathrm{out}_j u}\, y^u(t-1), \qquad y^{\mathrm{out}_j}(t) = f\big(\mathrm{net}_{\mathrm{out}_j}(t)\big)$$

$$\mathrm{net}_{c_j}(t) = \sum_u w_{c_j u}\, y^u(t-1)$$

$$s_{c_j}(t) = s_{c_j}(t-1) + y^{\mathrm{in}_j}(t)\; g\big(\mathrm{net}_{c_j}(t)\big), \qquad s_{c_j}(0) = 0$$

$$y^{c_j}(t) = y^{\mathrm{out}_j}(t)\; h\big(s_{c_j}(t)\big)$$

- $j$ numbers the memory cell; $\mathrm{in}_j$ and $\mathrm{out}_j$ are its two
  gates and $c_j$ is the cell itself.
- $u$ runs over every unit that feeds them: input units, gates, other cells,
  and ordinary units if there are any. Section 4 leaves the connections to
  whoever builds the network. One of them can be a unit whose activation is
  always 1.0; the weight from it is then a constant added to the net input,
  which the papers call a **bias** and this page calls an **offset** (Gers et
  al., 2000, [Section 3.3](../references/gers_2000_learning_to_forget.pdf#page=10)).
- The three net inputs are three separate weighted sums of the same previous
  activations, with three separate sets of weights.
- $f$ is the logistic sigmoid (Eq. 3), so both gates lie between 0 and 1.
- $s_{c_j}(t)$ is the **internal state**: the value the carousel holds.
- $g$ squashes the cell's net input before it enters; $h$ squashes the
  internal state on its way out.
- $y^{c_j}(t)$ is the **cell output**, which the rest of the network reads.

The fourth line is the carousel: the previous internal state is carried over
with weight 1.0 (the arrow marked 1.0 in Figure 1), and the gated, squashed
input is added to it. With invented numbers, a state of 0.5, an input gate of
0.1 and a squashed input of 1.5 give $0.5 + 0.1 \times 1.5 = 0.65$; with the
input gate at 0 the state stays 0.5 however large the input. The last line lets
the stored value out only as far as the output gate allows.

**Do not confuse the input gate with the cell input.** Both are computed from
the same previous activations. The squashed cell input
$g(\mathrm{net}_{c_j})$ is the proposed content, and it can be negative. The
input gate $y^{\mathrm{in}_j}$, between 0 and 1, is how much of that proposal
is added. PyTorch's documentation calls its version of the cell input the cell
gate (PyTorch, n.d.), but it is the quantity being multiplied, not a
multiplier.

Hochreiter and Schmidhuber (1997, Appendix A.1) give the two squashing curves
used in the experiments:

$$h(x) = \frac{2}{1 + e^{-x}} - 1 \quad (\text{Eq. 4}), \qquad g(x) = \frac{4}{1 + e^{-x}} - 2 \quad (\text{Eq. 5})$$

Both are the logistic curve stretched and shifted: $h$ runs from −1 to 1 and
$g$ from −2 to 2. Written with the **hyperbolic tangent**, $\tanh$, an S-shaped
curve from −1 to 1, they are $h(x) = \tanh(x/2)$ and $g(x) = 2\tanh(x/2)$; the
code below checks that on a grid of 25 points.

The paper is not consistent about these two ranges. The caption of its
Table 10 describes the setting it calls h1 as a sigmoid in [−2, 2] and the
setting g2 as one in [−1, 1]: the two ranges the other way round (Hochreiter &
Schmidhuber, 1997, [Table 10](../references/hochreiter_1997_long_short_term_memory.pdf#page=21)).
Section 5.1, the description of Task 2c in Section 5.2, and Eqs. 4 and 5 all
agree with each other, so this page follows them.

<details class="process-example">
<summary>Example in Python</summary>

```python
def g_1997(z):
    """Eq. 5: squashes the cell's net input into the range -2 to 2."""
    return 4.0 * sigmoid(z) - 2.0

def h_1997(z):
    """Eq. 4: squashes the internal state into the range -1 to 1 on its way out."""
    return 2.0 * sigmoid(z) - 1.0

grid = np.linspace(-6.0, 6.0, 25)
np.testing.assert_allclose(g_1997(grid), 2.0 * np.tanh(grid / 2.0), atol=1e-14)
np.testing.assert_allclose(h_1997(grid), np.tanh(grid / 2.0), atol=1e-14)
```

</details>

### One step of the 1997 cell on real readings

The cell below reads engine 1's T50 at its last two cycles, 191 and 192,
starting with zero internal state and zero output. Its multipliers are set by
hand, so it is again an **untrained demonstration** that predicts nothing. Each
gate's net input is the reading times its reading multiplier, plus the cell's
own previous output times its previous-output multiplier, plus its offset. The
forget-gate row of the table is used only from the forget-gate section on,
because the 1997 cell has no forget gate.

<details class="process-example">
<summary>Example in Python</summary>

```python
reading_w = np.array([1.0, 0.5, 1.0, 0.5])   # input gate, forget gate, cell input, output gate
hidden_w = np.array([0.5, 0.5, -1.0, 1.0])
offset = np.array([-1.0, 2.0, 0.0, -0.5])
multipliers = pd.DataFrame({"Part": ["Input gate", "Forget gate", "Cell input", "Output gate"],
                            "Reading multiplier": reading_w,
                            "Previous-output multiplier": hidden_w, "Offset": offset})
two = t50_1[-2:]   # NumPy array of shape (2,): T50 at cycles 191 and 192

def step_1997(x, y, s):
    """One step of the 1997 cell (Section 4, Eqs. 4 and 5): no forget gate."""
    net = reading_w * x + hidden_w * y + offset
    i, g, o = sigmoid(net[0]), g_1997(net[2]), sigmoid(net[3])
    s = 1.0 * s + i * g
    return [x, net[0], i, net[2], g, s, h_1997(s), o, o * h_1997(s)]

old = []
y = s = 0.0
for x in two:
    old.append(step_1997(x, y, s))
    y, s = old[-1][8], old[-1][5]
trace_1997 = pd.DataFrame({
    "Quantity": ["Reading", "Input gate net input", "Input gate", "Cell input net input",
                 "Squashed cell input g", "Internal state s", "Squashed internal state h(s)",
                 "Output gate", "Cell output"],
    "Cycle 191": old[0], "Cycle 192": old[1],
})
```

</details>

<!-- sample: multipliers -->
| Part | Reading multiplier | Previous-output multiplier | Offset |
|---|---|---|---|
| Input gate | 1.000000 | 0.500000 | -1.000000 |
| Forget gate | 0.500000 | 0.500000 | 2.000000 |
| Cell input | 1.000000 | -1.000000 | 0.000000 |
| Output gate | 0.500000 | 1.000000 | -0.500000 |

<!-- sample: trace_1997 -->
| Quantity | Cycle 191 | Cycle 192 |
|---|---|---|
| Reading | 2.074416 | 2.149321 |
| Input gate net input | 1.074416 | 1.314045 |
| Input gate | 0.745436 | 0.788189 |
| Cell input net input | 2.074416 | 1.819874 |
| Squashed cell input g | 1.553566 | 1.442204 |
| Internal state s | 1.158084 | 2.294814 |
| Squashed internal state h(s) | 0.521969 | 0.816893 |
| Output gate | 0.631163 | 0.711793 |
| Cell output | 0.329447 | 0.581459 |

At cycle 191 the previous output is 0, so the input gate's net input is
$1.0 \times 2.074416 + 0.5 \times 0 - 1.0 = 1.074416$ and the gate is
$f(1.074416) = 0.745436$. The cell input's net input is 2.074416, which $g$
squashes to 1.553566, and the state becomes
$0 + 0.745436 \times 1.553566 = 1.158084$. $h$ squashes that to 0.521969, and the output gate,
$f(0.5 \times 2.074416 - 0.5) = 0.631163$, lets
$0.631163 \times 0.521969 = 0.329447$ out. At cycle 192 every net input also
contains that output: the input gate's is
$1.0 \times 2.149321 + 0.5 \times 0.329447 - 1.0 = 1.314045$. The state keeps
all of cycle 191's content and adds $0.788189 \times 1.442204 = 1.136730$,
reaching $1.158084 + 1.136730 = 2.294814$.

### How the 1997 cell learns (Section 4 and Appendix A.1)

The paragraph on learning in Hochreiter and Schmidhuber (1997, Section 4)
describes a **truncated** gradient. An error
signal that reaches one of a cell's three net inputs (the cell's own, the input
gate's and the output gate's) is used to change that net input's incoming
weights, and is then not passed further back in time.
[Appendix A.1](../references/hochreiter_1997_long_short_term_memory.pdf#page=25)
does this by setting the derivative of each of those net inputs with respect to
every previous activation to zero. The one route left backwards in time runs
through the internal state, from $s_{c_j}(t)$ to $s_{c_j}(t-1)$, and the state
update multiplies along it by exactly 1 (Appendix A.2, Eq. 35). Section 4 calls the method a variant of RTRL:
for the weights into a cell and into its input gate, instead of sending errors
backwards, it carries forwards how much the internal state depends on each
weight. For a weight $w_{c_j m}$ from a unit $m$ into the cell, Appendix A.1
gives
([Eqs. 23, 27 and 28](../references/hochreiter_1997_long_short_term_memory.pdf#page=27)):

$$e_{s_{c_j}}(t) = y^{\mathrm{out}_j}(t)\; h'\big(s_{c_j}(t)\big) \sum_k w_{k c_j}\, e_k(t) \quad (\text{Eq. 23})$$

$$\frac{\partial s_{c_j}(t)}{\partial w_{c_j m}} = \frac{\partial s_{c_j}(t-1)}{\partial w_{c_j m}} + g'\big(\mathrm{net}_{c_j}(t)\big)\; y^{\mathrm{in}_j}(t)\; y^m(t-1) \quad (\text{Eq. 27})$$

$$\Delta w_{c_j m}(t) = \alpha\; e_{s_{c_j}}(t)\; \frac{\partial s_{c_j}(t)}{\partial w_{c_j m}} \quad (\text{Eq. 28})$$

- $e_k(t)$ is output unit $k$'s error signal (Eq. 19, the $\vartheta_k$ above),
  and the sum over $k$, each weighted by the connection $w_{k c_j}$ from the
  cell to unit $k$, is the error signal arriving at the cell output.
- $e_{s_{c_j}}(t)$, the **internal state error**, is minus the derivative of
  the error with respect to the internal state: the arriving error signal
  scaled by the output gate and by $h'$, the slope of $h$.
- $\partial s_{c_j}(t)/\partial w_{c_j m}$ is how much the internal state
  depends on the weight, carried forwards from step to step and started at 0.
  Each step adds the slope of $g$, the input gate, and the activation
  $y^m(t-1)$ that the weight multiplies. In the demonstration that activation
  is the reading the step reads, as in the experiments, where Section 5 sets the
  input units to the current input before it computes the cells.
- $\Delta w_{c_j m}(t)$ is time $t$'s contribution to the weight's update; the
  contributions of all steps are added up.

The input gate's weights follow the same pattern (Hochreiter & Schmidhuber,
1997, Appendix A.1, Eqs. 25 and 26), with
$g(\mathrm{net}_{c_j})\,f'_{\mathrm{in}_j}$ in place of
$g'\,y^{\mathrm{in}_j}$, and summed over the cells that share the gate. The
output gate's weights are updated by ordinary back-propagation, cut off at the
gate (Eq. 21).

The code below runs Eqs. 23, 27 and 28 on the two cycles of the trace, for the
cell input's reading multiplier. The demonstration has no output units, so the
arriving error signal is taken to be 1 at each cycle, and the last column is
then the weight's change per unit of arriving error signal and per unit of
learning rate. The code checks that number at cycle 192 against PyTorch's
automatic differentiation of the cell output, with the previous output treated
as a fixed number wherever it enters cycle 192's net inputs: that is the 1997
cut. PyTorch's `detach` does it: the value is used, but no derivative passes
back through it.

<details class="process-example">
<summary>Example in Python</summary>

```python
def g_slope(z):
    """Slope of Eq. 5's g: the derivative of 2 tanh(z / 2)."""
    return 1.0 - np.tanh(z / 2.0) ** 2

def h_slope(z):
    """Slope of Eq. 4's h: the derivative of tanh(z / 2)."""
    return 0.5 * (1.0 - np.tanh(z / 2.0) ** 2)

rows, carried = [], 0.0
for x, net_in, i, net_c, g, s, hs, o, out in old:
    carried = carried + g_slope(net_c) * i * x          # Eq. 27, started at 0
    entering = o * h_slope(s) * 1.0                      # Eq. 23, arriving error signal 1
    rows.append([x, i, g_slope(net_c), carried, entering, entering * carried])   # Eq. 28 / alpha
rule_1997 = pd.DataFrame(rows, columns=["Reading", "Input gate", "Slope of g", "ds/dw (Eq. 27)",
                                        "Output gate x slope of h (Eq. 23)", "Change of w per unit (Eq. 28)"])
rule_1997.insert(0, "Cycle", [191, 192])

def cell_output_1997(w_cell):
    """The two 1997 steps in PyTorch, with the previous output detached as the 1997 rule cuts it."""
    w = torch.tensor(reading_w)
    w[2] = w_cell
    u, b = torch.tensor(hidden_w), torch.tensor(offset)
    y = s = torch.zeros((), dtype=torch.float64)
    for x in torch.tensor(two):
        z = w * x + u * y.detach() + b
        s = s + torch.sigmoid(z[0]) * (4 * torch.sigmoid(z[2]) - 2)
        y = torch.sigmoid(z[3]) * (2 * torch.sigmoid(s) - 1)
    return y

w_cell = torch.tensor(reading_w[2], dtype=torch.float64, requires_grad=True)
cell_output_1997(w_cell).backward()
assert abs(w_cell.grad.item() - rule_1997.iloc[1, -1]) < 1e-12
```

</details>

<!-- sample: rule_1997 -->
| Cycle | Reading | Input gate | Slope of g | ds/dw (Eq. 27) | Output gate x slope of h (Eq. 23) | Change of w per unit (Eq. 28) |
|---|---|---|---|---|---|---|
| 191 | 2.074416 | 0.745436 | 0.396608 | 0.613292 | 0.229601 | 0.140812 |
| 192 | 2.149321 | 0.788189 | 0.480012 | 1.426467 | 0.118402 | 0.168896 |

At cycle 191, Eq. 27 starts from 0 and adds
$0.396608 \times 0.745436 \times 2.074416 = 0.613292$; cycle 192 adds
$0.480012 \times 0.788189 \times 2.149321 = 0.813175$, reaching 1.426467. One
unit of error signal arriving at the cell output at cycle 192 enters the
carousel as $0.711793 \times 0.166343 = 0.118402$, and Eq. 28 multiplies that by
1.426467 to give 0.168896, the number automatic differentiation also gives.
With one unit of error signal arriving at each of the two cycles, the weight
changes over both by $\alpha$ times $0.140812 + 0.168896 = 0.309708$.

Read backwards, the same product is the route Section 4 describes. Split
cycle 192's 1.426467 into its two shares:
$0.118402 \times 1.426467 = 0.118402 \times 0.813175 + 0.118402 \times 1 \times 0.613292$.
In the first term the error that entered at cycle 192 leaves at once, through
that cycle's input gate and $g$: $0.813175$ is
$0.788189 \times 0.480012$ times the cycle-192 reading. In the second it is
carried back one step through the carousel, multiplied by exactly 1, and leaves
at cycle 191, where $0.613292$ is $0.745436 \times 0.396608$ times that cycle's
reading. Carrying $\partial s / \partial w$ forwards is therefore the same
arithmetic as carrying the error backwards through the carousel.

Hochreiter and Schmidhuber (1997, Appendix A.1) give the reason for the cut:
with these routes removed, no error
that has left a cell can come back into it through a loop, so the flow through
the carousel stays constant. The cut also makes the method cheap. Its cost per
time step is proportional to the number of weights, and Section 4 calls it
**local in space and time**: in its footnote 3, the cost per step and weight
does not grow with the size of the network, and the memory needed does not
grow with the length of the sequence. The cut is a choice, not a necessity.
[Section 6](../references/hochreiter_1997_long_short_term_memory.pdf#page=22)
says the full gradient could be computed but does not recommend it: it costs
more, constant flow through the carousels can be shown only for the truncated
version, and a few experiments without truncation showed no significant
difference, which the section attributes to error outside the carousels
vanishing quickly anyway. The two derivatives are compared on real readings in
[What PyTorch differentiates](#what-pytorch-differentiates).

### Why the gates resolve the two conflicts

Hochreiter and Schmidhuber (1997, Section 4) match each gate to one conflict.
The multiplications above show how.

- **The input weight conflict.** The cell input's reading multiplier changes
  by $\alpha\,e_s\,\partial s/\partial w$, and $\partial s/\partial w$ gets one
  share per step, each multiplied by that step's input gate. Cycle 192's share
  is 0.813175 because the input gate was 0.788189 there; with the gate at 0 the
  share would be 0, $\partial s/\partial w$ would stay at 0.613292, and the
  reading of cycle 192 would neither change the state nor pull on the weight.
  The weight is shaped only by readings that arrive while the gate is open.
  Keeping later readings out is left to the gate, whose own weights are
  trained by Eqs. 25 and 26.
- **The output weight conflict.** The weight $w_{k c_j}$ from the cell to an
  output unit $k$ multiplies the cell output $y^{\mathrm{out}_j}\,h(s)$, which
  at cycle 191 is $0.631163 \times 0.521969 = 0.329447$. With the output gate
  at 0, unit $k$ receives 0 from the cell whatever that weight is, and the
  weight's own update, $\alpha\, e_k\, y^{c_j}$ (Eq. 22), is 0 as well. The
  same gate scales the error entering the cell, by Eq. 23: at 0, an error
  arising while the cell is not being read leaves the weights that decide what
  the cell stores unchanged.

Hochreiter and Schmidhuber (1997, Section 4) sum this up as the output gate
selecting which error signals are kept in the carousel, and the input gate deciding when they leave it, both by
the multiplications just shown. They add that a task does not always need both
gates: Experiments 2a and 2b use input gates only.

### Memory cell blocks

Several cells can share one input gate and one output gate. Hochreiter and
Schmidhuber (1997, Section 4) call $S$ such cells a **memory cell block of
size $S$**, say a block stores a **distributed** input, one whose information
is spread over several numbers, more easily than a single cell does, and note
that it costs the same two gates as one cell. A block of size 1 is a single
cell. Gers et al. (2000, [Section 2](../references/gers_2000_learning_to_forget.pdf#page=3)) call the
block the basic unit of an LSTM's **hidden layer**, the units between the
network's inputs and its outputs.

### Two problems the paper anticipates

Hochreiter and Schmidhuber (1997, Section 4) name two problems and remedies
for each.

- **Abuse.** Early in training, the error can sometimes be reduced without
  storing anything. The network may then hold a cell's output constant and use
  it as a fixed extra input to other units, and such a cell can take a long
  time to be freed for real storage. Remedies: add cells one at a time when the
  error stops falling, or start the output gates with negative offsets, so that
  the cells start near zero and are drawn into use one after another.
- **Internal state drift.** If the inputs a cell adds are mostly positive, or
  mostly negative, its internal state drifts away over time. The slope $h'$ of
  the output curve then becomes very small, and the error signal entering the
  cell through it vanishes. The remedy at the start of learning is to start the
  input gates with offsets that hold them near zero.

The 2000 paper describes a closely related failure, unbounded growth of the
state, on streams that are never reset (Gers et al., 2000, Section 2.1).

The key idea is that the carousel passes an error signal back unchanged by
multiplying it by exactly 1 per step, while two separate gate units, whose
activations lie between 0 and 1, multiply what enters and what leaves it; the
learning rule cuts every other route back in time, so each weight into the cell
is updated only for steps where the input gate let its contribution in, and
each weight out of it only for steps where the output gate let the cell be
read.

## What the 1997 paper claims, and on what evidence (Sections 5 to 7)

Hochreiter and Schmidhuber (1997, Abstract and Section 1) claim that LSTM
learns to bridge minimal time lags
of more than 1000 steps, also on noisy input that cannot be compressed, without
losing the ability to handle short lags, at a cost per time step and weight
that does not grow with the network. A **minimal time lag** is the smallest
number of steps between the last input that matters and the target that
depends on it (Table 10, column 3).

The evidence is six experiments on artificial sequences (Hochreiter &
Schmidhuber, 1997, [Section 5](../references/hochreiter_1997_long_short_term_memory.pdf#page=10)).
Five of them are built so that every training sequence has a long minimal lag,
with no short-lag examples to make learning easier. The first, the embedded
Reber grammar, allows lags as short as 9 steps; the paper includes it as a
popular benchmark on which the older methods do not fail completely, and to
show what output gates add (Section 5.1). Conditions shared by the experiments:
on-line learning, meaning the weights are updated example by example rather
than after a whole batch; logistic sigmoids as activation functions; starting
weights drawn from [−0.2, 0.2] or [−0.1, 0.1]; and the network's activations
reset after each input sequence. Gers et al.
(2000, [Section 2.1](../references/gers_2000_learning_to_forget.pdf#page=5)) describe
that last condition as resetting the cell states to zero before each new
sequence, and it is what the next section turns on.

Two results carry the claim. On Task 2a, where the first of 101 symbols must be
remembered until the last, across 100 steps, Hochreiter and Schmidhuber (1997,
[Table 2](../references/hochreiter_1997_long_short_term_memory.pdf#page=14))
report, as means of 18 trials, that RTRL and BPTT had no successful trial
within 5,000,000 training sequences, and that Schmidhuber's earlier two-network
sequence chunker, which bridges long lags only where parts of the sequence are
predictable (Section 2), succeeded in 33 percent of trials. LSTM succeeded in
100 percent, after 5,040 sequences. On Task 2c, the second symbol of a sequence
must be remembered across at least $q + 1$ steps filled with randomly chosen
**distractor symbols**, symbols that carry no information about the answer.
[Table 3](../references/hochreiter_1997_long_short_term_memory.pdf#page=15)
reports that with $q = 1{,}000$, so a minimal lag of 1,001 steps, and
$p = 1{,}000$ different distractor symbols to choose from, LSTM succeeded after
49,000 training sequences, the mean of 20 trials.

Hochreiter and Schmidhuber (1997, Section 6) list limitations. The truncated
version struggles with tasks that
cannot be learned one part at a time, such as the exclusive-or (true when
exactly one of two inputs is on) of two widely separated inputs, where storing
one input alone does not reduce the error. Like every gradient-based method, it
cannot count steps exactly, telling 99 from 100. Each block adds two gate
units. And all of the evidence is artificial: the conclusion
([Section 7](../references/hochreiter_1997_long_short_term_memory.pdf#page=23))
lists applying LSTM to real data, time series prediction among it, as future
work. Nothing in the 1997 paper is evidence about engine sensor data.

## Learning to forget (Gers, Schmidhuber and Cummins, 2000)

### The weakness on a continual stream (Section 2.1)

The 1997 experiments reset every cell before each new sequence (Hochreiter &
Schmidhuber, 1997, Section 5). Gers et al. (2000,
[Abstract and Section 1](../references/gers_2000_learning_to_forget.pdf#page=2))
consider a **continual** input stream: one that is not divided beforehand into
sequences with marked ends at which the network could be reset.

Gers et al. (2000, [Section 2.1](../references/gers_2000_learning_to_forget.pdf#page=5))
give the mechanism. Internal states often grow linearly while a series is
presented, and on a continual stream they can grow without bound. That
saturates $h$: the state sits on the flat part of the curve, where the curve's output is pinned
near −1 or 1. Two things follow:

- the slope $h'$ vanishes, so error signals can no longer enter the cell;
- the cell output becomes the output gate's activation times ±1, so the cell
  behaves as an ordinary unit and stops working as a memory.

The section also explains why three simpler remedies are not enough.

- **Weight decay**, which shrinks every weight a little towards zero at each
  update, made unbounded growth more likely, not less.
- **A fixed self-connection weight below 1**, which makes the state decay, has
  no principled value: a rate that helps on some tasks costs the ability to
  handle delays of unknown length, and it did not significantly improve their
  results (Gers et al., 2000,
  [Table 2](../references/gers_2000_learning_to_forget.pdf#page=13), the state
  decay row).
- **Resetting the states** at the start of each sequence needs someone who
  knows where the sequences start, which is exactly the knowledge a continual
  stream lacks.

### The forget gate (Section 3.1, Eqs. 6 and 7)

Gers et al. (2000, [Section 3](../references/gers_2000_learning_to_forget.pdf#page=6))
replace the carousel's constant weight 1.0 with the activation of a third
gate, the **forget gate**:

$$\mathrm{net}_{\varphi_j}(t) = \sum_m w_{\varphi_j m}\, y^m(t-1), \qquad y^{\varphi_j}(t) = f_{\varphi_j}\big(\mathrm{net}_{\varphi_j}(t)\big) \quad (\text{Eq. 6})$$

$$s_{c_j^v}(t) = y^{\varphi_j}(t)\; s_{c_j^v}(t-1) + y^{\mathrm{in}_j}(t)\; g\big(\mathrm{net}_{c_j^v}(t)\big), \qquad s_{c_j^v}(0) = 0 \quad (\text{Eq. 7})$$

- $j$ numbers the memory block and $v$ the cell within it, so $c_j^v$ is cell
  $v$ of block $j$; the gate belongs to the block.
- $\varphi_j$ is block $j$'s forget gate. Its net input is a weighted sum of
  the previous activations $y^m(t-1)$, exactly like the other two gates' (the
  paper's Eqs. 1 and 2), and $f_{\varphi_j}$ is the logistic sigmoid, so the
  gate lies between 0 and 1.
- Everything else is the 1997 cell: Eq. 7 replaces only the state update, and
  the rest of the forward computation is unchanged.

In words: before the new input is added, the old state is multiplied by the
forget gate. At 1 the cell is the 1997 cell; at 0 the old state is wiped in
one step; in between it fades. With invented numbers, an old state of 10.0, a
forget gate of 0.9 and a gated input of 0.5 give $0.9 \times 10.0 + 0.5 = 9.5$;
with the forget gate at 0 the state becomes 0.5. Section 3 counts the slow
fading as a reset too, a gradual one.

Gers et al. (2000, Section 3.1) also say how the gates start: negative offsets
for the input and output gates, as in 1997, and positive ones for the forget
gates, so that the forget gate starts near 1 and the cell begins as a 1997 cell.
[Section 4.2](../references/gers_2000_learning_to_forget.pdf#page=12) gives the
values used: +0.5 for the first block's forget gate, +1 for the second, and so
on, against −0.5, −1.0, −1.5 for the input and output gates of successive
blocks, and adds that the precise values are not critical.

### What the forget gate does to the learning rule (Section 3.2)

The learning rule keeps the 1997 truncation: errors are cut off once they leave
a cell or a gate (Gers et al., 2000,
[Section 3.2](../references/gers_2000_learning_to_forget.pdf#page=7)). The
section says which method each weight uses: output units ordinary
back-propagation through time; output gates a truncated version of it (Eq. 11);
and the weights into the cells, the input gates and the forget gates a
truncated RTRL. Its Eq. 13, the internal state error, and its Eq. 19, the
update of a weight into a cell, are the 1997 Eqs. 23 and 28 above, written with
$\delta_k$ for output unit $k$'s error signal (Eq. 10). What changes is how the
dependence of the state on a weight is carried forwards. For
a weight $w_{c_j^v m}$ into the cell,
[Eq. 15](../references/gers_2000_learning_to_forget.pdf#page=9) reads

$$\frac{\partial s_{c_j^v}(t)}{\partial w_{c_j^v m}} = \frac{\partial s_{c_j^v}(t-1)}{\partial w_{c_j^v m}}\; y^{\varphi_j}(t) + g'\big(\mathrm{net}_{c_j^v}(t)\big)\; y^{\mathrm{in}_j}(t)\; y^m(t-1)$$

- The left side is how much the internal state now changes per unit change of
  the weight from unit $m$ into the cell.
- The first term is the same quantity one step earlier, multiplied by the
  forget gate, because Eq. 7 multiplies the old state by it. With the gate at
  1 this is the 1997 Eq. 27.
- The second term is the weight's direct effect at this step: the slope of $g$,
  the input gate, and the activation $y^m(t-1)$ that the weight multiplies.
- Eqs. 16 and 17 do the same for the input gate's and the forget gate's
  weights, and Eq. 18 starts all three at zero. Eq. 19 updates a cell's weights
  from its own state; Eq. 20 updates a gate's weights by adding up the
  contributions of every cell in its block.

This record of past influence is what each update uses, and it is multiplied by
the forget gate at every step. When the gate drops to 0, the record is wiped
together with the state; in the paper's words, "forgetting includes forgiving
previous mistakes" (Gers et al., 2000, Section 3.2,
[p. 9](../references/gers_2000_learning_to_forget.pdf#page=9)). The same
multiplication applies to the error
passing back through the carousel: by Eq. 7, a change in the old state moves
the new one by the forget gate's activation, so the 1997 carousel's flow is
exactly constant only while the gate is at 1. Eq. 15 is run on two real cycles
in [What PyTorch differentiates](#what-pytorch-differentiates).

### The evidence (Section 4)

The main test is the **continual embedded Reber grammar** (Gers et al., 2000,
[Section 4.1](../references/gers_2000_learning_to_forget.pdf#page=10)). A
grammar here is a set of rules for which symbol may follow which, drawn as a
graph in Figure 2. Strings generated by it are joined into one stream with no
resets, and the network must predict the next symbol at every step. Getting the
symbol before last right requires remembering the string's second symbol across
a lag of at least 7 steps and with no upper limit. A test stream ends at the
first wrong prediction, or after 100,000 symbols.

Gers et al. (2000, [Table 2](../references/gers_2000_learning_to_forget.pdf#page=13))
report, as percentages of 100 independently trained networks each, how many
found a perfect solution, which the table's caption defines as predicting 10
test streams of 100,000 symbols each without a mistake: 74 for the 1997 LSTM
given resets at the end of every string, 0 for the 1997 LSTM on the continual stream, 0 for a
version whose state decays by a fixed factor of 0.9, 18 with forget gates, and
62 with forget gates and a learning rate that decreases during each stream.
[Figure 4](../references/gers_2000_learning_to_forget.pdf#page=14) shows the
mechanism of the failure. In one test stream, the internal states of the 1997
LSTM grow from string to string, linearly as Section 4.4 describes it, until,
after 13 strings, the cell outputs saturate and prediction breaks down.
[Figure 5](../references/gers_2000_learning_to_forget.pdf#page=15) shows the
remedy at work after training. One block's forget gate sits near 1 for most of
the stream and drops to near 0 at the end of each string, resetting the states,
which swing up and down around zero but no longer drift away as in Figure 4
(Section 4.4).

A second, continual version of the 1997 temporal order task (Gers et al.,
2000, [Section 4.5](../references/gers_2000_learning_to_forget.pdf#page=16))
gives the same picture. The 1997 LSTM found no perfect solution, against 24
percent with forget gates and 37 percent with forget gates and a decreasing learning
rate ([Table 3](../references/gers_2000_learning_to_forget.pdf#page=17)). On
the original task, where a reset follows each sequence anyway, forget gates
whose starting offset was 5.0 learned as fast as the 1997 cell, and with an
offset of 1.0 about three times slower: there the forget gate first had to
learn not to forget. Like the 1997 evidence, all of it comes from artificial
symbol streams.

### Worked example: 80 engines read as one stream

Joining the T50 readings of the 80 training engines (assumption A-06) end to
end, in order of engine number, gives one stream of 16,779 cycles with no
marker where one engine ends and the next begins: the continual setting of
Section 2.1. One cell reads it.
Its input gate is $f(1.0\,x - 1.0)$ and its squashed input is $g(1.0\,x)$, the
reading multipliers and offsets of the
[two-cycle example above](#one-step-of-the-1997-cell-on-real-readings), with the
cell's own previous output left out, so that each cycle's addition depends on
that cycle's reading alone. Without a forget gate the internal state is then a
running total of the additions. With one, the gate is $f(0.5\,x + 2.0)$, the
forget-gate row of the same table: a positive offset, as Section 3.1 recommends
for the start of training. This is an untrained demonstration of the
arithmetic, not of what a trained cell would do on FD001.

<details class="process-example">
<summary>Example in Python</summary>

```python
training_engines = np.unique(prepared.train.units)
assert len(training_engines) == 80
pieces = [scaled.loc[scaled["unit"] == unit, t50].to_numpy(dtype=np.float64) for unit in training_engines]
stream = np.concatenate(pieces)   # NumPy array of shape (16779,)

def additions(x):
    """What one cycle adds to the internal state: input gate times squashed cell input."""
    return sigmoid(reading_w[0] * x + offset[0]) * g_1997(reading_w[2] * x + offset[2])

added = additions(stream)
keep = sigmoid(reading_w[1] * stream + offset[1])
no_forget = np.cumsum(added)                        # the 1997 cell: a running total
with_forget = np.empty_like(added)                  # Eq. 7: multiply by the gate, then add
state = 0.0
for t in range(len(added)):
    state = keep[t] * state + added[t]
    with_forget[t] = state
np.testing.assert_allclose([added[0], added[1], keep[1], no_forget[1], with_forget[1]],
                           [-0.110186, -0.108832, 0.829098, -0.219017, -0.200187], atol=5e-7)
last_of = np.cumsum([len(piece) for piece in pieces]) - 1

def log10_slope_h(s):
    """log10 of h's slope at s, 2 sigma(s) sigma(-s), computed so that it cannot round to zero."""
    return (np.log(2.0) - np.logaddexp(0.0, -s) - np.logaddexp(0.0, s)) / np.log(10.0)

counts = [1, 2, 5, 10, 20, 40, 80]
ends = [last_of[k - 1] for k in counts]
stream_table = pd.DataFrame({
    "Engines read": counts,
    "Cycles read": [int(e + 1) for e in ends],
    "Internal state, no forget gate": [no_forget[e] for e in ends],
    "Slope of h, power of ten": [log10_slope_h(no_forget[e]) for e in ends],
    "Internal state, forget gate": [with_forget[e] for e in ends],
})
largest_add = np.abs(added).max()
stream_facts = pd.DataFrame({
    "Quantity": ["Cycles in the stream", "Average reading", "Average addition",
                 "Share of additions above zero", "Largest addition in size",
                 "Smallest forget gate", "Largest forget gate", "Bound: largest addition / (1 - largest forget gate)",
                 "Largest internal state reached with the forget gate"],
    "Value": pd.Series([len(stream), stream.mean(), added.mean(), (added > 0).mean(), largest_add,
                        keep.min(), keep.max(), largest_add / (1.0 - keep.max()), np.abs(with_forget).max()], dtype=object),
})
assert np.abs(with_forget).max() <= largest_add / (1.0 - keep.max())

# Restarting at zero: at each engine, as the 1997 experiments restart at each
# sequence, and at each 30-cycle window, as this project's stage does.
per_engine = [np.cumsum(additions(piece)) for piece in pieces]
windows_30 = np.concatenate([np.lib.stride_tricks.sliding_window_view(additions(piece), 30) for piece in pieces])
assert len(windows_30) == len(prepared.train.features)
within_window = np.cumsum(windows_30, axis=1)
largest = [np.abs(no_forget).max(), np.abs(with_forget).max(),
           max(np.abs(run).max() for run in per_engine), np.abs(within_window).max()]
arrangements = pd.DataFrame({
    "How the readings are fed": ["One stream, no forget gate", "One stream, forget gate",
                                 "Restart at each engine, no forget gate",
                                 "Restart at each 30-cycle window, no forget gate"],
    "Largest internal state": largest,
    "Smallest slope of h, power of ten": [log10_slope_h(v) for v in largest],
})
```

</details>

The loop over the stream, written out for engine 1's first two cycles:

```text
cycle 1   reading x = −0.995210
          input gate  f(1.0 × x − 1.0) = 0.119707
          cell input  g(1.0 × x)       = −0.920463
          addition    0.119707 × −0.920463               = −0.110186
          forget gate f(0.5 × x + 2.0)                   = 0.817931
          no forget gate:  0 + (−0.110186)                = −0.110186
          forget gate:     0.817931 × 0 + (−0.110186)     = −0.110186
cycle 2   reading −0.841500, addition −0.108832, forget gate 0.829098
          no forget gate:  −0.110186 + (−0.108832)        = −0.219017
          forget gate:     0.829098 × −0.110186 + (−0.108832) = −0.200187
```

"Slope of h, power of ten" is $\log_{10} h'(s)$: a value of −7.617237 means the
slope is $10^{-7.617237}$, and every error signal entering the cell is
multiplied by that slope.

<!-- sample: stream_table -->
| Engines read | Cycles read | Internal state, no forget gate | Slope of h, power of ten | Internal state, forget gate |
|---|---|---|---|---|
| 1 | 192 | 18.232484 | -7.617237 | 15.401310 |
| 2 | 479 | 36.929246 | -15.737138 | 18.828952 |
| 5 | 1115 | 133.917575 | -57.858634 | 18.038795 |
| 10 | 2098 | 244.300007 | -105.797115 | 19.489318 |
| 20 | 4072 | 564.775344 | -244.977785 | 14.385485 |
| 40 | 7907 | 1120.212683 | -486.201157 | 22.025286 |
| 80 | 16779 | 2537.499825 | -1101.721142 | 16.172269 |

<!-- sample: stream_facts -->
| Quantity | Value |
|---|---|
| Cycles in the stream | 16779 |
| Average reading | 0.000000 |
| Average addition | 0.151231 |
| Share of additions above zero | 0.449312 |
| Largest addition in size | 1.634459 |
| Smallest forget gate | 0.654997 |
| Largest forget gate | 0.972340 |
| Bound: largest addition / (1 - largest forget gate) | 59.091588 |
| Largest internal state reached with the forget gate | 23.559395 |

The readings average zero over the stream, because the rescaling makes them
do so, but the additions do not. The input gate $f(x - 1)$ opens wider for high
readings than for low ones, so a positive reading adds more than a negative
reading of the same size takes away. Fewer than half of the additions are
positive, yet their average is 0.151231, and without a forget gate the state
climbs with every engine read, to 2537.499825 after all 80. By then the slope
of $h$ is $10^{-1101.721142}$, and no error signal can enter the cell: Section
2.1's breakdown, reached here through additions that are mostly of one sign in
size, the condition the 1997 paper's Section 4 describes as drift.

With the forget gate, the size of the state can never exceed the largest
addition divided by one minus the largest forget gate, because each step
multiplies the old state by at most that gate and adds at most that much. The
bound here is 59.091588, and the largest state reached is 23.559395. Bounded is
not the same as useful, though. With its positive offset, as Section 3.1
intends, this gate stays between 0.654997 and 0.972340 and never comes near 0,
so it forgets slowly: at that largest state the slope of $h$ is still
$10^{-9.930685}$, as the table below shows. What the 2000 paper shows is a gate
that *learns* to drop to 0 at the right moments (Figure 5), and an untrained
gate cannot show that.

<!-- sample: arrangements -->
| How the readings are fed | Largest internal state | Smallest slope of h, power of ten |
|---|---|---|
| One stream, no forget gate | 2537.499825 | -1101.721142 |
| One stream, forget gate | 23.559395 | -9.930685 |
| Restart at each engine, no forget gate | 85.707215 | -36.921140 |
| Restart at each 30-cycle window, no forget gate | 35.771487 | -15.234329 |

Restarting at zero limits how many additions can pile up. Restarting at each
engine, the arrangement of the 1997 experiments, keeps this cell's largest
state at 85.707215; restarting at each 30-cycle window, the window length of
Wen et al. (2026, Section 3.4, Table 6), as the project's stage
does, keeps it at 35.771487. Even 30 additions of this hand-set cell are
enough to reach far along $h$'s flat part. How far the state travels within a
window depends on the multipliers, which training sets.

The key idea is that without a forget gate the internal state is a running
total of everything the cell has ever let in, so on an unbroken stream any
imbalance between positive and negative additions accumulates without limit
until $h$ is flat and the cell stops learning; a forget gate below 1 makes old
additions fade, which bounds the total, and training is what teaches the gate
when to drop to zero.

## The cell as PyTorch computes it

### The equations, and where each piece comes from

PyTorch's documentation for `torch.nn.LSTM` gives the cell it computes as six
equations (PyTorch, n.d.), three of which the code in the next section checks
against the installed library's own documentation:

$$\begin{aligned}
i_t &= \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
f_t &= \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
g_t &= \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
o_t &= \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}$$

- $x_t$ is the input at step $t$; in this project, one rescaled reading of one
  sensor, so a list of length 1.
- $h_{t-1}$ is the previous **hidden state**, the cell outputs of the step
  before, and $c_{t-1}$ the previous **cell state**. With hidden size $H$, each
  is a list of $H$ numbers.
- $i_t$, $f_t$ and $o_t$ are the input, forget and output gates; $g_t$ is the
  squashed cell input, which PyTorch calls the cell gate.
- Each $W$ is a matrix of multipliers and each $b$ a list of $H$ offsets; the
  first letter of the subscript says whether it multiplies the input ($i$) or
  the hidden state ($h$), and the second which gate it belongs to. Here each
  $W_{i\cdot}$ has shape $[H, 1]$, one reading in, and each $W_{h\cdot}$ has
  shape $[H, H]$, so every cell's previous output feeds every gate of every
  cell.
- $\sigma$ is the logistic sigmoid; $\odot$ multiplies two lists of equal
  length entry by entry.

An `nn.LSTM` with hidden size $H$ (the stage uses 32, assumption A-26)
therefore holds $H$ cells side by side, each with its own entry in each gate:
in the 1997 vocabulary, $H$ blocks of size 1. PyTorch stores the four input-side matrices stacked into
one array of $4H$ rows, in the order $i, f, g, o$, and the four hidden-side
ones likewise (PyTorch, n.d.); the
[implementation guide](../../src/thesis_gnn_rul_xai/models/temporal/README.md)
counts what that comes to for the stage.

| Piece of the PyTorch cell | Where it comes from |
|---|---|
| Input gate $i_t$ and output gate $o_t$ | Hochreiter and Schmidhuber (1997, Section 4) |
| Forget gate $f_t$ | Gers et al. (2000, Eq. 6) |
| $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ | Gers et al. (2000, Eq. 7); with $f_t = 1$ it is the 1997 state update. The name cell state is from Gers et al. (2000, Section 2) |
| $\tanh$ for $g_t$ | Neither paper: both use a $g$ from −2 to 2 (Hochreiter & Schmidhuber, 1997, Eq. 5; Gers et al., 2000, Section 2). The 1997 Eq. 5 is $2\tanh(x/2)$; Gers et al. give only the range |
| $\tanh(c_t)$ in $h_t = o_t \odot \tanh(c_t)$ | Not the 1997 paper, whose $h$ is $\tanh(x/2)$, from −1 to 1 (Hochreiter & Schmidhuber, 1997, Eq. 4). Gers et al. (2000, Section 2) give only the range, −1 to 1, which $\tanh$ also has, so their paper does not settle whether their $h$ is $\tanh(x/2)$ or $\tanh$. The product with the output gate is the 1997 cell output $y^{\mathrm{out}}\,h(s)$ |
| Gates read only $x_t$ and $h_{t-1}$ | Narrower than 1997, where a gate may read any unit, other gates included (Hochreiter & Schmidhuber, 1997, Section 4) |
| $h_0 = c_0 = 0$ unless given | The 1997 cell's $s_{c_j}(0) = 0$ (Hochreiter & Schmidhuber, 1997, Section 4); PyTorch's documentation (PyTorch, n.d.) |
| Derivatives through every route | Not the truncated rule of either paper; see below |

**Do not confuse the hidden state with the cell state.** In the 1997
vocabulary the cell state $c_t$ is the internal state $s_c$, the carousel's
content, and the hidden state $h_t$ is the cell output $y^c$. The gates read
the hidden state, never the cell state; the cell state is reached only through
the output gate and $\tanh$.

Neither paper lets a gate read the internal state directly either: the gates'
net inputs are sums over other units' outputs $y^u(t-1)$ (Hochreiter &
Schmidhuber, 1997, Appendix A.1, Eqs. 7 and 8; Gers et al., 2000, Eqs. 1, 2
and 6). LSTM variants that do give the gates the cell
state exist under the name of **peephole connections**. This project holds no
paper that introduces them, so this page says only that PyTorch's equations
contain no such term.

### Two steps of the PyTorch cell on the same readings

The cell below is the PyTorch cell with one entry per list, $H = 1$, run on
the same two readings with the same multipliers as the
[1997 cell](#one-step-of-the-1997-cell-on-real-readings), now including the
forget-gate row.

<details class="process-example">
<summary>Example in Python</summary>

```python
def step_pytorch(x, h, c):
    """One step of the cell nn.LSTM computes, written out by hand."""
    z = reading_w * x + hidden_w * h + offset
    i, f, g, o = sigmoid(z[0]), sigmoid(z[1]), np.tanh(z[2]), sigmoid(z[3])
    c = f * c + i * g
    return [x, z[0], i, f, z[2], g, c, np.tanh(c), o, o * np.tanh(c)]

new = []
h = c = 0.0
for x in two:
    new.append(step_pytorch(x, h, c))
    h, c = new[-1][9], new[-1][6]
trace_pytorch = pd.DataFrame({
    "Quantity": ["Reading", "Input gate net input", "Input gate i", "Forget gate f",
                 "Cell input net input", "Squashed cell input g", "Cell state c",
                 "tanh(c)", "Output gate o", "Hidden state h (cell output)"],
    "Cycle 191": new[0], "Cycle 192": new[1],
})

# The same multipliers in a one-cell nn.LSTM, in 64-bit numbers.
lstm = torch.nn.LSTM(input_size=1, hidden_size=1, batch_first=True, dtype=torch.float64)
with torch.no_grad():
    lstm.weight_ih_l0[:, 0] = torch.tensor(reading_w)   # rows in PyTorch's order: i, f, g, o
    lstm.weight_hh_l0[:, 0] = torch.tensor(hidden_w)
    lstm.bias_ih_l0[:] = torch.tensor(offset)
    lstm.bias_hh_l0[:] = 0.0                           # nn.LSTM adds a second offset; set it to zero
    # Input shape [1 window, 2 cycles, 1 reading per cycle]; batch_first=True puts the window axis first.
    outputs, (h_n, c_n) = lstm(torch.tensor(two).reshape(1, 2, 1))
np.testing.assert_allclose(outputs[0, :, 0].numpy(), [new[0][9], new[1][9]], rtol=0, atol=1e-14)
assert abs(c_n.item() - new[1][6]) < 1e-14

# Wen et al.'s Eq. 2 joins h and x into one list: i_t = sigma(W_i . [h_{t-1}, x_t] + b_i).
W_i = np.array([hidden_w[0], reading_w[0]])            # [0.5, 1.0], in the order h, x
joined = sigmoid(W_i @ np.array([new[0][9], two[1]]) + offset[0])
reversed_order = sigmoid(W_i @ np.array([two[1], new[0][9]]) + offset[0])
assert abs(joined - new[1][2]) < 1e-14 and abs(reversed_order - new[1][2]) > 0.05

doc = torch.nn.LSTM.__doc__
for line in (r"f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})",
             r"c_t = f_t \odot c_{t-1} + i_t \odot g_t", r"h_t = o_t \odot \tanh(c_t)",
             "Defaults to zeros if (h_0, c_0) is not provided."):
    assert line in doc, line
```

</details>

<!-- sample: trace_pytorch -->
| Quantity | Cycle 191 | Cycle 192 |
|---|---|---|
| Reading | 2.074416 | 2.149321 |
| Input gate net input | 1.074416 | 1.344450 |
| Input gate i | 0.745436 | 0.793221 |
| Forget gate f | 0.954227 | 0.963378 |
| Cell input net input | 2.074416 | 1.759064 |
| Squashed cell input g | 0.968925 | 0.942398 |
| Cell state c | 0.722271 | 1.443350 |
| tanh(c) | 0.618314 | 0.894370 |
| Output gate o | 0.631163 | 0.724105 |
| Hidden state h (cell output) | 0.390257 | 0.647618 |

At cycle 191 both cells start from zero, so their input and output gates are
those of the 1997 table, 0.745436 and 0.631163. They part at the cell input,
which the 1997 curve $g$ stretches to 1.553566 and $\tanh$ holds at 0.968925,
and again on the way out, where $h$ turned the 1997 state into 0.521969 and
$\tanh$ turns this one into 0.618314. The forget gate, 0.954227, has nothing to
act on yet, because the old cell state is zero. At cycle 192 each cell reads
its own previous output, so every net input differs; the 1997 state kept all
of its 1.158084, and this cell state keeps 0.963378 of its 0.722271. Two cycles
are enough to show each difference between the two cells' forward
computations as this demonstration sets them up: the forget gate and the two
squashing curves.

The code also loads the same multipliers into a one-cell `torch.nn.LSTM` and
checks that its outputs and final cell state equal the hand-written column to
within $10^{-14}$. The same agreement over a whole window, with the stage's own
starting multipliers, is shown in
[Temporal algorithm](../learn/09_temporal.md#one-lstm-per-sensor). Finally it
recomputes the input gate at cycle 192 in the joined form of Eq. 2 of Wen et
al. (2026, Section 2.2), $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$, where
$[h_{t-1}, x_t]$ is the previous output and the reading written as one list and $W_i$ is the row $[0.5, 1.0]$ of the gate's two multipliers in that order. It gives the
table's 0.793221; joining the two the other way round gives a different number,
so the order matters.

### What PyTorch differentiates

PyTorch obtains derivatives by automatic differentiation through every
operation, so it follows every route from a multiplier to the result,
including the ones both papers cut: the previous hidden state feeding the gates
and the cell input. The example below differentiates the hidden state at cycle
192 with respect to the cell input's reading multiplier, 1.0 in the
multipliers table, in the PyTorch cell above. It does so four ways. The first
nudges the multiplier up and down by 0.000001 and divides the change in output
by the change in multiplier. The second is automatic differentiation. The third
is automatic differentiation with the routes cut as both papers cut them,
using `detach` on cycle 191's hidden state wherever it enters cycle 192's net
inputs. The fourth runs Gers et al.'s Eq. 15 by hand over the two cycles and
multiplies the result by what a change in the cell state does to the hidden
state, $o_t\,(1 - \tanh^2 c_t)$.

<details class="process-example">
<summary>Example in Python</summary>

```python
def hidden_192(w_cell, truncate):
    """Run the two steps in PyTorch; truncate=True cuts the routes the 1997 and 2000 rules cut."""
    w = torch.tensor(reading_w)
    w[2] = w_cell
    u, b = torch.tensor(hidden_w), torch.tensor(offset)
    h = c = torch.zeros((), dtype=torch.float64)
    for x in torch.tensor(two):
        previous = h.detach() if truncate else h
        z = w * x + u * previous + b
        i, f, g, o = torch.sigmoid(z[0]), torch.sigmoid(z[1]), torch.tanh(z[2]), torch.sigmoid(z[3])
        c = f * c + i * g
        h = o * torch.tanh(c)
    return h

def derivative(truncate):
    """d(hidden state at 192) / d(cell input's reading multiplier), by automatic differentiation."""
    multiplier = torch.tensor(reading_w[2], dtype=torch.float64, requires_grad=True)
    hidden_192(multiplier, truncate).backward()
    return multiplier.grad.item()

rows, record = [], 0.0
for x, z_i, i, f, z_c, g, c, tanh_c, o, h in new:
    record = record * f + (1.0 - g ** 2) * i * x        # Eq. 15, started at 0 (Eq. 18)
    rows.append([x, f, i, 1.0 - g ** 2, record])
carried = pd.DataFrame(rows, columns=["Reading", "Forget gate", "Input gate", "Slope of tanh at the cell input",
                                      "dc/dw (Eq. 15)"])
carried.insert(0, "Cycle", [191, 192])
by_hand = new[1][8] * (1.0 - new[1][7] ** 2) * record

nudge = 1e-6
up = hidden_192(torch.tensor(reading_w[2] + nudge, dtype=torch.float64), False).item()
down = hidden_192(torch.tensor(reading_w[2] - nudge, dtype=torch.float64), False).item()
derivatives = pd.DataFrame({
    "Method": ["Nudge the multiplier by 0.000001 each way", "Automatic differentiation, every route",
               "Automatic differentiation, routes cut", "Eq. 15 by hand, times o (1 - tanh(c)^2)"],
    "Change in the cycle-192 hidden state per unit of multiplier": [(up - down) / (2 * nudge), derivative(False),
                                                                     derivative(True), by_hand],
})
assert abs(by_hand - derivative(True)) < 1e-12
```

</details>

<!-- sample: carried -->
| Cycle | Reading | Forget gate | Input gate | Slope of tanh at the cell input | dc/dw (Eq. 15) |
|---|---|---|---|---|---|
| 191 | 2.074416 | 0.954227 | 0.745436 | 0.061185 | 0.094613 |
| 192 | 2.149321 | 0.963378 | 0.793221 | 0.111885 | 0.281899 |

<!-- sample: derivatives -->
| Method | Change in the cycle-192 hidden state per unit of multiplier |
|---|---|
| Nudge the multiplier by 0.000001 each way | 0.047443 |
| Automatic differentiation, every route | 0.047443 |
| Automatic differentiation, routes cut | 0.040846 |
| Eq. 15 by hand, times o (1 - tanh(c)^2) | 0.040846 |

Eq. 15 starts at 0 (Eq. 18), so cycle 191 gives
$0 \times 0.954227 + 0.061185 \times 0.745436 \times 2.074416 = 0.094613$, and
cycle 192 gives
$0.094613 \times 0.963378 + 0.111885 \times 0.793221 \times 2.149321 = 0.281899$;
the forget gate is what multiplies the old record.
Multiplied by $0.724105 \times (1 - 0.894370^2)$ it gives the "routes cut" row
exactly. Nudging and automatic differentiation agree, so the second row is the
true derivative. The cut version is smaller because it drops the routes through
cycle 191's hidden state: the multiplier changed that hidden state, which
entered all four of cycle 192's net inputs. Training with PyTorch therefore
follows every route within a window, which the 1997 paper chose not to do.
Hochreiter and Schmidhuber (1997, Section 6) found no significant difference in
a few experiments, and nothing on this page measures the difference for this
project's stage.

**Do not confuse the papers' truncation with cutting the history into
windows.** The 1997 and 2000 rules keep the whole sequence but drop, at every
step, each route that leaves a cell or a gate, so only the carousel carries
derivatives back (Hochreiter & Schmidhuber, 1997, Appendix A.1; Gers et al.,
2000, Section 3.2). The project keeps every route, but only within one 30-cycle
window: each window starts from zero, so no derivative reaches more than 29
cycles back.

The key idea is that PyTorch's cell is the 2000 cell with $\tanh$ in place of
the 1997 squashing curves, gates that read only the input and the previous
hidden state, and derivatives taken through every route within a window rather
than the truncated ones both papers used.

## Do not confuse the LSTM with the GRU (Cho et al., 2014)

The project also offers, as an option of its own (assumption A-36), a **gated
recurrent unit**, or **GRU**, the name PyTorch's `torch.nn.GRU` uses for the
hidden unit that Cho et al.
(2014, [Section 2.3](../references/cho_2014_rnn_encoder_decoder.pdf#page=3))
propose. The paper itself calls it only a new type of hidden unit,
motivated by the LSTM but simpler to compute. For hidden unit $j$, Eqs. 5 to 8
are:

$$r_j = \sigma\big([W_r x]_j + [U_r h_{\langle t-1 \rangle}]_j\big), \qquad z_j = \sigma\big([W_z x]_j + [U_z h_{\langle t-1 \rangle}]_j\big)$$

$$h_j^{\langle t \rangle} = z_j\, h_j^{\langle t-1 \rangle} + (1 - z_j)\, \tilde h_j^{\langle t \rangle}, \qquad \tilde h_j^{\langle t \rangle} = \phi\big([W x]_j + [U (r \odot h_{\langle t-1 \rangle})]_j\big)$$

- $x$ is the input and $h_{\langle t-1 \rangle}$ the previous hidden state;
  $[\,\cdot\,]_j$ takes entry $j$ of a list.
- $r_j$ is the **reset gate** and $z_j$ the **update gate**, both logistic;
  $r$ without an index is the list of every unit's reset gate.
- $\tilde h_j$ is the proposed new value, and $\phi$ is $\tanh$ in the paper's
  experiments ([Section 4.1.1](../references/cho_2014_rnn_encoder_decoder.pdf#page=6)).
- No offsets are printed in these equations.

The differences from the LSTM are mechanical:

- **One state, not two.** The GRU keeps only its hidden state; there is no
  separate cell state behind an output gate, and the whole state is visible to
  the next layer at every step.
- **Fewer gates.** As the two LSTM papers name them, the cell with a forget
  gate has three gates, input, output and forget, and the 1997 cell the first
  two (Gers et al., 2000, Section 3.1; Hochreiter & Schmidhuber, 1997,
  Section 4); Cho et al. (2014, Section 2.3, footnote 1) credit the LSTM with
  four, without listing them, and PyTorch's documentation also lists four,
  counting $g_t$ (PyTorch, n.d.). The GRU has two. Its update
  gate does in one blend what the LSTM's forget and input gates do separately:
  $z$ keeps the old value and $1 - z$ admits the new, and the two always add up
  to 1, whereas an LSTM's forget and input gates are free to be both high or
  both low.
- **The reset gate acts before the proposal is formed.** When the reset gate
  $r$ is near 0, the whole list and not a single entry, since $U$ mixes every
  entry of $r \odot h$, the proposal ignores the previous state and is computed
  from the input alone (Section 2.3).

One detail separates the paper from PyTorch. Cho et al. (2014, Section 2.3,
Eq. 8) multiply the previous state by the reset gate *before* the matrix $U$;
PyTorch's `nn.GRU` documentation applies the reset gate *after* its matrix, to
$W_{hn} h_{t-1} + b_{hn}$ (PyTorch, n.d.). The two placements give different results in
general. The project's GRU route uses PyTorch's version (assumption A-36),
worked through with numbers in
[Temporal algorithm](../learn/09_temporal.md#one-gru-per-sensor).
Wen et al.'s own model contains no GRU; a GRU-based method from the
literature, their reference [55], appears only among the methods compared in
their Tables 7 to 10 (Wen et al., 2026,
[Section 4.1](../references/wen_2026_estgnn_dynamic_shap.pdf#page=10)).

## What this project takes from these papers

Wen et al. (2026, [Section 2.2](../references/wen_2026_estgnn_dynamic_shap.pdf#page=4))
print, as their Eqs. 1 to 6, the cell PyTorch computes: the 1997 cell with the
forget gate of Gers et al. and $\tanh$ squashing, written in the joined form
$W \cdot [h_{t-1}, x_t]$. Their Eq. 7 gives every sensor its own LSTM over its
window of readings. Neither Hochreiter and Schmidhuber nor Gers et al. appear
anywhere in Wen et al.'s text, so the equations are all the paper says about
which version of the cell it means.

The project's `LSTMStage` builds that stage from `torch.nn.LSTM`, one network
per sensor as Eq. 7 says. Its width, depth and the conversion of its last
hidden state into one number per sensor are this project's choices, because
Wen et al. give none of them (assumption A-26). It is not the default stage
(assumption A-11). What it adds on FD001 depends on whether a training run
ends by putting back the weights from its best point on the engines held out
of training to check it, a step Wen et al. never describe (finding F-16). From
the papers on this page it inherits:

- **The forget gate** of Gers et al., and with it a carousel whose flow is
  constant only while that gate is at 1.
- **$\tanh$ squashing** rather than the curves of 1997 Eqs. 4 and 5.
- **Blocks of size 1**: each of the 32 cells has its own gates.
- **No peephole connections**, as in both papers.
- **Derivatives through every route within a window**, not the truncation both
  papers used.
- **A fresh start for every window.** The stage's memory starts at zero for
  every 30-cycle window, as
  [Temporal algorithm](../learn/09_temporal.md#2--read-cycles-with-memory)
  shows. Wen et al. (2026, Section 2.2, Eq. 7) run each sensor's LSTM over one
  window but state no starting memory; the zero start is PyTorch's default
  (PyTorch, n.d.), which this project keeps. The restart at every window is
  the arrangement of the 1997 experiments, which reset after
  each sequence, not the continual stream of the 2000 paper. The problem the
  forget gate was invented for does not arise in that form here, although the
  gate is present; what it contributes to this project's stage has not been
  measured.
- **PyTorch's default starting values, not those of either paper.** Wen et al.
  state no starting values for the stage, and the project keeps PyTorch's
  (assumption A-34): every multiplier and offset is drawn evenly from
  $\pm 1/\sqrt{32}$ (PyTorch, n.d.). Each gate
  has two offsets, so their sum lies within $\pm 2/\sqrt{32}$. At the first
  cycle of a window, where the previous hidden state is zero, a reading of zero
  therefore gives a forget gate between 0.412521 and 0.587479. The offsets of
  Gers et al. (2000, Section 4.2) give 0.622459 for the first block and
  0.731059 for the second, rising towards 1 for later blocks.

<details class="process-example">
<summary>Example in Python</summary>

```python
stage = LSTMStage(window_size=30, n_nodes=14)
bound = 1.0 / np.sqrt(stage.hidden_size)
assert len(stage.cells) == 14 and stage.hidden_size == 32
for cell in stage.cells:
    for values in (cell.weight_ih_l0, cell.weight_hh_l0, cell.bias_ih_l0, cell.bias_hh_l0):
        assert values.abs().max().item() <= bound
starting = pd.DataFrame({
    "Quantity": ["Largest starting multiplier or offset", "Largest sum of a gate's two offsets",
                 "Forget gate at zero input, PyTorch's lowest", "Forget gate at zero input, PyTorch's highest",
                 "Forget gate at zero input, Gers et al.'s first block (+0.5)",
                 "Forget gate at zero input, Gers et al.'s second block (+1)"],
    "Value": [bound, 2 * bound, sigmoid(-2 * bound), sigmoid(2 * bound), sigmoid(0.5), sigmoid(1.0)],
})
```

</details>

<!-- sample: starting -->
| Quantity | Value |
|---|---|
| Largest starting multiplier or offset | 0.176777 |
| Largest sum of a gate's two offsets | 0.353553 |
| Forget gate at zero input, PyTorch's lowest | 0.412521 |
| Forget gate at zero input, PyTorch's highest | 0.587479 |
| Forget gate at zero input, Gers et al.'s first block (+0.5) | 0.622459 |
| Forget gate at zero input, Gers et al.'s second block (+1) | 0.731059 |

How the stage runs on a real window, cycle by cycle, is taught in
[Temporal algorithm](../learn/09_temporal.md#one-lstm-per-sensor); how its last
hidden state becomes the number each sensor passes to the graph is taught in
the same page's steps
[3](../learn/09_temporal.md#3--keep-the-last-hidden-state) and
[4](../learn/09_temporal.md#4--turn-into-numbers-per-sensor).

## References

- Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
  Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN
  encoder–decoder for statistical machine translation. In A. Moschitti, B.
  Pang, & W. Daelemans (Eds.), *Proceedings of the 2014 Conference on
  Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1724–1734).
  Association for Computational Linguistics. https://doi.org/10.3115/v1/D14-1179
  [Copy](../references/cho_2014_rnn_encoder_decoder.pdf)
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget:
  Continual prediction with LSTM. *Neural Computation*, *12*(10), 2451–2471.
  https://doi.org/10.1162/089976600300015015
  [Copy](../references/gers_2000_learning_to_forget.pdf). The copy held is the
  authors' accepted manuscript, cited here by section and equation; the one
  page number, for a quotation, is that of the PDF file, one more than the
  manuscript's printed page number.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural
  Computation*, *9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
  [Copy](../references/hochreiter_1997_long_short_term_memory.pdf). The copy
  held is the authors' 32-page manuscript, cited here by section and equation.
- NASA Prognostics Center of Excellence. (2008). readme.txt, the description of
  the C-MAPSS data set distributed inside its zip file. NASA Prognostics Data
  Repository. [Copy](../references/nasa_2008_cmapss_readme.txt)
- PyTorch. (n.d.). https://pytorch.org/. Software, so no copy is kept; the code
  on this page reads the documentation of the installed library.
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
