# Reading a result

A prediction $\hat{y}$ is a number of cycles, and so is $y$, the remaining life
it is scored against. In this project's scores $y$ is the test engine's label:
its true remaining life, capped at 125 cycles like the training labels
(assumptions A-01 and A-02), so 125 for the 11 FD001 test engines that have
more than that left (the second warning under "What may and may not be compared
with the paper", below, says what the cap does to each measure). NASA's description of
the data counts that life in "operational cycles" (NASA Prognostics Center of
Excellence [NASA PCoE], 2008). The error is the
difference between the prediction $\hat{y}$ and the label $y$, positive when the model was optimistic, the same way
round as the dataset paper's "Estimated RUL – True RUL" (Saxena et al., 2008,
p. 7):

$$d = \hat{y} - y$$

An optimistic model expects failure later than it comes, which leaves a failing
engine flying. A pessimistic one, with a negative error, sends an engine for
maintenance while it still had life in it. **The two are not equally bad**
(Saxena et al., 2008, p. 7; Wen et al., 2026, p. 9), and the measures below
differ in whether they take that into account.

(Corrected 2026-10-03: this page called $y$ the true remaining life, while
every score it reads is computed against the capped label. Found by a review
of the wording around the capped label.)

---

## The four measures

### RMSE — root mean squared error

$$\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} d_i^{\,2}}$$

The typical size of the error over the $N$ test engines, in cycles: an
RMSE of 16 means the model is typically about 16 cycles out. One cycle is one
flight on this project's reading of NASA's description of the files and of the
dataset paper's "one measurement snapshot per flight" (NASA PCoE, 2008; Saxena
et al., 2008, p. 5); the [Data](01_the_problem.md#what-one-cycle-is) page sets
out that reading.

Squaring makes it **symmetric**: 10 cycles too high counts exactly like 10 too
low, so it is blind to the asymmetry above. It is "frequently employed to
evaluate the accuracy of Remaining Useful Life (RUL) predictions" anyway (Dida
et al., 2025, p. 5), and using the same measure as earlier papers is what makes
results comparable (Li et al., 2021, p. 5).

### The NASA score

$$s_i = \begin{cases}
e^{-d_i/13} - 1 & d_i < 0 \quad \text{early} \\
e^{\;d_i/10} - 1 & d_i \ge 0 \quad \text{late}
\end{cases}$$

Each test engine $i$ gets a penalty $s_i$, and the score, from NASA's 2008
prognostics data challenge (Saxena et al., 2008, p. 7; Wen et al., 2026, p. 9),
is the sum of those penalties. In the form this project uses (assumption A-15),
the divisor for a late prediction is 10, smaller than the 13 for an early one,
so **the same error costs more when the model was optimistic**: 10 cycles late
costs $e^{1} - 1 = 1.72$, and 10 cycles early costs $e^{10/13} - 1 = 1.16$.

It is exponential, so one bad engine can dominate: a single engine 40 cycles
late costs $e^{4} - 1 = 53.6$, more than 99 engines each 2 cycles late put
together, $99 \times (e^{0.2} - 1) = 21.9$. Because it is a sum, it grows with
the number of test engines, so it compares only runs on the same sub-dataset.
Lower is better, and the $- 1$ makes an exact prediction cost $e^{0} - 1 = 0$.

> The paper's Equations 20 and 21, taken together, give the *gentler* penalty
> to late predictions: the opposite of what the paper's own text says (Wen et
> al., 2026, p. 9, Eqs. 20–21). The NASA dataset paper behind the challenge,
> Saxena et al. (2008), has the same mismatch: its printed equation also gives
> late predictions the gentler 13 (p. 7, Eq. 11), while its text says late
> predictions were "more heavily penalized" (p. 7) and its Figure 9 divides
> late errors by 10 (p. 8, Fig. 9; finding F-20). This project follows what
> both papers say in words (assumption A-15).

### R² — coefficient of determination

$$R^2 = 1 - \frac{\sum_i (\hat{y}_i - y_i)^2}{\sum_i (y_i - \bar{y})^2}$$

The bottom sum is the squared error of always guessing $\bar{y}$, the average
true label; the top sum is the model's own squared error. The fraction is the
model's error as a share of that guessing error, so a model with no error scores
1 − 0 = 1, **one exactly as bad as always guessing the average scores
1 − 1 = 0**, and a worse one scores below 0.

> The paper's Equation 23 puts the predicted values in the denominator instead
> of the true ones (Wen et al., 2026, p. 9, Eq. 23), so a model could raise its
> score by spreading its predictions more widely without becoming any more
> accurate. This project uses the definition above (assumption A-16), which is
> how this project reads the paper's own text beside the equation: R-squared
> measures "the proportion of variance in the dependent variable that is
> predictable from the independent variables" (Wen et al., 2026, p. 9). The
> dependent variable is the quantity being predicted, here the remaining life,
> and the independent variables are the inputs it is predicted from, so the
> variance meant is that of the labels $y_i$, not of the predictions: the
> bottom sum above divided by the number of engines.

### ASUE — average safe underestimation error

$$\mathrm{ASUE} = \frac{1}{N_{\text{under}}}
\sum_{i\,:\,\hat{y}_i < y_i} \max\left(0,\; y_i - \hat{y}_i\right)$$

The average shortfall, counted **only over the $N_{\text{under}}$ engines the
model was pessimistic about** (Wen et al., 2026, p. 9, Eq. 24). It is the cost
of caution: how much life, on average, is thrown away on the engines serviced
early. Because the NASA score charges early errors less, a model can lower
that score by leaning early, and ASUE shows the price.

Read it together with $N_{\text{under}}$, which every results file written
since 2026-10-03 holds beside each ASUE as `n_early`. A file stored before then
does not; for it, $N_{\text{under}}$ is the number of test engines minus
`n_late`, the count of engines predicted late, which every stored run's
`results.csv` holds.

The real FD001 test labels show why the count matters. A model that predicts
124 cycles for all 100 test engines is below the label only on the 11 engines
whose labels were capped at 125, by one cycle each, so its ASUE is 1.0 over 11
early engines. One cycle higher, at 125, it is below no label at all:
$N_{\text{under}}$ is 0, the sum is 0, and the formula is 0 / 0, which has no
value. The code then gives "not a number" with a warning, not 0, since this
model is late on all 100 engines, the costly direction, and an ASUE of 0 would
have ranked it the best there is (assumption A-39).

---

## One seed is not a result

Training draws random numbers: the network's weights start from random values,
the training examples arrive in a shuffled order, and the held-back engines, a
fifth of the training engines (assumption A-06), are picked at random. A
**seed** is the starting number those draws are generated from, so the same
network trained with two different seeds gives two different results.

So **anything quoted as a result here is a mean over at least five seeds, with
its standard deviation**:

```bash
python scripts/train.py --seeds 5
```

On FD001, the current default settings of `scripts/train.py`, which leave out
the LSTM stage (assumption A-11), trained with seeds 42 to 46 gave a test RMSE
of 16.695, 16.131, 16.843, 15.970 and 16.843 cycles. Their mean is 16.496, and their standard
deviation, the typical distance of one seed's RMSE from that mean, is 0.415, so
the result is quoted as **16.496 ± 0.415**. It is the stored
run `fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z`,
listed in the playground's Results tab as finished at 2026-09-14 23:39 UTC;
opening it there shows its other three measures the same way.

The `--fixed-split` flag holds back the same engines for every seed. That
removes one of the three random draws above, so the spread it gives is smaller
than a genuine repetition would show, and a number produced with it must say so.

---

## When is a difference real?

To test whether one configuration, a full set of settings, beats another, train
both with the same five seeds and subtract within each seed, so that every
difference compares two runs that differ only in the setting under test. Under
the current default, replacing the paper's sensor graph (Wen et al., 2026,
Table 3) with one joining every sensor to every other (assumption A-19) raised
the RMSE by 2.125, 2.822, 2.997, 3.636 and 2.342 cycles:

```
mean 2.784        standard deviation 0.592
t = mean ÷ (standard deviation ÷ √5) = 2.784 ÷ 0.265 = 10.5
```

Dividing the standard deviation by √5 gives the uncertainty of an average of
five numbers, so $t$ counts how many of those uncertainties the mean lies from
zero. A paired *t*-test turns $t$ into *p*: the probability, if the setting made
no difference at all, of a $t$ at least that far from zero. Here *p* = 0.00046.
With five pairs, *p* falls below 0.05, the line this project uses for calling
a single difference real (assumption A-37), once $t$ lies more than 2.776 from
zero.

A graph built from the data instead, joining the 41 most strongly correlated
sensor pairs (as many edges as the paper's graph; Wen et al., 2026, Table 3),
changed the RMSE by 0.238, 0.311, 0.146, 0.457 and −0.672: mean 0.096, standard deviation 0.444, $t$ =
0.48, *p* = 0.655. One seed points the other way, the spread swamps the mean,
and five seeds cannot tell the two graphs apart by test error. So, on these
measurements (assumption A-27), most of the 2.784 is the cost of every sensor
receiving the same average, not the worth of the engineering knowledge the
paper built its graph from (Wen et al., 2026, p. 4, Section 2.3.1).

---

## The rule that catches people out

> **A comparison is a fact about one whole configuration — architecture,
> preprocessing and split together — not about the two options being
> compared.**

The first comparison above shows it. In an earlier configuration the inputs
were squeezed into the range 0 to 1, the graph was rescaled in its symmetric
form, and the learning rate shrank a little after every pass over the training
data (assumptions A-03, A-21 and A-17 record what replaced each). There the
same swap of
graphs cost 0.044 cycles (*p* = 0.913), against 2.784 under the current default.
The graph was the same both times. At least those three settings differed, and
no stored run changes one of them on its own, so the runs cannot say what
caused the change (finding F-04).

A verdict close to the 0.05 line can flip when only the learning-rate schedule
changes. With no engines held back, the row form of the graph's rescaling beat
the symmetric form by 0.35 cycles at *p* = 0.047 when the rate shrank after
every pass, and by 0.25 at *p* = 0.068 under the single drop at the halfway
point that the project now uses (assumption A-17). With a fifth of the engines
held back (assumption A-06), no difference survives under either, with or
without early stopping (assumption A-18): the four
comparisons, all re-run at one commit on 2026-10-02, give differences between
0.091 and 0.199 cycles at *p* between 0.276 and 0.735, and which form comes out
nominally ahead changes from one to the next (assumption A-21).

So a comparison must name the configuration it was made under, and be re-run
whenever any part of it changes.

---

## What may and may not be compared with the paper

The paper reports several variants (Wen et al., 2026, p. 12, Table 11). Under
the default settings, which leave the LSTM stage out (assumption A-11), only one
of them is comparable:

| Paper's variant | RMSE | Comparable? |
|---|---|---|
| ESTGNN, full model | 11.78 | **Not under the default** — includes the LSTM stage |
| ESTGNN w/o LSTM | 14.93 | **Yes** — this is the matching row |
| ESTGNN w/o GCN | 16.42 | No |
| ESTGNN w/o physics prior | 12.34 | No |

Even that comparison carries three warnings:

1. **The settings were chosen to land near the target.** Where the paper is
   silent, the option whose test error came closest to the paper's was picked,
   so the
   configuration is not independent of the number it is compared with. The
   settings used before that rule give 20.35 ± 0.75 instead of the default's
   result above, and the paper specifies none of the settings that differ, so
   it gives no reason to prefer either number (assumption A-22).
2. **Capping the test labels at 125 can only lower the RMSE.** No prediction
   exceeds 125, so for an engine whose true remaining life is above 125 the
   prediction was early, and capping the label shrinks that early error
   without changing its direction: engine 25, with 145 cycles left, is
   charged exactly 20 cycles less whatever the model predicted. The NASA
   score can only fall for the same reason, but R-squared and ASUE are not
   guaranteed to improve; the
   [Preprocessing](04_the_pipeline.md#1--cap-the-labels) page notes in general
   that capping does not guarantee every measure improves. On the current default capping lowers
   the RMSE by 1.51 cycles, from 18.01 to 16.50 (1.31 in an earlier
   configuration), so measured without it the gap to 14.93 would be wider
   (assumption A-02; the current figure added 2026-10-02). (Corrected 2026-10-02: this said capping
   can only lower "the error", for a reason the Preprocessing page does not
   give; found by the
   search for that claim after the review of the playground's information
   cards on 2026-09-26.)
3. **The paper gives its numbers with no spread** (Wen et al., 2026, Tables
   7–11) and never mentions seeds or repeated runs (assumption A-37 records the
   search of its text), so they have an unknown spread of their own.

## References

- Dida, M., Cheriet, A., & Belhadj, M. (2025). Remaining useful life
  prediction using attention-LSTM neural network of aircraft engines.
  *International Journal of Prognostics and Health Management*, *16*(2).
  https://doi.org/10.36001/ijphm.2025.v16i2.4274
  [Copy](../references/dida_2025_attention_lstm_rul.pdf)
- Li, T., Zhao, Z., Sun, C., Yan, R., & Chen, X. (2021). Hierarchical
  attention graph convolutional network to fuse multi-sensor signals for
  remaining useful life prediction. *Reliability Engineering & System Safety*,
  *215*, Article 107878. https://doi.org/10.1016/j.ress.2021.107878
  [Copy](../references/li_2021_hierarchical_attention_gcn.pdf)
- NASA Prognostics Center of Excellence. (2008). readme.txt, the description
  of the C-MAPSS data set distributed inside its zip file. NASA Prognostics
  Data Repository.
  [Copy](../references/nasa_2008_cmapss_readme.txt)
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
