# Assumptions

## Why this folder exists

This project re-implements the method of Wen et al. (2026). Their paper does
not describe every detail needed to write the code. Wherever a value or a
procedure had to be chosen by us instead of read from the paper, the choice is
recorded here, in a file of its own named after its code: assumption A-07 is
[`A-07.md`](A-07.md). What this project found out, about the paper, its data and
its own results, is not a choice, and are kept apart in [`../findings/`](../findings/README.md),
one file per finding, named the same way (`F-16.md`).

Keeping this log is not paperwork. It is what separates a re-implementation
from a guess. When a result from this code differs from the published one, the
first question a reader (or an examiner) will ask is "what did you do
differently?", and this folder is the answer. It also marks the boundary
between what the paper claims and what this code does, so that neither is
mistaken for the other.

**Rule followed in the code:** every one of these choices appears at the point
where it is used, in a comment beginning with `ASSUMPTION`, together with the
identifier in the table below. A reader who meets `ASSUMPTION A-17` in the code
finds the reasoning in `A-17.md` in this folder.

Until 2026-10-03 the assumptions and the findings were one file,
`docs/ASSUMPTIONS.md`. That day each entry became a file of its own under the
same code, its text unchanged apart from four things:

- its links, rewritten so that they work from the new folder;
- its headings, each moved up one level (two for a finding), so that the
  entry's title is the file's main heading;
- the paths of the audits, which moved to [`../audits/`](../audits/README.md)
  the same day;
- six phrases that located something by its place in the single file, such
  as "F-04 below" or "the summary table at the top of this file", or that
  called the whole log "this file"; they now link to what they meant or say
  "this log".

## Summary

| ID | Topic | What the paper says | What this project does |
|----|-------|---------------------|------------------------|
| [A-01](A-01.md) | Label ceiling `RUL_max` | Defines it symbolically in Equation 19, never prints a value | 125 cycles |
| [A-02](A-02.md) | Ceiling on test labels | Not discussed | Applied, because the model's output layer cannot exceed the ceiling |
| [A-03](A-03.md) | Input rescaling | Not described at all | **`zscore`** — the largest measured assumption in the project (3.83 cycles, F-10); selected under A-22 |
| [A-04](A-04.md) | Smoothing strength | Self-contradictory sentence in Section 3.2.3 | Read as `alpha = 0.5` |
| [A-05](A-05.md) | Window stride | Not in the text; **Fig. 1 indicates one time step** | 1 cycle |
| [A-06](A-06.md) | Validation split | None; trains for a fixed 200 epochs | **20 % of the engines**, reinstated 2026-09-08 after the A-22 removal (date corrected 2026-10-02 from 2026-09-02; see A-06); worth 0.32 cycles, which five seeds cannot distinguish from zero (p = 0.234) |
| [A-07](A-07.md) | Node feature content | Table 4 **prints** one number per sensor (`W(1) in R^(1x32)`), produced by an LSTM that is never specified | The last cycle of the window, keeping Table 4's shape |
| [~~A-08~~](A-08.md) | Order: smoothing and scaling | **Fig. 1 specifies it**: smoothing then normalisation | Withdrawn as an assumption; the paper's order is followed |
| [A-09](A-09.md) | Smoothing formula variant | Gives the recursive form as Equation 18 | Recursive form (`adjust=False` in pandas) |
| [A-10](A-10.md) | Identity of the 7th dropped sensor | Says "seven", names six | The seventh is `NRc`, deduced from their Table 3 |
| [A-11](A-11.md) | The LSTM stage | Central to their architecture | Omitted, matching their own published ablation |
| [A-12](A-12.md) | Bias in the graph layers | Equations 8-10 print only `W` | No bias, as printed; configurable |
| [A-13](A-13.md) | Scaling of the target | Sigmoid output called a "normalized RUL value" | Targets divided by `rul_max` for training |
| [A-14](A-14.md) | Weight initialisation | Not stated | Glorot uniform, following Kipf & Welling (2017) |
| [A-15](A-15.md) | Sign of the score function | Equations 20 and 21 contradict each other | The NASA convention the paper's own text describes |
| [A-16](A-16.md) | R-squared denominator | Equation 23 uses the predicted values | The standard definition, using the observed ones |
| [A-17](A-17.md) | Learning-rate schedule | Both endpoints given, mechanism not | **One drop of x0.1 at the halfway epoch**, so only the two printed rates are used; indistinguishable from exponential (p = 0.39) |
| [A-18](A-18.md) | Early stopping | None; trains a fixed 200 epochs | **On**, since the split returned on 2026-09-08 (date corrected 2026-10-02 from 2026-09-02; see A-06); fires between epochs 8 and 55 of a nominal 200 |
| [A-19](A-19.md) | What "without physics prior" removes | Names the ablation, never defines it | Replace the graph with a fully connected one |
| [A-20](A-20.md) | What "a sensor not considered" means | Equation 15 needs `f(S)`, but a network cannot take 13 inputs | Replace the sensor with its average training window |
| [A-21](A-21.md) | Adjacency normalisation formula | Says only "normalized ... incorporating self-loops"; never prints it | **Row `D^(-1)(A+I)`** — 0.25 cycles closer *with nothing held back* under the step schedule (p = 0.068, **no longer significant**), selected under A-22 |
| [A-22](A-22.md) | How the settings the paper leaves open were chosen | Leaves them unstated | On the author's instruction of 2026-08-29: where the paper is silent, the option that brings the result closest to its figure; where it states something, the paper. This selected the z-score (A-03) and row rescaling (A-21), so the default configuration is not independent of the figure it is compared with |
| [A-23](A-23.md) | Engines shorter than the window | Not discussed; 17 test engines across FD002 and FD004 are affected | Left out, following the only published treatment found; no FD001 engine is affected |
| [A-24](A-24.md) | What `phi_0` is in Equation 14 | Calls it "the average value of target variable across all samples" | The prediction at the average window, forced by A-20; **10.719 cycles** from the paper's wording |
| [A-26](A-26.md) | Size and depth of the recurrent stage | Table 4 has **no recurrent row**; Table 6 lists no hidden size, depth or projection | One layer, hidden size 32, a linear projection to the one number Table 4 prints; **fourteen dedicated cells**, which the paper does state |
| [A-25](A-25.md) | Where the scaling is fitted | Fig. 1 puts normalisation **before** the train and test datasets; never names the fitting set | Split first, fit on training engines only |
| [A-27](A-27.md) | A sensor graph built from the data | Describes none for its network, using correlations only to choose sensors, and never says what replaced its graph in Table 11; its Conclusions claim its own graph goes "beyond data-driven correlations" | The 41 most strongly correlated pairs on the training engines, as many edges as Table 3; **indistinguishable from Table 3 by test RMSE** (+0.096, p = 0.655), lower held-back error on every seed, by 1.8 to 2.0 cycles on average on each of four summaries of it (p 0.010 to 0.020), but its test predictions run later; with the recurrent stage, indistinguishable again by test RMSE (+0.002, p = 0.995) and no held-back advantage that survives a summary no pass was chosen by |
| [A-28](A-28.md) | Each sensor's last reading, average and change as node features | Never considered; Table 4 prints one number per sensor, from the LSTM stage | A stage outside the paper, not the default; **2.910 cycles below the last cycle** (p = 0.00025, engine interval [−5.19, −0.63]), indistinguishable from the recurrent stage on test error |
| [A-29](A-29.md) | Explaining the sensor graph itself | Explains sensor nodes only (Section 2.4), yet says its rankings set "edge weights" and validate its graph (Section 4.4) | Walk shares from GNN-LRP, of the score before the sigmoid, with the authors' own setting of gamma (0, 1.5, 0), 0 at the first graph layer; on the default network about half the score is the dense layers' constant (51 ± 4 %), and how much stays within a sensor's group is mostly set by the graph |
| [A-30](A-30.md) | What removing one connection means | Removes the whole graph once and retrains; never one connection | Delete it from the raw graph both ways, rescale, rerun without retraining; on the default network its effect ranks the connections against what flowed along them (rank correlation −0.26 ± 0.14, p = 0.013) |
| [A-31](A-31.md) | Placing explanation nodes on an engine | Table 2 names variables and component regions; Fig. 3 places a separate graph beside a schematic engine | An original schematic cutaway with section anchors, distinct markers for calculated and system quantities, and the stored model's graph overlaid; no claim of exact probe coordinates |
| [A-32](A-32.md) | Learning-rate sensitivity levels | Table 6 gives `0.001` at the start and `0.0001` at the end; it gives no sensitivity study | Test those rates, one tenth of both, and ten times both, keeping the ten-to-one ratio |
| [A-33](A-33.md) | Running the GNN-LRP paper's own check on this network | Not in the paper; the GNN-LRP paper's node flipping never says how a network on a fixed rescaled graph runs with nodes missing | A sensor's row and column of the rescaled graph zeroed, nothing rescaled; 20 random orders; walks beat random orders on every seed of all five configurations |
| [A-34](A-34.md) | Starting values of the recurrent stage | Prints the gate equations with one offset each (Equations 1, 2, 3 and 5); never says what any number starts from | PyTorch's default, every number drawn evenly from ±1/√32, so each forget gate starts between 0.412521 and 0.587479 at a zero reading; Gers et al. (2000) start it above one half, at 0.622459 and 0.731059 for their first two blocks; effect not measured |
| [A-35](A-35.md) | Which window a graph of Table 12 and Fig. 10 is | Numbers the graphs 1 to 191 under a column headed "Time"; never says which cycle of the engine a graph is | The engine's *g*-th window, ending at cycle *g* + 29; under every placement one window apart the default network's four largest values still change more often than Table 12's (8 to 28 joins against 1) |
| [A-36](A-36.md) | A GRU in place of each sensor's LSTM | Has no such stage; its "GRU [55]" row is another published method | An optional stage at the LSTM stage's size, PyTorch's cell; on FD001 against the LSTM stage −0.416 test RMSE (p = 0.455) and −0.999 held-back (p = 0.0047), neither clearing the bar |
| [A-37](A-37.md) | The bar when several comparisons are read together | Tests no difference: one number per cell, no seeds, no spread | p below 0.05 divided by the number of comparisons; among the seventeen of the FD001 sensitivity study it sets aside one test and two held-back differences |
| [A-38](A-38.md) | Ordinary models' predictions outside 0 to the label ceiling | Never says how its compared methods' numbers were obtained or bounded | Clipped to 0 to 125 before scoring; lowers linear regression on the 42 window summaries by 1.295 cycles of RMSE, every other model by at most 0.144; no comparison with the network changes side |
| [A-39](A-39.md) | ASUE when no engine is predicted early | Equation 24 divides by `N_under`, the number predicted early, and never says what the measure is when that is 0 | "Not a number", with a warning, and the count of early engines written beside every ASUE; changes no stored figure (fewest early engines in any of 464 stored prediction files: 25 of 100) |
| [A-40](A-40.md) | Regularisation: anything added to keep a network from memorising its training engines | Section 3.4 and Table 6 list the loss, optimiser, rates, passes, batch size and window, and no weight decay, dropout, clipping or early stopping | None of the first three; the held-back engines and early stopping (A-06, A-18) are the only guard. Over 200 passes the default's held-back error rises 0.411 ± 0.398 cycles after its best pass, the recurrent stage's 3.574 ± 0.721 |
| [A-41](A-41.md) | Graphs with the paper's 41 connections and none of its knowledge | Never describes drawing a graph at random or rewiring one; its "w/o Physics prior" row never says what replaced Table 3 | 41 pairs picked at random, or Table 3 rewired by 4,000 proposed swaps keeping every sensor's number of connections; one graph per seed, redrawn until in one piece (0.26 % of random draws are not); **not yet trained** |
| [F-22](../findings/F-22.md) | The networks beside ordinary models with no sensor graph | Compares its network with eight published methods, each row citing a source (Tables 7 to 10); never says how their numbers were obtained (A-38) | Measured on FD001 at `6733d77`, beside both networks with their best held-back pass restored and trained the paper's way: given the default's fourteen latest readings, gradient boosting, a random forest and a network of 64 trail it on four or five seeds of five by 0.12 to 1.57 cycles, no interval over the test engines excluding zero; given each window's latest reading, average and slope, untuned gradient boosting beats the default by 4.36 and 4.35 (p = 0.0001 and < 0.0001) and the recurrent-stage network on every seed, by 1.09 with its best pass restored (p = 0.0048) and 3.58 trained the paper's way (p = 0.0012, its interval over engines clear of zero), neither under the bar 0.05 / 58; given the latest reading and slope without the average, a row added after the first results were seen, by 3.91 trained the paper's way (p = 0.00085, just under that bar; interval over the test engines +1.93 to +5.95) |

The table lists the assumptions, the choices this project made. Findings,
facts established by measurement or by checking the paper, are listed in
[their own folder](../findings/README.md), with one exception kept on
purpose: the comparison with ordinary models (finding F-22) has a row here,
because it answers the question a reader brings to every result in this table,
whether a model with no sensor graph does as well. Its last column is a
measurement, not a choice. (Corrected 2026-10-03: the table had no row for
A-22, the rule by which the settings the paper leaves open were chosen; the
test that every entry has a row found the gap when the log became a folder.)

## References

Full citations for every work mentioned in these entries are in
[`REFERENCES.md`](../REFERENCES.md).
