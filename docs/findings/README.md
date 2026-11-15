# Findings

These are not choices made by this project; the choices are in
[`../assumptions/`](../assumptions/README.md). Some come from running the pipeline
on the real FD001 files, others from checking the paper against itself or
against the papers around it, and one (F-09) from this project's own
practice; each entry's opening says which. They belong in the thesis. Each is in a file of its own named after its code: finding F-16 is
[`F-16.md`](F-16.md).

| ID | Finding | Logged after |
|----|---------|--------------|
| [F-01](F-01.md) | P15 is in the graph but carries almost no information | [A-10](../assumptions/A-10.md) |
| [F-02](F-02.md) | Eleven percent of test engines are affected by the label ceiling | [A-10](../assumptions/A-10.md) |
| [F-03](F-03.md) | The shortest test history is exactly 31 cycles | [A-10](../assumptions/A-10.md) |
| [F-04](F-04.md) | What the physics graph is worth has now been measured three times, with three answers | [A-19](../assumptions/A-19.md) |
| [F-05](F-05.md) | What SHAP says about P15, and what it does not | [A-20](../assumptions/A-20.md) |
| [F-06](F-06.md) | Where this code lands against the paper's FD001 figures | [A-20](../assumptions/A-20.md) |
| [F-07](F-07.md) | The paper's construction rules require an edge its matrix does not contain | [A-22](../assumptions/A-22.md) |
| [F-08](F-08.md) | The paper's Table 12 cannot hold signed Shapley values | [A-22](../assumptions/A-22.md) |
| [F-09](F-09.md) | A results file survived in the repository that no committed code produced | [A-22](../assumptions/A-22.md) |
| [F-10](F-10.md) | The input scaling the paper never describes accounts for most of the gap | [A-22](../assumptions/A-22.md) |
| [F-11](F-11.md) | The Conclusions claim a "highest R²" the paper's own Table 9 refutes | [A-22](../assumptions/A-22.md) |
| [F-12](F-12.md) | No other paper in the surrounding literature gives a graph node one number | [A-22](../assumptions/A-22.md) |
| [F-13](F-13.md) | Fig. 1 lists node features the paper's own feature selection removes | [A-25](../assumptions/A-25.md) |
| [F-14](F-14.md) | The text under Equations 9 and 10 misnumbers the weight matrices | [A-25](../assumptions/A-25.md) |
| [F-15](F-15.md) | Fig. 9 and Table 12 disagree about which sensors matter most | [A-25](../assumptions/A-25.md) |
| [F-16](F-16.md) | What the paper's LSTM is worth depends on a training procedure the paper does not describe | [A-25](../assumptions/A-25.md) |
| [F-17](F-17.md) | Fig. 10 is drawn from Table 12, except in two of its fifteen graphs | [A-25](../assumptions/A-25.md) |
| [F-18](F-18.md) | The paper's graph cannot tell three groups of sensors apart, and training never separates their weights | [A-30](../assumptions/A-30.md) |
| [F-19](F-19.md) | On FD001, the temporal stage dominates the measured one-factor changes | [A-32](../assumptions/A-32.md) |
| [F-20](F-20.md) | The dataset paper's scoring equation prints its two constants the other way round from its own text and figure | [A-33](../assumptions/A-33.md) |
| [F-21](F-21.md) | As printed, the paper's graph layers can only scale with their input, and after the first one each sensor's 32 numbers come from one | [A-34](../assumptions/A-34.md) |
| [F-22](F-22.md) | Flexible ordinary models given the default network's fourteen numbers trail it on four or five seeds of five, by less than the test engines can resolve; given each window's latest reading and slope, an untuned gradient booster beats the default clearly, and the recurrent-stage network on every seed, by enough to clear the bar for the whole entry only when that network is trained the paper's way, and then only just, on a row chosen after the first results were seen | [A-40](../assumptions/A-40.md) |
| [F-23](F-23.md) | The paper reads its correlation heatmap as physical coupling, but the sizes in it are set by how much scatter each sensor carries | — |

**Logged after** names the assumption in whose section each finding sat until
2026-10-03, when the assumptions and the findings were one file,
`docs/ASSUMPTIONS.md`. F-01 to F-03 sat instead after A-10, under a heading of
their own, "Findings about the source paper, established from the data". That
day each entry became a file of its own under the same code, its text
unchanged apart from the four things listed in
[the assumptions folder's README](../assumptions/README.md#why-this-folder-exists).
A finding added after that day sat in no assumption's section, and its row has
a dash in that column.

## References

Full citations for every work mentioned in these entries are in
[`REFERENCES.md`](../REFERENCES.md).
