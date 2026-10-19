# Audit of 2026-10-19: every statement on the documentation pages against its source

## What this file is

A record of giving every statement on the website's documentation pages a
source, and of checking each source against the copy this project holds. It
covers the twelve pages under Learn (`docs/learn/`) and the five under
Resources (`docs/resources/`), and the two READMEs that repeated one of their
definitions.

It was asked for by the author on 2026-10-18:

> "all the things you write should have a source. Especially the ones that are
> from a paper. These can be shown the same way the sources in papers are
> indicated. Make sure all the docs pages are following this rule. For instance
> when you are saying what a cycle is exactly in this dataset, you should
> source it."

The same message asked for the Data page to open with the data it uses,
NASA's C-MAPSS, to say what one cycle is, and to say why it defines remaining
useful life. Those three changes are in the page's first four sections.

## How the pages now show their sources

The way a journal paper does. A claim taken from a published work carries an
author–date citation with the most specific place a reader can find it, such
as (Saxena et al., 2008, p. 5) or Kipf and Welling (2017, Eq. 2), and the page
ends with a list headed "References" that holds the reading list's entry for
each work it cites, with a link to the copy. A choice this project made is not
credited to a paper: the page says the choice in words and points to its entry
in the log, as in "(assumption A-07)". A measurement is computed on the page
or points to the finding that records it. Where a sentence joins a source to
this project's own reading of it, it says which part is which.
[`../REFERENCES.md`](../REFERENCES.md) states the rule, under "How the taught
pages cite these works".

After the work, the 17 pages carry 95 reference-list entries between them,
naming 29 different works. Wen et al. (2026), the paper re-implemented, is
cited on all 17. Only works already in the reading list were cited; none was
added.

## How it was checked

1. **Citing.** One page at a time, each claim was matched to a held copy and
   the citation written with its locator. Every locator was checked in the
   copy: in a text extraction of each PDF, page by page, and, where a claim
   rests on a drawing, in the page rendered at high resolution.
2. **Two readers per page.** Two independent readers then opened every cited
   copy at every locator and reported what did not match: a quotation not word
   for word, a page that says something else, a pointer to the wrong log entry,
   a sentence whose meaning the citing had changed.
3. **Fixing.** Each reported problem was checked again before it was applied,
   because readers can be wrong; several were rejected with a reason.
4. **One read across all pages.** A final reader compared the pages with each
   other and with the READMEs, looking for a fact stated two ways, a pointer
   that leads to the wrong entry, a pronoun or bare page number that now points
   at the wrong work, and text the website renders wrongly. Its 28 findings
   were applied the same way, and the changes they made were read once more
   against the copies.
5. **Automatic checks.** A new test, `tests/test_citations.py`, checks on every
   page that a list headed "References" ends it, that every work it lists is in
   the reading list and cited on the page, that every work cited on the page is
   listed, and that every copy link names a held file. Whether a citation
   supports its sentence cannot be tested; that was steps 1 to 4.

## What was wrong, and how it was found

Most of the work was adding sources to sentences that were true. These are the
places where checking a source showed that a sentence, or an earlier citation,
was wrong. Each is corrected on its page.

| Where | What it said | What the source says | How it was found |
|---|---|---|---|
| `README.md`, glossary; `data/README.md` | A flight cycle is "one complete take-off, flight, and landing". | NASA's readme says each row is "a snapshot of data taken during a single operational cycle"; Saxena et al. (2008, p. 5) take "one measurement snapshot per flight". Neither describes take-off and landing. Both READMEs now quote the two and call "one cycle is one flight" this project's reading. | The final read across pages: the Data page already said no source defines a cycle in those words. One held paper does define "flight cycle data" as one mission "from take-off to landing" (Li & Dong, 2026, p. 402), but for NASA's later N-CMAPSS data, not these files, so it is not cited for them. |
| Start here | Saxena et al. (2008) say reliable estimates promise cost savings by "using equipment for longer". | "by increasing equipment usage" (p. 2). | A reader comparing the quotation with the copy. |
| Data page | Frederick et al. (2007, p. 7) cited for the fan, the bypass and the combustor. | PDF page 7 is the list of symbols, which shows none of them; the engine drawing is Fig. 1.1 on PDF page 9. | A reader opening the locator. |
| C-MAPSS page | The figure was read as marking the outer shaft N2 and the inner N1. | In Saxena et al.'s Fig. 1 (p. 2) and the guide's Fig. 1.1, the arrow from N1 ends on the outer shaft (compressor and turbine of the core) and the arrow from N2 on the inner one, the reverse of the guide's list of symbols, which names N1 the fan speed and N2 the core speed (p. 7). The page now says both and follows the list of symbols. | Both figures rendered at 1152 and 1200 dots per inch and the shafts followed by eye. |
| C-MAPSS page | "one snapshot per flight" cited to Section V.C (p. 6). | Section V.C says "at cruise snapshots"; the per-flight wording is in Section V on p. 5. | The final read across pages. |
| Start here | The paper "makes no statement about code at all". | It mentions its pseudocode (p. 4), its software environment and a Software contribution role. It never says whether its code is available, which is what the page now says. | The final read across pages. |
| Worked examples | Every addition "includes its learned offset". | Only the fully connected layers have an offset (Wen et al., 2026, Eqs. 12–13); the graph layers' equations print none (Eqs. 8–10), and the project follows them (assumption A-12). The page now says why their tables still show an offset of 0. | A reader comparing the equations, then the final read comparing the page with its figure. |
| Pipeline page | Cites the paper's Fig. 1 for the order of its steps. | Fig. 1 draws smoothing before capping the labels; the page's steps run the other way. The page now says so, and why it makes no difference: each step changes only its own columns. | A reader rendering Fig. 1. |
| Pipeline page | Wen et al. (2026, Table 2) cited for `T24` being the file's `sensor_2` column. | Table 2 numbers the sensors; nothing in it, or in NASA's readme, says which file column holds which. That is this project's reading, checked on the C-MAPSS page. | The final read across pages. |
| GCN page | A bare "Eq. 16" after a citation of He et al. (2015). | The equation meant is Glorot and Bengio's (2010, Eq. 16); He et al. have their own Eq. 16 (p. 1029). | A reader following the number. |

## What no source we hold supports

These sentences stay on the pages because they teach something the page needs,
but no held copy says them. Each needs a decision: accept it as the page's own
explanation, find a source and add it to the reading list, or reword it. They
fall into four groups.

### A. Plain-word explanations of standard terms (39)

The repository's own rule is that a borrowed word is explained where it is first used, so the pages gloss terms such as a spool, the Rankine scale, a paired t-test or a softmax. These are textbook definitions, but no held copy states them, and the papers that use the terms do not define them. Accepting them as the page's own explanations is the usual practice in a thesis; the alternative is a textbook added to the reading list.

| Page | Sentence | What is missing |
|---|---|---|
| `learn/01_the_problem.md` | A model learns by being shown inputs together with the right answer, and the files contain no remaining-life column for the training engines (NASA PCoE, 2008). | The first clause is a general statement of supervised learning, and no held source states it. The page treats it as its own definition. The second clause is sourced. |
| `learn/01_the_problem.md` | A model learns from examples paired with known answers, called **labels**. | The same general point, at the start of 'Calculate training labels'. No held source defines learning from inputs paired with answers. It reads as the page introducing the word 'labels', but it is not marked as the page's own definition. |
| `learn/01_the_problem.md` | Correlation describes co-variation, not a causal connection. | A textbook statistical caution that no held source states. The only listed source that might, the SHAP documentation page on causal insights, is not held. |
| `learn/03_explainability.md` | Training changes the model's multipliers and offsets using examples with known answers. | A description of training in general, with no citation; it reads as a definition the page introduces. Wen et al. (2026, Section 3.4) describe training on the error between predicted and actual values, but have not been checked at that locator for this wording. |
| `learn/04_the_pipeline.md` | This is called **z-score rescaling**. | The standard name for subtracting the mean and dividing by the standard deviation, with no held source cited. The only held paper that uses the name (Wang et al., 2024, DVGTformer) prints a different formula, so citing it would attach the wrong formula. |
| `learn/06_reading_results.md` | A paired *t*-test turns $t$ into *p*: the probability, if the setting made no difference at all, of a $t$ at least that far from zero. | This is a textbook definition of the paired t-test and of a p-value. No work in docs/REFERENCES.md covers the t-test: a grep of the held texts for 't-test', 't test' and 'Student' finds no definition. Dunn, Holm and Benjamini–Hochberg concern multiple comparisons and are not held. A statistics reference would need adding to docs/REFERENCES.md. |
| `learn/06_reading_results.md` | Dividing the standard deviation by √5 gives the uncertainty of an average of five numbers, so $t$ counts how many of those uncertainties the mean lies from zero. | This is the standard error of a mean, a standard statistical fact, and no held source covers it. It could be treated as a definition the page introduces, but an examiner could ask for a reference. |
| `learn/09_temporal.md` | This gives the change per cycle of the straight line whose squared gaps from the readings are smallest. | No held source covers least squares. The page's own code checks only that the arithmetic matches WindowSummary, not that the line is the least-squares fit. This is a standard mathematical fact, so it may fall under the convention's 'arithmetic' category. The author can decide whether it needs a textbook citation. |
| `resources/cmapss.md` | The fan, the low-pressure compressor (LPC) and the high-pressure compressor (HPC) squeeze the air; fuel burns in the burner, which Figure 1 calls the combustor (Saxena et al., 2008, Figs. 1–2, p. 2); the high-pressure and low-pressure turbines (HPT and LPT) take energy out of the hot gas. | The module names (guide p. 7) and the burner/combustor identity (Saxena Figs. 1–2) are supported. No held copy says what each module does to the air: 'squeeze the air' and 'take energy out of the hot gas' are unsourced. |
| `resources/cmapss.md` | A **spool** is one shaft with compressor and turbine stages fixed to it, so that the turbine turns the compressor. | The guide's nomenclature only uses 'spool' (N1 fan spool speed, N2 core spool speed) and never defines it. A grep of all held extractions for 'spool' finds only those two lines. |
| `resources/cmapss.md` | In standard engineering usage, which neither source spells out, a compressor's **pressure ratio** is its outlet pressure divided by its inlet pressure, and its **efficiency** is the share of the work put into it that goes into raising the pressure rather than into heating the air: a less efficient compressor heats the air more for the same rise in pressure. | The sentence is flagged as standard usage, but no held copy defines either term. |
| `resources/cmapss.md` | The glossary column headed 'In plain words, by standard usage': psia as 'pounds per square inch counted from vacuum', rpm as 'revolutions per minute', and the corrected speeds NRf and NRc as the physical speeds rescaled to a standard inlet temperature. | The names and units come from Saxena Table 2 (cited in the lead-in), and Wen p. 5 supports only 'derived from the physical speeds and the inlet temperature'. No held copy gives these definitions. |
| `resources/cmapss.md` | **°R**, degrees Rankine, are Fahrenheit-sized degrees counted from absolute zero, so a temperature in °F is the one in °R minus 459.67. | No held copy defines the Rankine scale or the 459.67 offset (grep finds neither). The T2 check against the guide's 59 °F (Table 1.4, p. 11) agrees with it but does not establish it. |
| `resources/cmapss.md` | Static pressure is what the moving air presses on a wall it flows along; total pressure is what the air would reach if it were brought to rest, so it also counts the part due to the air's speed. | Saxena Table 2 names P30 total and Ps30 static pressure at the HPC outlet (cited), but no held copy defines the two. |
| `resources/cmapss.md` | A **stall margin** is how far a compressor is operating from its stall limit (Saxena et al., 2008, Section V.D, p. 6), which by standard usage is the point where its airflow breaks down; the smaller the margin, the closer it is to stalling. | Only the first half is supported. No held copy says that stall is where a compressor's airflow breaks down. |
| `resources/cmapss.md` | FD001's inlet is at the standard-day temperature, as the T2 check showed, so the corrected fan speed nearly equals the physical one, 2388.02 against 2388.06, as the standard definition of a corrected speed, from the physical speed and the inlet temperature (Wen et al., 2026, p. 5), leads one to expect. | Wen p. 5 supports only that the corrected speed is derived from the physical speed and the inlet temperature. That it equals the physical speed on a standard day comes from the standard definition, which no held source states. |
| `resources/cmapss.md` | written in MATLAB and Simulink (two numerical computing environments) ... Mach numbers (speed as a fraction of the speed of sound) ... / The guide's own menu labels FC01 as sea level, static (the engine is not moving), standard day ... / For the four sensors on the gas path (the air's route through the engine) ... | These are small glosses on the sources' terms. The sources use the terms (Saxena p. 2 'MATLAB and Simulink environment', 'Mach numbers'; guide Fig. 4.6 'static') but do not define them. |
| `resources/gcn.md` | **What an eigenvalue of $L$ measures.** For any list $x$, $$x^\top L x = \sum_{(i,j)} \left(\frac{x_i}{\sqrt{d_i}} - \frac{x_j}{\sqrt{d_j}}\right)^2$$ where the sum runs over the connections, each counted once (41 here): ... | A general identity that neither Kipf and Welling nor any other held source states. The page's code checks it only for the 14 eigenvectors of this graph, not for every list, and the page does not derive it. |
| `resources/gcn.md` | The paper calls it the graph Fourier transform of $x$, a name from signal processing, where the Fourier transform splits a sound into components that change slowly or quickly; on a graph the eigenvectors of $L$, ordered by eigenvalue from smooth to sharply changing, play that part. | Only the name "graph Fourier transform" is in Kipf and Welling (2017, Section 2.1, p. 2). The signal-processing analogy has no held source. The smooth-to-sharp ordering is shown only on this graph, by the page's own table. |
| `resources/gcn.md` | A **symmetric** matrix, one whose entry $(i, j)$ equals its entry $(j, i)$ as $L$'s does, has $N$ eigenvectors at right angles to one another. | Standard linear algebra (the spectral theorem) that no held source states. Low stakes. |
| `resources/gcn.md` | ... the reason, supplied here, is that Chebyshev polynomials stay between −1 and 1 for inputs between −1 and 1 and grow fast outside that interval | The page marks this reason as its own ("supplied here"), but the mathematical fact has no held source. It is shown only for T_2 at two points. Hammond et al. (2011), where the paper sends the reader, is not held. Listed so the author can decide. |
| `resources/gcn.md` | Within a run, the randomness in training comes from **dropout** (Kipf & Welling, 2017, Section 3.1), which sets a random share of the values to zero during training; ... | The citation now covers only the claim that dropout is where the randomness comes from. The description of what dropout does has no held source. Kipf and Welling cite Srivastava et al. (2014) for it, which is neither held nor in docs/REFERENCES.md. |
| `resources/gcn.md` | They use 5-fold cross-validation, in which the labelled nodes are split into five parts and each part is held out once for testing while the model trains on the other four, with all labels available (unlike the main experiments), ... | Kipf and Welling (2017, Appendix B, p. 14) name "a 5-fold cross-validation experiment ... using all labels" but do not define it. The definition is the page's own, with no held source. |
| `resources/gcn.md` | The penalty is **L2 regularisation**: a multiple of the sum of the squared multipliers added to the loss, which pulls them towards zero. | Kipf and Welling (Section 5.2) name the "L2 regularization factor for the first GCN layer" but do not define it. No held source gives the definition. |
| `resources/gcn.md` | On 10 random splits of the same size the paper also reports a mean and its standard error, the uncertainty of a mean over 10 runs: for example 80.1 ± 0.5 % on Cora. | That the paper reports a mean and its standard error is in Kipf and Welling, p. 7. The gloss "the uncertainty of a mean over 10 runs" is the page's own definition, with no held source. |
| `resources/gcn.md` | Gradient descent, which moves each multiplier a small step against its gradient, then adjusts $W^{(0)}$ and $W^{(1)}$ to lower $\mathcal{L}$ (Kipf & Welling, 2017, Section 3.1). | Section 3.1 says only that the weights "are trained using gradient descent". The description of what gradient descent does is the page's own, with no held source. Low stakes. |
| `resources/gnn_lrp.md` | The graph of each sentence is its dependency tree, which joins each word to the word it grammatically depends on (Appendix F.2). | Appendix F.2 (PDF p. 19) supports only 'the dependency tree from the spaCy model'. No held source gives the gloss on what a dependency tree joins. It is a standard linguistic definition, kept so the term is explained. |
| `resources/gnn_lrp.md` | ... and the dipole moment (a measure of how a molecule's electric charge is separated) after the paper replaces that output by a linear one (Section 5.2). | The bracketed gloss of 'dipole moment' is a physics definition that no held source gives. Schnake et al. Section 5.2 uses the term without defining it. |
| `resources/gnn_lrp.md` | ... the difference between the positive and the negative class scores before a softmax (the step that turns class scores into probabilities) is applied ... | The bracketed gloss of 'softmax' is a standard definition. No held source states it: Schnake et al. Fig. 7 and Appendix F.2 support only the use of the logit difference. |
| `resources/lstm.md` | The code computes the products twice. Once from the formula, and once with PyTorch's **automatic differentiation**, which records every arithmetic operation as it runs and then applies the chain rule backwards through all of them to get exact derivatives. | The page's code shows automatic differentiation agreeing with the formula, but no held source describes how PyTorch records operations, and the description has not been checked against PyTorch's documentation. It could cite (PyTorch, n.d.) if the reading list's software entry is accepted for facts from documentation. |
| `resources/shap.md` | Results from cooperative game theory, the mathematics of dividing a jointly produced payout among the players who produced it, apply to that whole class and guarantee a unique solution. | No held source gives the gloss of cooperative game theory. Lundberg and Lee (pp. 2, 4) use the term without defining it, and Shapley (1953) is not held. The rest of the sentence is supported (Lundberg & Lee, 2017, p. 2). |
| `resources/shap.md` | In their standard meaning, °R (`degR` in this project's code) is degrees Rankine, an absolute temperature scale whose degree is the size of a Fahrenheit degree, which the C-MAPSS page checks against the user's guide, and psia is pounds per square inch absolute. | The page now says these are standard meanings, not the sources' words, but no held source defines Rankine or spells out psia. Saxena Table 2 prints only the symbols. The C-MAPSS page's check (T2 = 518.67 °R minus 459.67 gives the 59 °F standard day of Frederick Table 1.4) fits the Rankine reading but does not define it. Nothing checks psia. |
| `resources/shap.md` | (A lasso is a regression whose penalty on the sizes of the coefficients sets small ones to exactly zero; the paper does not say how it removes the penalty's bias.) | No held source defines the lasso. Lundberg only names a "debiased lasso" (Fig. 3, p. 8), and Kapoor's preprint only mentions a Lasso model whose coefficients all came out zero. The second half is a correct reading of the paper. |
| `resources/shap.md` | Figure 4 of Lundberg and Lee (2017, p. 9) reports a study with two small models, run on Amazon Mechanical Turk (Section 5.2, p. 8), a paid online service on which people complete small tasks. | The citation covers only the name. No held source describes Mechanical Turk. |
| `resources/shap.md` | Each method estimates the value of one feature in two decision-tree models (models that predict by a sequence of yes/no threshold tests), ... | The figure caption (p. 8) supports the two decision-tree models, but no held source gives the parenthetical definition of a decision tree. |
| `resources/shap.md` | the same network as in Lundberg and Lee's reference [7] (p. 9): two convolution layers, which slide small learned filters across the image, then two fully connected layers (every input joined to every output) ... | p. 9 supports the network being the one of reference [7], and the layer counts. No held source gives the plain-word glosses of convolution and fully connected layers. |
| `resources/shap.md` | ... and the change in log-odds (the logarithm of a class's probability divided by one minus it) over 20 random images compares an original and an updated DeepLIFT, SHAP and LIME. | No held source gives the gloss, and Lundberg and Lee do not define log-odds. Caution: the experiment masks pixels to turn an 8 into a 3, following their reference [7] (Shrikumar et al., not held), so the quantity may be the log-odds between the two classes rather than one class against all others. The gloss may not match what Figure 5B plots, and no held copy can settle it. |
| `resources/shap.md` | ... a linear layer, max pooling (a layer that keeps the largest of a few neighbouring values), or a function of one input. | p. 7 names max pooling but does not define it, and no held source gives the gloss. |
| `resources/shap.md` | A footnote of Lundberg and Lee (2017, p. 6, footnote 2) points out that econometrics, the statistics of economic data, had already stated Shapley values in an equivalent way, ... | The footnote says only "proposed in econometrics [2]". No held source gives the gloss "the statistics of economic data". |

### B. Why information must not leak from the test engines (6)

The reading list's source for leakage is Kapoor and Narayanan's 2023 article, but only its 2022 preprint is held, and the two have not been compared. The preprint makes most of these points, and the right-hand column says where; the article itself may differ.

| Page | Sentence | What is missing |
|---|---|---|
| `learn/04_the_pipeline.md` | The order protects against **leakage**, which on this page means information from test engines, held-back engines or future cycles entering a calculation that should not use it. | This is now marked as the page's working definition, but leakage is a method concept, and the work REFERENCES.md lists for it (Kapoor & Narayanan, 2023) is held only as its unchecked 2022 preprint. The preprint supports the idea: p. 4 gives its definition of data leakage; p. 5 covers pre-processing on training and test together, temporal leakage and non-independent samples. The author may want to cite the article once a copy is held. |
| `learn/04_the_pipeline.md` | We split by engine, not by individual window. Neighbouring 30-cycle windows share 29 cycles; putting them on opposite sides would test the model on almost the same history it learned from. | This is the project's reasoning plus arithmetic, and it is not marked as such. The published support would be the preprint's "[L3.2] Nonindependence between train and test samples constitutes leakage" (p. 5), which has the same version problem as above. The split itself carries pointers later in the section (assumptions A-06 and A-25). |
| `learn/04_the_pipeline.md` | No row reads a later cycle or another engine; otherwise the current input could reveal information unavailable at that time. | The first clause describes what the code does. The second is leakage reasoning (temporal leakage, preprint [L3.1], p. 5), and no held, checked source supports it. |
| `learn/04_the_pipeline.md` | The held-back and test tables use the **same** fitted numbers. Recalculating them from those tables would let unseen engines influence preparation. | The first sentence describes the code. The second is leakage reasoning (pre-processing on training and test together, preprint [L1.2], p. 5). No held, checked source supports it, and it is not marked as the project's reasoning. |
| `learn/04_the_pipeline.md` | Test answers remain for final evaluation. They must not replace validation answers for choosing settings or deciding when to stop. | This is a rule of research practice with no held, checked source (the same Kapoor preprint problem). It reads as general practice rather than as a project choice; the next sentence records the page's own exception (assumption A-22). |
| `resources/gcn.md` | Letting their readings in would be **leakage**: information about the cases being scored reaching the model before it is scored. | A definition of a method term with no citable held source. The held Kapoor and Narayanan copy is the 2022 preprint, which docs/REFERENCES.md says is not the cited 2023 article and has not been checked against it. It can be cited as (Kapoor & Narayanan, 2023) once the article is held and checked. |

### C. Sources the pages need but the project does not hold (2)

The GNN-LRP authors' published code is in the reading list but could not be downloaded from this environment, and no work in the reading list introduces the peephole connections the LSTM page mentions.

| Page | Sentence | What is missing |
|---|---|---|
| `learn/11_gnn_lrp.md` | The authors' published code (Schnake and colleagues, n.d.) starts at 3 and falls evenly to 0, as this project read it in the function `explain` of its file `xaigraph-synth-code/explainers.py` (assumption A-29). | The only source is the authors' code repository, which is listed in docs/REFERENCES.md but not held. It refused connections from this environment (403), so neither the linspace(3, 0, depth) schedule nor the file and function could be checked. The held papers do not state this schedule: grepping schnake_2022 (arXiv v3) and schnake_2020 (arXiv v1) finds only γ = 3 for the sentiment model (Fig. 7) and γ = 0.25 in v1. The sentence is marked as this project's reading. The author may want to keep a copy of that file, or record the commit in A-29. |
| `resources/lstm.md` | LSTM variants that do give the gates the cell state exist under the name of **peephole connections**. | No held source mentions peephole connections: there is no match in any text copy, and nothing in Gers et al.'s closing pages. REFERENCES.md has no entry for the paper that introduced them. The page admits this in the next sentence. |

### D. The project's own reasoning, not yet marked as such (11)

Each of these follows from the page's own argument or from a cited source by a step the source does not take. Most need only a few words saying the reasoning is the page's; two describe this project's own options (Mean and Full window), which the paper does not have and no log entry records.

| Page | Sentence | What is missing |
|---|---|---|
| `learn/00_start_here.md` | It would be better to estimate, for each engine, how many more flights it will survive, and service it just before that point. | This is a recommendation, and the Saxena sentences that follow (p. 2, cost savings and safety; p. 7, early estimates preferred) support it in general. No held source says to service "just before that point", and "flights" depends on the project's cycle-equals-flight reading, which is marked only two paragraphs later. Borderline. It is not marked as the project's reasoning, so it is listed for the author to decide. |
| `learn/03_explainability.md` | A remaining-life prediction tells us what the model expects, but not which readings led to it. | The page's opening framing. No held source on this page says it, and the page does not mark it as its own reasoning. It may count as a definition the page introduces. |
| `learn/03_explainability.md` | Applying the sigmoid to each share and adding would not give the same prediction, because that conversion is not an addition-preserving operation. | An arithmetic fact, not shown with numbers on the page and not marked as the project's reasoning. A-29 gives the same reasoning as the project's own ("The sigmoid bends ..."), and Schnake never discusses a sigmoid. A pointer to A-29 or a one-line worked example would settle it. |
| `learn/03_explainability.md` | A correct sum checks the calculation, not the quality of the model or the physical truth of an explanation. | The project's own reasoning, not marked as such. It follows from local accuracy holding for any model (Lundberg & Lee, 2017, Eq. 5) and from conservation in GNN-LRP, but neither source says it in these words. |
| `learn/07_evidence.md` | Everything on this site is one of three kinds of statement, and the value of a re-implementation study depends entirely on a reader being able to tell which is which. | This is the project's own position, from CLAUDE.md rule 5 and its preamble, but the page does not mark it as such. "Depends entirely" is a judgement, not something a held source states. It probably needs no citation; the author may want to mark it as the project's stance or leave it as framing. |
| `learn/08_worked_examples.md` | Mean instead averages a sensor's values in the window. | This describes this repository's own option (models/temporal MeanWindow, "Average each sensor's window into one number"), so the code supports it and it needs no citation. But nothing on the page says the option is this project's and not the paper's, and no assumption entry covers it. The neighbouring options are marked: Window summary (A-28), the GRU (A-36) and Last cycle (A-07). The author may want to mark it the same way. |
| `learn/08_worked_examples.md` | Full window keeps all the values. | Same situation as Mean. It describes this repository's RawWindow option ("Hand the graph layer every cycle"), which the paper does not have. It is not marked as this project's option, and no assumption entry covers it. |
| `learn/10_dynamic_shap.md` | Giving each sensor all of the change caused by removing it can therefore count the same interaction more than once. | No held source says this. Lundberg & Lee (2017, p. 3), cited on the sentence before, say only that "the effect of withholding a feature depends on other features in the model". The sentence is the page's own reasoning, introduced with "therefore", but not labelled as the project's reading. |
| `learn/11_gnn_lrp.md` | When each sensor hands the graph one number, as in the default network, every number reaching one first-layer output passes through the same multiplier, so gamma's extra weight scales them all alike and the rescaling undoes it: the replacement changes nothing there. It matters when each sensor hands the graph several numbers. | Borderline. No held source says this. The cancellation follows by arithmetic from Schnake et al. (2022) Table 1, Eq. 9: one input neuron per node means the shared (1 + γ) factor appears in every numerator and the denominator. "It matters when ... several numbers" is a project measurement recorded in A-29: a median change of 106 ± 42 % for the window-summary network. Neither sentence carries its own pointer; the nearest is "(assumption A-29)" at the end of the paragraph. The author may want "(assumption A-29)" after "changes nothing there". |
| `resources/cmapss.md` | No two real engines leave the factory identical, and an engine in service is already slightly worn. | A borderline case. Saxena Section V.A (p. 5, cited at the end of the next sentence) says 'Initial wear can occur due to manufacturing inefficiencies and are commonly observed in real systems'. That is close, but it does not say that no two engines are identical, and this sentence carries no citation of its own. |
| `resources/gnn_lrp.md` | **γ** (gamma), at least 0, is a setting of the explanation rule. | Montavon et al. (2019, Section 10.2.1) say only that γ 'controls by how much positive contributions are favored' and that 'as γ increases' negatives fade. Schnake et al. (2022, Table 1) call it 'a hyperparameter of the propagation rule'. Neither states γ ≥ 0, although every setting either uses is 0 or more. |

## What the automatic check cannot see

`tests/test_citations.py` compares each page's citations with its own list
and with the reading list. Three things are outside its reach:

- **Whether a citation supports its sentence.** Only reading the copy at the
  locator can tell, and steps 1 to 4 above are the record of that reading as
  of this date. A sentence edited later is unchecked until someone reads it
  against its source again.
- **A sentence that needs a source and has none.** The test cannot tell a
  claim from an explanation; the four groups above are what readers found.
- **Two works with the same first author and year.** The test identifies a
  work by its first author's surname and its year, so the reading list's two
  papers by Wang et al. (2024) share one key, and a page citing one while
  listing the other would pass. No page cites either at present.

## What was not part of this check

The notebooks, `docs/assumptions/`, `docs/findings/`, the folder READMEs other
than the data folder's one sentence, and `README.md` apart from its glossary
entry on remaining useful life and its count of the Learn pages. The
glossary in `README.md` cites works by author and year, but not every entry
has a source (the one on predictive maintenance has none), and most carry no
page or section.

## What is left

The sentences in the four groups above, each needing the author's decision;
the works cited but not held, which were already written up before this audit;
and the Mean and Full window options for the step before the graph layers,
which this project added, the worked examples page describes, and no log entry
records. All three are in [`../PENDING.md`](../PENDING.md).
