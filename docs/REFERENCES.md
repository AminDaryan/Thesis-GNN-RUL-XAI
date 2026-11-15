# References

Every source this project cites, in the code, the notebooks, the taught pages
or the assumptions log, with a note on what it is used for. Citations follow
APA style.

**Where the papers are.** Each entry ends with a line saying where its copy
is. "Copy:" links the PDF kept in [`docs/references/`](references/), which
opens in the browser's own PDF viewer; the words after it say which version
the copy is, because page and section numbers in this project refer to that
version and no other. The copies are kept on the author's computer and are
not committed, since many are publishers' files that may not be passed on.
`docs/references/manifest.json` records where each came from and a checksum
of it, and `python scripts/fetch_references.py` downloads again every copy
that is free to download, refusing any file that is not the version the
citations were checked against. "Not held:" marks a work this project cites
without having a copy, and says how to get one.

The pages under **Resources** in the site's menu explain the five works the
project is built on most directly — the dataset, the LSTM, the GCN, SHAP and
GNN-LRP — from the papers themselves.

**How the taught pages cite these works.** Every page under Learn and
Resources names its sources the way a journal paper does. A claim taken from a
published work carries an author–date citation in the text, such as
(Saxena et al., 2008, p. 5) or Kipf and Welling (2017, Eq. 2), with the most
specific place a reader can find: a page, section, table, equation or figure
of the copy listed here. The page ends with a list headed "References" that
holds this file's entry for every work the page cites, and only those, each
followed by a link to its copy. A choice this project made is not credited to
a paper: the page says the choice in words and points to its entry, as in
"(assumption A-07)". A measurement is either computed on the page or points
to the finding that records it, as in "(finding F-16)". Where a sentence joins
a source to this project's own reading of it, it says which part is which.
`tests/test_citations.py` checks that each page's citations and its list agree
with each other and with this file; whether a citation supports its sentence
can only be checked by reading the copy, and the record of that check is
[`audits/AUDIT_2026-10-19_docs_sources.md`](audits/AUDIT_2026-10-19_docs_sources.md).

---

## The paper being re-implemented

**Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026).** Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
*168*, 110875. https://doi.org/10.1016/j.ast.2025.110875

> The method this project re-implements. Received 5 July 2025, accepted
> 1 September 2025. The word "neutral" in the title is the publisher's own
> spelling, kept here because it is how the paper is indexed; the paper is
> about a graph *neural* network. The authors publish no source code, and the
> data availability statement reads "Data will be made available on request".
>
> Used for: the 14-sensor selection, the hand-designed adjacency matrix
> (their Table 3), the network layer sizes (their Table 4), and the training
> settings (their Table 6).
>
> Copy: [wen_2026_estgnn_dynamic_shap.pdf](references/wen_2026_estgnn_dynamic_shap.pdf),
> the publisher's version of record (16 pages). Elsevier subscription article:
> through a university library.

---

## The dataset

**Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).** Damage propagation
modeling for aircraft engine run-to-failure simulation. In *2008 International
Conference on Prognostics and Health Management* (pp. 1–9). IEEE.
https://doi.org/10.1109/PHM.2008.4711414

> The paper that introduced the C-MAPSS benchmark dataset and explains how the
> simulated engine degradation was produced. This is the correct citation for
> the data itself. Its scoring function, Eq. (11) on page 7, prints its two
> constants the opposite way round from the paper's own sentence above it and
> from its Figure 9 (finding F-20).
>
> Copy: [saxena_2008_damage_propagation_modeling.pdf](references/saxena_2008_damage_propagation_modeling.pdf),
> the authors' 9-page manuscript as NASA distributes it on DASHlink,
> byte-identical to the copy in the author's NASA download. NASA's Technical
> Reports Server also holds a file under report number 20090029214; that one is
> an **earlier draft** with a different title ("Damage Propagation Modeling for
> Aircraft Engine Prognostics") and different numbering — its scoring function
> is Eq. (9) — so page and equation numbers in this project do not refer to it.

**NASA Prognostics Center of Excellence. (2008).** readme.txt, the description
of the C-MAPSS data set distributed inside its zip file. NASA Prognostics Data
Repository.

> Lists the four subsets, their engine counts, operating conditions and fault
> modes, and the 26 columns of every data file.
>
> Copy: [nasa_2008_cmapss_readme.txt](references/nasa_2008_cmapss_readme.txt),
> taken from NASA's zip file; `data/raw/README.md` gives the download address.

**NASA Prognostics Center of Excellence Data Set Repository.**
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

> The official distribution point for the C-MAPSS files. A web page, so no copy
> is kept.

**Frederick, D. K., DeCastro, J. A., & Litt, J. S. (2007).** *User's guide for
the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)*
(NASA/TM-2007-215026). NASA Glenn Research Center.
https://ntrs.nasa.gov/citations/20070034949

> Checked for engine structure and variable definitions. The overlay follows
> the dataset paper's Table 2 for W32 because this guide's Table 1.3 labels
> both coolant flows as high-pressure turbine flows. This disagreement and
> the schematic placements are recorded in the engine-overlay choice
> (assumption A-31).
>
> Copy: [frederick_2007_cmapss_users_guide.pdf](references/frederick_2007_cmapss_users_guide.pdf),
> the NASA Technical Reports Server file (47 pages), free.

---

## Engine illustration and interaction

**NASA Glenn Research Center. (n.d.).** Engine Parameter Interactive
(EngineSimU). Retrieved September 27, 2026.
[Interactive and instructions](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/enginesimu/).
[Published source archive](https://github.com/nasa/BGA).

> A documented, existing example of selecting components on an engine
> schematic. Used as an interaction reference, not as evidence of C-MAPSS
> probe coordinates or the validity of GNN-LRP. No code or artwork was copied.
> A web page and a code archive, so no copy is kept.

---

## The label design

**Heimes, F. O. (2008).** Recurrent neural networks for remaining useful life
estimation. In *2008 International Conference on Prognostics and Health
Management* (pp. 1–6). IEEE. https://doi.org/10.1109/PHM.2008.4711422

> Introduced the piecewise-linear Remaining Useful Life target: hold the label
> at a ceiling while the engine is healthy, then let it fall linearly. Reported
> that a ceiling between roughly 120 and 130 cycles works well. This is the
> origin of the value 125 used almost universally since.
>
> Not held: an IEEE paper with no free copy, through a university library. The
> two sentences above therefore come from papers that cite it, Jin et al.
> (2025, p. 7) among them, and have not been checked against Heimes himself.

---

## The network's building blocks

**Kipf, T. N., & Welling, M. (2017).** Semi-supervised classification with
graph convolutional networks. *International Conference on Learning
Representations (ICLR)*. https://arxiv.org/abs/1609.02907v4

> The graph convolutional network (GCN) layer, the kind of layer Wen et al. use
> and this project builds. Wen et al. do not cite this paper (their reference
> list has no entry for it). The source of the symmetric normalisation
> `D^(-1/2) (A + I) D^(-1/2)`
> implemented in `graph/adjacency.py` (introduced in the display just before
> their Eq. 8), of the propagation rule without a bias term (their Eq. 2), and
> of Glorot initialisation for this layer (their Section 5.2).
>
> Copy: [kipf_2017_graph_convolutional_networks.pdf](references/kipf_2017_graph_convolutional_networks.pdf),
> arXiv version 4, the ICLR 2017 paper, free.

**Glorot, X., & Bengio, Y. (2010).** Understanding the difficulty of training
deep feedforward neural networks. In Y. W. Teh & M. Titterington (Eds.),
*Proceedings of the Thirteenth International Conference on Artificial
Intelligence and Statistics* (Vol. 9, pp. 249–256). PMLR.
https://proceedings.mlr.press/v9/glorot10a.html

> The Glorot (or "Xavier", after Xavier Glorot) initialisation of the graph
> layers' multipliers (assumption A-14).
>
> Copy: [glorot_2010_understanding_deep_feedforward.pdf](references/glorot_2010_understanding_deep_feedforward.pdf),
> the PMLR proceedings paper, free.

**He, K., Zhang, X., Ren, S., & Sun, J. (2015).** Delving deep into rectifiers:
Surpassing human-level performance on ImageNet classification. In
*Proceedings of the IEEE International Conference on Computer Vision (ICCV)*
(pp. 1026–1034). https://doi.org/10.1109/ICCV.2015.123

> PyTorch names the default starting values of its fully connected layers,
> which this project keeps (assumption A-14), after this paper's first author:
> `kaiming_uniform_`. With the setting PyTorch passes, its own source says that
> default is the same as drawing every multiplier evenly from
> ±1/√(number of inputs), which is not the distribution the paper derives (a
> bell curve with spread √(2/n), page 1029). The paper is also the source for
> the caution that Glorot's derivation assumes activations that are linear, an
> assumption it calls invalid for the ReLU used here (page 1029).
>
> Copy: [he_2015_delving_deep_rectifiers.pdf](references/he_2015_delving_deep_rectifiers.pdf),
> the Computer Vision Foundation's open-access copy of the ICCV paper, free.

**Hochreiter, S., & Schmidhuber, J. (1997).** Long short-term memory.
*Neural Computation*, *9*(8), 1735–1780.
https://doi.org/10.1162/neco.1997.9.8.1735

> The Long Short-Term Memory network. Wen et al. place one of these before each
> sensor node. This project builds that stage but does not use it by default
> (assumption A-11).
>
> Copy: [hochreiter_1997_long_short_term_memory.pdf](references/hochreiter_1997_long_short_term_memory.pdf),
> the authors' 32-page manuscript from Hochreiter's institute, free. Its pages
> are numbered 1 to 32, not as in the journal, so it is cited here by section.

**Gers, F. A., Schmidhuber, J., & Cummins, F. (2000).** Learning to forget:
Continual prediction with LSTM. *Neural Computation*, *12*(10), 2451–2471.
https://doi.org/10.1162/089976600300015015

> Added the forget gate to the 1997 LSTM. The LSTM that PyTorch computes, and
> so the one this project runs, has it. Its Sections 3.1 and 4.2 start the
> forget gate's offset positive, so the gate begins above one half. Section
> 3.1 calls the starting gate "almost 1.0"; the offsets Section 4.2 uses,
> +0.5 for the first block of cells and +1 for the second, rising with each
> later block, give 0.622459 and 0.731059 at a zero input. This project keeps
> PyTorch's starting values instead, which put it near one half (assumption
> A-34).
>
> Copy: [gers_2000_learning_to_forget.pdf](references/gers_2000_learning_to_forget.pdf),
> the authors' accepted manuscript from IDSIA, free. Its text cannot be
> searched or copied (its fonts carry no text), so it is read from the page
> images.

**Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
Schwenk, H., & Bengio, Y. (2014).** Learning phrase representations using RNN
encoder–decoder for statistical machine translation. In A. Moschitti, B. Pang,
& W. Daelemans (Eds.), *Proceedings of the 2014 Conference on Empirical
Methods in Natural Language Processing (EMNLP)* (pp. 1724–1734). Association
for Computational Linguistics. https://doi.org/10.3115/v1/D14-1179

> The origin of the gated recurrent unit (GRU), built here as an alternative
> temporal stage that is not part of Wen et al.'s model (Section 2.3,
> Eqs. 5–8); Wen et al. list a GRU network only as one of the methods they
> compare against (their Table 7, reference [55]).
>
> Copy: [cho_2014_rnn_encoder_decoder.pdf](references/cho_2014_rnn_encoder_decoder.pdf),
> the ACL Anthology proceedings paper, free.

**Kingma, D. P., & Ba, J. (2015).** Adam: A method for stochastic
optimization. *3rd International Conference on Learning Representations
(ICLR)*. https://arxiv.org/abs/1412.6980v9

> The optimiser every network here is trained with; Wen et al.'s Table 6 names
> it.
>
> Copy: [kingma_2015_adam.pdf](references/kingma_2015_adam.pdf), arXiv version
> 9, free.

**PyTorch documentation. (n.d.).** torch.optim.Adam.
https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html

> The update rule as PyTorch computes it, cited by the website's README for how
> the learning rate scales a step. A web page, so no copy is kept.

**MacQueen, J. (1967).** Some methods for classification and analysis of
multivariate observations. In L. M. Le Cam & J. Neyman (Eds.), *Proceedings of
the Fifth Berkeley Symposium on Mathematical Statistics and Probability: Vol.
1. Statistics* (pp. 281–297). University of California Press.

> The paper that named k-means, used in notebook 01 to find operating
> conditions before scaling each one separately.
>
> Copy: [macqueen_1967_k_means.pdf](references/macqueen_1967_k_means.pdf), a
> scan of the published chapter from the UC Berkeley Library, free.

---

## Graph neural networks, background

**Corso, G., Stark, H., Jegelka, S., Jaakkola, T., & Barzilay, R. (2024).**
Graph neural networks. *Nature Reviews Methods Primers*, *4*, Article 17.
https://doi.org/10.1038/s43586-024-00294-7

> A clear, recent introduction to message passing on graphs: each node
> receiving numbers from its neighbours and combining them. Good background
> reading for the thesis chapter that introduces graph neural networks.
>
> Copy: [corso_2024_graph_neural_networks.pdf](references/corso_2024_graph_neural_networks.pdf),
> the publisher's PDF (13 pages). Subscription article: through a university
> library.

---

## Explainability

**Shapley, L. S. (1953).** A value for n-person games. In H. W. Kuhn &
A. W. Tucker (Eds.), *Contributions to the Theory of Games, Volume II*
(pp. 307–317). Princeton University Press.
https://doi.org/10.1515/9781400881970-018

> The origin of the Shapley value in cooperative game theory. *Cooperative game
> theory* is the mathematical study of how a group that achieves something
> together should divide the credit for it. The Shapley value is the unique way
> of dividing that credit which satisfies a small set of fairness conditions.
>
> Not held: a publisher's book chapter, through a university library. RAND
> published a 1952 preprint of it, paper P-295
> (https://www.rand.org/pubs/papers/P295.html), which can be read in a browser
> but may not be reposted; it is a different version, so its pages do not
> match. What this project says about Shapley's work comes through Lundberg
> and Lee below, who credit the uniqueness result they rely on to Young (1985),
> their reference [10] (page 4 of the copy held here), not to Shapley; the
> sentence above about uniqueness is therefore unchecked against Shapley's
> chapter.

**Lundberg, S. M., & Lee, S.-I. (2017).** A unified approach to interpreting
model predictions. In I. Guyon, U. von Luxburg, S. Bengio, H. Wallach,
R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), *Advances in Neural
Information Processing Systems 30 (NIPS 2017)* (pp. 4765–4774).
https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

> Introduced SHAP (SHapley Additive exPlanations), which applies the Shapley
> value to machine learning. The "players" become the input features and the
> "payout" becomes the model's prediction, so each feature receives a number
> saying how much it pushed the prediction up or down. This is the method Wen
> et al. apply after training, and the starting point for the explainability
> half of this thesis.
>
> Copy: [lundberg_2017_unified_shap.pdf](references/lundberg_2017_unified_shap.pdf),
> the proceedings paper (10 pages, numbered 1 to 10), free.

**Zeiler, M. D., & Fergus, R. (2014).** Visualizing and understanding
convolutional networks. In D. Fleet, T. Pajdla, B. Schiele, & T. Tuytelaars
(Eds.), *Computer Vision – ECCV 2014* (Lecture Notes in Computer Science,
Vol. 8689, pp. 818–833). Springer. https://doi.org/10.1007/978-3-319-10590-1_53

> Popularised occlusion — hiding part of the input and watching the output
> move — for images (their Section 4.2). Notebook xai/02 sets it beside
> Shapley values as the cheaper method.
>
> Copy: [zeiler_2014_visualizing_convnets.pdf](references/zeiler_2014_visualizing_convnets.pdf),
> the published chapter as posted on Fergus's own web page, free.

**Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., & Samek, W.
(2015).** On pixel-wise explanations for non-linear classifier decisions by
layer-wise relevance propagation. *PLOS ONE*, *10*(7), Article e0130140.
https://doi.org/10.1371/journal.pone.0130140

> The origin of layer-wise relevance propagation (LRP): a prediction is divided
> into shares that are handed backwards through the network, layer by layer,
> with the total kept the same. GNN-LRP extends it.
>
> Copy: [bach_2015_layerwise_relevance_propagation.pdf](references/bach_2015_layerwise_relevance_propagation.pdf),
> the open-access article (46 pages, 21 MB), free.

**Montavon, G., Binder, A., Lapuschkin, S., Samek, W., & Müller, K.-R.
(2019).** Layer-wise relevance propagation: An overview. In W. Samek,
G. Montavon, A. Vedaldi, L. K. Hansen, & K.-R. Müller (Eds.), *Explainable AI:
Interpreting, explaining and visualizing deep learning* (Lecture Notes in
Computer Science, Vol. 11700, pp. 193–209). Springer.
https://doi.org/10.1007/978-3-030-28954-6_10

> Defines the LRP-γ rule, the setting γ that GNN-LRP and `explain/walks.py`
> apply (their Section 10.2.1 and the table of rules in Appendix 10.A).
>
> Copy: [montavon_2019_lrp_overview.pdf](references/montavon_2019_lrp_overview.pdf),
> the authors' copy from Samek's Fraunhofer HHI page, free. Its printed pages
> run 195–212, not as in the book, so it is cited by section only.

**Schnake, T., Eberle, O., Lederer, J., Nakajima, S., Schütt, K. T., Müller,
K.-R., & Montavon, G. (2022).** Higher-order explanations of graph neural
networks via relevant walks. *IEEE Transactions on Pattern Analysis and
Machine Intelligence*, *44*(11), 7581–7596.
https://doi.org/10.1109/TPAMI.2021.3115452

> Introduced GNN-LRP, which divides a graph network's output into one share
> per walk: the list of nodes a contribution passes through, one per graph
> layer. This project uses it in `explain/walks.py` to ask how much of a
> prediction travelled through the sensor graph (assumption A-29). Section,
> table and appendix numbers cited in this project are those of arXiv
> version 3.
>
> Copy: [schnake_2022_gnn_lrp_relevant_walks.pdf](references/schnake_2022_gnn_lrp_relevant_walks.pdf),
> arXiv version 3 (https://arxiv.org/abs/2006.03589v3), free. The journal
> version is open access on IEEE Xplore but was not compared, so its numbering
> may differ.
>
> The first version, arXiv 2006.03589v1, titled *XAI for Graphs: Explaining
> Graph Neural Network Predictions by Identifying Relevant Walks*, is cited
> once, for the γ of 0.25 at every layer it used:
> [schnake_2020_gnn_lrp_arxiv_v1.pdf](references/schnake_2020_gnn_lrp_arxiv_v1.pdf), free.

**Schnake, T., and colleagues. (n.d.).** paper_gnn_lrp, the GNN-LRP authors'
published code. https://git.tu-berlin.de/thomas_schnake/paper_gnn_lrp

> Source of the default γ schedule this project adopts as the authors' own
> setting, and the reference the walk explanation was audited against
> (assumption A-29). A code repository, so no copy is kept.

**Pope, P. E., Kolouri, S., Rostami, M., Martin, C. E., & Hoffmann, H.
(2019).** Explainability methods for graph convolutional neural networks. In
*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR)* (pp. 10772–10781).
https://openaccess.thecvf.com/content_CVPR_2019/html/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.html

> The node-level explanation method that the GNN-LRP paper compares its walks
> with; this project names its "starting sensors" ranking after it
> (assumption A-33).
>
> Copy: [pope_2019_explainability_gcn.pdf](references/pope_2019_explainability_gcn.pdf),
> the Computer Vision Foundation's open-access copy, free. Its pages are
> numbered 10772–10781 as above; IEEE Xplore numbers the same paper
> 10764–10773 (https://doi.org/10.1109/CVPR.2019.01103).

**Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019).**
GNNExplainer: Generating explanations for graph neural networks. In
*Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*.
https://arxiv.org/abs/1903.03894v4

> Learns, for one prediction, a weight between 0 and 1 for every connection of
> the graph, so that a small set of connections keeps the prediction. Considered
> for this project and not used; assumption A-29 gives the reasons.
>
> Copy: [ying_2019_gnnexplainer.pdf](references/ying_2019_gnnexplainer.pdf),
> arXiv version 4, free. It is the version the project links, and the only one
> that carries the settings in its own Appendix C; the proceedings paper
> leaves them to a separate supplement.

**Luo, D., Cheng, W., Xu, D., Yu, W., Zong, B., Chen, H., & Zhang, X. (2020).**
Parameterized explainer for graph neural network. In *Advances in Neural
Information Processing Systems 33 (NeurIPS 2020)* (pp. 19620–19631).
https://arxiv.org/abs/2011.04573v1

> PGExplainer: trains a second, small network that produces those connection
> weights for any prediction. Considered and not used (assumption A-29).
>
> Copy: [luo_2020_pgexplainer.pdf](references/luo_2020_pgexplainer.pdf), arXiv
> version 1, free; its Appendix B holds the settings the project refers to.

**Holdijk, L., Boon, M., Henckens, S., & de Jong, L. (2021).** [Re]
Parameterized explainer for graph neural network. *ReScience C*, *7*(2), #7.
https://doi.org/10.5281/zenodo.4834242

> A replication of PGExplainer that could not reproduce its results from the
> paper alone and found them to depend strongly on its penalty weights and
> temperature. Part of the reason given in assumption A-29.
>
> Not held: free under CC BY 4.0 from
> https://zenodo.org/records/4834242/files/article.pdf, but Zenodo refused
> every download from this computer's network on 2026-10-02. Until a copy is
> obtained, the description above is unchecked against the article. The third
> author's name, Henckens, was spelt Henckels here until 2026-10-02.

**Yuan, H., Yu, H., Gui, S., & Ji, S. (2023).** Explainability in graph neural
networks: A taxonomic survey. *IEEE Transactions on Pattern Analysis and
Machine Intelligence*, *45*(5), 5782–5799.
https://doi.org/10.1109/TPAMI.2022.3204236

> A survey that also compares graph explanation methods on shared tests, the
> only comparison found that includes GNN-LRP and was not run by the authors
> of GNN-LRP, GNNExplainer or PGExplainer (its Sections 8.1 and 8.2). Three of
> its four authors wrote SubgraphX, another method it compares (its reference
> [48]). Its comparison covers classification only. Cited by the explanation
> folder's README for that comparison. This entry gave the year as 2022 with
> no volume until 2026-10-02: that was the early-access record. Until the same
> date it also called the survey the only independent comparison, which it is
> not where SubgraphX is concerned. Both were found when the cited papers were
> collected into `docs/references/` and checked against the places that cite
> them.
>
> Copy: [yuan_2023_gnn_explainability_survey.pdf](references/yuan_2023_gnn_explainability_survey.pdf),
> arXiv version 3 (https://arxiv.org/abs/2012.15445v3), free, the version the
> project read.

**SHAP documentation. (n.d.).** Be careful when interpreting predictive models
in search of causal insights.
https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html

> Cited by the website's README for the difference between explaining a
> prediction and establishing a physical cause. A web page, so no copy is kept.

**PyTorch Geometric. (n.d.).** The graph-network library's GNNExplainer and
PGExplainer implementations. https://pytorch-geometric.readthedocs.io/

> Cited in `explain/README.md` for how the common library adapts those two
> methods to a network that predicts a number. Not used by the project. Code
> and documentation, so no copy is kept.

---

## Research practice

**Kapoor, S., & Narayanan, A. (2023).** Leakage and the reproducibility crisis
in machine-learning-based science. *Patterns*, *4*(9), 100804.
https://doi.org/10.1016/j.patter.2023.100804

> A survey of how data leakage — information about the test data reaching the
> model during training — produces published results that cannot be
> reproduced. The justification for two rules enforced throughout this
> pipeline: all statistics are learned from training data only, and the
> validation split is made by engine rather than by window.
>
> Copy: [kapoor_2022_leakage_arxiv_preprint.pdf](references/kapoor_2022_leakage_arxiv_preprint.pdf)
> is the authors' 2022 preprint (arXiv 2207.07048v1), free, **not** the cited
> article: its title differs ("... ML-based Science"), and nothing in it has
> been checked against the article. The article is open access
> (CC BY 4.0) but has to be saved from a web browser,
> https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9.

**Dunn, O. J. (1961).** Multiple comparisons among means. *Journal of the
American Statistical Association*, *56*(293), 52–64.
https://doi.org/10.1080/01621459.1961.10482090

> Sets out what is now called the Bonferroni correction for many comparisons
> read at once: hold each one to the overall bar divided by their number, so
> that the chance of any false alarm among them stays within the overall bar
> whatever the comparisons have in common. Assumption A-37 uses it as this
> project's bar.
>
> Not held: a publisher's article, through a university library. The
> description above has not been checked against the article; the citation was
> checked against its publisher's record on 2026-10-03.

**Holm, S. (1979).** A simple sequentially rejective multiple test procedure.
*Scandinavian Journal of Statistics*, *6*(2), 65–70.
https://www.jstor.org/stable/4615733

> The procedure that tests the smallest p against the overall bar divided by
> the number of comparisons, the next against the bar divided by one fewer, and
> so on, stopping at the first that fails. It keeps the Bonferroni correction's
> guarantee and never calls fewer differences real. Assumption A-37 sets it
> beside this project's bar.
>
> Not held: a journal article on JSTOR, through a university library. The
> description above has not been checked against the article.

**Benjamini, Y., & Hochberg, Y. (1995).** Controlling the false discovery
rate: A practical and powerful approach to multiple testing. *Journal of the
Royal Statistical Society: Series B (Methodological)*, *57*(1), 289–300.
https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

> The procedure that bounds the expected share of false alarms among the
> differences called real, rather than the chance of any false alarm at all.
> Assumption A-37 sets it beside this project's bar.
>
> Not held: a publisher's article, through a university library. The
> description above has not been checked against the article; the citation was
> checked against its publisher's record on 2026-10-03.

**Spearman, C. (1904).** The proof and measurement of association between two
things. *The American Journal of Psychology*, *15*(1), 72–101.
https://doi.org/10.2307/1412159

> The paper that named the weakening of a correlation by errors of measurement
> "attenuation": when each of two series carries "accidental" errors, the
> correlation measured between them is smaller than the one between the true
> values, and the shortfall "cannot be in the least eliminated by any
> prolongation of the series" (pp. 89–90). The C-MAPSS resource page cites it
> for that name and that observation, where it explains why the paper's
> sensor correlations are as large as each sensor's scatter allows (finding
> F-23). The formula used there, a correlation equal to the slow parts'
> correlation times the square root of the two shares of slow drift, is
> derived on that page, not taken from this paper.
>
> Not held: free, as a scan of the journal, from the Internet Archive,
> https://archive.org/details/jstor-1412159, which refused this network. The
> passages on pp. 89–90 were checked on 2026-11-14 against the text the
> Internet Archive made from that scan by character recognition; its formulas
> are unreadable in that text, which is why the page derives its own. The page
> range is the scan's own, since Crossref's record gives only the first page.

---

## Comparable work on the same dataset

**Jin, Y., Meng, Z., Hu, Q., & Chen, D. (2025).** Heterogeneous dynamic-aware
GNN for RUL prediction of aeroengine. *IEEE Transactions on Instrumentation and
Measurement*, *74*, Article 3538014. https://doi.org/10.1109/TIM.2025.3569885

> A graph-based method on the same dataset that documents its preprocessing
> explicitly (its page 7): k-means clustering of the operating conditions for
> the subsets that have several (FD002 and FD004), scaling to (0, 1) within each
> condition, and a label ceiling of 125. This is the source of the substituted
> preprocessing described in assumption A-03, and the supporting evidence for
> A-01. Its study of the window length (its page 12, Fig. 14) is cited in
> `docs/LITERATURE_BASIS.md`, under holding engines out, as consistent with
> choosing a setting on the test set.
>
> Copy: [jin_2025_heterogeneous_dynamic_aware_gnn.pdf](references/jin_2025_heterogeneous_dynamic_aware_gnn.pdf),
> the publisher's version (licensed download). Through a university library.

**Li, X., Ding, Q., & Sun, J.-Q. (2018).** Remaining useful life estimation in
prognostics using deep convolution neural networks. *Reliability Engineering &
System Safety*, *172*, 1–11. https://doi.org/10.1016/j.ress.2017.11.021

> A widely cited convolutional baseline on C-MAPSS, reported in the comparison
> table of Wen et al. Recommended as the first baseline to reproduce, because
> a pipeline that cannot reach its published error is a pipeline with a
> problem.
>
> Not held: a publisher's article, through a university library. A pre-review
> preprint with different pages is at https://escholarship.org/uc/item/5ns8r3fs.

**Zou, J., & Lin, P. (2025).** Multichannel attention-based TCN-GRU network for
remaining useful life prediction of aero-engines. *Energies*, *18*(8), 1899.
https://doi.org/10.3390/en18081899

> Wen et al.'s reference [55], the source of the row "GRU [55]" in their
> Tables 7 to 10. Its title names a network that joins a temporal
> convolutional network (TCN), attention and a gated recurrent unit (GRU), so
> by its title alone it is not a plain GRU, and it is not Wen et al.'s network
> with another recurrent cell; assumption A-36 says why no published row may
> be set beside this project's GRU stage. The citation is the one Wen et al.
> print, with the DOI from the publisher's record, checked on 2026-10-03.
>
> Not held: free under CC BY 4.0 from the publisher; nothing in it has been
> checked against the article.

**Dida, M., Cheriet, A., & Belhadj, M. (2025).** Remaining useful life
prediction using attention-LSTM neural network of aircraft engines.
*International Journal of Prognostics and Health Management*, *16*(2).
https://doi.org/10.36001/ijphm.2025.v16i2.4274

> The only published precedent found for leaving out engines shorter than the
> 30-cycle window rather than padding them (assumption A-23), and a precedent
> for windows one cycle apart (A-05), the ceiling of 125 (A-01) and a 14-sensor
> selection that differs from Wen et al.'s (`docs/LITERATURE_BASIS.md`).
>
> Copy: [dida_2025_attention_lstm_rul.pdf](references/dida_2025_attention_lstm_rul.pdf),
> the open-access article (CC BY 3.0), free; the journal's server sometimes
> needs a second try.

The remaining twelve were read as a group, for what other work on C-MAPSS
gives a graph node to carry (finding F-12) and for the precedents gathered in
`docs/LITERATURE_BASIS.md`. Page numbers quoted from them in this project are
page numbers of these files.

**Li, T., Zhao, Z., Sun, C., Yan, R., & Chen, X. (2021).** Hierarchical
attention graph convolutional network to fuse multi-sensor signals for
remaining useful life prediction. *Reliability Engineering & System Safety*,
*215*, Article 107878. https://doi.org/10.1016/j.ress.2021.107878

> HAGCN. Copy: [li_2021_hierarchical_attention_gcn.pdf](references/li_2021_hierarchical_attention_gcn.pdf),
> publisher's version, through a university library.

**Li, T., & Dong, J. (2026).** Multi-level attention graph feature fusion
smooth prognostics approach for aircraft engines remaining useful life
prediction. *Journal of Control and Decision*, *13*(2), 399–415 (published
online 2024). https://doi.org/10.1080/23307706.2024.2398536

> MAGF-SPA, which uses the newer N-CMAPSS data only. Copy:
> [li_2026_multilevel_attention_graph.pdf](references/li_2026_multilevel_attention_graph.pdf),
> publisher's download with a cover sheet, so its PDF page 10 is the journal's
> page 407; through a university library.

**Wang, Y., Wu, M., Li, X., Xie, L., & Chen, Z. (2025).** A survey on graph
neural networks for remaining useful life prediction: Methodologies,
evaluation and future trends. *Mechanical Systems and Signal Processing*,
*229*, Article 112449. https://doi.org/10.1016/j.ymssp.2025.112449

> Copy: [wang_2025_gnn_rul_survey.pdf](references/wang_2025_gnn_rul_survey.pdf),
> publisher's version, through a university library. A free preprint,
> arXiv 2409.19629v1, has different page numbers.

**Wen, Z., Fang, Y., Wei, P., Liu, F., Chen, Z., & Wu, M. (2025).** Temporal
and heterogeneous graph neural network for remaining useful life prediction.
*IEEE Transactions on Neural Networks and Learning Systems*, *36*(11),
19748–19761. https://doi.org/10.1109/TNNLS.2025.3592788

> THGNN; a different Wen from the paper this project re-implements. Copy:
> [wen_2025_temporal_heterogeneous_gnn.pdf](references/wen_2025_temporal_heterogeneous_gnn.pdf),
> publisher's version (licensed download). The free author version
> arXiv 2405.04336v3 has the passages this project cites on the same pages.

**Zhang, Q., Liu, J., Xie, K., & Liu, Q. (2025).** A dynamic spatio-temporal
graph neural network with global and local perception for remaining useful
life prediction. *Expert Systems with Applications*, *282*, Article 127707.
https://doi.org/10.1016/j.eswa.2025.127707

> GLSTGN. Copy: [zhang_2025_dynamic_spatiotemporal_gnn.pdf](references/zhang_2025_dynamic_spatiotemporal_gnn.pdf),
> publisher's version, through a university library.

**Zheng, L., Wu, F., Li, C., Estupinan, E., & Qin, Y. (2026).** Dynamic
curvature pooling graph convolutional network to fuse multi-sensor signals for
remaining useful life prediction. *Advanced Engineering Informatics*, *74*,
Article 104693. https://doi.org/10.1016/j.aei.2026.104693

> Copy: [zheng_2026_dynamic_curvature_pooling_gcn.pdf](references/zheng_2026_dynamic_curvature_pooling_gcn.pdf),
> publisher's version, through a university library.

**Ma, M., Wang, Z., & Zhong, Z. (2024).** Transformer encoder enhanced by an
adaptive graph convolutional neural network for prediction of aero-engines'
remaining useful life. *Aerospace*, *11*(4), Article 289.
https://doi.org/10.3390/aerospace11040289

> AGCTE. Copy: [ma_2024_transformer_adaptive_gcn.pdf](references/ma_2024_transformer_adaptive_gcn.pdf),
> the open-access article (CC BY 4.0). MDPI's site refuses scripted downloads,
> so it has to be saved from a web browser.

**Wang, L., Cao, H., Ye, Z., Xu, H., & Yan, J. (2024).** DVGTformer: A
dual-view graph Transformer to fuse multi-sensor signals for remaining useful
life prediction. *Mechanical Systems and Signal Processing*, *207*, Article
110935. https://doi.org/10.1016/j.ymssp.2023.110935

> Copy: [wang_2024_dvgtformer.pdf](references/wang_2024_dvgtformer.pdf),
> publisher's version, through a university library.

**Yuan, Y., Yang, Q., Wang, G., Ren, J., Wang, Z., Qiu, F., Li, K., & Liu, H.
(2025).** Combined improved tuna swarm optimization with graph convolutional
neural network for remaining useful life of engine. *Quality and Reliability
Engineering International*, *41*(1), 174–191. https://doi.org/10.1002/qre.3651

> IMTSO-GCN. Copy: [yuan_2025_tuna_swarm_gcn.pdf](references/yuan_2025_tuna_swarm_gcn.pdf),
> publisher's version (licensed download); its PDF page 10 is the journal's
> page 183.

**Wang, Y., Wu, M., Li, X., Xie, L., & Chen, Z. (2024).** Multivariate
time-series representation learning via hierarchical correlation pooling
boosted graph neural network. *IEEE Transactions on Artificial Intelligence*,
*5*(1), 321–333. https://doi.org/10.1109/TAI.2023.3241896

> HierCorrPool. Copy: [wang_2024_hierarchical_correlation_pooling.pdf](references/wang_2024_hierarchical_correlation_pooling.pdf),
> publisher's version (licensed download).

**Cao, X., Cheng, B., Yang, X., Duan, Y., & Gao, J. (2025).** Remaining useful
life prediction method for turbofan engines based on graph neural networks.
*IEEE Sensors Journal*, *25*(8), 13566–13578.
https://doi.org/10.1109/JSEN.2025.3546498

> ADFG. Copy: [cao_2025_turbofan_rul_gnn.pdf](references/cao_2025_turbofan_rul_gnn.pdf),
> publisher's version (licensed download).

**Ren, Y., Meng, Z., & Li, J. (2026).** Spatial-temporal graph hybrid neural
network for remaining useful life prediction of aero-engines. *Reliability
Engineering & System Safety*, *274*, Article 112433.
https://doi.org/10.1016/j.ress.2026.112433

> Copy: [ren_2026_spatial_temporal_graph_hybrid.pdf](references/ren_2026_spatial_temporal_graph_hybrid.pdf),
> publisher's version, through a university library.

**Shi, J., Zhong, J., Zhang, Y., Xiao, B., Xiao, L., & Zheng, Y. (2024).** A
dual attention LSTM lightweight model based on exponential smoothing for
remaining useful life prediction. *Reliability Engineering & System Safety*,
*243*, 109821. https://doi.org/10.1016/j.ress.2023.109821

> Named in `docs/LITERATURE_BASIS.md` as a possible precedent for the smoothing
> choices, found only in other papers' reference lists and not treated as
> evidence. Not held: a publisher's article with no free copy, through a
> university library.

---

## Software

**Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen,
P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R.,
Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., Fernández
del Río, J., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020).** Array
programming with NumPy. *Nature*, *585*(7825), 357–362.
https://doi.org/10.1038/s41586-020-2649-2

> Copy: [harris_2020_numpy.pdf](references/harris_2020_numpy.pdf), the
> open-access article (CC BY 4.0), free.

**The pandas development team. (2024).** pandas-dev/pandas: Pandas. Zenodo.
https://doi.org/10.5281/zenodo.3509134

> A software archive, so no copy is kept.

**Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel,
O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J.,
Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É.
(2011).** Scikit-learn: Machine learning in Python. *Journal of Machine
Learning Research*, *12*, 2825–2830.
https://www.jmlr.org/papers/v12/pedregosa11a.html

> The k-means implementation used for operating-condition detection comes from
> scikit-learn. Cite these three in the thesis if a software section is
> required.
>
> Copy: [pedregosa_2011_scikit_learn.pdf](references/pedregosa_2011_scikit_learn.pdf),
> the JMLR paper, free.

**PyTorch. (n.d.).** https://pytorch.org/

> The library the network, its training and its explanations are written in.
> Software, so no copy is kept.

---

## The website's design rules

`CLAUDE.md` (rule 11) builds the website's sizes and colours on three sources.
None is a paper, so no copies are kept.

- **Apple. (n.d.).** *Human Interface Guidelines*: typography.
  https://developer.apple.com/design/human-interface-guidelines/typography —
  the smallest and default text sizes on a Mac, and the advice against light
  font weights.
- **World Wide Web Consortium. (2023).** *Web Content Accessibility Guidelines
  (WCAG) 2.2*. https://www.w3.org/TR/WCAG22/ — the contrast ratio between text
  and its background, at least 4.5 to 1 for ordinary text and 3 to 1 for the
  edge of a control.
- **Filipiuk, M. (n.d.).** *UI Design Principles* [E-book]. — the method used
  to build the scale of text sizes.
