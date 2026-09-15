# The paper, item by item, against the code

## What this file is

One row for every numbered part of Wen et al. (2026): each equation, table,
figure and section. Each row says what the paper prints, where this repository
does it, which automated check fails if the code stops doing it, and where the
two part company.

It replaces nothing. The audit of 2026-08-31
(`docs/AUDIT_2026-08-31_paper_against_code.md`) is the record of how each row
was first checked, and `docs/ASSUMPTIONS.md` holds the reasoning behind every
departure. This file is the part that stays true as the code changes.

## Why it is called living, and what that does and does not mean

A document like the audit is right on the day it is written and then silently
goes out of date: a function is renamed, a test is deleted, and nothing tells
the reader that the row they are relying on now points at nothing.

`tests/test_paper_to_code.py` reads this table every time `pytest` runs, and
fails when:

- a function, class or constant named in the **Code** column no longer exists;
- a test or check named in the **Checked by** column no longer exists;
- a numbered choice or finding named in the **Standing** column, such as
  `A-21`, has no entry in `docs/ASSUMPTIONS.md`;
- a numbered part of the paper has no row, or has two;
- a row says the code does something and names no check for it.

**What it cannot do** is tell whether the code is right. That is the job of
the test each row names. This file keeps the pointers from the paper to those
tests alive; the tests decide whether the code still does what the paper says.

## How every row was checked

Every entry in the **What the paper prints** column was read on 2026-09-15 from
the PDF's text layer, extracted with `pypdfium2`. Where the text layer loses a
mark that changes the meaning, the region was rendered and read instead:
Equation 23's denominator at scale 8, and pages 3, 4 and 5 for Figures 1, 2 and
3 and the missing pseudocode. Nothing here is from memory.

## How to read a row

- **Paper**: the part of the paper.
- **What the paper prints**: in short, and quoted where the wording matters.
- **Code**: where in `src/thesis_gnn_rul_xai/` it is done, written
  `module:name`. A dash means there is nothing to build.
- **Checked by**: the test in `tests/`, or the check in
  `scripts/verify_claims.py`, that fails if the code stops doing it.
- **Standing**: whether the code follows the paper, and if not, what this
  project chose instead, with the entry that argues it in parentheses.

---

## The table

| Paper | What the paper prints | Code | Checked by | Standing |
|---|---|---|---|---|
| Section 1 | Why remaining-life prediction matters, and earlier work | — | — | Nothing to build. |
| Table 1 | Five earlier works compared on graph, physics prior, interpretability and time sequence | — | — | Nothing to build: a summary of other papers. |
| Section 2 | Four parts: parallel LSTM networks, a physics-based graph, graph convolution, dynamic SHAP; then "The algorithm pseudocode is shown as follows:" | `thesis_gnn_rul_xai.models.estgnn:ESTGNN` | `tests/test_models.py::test_activations_and_shapes_match_table_4` | Built. The promised pseudocode is not printed: on the rendered page 4 the next line is the heading of Section 2.1. |
| Fig. 1 | The whole method: feature selection, smoothing, piecewise remaining life, normalisation, then train and test sets; a sliding window marked "t=1s"; parallel LSTMs; node features; the graph; SHAP graphs | `thesis_gnn_rul_xai.data.pipeline:run_preprocessing` | `tests/test_pipeline.py::test_validation_engines_never_influence_the_zscore_statistics`, `tests/test_normalization.py::test_test_data_never_influences_the_scaling` | Built in a different order: the scaling is measured on the training engines after the split, the reverse of the figure (A-25). Windows one cycle apart, read from "t=1s" (A-05). Its node-feature list names two sensors the paper's own selection removes (F-13). |
| Section 2.1 | A multivariate series X = {x1, …, xt} with xt in R^n for n sensors; a graph G = (V, E, A); learn f:(X, G) → y | `thesis_gnn_rul_xai.data.windowing:WindowedDataset` | `tests/test_windowing.py::test_window_shape_is_nodes_then_every_cycle` | Built: each example is 14 sensors by 30 cycles, labelled with the remaining life at its last cycle. |
| Section 2.2 | "Each sensor s ∈ {1, 2, …, n} is associated with a dedicated LSTM network" | `thesis_gnn_rul_xai.models.temporal:LSTMStage` | `tests/test_models.py::test_the_paper_gives_every_sensor_its_own_network` | Built, and not the default (A-11). |
| Eqs. 1–6 | The LSTM's forget, input and output gates, candidate values, cell state and hidden state | `thesis_gnn_rul_xai.models.temporal:LSTMStage` | `tests/test_models.py::test_the_paper_gives_every_sensor_its_own_network` | Built on PyTorch's LSTM, which computes these six equations with two bias vectors per gate where the paper prints one. The two add into one number, so the function computed is the same. |
| Fig. 2 | A diagram of one LSTM cell | `thesis_gnn_rul_xai.models.temporal:LSTMStage` | `tests/test_models.py::test_the_paper_gives_every_sensor_its_own_network` | As Eqs. 1–6. |
| Eq. 7 | h_t^(s) = LSTM([x_{t−w+1,s}, …, x_{t,s}]), one network per sensor over its own window | `thesis_gnn_rul_xai.models.temporal:LSTMStage.forward` | `tests/test_models.py::test_the_paper_gives_every_sensor_its_own_network` | Built: fourteen networks, a citation. Their size, depth and how the output becomes one number are never stated, so they are this project's choice (A-26). The default passes each sensor's last reading instead (A-07). |
| Section 2.3.1 | Five rules for joining sensors; binary entries; a 14 × 14 matrix | `thesis_gnn_rul_xai.graph.adjacency:PHYSICS_ADJACENCY_ROWS` | `tests/test_adjacency.py::test_the_matrix_matches_the_papers_own_construction_rules_except_one` | Table 3 is used as printed. The five rules require one edge the printed matrix does not have (F-07). |
| Table 2 | The 21 sensors: number, symbol, description, unit | `thesis_gnn_rul_xai.data.sensors:SENSORS` | `tests/test_sensors.py::test_there_are_twenty_one_sensors` | Built. Fig. 1 names two of them differently (F-13). |
| Table 3 | The 14 × 14 matrix of which sensors are joined | `thesis_gnn_rul_xai.graph.adjacency:PHYSICS_ADJACENCY_ROWS` | `tests/test_adjacency.py::test_the_graph_is_symmetric`, `tests/test_adjacency.py::test_the_graph_matches_the_published_edge_count` | Transcribed exactly, and checked when the module is imported. |
| Fig. 3 | The engine's layout, the same matrix as a coloured grid, and the graph drawn from it | `thesis_gnn_rul_xai.graph.adjacency:build_physics_adjacency` | `tests/test_adjacency.py::test_the_graph_matches_the_published_edge_count` | Agrees with Table 3, as the audit of 2026-08-31 found. |
| Section 2.3.2 | Three graph layers with ReLU, a flatten, two fully connected layers; node features "from parallel LSTM networks" | `thesis_gnn_rul_xai.models.estgnn:ESTGNN` | `tests/test_models.py::test_activations_and_shapes_match_table_4` | Built. |
| Eq. 8 | H(1) = ReLU(Ã X W(1)), Ã "the normalized adjacency matrix incorporating self-loops" | `thesis_gnn_rul_xai.models.layers:GraphConvolution.forward`, `thesis_gnn_rul_xai.graph.adjacency:normalize_adjacency` | `tests/test_models.py::test_layer_averages_over_neighbours`, `tests/test_adjacency.py::test_normalisation_adds_self_loops` | Built. The formula that rescales the matrix is never printed; each row is divided by its sum (A-21). |
| Eq. 9 | H(2) = ReLU(Ã H(1) W(2)), with no bias term | `thesis_gnn_rul_xai.models.estgnn:ESTGNN.forward` | `tests/test_models.py::test_activations_and_shapes_match_table_4` | Built with no bias in the graph layers (A-12). The sentence beneath misnumbers the matrices (F-14). |
| Eq. 10 | H(3) = ReLU(Ã H(2) W(3)), with no bias term | `thesis_gnn_rul_xai.models.estgnn:ESTGNN.forward` | `tests/test_models.py::test_activations_and_shapes_match_table_4` | As Eq. 9. |
| Eq. 11 | z = Flatten(H(3)) | `thesis_gnn_rul_xai.models.estgnn:ESTGNN.forward` | `tests/test_models.py::test_activations_and_shapes_match_table_4` | Built. |
| Eq. 12 | h_fc1 = ReLU(W_fc1 · z + b_fc1) | `thesis_gnn_rul_xai.models.estgnn:ESTGNN.forward` | `tests/test_models.py::test_parameter_counts_match_table_4` | Built, with the bias the equation prints. |
| Eq. 13 | ŷ = Sigmoid(W_fc2 · h_fc1 + b_fc2), "normalized RUL values within [0,1]" | `thesis_gnn_rul_xai.models.estgnn:ESTGNN.predict_cycles` | `tests/test_models.py::test_output_is_one_number_per_example_inside_the_unit_interval`, `tests/test_models.py::test_the_prediction_can_never_exceed_the_ceiling` | Built. The output is multiplied by the label ceiling to give cycles (A-13). |
| Table 4 | Input [batch, 14, 1]; W(1) 1 × 32, W(2) 32 × 64, W(3) 64 × 32; flatten to 448; W_fc1 448 × 128; W_fc2 128 × 1 | `thesis_gnn_rul_xai.models.estgnn:GCN_HIDDEN_SIZES`, `thesis_gnn_rul_xai.models.estgnn:FC_HIDDEN_SIZE` | `tests/test_models.py::test_parameter_counts_match_table_4`, `tests/test_models.py::test_the_default_width_reproduces_table_4` | Built. It prints no total; 61,729 is derived from its shapes and the two printed biases (A-07). A node carrying one number is kept by the default; the window-summary stage gives it three and departs from this table (A-28). |
| Fig. 4 | Three GCN layers, flatten, two fully connected layers | `thesis_gnn_rul_xai.models.estgnn:ESTGNN` | `tests/test_models.py::test_activations_and_shapes_match_table_4` | Built. |
| Section 2.4 | SHAP applied to "individual sensor nodes" | `thesis_gnn_rul_xai.explain.shap:shap_values` | `tests/test_shap.py::test_efficiency_holds_to_machine_precision` | Built. |
| Eq. 14 | g(z) = φ0 + Σ φi zi, with φ0 "the average value of target variable across all samples" | `thesis_gnn_rul_xai.explain.shap:efficiency_error` | `tests/test_shap.py::test_efficiency_holds_to_machine_precision` | Built with φ0 the prediction at the average window, not the average label (A-24). Table 12 cannot satisfy this equation (F-08). |
| Eq. 15 | φi = Σ over S ⊆ N∖{i} of \|S\|!(n − \|S\| − 1)!/n! · (f(S ∪ {i}) − f(S)), n = 14 | `thesis_gnn_rul_xai.explain.shap:coalition_weights`, `thesis_gnn_rul_xai.explain.shap:shap_values` | `tests/test_shap.py::test_coalition_weights_account_for_every_ordering_once`, `tests/test_shap.py::test_interchangeable_sensors_receive_equal_values` | Built exactly, over every subset of the other sensors. What f(S) means when a sensor is absent is never said; it gets its average training window (A-20). |
| Eq. 16 | The same sum at each time step t | `thesis_gnn_rul_xai.explain.shap:dynamic_shap` | `tests/test_shap.py::test_dynamic_shap_covers_one_engine_in_cycle_order`, `tests/test_shap.py::test_dynamic_shap_rows_each_satisfy_efficiency` | Built. |
| Section 3.1 | An Intel 13000KF, an RTX 4090, Windows 11, Python 3.9, NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.1.3 and LightGBM 4.5.0; no deep learning library named | `thesis_gnn_rul_xai.utils.provenance:environment` | `tests/test_run_records.py::test_the_environment_names_python_and_every_library_a_number_depends_on` | Not reproducible from what is printed. Every run of this project records its own machine and library versions instead. |
| Section 3.2.1 | Four sub-datasets; training runs to failure, test histories stop before it | `thesis_gnn_rul_xai.data.loading:load_subset` | `tests/test_loading.py::test_training_labels_reach_zero_at_the_end_of_every_engine`, `tests/test_loading.py::test_test_labels_match_the_official_file_at_the_last_cycle` | Built. |
| Table 5 | Training engines 100, 260, 100, 249; test engines 100, 259, 100, 248; conditions 1, 6, 1, 6; fault modes 1, 1, 2, 2 | `thesis_gnn_rul_xai.data.loading:EXPECTED_SHAPES`, `thesis_gnn_rul_xai.evaluation.published:DATASET_DESCRIPTION` | `tests/test_loading.py::test_the_files_match_the_published_release`, `scripts/verify_claims.py::check_windows` | FD001's counts are checked against the NASA files by the test; all four agreed in the audit of 2026-08-31. Test engines shorter than the window are left out, not padded (A-23). |
| Section 3.2.2 | Pearson correlation; remove constant features and those correlated above 0.95; "seven sensors are removed … including T2, P2, epr, farB, Nfdmd and PCNRdmd" | `thesis_gnn_rul_xai.data.sensors:PAPER_EXCLUDED_SYMBOLS`, `thesis_gnn_rul_xai.data.sensors:detect_uninformative_sensors` | `tests/test_sensors.py::test_six_of_the_seven_dropped_sensors_are_constant`, `scripts/verify_claims.py::check_sensors` | Built. The unnamed seventh is NRc, and the paper's own criterion confirms it (A-10). |
| Eq. 17 | The Pearson correlation coefficient | `thesis_gnn_rul_xai.graph.adjacency:build_correlation_adjacency` | `scripts/verify_claims.py::check_sensors`, `tests/test_adjacency.py::test_correlation_graph_keeps_exactly_the_strongest_pairs_asked_for` | Used to confirm the feature selection (A-10) and to build the graph from the data (A-27). |
| Fig. 5 | Correlation heatmap of FD001 "after removing the constant features" | `thesis_gnn_rul_xai.data.sensors:PAPER_SENSOR_SYMBOLS` | `tests/test_sensors.py::test_kept_sensor_order_matches_the_paper_table` | Shows the fourteen kept sensors with NRc absent, which supports the reading of the unnamed seventh (A-10). |
| Section 3.2.3 | Exponentially weighted moving average to remove noise | `thesis_gnn_rul_xai.data.smoothing:apply_ewma` | `tests/test_smoothing.py::test_engines_are_smoothed_independently` | Built. |
| Eq. 18 | S_t = α·X_t + (1 − α)·S_{t−1}, with "which is equal to 0.5" attached to S_{t−1} | `thesis_gnn_rul_xai.data.smoothing:ewma_series` | `tests/test_smoothing.py::test_matches_the_recursive_formula_by_hand`, `tests/test_smoothing.py::test_the_filter_only_looks_backwards` | The recursive form (A-09), with α = 0.5 read from a sentence that attaches the value to the wrong symbol (A-04). |
| Section 3.2.3.1 | Remaining life constant while healthy, then falling linearly after a change point | `thesis_gnn_rul_xai.data.labeling:apply_piecewise_rul` | `tests/test_labeling.py::test_the_curve_is_flat_then_straight` | Built. |
| Eq. 19 | RUL_t = min(T_total − T_end, RUL_max), with RUL_max never given a value | `thesis_gnn_rul_xai.data.labeling:apply_piecewise_rul` | `tests/test_labeling.py::test_values_above_the_ceiling_are_capped`, `tests/test_pipeline.py::test_labels_respect_the_ceiling` | A ceiling of 125 cycles, read from Fig. 7 (A-01), applied to the test labels too (A-02). |
| Fig. 6 | The truncation schematic | `thesis_gnn_rul_xai.data.labeling:piecewise_rul_curve` | `tests/test_labeling.py::test_the_curve_is_flat_then_straight` | Same shape. It draws the ceiling on the 120 gridline, which disagrees with Fig. 7 (A-01). |
| Section 3.3 | Four measures: score function, RMSE, R² and ASUE | `thesis_gnn_rul_xai.evaluation.metrics:evaluate` | `tests/test_metrics.py::test_evaluate_returns_every_measure` | Built. |
| Eq. 20 | δi = yi − ŷi, actual minus predicted | `thesis_gnn_rul_xai.evaluation.metrics:prediction_error` | `tests/test_metrics.py::test_prediction_error_is_positive_when_the_model_is_optimistic` | The sign is reversed to predicted minus actual. The text under Eq. 21 calls δi ≥ 0 over-prediction and punishes it harder, which is true only of predicted minus actual, the NASA convention (A-15). |
| Eq. 21 | SF = Σ si, with e^(−δi/13) − 1 for δi < 0 and e^(δi/10) − 1 for δi ≥ 0 | `thesis_gnn_rul_xai.evaluation.metrics:nasa_score` | `tests/test_metrics.py::test_late_predictions_are_punished_harder_than_early_ones`, `tests/test_metrics.py::test_the_score_is_a_sum_not_a_mean` | The divisors 13 and 10 as printed, under the NASA sign (A-15). |
| Eq. 22 | RMSE = √((1/N) Σ δi²) | `thesis_gnn_rul_xai.evaluation.metrics:rmse` | `tests/test_metrics.py::test_rmse_is_in_cycles` | Built. |
| Eq. 23 | R² = 1 − Σ δi² / Σ (ŷi − ȳi)², ȳ "the mean of the observed values" | `thesis_gnn_rul_xai.evaluation.metrics:r_squared` | `tests/test_metrics.py::test_r_squared_uses_the_true_labels_in_its_denominator` | The printed denominator uses the predicted values; the code uses the observed ones, which is what R² is (A-16). |
| Eq. 24 | ASUE = (1/N_under) Σ max(0, yi − ŷi) | `thesis_gnn_rul_xai.evaluation.metrics:asue` | `tests/test_metrics.py::test_asue_counts_only_the_pessimistic_engines` | Built. |
| Section 3.4 | Mean squared error; Adam; a learning rate "beginning at 1 × 10⁻³ and reducing to 1 × 10⁻⁴"; 200 epochs; batches of 32; a window of 30 time steps (an epoch is one pass over the training data) | `thesis_gnn_rul_xai.training.config:TrainingConfig` | `tests/test_training.py::test_defaults_are_the_values_of_table_6`, `tests/test_training.py::test_the_learning_rate_reaches_its_stated_final_value` | Built. How the rate moves between the two values is not said: one tenfold drop at the halfway pass (A-17). No held-out engines are described; this project holds a fifth back and stops early (A-06, A-18). The starting weights are not described (A-14). |
| Table 6 | Learning rate 1e−3 to 1e−4; epochs 200; batch size 32; time steps 30; Adam; MSE loss. Its caption repeats Table 2's | `thesis_gnn_rul_xai.training.config:TrainingConfig`, `thesis_gnn_rul_xai.evaluation.published:TRAINING_SETTINGS` | `tests/test_training.py::test_defaults_are_the_values_of_table_6`, `tests/test_playground.py::test_the_defaults_the_paper_states_are_the_ones_it_prints` | As Section 3.4. |
| Section 4.1 | Eight methods compared with ESTGNN on four measures | `thesis_gnn_rul_xai.evaluation.published:published_row` | `tests/test_playground.py::test_the_transcribed_rmse_row_averages_to_what_table_8_prints` | ESTGNN's rows are transcribed for comparison. None of the other methods is built here. |
| Tables 7–10 | ESTGNN on FD001: score 278.6 (Table 7), RMSE 11.78 (Table 8), R² 0.921 (Table 9), ASUE 8.312 (Table 10) | `thesis_gnn_rul_xai.evaluation.published:published_row` | `tests/test_playground.py::test_the_transcribed_rmse_row_averages_to_what_table_8_prints` | Transcribed. A configuration without the recurrent stage may not be set beside these rows, only beside Table 11's "w/o LSTM" (A-11). Table 9 contradicts the Conclusions' claim of the highest R² (F-11). |
| Section 4.2 | Prediction trajectories and error ranges | — | — | Nothing to build. |
| Fig. 7 | Remaining-life trajectories for FD001 unit 38 and one unit of each other sub-dataset | `thesis_gnn_rul_xai.config:PreprocessingConfig` | `tests/test_pipeline.py::test_labels_respect_the_ceiling` | Its plateau, read against its own axis, is the evidence for the ceiling of 125 (A-01). |
| Fig. 8 | True against predicted remaining life, four sub-datasets | — | — | Nothing to check against this code. |
| Section 4.3 | Three variants: without GCN, without LSTM, without physics prior, "under identical experimental conditions" | `thesis_gnn_rul_xai.evaluation.published:ABLATION_ROWS` | `tests/test_playground.py::test_each_ablation_row_matches_the_average_and_delta_table_11_prints` | The graph used "without physics prior" is never described; every sensor joined to every other stands in for it (A-19), and most of what that costs is averaging (F-04, A-27). "Without GCN" is not built. |
| Table 11 | RMSE only. FD001: full 11.78, w/o GCN 16.42, w/o LSTM 14.93, w/o physics prior 12.34 | `thesis_gnn_rul_xai.evaluation.published:published_row` | `tests/test_playground.py::test_each_ablation_row_matches_the_average_and_delta_table_11_prints`, `tests/test_playground.py::test_table_11_prints_rmse_only_so_nothing_else_is_invented_for_it` | Transcribed. The default configuration is compared with its "w/o LSTM" row (A-11). What the recurrent stage is worth depends on a training step the paper does not describe (F-16). |
| Section 4.4 | Early stage led by T50, Nc, Ps30, phi; intermediate by T50, P30, Ps30, phi; late by Nc, Ps30, BPR with T50 | `thesis_gnn_rul_xai.explain.shap:dynamic_shap` | `tests/test_playground.py::test_table_12_top_four_in_each_column_are_the_cells_the_paper_sets_in_bold` | The prose agrees with Table 12 and not with Fig. 9 (F-15). |
| Table 12 | SHAP values for 14 sensors at graphs 1–5, 78–82 and 187–191, the four largest in bold | `thesis_gnn_rul_xai.evaluation.published:SHAP_TABLE_12` | `tests/test_playground.py::test_table_12_column_sums_are_the_ones_finding_f08_prints`, `tests/test_playground.py::test_table_12_covers_the_paper_sensors_once_each_at_fifteen_graphs` | Transcribed. Every cell is positive, which signed Shapley values cannot all be (F-08). |
| Fig. 9 | How the dynamic graph is built: red for the four most critical sensors, blue for the rest | `thesis_gnn_rul_xai.evaluation.published:FIG_10_RING` | `tests/test_playground.py::test_table_12_top_four_in_each_column_are_the_cells_the_paper_sets_in_bold` | Its four critical sensors disagree with Table 12 (F-15). |
| Fig. 10 | Fifteen dynamic graphs, from the early to the advanced stage | `thesis_gnn_rul_xai.evaluation.published:FIG_10_RING` | `tests/test_playground.py::test_table_12_top_four_in_each_column_are_the_cells_the_paper_sets_in_bold` | Drawn from Table 12 except in graphs 81 and 82 (F-17). |
| Section 5 | "the highest R² of 0.742"; the graph captures "authentic sensor relationships beyond data-driven correlations" | `thesis_gnn_rul_xai.graph.adjacency:build_correlation_adjacency` | `scripts/verify_claims.py::check_correlation` | The R² claim is contradicted by Table 9 (F-11). The claim about correlations is tested by building a graph from them, which could not be told apart from the paper's graph on test error under the default configuration (A-27). |

---

## Keeping it true

**When code named here moves**, the test fails with the row and the name that
no longer resolves. Fix the row in the same change; a row pointing at nothing
is worse than no row, because it looks like a guarantee.

**When the code starts departing from the paper**, the row's Standing changes
in the same change as the code, and the departure gets its entry in
`docs/ASSUMPTIONS.md` first, as working rule 4 requires.

**When something about the paper is found**, it goes into `docs/ASSUMPTIONS.md`
as a finding and its identifier goes into the row it concerns.
