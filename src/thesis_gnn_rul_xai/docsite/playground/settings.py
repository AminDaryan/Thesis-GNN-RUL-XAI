"""Every setting the playground offers, where each default comes from, and how a request is checked."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ...evaluation.published import SUBSETS, TRAINING_SETTINGS, published_row

# Fewer seeds than this is marked as exploratory everywhere it appears.
# CLAUDE.md rule 3: one seed is not a result, and five is the floor.
RESULT_SEEDS = 5


class SettingsError(ValueError):
    """A setting the playground refuses, with the reason in words."""


def _source(kind: str, ref: str, text: str) -> dict[str, str]:
    """Where a setting's default comes from: the paper, a logged choice, or neither."""
    return {"kind": kind, "ref": ref, "text": text}


def _paper(ref: str, text: str) -> dict[str, str]:
    return _source("paper", ref, text)


def _choice(ref: str, text: str) -> dict[str, str]:
    return _source("assumption", ref, text)


def _project(text: str) -> dict[str, str]:
    return _source("project", "", text)


def _extension(text: str) -> dict[str, str]:
    return _source("extension", "", text)


# Every setting the page offers, in the order the form draws them. The groups
# follow the pipeline. Inside a group, the lists of options about what the
# group is for come first: the temporal stage, the sensor graph and its
# rescaling. The small settings are kept together so that they fill rows of
# cells, and a list about something else, the sensor scaling or the learning
# rate's course, follows them on a row of its own. That is not the order the
# preprocessing runs in, which data/pipeline/__init__.py sets; the page groups the settings
# by the steps they decide (SECTIONS in assets/playground.js). Each fact about where a default comes
# from is said once: on the setting when every option shares it, on each
# option when the options differ, and nowhere when nothing sets it.
# An empty text leaves the tag alone, where the value in the control says it.
FIELDS: tuple[dict[str, Any], ...] = (
    # -- data: the form's first tab --------------------------------------
    {"key": "subset", "group": "data", "label": "Sub-dataset", "kind": "choice",
     "default": "FD001",
     "choices": [{"value": s, "label": s, "source": None} for s in SUBSETS],
     "about": ("Selects one of NASA's four datasets of simulated engine wear. Each contains separate engines "
               "for training, where the model learns from examples, and testing, where its predictions are "
               "scored. The datasets differ in operating conditions, such as altitude and speed, and in the "
               "types of component deterioration included."),
     "source": _paper("Table 5", "")},
    # -- procedure: set by the choice of procedure at the top of Training ---
    {"key": "validation_fraction", "group": "procedure",
     "label": "Training engines held back",
     "kind": "float", "default": 0.2, "min": 0.0, "max": 0.5, "step": 0.05,
     "about": ("The share of training-file engines reserved for checking progress: 0.2 means 20%. Their "
               "readings do not enter the updates that adjust the model's learned numbers. Their prediction "
               "error selects which training pass to keep. These validation engines are separate from the "
               "test engines used for the final score."),
     "source": _choice("A-06", "")},
    {"key": "early_stopping", "group": "procedure", "label": "Stop when held-back error stalls",
     "kind": "bool", "default": True,
     "about": ("Stops training after the allowed number of passes without a sufficient reduction in "
               "prediction error on the held-back engines. The model from the best recorded pass is restored. "
               "This requires held-back engines; the training procedure turns it on or off."),
     "source": _choice("A-18", "")},
    {"key": "early_stopping_patience", "group": "procedure",
     "label": "Passes allowed without improvement", "kind": "int",
     "default": 20, "min": 1, "max": 1000,
     "about": ("How many consecutive passes without a sufficient improvement on the held-back engines are "
               "allowed before stopping. A larger value waits longer for improvement. Very small reductions "
               "in error do not reset the count. The 'Drop when held-back error stalls' learning-rate option "
               "uses half this value, rounded down, as its waiting period, with a minimum of one pass."),
     "source": _choice("A-18", "")},
    # -- preprocessing -----------------------------------------------------
    {"key": "window_size", "group": "preprocessing", "label": "Window length (cycles)",
     "kind": "int", "default": 30, "min": 1, "max": 100,
     "about": ("A window is one model input: a sensor history spanning this many consecutive flight cycles. "
               "Its training target is the remaining life at the last cycle, after applying the label ceiling. Longer "
               "windows contain more history, but the temporal stage decides how much is used. Engines with "
               "fewer recorded cycles than this are excluded."),
     "source": _paper("Table 6", "")},
    {"key": "window_stride", "group": "preprocessing", "label": "Cycles between windows",
     "kind": "int", "default": 1, "min": 1, "max": 30,
     "about": ("The gap between the start cycles of consecutive windows from the same training or held-back "
               "engine. At 1, a window moves forward by one cycle at a time. Larger gaps generally produce "
               "fewer examples and less overlap. Test engines use only their final complete window, regardless "
               "of this setting."),
     "source": _choice("A-05", "")},
    {"key": "rul_max", "group": "preprocessing", "label": "Label ceiling (cycles)",
     "kind": "int", "default": 125, "min": 10, "max": 400,
     "about": ("The largest remaining-life target used for training. Any longer remaining life is replaced "
               "by this ceiling, so the model is not asked to distinguish among those longer lifetimes. The "
               "model's output is also limited to this ceiling. This is a labeling choice, not a measurement "
               "of when an engine starts to deteriorate."),
     "source": _choice("A-01", "the chosen value and its evidence are documented in the assumptions log")},
    {"key": "clip_test_labels", "group": "preprocessing", "label": "Cap the test labels too",
     "kind": "bool", "default": True,
     "about": ("Replaces test targets above the label ceiling with the ceiling before scoring. Predictions "
               "are unchanged. With this off, the model is scored against the full remaining life, including "
               "values its output cannot reach. The two choices answer different evaluation questions, so "
               "every reported result must identify which was used."),
     "source": _choice("A-02", "changes the test targets used for scoring")},
    {"key": "smoothing", "group": "preprocessing", "label": "Smooth each sensor",
     "kind": "bool", "default": True,
     "about": ("Replaces each sensor reading with a weighted average of itself and the previous smoothed "
               "reading. The first reading is kept unchanged. Each engine is processed separately, using "
               "the current and earlier cycles only. This reduces rapid fluctuations but can also delay "
               "genuine changes. Turning it off leaves readings unchanged at this step."),
     "source": _paper("Section 3.2.3", "exponential smoothing before windowing")},
    {"key": "smoothing_alpha", "group": "preprocessing", "label": "Smoothing weight on the newest reading",
     "kind": "float", "default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
     "about": ("After the first cycle, this fraction of the current reading is combined with the remaining "
               "fraction of the previous smoothed value. At 0.5 they contribute equally. At 1, smoothing "
               "changes nothing. Smaller values give older readings more influence and respond more slowly "
               "to a sustained change. Used only when smoothing is on."),
     "source": _choice("A-04", "")},
    {"key": "normalization", "group": "preprocessing", "label": "Sensor scaling",
     "kind": "choice", "default": "zscore",
     "choices": [
         {"value": "zscore", "label": "Mean and spread of the training engines", "source": None},
         {"value": "minmax_per_condition", "label": "Training range 0 to 1, per operating condition",
          "source": None},
         {"value": "minmax_global", "label": "Training range 0 to 1, all training rows", "source": None},
     ],
     "about": ("Rescales each sensor using training engines only; the same rule then handles held-back and "
               "test engines."
               "\nMean and spread: subtract the training mean and divide by the standard deviation, a measure "
               "of spread around that mean."
               "\nTraining range: map the training minimum to 0 and maximum to 1, either separately for groups "
               "of similar operating settings or across all training rows. New readings can fall outside 0 "
               "to 1. Sensors with effectively no training variation become 0."),
     "source": _choice("A-03", "the scaling method is a documented project choice")},
    # -- temporal stage ----------------------------------------------------
    {"key": "temporal", "group": "temporal", "label": "Temporal stage", "kind": "choice",
     "default": "lstm",
     "choices": [
         {"value": "lstm", "label": "LSTM, one per sensor",
          "about": ("A long short-term memory network (LSTM) reads each sensor's window from oldest to newest. "
                    "It combines each reading with a numerical memory of earlier readings, using rules adjusted "
                    "during training. Its final output is converted into the numbers passed to the graph. "
                    "Memory starts afresh for each window."),
          "source": _paper("Section 2.2", "")},
         {"value": "last_cycle", "label": "w/o LSTM: keep only the last cycle",
          "about": ("Passes each sensor's final preprocessed reading to the graph, with no learned operation "
                    "in this step. Earlier entries in the window are discarded here, although smoothing may "
                    "already have mixed earlier readings into the final value. 'w/o LSTM' means without the "
                    "network that processes the readings in time order."),
          "source": _choice("A-07", "")},
         {"value": "gru", "label": "GRU, one per sensor",
          "about": ("A gated recurrent unit (GRU) reads each sensor's window from oldest to newest, combining "
                    "each reading with a numerical memory of earlier readings. It carries one memory vector, "
                    "a list of numbers, between cycles; the LSTM carries two. Memory starts afresh for each "
                    "window. This is an alternative model added by this project."),
          "source": _extension("an alternative to the paper's LSTM")},
         {"value": "mean_window", "label": "Average over the window",
          "about": ("Passes the ordinary average of each sensor's preprocessed readings in the window to the "
                    "graph: add them and divide by the window length. Each sensor contributes one number. "
                    "This step has no learned numbers and does not preserve the order of readings within "
                    "the window."),
          "source": _extension("no learned parameters")},
         {"value": "window_summary", "label": "Last reading, average and change over the window",
          "about": ("Passes three numbers per sensor: the final preprocessed reading, the window average, and "
                    "the change from start to end of a fitted straight line. The line minimizes the sum of "
                    "squared differences from the readings. This step has no learned numbers and requires "
                    "at least two cycles; the graph layers learn how to combine its outputs."),
          "source": _extension("three summaries per sensor (assumption A-28)")},
         {"value": "raw_window", "label": "The whole window straight into the graph layer",
          "about": ("Passes every preprocessed reading in the window to the graph, in time order. Each sensor "
                    "therefore contributes as many numbers as there are cycles in the window. This step has "
                    "no learned numbers; the first graph layer then learns weighted sums across those cycle "
                    "positions, using the same multipliers for every sensor."),
          "source": _extension("the arrangement the last-cycle stage replaced")},
     ],
     "about": ("Decides how to represent each sensor's history before sensors exchange information. It takes "
               "that sensor's window of preprocessed readings and produces one or more numbers for the graph "
               "layers. The options either compute fixed summaries, pass all readings through, or learn how "
               "to combine readings in time order."),
     "source": None},
    {"key": "temporal_hidden", "group": "temporal", "label": "Hidden size", "kind": "int",
     "default": 32, "min": 1, "max": 512,
     "about": ("The length of each memory vector, a list of numbers carried between cycles by the selected "
               "temporal network. A GRU carries one such list; an LSTM carries two. A larger value adds learned "
               "numbers to the model. It does not change the window length or directly set how many numbers "
               "reach the graph."),
     "source": _choice("A-26", "")},
    {"key": "temporal_out_features", "group": "temporal",
     "label": "Numbers handed to the graph per sensor", "kind": "int",
     "default": 1, "min": 1, "max": 64,
     "about": ("How many numbers each sensor passes from its LSTM or GRU to the graph layers. Each is a "
               "weighted sum of that sensor's final temporal output, plus an added constant. The multipliers "
               "and constants are learned during training and shared across sensors. This setting does not "
               "change the number of sensors."),
     "source": _choice("A-26", "Table 4 gives one number per sensor; the conversion to it is our choice")},
    {"key": "shared_across_sensors", "group": "temporal",
     "label": "One network shared by all sensors", "kind": "bool", "default": False,
     "about": ("Off: each sensor has its own temporal network with its own learned numbers. On: the same "
               "network processes every sensor's window separately, without a sensor identifier. Sharing "
               "reduces the number of learned numbers; it does not mix readings from different sensors. "
               "Each window still starts with a fresh memory."),
     "source": _extension("")},
    # -- graph layers: the graph they run on, then their sizes -------------
    # The sensor graph is built while the data is prepared, but it is drawn
    # with the layers that use it, where the page's pipeline shows it too.
    {"key": "graph", "group": "graph", "label": "Sensor graph", "kind": "choice",
     "default": "physics",
     "choices": [
         {"value": "physics", "label": "The paper's physics graph",
          "about": ("Connects the sensor pairs that the paper lists as related. The authors motivate "
                    "these connections using sensor locations and engine relationships. The connections "
                    "control which sensors exchange numbers in this model and remain fixed during training. "
                    "A connection does not by itself establish a causal relationship."),
          "source": _paper("Table 3", "")},
         {"value": "complete", "label": "Every sensor joined to every other",
          "about": ("Connects every sensor to every other. With either graph-rescaling option, each sensor "
                    "then receives the same average of all sensors' incoming numbers. The first graph layer "
                    "therefore produces identical representations for every sensor. This changes both the "
                    "connections and the averaging, so it cannot isolate the value of physical knowledge."),
          "source": _choice("A-19", "our interpretation of the paper's variant without a physics prior")},
         {"value": "correlation", "label": "The 41 most correlated sensor pairs",
          "about": ("Connects the 41 pairs with the strongest positive or negative linear association, "
                    "measured on preprocessed training readings using Pearson correlation, a score from -1 "
                    "to 1 for how closely two readings follow a straight line. A strong association "
                    "does not prove causation. "
                    "The graph has the same pair count as the paper's graph and stays fixed during training."),
          "source": _choice("A-27", "built from each seed's own training engines, so seeds "
                                    "can get slightly different graphs")},
     ],
     "about": ("A graph is a set of sensors and connections between them. Each graph layer combines a "
               "sensor's current numbers with those of its connected neighbours. The connections and mixing "
               "weights from graph rescaling are fixed before model training; the later multipliers are learned. "
               "Each layer can "
               "carry information across one further connection."),
     "source": None},
    {"key": "adjacency_normalization", "group": "graph",
     "label": "Graph rescaling", "kind": "choice", "default": "row",
     "choices": [
         {"value": "row", "label": "Each row sums to one", "source": None},
         {"value": "symmetric", "label": "Symmetric, by both degrees", "source": None},
     ],
     "about": ("First include each sensor's own numbers alongside its neighbours'. Then assign weights to "
               "these connections using their counts, including the connection to itself."
               "\nEach row sums to one: take the ordinary average of the sensor and its neighbours."
               "\nSymmetric, by both degrees: divide each connection's weight by the square root of the two "
               "sensors' connection counts multiplied together. This reduces weights involving many "
               "connections; it need not produce an average."),
     "source": _choice("A-21", "the paper describes the step but prints no formula")},
    {"key": "gcn_hidden", "group": "graph", "label": "Graph layer widths", "kind": "widths",
     "default": [32, 64, 32], "min": 1, "max": 512, "length": 3,
     "about": ("How many numbers each sensor has after each of the three graph layers. A layer taking in 32 "
               "numbers per sensor and giving out 64 learns a table of 32 times 64 multipliers, used for every "
               "sensor, plus an optional offset for each output. A wider layer has more to learn. The last width, times the 14 "
               "sensors, is how many numbers reach the first fully connected layer."),
     "source": _paper("Table 4", "")},
    {"key": "fc_hidden", "group": "graph", "label": "First fully connected width",
     "kind": "int", "default": 128, "min": 1, "max": 2048,
     "about": ("Every sensor's numbers from the graph layers are laid end to end in one row, and this layer "
               "turns that row into as many new numbers as set here. Each is made by multiplying the whole row "
               "by learned numbers and adding up the results plus a learned offset, with a negative total set to "
               "zero, so any sensor can affect any of them."),
     "source": _paper("Table 4", "")},
    {"key": "gcn_bias", "group": "graph", "label": "Bias in the graph layers",
     "kind": "bool", "default": False,
     "about": ("With this on, each graph layer adds a learned offset to every number it gives out, the same "
               "offsets for every sensor, before negative results are set to zero. With it off, a layer only "
               "multiplies and adds up what came in, so if everything coming in is zero, everything going out is "
               "zero too."),
     "source": _choice("A-12", "Table 4 lists only a weight matrix per layer")},
    # -- training ----------------------------------------------------------
    {"key": "epochs", "group": "training", "label": "Passes over the training data",
     "kind": "int", "default": TRAINING_SETTINGS["epochs"], "min": 1, "max": 1000,
     "about": ("How many times training goes through all the training windows, in a new shuffled order each "
               "time, correcting the network's learned numbers after every group of windows. More passes give "
               "more chances to improve but take longer. When engines are held back, training can stop before "
               "the last pass."),
     "source": _paper("Table 6", "")},
    {"key": "batch_size", "group": "training", "label": "Windows per update",
     "kind": "int", "default": TRAINING_SETTINGS["batch_size"], "min": 1, "max": 4096,
     "about": ("Each pass splits the shuffled training windows into groups of this many. After each group the "
               "network corrects its learned numbers once, based on how far its predictions for that group were "
               "from the remaining life each window is labelled with. Smaller groups mean more corrections per "
               "pass, each based on fewer windows."),
     "source": _paper("Table 6", "")},
    {"key": "learning_rate", "group": "training", "label": "First learning rate",
     "kind": "float", "default": TRAINING_SETTINGS["learning_rate"],
     "min": 1e-4, "max": 1.0, "step": 1e-4,
     "about": ("Scales the updates to the model's learned numbers at the start of training. Adam, the training "
               "algorithm, measures how prediction error changes with each learned number and uses the history "
               "of those measurements to decide each update. "
               "A larger rate allows larger steps, which can speed learning or make it unstable. A smaller "
               "rate can slow progress. Neither guarantees a better result."),
     "source": _paper("Table 6", "")},
    {"key": "final_learning_rate", "group": "training", "label": "Last learning rate",
     "kind": "float", "default": TRAINING_SETTINGS["final_learning_rate"],
     "min": 1e-5, "max": 1.0, "step": 1e-5,
     "about": ("The intended final scale for updates to the model's learned numbers. The schedule determines "
               "how the rate approaches it; stopping early may prevent it being reached. For the option that "
               "waits for stalled error, this is a lower limit. 'Never changes' ignores it. It cannot exceed "
               "the first rate, and a smaller rate does not guarantee smaller actual updates."),
     "source": _paper("Table 6", "")},
    {"key": "lr_schedule", "group": "training", "label": "How the rate gets there",
     "kind": "choice", "default": "step",
     "choices": [
         {"value": v, "label": label, "source": None}
         for v, label in (
             ("step", "One drop, halfway"),
             ("exponential", "The same factor every pass"),
             ("cosine", "A half cosine"),
             ("plateau", "Drop when held-back error stalls"),
             ("constant", "Never changes"),
         )
     ],
     "about": ("Controls how the update scale changes between passes."
               "\nOne drop, halfway: switch from first to last rate halfway through."
               "\nThe same factor every pass: multiply by a fixed factor, reaching the last rate."
               "\nA half cosine: decrease smoothly, slowest near both ends."
               "\nDrop when held-back error stalls: cut to one tenth after error stalls, never below the last "
               "rate; requires held-back engines."
               "\nNever changes: retain the first rate."
               "\nOne-pass runs use the first rate. Early stopping can prevent reaching the last rate."),
     "source": _choice("A-17", "")},
    # -- repeats -----------------------------------------------------------
    {"key": "seed", "group": "training", "label": "First seed", "kind": "int",
     "default": 42, "min": 0, "max": 2**31 - 1,
     "about": ("Sets the starting point for random choices: which engines are held back, the model's initial "
               "learned numbers, and the order of training windows. Reusing it helps repeat a run, but does not "
               "guarantee identical results across computers or software versions. Each further repeat uses "
               "the next whole number. The seed is an identifier, not a quality score."),
     "source": None},
    {"key": "seeds", "group": "training", "label": "Number of repeats", "kind": "int",
     "default": RESULT_SEEDS, "min": 1, "max": 20,
     "about": ("Trains a new model with each successive seed, then reports the mean score and standard "
               "deviation, a measure of the scores' spread. This measures variation across these random "
               "choices, not uncertainty for an individual engine's prediction. The project requires at "
               "least five repeats for a reportable result; fewer are exploratory."),
     "source": _project("")},
    {"key": "fixed_split", "group": "training", "label": "Same held-back engines for every seed",
     "kind": "bool", "default": False,
     "about": ("Draws the held-back engines once, with the first seed, and gives every repeat the same ones; "
               "only the learned numbers' starting values and the windows' order still change. The spread "
               "between repeats then leaves out how much the choice of engines moves the result, and the run's "
               "page warns of it. It matters only when engines are held back."),
     "source": _project("excludes variation from changing the held-back engines; report this choice")},
    {"key": "device", "group": "training", "label": "Device", "kind": "choice",
     "default": "auto",
     "choices": [{"value": v, "label": v, "source": None} for v in ("auto", "cpu", "cuda")],
     "about": ("Decides which chip does the network's arithmetic, the computer's main processor or an NVIDIA "
               "graphics card. The settings and the steps of the run stay the same."
               "\nauto: the graphics card when one can be used, otherwise the processor."
               "\ncpu: always the processor."
               "\ncuda: always the graphics card; the run stops with an error if none can be used."),
     "source": None},
)

_BY_KEY = {spec["key"]: spec for spec in FIELDS}

# The steps of the form, its five tabs, numbered on the page in this order. The "procedure"
# group is left out on purpose: its settings are drawn at the top of Training,
# under the choice of procedure, which is their only control.
GROUPS = (
    ("data", "Data"),
    ("preprocessing", "Preprocessing"),
    ("temporal", "Temporal stage"),
    ("graph", "Graph layers"),
    ("training", "Training config"),
)

# The four model variants of Table 11, as settings: a stored run is set beside
# the row whose settings it has. Taking the LSTM out of the page's pipeline
# gives "w/o LSTM", taking the physics prior out gives "w/o Physics prior".
# "w/o GCN" has no settings: the graph layers are the one part this playground
# keeps, so their box has nothing to take them out with.
ABLATIONS = (
    {"key": "full", "settings": {"temporal": "lstm", "graph": "physics"}},
    {"key": "without_lstm", "settings": {"temporal": "last_cycle", "graph": "physics"}},
    {"key": "without_physics_prior", "settings": {"temporal": "lstm", "graph": "complete"}},
    {"key": "without_gcn", "settings": None},
)

# What the choice of procedure is for, behind the i beside its name.
PROCEDURE_ABOUT = ("Decides whether some training engines are set aside to check the network's progress, "
                   "and so which pass's learned numbers are kept at the end. The engine groups are chosen "
                   "during preprocessing, before calculating rescaling statistics.")

# The two training procedures, and the only control for how many engines are
# held back and whether the best pass's weights are restored. They are one
# choice because it moves what the LSTM is worth (F-16), and the first one in
# the Training group. Neither sets the number of passes, which is the next one.
PROCEDURES = (
    {"key": "paper", "label": "The paper's procedure",
     "about": ("Uses every training-file engine, runs the selected number of complete passes, and keeps the "
               "final learned numbers. No engines are reserved for selecting an earlier pass or stopping "
               "when their prediction error stalls. This follows the paper's stated fixed training length; "
               "the paper does not describe a separate validation group."),
     "source": _paper("Table 6", "fixed training length; other unspecified choices remain documented separately"),
     "settings": {"validation_fraction": 0.0, "early_stopping": False}},
    {"key": "project", "label": "Held back, best pass",
     "about": ("Reserves some training-file engines to check progress after every pass. They do not update "
               "the model's learned numbers. A sufficient reduction in their prediction error saves a new "
               "best version; training stops after too many passes without one and restores that version. "
               "This leaves fewer engines for learning and makes validation part of model selection."),
     "source": _choice("A-06", "validation and early stopping are project choices (assumption A-18); their effect "
                               "on the LSTM comparison is documented separately (finding F-16)."),
     "settings": {"validation_fraction": 0.2, "early_stopping": True}},
)


# Where the "Read more" at the foot of an i leads, opened in a new tab: the part
# of the documentation that explains a setting (by its key), an option
# ("key=value") or an option of
# the Explain tab ("explain=value"). A heading is written as the document has
# it; "entry" leads to an entry of the assumptions log instead.
_PIPELINE, _MODEL, _GRAPHS = "docs/learn/04_the_pipeline.md", "docs/learn/05_the_model.md", "docs/learn/02_graphs.md"
_TEMPORAL, _TRAINING = "src/thesis_gnn_rul_xai/models/temporal/README.md", "src/thesis_gnn_rul_xai/training/README.md"
_READ_MORE: dict[str, tuple[str, str]] = {
    "subset": ("docs/learn/01_the_problem.md", "NASA's benchmark data: C-MAPSS"),
    "procedure": (_TRAINING, "Validation is optional, and what changes without it"),
    "validation_fraction": (_TRAINING, "Validation is optional, and what changes without it"),
    "early_stopping": (_TRAINING, "Validation is optional, and what changes without it"),
    "early_stopping_patience": (_TRAINING, "Validation is optional, and what changes without it"),
    "window_size": (_PIPELINE, "5 — Cut into windows"),
    "window_stride": (_PIPELINE, "5 — Cut into windows"),
    "rul_max": (_PIPELINE, "1 — Cap the labels"),
    "clip_test_labels": (_PIPELINE, "1 — Cap the labels"),
    "smoothing": (_PIPELINE, "2 — Smooth each sensor"),
    "smoothing_alpha": (_PIPELINE, "2 — Smooth each sensor"),
    "normalization": (_PIPELINE, "Rescale each sensor"),
    "temporal": ("docs/learn/09_temporal.md", "Choose a route"),
    "temporal=lstm": ("docs/learn/09_temporal.md", "One LSTM per sensor"),
    "temporal=last_cycle": (_MODEL, "The stage left out by default"),
    "temporal=gru": ("docs/learn/09_temporal.md", "One GRU per sensor"),
    "temporal=mean_window": (_TEMPORAL, "The stages"),
    "temporal=window_summary": (_TEMPORAL, "`window_summary` — three numbers a person could read off a plot"),
    "temporal=raw_window": (_TEMPORAL, "`raw_window` — kept, named, and gated"),
    "temporal_hidden": ("docs/learn/09_temporal.md", "One LSTM per sensor"),
    "temporal_out_features": ("docs/learn/09_temporal.md", "4 — Turn into numbers per sensor"),
    "shared_across_sensors": ("docs/learn/09_temporal.md", "One LSTM for all sensors"),
    "graph": (_GRAPHS, "1 — Sensor graph"),
    "graph=physics": ("src/thesis_gnn_rul_xai/graph/README.md", "Where this graph comes from"),
    "graph=complete": ("entry", "A-19"),
    "graph=correlation": ("src/thesis_gnn_rul_xai/graph/README.md", "The graph built from the data instead"),
    "adjacency_normalization": ("src/thesis_gnn_rul_xai/graph/README.md", "Normalising the matrix before use"),
    "gcn_hidden": (_MODEL, "1 — Graph layer 1"),
    "gcn_bias": (_MODEL, "1 — Graph layer 1"),
    "fc_hidden": (_MODEL, "5 — Fully connected"),
    "epochs": (_MODEL, "Training"),
    "batch_size": (_MODEL, "Training"),
    "learning_rate": (_TRAINING, "The learning-rate schedule"),
    "final_learning_rate": (_TRAINING, "The learning-rate schedule"),
    "lr_schedule": (_TRAINING, "The learning-rate schedule"),
    "seed": ("docs/learn/06_reading_results.md", "One seed is not a result"),
    "seeds": ("docs/learn/06_reading_results.md", "One seed is not a result"),
    "fixed_split": ("docs/learn/06_reading_results.md", "One seed is not a result"),
    "device": (_TRAINING, "What one run does"),
    "explain=shap": ("docs/learn/10_dynamic_shap.md", "3 — Average what each sensor adds"),
    "explain=occlusion": ("docs/learn/10_dynamic_shap.md", "3 — Average what each sensor adds"),
    "explain=walks": ("docs/learn/11_gnn_lrp.md", "1 — Follow the graph calculation"),
    "explain=engine": ("docs/learn/10_dynamic_shap.md", "5 — Repeat along an engine's history"),
    "explain=test": ("docs/learn/03_explainability.md", "Why a number is not enough"),
    "explain=every": ("docs/learn/10_dynamic_shap.md", "5 — Repeat along an engine's history"),
}

# Diagram steps link by their displayed names; a shared picture does not imply
# a shared operation (the last reading and the last hidden state differ).
_STEP_DOCUMENTS = {
    **dict.fromkeys(("Training engines", "Test engines", "Calculate training labels",
                     "Select 14 sensor columns for model inputs"), "docs/learn/01_the_problem.md"),
    **dict.fromkeys(("Cap the labels", "Smooth each sensor", "Hold engines back", "Find operating conditions",
                     "Rescale each sensor", "Cut into windows"), _PIPELINE),
    **dict.fromkeys(("Split by sensor", "One LSTM per sensor", "One LSTM for all sensors",
                     "One GRU per sensor", "One GRU for all sensors", "Keep the last hidden state",
                     "Turn into numbers per sensor", "Keep the last reading", "Average the window",
                     "Last reading", "Average", "Change over the window", "Three numbers per sensor",
                     "Pass every cycle"), "docs/learn/09_temporal.md"),
    **dict.fromkeys(("Sensor graph", "Rescale the graph"), _GRAPHS),
    **dict.fromkeys(("Graph layer 1", "Graph layer 2", "Graph layer 3", "Flatten", "Fully connected",
                     "Predicted life fraction", "Scale to cycles"), _MODEL),
}

_STEP_HEADINGS = {
    "Select 14 sensor columns for model inputs": "3 — Select 14 sensor columns for model inputs",
    "Cap the labels": "1 — Cap the labels",
    "Smooth each sensor": "2 — Smooth each sensor",
    "Hold engines back": "3 — Hold engines back",
    "Cut into windows": "5 — Cut into windows",
    "Sensor graph": "1 — Sensor graph",
    "Rescale the graph": "2 — Rescale the graph",
    "Split by sensor": "1 — Split by sensor",
    "Keep the last hidden state": "3 — Keep the last hidden state",
    "Turn into numbers per sensor": "4 — Turn into numbers per sensor",
    "Last reading": "Keep the last reading",
    "Average": "Average the window",
    "Graph layer 1": "1 — Graph layer 1",
    "Graph layer 2": "2 — Graph layer 2",
    "Graph layer 3": "3 — Graph layer 3",
    "Flatten": "4 — Flatten",
    "Fully connected": "5 — Fully connected",
    "Predicted life fraction": "6 — Predicted life fraction",
    "Scale to cycles": "7 — Scale to cycles",
}
_READ_MORE.update({f"operation={name}": (path, _STEP_HEADINGS.get(name, name))
                   for name, path in _STEP_DOCUMENTS.items()})
_READ_MORE.update({f"temporal={value}": (_STEP_DOCUMENTS[name], name) for value, name in {
    "last_cycle": "Keep the last reading", "mean_window": "Average the window",
    "window_summary": "Three numbers per sensor", "raw_window": "Pass every cycle",
}.items()})


def read_more(key: str) -> str:
    """The site address of the documentation for a setting or option."""
    from ..markup import slugify

    path, heading = _READ_MORE[key]
    return f"#/entry/{heading}" if path == "entry" else f"#/doc/{path}#{slugify(heading)}"


def _with_more(item: Mapping[str, Any], key: str) -> dict[str, Any]:
    """A setting, option or procedure with the address of its Read more, when it has one."""
    return {**item, "more": read_more(key)} if key in _READ_MORE else dict(item)


# -- settings ---------------------------------------------------------------


def _number(spec: Mapping[str, Any], value: Any, integer: bool) -> float | int:
    """One numeric setting, converted and checked against its range."""
    if isinstance(value, bool):
        raise SettingsError(f"{spec['label']} must be a number, got {value!r}.")
    try:
        number = int(value) if integer and float(value) == int(float(value)) else float(value)
    except (TypeError, ValueError):
        raise SettingsError(f"{spec['label']} must be a number, got {value!r}.") from None
    if integer and not isinstance(number, int):
        raise SettingsError(f"{spec['label']} must be a whole number, got {value!r}.")
    if not math.isfinite(number) or not spec["min"] <= number <= spec["max"]:
        label = f"{spec['label']} (Seeds)" if spec["key"] == "seeds" else spec["label"]
        raise SettingsError(
            f"{label} must lie between {spec['min']} and {spec['max']}, got {value!r}."
        )
    return number


def validate_settings(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Every setting, checked and filled from its default. Unknown keys are refused."""
    if not isinstance(payload, Mapping):
        raise SettingsError("The settings must be a JSON object.")
    unknown = set(payload) - set(_BY_KEY)
    if unknown:
        raise SettingsError(f"Unknown setting(s): {sorted(unknown)}.")

    settings: dict[str, Any] = {}
    for spec in FIELDS:
        value = payload.get(spec["key"], spec["default"])
        kind = spec["kind"]
        if kind == "choice":
            allowed = [choice["value"] for choice in spec["choices"]]
            if value not in allowed:
                raise SettingsError(f"{spec['label']} must be one of {allowed}, got {value!r}.")
        elif kind == "bool":
            if not isinstance(value, bool):
                raise SettingsError(f"{spec['label']} must be true or false, got {value!r}.")
        elif kind == "int":
            value = _number(spec, value, integer=True)
        elif kind == "float":
            value = float(_number(spec, value, integer=False))
        elif kind == "widths":
            if not isinstance(value, list | tuple) or len(value) != spec["length"]:
                raise SettingsError(
                    f"{spec['label']} must be {spec['length']} widths: the graph layers "
                    "are held at Table 4's three."
                )
            value = [_number(spec, width, integer=True) for width in value]
        settings[spec["key"]] = value
    return settings


def paper_row_for(
    subset: str, temporal: str | None, graph: str | None
) -> tuple[dict[str, Any] | None, str]:
    """The row of the paper a configuration may be set beside, or why there is none."""
    if graph == "correlation":
        return None, "The paper never builds a graph from correlations, so no row describes one."
    for variant in ABLATIONS:
        if variant["settings"] == {"temporal": temporal, "graph": graph}:
            row = published_row(variant["key"], subset)
            return {"key": row.key, "label": row.label, "source": row.source, "rmse": row.rmse,
                    "score": row.score, "r2": row.r2, "asue": row.asue}, ""
    if temporal is None:
        return None, "The run did not record its temporal stage, so no row can be chosen."
    if temporal == "last_cycle" and graph == "complete":
        return None, "Two parts are removed at once; Table 11 removes one at a time."
    return None, "The paper has no row for this temporal stage."


# The one vocabulary the page uses for temporal stages and explanation methods.
_STAGE_NAMES = {"lstm": "LSTM", "gru": "GRU", "last_cycle": "w/o LSTM",
                "mean_window": "window average", "window_summary": "window summary",
                "raw_window": "whole window"}
_METHOD_NAMES = {"shap": "Exact Shapley values", "occlusion": "Occlusion",
                 "walks": "Walks through the sensor graph"}
