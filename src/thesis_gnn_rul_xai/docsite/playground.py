"""The playground behind the website: checks settings, runs the real scripts, and reads what they stored.

Read README.md in this folder. It explains why every run started here goes
through scripts/train.py and lands in reports/, and what the page shows.
"""

from __future__ import annotations

import calendar
import contextlib
import json
import math
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ..evaluation.published import (
    DATASET_DESCRIPTION,
    FIG_10_RING,
    SHAP_TABLE_12,
    SHAP_TABLE_12_GRAPHS,
    SUBSETS,
    TRAINING_SETTINGS,
    published_row,
)
from ..utils.logging_setup import get_logger

__all__ = [
    "FIELDS",
    "STAGES",
    "JobManager",
    "SettingsError",
    "compare_runs",
    "explanation_detail",
    "explanation_label",
    "job_label",
    "list_explanations",
    "list_runs",
    "options",
    "paper_row_for",
    "preview",
    "run_detail",
    "stage_detail",
    "train_arguments",
    "validate_explain",
    "validate_settings",
]

logger = get_logger(__name__)

# Where the playground keeps the files of the jobs it starts: the settings it
# wrote, the command's printed output and its progress. Inside reports/, which
# is not committed, and hidden so the run listing never mistakes it for a run.
JOBS_FOLDER = Path("reports") / ".playground"

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
     "about": ("Chooses which of NASA's four sets of simulated engines to train and test on. The sets differ in "
               "how many ways the engines are flown (height, speed, throttle) and how many kinds of fault wear "
               "them out. Pick another to see whether the same settings still work when engines are flown or "
               "fail in more than one way."),
     "source": _paper("Table 5", "")},
    # -- procedure: set by the choice of procedure at the top of Training ---
    {"key": "validation_fraction", "group": "procedure",
     "label": "Training engines held back",
     "kind": "float", "default": 0.2, "min": 0.0, "max": 0.5, "step": 0.05,
     "about": ("To measure the network's error after every pass on engines it never learns from, this share of "
               "the training engines is held back: 0.2 is one in five. Each seed picks its own group at random "
               "unless 'Same held-back engines for every seed' is on. More held back means more engines checked "
               "and fewer to learn from."),
     "source": _choice("A-06", "")},
    {"key": "early_stopping", "group": "procedure", "label": "Stop when held-back error stalls",
     "kind": "bool", "default": True,
     "about": ("Ends training before the last pass once the network's error on the held-back engines has gone a "
               "set number of passes in a row without beating its best, saving the time the remaining passes "
               "would take. It only acts when engines are held back; the choice of training procedure switches "
               "it on or off."),
     "source": _choice("A-18", "")},
    {"key": "early_stopping_patience", "group": "procedure",
     "label": "Passes allowed without improvement", "kind": "int",
     "default": 20, "min": 1, "max": 1000,
     "about": ("How many passes in a row the network's error on the held-back engines may go without beating its "
               "best before training stops. Larger gives a slowly improving run more time; smaller stops it "
               "sooner. With 'Drop when held-back error stalls' chosen, the learning rate, the size of each "
               "correction, is lowered after more than half this many."),
     "source": _choice("A-18", "")},
    # -- preprocessing -----------------------------------------------------
    {"key": "window_size", "group": "preprocessing", "label": "Window length (cycles)",
     "kind": "int", "default": 30, "min": 1, "max": 100,
     "about": ("Engines run for different numbers of flight cycles, but the network needs every example the same "
               "size. So each engine's history is cut into windows of this many consecutive cycles, each "
               "labelled with the remaining life at its last cycle. A longer window gives each example more "
               "history, but engines shorter than it are left out."),
     "source": _paper("Table 6", "")},
    {"key": "window_stride", "group": "preprocessing", "label": "Cycles between windows",
     "kind": "int", "default": 1, "min": 1, "max": 30,
     "about": ("Each new window starts this many cycles after the previous one from the same engine. At 1, "
               "neighbouring windows share all but one cycle and each training engine gives every window it can; "
               "a larger step gives fewer, less overlapping windows and quicker passes. Test engines each give "
               "only their last window."),
     "source": _choice("A-05", "")},
    {"key": "rul_max", "group": "preprocessing", "label": "Label ceiling (cycles)",
     "kind": "int", "default": 125, "min": 10, "max": 400,
     "about": ("An engine early in its life is treated as simply healthy: in training, any remaining life above "
               "this ceiling is counted as the ceiling, so the network is never asked to tell a long remaining "
               "life from a longer one. Its prediction can never go above the ceiling either. Lowering it treats "
               "more of each engine's early cycles this way."),
     "source": _choice("A-01", "read off Fig. 7 by calibrating its plateau against its own axis")},
    {"key": "clip_test_labels", "group": "preprocessing", "label": "Cap the test labels too",
     "kind": "bool", "default": True,
     "about": ("Before the network is scored on the test engines, any true remaining life above the label "
               "ceiling is cut down to the ceiling. The network can never predict above the ceiling, so with "
               "this off, life beyond it counts as error the network could never avoid; with it on, that part is "
               "ignored."),
     "source": _choice("A-02", "can only move reported numbers favourably")},
    {"key": "smoothing", "group": "preprocessing", "label": "Smooth each sensor",
     "kind": "bool", "default": True,
     "about": ("Each sensor's readings jump up and down from one cycle to the next. With this on, every reading "
               "is replaced by part of itself plus part of the smoothed value from the cycle before, one engine "
               "at a time and using only earlier cycles, which evens out the jumps. With it off, the readings go "
               "on unchanged."),
     "source": _paper("Section 3.2.3", "exponential smoothing before windowing")},
    {"key": "smoothing_alpha", "group": "preprocessing", "label": "Smoothing weight on the newest reading",
     "kind": "float", "default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
     "about": ("Each smoothed value is this share of the newest reading plus the remaining share of the smoothed "
               "value before it: at 0.5, half and half. At 1 the readings go on unchanged; a smaller weight "
               "evens out more of the jumps, but a real change in a sensor then shows up more slowly."),
     "source": _choice("A-04", "")},
    {"key": "normalization", "group": "preprocessing", "label": "Sensor scaling",
     "kind": "choice", "default": "zscore",
     "choices": [
         {"value": "zscore", "label": "Mean and spread of the training engines", "source": None},
         {"value": "minmax_per_condition", "label": "Range 0 to 1, per operating condition",
          "source": None},
         {"value": "minmax_global", "label": "Range 0 to 1, over all rows", "source": None},
     ],
     "about": ("Sensors use different units, so some read far bigger than others. Rescaling brings them to a "
               "similar size: the network multiplies readings by learned numbers and adds up the results, so "
               "without it the biggest would make up most of each sum they enter."
               "\nMean and spread: take away the average, then divide by how far readings usually stray from it."
               "\nPer operating condition: the lowest training reading becomes 0 and the highest 1, separately for "
               "each mix of height, speed and throttle."
               "\nOver all rows: the same, one range for all cycles."),
     "source": _choice("A-03", "the paper does not describe its scaling (F-10)")},
    # -- temporal stage ----------------------------------------------------
    {"key": "temporal", "group": "temporal", "label": "Temporal stage", "kind": "choice",
     "default": "lstm",
     "choices": [
         {"value": "lstm", "label": "LSTM, one per sensor",
          "about": ("Each sensor's window goes through an LSTM, a network that takes the readings one cycle at a "
                    "time, oldest first. It combines each new reading with numbers carried over from the cycle "
                    "before, in a way learned in training. What it gives out after the last cycle is turned into the "
                    "numbers the graph layers receive."),
          "source": _paper("Section 2.2", "")},
         {"value": "last_cycle", "label": "w/o LSTM: keep only the last cycle",
          "about": ("Each sensor keeps only its most recent reading in the window; the earlier readings are not used, "
                    "and nothing is learned in this step. 'w/o' is short for without: this is the network without "
                    "its LSTM, the part that takes each sensor's window one cycle at a time."),
          "source": _choice("A-07", "")},
         {"value": "gru", "label": "GRU, one per sensor",
          "about": ("A simpler network of the same kind as the LSTM: it also takes each sensor's window one cycle at "
                    "a time, oldest first, but carries one list of numbers from cycle to cycle where the LSTM carries "
                    "two. With the same 'Hidden size' it has fewer learned numbers."),
          "source": _extension("a smaller recurrent cell")},
         {"value": "mean_window", "label": "Average over the window",
          "about": ("Each sensor's readings across the window are replaced by their plain average, so each sensor "
                    "hands on one number that describes the whole window rather than its last cycle. If a sensor has "
                    "been rising steadily, its average sits below its latest reading."),
          "source": _extension("no learned parameters")},
         {"value": "window_summary", "label": "Last reading, average and change over the window",
          "about": ("Each sensor hands on three numbers: its latest reading, its average over the window, and how far "
                    "it rose or fell across the window, read off the straight line that best fits its readings. The "
                    "first graph layer learns how to combine the three. It needs a window of at least 2 cycles."),
          "source": _extension("three numbers per sensor, no learned parameters (A-28)")},
         {"value": "raw_window", "label": "The whole window straight into the graph layer",
          "about": ("Every reading in the window goes on unchanged, so each sensor hands the graph layers as many "
                    "numbers as the window has cycles. Nothing is learned in this step; instead the first graph layer "
                    "learns how much each cycle counts, the same for every sensor."),
          "source": _extension("the arrangement the last-cycle stage replaced")},
     ],
     "about": ("Each sensor reaches this step as a window: its readings over a run of consecutive flight cycles. "
               "This step turns those readings into one or more numbers per sensor, and only those numbers go "
               "on to the graph layers. The options below differ in what those numbers are, and in whether "
               "anything is learned here."),
     "source": None},
    {"key": "temporal_hidden", "group": "temporal", "label": "Hidden size", "kind": "int",
     "default": 32, "min": 1, "max": 512,
     "about": ("Sets how many numbers the LSTM or GRU, the network that takes each sensor's window one cycle at "
               "a time, carries from one cycle to the next; the LSTM carries two lists this long. The list it "
               "gives out after the last cycle is turned into what the graph layers receive. A larger size means "
               "more learned numbers to train."),
     "source": _choice("A-26", "")},
    {"key": "temporal_out_features", "group": "temporal",
     "label": "Numbers handed to the graph per sensor", "kind": "int",
     "default": 1, "min": 1, "max": 64,
     "about": ("What the LSTM or GRU, the network that takes each sensor's window one cycle at a time, gives out "
               "after the last cycle is turned into this many numbers per sensor. Each is made by multiplying "
               "those outputs by learned numbers and adding up the results plus a learned offset, the same for "
               "every sensor. More gives the first graph layer more to combine."),
     "source": _choice("A-26", "1 is the node width Table 4 prints")},
    {"key": "shared_across_sensors", "group": "temporal",
     "label": "One network shared by all sensors", "kind": "bool", "default": False,
     "about": ("With this off, each sensor has its own LSTM or GRU, the network that takes its window one cycle "
               "at a time, with its own learned numbers. With it on, one network takes every sensor's window, "
               "and nothing in its input says which sensor a window came from. This leaves far fewer learned "
               "numbers, but every sensor's readings are handled exactly alike."),
     "source": _extension("")},
    # -- graph layers: the graph they run on, then their sizes -------------
    # The sensor graph is built while the data is prepared, but it is drawn
    # with the layers that use it, where the page's pipeline shows it too.
    {"key": "graph", "group": "graph", "label": "Sensor graph", "kind": "choice",
     "default": "physics",
     "choices": [
         {"value": "physics", "label": "The paper's physics graph",
          "about": ("Joins 41 fixed pairs of sensors, listed in a table typed into the code by hand. For example, the "
                    "two pressures measured at the same spot, where air leaves the high-pressure compressor, are "
                    "joined. Neither the data nor training ever changes which pairs are joined."),
          "source": _paper("Table 3", "")},
         {"value": "complete", "label": "Every sensor joined to every other",
          "about": ("Joins every sensor to every other, so the graph says nothing about which readings belong "
                    "together. Every sensor then receives the same average of all 14, so after the first graph layer "
                    "all 14 carry identical numbers, and 'Graph rescaling' makes no difference."),
          "source": _choice("A-19", "averages every sensor equally, so it measures the cost "
                                    "of averaging as much as the value of the physics (audit item D-2)")},
         {"value": "correlation", "label": "The 41 most correlated sensor pairs",
          "about": ("Joins the 41 sensor pairs whose readings on the training engines move most closely in step, "
                    "whether rising together or one rising as the other falls, instead of a table typed in by hand. "
                    "It has as many pairs as the physics graph, so the two differ only in which sensors are joined."),
          "source": _choice("A-27", "built from each seed's own training engines, so seeds "
                                    "can get slightly different graphs")},
     ],
     "about": ("Decides which sensors pass numbers to which. In each graph layer, every sensor first takes its "
               "own numbers and those of the sensors it is joined to, added up in fixed proportions, so "
               "information travels one join per layer. The joins are fixed before training, so which readings "
               "belong together is set in advance rather than learned."),
     "source": None},
    {"key": "adjacency_normalization", "group": "graph",
     "label": "Graph rescaling", "kind": "choice", "default": "row",
     "choices": [
         {"value": "row", "label": "Each row sums to one", "source": None},
         {"value": "symmetric", "label": "Symmetric, by both degrees", "source": None},
     ],
     "about": ("Each sensor is first joined to itself, so its own numbers are kept. Then each join is divided by "
               "join counts, so a sensor with many joins does not get bigger numbers just because it adds up "
               "more of them."
               "\nEach row sums to one: every sensor takes the plain average of itself and the sensors it is "
               "joined to."
               "\nSymmetric, by both degrees: each join is divided by the square root of the two sensors' join "
               "counts multiplied together, so a join between two well-joined sensors counts for less."),
     "source": _choice("A-21", "the paper describes the step but prints no formula")},
    {"key": "gcn_hidden", "group": "graph", "label": "Graph layer widths", "kind": "widths",
     "default": [32, 64, 32], "min": 1, "max": 512, "length": 3,
     "about": ("How many numbers each sensor has after each of the three graph layers. A layer taking in 32 "
               "numbers per sensor and giving out 64 learns a table of 32 times 64 numbers, used for every "
               "sensor, so a wider layer has more to learn. The last width, times the 14 sensors, is how many "
               "numbers reach the next layer."),
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
     "about": ("Sets how big a correction the network makes to its learned numbers after each group of windows, "
               "at the start of training; that size is called the learning rate. Larger means bigger changes "
               "each time, which can speed learning up or make the error jump about; smaller can give steadier "
               "but slower progress."),
     "source": _paper("Table 6", "")},
    {"key": "final_learning_rate", "group": "training", "label": "Last learning rate",
     "kind": "float", "default": TRAINING_SETTINGS["final_learning_rate"],
     "min": 1e-5, "max": 1.0, "step": 1e-5,
     "about": ("The size of each correction to the learned numbers, called the learning rate, starts at the "
               "first rate and falls towards this one in the way chosen under 'How the rate gets there', so "
               "later passes make finer changes. It must not be larger than the first, or the run fails; 'Never "
               "changes' ignores it."),
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
     "about": ("Chooses how the learning rate, the size of each correction to the learned numbers, changes over "
               "the passes."
               "\nOne drop, halfway: the first rate for half the passes, then the last rate."
               "\nThe same factor every pass: shrinks by the same fraction each pass, ending on the last."
               "\nA half cosine: falls smoothly, slowest near the start and end, ending on the last."
               "\nDrop when held-back error stalls: cut to a tenth each time that error stalls, never below the "
               "last; needs held-back engines, or it never moves."
               "\nNever changes: stays at the first rate."),
     "source": _choice("A-17", "")},
    # -- repeats -----------------------------------------------------------
    {"key": "seed", "group": "repeats", "label": "First seed", "kind": "int",
     "default": 42, "min": 0, "max": 2**31 - 1,
     "about": ("Fixes the run's random draws: which training engines are held back, the starting values of the "
               "learned numbers, and the order in which the windows are handed to the network. The same number "
               "with the same settings makes the same draws again. The number itself means nothing; each further "
               "repeat uses the next one up."),
     "source": None},
    {"key": "seeds", "group": "repeats", "label": "Seeds", "kind": "int",
     "default": RESULT_SEEDS, "min": 1, "max": 20,
     "about": ("Repeats the run this many times, each time training a new network with the next seed, and "
               "reports the average result and how far the repeats usually stray from it. That spread shows how "
               "much any one result owed to chance. Fewer than 5 is flagged as a first look, not a result to "
               "quote."),
     "source": _project("")},
    {"key": "fixed_split", "group": "repeats", "label": "Same held-back engines for every seed",
     "kind": "bool", "default": False,
     "about": ("Draws the held-back engines once, with the first seed, and gives every repeat the same ones; "
               "only the learned numbers' starting values and the windows' order still change. The spread "
               "between repeats then leaves out how much the choice of engines moves the result, and the run's "
               "page warns of it. It matters only when engines are held back."),
     "source": _project("understates the spread; say so wherever the number appears")},
    {"key": "device", "group": "repeats", "label": "Device", "kind": "choice",
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

# The steps of the form, its six tabs, numbered on the page in this order. The "procedure"
# group is left out on purpose: its settings are drawn at the top of Training,
# under the choice of procedure, which is their only control.
GROUPS = (
    ("data", "Data"),
    ("preprocessing", "Preprocessing"),
    ("temporal", "Temporal stage"),
    ("graph", "Graph layers"),
    ("training", "Training"),
    ("repeats", "Repeats"),
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
                   "and so which pass's learned numbers are kept at the end.")

# The two training procedures, and the only control for how many engines are
# held back and whether the best pass's weights are restored. They are one
# choice because it moves what the LSTM is worth (F-16), and the first one in
# the Training group. Neither sets the number of passes, which is the next one.
PROCEDURES = (
    {"key": "paper", "label": "The paper's procedure",
     "about": ("Every training engine is learned from, every pass is run, and the learned numbers from the very "
               "last pass are kept. No engines are held back to measure progress on, so nothing can stop "
               "training early or pick out a better pass."),
     "source": _paper("Table 6", ""),
     "settings": {"validation_fraction": 0.0, "early_stopping": False}},
    {"key": "project", "label": "Held back, best pass",
     "about": ("Some training engines are held back and never learned from. After every pass the network's error "
               "on them is measured; the learned numbers from the pass with the lowest error are kept, and "
               "training stops early once that error has gone a set number of passes in a row without beating "
               "its best. The cost is fewer engines to learn from."),
     "source": _choice("A-06", "a step the paper never describes (A-18). On FD001 it moved what the "
                               "LSTM is worth from 0.77 to 3.19 cycles (F-16)."),
     "settings": {"validation_fraction": 0.2, "early_stopping": True}},
)


# Where the "Read more" at the foot of an i leads, opened in a new tab: the part
# of the documentation that explains a setting (by its key), an option
# ("key=value"), a step of a box's flowchart (by its picture) or an option of
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
    "window_size": (_PIPELINE, "8 — Window"),
    "window_stride": (_PIPELINE, "8 — Window"),
    "rul_max": (_PIPELINE, "4 — Cap the labels"),
    "clip_test_labels": (_PIPELINE, "4 — Cap the labels"),
    "smoothing": (_PIPELINE, "5 — Smooth *(leakage-critical)*"),
    "smoothing_alpha": (_PIPELINE, "5 — Smooth *(leakage-critical)*"),
    "normalization": (_PIPELINE, "7 — Rescale *(leakage-critical)*"),
    "temporal": (_MODEL, "Before Table 4: the stage the paper does not tabulate"),
    "temporal=lstm": (_TEMPORAL, "`lstm` — what the paper actually describes"),
    "temporal=last_cycle": (_MODEL, "The stage left out by default"),
    "temporal=gru": (_TEMPORAL, "The stages"),
    "temporal=mean_window": (_TEMPORAL, "The stages"),
    "temporal=window_summary": (_TEMPORAL, "`window_summary` — three numbers a person could read off a plot"),
    "temporal=raw_window": (_TEMPORAL, "`raw_window` — kept, named, and gated"),
    "temporal_hidden": (_TEMPORAL, "`lstm` — what the paper actually describes"),
    "temporal_out_features": (_TEMPORAL, "`lstm` — what the paper actually describes"),
    "shared_across_sensors": (_TEMPORAL, "`lstm` — what the paper actually describes"),
    "graph": (_GRAPHS, "The graph this project uses"),
    "graph=physics": ("src/thesis_gnn_rul_xai/graph/README.md", "Where this graph comes from"),
    "graph=complete": ("entry", "A-19"),
    "graph=correlation": ("src/thesis_gnn_rul_xai/graph/README.md", "The graph built from the data instead"),
    "adjacency_normalization": ("src/thesis_gnn_rul_xai/graph/README.md", "Normalising the matrix before use"),
    "gcn_hidden": (_GRAPHS, "One graph layer, as a formula"),
    "gcn_bias": (_GRAPHS, "One graph layer, as a formula"),
    "fc_hidden": (_MODEL, "The six layers, and how many numbers each learns"),
    "epochs": (_MODEL, "Training"),
    "batch_size": (_MODEL, "Training"),
    "learning_rate": (_TRAINING, "The learning-rate schedule"),
    "final_learning_rate": (_TRAINING, "The learning-rate schedule"),
    "lr_schedule": (_TRAINING, "The learning-rate schedule"),
    "seed": ("docs/learn/06_reading_results.md", "One seed is not a result"),
    "seeds": ("docs/learn/06_reading_results.md", "One seed is not a result"),
    "fixed_split": ("docs/learn/06_reading_results.md", "One seed is not a result"),
    "device": (_TRAINING, "What one run does"),
    "explain=shap": ("docs/learn/03_explainability.md", "SHAP: the same idea, applied to a model"),
    "explain=occlusion": ("docs/learn/03_explainability.md", 'What "removing a sensor" has to mean'),
    "explain=engine": ("docs/learn/03_explainability.md", "Dynamic: the same calculation at every cycle"),
    "explain=test": ("docs/learn/03_explainability.md", "Why a number is not enough"),
    "explain=every": ("docs/learn/03_explainability.md", "Dynamic: the same calculation at every cycle"),
    "file": ("docs/learn/01_the_problem.md", "NASA's benchmark data: C-MAPSS"),
    "life": ("docs/learn/01_the_problem.md", "What the model is asked to predict"),
    "select": ("docs/learn/01_the_problem.md", "Only 14 of the 21 sensors"),
    "cap": (_PIPELINE, "4 — Cap the labels"),
    "smooth": (_PIPELINE, "5 — Smooth *(leakage-critical)*"),
    "hold": (_PIPELINE, "6 — Split *(leakage-critical)*"),
    "conditions": ("src/thesis_gnn_rul_xai/data/normalization/README.md", "What an operating condition is"),
    "rescale": (_PIPELINE, "7 — Rescale *(leakage-critical)*"),
    "window": (_PIPELINE, "8 — Window"),
    "sets": (_PIPELINE, "8 — Window"),
    "split": (_TEMPORAL, "`lstm` — what the paper actually describes"),
    "recurrent": (_TEMPORAL, "`lstm` — what the paper actually describes"),
    "last": (_TEMPORAL, "The stages"),
    "project": (_TEMPORAL, "`lstm` — what the paper actually describes"),
    "average": (_TEMPORAL, "The stages"),
    "slope": (_TEMPORAL, "`window_summary` — three numbers a person could read off a plot"),
    "stack": (_TEMPORAL, "`window_summary` — three numbers a person could read off a plot"),
    "graph-step": (_GRAPHS, "The graph this project uses"),
    "table": ("src/thesis_gnn_rul_xai/graph/README.md", "Normalising the matrix before use"),
    "gcn": (_GRAPHS, "One graph layer, as a formula"),
    "flatten": (_MODEL, "The six layers, and how many numbers each learns"),
    "dense": (_MODEL, "The six layers, and how many numbers each learns"),
    "sigmoid": (_MODEL, "The last layer, and what it makes impossible"),
    "scale": (_MODEL, "The last layer, and what it makes impossible"),
}


def read_more(key: str) -> str:
    """The site address of the documentation for a setting, an option, a step's picture or an Explain option."""
    from .markup import slugify

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
        raise SettingsError(
            f"{spec['label']} must lie between {spec['min']} and {spec['max']}, got {value!r}."
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


def validate_explain(payload: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """The request to explain one stored run, checked."""
    if not isinstance(payload, Mapping):
        raise SettingsError("The request must be a JSON object.")
    run = _run_directory(root, str(payload.get("run", "")))
    method = payload.get("method", "shap")
    if method not in ("shap", "occlusion"):
        raise SettingsError(f"method must be 'shap' or 'occlusion', got {method!r}.")
    scope = payload.get("scope", "engine")
    if scope not in ("engine", "test"):
        raise SettingsError(f"scope must be 'engine' or 'test', got {scope!r}.")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SettingsError(f"seed must be a whole number, got {seed!r}.")
    if not (run / f"seed_{seed}" / "model.pt").is_file():
        raise SettingsError(
            f"This run saved no weights for seed {seed}. Runs made before 2026-09-14 "
            "saved none; train the configuration again to explain it."
        )
    engine = payload.get("engine")
    if engine is not None and (isinstance(engine, bool) or not isinstance(engine, int)):
        raise SettingsError(f"engine must be a whole number or empty, got {engine!r}.")
    every = payload.get("every", 1)
    if isinstance(every, bool) or not isinstance(every, int) or not 1 <= every <= 100:
        raise SettingsError(f"every must be a whole number from 1 to 100, got {every!r}.")
    return {"run": run.name, "method": method, "scope": scope, "seed": seed,
            "engine": engine, "every": every}


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


def caveats(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What a reader must be told before setting this configuration beside the paper."""
    notes: list[dict[str, Any]] = []

    # "run": it qualifies a stored run's numbers, so the run's page lists it.
    # "design": it flags a conflict between settings the reader may not have
    # open, so the Design tab shows it beside the Train button as well.
    def add(level: str, ref: str, text: str, where: tuple[str, ...] = ("run",)) -> None:
        notes.append({"level": level, "ref": ref, "text": text, "where": list(where)})

    seeds = int(settings.get("seeds") or 0)
    if seeds < RESULT_SEEDS:
        add("warn", "", f"{seeds} seed{'' if seeds == 1 else 's'} is exploratory: quote "
            "nothing from it. A result is a mean over at least five seeds, with its spread.",
            ("design", "run"))
    if settings.get("temporal") in ("lstm", "gru"):
        add("info", "A-26", "The hidden size and the numbers handed to the graph per "
            "sensor are chosen here; the paper states neither.")
    if settings.get("temporal") == "last_cycle":
        add("info", "D-6", "Only each window's last cycle reaches the graph layers, which "
            "the paper's w/o LSTM model may not have done.")
    if settings.get("shared_across_sensors") and settings.get("temporal") in ("lstm", "gru"):
        add("info", "Section 2.2", "One shared network is not the paper's stage, which "
            "gives each sensor its own.")
    if list(settings.get("gcn_hidden") or [32, 64, 32]) != [32, 64, 32] or \
            int(settings.get("fc_hidden") or 128) != 128:
        add("info", "Table 4", "The layer widths differ from Table 4.")
    if float(settings.get("validation_fraction") or 0.0) > 0:
        add("warn", "F-16", "The paper never describes restoring the best pass; on FD001 "
            "it moved what the LSTM is worth from 0.77 to 3.19 cycles.")
    elif settings.get("lr_schedule") == "plateau":
        add("warn", "A-17", "The plateau schedule needs held-back engines; with none the "
            "rate never leaves its first value.", ("design", "run"))
    if settings.get("clip_test_labels", True):
        add("info", "A-02", "Test labels are capped too, which can only flatter the numbers.")
    if settings.get("graph") == "complete":
        # Every sensor of the complete graph has the same number of neighbours,
        # so both rescalings give the same matrix, 1/14 everywhere.
        add("info", "D-2", "The complete graph averages every sensor equally, so it "
            "measures the cost of averaging as much as the value of the physics.")
    # Only with engines held back, each seed drawing its own, do seeds get different graphs.
    varies = float(settings.get("validation_fraction") or 0.0) > 0 and not settings.get("fixed_split")
    if settings.get("graph") == "correlation" and varies:
        add("info", "A-27", "The graph is rebuilt from each seed's own training engines, "
            "so two seeds can be given slightly different graphs.")
    subset, window = settings.get("subset"), settings.get("window_size")
    if subset in SUBSETS and isinstance(window, int) and not isinstance(window, bool):
        counted = dataset_facts(str(subset), window)["counted"]
        if counted and counted["left_out"] > 0:
            one = counted["left_out"] == 1
            add("warn", "A-23", f"{counted['left_out']} of {counted['test_engines']} test "
                f"engines {'is' if one else 'are'} shorter than the window and "
                f"{'was' if one else 'were'} left out, so the score cannot be set beside "
                "Table 7.")
    if settings.get("fixed_split"):
        add("info", "", "The same held-back engines for every seed understates the spread.")
    return notes


def _test_lengths(subset: str) -> tuple[int, ...]:
    """How many cycles each test engine of a sub-dataset recorded, or nothing while its file is missing."""
    from ..utils.paths import raw_data_dir

    return _read_test_lengths(subset) if (raw_data_dir() / f"test_{subset}.txt").is_file() else ()


@lru_cache(maxsize=8)
def _read_test_lengths(subset: str) -> tuple[int, ...]:
    """How many cycles each test engine of a sub-dataset recorded, from the real file."""
    from ..data.loading import read_cmapss_frame
    from ..data.sensors import CYCLE_COL, UNIT_COL
    from ..utils.paths import raw_data_dir

    frame = read_cmapss_frame(raw_data_dir() / f"test_{subset}.txt")
    return tuple(int(n) for n in frame.groupby(UNIT_COL)[CYCLE_COL].max().tolist())


def dataset_facts(subset: str, window_size: int) -> dict[str, Any]:
    """What Table 5 says about a sub-dataset, and how many test engines this window leaves out."""
    facts: dict[str, Any] = {"table5": DATASET_DESCRIPTION[subset], "counted": None}
    lengths = _test_lengths(subset)
    if lengths:
        # An engine shorter than the window yields no window at all (A-23).
        facts["counted"] = {
            "test_engines": len(lengths),
            "left_out": sum(1 for n in lengths if n < window_size),
        }
    return facts


def preview(settings: Mapping[str, Any]) -> dict[str, Any]:
    """What the Design tab draws for these settings: the data, the cautions, the network built for real."""
    answer: dict[str, Any] = {
        "dataset": dataset_facts(settings["subset"], int(settings["window_size"])),
        "design_cautions": [note for note in caveats(settings) if "design" in note["where"]],
        "torch": True,
    }
    try:
        model = _network(settings)
    except _NoTorch:
        answer["torch"] = False
        return answer
    except ValueError as refused:
        answer["network_error"] = str(refused)
        return answer
    answer["layers"] = model.layer_summary()
    answer["parameters"] = model.count_parameters()
    return answer


class _NoTorch(Exception):
    """PyTorch is missing, so no network can be built."""


def _network(settings: Mapping[str, Any], matrix: Any = None) -> Any:
    """The network these settings build, for real, on the given sensor graph or one that stands in for it."""
    try:
        from ..graph.adjacency import (
            build_fully_connected_adjacency,
            build_physics_adjacency,
            normalize_adjacency,
        )
        from ..models import ModelConfig, build_model
    except ImportError as missing:
        # Only a missing PyTorch is expected here; any other failed import is a fault.
        if not (missing.name or "").startswith("torch"):
            raise
        raise _NoTorch() from None

    # The correlation graph needs the training engines, which a preview does not
    # load. The layers and their parameter counts do not depend on which edges
    # the graph has, so the paper's graph stands in for it when none is given.
    if matrix is None:
        matrix, symbols = build_physics_adjacency()
        if settings["graph"] == "complete":
            matrix, _ = build_fully_connected_adjacency(len(symbols))
    return build_model(
        normalize_adjacency(matrix, method=settings["adjacency_normalization"]),
        _model_config(settings, ModelConfig),
        window_size=int(settings["window_size"]),
        rul_max=int(settings["rul_max"]),
    )


def _model_config(settings: Mapping[str, Any], factory: Any) -> Any:
    """The ModelConfig the settings describe."""
    return factory(
        temporal=settings["temporal"],
        temporal_hidden=int(settings["temporal_hidden"]),
        temporal_out_features=int(settings["temporal_out_features"]),
        shared_across_sensors=bool(settings["shared_across_sensors"]),
        gcn_hidden=tuple(settings["gcn_hidden"]),
        fc_hidden=int(settings["fc_hidden"]),
        gcn_bias=bool(settings["gcn_bias"]),
    )


# -- one box of the pipeline, as its own flowchart -----------------------------
# README.md in this folder says what each box's flowchart shows and why. Every
# number in them is read from the files, from a real preprocessing run or from
# the network built for real; the words say what the code does, in its order.
# A step's tag says where its operation comes from; a value on screen that is
# not the default says so in words, with where the default came from, so a
# value the paper never printed never carries the paper's tag.

STAGES = ("data", "preprocessing", "temporal", "graph")

_SOURCE_WORDS = {"paper": "the paper's {ref}", "assumption": "assumption {ref}",
                 "project": "a rule of this repository", "extension": "an option the paper does not have"}

# One preprocessing run at a time: a second request for the same settings then
# finds the first one's answer kept, instead of running it again beside it.
_PREPARE_LOCK = threading.Lock()


class _Undescribable(Exception):
    """A box whose flowchart cannot be drawn for these settings, with the reason in words."""


def stage_detail(stage: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    """One box of the page's pipeline as its own flowchart: its steps, in the order the code runs them."""
    builders = {"data": _data_steps, "preprocessing": _preprocessing_steps,
                "temporal": _temporal_steps, "graph": _graph_steps}
    if stage not in builders:
        raise SettingsError(f"stage must be one of {list(STAGES)}, got {stage!r}.")
    from ..data.loading import DOWNLOAD_URL

    answer: dict[str, Any] = {"stage": stage, "steps": []}
    needs_files = stage in ("data", "preprocessing") or (stage == "graph" and settings["graph"] == "correlation")
    if needs_files and not _files_present(settings["subset"]):
        answer["note"] = (f"The NASA files for {settings['subset']} are not in data/raw/, so nothing here is "
                          f"counted. They can be downloaded from {DOWNLOAD_URL}")
    try:
        answer["steps"] = builders[stage](settings)
    except _Undescribable as reason:
        answer["error"] = str(reason)
    for item in answer["steps"]:
        for step in item.get("steps", [item]):
            key = "graph-step" if step.get("picture") == "graph" else step.get("picture", "")
            if key in _READ_MORE:
                step["more"] = read_more(key)
    return answer


def _step(name: str, does: str, *, parts: list[tuple[Any, ...]] | None = None, shape: str = "",
          learned: int | None = None, source: Mapping[str, str] | None = None,
          picture: str = "") -> dict[str, Any]:
    """One step of a box's flowchart: its name, the values that decide it, what it hands on and its picture."""
    return {"name": name, "does": " ".join(does.split()), "shape": shape, "learned": learned,
            "parts": [{"text": part[0], "setting": part[1], "index": part[2] if len(part) > 2 else None}
                      for part in (parts or [])],
            "source": dict(source) if source else None, "picture": picture}


def _edge(name: str, shape: str = "", icon: str = "") -> dict[str, Any]:
    """What enters or leaves a box, drawn as words beside the picture of the box it comes from or goes to."""
    return {"kind": "edge", "name": name, "shape": shape, "icon": icon}


def _together(*steps: dict[str, Any], shape: str = "") -> dict[str, Any]:
    """Steps that work side by side on the same input, drawn in one row, and what the row hands on."""
    return {"kind": "group", "steps": list(steps), "shape": shape}


def _branch(*steps: dict[str, Any]) -> dict[str, Any]:
    """Steps that make a second input for the step after them, drawn beside the main line."""
    return {"kind": "branch", "steps": list(steps)}


def _networks(stage: Any, kind: str, window: int, shared: bool) -> dict[str, Any]:
    """The recurrent networks of a temporal stage, for drawing them one by one and one from inside."""
    from ..data.sensors import PAPER_SENSOR_SYMBOLS

    cell = stage.cells[0]
    gates = 4 if kind == "lstm" else 3
    learned = sum(p.numel() for p in cell.parameters())
    return {"kind": kind, "shared": shared, "sensors": list(PAPER_SENSOR_SYMBOLS), "window": window,
            "hidden": int(cell.hidden_size), "inputs": int(cell.input_size), "learned_each": int(learned),
            "per_gate": int(learned // gates), "gates": gates}


def _cite(ref: str) -> dict[str, str]:
    """A step's tag: the part of the paper its operation comes from."""
    return {"kind": "paper", "ref": ref, "text": ""}


def _count(n: int) -> str:
    """A whole number with a space between thousands, as the page prints it: "20 631"."""
    return f"{int(n):,}".replace(",", " ")


def _plural(n: int, word: str) -> str:
    """A count and its noun, singular or plural: "1 cycle", "30 cycles"."""
    return f"{_count(n)} {word}{'' if n == 1 else 's'}"


def _default_words(key: str) -> str:
    """A setting's default, in the form's words: "30 cycles", "on", "20%", "Each row sums to one"."""
    spec = _BY_KEY[key]
    value = spec["default"]
    if spec["kind"] == "choice":
        return f'"{_label_of(key, value)}"'
    if key == "validation_fraction":
        return f"{value:.0%}"
    if spec["kind"] == "bool":
        return "on" if value else "off"
    if spec["kind"] == "widths":
        return ", ".join(str(w) for w in value[:-1]) + f" and {value[-1]}"
    unit = _unit_of(spec["label"])
    return f"{value:g} {unit}".strip() if isinstance(value, float) else f"{value} {unit}".strip()


def _unit_of(label: str) -> str:
    """The unit a setting's label ends with, "(cycles)", or nothing."""
    return label[label.rindex("(") + 1:-1] if label.endswith(")") and "(" in label else ""


def _decided(key: str, value: Any) -> tuple[dict[str, str] | None, str]:
    """Where a value on screen comes from: its tag, or none and a sentence saying it is not the default."""
    spec = _BY_KEY[key]
    for choice in spec.get("choices", []):
        if choice["value"] == value and choice["source"]:
            return choice["source"], ""
    source = spec["source"]
    # A paper's list of options, or an option the paper does not have, describes every value alike.
    if value == spec["default"] or (source and (source["kind"] == "extension" or (
            spec["kind"] == "choice" and source["kind"] == "paper"))):
        return source, ""
    where = f" ({_SOURCE_WORDS[source['kind']].format(ref=source['ref'])})" if source else ""
    return None, f'"{spec["label"].split(" (")[0]}" was changed from its default, {_default_words(key)}{where}.'


def _label_of(key: str, value: Any) -> str:
    """An option's words, as the form shows them."""
    for choice in _BY_KEY[key].get("choices", []):
        if choice["value"] == value:
            return str(choice["label"])
    return str(value)


def _files_present(subset: str) -> bool:
    """Whether all three NASA files of a sub-dataset are in data/raw/."""
    from ..utils.paths import raw_data_dir

    return all((raw_data_dir() / f"{kind}_{subset}.txt").is_file() for kind in ("train", "test", "RUL"))


@lru_cache(maxsize=4)
def _file_counts(subset: str) -> dict[str, Any]:
    """Rows and engines in one sub-dataset's three files, and the kept sensors that barely vary, counted from them."""
    from ..data.loading import read_cmapss_frame, read_rul_file
    from ..data.sensors import (
        PAPER_SENSOR_SYMBOLS,
        UNIT_COL,
        detect_uninformative_sensors,
    )
    from ..utils.paths import raw_data_dir

    folder = raw_data_dir()
    train, test = read_cmapss_frame(folder / f"train_{subset}.txt"), read_cmapss_frame(folder / f"test_{subset}.txt")
    check = detect_uninformative_sensors(train)
    return {"train_rows": len(train), "train_engines": int(train[UNIT_COL].nunique()),
            "test_rows": len(test), "test_engines": int(test[UNIT_COL].nunique()),
            "labels": len(read_rul_file(folder / f"RUL_{subset}.txt")),
            "flat": [s for s in PAPER_SENSOR_SYMBOLS if s in check["constant"] + check["low_resolution"]]}


def _file_facts(subset: str) -> dict[str, Any] | None:
    """The counts above, or None while the files are missing; a missing file is never remembered."""
    return _file_counts(subset) if _files_present(subset) else None


@lru_cache(maxsize=16)
def _prepared(key: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """What a flowchart reads from a real preprocessing run: its counts and its sensor graph, not its windows."""
    import numpy as np

    from ..config import PreprocessingConfig
    from ..data.pipeline import run_preprocessing

    run = run_preprocessing(PreprocessingConfig(**dict(key)))
    return {"train": len(run.train), "validation": len(run.validation), "test": len(run.test),
            "kept": len(np.unique(run.train.units)), "held": len(np.unique(run.validation.units)),
            "conditions": int(run.condition_encoder.n_detected), "adjacency": run.adjacency_raw.copy()}


def _prepared_for(settings: Mapping[str, Any]) -> dict[str, Any] | None:
    """The preprocessing run these settings describe, or None when the NASA files are missing."""
    if not _files_present(settings["subset"]):
        return None
    key = preprocessing_settings(settings)
    # The rescaling of the graph, and the graph unless it is built from the
    # data, change nothing a flowchart reads from the run.
    del key["adjacency_normalization"]
    if key["graph"] != "correlation":
        del key["graph"]
    with _PREPARE_LOCK:
        return _prepared(tuple(sorted(key.items())))


def _data_steps(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Box 1: the three files of a sub-dataset, their labels, and the sensors kept from them."""
    from ..data.sensors import PAPER_SENSOR_SYMBOLS

    subset = settings["subset"]
    table5 = DATASET_DESCRIPTION[subset]
    facts = _file_facts(subset)
    tag, _ = _decided("subset", subset)

    def rows(key: str, columns: int) -> str:
        """A table's shape as the files give it, or nothing without them."""
        return f"({_count(facts[key])}, {columns})" if facts else ""

    flat = facts["flat"] if facts else []
    check = ""
    if flat:
        check = (f" A check the code runs every time finds that {' and '.join(flat)} barely "
                 f"var{'ies' if len(flat) == 1 else 'y'} in {subset}; it changes nothing, so "
                 f"{'it is' if len(flat) == 1 else 'they are'} kept"
                 + (" (finding F-01)." if "P15" in flat else "."))
    return [
        _together(
            _step("Training engines",
                  f"train_{subset}.txt holds one row per engine per flight cycle, 26 columns: the engine's number, "
                  "the cycle, 3 operating settings and 21 sensor readings, recorded until the engine fails.",
                  parts=[(f"{_plural(table5['train_engines'], 'engine')} · {_plural(table5['conditions'], 'condition')}"
                          f" · {_plural(table5['fault_modes'], 'fault mode')}", "subset")],
                  shape=rows("train_rows", 26), source=tag, picture="file"),
            _step("Test engines",
                  f"test_{subset}.txt has the same columns, but each record stops some cycles before the engine "
                  "fails. How many cycles were left is what the network has to predict.",
                  parts=[(_plural(table5["test_engines"], "engine"), "subset")],
                  shape=rows("test_rows", 26), source=tag, picture="file"),
            _step("Their remaining life",
                  f"RUL_{subset}.txt holds one number per line: the remaining life of test engine i at its last "
                  "recorded cycle. The file names no engines, so line i belongs to engine i by its position.",
                  parts=[("one per test engine", None)],
                  shape=f"({_count(facts['labels'])},)" if facts else "", picture="file"),
            shape="3 tables",
        ),
        _step("Label every row",
              "Each row's remaining life is its engine's last recorded cycle minus this cycle, plus, for a test "
              "engine, its number from the label file. Nothing is capped yet: the ceiling comes in preprocessing.",
              parts=[("cycles left until failure", None)], shape=rows("train_rows", 27),
              source=_cite("Section 3.2.1"), picture="life"),
        _step("Keep 14 of the 21 sensors",
              "The paper's Table 3 keeps " + ", ".join(PAPER_SENSOR_SYMBOLS) + ", in that order, and the sensor "
              "graph depends on that order. The paper removes seven sensors and names six; the seventh, NRc, is "
              "this project's reading (assumption A-10). The other seven sensor columns stay in the table and are "
              "never read." + check,
              parts=[("Table 3's sensors, in its order", None)],
              shape="14 of 27 columns" if facts else "", source=_cite("Table 3"), picture="select"),
        _edge("To preprocessing", icon="preprocessing"),
    ]


def _preprocessing_steps(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Box 2: from labelled rows to windows, in the order the pipeline runs its steps."""
    run = _prepared_for(settings)
    subset, seed = settings["subset"], settings["seed"]
    ceiling, window, stride = int(settings["rul_max"]), int(settings["window_size"]), int(settings["window_stride"])
    fraction = float(settings["validation_fraction"])
    several = DATASET_DESCRIPTION[subset]["conditions"] > 1
    # Until the windows are cut, what passes from step to step is the training
    # file's table: its rows, and 27 columns, one more once the operating
    # condition is added. The test file's table goes the same way beside it.
    facts = _file_facts(subset)
    rows = _count(facts["train_rows"]) if facts else "rows"
    table, table_more = f"({rows}, 27)", f"({rows}, 28)"
    steps = [_edge("Labelled rows from the data", table, icon="data")]

    _, ceiling_note = _decided("rul_max", ceiling)
    capped = bool(settings["clip_test_labels"])
    _, test_note = _decided("clip_test_labels", capped)
    steps.append(_step(
        "Cap the labels",
        "The paper takes remaining life to stay level while an engine is healthy and to fall in a straight line "
        f"after that, so every label above the ceiling becomes the ceiling: min(life, {ceiling}). "
        + (ceiling_note or "The paper gives the ceiling no number; 125 was read off its Fig. 7 (assumption A-01).")
        + " The network's output is scaled to the same ceiling, so it can never predict more (assumption A-13). "
        + "The paper's Fig. 1 caps after smoothing; the two touch different columns, so the order changes nothing. "
        + ("The test labels are capped too, which can only flatter the scores (assumption A-02)." if capped else
           "The test labels keep their true values.") + (f" {test_note}" if test_note else ""),
        parts=[(f"at {_plural(ceiling, 'cycle')}", "rul_max"),
               ("test labels too" if capped else "not the test labels", "clip_test_labels")],
        shape=table, source=_cite("Eq. 19"), picture="cap"))

    if settings["smoothing"]:
        alpha = float(settings["smoothing_alpha"])
        _, alpha_note = _decided("smoothing_alpha", alpha)
        blends = alpha < 1
        steps.append(_step(
            "Smooth each sensor",
            (f"Runs down each engine's history: smoothed = {alpha:g} x reading + {1 - alpha:g} x the smoothed "
             "value before it, on the 14 kept sensors in their own units. It uses only the same engine's earlier "
             "readings, so nothing from a later cycle reaches an earlier one." if blends else
             "With all the weight on the newest reading, each reading passes through unchanged.")
            + " It comes before the rescaling, the order the paper's Fig. 1 prints. "
            + (alpha_note or "The weight 0.5 is this project's reading of a sentence that attaches it to the "
                             "wrong symbol (assumption A-04).")
            + (f" On {subset} the operating condition changes from one cycle to the next, so smoothing in raw "
               "units blends readings taken under different conditions; the log measures what that costs "
               "(entry A-08)." if several and blends else ""),
            parts=[(f"weight {alpha:g} on the newest reading", "smoothing_alpha")], shape=table,
            source=_cite("Eq. 18"), picture="smooth"))
    else:
        _, off_note = _decided("smoothing", False)
        steps.append(_step(
            "Smooth each sensor", "Switched off: the readings go on as recorded. " + off_note,
            parts=[("off", "smoothing")], shape=table, picture="smooth"))

    if fraction > 0:
        tag, note = _decided("validation_fraction", fraction)
        count = f"{run['held']} of {run['held'] + run['kept']} engines" if run else f"{fraction:.0%} of the engines"
        steps.append(_step(
            "Hold engines back",
            f"Before anything is measured from the data, {fraction:.0%} of the training engines are set aside, "
            f"here drawn with seed {seed}; each seed of a run draws its own unless the same held-back engines are "
            "kept for every seed. After every pass the network is scored on them, and the pass that scores best is "
            "the one whose weights are kept" + (", and training stops once they stop improving"
                                                if settings["early_stopping"] else "")
            + ". Whole engines go, never single windows, because neighbouring windows share most of their cycles. "
            "Setting them aside before the rescaling is measured reverses the order the paper's Fig. 1 prints, "
            "where the rescaling comes before the sets are formed (assumption A-25). " + note,
            parts=[(count, "validation_fraction")], shape=table, source=tag, picture="hold"))
    else:
        _, note = _decided("validation_fraction", fraction)
        steps.append(_step(
            "Hold engines back",
            "Nothing is held back, as in the paper's procedure, which describes no held-out set: every training "
            "engine teaches the network, every pass is run and the final weights are kept. " + note,
            parts=[("none", "validation_fraction")], shape=table, picture="hold"))

    method = settings["normalization"]
    conditions = run["conditions"] if run else None
    steps.append(_step(
        "Find operating conditions",
        "Rounds the three operating settings of the training engines' rows to whole numbers and counts the "
        "combinations, at most 6. With more than one, a clustering method (k-means) puts each row with the "
        f"nearest of that many centres, found from the unrounded settings and started from seed {seed}. " + ("The rescaling below uses the groups." if method == "minmax_per_condition"
                                          else "The rescaling chosen below does not use them."),
        parts=[(f"{_plural(conditions, 'condition')} found" if conditions else "", None)], shape=table_more,
        picture="conditions"))

    tag, note = _decided("normalization", method)
    formula = "(reading - mean) / spread" if method == "zscore" else "(reading - smallest) / (largest - smallest)"
    where = {"zscore": "one mean and one spread per sensor",
             "minmax_per_condition": "one range per sensor within each operating condition",
             "minmax_global": "one range per sensor over all rows"}[method]
    applied = "every engine, the held-back and test engines included" if fraction > 0 else "every engine"
    pooled = (f" Here the {DATASET_DESCRIPTION[subset]['conditions']} operating conditions share one scale; the "
              "range per condition is the choice the log recommends for this sub-dataset (assumption A-03)."
              if several and method != "minmax_per_condition" else "")
    steps.append(_step(
        "Rescale each sensor",
        f"{formula}, with {where}, measured on the training engines only and then applied to {applied}. "
        + note + pooled,
        parts=[(_label_of("normalization", method).lower(), "normalization")], shape=table_more, source=tag,
        picture="rescale"))

    tag, note = _decided("window_size", window)
    _, stride_note = _decided("window_stride", stride)
    every = "every cycle" if stride == 1 else f"every {stride} cycles"
    left_out = (dataset_facts(subset, window)["counted"] or {}).get("left_out", 0)
    steps.append(_step(
        "Cut into windows",
        f"A run of {_plural(window, 'consecutive cycle')} of a training-file engine, starting {every}, becomes one "
        f"example, labelled with the remaining life at its last cycle and laid out as 14 sensors by "
        f"{_plural(window, 'cycle')}. " + note + " "
        + (stride_note or "One cycle apart is this project's choice (assumption A-05).")
        + " A test engine gives only the window that ends at its last recorded cycle; the spacing does not apply "
        "to it." + (f" {_plural(left_out, 'test engine')} shorter than {_plural(window, 'cycle')} "
                    f"{'is' if left_out == 1 else 'are'} left out (assumption A-23)." if left_out else ""),
        parts=[(_plural(window, "cycle"), "window_size"), (f"a new one {every}", "window_stride")],
        shape=f"({_count(run['train'] + run['validation'])}, 14, {window})" if run else f"(windows, 14, {window})",
        source=tag, picture="window"))

    if run:
        if fraction > 0:
            steps.append(_step(
                "Three sets of windows",
                f"The windows are divided by engine as chosen above; the counts are seed {seed}'s. Only the "
                "training windows teach the network, the held-back ones score it after every pass, and the test "
                "windows are scored once, at the end.",
                parts=[(f"{_count(run['train'])} training · {_count(run['validation'])} held back · "
                        f"{_count(run['test'])} test", None)],
                shape=f"({_count(run['train'])}, 14, {window})", picture="sets"))
        else:
            steps.append(_step(
                "Two sets of windows",
                "Every training-file window teaches the network, and the test windows are scored once, at the end.",
                parts=[(f"{_count(run['train'])} training · {_count(run['test'])} test", None)],
                shape=f"({_count(run['train'])}, 14, {window})", picture="sets"))
    steps.append(_edge("To the temporal stage, B windows at a time", f"(B, 14, {window})", icon="temporal"))
    return steps


def _built(settings: Mapping[str, Any], matrix: Any = None) -> Any:
    """The network for a box's flowchart, or the reason it cannot be built."""
    if settings["temporal"] == "window_summary" and int(settings["window_size"]) < 2:
        raise _Undescribable("The window summary measures the change over the window, so the window length must "
                             "be at least 2 cycles.")
    try:
        return _network(settings, matrix)
    except _NoTorch:
        raise _Undescribable("PyTorch is not installed, so the network cannot be built to describe it.") from None
    except ValueError as refused:
        raise _Undescribable(str(refused)) from None


def _temporal_steps(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Box 3: how each sensor's window becomes the numbers the graph layers receive."""
    model = _built(settings)
    stage, kind = model.temporal, settings["temporal"]
    window, nodes, k = int(settings["window_size"]), model.n_nodes, int(model.in_features)
    readings = _plural(window, "reading")
    chosen, _ = _decided("temporal", kind)
    steps = [_edge("Windows from preprocessing, B at a time", f"(B, {nodes}, {window})", icon="preprocessing")]
    if kind in ("lstm", "gru"):
        hidden, shared = int(settings["temporal_hidden"]), bool(settings["shared_across_sensors"])
        _, hidden_note = _decided("temporal_hidden", hidden)
        cells = sum(p.numel() for p in stage.cells.parameters())
        each = cells // len(stage.cells)
        name = "LSTM" if kind == "lstm" else "GRU"
        rows = f"B x {nodes}" if shared else "B"
        inside = ("its input, forget and output gates decide what to keep in a cell state and what to pass on"
                  if kind == "lstm" else
                  "its reset and update gates decide how much of what it carries to renew; there is no separate "
                  "cell state")
        steps.append(_step(
            "Split by sensor",
            (f"Each window's {nodes} sensors become {nodes} sequences of one number per cycle, all read by the same "
             "network." if shared else
             f"Each sensor's {readings} {'becomes' if window == 1 else 'become'} a sequence of its own, one "
             "number per cycle. Inside this stage no "
             "sensor sees another sensor's readings."),
            parts=[("one sequence per window and sensor" if shared else f"{nodes} sequences of {readings}", None)],
            shape=f"({rows}, {window}, 1)" if shared else f"{nodes} x (B, {window}, 1)",
            source=_decided("shared_across_sensors", True)[0] if shared else _cite("Section 2.2"),
            picture="split"))
        steps.append(_step(
            f"One {name} for all sensors" if shared else f"One {name} per sensor",
            ("A single network reads every sensor's window in turn, with nothing telling it which sensor it is "
             "reading; the paper gives each sensor its own (Section 2.2)." if shared else
             "Each sensor's own network reads its readings oldest first.")
            + f" At every cycle {inside}. It carries {_plural(hidden, 'number')} from one cycle to the next, its "
            f"hidden state, which {'starts' if hidden == 1 else 'start'} at zero for every window. "
            + (hidden_note or "That width is this project's choice (assumption A-26).")
            + f" {_count(each)} learned numbers per network.",
            parts=[(f"hidden {hidden}", "temporal_hidden"),
                   ("shared" if shared else f"{len(stage.cells)} networks", "shared_across_sensors")],
            shape=f"({rows}, {window}, {hidden})" + ("" if shared else " each"), learned=cells,
            source=_decided("shared_across_sensors", True)[0] if shared else chosen, picture="recurrent"))
        steps[-1]["networks"] = _networks(stage, kind, window, shared)
        dropped = ([f"the {_plural(window - 1, 'earlier one')}"] if window > 1 else []) + (
            ["the cell state"] if kind == "lstm" else [])
        one = len(dropped) == 1 and (kind == "lstm" or window == 2)
        steps.append(_step(
            "Keep the last hidden state",
            "Only the hidden state after the most recent cycle leaves the network"
            + (f"; {' and '.join(dropped)} {'is' if one else 'are'} dropped." if dropped else "."),
            parts=[(f"{_plural(hidden, 'number')} per sensor", None)],
            shape=f"(B x {nodes}, {hidden})" if shared else f"(B, {nodes}, {hidden})",
            source=_cite("Eq. 7"), picture="last"))
        projection = sum(p.numel() for p in stage.project.parameters())
        _, out_note = _decided("temporal_out_features", k)
        steps.append(_step(
            f"Turn into {_plural(k, 'number')} per sensor",
            (f"A weighted sum of each sensor's {_plural(hidden, 'number')} plus an offset, with the same weights "
             f"for all {nodes} sensors, gives 1" if k == 1 else
             f"{k} weighted sums of each sensor's {_plural(hidden, 'number')}, each plus an offset and each with "
             f"the same weights for all {nodes} sensors, give {k}")
            + "; nothing sets their negative results to zero. The paper prints no such layer; "
            "its Table 4 hands the graph layers one number per sensor (assumption A-26)."
            + (f" The {nodes} sensors' rows are then put back in place, one per window." if shared else "")
            + (f" {out_note}" if out_note else ""),
            parts=[(f"{k} per sensor", "temporal_out_features")], shape=f"(B, {nodes}, {k})",
            learned=projection, source=_decided("temporal_out_features", 1)[0], picture="project"))
    elif kind == "last_cycle":
        steps.append(_step(
            "Keep the last reading",
            "Each sensor keeps only its most recent reading of the window"
            + (f"; the other {_plural(window - 1, 'reading')} {'is' if window == 2 else 'are'} dropped"
               if window > 1 else "")
            + ". Nothing is learned here. This stage stands for the paper's model without its LSTM (assumption A-07).",
            parts=[(f"cycle {window} of {window}", "temporal")], shape=f"(B, {nodes}, 1)", source=chosen,
            picture="last"))
    elif kind == "mean_window":
        steps.append(_step(
            "Average the window",
            (f"Each sensor's {readings} are replaced by their plain average." if window > 1 else
             "Each sensor's single reading is its own average.") + " Nothing is learned here; the paper has "
            "no such stage.",
            parts=[(f"{readings} to 1", "temporal")], shape=f"(B, {nodes}, 1)", source=chosen,
            picture="average"))
    elif kind == "window_summary":
        steps += [
            _together(
                _step("Last reading", "Each sensor's reading at the most recent cycle.",
                      parts=[("1 number", None)], shape=f"(B, {nodes})", picture="last"),
                _step("Average", f"The plain average of the {readings}.", parts=[("1 number", None)],
                      shape=f"(B, {nodes})", picture="average"),
                _step("Change over the window",
                      f"The slope of the straight line that best fits the {readings}, times {window - 1}, so the "
                      "change is in the readings' own units.", parts=[("1 number", None)], shape=f"(B, {nodes})",
                      picture="slope"),
            ),
            _step("Three numbers per sensor",
                  "The three are stacked for each sensor. Nothing is learned here, so the first graph layer's "
                  "weights mix the three, which lets time into the graph layer (assumption A-28).",
                  parts=[("last · average · change", "temporal")], shape=f"(B, {nodes}, 3)", source=chosen,
                  picture="stack"),
        ]
    else:
        first = int(settings["gcn_hidden"][0])
        steps.append(_step(
            "Pass every cycle",
            (f"All {readings} go on unchanged" if window > 1 else "The single reading goes on unchanged")
            + f", so the first graph layer's {window} x {first} weights become {_plural(first, 'learned pattern')} "
            "over the cycles, the same for every sensor; the fixed graph still does the mixing "
            "between sensors. This is the arrangement the last-cycle stage replaced (assumption A-07).",
            parts=[(f"all {readings}", "temporal")], shape=f"(B, {nodes}, {window})", source=chosen,
            picture="window"))
    steps.append(_edge("To the graph layers", f"(B, {nodes}, {k})", icon="graph"))
    return steps


def _graph_steps(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Box 4: the sensor graph, the three graph layers and the layers that turn them into remaining life."""
    import numpy as np

    from ..data.sensors import PAPER_SENSOR_SYMBOLS
    from ..graph.adjacency import (
        build_fully_connected_adjacency,
        build_physics_adjacency,
        normalize_adjacency,
        pairs_beyond_reach,
    )

    graph, rescaling, seed = settings["graph"], settings["adjacency_normalization"], settings["seed"]
    symbols = list(PAPER_SENSOR_SYMBOLS)
    if graph == "correlation":
        run = _prepared_for(settings)
        matrix = run["adjacency"] if run else None
    elif graph == "complete":
        matrix, _ = build_fully_connected_adjacency(len(symbols))
    else:
        matrix, _ = build_physics_adjacency()
    model = _built(settings, matrix)
    layers = {row["layer"]: row for row in model.layer_summary()}
    nodes, k = model.n_nodes, int(model.in_features)

    graph_words = {
        "physics": "the pairs the paper's Table 3 joins, built by hand from how the engine works",
        "complete": "every sensor joined to every other, which averages every sensor equally, so it measures the "
                    "cost of averaging as much as the value of the physics (audit item D-2)",
        "correlation": "the sensor pairs whose readings are most strongly correlated on the training engines",
    }[graph]
    extra, pairs, sums, beyond, apart = "", "", "", [], 0
    if matrix is not None:
        joined = np.asarray(matrix) > 0
        np.fill_diagonal(joined, False)
        if graph != "correlation":  # its name already says how many pairs it keeps
            pairs = f" · {_count(int(joined.sum() // 2))} pairs"
        rescaled = normalize_adjacency(np.asarray(matrix), method=rescaling)
        alone = [s for s, row in zip(symbols, joined, strict=True) if not row.any()]
        twins: dict[bytes, list[str]] = {}
        for symbol, row in zip(symbols, np.round(rescaled, 6), strict=True):
            twins.setdefault(row.tobytes(), []).append(symbol)
        groups = [group for group in twins.values() if len(group) > 1]
        if graph == "correlation":
            if float(settings["validation_fraction"]) == 0:
                extra += " With nothing held back, every seed builds this same graph."
            elif settings["fixed_split"]:
                extra += " With the same held-back engines for every seed, every seed uses this graph."
            else:
                extra += (f" This is seed {seed}'s graph, the first of the run; each seed builds its own from its "
                          "own training engines.")
        if alone:
            extra += (f" {' and '.join(alone)} {'is' if len(alone) == 1 else 'are'} joined to nothing, so "
                      f"{'it only ever sees itself' if len(alone) == 1 else 'each only ever sees itself'}.")
        if graph == "complete":
            extra += " All 14 sensors then carry identical numbers from the first graph layer on."
        elif groups:
            named = "; ".join(", ".join(g[:-1]) + " and " + g[-1] for g in groups)
            extra += (f" {_plural(len(groups), 'group')} of sensors get identical rows here ({named}), so the "
                      "layers after this can read each group only as the sum of its sensors (audit item D-1).")
        totals = rescaled.sum(axis=1)
        sums = (" Every row sums to 1." if np.allclose(totals, 1, atol=1e-5) else
                f" Its rows sum to between {totals.min():.4f} and {totals.max():.4f}, not to 1.")
        if graph == "complete":
            sums = " On this graph both rescalings give the same table, 1/14 everywhere, so this setting changes nothing."
        beyond = sorted((min(a, b, key=symbols.index), max(a, b, key=symbols.index))
                        for a, b, _ in pairs_beyond_reach(np.asarray(matrix), symbols, 3))
        # Pairs no path joins at all: pairs_beyond_reach counts only pairs a path joins.
        part = list(range(len(symbols)))
        for i, j in zip(*np.nonzero(joined), strict=True):
            a, b = part[i], part[j]
            part = [a if x == b else x for x in part]
        apart = sum(1 for i in range(len(symbols)) for j in range(i + 1, len(symbols)) if part[i] != part[j])
    else:
        extra = " It is built from the training engines, which are not in data/raw/, so it cannot be shown here."

    graph_tag, graph_note = _decided("graph", graph)
    graph_steps = [
        _step("Sensor graph",
              f"Which sensors exchange numbers: {graph_words}. It is a fixed table of 1s where two sensors are "
              f"joined and 0s elsewhere, the diagonal included; training never changes it.{extra} {graph_note}",
              parts=[(_label_of("graph", graph) + pairs, "graph")], shape=f"({nodes}, {nodes})", source=graph_tag,
              picture="graph"),
    ]
    tag, note = _decided("adjacency_normalization", rescaling)
    graph_steps.append(_step(
        "Rescale the graph",
        ("Each sensor is first joined to itself, a 1 on the diagonal. Then each entry is divided by its sensor's "
         "count of connections, itself included, so every sensor takes the plain average of itself and its "
         "neighbours: D^-1 (A + I), where A is the table, I the 1s added on the diagonal and D the counts."
         if rescaling == "row" else
         "Each sensor is first joined to itself, a 1 on the diagonal. Then each entry is divided by the square "
         "root of both sensors' counts of connections, itself included: D^-1/2 (A + I) D^-1/2, where A is the "
         "table, I the 1s added on the diagonal and D the counts. The table is symmetric.")
        + sums + (f" {note}" if note else ""),
        parts=[(_label_of("adjacency_normalization", rescaling).lower(), "adjacency_normalization")],
        shape=f"({nodes}, {nodes})", source=tag, picture="table"))
    # The graph is a second input of the first graph layer, beside the windows.
    steps = [_edge("From the temporal stage, B windows at a time", f"(B, {nodes}, {k})", icon="temporal"),
             _branch(*graph_steps)]

    bias = bool(settings["gcn_bias"])
    defaults = _BY_KEY["gcn_hidden"]["default"]
    width_out = str(k)
    index = 1
    while f"GCN-{index}" in layers:
        row = layers[f"GCN-{index}"]
        width_in = row["in"].split(",")[-1].strip(" )")
        width_out = row["out"].split(",")[-1].strip(" )")
        reach = {1: " Each sensor now carries what its neighbours held.",
                 2: " Each sensor now carries what its neighbours' neighbours held.",
                 3: " Each sensor now carries what sensors up to three steps away held."}.get(index, "")
        if index == 3 and (beyond or apart):
            far = [f"{_plural(len(beyond), 'pair')} {'is' if len(beyond) == 1 else 'are'} further apart than that ("
                   + ", ".join(f"{a}-{b}" for a, b in beyond) + ")"] if beyond else []
            far += [f"{_plural(apart, 'pair')} {'is' if apart == 1 else 'are'} joined by no path at all"] if apart else []
            reach += (" On this graph " + " and ".join(far) + ": they exchange nothing inside the graph layers, and "
                      "only the fully connected layer combines them.")
        width_note = (f' "Graph layer widths" was changed from its default for this layer, {defaults[index - 1]} '
                      "(the paper's Table 4)." if index <= len(defaults) and int(width_out) != defaults[index - 1]
                      else "")
        steps.append(_step(
            f"Graph layer {index}",
            f"Mixes each sensor's {_plural(int(width_in), 'number')} with its neighbours' through "
            f"the rescaled graph, multiplies by a learned {width_in} x {width_out} table of weights, "
            + ("adds an offset the paper's equation does not print (assumption A-12), " if bias else
               "adds no offset, as the paper's equation prints none (assumption A-12), ")
            + "then sets every negative result to zero (ReLU)." + reach
            + width_note,
            parts=[(f"{_plural(int(width_out), 'number')} per sensor", "gcn_hidden", index - 1),
                   ("offset" if bias else "no offset", "gcn_bias")],
            shape=row["out"], learned=int(row["parameters"]), source=_cite(f"Eq. {7 + index}"),
            picture="gcn"))
        index += 1
    flat, fc1, fc2 = layers["Flatten"], layers["FC-1"], layers["FC-2"]
    flat_width = flat["out"].split(",")[-1].strip(" )")
    fc = int(settings["fc_hidden"])
    _, fc_note = _decided("fc_hidden", fc)
    _, ceiling_note = _decided("rul_max", int(settings["rul_max"]))
    steps += [
        _step("Flatten",
              f"Lays the {nodes} sensors' numbers end to end in Table 3's order. Nothing is averaged, so each sensor "
              "keeps its own weights in the next layer.",
              parts=[(f"{nodes} x {width_out} = {flat_width}", None)], shape=flat["out"], source=_cite("Eq. 11"),
              picture="flatten"),
        _step("Fully connected",
              (f"Each of the {fc} outputs is" if fc > 1 else "Its one output is")
              + f" a weighted sum of all {flat_width} numbers plus an offset, then every negative result is set to "
              "zero (ReLU)." + (f" {fc_note}" if fc_note else ""),
              parts=[(_plural(fc, "number"), "fc_hidden")], shape=fc1["out"], learned=int(fc1["parameters"]),
              source=_cite("Eq. 12"), picture="dense"),
        _step("Output",
              f"One weighted sum of the {_plural(fc, 'number')} plus an offset, squeezed by the sigmoid into a "
              "fraction between "
              "0 and 1.",
              parts=[("a fraction from 0 to 1", None)], shape="(B,)", learned=int(fc2["parameters"]),
              source=_cite("Eq. 13"), picture="sigmoid"),
        _step("Scale to cycles",
              "The fraction times the ceiling gives remaining life in flight cycles, so no prediction can exceed "
              f"{settings['rul_max']}. Training divides its labels by the same number." + (
                  f" {ceiling_note}" if ceiling_note else ""),
              parts=[(f"x {settings['rul_max']}", "rul_max")], shape="(B,)",
              source={"kind": "assumption", "ref": "A-13", "text": ""}, picture="scale"),
        _edge("Remaining life, one number per window", "(B,)", icon="output"),
    ]
    return steps


def preprocessing_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The part of the settings that becomes preprocessing_config.yaml."""
    return {
        "subset": settings["subset"],
        "window_size": int(settings["window_size"]),
        "window_stride": int(settings["window_stride"]),
        "rul_max": int(settings["rul_max"]),
        "clip_test_labels": bool(settings["clip_test_labels"]),
        "normalization": settings["normalization"],
        "smoothing_alpha": float(settings["smoothing_alpha"]) if settings["smoothing"] else None,
        "graph": settings["graph"],
        "adjacency_normalization": settings["adjacency_normalization"],
        "validation_fraction": float(settings["validation_fraction"]),
        "seed": int(settings["seed"]),
        "notes": "Written by the playground of the documentation website.",
    }


def train_arguments(
    settings: Mapping[str, Any],
    config_path: Path,
    progress_path: Path | None,
    *,
    display: bool = False,
) -> list[str]:
    """The command that trains these settings: a list for the process, or words to show."""
    program = ["python"] if display else [sys.executable, "-u"]
    arguments = [
        *program, "scripts/train.py",
        "--config", str(config_path).replace("\\", "/") if display else str(config_path),
        "--seeds", str(settings["seeds"]),
        "--temporal", settings["temporal"],
    ]
    if settings["temporal"] in ("lstm", "gru"):
        arguments += ["--temporal-hidden", str(settings["temporal_hidden"]),
                      "--temporal-out-features", str(settings["temporal_out_features"])]
        if settings["shared_across_sensors"]:
            arguments.append("--shared-across-sensors")
    arguments += ["--gcn-hidden", *(str(w) for w in settings["gcn_hidden"]),
                  "--fc-hidden", str(settings["fc_hidden"])]
    if settings["gcn_bias"]:
        arguments.append("--gcn-bias")
    arguments += [
        "--epochs", str(settings["epochs"]),
        "--batch-size", str(settings["batch_size"]),
        "--learning-rate", repr(float(settings["learning_rate"])),
        "--final-learning-rate", repr(float(settings["final_learning_rate"])),
        "--lr-schedule", settings["lr_schedule"],
    ]
    if settings["early_stopping"]:
        arguments += ["--early-stopping-patience", str(settings["early_stopping_patience"])]
    else:
        arguments.append("--no-early-stopping")
    if settings["fixed_split"]:
        arguments.append("--fixed-split")
    arguments += ["--device", settings["device"]]
    if progress_path is not None:
        arguments += ["--progress", str(progress_path)]
    return arguments


def explain_arguments(request: Mapping[str, Any], progress_path: Path | None,
                      *, display: bool = False) -> list[str]:
    """The command that explains a stored run."""
    program = ["python"] if display else [sys.executable, "-u"]
    arguments = [*program, "scripts/explain.py", "--run", request["run"],
                 "--seed", str(request["seed"]), "--method", request["method"],
                 "--scope", request["scope"]]
    if request["scope"] == "engine":
        if request.get("engine") is not None:
            arguments += ["--engine", str(request["engine"])]
        if int(request.get("every") or 1) > 1:
            arguments += ["--every", str(request["every"])]
    if progress_path is not None:
        arguments += ["--progress", str(progress_path)]
    return arguments


def options() -> dict[str, Any]:
    """Everything the page needs to draw its form, before anything is chosen."""
    from .figures import _graph_layout

    graph, positions = _graph_layout()[:2]
    return {
        "fields": [{**_with_more(spec, spec["key"]),
                    "choices": [_with_more(c, f"{spec['key']}={c['value']}") for c in spec.get("choices", [])]}
                   if "choices" in spec else _with_more(spec, spec["key"]) for spec in FIELDS],
        "groups": [{"key": k, "label": label} for k, label in GROUPS],
        "procedures": [_with_more(procedure, "procedure") for procedure in PROCEDURES],
        "procedure_about": PROCEDURE_ABOUT,
        "procedure_more": read_more("procedure"),
        "explain_more": {key.split("=")[1]: read_more(key) for key in _READ_MORE if key.startswith("explain=")},
        "defaults": {spec["key"]: spec["default"] for spec in FIELDS},
        "result_seeds": RESULT_SEEDS,
        "dataset": DATASET_DESCRIPTION,
        "graph": {
            "positions": {node: [round(x, 1), round(y, 1)] for node, (x, y) in positions.items()},
            "edges": [list(edge) for edge in graph.edges()],
            "width": 860,
            "height": 520,
        },
        "table12": {
            "graphs": list(SHAP_TABLE_12_GRAPHS),
            "values": {sensor: list(values) for sensor, values in SHAP_TABLE_12.items()},
            "ring": list(FIG_10_RING),
        },
    }


# -- stored runs ------------------------------------------------------------


def _clean(value: Any) -> Any:
    """Replace NaN and infinity with None, which JSON can carry and they cannot."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    return value


def _run_directory(root: Path, name: str) -> Path:
    """A run folder in reports/, refusing any name that is not one."""
    reports = (root / "reports").resolve()
    path = (root / "reports" / name).resolve()
    # Looking for separators is not enough on Windows: a name carrying a
    # drive, "C:x", holds none of them, and joining it throws the base away.
    if not name or name.startswith(".") or path.parent != reports:
        raise SettingsError(f"{name!r} is not the name of a run.")
    if not (path / "summary.json").is_file():
        raise SettingsError(f"There is no finished run called {name!r} in reports/.")
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """A CSV file as a list of rows, with missing values as None."""
    import pandas as pd

    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    return _clean(frame.astype(object).where(frame.notna(), None).to_dict(orient="records"))


def _describe(path: Path) -> dict[str, Any] | None:
    """One stored run, as the results table lists it."""
    try:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if "mean" not in summary or not (path / "results.csv").is_file():
        return None
    config = _read_yaml(path / "preprocessing_config.yaml")
    model = summary.get("model") or (
        _read_yaml(path / "model_config.yaml") if (path / "model_config.yaml").is_file() else None
    )
    temporal = summary.get("temporal_stage")
    subset = str(summary.get("subset", config.get("subset", "")))
    validation_fraction = config.get("validation_fraction", summary.get("validation_fraction"))
    row, reason = (paper_row_for(subset, temporal, summary.get("graph"))
                   if subset in SUBSETS else (None, "Unknown sub-dataset."))
    seeds = int(summary.get("seeds", 0))
    commit = str(summary.get("commit", "unknown"))
    settings_for_caveats = {
        "seeds": seeds, "temporal": temporal, "subset": subset,
        "window_size": config.get("window_size"),
        "graph": summary.get("graph"), "clip_test_labels": summary.get("clip_test_labels"),
        "validation_fraction": validation_fraction or 0.0,
        "fixed_split": summary.get("split") == "fixed",
        "gcn_hidden": (model or {}).get("gcn_hidden"),
        "fc_hidden": (model or {}).get("fc_hidden"),
        "shared_across_sensors": (model or {}).get("shared_across_sensors"),
        "lr_schedule": summary.get("lr_schedule"),
    }
    return _clean({
        "name": path.name,
        "commit": commit,
        "dirty": commit.endswith("-dirty"),
        "finished": str(summary.get("finished_utc", "")),
        "subset": subset,
        "seeds": seeds,
        "exploratory": seeds < RESULT_SEEDS,
        "temporal": temporal,
        "temporal_hidden": summary.get("temporal_hidden"),
        "graph": summary.get("graph"),
        "adjacency_normalization": (summary.get("adjacency_normalization")
                                    or config.get("adjacency_normalization")),
        "split": summary.get("split"),
        "normalization": config.get("normalization"),
        "smoothing_alpha": config.get("smoothing_alpha"),
        "window_size": config.get("window_size"),
        "rul_max": config.get("rul_max"),
        "clip_test_labels": summary.get("clip_test_labels"),
        "validation_fraction": validation_fraction,
        # What train() does, read from the settings rather than trusted from a
        # field: with nothing held back it can only keep the last pass.
        "reported_weights": ("final_epoch" if not validation_fraction
                             else "best_validation_epoch"),
        "lr_schedule": summary.get("lr_schedule"),
        "model": model,
        "training": summary.get("training"),
        "has_weights": any(path.glob("seed_*/model.pt")),
        "mean": summary.get("mean"),
        "std": summary.get("std"),
        "paper_row": row,
        "no_row_reason": reason,
        "caveats": caveats(settings_for_caveats),
    })


def list_runs(root: Path) -> list[dict[str, Any]]:
    """Every finished run in reports/, newest first."""
    reports = root / "reports"
    if not reports.is_dir():
        return []
    runs = [described for path in sorted(reports.iterdir())
            if path.is_dir() and not path.name.startswith(".")
            and (described := _describe(path)) is not None]
    runs.sort(key=lambda run: run["finished"], reverse=True)
    return runs


def run_detail(root: Path, name: str) -> dict[str, Any]:
    """One run in full: every seed's numbers, curves, predictions and explanations."""
    path = _run_directory(root, name)
    described = _describe(path)
    if described is None:
        raise SettingsError(f"{name!r} is not a training run.")
    seeds = {}
    for folder in sorted(path.glob("seed_*")):
        seeds[folder.name.removeprefix("seed_")] = {
            "history": _read_csv(folder / "training_history.csv"),
            "predictions": _read_csv(folder / "test_predictions.csv"),
            "has_weights": (folder / "model.pt").is_file(),
        }
    explanations = []
    for summary_path in sorted((path / "explanations").glob("*/summary.json")):
        try:
            explanations.append(
                {"name": summary_path.parent.name,
                 **json.loads(summary_path.read_text(encoding="utf-8"))}
            )
        except (OSError, json.JSONDecodeError):
            continue
    explanations.sort(key=lambda item: str(item.get("finished_utc", "")), reverse=True)
    return _clean({
        **described,
        "results": _read_csv(path / "results.csv"),
        "preprocessing": _read_yaml(path / "preprocessing_config.yaml"),
        "per_seed": seeds,
        "explanations": explanations,
    })


def list_explanations(root: Path) -> list[dict[str, Any]]:
    """Every stored explanation in reports/, with the run it explains, newest first."""
    found = []
    for run in list_runs(root):
        for summary_path in sorted((root / "reports" / run["name"] / "explanations").glob("*/summary.json")):
            try:
                stored = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            found.append({**stored, "name": summary_path.parent.name, "run": run["name"],
                          "run_subset": run.get("subset"), "run_temporal": run.get("temporal"),
                          "run_temporal_hidden": run.get("temporal_hidden"), "run_graph": run.get("graph"),
                          "run_finished": run.get("finished")})
    found.sort(key=lambda item: str(item.get("finished_utc", "")), reverse=True)
    return _clean(found)


def explanation_detail(root: Path, run: str, name: str) -> dict[str, Any]:
    """One stored explanation: its summary and every value it computed."""
    path = (_run_directory(root, run) / "explanations").resolve()
    folder = (path / name).resolve()
    # The same hazard as a run name: a drive carried in the name escapes.
    if not name or name.startswith(".") or folder.parent != path:
        raise SettingsError(f"{name!r} is not the name of an explanation.")
    if not (folder / "summary.json").is_file():
        raise SettingsError(f"Run {run!r} has no explanation called {name!r}.")
    return _clean({
        "summary": json.loads((folder / "summary.json").read_text(encoding="utf-8")),
        "values": _read_csv(folder / "values.csv"),
    })


_COMPARED = ("rmse", "score", "r2", "asue")


def _flat_settings(detail: Mapping[str, Any]) -> dict[str, Any]:
    """Every setting of a run on one level, for spotting what two runs change."""
    flat: dict[str, Any] = {}
    for key, value in (detail.get("preprocessing") or {}).items():
        if key not in ("notes", "seed"):
            flat[f"preprocessing.{key}"] = value
    for key, value in (detail.get("model") or {"temporal": detail.get("temporal")}).items():
        flat[f"model.{key}"] = value
    for key, value in (detail.get("training") or {}).items():
        flat[f"training.{key}"] = value
    flat["seeds"] = detail.get("seeds")
    flat["split"] = detail.get("split")
    flat["commit"] = detail.get("commit")
    return flat


def compare_runs(root: Path, first: str, second: str) -> dict[str, Any]:
    """Two runs paired seed by seed: what differs, by how much, and whether seeds explain it."""
    a, b = run_detail(root, first), run_detail(root, second)
    flat_a, flat_b = _flat_settings(a), _flat_settings(b)
    differences = [
        {"setting": key, "first": flat_a.get(key), "second": flat_b.get(key)}
        for key in sorted(set(flat_a) | set(flat_b))
        if flat_a.get(key) != flat_b.get(key)
    ]
    by_seed_a = {row["seed"]: row for row in a["results"]}
    by_seed_b = {row["seed"]: row for row in b["results"]}
    shared = sorted(set(by_seed_a) & set(by_seed_b))

    measures: dict[str, Any] = {}
    for measure in _COMPARED:
        pairs = [(by_seed_a[s][measure], by_seed_b[s][measure]) for s in shared
                 if by_seed_a[s].get(measure) is not None
                 and by_seed_b[s].get(measure) is not None]
        entry: dict[str, Any] = {
            "per_seed": [{"seed": s, "first": x, "second": y, "difference": y - x}
                         for s, (x, y) in zip(shared, pairs, strict=False)],
            "mean_difference": None, "t": None, "p": None,
        }
        if pairs:
            differences_here = [y - x for x, y in pairs]
            entry["mean_difference"] = sum(differences_here) / len(differences_here)
        if len(pairs) >= 2:
            from scipy import stats

            test = stats.ttest_rel([y for _, y in pairs], [x for x, _ in pairs])
            entry["t"], entry["p"] = float(test.statistic), float(test.pvalue)
        measures[measure] = entry
    return _clean({
        "first": {"name": a["name"], "mean": a["mean"], "std": a["std"], "seeds": a["seeds"]},
        "second": {"name": b["name"], "mean": b["mean"], "std": b["std"], "seeds": b["seeds"]},
        "shared_seeds": shared,
        "differences": differences,
        "measures": measures,
    })


# -- jobs -------------------------------------------------------------------


# The states a job can no longer leave.
_ENDED = ("done", "failed", "cancelled", "interrupted")

# The one vocabulary the page uses for temporal stages and explanation methods.
_STAGE_NAMES = {"lstm": "LSTM", "gru": "GRU", "last_cycle": "w/o LSTM",
                "mean_window": "window average", "window_summary": "window summary",
                "raw_window": "whole window"}
_METHOD_NAMES = {"shap": "Exact Shapley values", "occlusion": "Occlusion"}
# What a job is called when its folder holds nothing to name it by.
_UNNAMED = {"train": "Training run", "explain": "Explanation"}


def job_label(settings: Mapping[str, Any]) -> str:
    """The name a training job is listed under: what it trains, on what, how many times."""
    parts = [str(settings.get("subset") or "")]
    temporal = settings.get("temporal")
    if temporal is not None:
        stage = _STAGE_NAMES.get(str(temporal), str(temporal))
        if temporal in ("lstm", "gru") and settings.get("temporal_hidden") is not None:
            stage += f" {settings['temporal_hidden']}"
        parts.append(stage)
    graph = str(settings.get("graph") or "physics")
    rescaling = settings.get("adjacency_normalization")
    parts.append(f"{graph} graph" if rescaling in (None, "row") else f"{graph} · {rescaling} graph")
    parts.append("best pass" if float(settings.get("validation_fraction") or 0.0) > 0
                 else "final pass")
    seeds = int(settings.get("seeds") or 0)
    parts.append(f"{seeds} seed{'' if seeds == 1 else 's'}")
    return " · ".join(part for part in parts if part)


def explanation_label(request: Mapping[str, Any]) -> str:
    """The name an explanation job is listed under: which method, on what, with which seed's weights."""
    method = request.get("method")
    parts = [_METHOD_NAMES.get(str(method), str(method or "Explanation"))]
    if request.get("scope") == "test":
        parts.append("test engines")
    elif request.get("scope") == "engine":
        engine = request.get("engine")
        parts.append("longest engine" if engine is None else f"engine {engine}")
    if request.get("seed") is not None:
        parts.append(f"seed {request['seed']}")
    return " · ".join(parts)


def _plain_name(name: Any) -> bool:
    """Whether a name read from a file is one plain folder name, with no way out of its folder."""
    return (isinstance(name, str) and bool(name) and not name.startswith(".")
            and ":" not in name and Path(name).name == name)


def _explain_request(folder: Path, command: str, progress: Mapping[str, Any]) -> dict[str, Any]:
    """What an explanation job asked for, from its command, and the engine it chose once it finished."""
    request: dict[str, Any] = {"method": progress.get("method"), "scope": progress.get("scope")}
    words = command.split()
    for flag, key in (("--method", "method"), ("--scope", "scope"),
                      ("--seed", "seed"), ("--engine", "engine")):
        if flag in words[:-1]:
            value = words[words.index(flag) + 1]
            request[key] = int(value) if value.isdigit() else value
    # With no engine asked for, the script picks the longest, and only its
    # stored summary says which one that was. The jobs folder sits inside
    # reports/, beside the run the explanation was written into.
    result = progress.get("result") or {}
    if _plain_name(result.get("run")) and _plain_name(result.get("directory")):
        stored = (folder.parent.parent / result["run"] / "explanations"
                  / result["directory"] / "summary.json")
        try:
            summary = json.loads(stored.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
        if isinstance(summary, dict):
            request.update({key: summary[key] for key in ("method", "scope", "seed")
                            if summary.get(key) is not None})
            if summary.get("unit") is not None:
                request["engine"] = summary["unit"]
    return request


def _job_name(folder: Path, kind: str, command: str, progress: Mapping[str, Any],
              fallback: str) -> str:
    """The name a job is listed under, built the same way for this server's jobs and for folders read back."""
    if kind == "train":
        try:
            settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            settings = None
        if isinstance(settings, dict) and settings:
            return job_label(settings)
    elif kind == "explain":
        request = _explain_request(folder, command, progress)
        if request.get("method"):
            return explanation_label(request)
    return fallback


def read_progress(folder: Path) -> dict[str, Any]:
    """What a job's progress file says so far, summarised for the page."""
    path = folder / "progress.jsonl"
    if not path.is_file():
        return {}
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # the last line may be half-written
    summary: dict[str, Any] = {"seeds_done": [], "curve": []}
    for event in events:
        kind = event.get("event")
        if kind == "start":
            summary.update({k: v for k, v in event.items() if k != "event"})
        elif kind == "seed":
            summary["seed"], summary["seed_index"], summary["curve"] = (
                event.get("seed"), event.get("index"), [])
        elif kind == "epoch":
            summary["epoch"] = event.get("epoch")
            summary["curve"].append({
                "epoch": event.get("epoch"),
                "train_loss": event.get("train_loss"),
                "validation_rmse": event.get("validation_rmse_cycles"),
            })
        elif kind == "seed_done":
            summary["seeds_done"].append({k: v for k, v in event.items() if k != "event"})
        elif kind == "window":
            summary.update(done=event.get("done"), total=event.get("total"),
                           seconds=event.get("seconds"))
        elif kind == "done":
            summary["result"] = {k: v for k, v in event.items() if k != "event"}
    return summary


def _process_alive(pid: Any) -> bool:
    """Whether a process with this number is still running on this machine."""
    # The number is read back from a file, so it may be anything at all. A
    # damaged one means "nothing is running", never an error raised out of
    # the listing that every job on the page is drawn from.
    try:
        number = int(pid)
    except (TypeError, ValueError):
        return False
    if number <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        # 0x1000 asks only "does this exist and how did it end", which needs no
        # right over the process; 259 is what Windows reports while it runs.
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, number)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            asked = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(asked) and code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(number, 0)
    except OSError:
        return False
    return True


def _log_tail(folder: Path, lines: int = 25) -> str:
    """The last lines a command printed, which is where a failure explains itself."""
    path = folder / "output.txt"
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


@dataclass
class Job:
    """One command the playground started, or will start when the one before it ends."""

    id: str
    kind: str
    label: str
    arguments: list[str]
    display: str
    folder: Path
    created: float = field(default_factory=time.time)
    status: str = "queued"
    started: float | None = None
    finished: float | None = None
    returncode: int | None = None
    process: subprocess.Popen[bytes] | None = field(default=None, repr=False)

    def progress(self) -> dict[str, Any]:
        """What the job's progress file says so far, summarised for the page."""
        return read_progress(self.folder)

    def log_tail(self, lines: int = 25) -> str:
        """The last lines the command printed, which is where a failure explains itself."""
        return _log_tail(self.folder, lines)

    def record(self) -> None:
        """Write what the page shows about this job beside its files, so a later server can list it."""
        summary = {
            "id": self.id, "kind": self.kind, "label": self.label, "command": self.display,
            "created": self.created, "status": self.status, "started": self.started,
            "finished": self.finished, "returncode": self.returncode,
            # So that a later server can tell a job still running under the
            # server that started it from one whose server died with it.
            "pid": self.process.pid if self.process is not None else None,
        }
        # If the folder was removed meanwhile, the listing is the loss, not the run.
        with contextlib.suppress(OSError):
            (self.folder / "job.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        """The job as the page shows it."""
        now = time.time()
        progress = self.progress()
        return _clean({
            "id": self.id, "kind": self.kind,
            "label": _job_name(self.folder, self.kind, self.display, progress, self.label),
            "status": self.status,
            "command": self.display, "created": self.created, "started": self.started,
            "finished": self.finished, "returncode": self.returncode,
            "elapsed": (self.finished or now) - self.started if self.started else None,
            "progress": progress,
            "log": self.log_tail() if self.status in ("failed", "cancelled") else "",
            "from_disk": False,
        })


def _stamp_to_time(stamp: str) -> float | None:
    """The moment a job folder's name records, as seconds since the epoch."""
    try:
        return float(calendar.timegm(time.strptime(stamp, "%Y%m%dT%H%M%S")))
    except ValueError:
        return None


def _describe_job_folder(path: Path) -> dict[str, Any] | None:
    """One job an earlier server left behind, read from its folder."""
    parts = path.name.split("_")
    if len(parts) < 3:
        return None
    recorded: dict[str, Any] = {}
    if (path / "job.json").is_file():
        try:
            recorded = json.loads((path / "job.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            recorded = {}
    progress = read_progress(path)
    kind = str(recorded.get("kind") or parts[1])
    status = str(recorded.get("status") or "")
    if status not in _ENDED:
        if "result" in progress:
            status = "done"
        elif _process_alive(recorded.get("pid")):
            # The server that started it is still running it. Calling this
            # ended would offer it to "remove all ended", which would delete
            # the folder the command is at that moment writing into.
            status = "running"
        else:
            # A job still running when its server stopped never wrote its end.
            status = "interrupted"
    started, finished = recorded.get("started"), recorded.get("finished")
    command = str(recorded.get("command") or "")
    return _clean({
        "id": path.name,
        "kind": kind,
        # Rebuilt rather than read from job.json, so a folder an older server
        # wrote is named in the same words as a job this server is running.
        "label": _job_name(path, kind, command, progress, str(
            recorded.get("label") or _UNNAMED.get(kind, f"{kind} job"))),
        "status": status,
        "command": command,
        "created": recorded.get("created") or _stamp_to_time(parts[0]) or path.stat().st_mtime,
        "started": started,
        "finished": finished,
        "elapsed": (finished - started) if started and finished else None,
        "progress": progress,
        "log": _log_tail(path) if status == "failed" else "",
        "from_disk": True,
    })


def job_history(root: Path) -> list[dict[str, Any]]:
    """Every job any server left in reports/.playground/, read from the folders."""
    folder = root / JOBS_FOLDER
    if not folder.is_dir():
        return []
    return [described for path in sorted(folder.iterdir())
            if path.is_dir() and (described := _describe_job_folder(path)) is not None]


class JobManager:
    """Runs the playground's commands one at a time, in the order they were asked for."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.jobs: dict[str, Job] = {}
        self.queue: list[str] = []
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stopping = False
        self.worker: threading.Thread | None = None

    def _ensure_worker(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(target=self._work, name="playground-jobs", daemon=True)
            self.worker.start()

    def _new_folder(self, kind: str) -> tuple[str, Path]:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        identifier = f"{stamp}_{kind}_{secrets.token_hex(3)}"
        folder = self.root / JOBS_FOLDER / identifier
        folder.mkdir(parents=True, exist_ok=True)
        return identifier, folder

    def submit_training(self, settings: Mapping[str, Any]) -> Job:
        """Queue one training run of these settings, under the name job_label gives it."""
        identifier, folder = self._new_folder("train")
        config_path = folder / "preprocessing_config.yaml"
        config_path.write_text(
            yaml.safe_dump(preprocessing_settings(settings), sort_keys=False), encoding="utf-8"
        )
        (folder / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        relative = config_path.relative_to(self.root)
        job = Job(
            id=identifier, kind="train", label=job_label(settings), folder=folder,
            arguments=train_arguments(settings, config_path, folder / "progress.jsonl"),
            display=" ".join(train_arguments(settings, relative, None, display=True)),
        )
        return self._enqueue(job)

    def submit_explanation(self, request: Mapping[str, Any]) -> Job:
        """Queue one explanation of a stored run, under the name explanation_label gives it."""
        identifier, folder = self._new_folder("explain")
        job = Job(
            id=identifier, kind="explain", label=explanation_label(request), folder=folder,
            arguments=explain_arguments(request, folder / "progress.jsonl"),
            display=" ".join(explain_arguments(request, None, display=True)),
        )
        return self._enqueue(job)

    def submit_command(self, kind: str, label: str, arguments: list[str]) -> Job:
        """Queue any command. Only the tests use this directly."""
        identifier, folder = self._new_folder(kind)
        job = Job(id=identifier, kind=kind, label=label, folder=folder,
                  arguments=list(arguments), display=" ".join(arguments))
        return self._enqueue(job)

    def _enqueue(self, job: Job) -> Job:
        with self.lock:
            self.jobs[job.id] = job
            self.queue.append(job.id)
            # Written under the lock: the worker can take this job the moment
            # the lock is released, and its own record must land after this one.
            job.record()
        self._ensure_worker()
        self.wake.set()
        return job

    def cancel(self, identifier: str) -> Job:
        """Remove a queued job, or stop a running one. A stopped run writes nothing."""
        with self.lock:
            job = self.jobs.get(identifier)
            if job is None:
                raise SettingsError(f"There is no job called {identifier!r}.")
            if job.status == "queued":
                self.queue.remove(identifier)
                job.status, job.finished = "cancelled", time.time()
                job.record()
            elif job.status == "running":
                job.status = "cancelling"
                # A job is marked running a moment before its process exists.
                # Asked to stop in that moment, there is nothing to end yet, so
                # _run sees "cancelling" and never starts it.
                if job.process is not None:
                    job.process.terminate()
        return job

    def forget(self, identifier: str = "", *, all_finished: bool = False) -> list[str]:
        """Drop finished jobs from the list and delete their folders. A running or queued job stays."""
        if all_finished:
            targets = [item["id"] for item in self.snapshot() if item["status"] in _ENDED]
        else:
            targets = [identifier]
        forgotten = []
        jobs_folder = (self.root / JOBS_FOLDER).resolve()
        for target in targets:
            # The name has to be one plain folder name inside the jobs folder.
            # Looking for separators is not enough: a Windows name carrying a
            # drive, "C:x", has none, and joining it throws the base away.
            folder = (self.root / JOBS_FOLDER / target).resolve()
            if not target or target.startswith(".") or folder.parent != jobs_folder:
                raise SettingsError(f"{target!r} is not the name of a job.")
            with self.lock:
                job = self.jobs.get(target)
                if job is not None:
                    if job.status not in _ENDED:
                        raise SettingsError(
                            f"Job {target!r} is {job.status}; stop it before removing it."
                        )
                    del self.jobs[target]
            if job is None:
                # Not this server's job. It may still belong to a running one.
                described = _describe_job_folder(folder)
                if described is not None and described["status"] not in _ENDED:
                    raise SettingsError(
                        f"Job {target!r} is {described['status']}; stop it before removing it."
                    )
            if folder.is_dir():
                # The job's own folder only: what it printed and its progress.
                # The run it produced lives elsewhere in reports/ and stays.
                shutil.rmtree(folder, ignore_errors=True)
            elif job is None:
                raise SettingsError(f"There is no job called {target!r}.")
            forgotten.append(target)
        return forgotten

    def snapshot(self) -> list[dict[str, Any]]:
        """Every job this server was asked for, then those earlier servers left on disk, newest first."""
        with self.lock:
            jobs = list(self.jobs.values())
        listed = [job.to_dict() for job in jobs]
        known = {job.id for job in jobs}
        listed += [item for item in job_history(self.root) if item["id"] not in known]
        listed.sort(key=lambda item: item["created"], reverse=True)
        return listed

    def get(self, identifier: str) -> Job | None:
        with self.lock:
            return self.jobs.get(identifier)

    def stop(self) -> None:
        """Stop the running job and forget the queue. Called when the server stops."""
        self.stopping = True
        with self.lock:
            self.queue.clear()
            running = [job for job in self.jobs.values() if job.process is not None
                       and job.process.poll() is None]
        for job in running:
            job.process.terminate()  # type: ignore[union-attr]
        self.wake.set()

    def _work(self) -> None:
        while not self.stopping:
            with self.lock:
                identifier = self.queue.pop(0) if self.queue else None
                job = self.jobs.get(identifier) if identifier else None
                if job is not None:
                    job.status, job.started = "running", time.time()
            if job is None:
                self.wake.clear()
                self.wake.wait(timeout=1.0)
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        logger.info("Playground job %s: %s", job.id, job.display)
        with (job.folder / "output.txt").open("wb") as output:
            try:
                # Started and published under the lock, so a stop asked for
                # before this point is seen here, and one asked for after it
                # finds the process to end.
                with self.lock:
                    if job.status == "cancelling":
                        job.returncode = None
                    else:
                        job.process = subprocess.Popen(
                            job.arguments, cwd=self.root, stdout=output,
                            stderr=subprocess.STDOUT,
                            env={**_environment(), "PYTHONIOENCODING": "utf-8"},
                        )
                if job.process is not None:
                    # Recorded now the process exists, so its number reaches
                    # the file while the command is still running.
                    job.record()
                    job.returncode = job.process.wait()
            except OSError as failure:
                output.write(f"Could not start the command: {failure}\n".encode())
                job.returncode = -1
        with self.lock:
            job.finished = time.time()
            if job.status == "cancelling":
                job.status = "cancelled"
            else:
                job.status = "done" if job.returncode == 0 else "failed"
            job.process = None
        job.record()


def _environment() -> dict[str, str]:
    import os

    return dict(os.environ)
