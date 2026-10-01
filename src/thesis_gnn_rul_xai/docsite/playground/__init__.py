"""The playground behind the website: checks settings, runs the real scripts, and reads what they stored.

Read README.md in this folder. It explains why every run started here goes
through scripts/train.py and lands in reports/, and what the page shows.
"""

from __future__ import annotations

from .commands import options, preprocessing_settings, train_arguments
from .design import STAGES, caveats, preview, stage_detail
from .jobs import JobManager, explanation_label, job_label
from .runs import (
    compare_runs,
    explanation_detail,
    list_explanations,
    list_runs,
    run_detail,
    validate_explain,
)
from .settings import (
    ABLATIONS,  # ablations: the paper's Table 11 variants, each with one part taken out
    FIELDS,
    GROUPS,
    PROCEDURE_ABOUT,
    PROCEDURES,
    SettingsError,
    paper_row_for,
    read_more,
    validate_settings,
)

__all__ = [
    "ABLATIONS",
    "FIELDS",
    "GROUPS",
    "PROCEDURES",
    "PROCEDURE_ABOUT",
    "STAGES",
    "JobManager",
    "SettingsError",
    "caveats",
    "compare_runs",
    "explanation_detail",
    "explanation_label",
    "job_label",
    "list_explanations",
    "list_runs",
    "options",
    "paper_row_for",
    "preprocessing_settings",
    "preview",
    "read_more",
    "run_detail",
    "stage_detail",
    "train_arguments",
    "validate_explain",
    "validate_settings",
]
