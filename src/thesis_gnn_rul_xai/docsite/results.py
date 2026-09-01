"""Reads the stored runs in reports/ and the assumptions log, for the website.

Read README.md in this folder. It explains everything this file does, and why
a run whose commit is unknown is shown as such rather than hidden.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ..utils.paths import project_root

__all__ = [
    "Entry",
    "Run",
    "Verification",
    "latest_verification",
    "load_entries",
    "load_runs",
]

# A heading in docs/ASSUMPTIONS.md that introduces one entry, such as
# "## A-01 — The label ceiling is 125 cycles" or "### F-04 — WITHDRAWN AND ...".
_ENTRY_HEADING = re.compile(
    r"^#{2,3}\s+(~~)?([AF]-\d{2})(~~)?\s*[—-]+\s*(.+?)\s*$", re.MULTILINE
)

# The line inside an entry that names the code the choice is made in.
_WHERE = re.compile(r"^\*\*Where:\*\*\s*(.+?)$", re.MULTILINE)

# The log marks a retired entry by opening its title with a capitalised status
# word, as in "WITHDRAWN: the paper does state the order". Matching the word
# anywhere in the title instead would catch ordinary prose: A-25's title says
# an order "reverses" and that entry is perfectly current.
_RETIRED = re.compile(r"^\s*(WITHDRAWN(\s+AND\s+REVERSED)?|REVERSED|SUPERSEDED)\b")


@dataclass(frozen=True)
class Run:
    """One stored training run: its settings, its four numbers and its commit."""

    directory: str
    commit: str
    finished: str
    subset: str
    graph: str
    adjacency_normalization: str
    seeds: int
    split: str
    clip_test_labels: bool
    early_stopping: bool
    mean: dict[str, float | None]
    std: dict[str, float | None]
    dirty: bool


@dataclass(frozen=True)
class Verification:
    """One run of scripts/verify_claims.py: how many claims it re-checked."""

    directory: str
    commit: str
    finished: str
    checked: int
    failed: int


@dataclass(frozen=True)
class Entry:
    """One assumption or finding, as the website lists it on a card."""

    identifier: str
    kind: str
    title: str
    anchor: str
    where: str
    status: str
    summary: str


def _numbers(block: object) -> dict[str, float | None]:
    """One of the mean or standard-deviation blocks of a summary file.

    A run over a single seed writes null for every standard deviation, because
    one run has no spread. That is kept as null rather than turned into a zero,
    which would read as "measured, and it was zero".
    """
    if not isinstance(block, dict):
        return {}
    out: dict[str, float | None] = {}
    for key, value in block.items():
        try:
            out[str(key)] = None if value is None else float(value)
        except (TypeError, ValueError):
            out[str(key)] = None
    return out


def load_runs(root: Path | None = None) -> list[Run]:
    """Every run in reports/ that wrote a summary, newest first."""
    root = root or project_root()
    reports = root / "reports"
    if not reports.is_dir():
        return []

    runs: list[Run] = []
    for summary_path in sorted(reports.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        # reports/ holds two kinds of directory. A training run writes a
        # per-seed results.csv beside its summary; scripts/verify_claims.py
        # writes a claims.csv and no scores at all. Only the first kind is a
        # run, and showing the second as one with every column blank would
        # read as a training run that failed.
        if not (summary_path.parent / "results.csv").is_file():
            continue
        if "mean" not in summary:
            continue

        commit = str(summary.get("commit", "unknown"))
        runs.append(
            Run(
                directory=summary_path.parent.name,
                commit=commit,
                finished=str(summary.get("finished_utc", "")),
                subset=str(summary.get("subset", "")),
                graph=str(summary.get("graph", "")),
                adjacency_normalization=str(summary.get("adjacency_normalization", "")),
                seeds=int(summary.get("seeds", 0)),
                split=str(summary.get("split", "")),
                clip_test_labels=bool(summary.get("clip_test_labels", False)),
                early_stopping=bool(summary.get("early_stopping", False)),
                mean=_numbers(summary.get("mean")),
                std=_numbers(summary.get("std")),
                dirty=commit.endswith("-dirty"),
            )
        )
    runs.sort(key=lambda run: run.finished, reverse=True)
    return runs


def latest_verification(root: Path | None = None) -> Verification | None:
    """The most recent run of verify_claims.py, or nothing if it never ran.

    That script recomputes the figures docs/ASSUMPTIONS.md quotes and fails if
    one disagrees, so its last result is how far the log can be trusted.
    """
    root = root or project_root()
    reports = root / "reports"
    if not reports.is_dir():
        return None

    found: list[Verification] = []
    for summary_path in sorted(reports.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "claims_checked" not in summary:
            continue
        found.append(
            Verification(
                directory=summary_path.parent.name,
                commit=str(summary.get("commit", "unknown")),
                finished=str(summary.get("finished_utc", "")),
                checked=int(summary.get("claims_checked", 0)),
                failed=int(summary.get("claims_failed", 0)),
            )
        )
    return max(found, key=lambda v: v.finished) if found else None


def _status_of(identifier: str, heading: str, body: str) -> str:
    """Whether an entry still stands, was withdrawn, or was superseded."""
    retired = _RETIRED.match(heading)
    if retired:
        word = retired.group(1).upper()
        if word.startswith("WITHDRAWN"):
            return "withdrawn"
        return "reversed" if word == "REVERSED" else "superseded"
    # A-06 and A-18 carry their status in the body instead, as a bold
    # "Superseded <date> by A-22." line at the top of the entry.
    if re.match(r"\s*\*\*Superseded\b", body):
        return "superseded"
    return "active"


def _summarise(body: str) -> str:
    """The first sentence of prose in an entry, for the card's front."""
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "|", "```", ">", "**Where:**")):
            continue
        plain = re.sub(r"[*`_]", "", stripped)
        plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain)
        if len(plain) > 40:
            return plain[:260] + ("…" if len(plain) > 260 else "")
    return ""


def load_entries(root: Path | None = None) -> list[Entry]:
    """Every assumption and finding in the log, in the order it is written."""
    root = root or project_root()
    path = root / "docs" / "ASSUMPTIONS.md"
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")

    matches = list(_ENTRY_HEADING.finditer(text))
    entries: list[Entry] = []
    seen: set[str] = set()
    for index, match in enumerate(matches):
        identifier = match.group(2)
        if identifier in seen:  # a later heading repeating an entry, not a new one
            continue
        seen.add(identifier)
        title = match.group(4).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        where = _WHERE.search(body)
        entries.append(
            Entry(
                identifier=identifier,
                kind="assumption" if identifier.startswith("A") else "finding",
                title=title,
                anchor=_anchor_for(match.group(0)),
                where=re.sub(r"[`*]", "", where.group(1)).strip() if where else "",
                status=_status_of(identifier, title, body),
                summary=_summarise(body),
            )
        )
    return entries


def _anchor_for(heading_line: str) -> str:
    """The anchor the rendered log gives this heading, so a card can link to it."""
    from .markup import slugify

    return slugify(heading_line.lstrip("#").strip())


def to_json(root: Path | None = None) -> dict[str, object]:
    """Runs and entries together, as the browser asks for them."""
    return {
        "runs": [asdict(run) for run in load_runs(root)],
        "entries": [asdict(entry) for entry in load_entries(root)],
    }
