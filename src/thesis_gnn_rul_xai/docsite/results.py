"""Reads the assumptions log and the last re-check of its numbers, for the website.

Read README.md in this folder. It explains everything this file does. The
stored training runs are read by playground/runs.py, which the playground draws from.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..utils.paths import log_entry_files, project_root

__all__ = [
    "Entry",
    "Verification",
    "latest_verification",
    "load_entries",
]

# The first line of an entry's file, such as "# A-01 — The label ceiling is
# 125 cycles" in docs/assumptions/A-01.md or "# A-08 — WITHDRAWN: the paper
# does state the order" in docs/assumptions/A-08.md.
_ENTRY_HEADING = re.compile(r"#\s+(~~)?([AF]-\d{2})(~~)?\s*[—-]+\s*(.+?)\s*$", re.MULTILINE)

# The line inside an entry that names the code the choice is made in.
_WHERE = re.compile(r"^\*\*Where:\*\*\s*(.+?)$", re.MULTILINE)

# The log marks a retired entry by opening its title with a capitalised status
# word, as in "WITHDRAWN: the paper does state the order". Matching the word
# anywhere in the title instead would catch ordinary prose: A-25's title says
# an order "reverses" and that entry is perfectly current.
_RETIRED = re.compile(r"^\s*(WITHDRAWN(\s+AND\s+REVERSED)?|REVERSED|SUPERSEDED)\b")


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
    """One assumption or finding: its heading, the file it is in and whether it still stands."""

    identifier: str
    kind: str
    title: str
    path: str
    where: str
    status: str
    summary: str


def latest_verification(root: Path | None = None) -> Verification | None:
    """The most recent run of verify_claims.py, or nothing if it never ran."""
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
    # An entry may instead open its body with a bold "Superseded <date> by
    # A-xx." line; A-06 and A-18 did until both were reinstated on 2026-09-08.
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
    """Every assumption in number order, then every finding, each read from its own file."""
    root = root or project_root()
    entries: list[Entry] = []
    for path in log_entry_files(root):
        text = path.read_text(encoding="utf-8")
        match = _ENTRY_HEADING.match(text)
        if not match:  # a file that does not open with its entry's heading is not read as one
            continue
        identifier = match.group(2)
        title = match.group(4).strip()
        body = text[match.end() :]
        where = _WHERE.search(body)
        entries.append(
            Entry(
                identifier=identifier,
                kind="assumption" if identifier.startswith("A") else "finding",
                title=title,
                path=path.relative_to(root).as_posix(),
                where=re.sub(r"[`*]", "", where.group(1)).strip() if where else "",
                status=_status_of(identifier, title, body),
                summary=_summarise(body),
            )
        )
    return entries

