"""Which code produced a stored result, and when: the stamp rule 3a asks every run to carry."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["current_commit", "utc_stamp"]


def current_commit(cwd: Path | None = None) -> str:
    """The short commit, with "-dirty" if anything is uncommitted, or "unknown"."""
    where = cwd or Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, cwd=where,
        )
        if head.returncode != 0:
            return "unknown"
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, cwd=where,
        )
        return head.stdout.strip() + ("-dirty" if status.stdout.strip() else "")
    except Exception:  # pragma: no cover - git absent, or not a clone
        return "unknown"


def utc_stamp() -> str:
    """The current time in UTC, in the compact form run directory names use."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
