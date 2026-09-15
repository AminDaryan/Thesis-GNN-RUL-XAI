"""Which code produced a stored result, and when: the stamp rule 3a asks every run to carry."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

__all__ = [
    "CODE_PATHS",
    "capture_code_state",
    "current_commit",
    "environment",
    "file_fingerprints",
    "utc_stamp",
    "write_code_state",
]

# The parts of the repository on the path from the data files to a number.
CODE_PATHS = ("src", "scripts", "config", "pyproject.toml")


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


def environment() -> dict[str, object]:
    """The Python, the machine and the version of every library a number depends on."""
    versions: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "torch", "pyyaml"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    details: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": versions,
    }
    try:
        import torch

        details["torch_threads"] = torch.get_num_threads()
        details["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - PyTorch is optional
        pass
    return details


def file_fingerprints(paths: list[Path]) -> dict[str, str]:
    """The SHA-256 of each file's bytes, keyed by file name."""
    fingerprints: dict[str, str] = {}
    for path in paths:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        fingerprints[Path(path).name] = digest.hexdigest()
    return fingerprints


def capture_code_state(cwd: Path | None = None) -> dict[str, object]:
    """Read the uncommitted changes to the code now, before anything runs."""
    where = cwd or Path(__file__).resolve().parent
    empty: dict[str, object] = {"status": [], "patch": "", "untracked": {}}
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10, cwd=where,
        )
        if top.returncode != 0:
            return empty
        root = Path(top.stdout.strip())
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS],
            capture_output=True, text=True, timeout=10, cwd=root,
        ).stdout.splitlines()
        patch = subprocess.run(
            ["git", "diff", "HEAD", "--", *CODE_PATHS],
            capture_output=True, text=True, timeout=60, cwd=root,
        ).stdout
    except Exception:  # pragma: no cover - git absent, or not a clone
        return empty
    # A patch cannot hold a file git has never seen, so those are read whole.
    untracked = {
        line[3:].strip().strip('"'): (root / line[3:].strip().strip('"')).read_bytes()
        for line in status
        if line.startswith("??") and "__pycache__" not in line
        and (root / line[3:].strip().strip('"')).is_file()
    }
    return {"status": status, "patch": patch, "untracked": untracked}


def write_code_state(directory: Path, state: dict[str, object]) -> list[str]:
    """Write what capture_code_state() read into the run's folder; return the changed files."""
    status = list(state["status"])  # type: ignore[call-overload]
    if not status:
        return []
    folder = Path(directory) / "code_state"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "status.txt").write_text("\n".join(status) + "\n", encoding="utf-8")
    (folder / "uncommitted.patch").write_text(str(state["patch"]), encoding="utf-8")
    for relative, content in dict(state["untracked"]).items():  # type: ignore[call-overload]
        target = folder / "untracked" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return status
