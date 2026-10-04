"""Finds the project folder and the data, report and figure folders inside it.

Read README.md in this folder. It explains everything this file does.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "ensure_dir",
    "figures_dir",
    "log_entry_files",
    "processed_data_dir",
    "project_root",
    "raw_data_dir",
    "reports_dir",
]

# A marker file that exists only in the project root directory.
_ROOT_MARKER = "pyproject.toml"


def project_root(start: Path | None = None) -> Path:
    """Return the full path of the folder that holds pyproject.toml."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / _ROOT_MARKER).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate the project root: no '{_ROOT_MARKER}' found in "
        f"'{here}' or any of its parent directories."
    )


def log_entry_files(root: Path | None = None) -> list[Path]:
    """Every docs/assumptions/A-NN.md in number order, then every docs/findings/F-NN.md."""
    root = root or project_root()
    found: list[Path] = []
    for folder, letter in (("assumptions", "A"), ("findings", "F")):
        named = re.compile(rf"{letter}-\d{{2}}\.md")
        files = [f for f in (root / "docs" / folder).glob(f"{letter}-*.md") if named.fullmatch(f.name)]
        found += sorted(files, key=lambda f: int(f.stem[2:]))
    return found


def ensure_dir(path: Path) -> Path:
    """Create a directory (and any missing parents) and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_data_dir() -> Path:
    """Directory holding the original C-MAPSS text files. Treat as read-only."""
    return project_root() / "data" / "raw"


def processed_data_dir() -> Path:
    """Directory holding arrays produced by the preprocessing pipeline."""
    return ensure_dir(project_root() / "data" / "processed")


def reports_dir() -> Path:
    """Directory for generated tables, logs and summaries."""
    return ensure_dir(project_root() / "reports")


def figures_dir() -> Path:
    """Directory for generated figures."""
    return ensure_dir(project_root() / "reports" / "figures")
