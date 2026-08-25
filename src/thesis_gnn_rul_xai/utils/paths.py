"""
Project paths
=============

Every script and notebook in this project needs to know where the project root
is, so that data and results are always written to the same place no matter
which directory the code is started from.

The root is found by walking up the directory tree from this file until a
directory containing a ``pyproject.toml`` file is found. This is the usual
convention in Python projects and means the code works identically whether it
is started from a notebook, from a terminal, or from an automated test.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "project_root",
    "raw_data_dir",
    "processed_data_dir",
    "reports_dir",
    "figures_dir",
    "ensure_dir",
]

# A marker file that exists only in the project root directory.
_ROOT_MARKER = "pyproject.toml"


def project_root(start: Path | None = None) -> Path:
    """Return the absolute path of the project root directory.

    Parameters
    ----------
    start : Path, optional
        Directory to start searching from. Defaults to the directory that
        contains this source file.

    Returns
    -------
    Path
        Absolute path to the directory that contains ``pyproject.toml``.

    Raises
    ------
    FileNotFoundError
        If no parent directory contains the marker file. This normally means
        the package was copied somewhere without its project files.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / _ROOT_MARKER).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate the project root: no '{_ROOT_MARKER}' found in "
        f"'{here}' or any of its parent directories."
    )


def ensure_dir(path: Path) -> Path:
    """Create a directory (and any missing parents) and return it.

    This is safe to call repeatedly: if the directory already exists, nothing
    happens and no error is raised.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_data_dir() -> Path:
    """Directory holding the original, unmodified C-MAPSS text files.

    Files in this directory are treated as read-only. Nothing in this project
    ever writes to them, which guarantees that the original measurements can
    always be recovered.
    """
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
