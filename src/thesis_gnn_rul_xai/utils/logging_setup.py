"""
Logging
=======

The pipeline prints short progress messages so that a reader can see what the
code did and in which order. Python's :mod:`logging` module is used rather than
``print`` statements, because logging can be switched off, redirected to a
file, or filtered by severity without editing the code.

Severity levels used in this project
------------------------------------
``INFO``
    Normal progress: "loaded 20631 rows", "kept 14 of 21 sensors".
``WARNING``
    Something unexpected that the code can still handle, for example an engine
    unit that is shorter than the chosen time window.
``ERROR``
    A condition that stops the pipeline.
"""

from __future__ import annotations

import logging
import sys

__all__ = ["configure_logging", "get_logger"]

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once, writing to standard output.

    Parameters
    ----------
    level : int
        Minimum severity that will be shown. Use ``logging.DEBUG`` for very
        detailed output, ``logging.WARNING`` to see only problems.

    Notes
    -----
    Calling this function repeatedly is harmless. Existing handlers are removed
    first, which prevents the duplicated log lines that otherwise appear when a
    notebook cell is executed more than once.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a module.

    Parameters
    ----------
    name : str
        Normally ``__name__``, so that log lines show which module produced
        them.
    """
    return logging.getLogger(name)
