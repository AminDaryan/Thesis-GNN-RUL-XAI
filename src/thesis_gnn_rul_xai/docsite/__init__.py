"""The local documentation website: every README and notebook, in a browser.

Read README.md in this folder. It explains everything this package does.
"""

from __future__ import annotations

from .catalog import Document, Section, build_catalog
from .server import DEFAULT_PORT, serve

__all__ = [
    "DEFAULT_PORT",
    "Document",
    "Section",
    "build_catalog",
    "serve",
]
