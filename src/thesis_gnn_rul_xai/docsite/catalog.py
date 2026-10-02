"""Finds every document in the repository and arranges them for the website.

Read README.md in this folder. It explains everything this file does, and why
the menu's order is not the order of the directory tree.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..utils.paths import project_root

__all__ = [
    "Catalog",
    "Document",
    "Section",
    "build_catalog",
    "read_document",
    "search",
]

# How many words a minute this project assumes a reader gets through. Used only
# to print an honest "about n minutes" at the foot of a long document.
_WORDS_PER_MINUTE = 220

# The menu, written out by hand: each section has a key, the name the menu
# shows, and its pages as (path, name, kind, depth). The name is the page's one
# name: the site shows it in the page's first heading and in the browser tab as
# well as in the menu, whatever heading the file itself opens with, so a reader
# never meets two names for one page. Depth 1 indents a page under the one
# above it. A page whose file is missing is left out rather than listed.
#
# Some files are deliberately not listed. README.md is the repository's front
# page on GitHub, and each of its sections has a menu page of its own. A page
# that is not listed still opens from a link, under its own heading.
_Entry = tuple[str, str, str, int]

_PRIMARY: _Entry = ("explore/playground", "Playground", "explore", 0)

_LEARN: tuple[_Entry, ...] = (
    ("docs/learn/00_start_here.md", "Start here", "learn", 0),
    ("docs/learn/01_the_problem.md", "Data", "learn", 0),
    ("docs/learn/04_the_pipeline.md", "Preprocessing", "learn", 0),
    ("docs/learn/09_temporal.md", "Temporal algorithm", "learn", 0),
    ("docs/learn/02_graphs.md", "GNN", "learn", 0),
    ("docs/learn/05_the_model.md", "Architecture and training", "learn", 1),
    ("docs/learn/03_explainability.md", "XAI", "learn", 0),
    ("docs/learn/10_dynamic_shap.md", "Dynamic SHAP", "learn", 1),
    ("docs/learn/11_gnn_lrp.md", "GNN-LRP", "learn", 1),
    ("docs/learn/08_worked_examples.md", "Follow one real example", "learn", 0),
    ("docs/learn/06_reading_results.md", "Reading a result", "learn", 0),
    ("docs/learn/07_evidence.md", "Paper, choice or measurement", "learn", 0),
)

_NOTEBOOKS: tuple[_Entry, ...] = (
    ("notebooks/README.md", "About the notebooks", "reference", 0),
    ("notebooks/01_preprocessing_fd001.ipynb", "Preprocessing FD001", "notebook", 0),
    ("notebooks/02_model_fd001.ipynb", "Building and training the network", "notebook", 0),
    ("notebooks/xai/README.md", "About the explanation notebooks", "reference", 0),
    ("notebooks/xai/01_shap_fd001.ipynb", "Exact Shapley values", "notebook", 0),
    ("notebooks/xai/02_occlusion_fd001.ipynb", "Occlusion: hiding one sensor", "notebook", 0),
    ("notebooks/xai/03_walks_fd001.ipynb", "Walks through the sensor graph", "notebook", 0),
)

# The folders under src/thesis_gnn_rul_xai/ in the order the pipeline runs
# them rather than alphabetically, each followed by the pages of its parts.
_PACKAGE = "src/thesis_gnn_rul_xai"
_CODE: tuple[_Entry, ...] = (
    (f"{_PACKAGE}/README.md", "Package overview", "reference", 0),
    ("data/raw/README.md", "Getting the NASA data", "reference", 0),
    (f"{_PACKAGE}/data/README.md", "Preparing the data", "reference", 0),
    (f"{_PACKAGE}/data/loading/README.md", "Reading the files", "reference", 1),
    (f"{_PACKAGE}/data/sensors/README.md", "Choosing the sensors", "reference", 1),
    (f"{_PACKAGE}/data/labeling/README.md", "Capping the label", "reference", 1),
    (f"{_PACKAGE}/data/smoothing/README.md", "Smoothing", "reference", 1),
    (f"{_PACKAGE}/data/normalization/README.md", "Normalising", "reference", 1),
    (f"{_PACKAGE}/data/windowing/README.md", "Cutting windows", "reference", 1),
    (f"{_PACKAGE}/data/pipeline/README.md", "All steps in order", "reference", 1),
    (f"{_PACKAGE}/graph/README.md", "Building the graph", "reference", 0),
    (f"{_PACKAGE}/models/README.md", "Building the network", "reference", 0),
    (f"{_PACKAGE}/models/temporal/README.md", "What each sensor carries", "reference", 1),
    (f"{_PACKAGE}/training/README.md", "Training", "reference", 0),
    (f"{_PACKAGE}/evaluation/README.md", "Scoring", "reference", 0),
    (f"{_PACKAGE}/explain/README.md", "Explaining", "reference", 0),
    (f"{_PACKAGE}/utils/README.md", "Shared helpers", "reference", 0),
    (f"{_PACKAGE}/docsite/README.md", "This website", "reference", 0),
    (f"{_PACKAGE}/docsite/playground/README.md", "The playground", "reference", 1),
)

# The dated audits and reviews are found on disk rather than listed, so a new
# one appears without an edit here. They sit between these two groups.
_CHECKS_BEFORE: tuple[_Entry, ...] = (
    ("docs/ASSUMPTIONS.md", "Assumptions log", "reference", 0),
    ("docs/PAPER_TO_CODE.md", "Paper-to-code table", "reference", 0),
    ("docs/LITERATURE_BASIS.md", "Published basis for each choice", "reference", 0),
    ("docs/PENDING.md", "Open work", "reference", 0),
    ("docs/EXPERIMENT_A-07_mean_window.md", "Planned experiment: last cycle or mean", "reference", 0),
    ("docs/DESIGN_swappable_stages.md", "Design of the swappable stages", "reference", 0),
)
_CHECKS_AFTER: tuple[_Entry, ...] = (
    ("CLAUDE.md", "Working rules", "reference", 0),
)

# One page per method or dataset the project builds on, each explained from its
# original paper, then the list of every work cited with a link to its PDF. The
# year names the paper, and keeps "GNN-LRP" apart from the taught page of that name.
_RESOURCES: tuple[_Entry, ...] = (
    ("docs/resources/cmapss.md", "C-MAPSS dataset (2008)", "reference", 0),
    ("docs/resources/lstm.md", "LSTM (1997)", "reference", 0),
    ("docs/resources/gcn.md", "GCN (2017)", "reference", 0),
    ("docs/resources/shap.md", "SHAP (2017)", "reference", 0),
    ("docs/resources/gnn_lrp.md", "GNN-LRP (2022)", "reference", 0),
    ("docs/REFERENCES.md", "References", "reference", 0),
)

# A REVIEW_ file is an outside review *of this repository*; an AUDIT_ file is a
# check *of the paper* against the code, which runs in the other direction.
_DATED_NAMES = {
    "REVIEW": "Outside review",
    "AUDIT": "Audit",
    "AUDIT_scientific_validity": "Validity audit",
}

# Dated files kept in docs/ but not listed. The paper-against-code audit is the
# record of how each row of docs/PAPER_TO_CODE.md was first checked; the table,
# which a test keeps current, is the page the menu offers for the same question.
_UNLISTED_DATED = frozenset({"AUDIT_paper_against_code"})

_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_MARKUP = re.compile(r"[*_`>#\[\]()!|]")
_DATED = re.compile(r"^(REVIEW|AUDIT)_(\d{4}-\d{2}-\d{2})(?:_(.+))?\.md$")


@dataclass(frozen=True)
class Document:
    """One page of the website: where its text lives and how it is listed."""

    path: str
    title: str
    kind: str = "reference"
    words: int = 0
    depth: int = 0

    @property
    def minutes(self) -> int:
        """About how long this document takes to read, rounded up to a minute."""
        return max(1, round(self.words / _WORDS_PER_MINUTE))


@dataclass(frozen=True)
class Section:
    """One group in the navigation sidebar, with the pages it contains."""

    key: str
    title: str
    documents: tuple[Document, ...] = ()


@dataclass
class Catalog:
    """Every document the website can show, grouped into sections."""

    sections: tuple[Section, ...] = ()
    primary: Document | None = None
    by_path: dict[str, Document] = field(default_factory=dict)

    def get(self, path: str) -> Document | None:
        """The document stored at this repository-relative path, if it exists."""
        return self.by_path.get(path)

    def to_dict(self) -> dict[str, object]:
        """The whole catalog as plain data, for the browser to draw a menu."""

        def plain(doc: Document) -> dict[str, object]:
            return {**asdict(doc), "minutes": doc.minutes}

        return {
            "primary": plain(self.primary) if self.primary else None,
            "sections": [
                {
                    "key": section.key,
                    "title": section.title,
                    "documents": [plain(doc) for doc in section.documents],
                }
                for section in self.sections
            ],
        }


def _count_words(text: str) -> int:
    """How many words of prose a Markdown file holds, ignoring its code."""
    prose = _MARKUP.sub(" ", _CODE_FENCE.sub(" ", text))
    return len(prose.split())


def _describe(root: Path, entry: _Entry) -> Document | None:
    """The catalog entry for one line of the menu, or nothing if its file is missing."""
    path, title, kind, depth = entry
    if kind == "explore":
        return Document(path=path, title=title, kind=kind, depth=depth)
    file = root / path
    if not file.is_file():
        return None
    words = 0 if kind == "notebook" else _count_words(file.read_text(encoding="utf-8"))
    return Document(path=path, title=title, kind=kind, words=words, depth=depth)


def _dated(root: Path) -> list[_Entry]:
    """Every dated audit and review in docs/, newest first, named for the menu."""
    found: list[tuple[str, _Entry]] = []
    for file in (root / "docs").glob("*.md"):
        match = _DATED.match(file.name)
        if not match:
            continue
        kind, date, topic = match.groups()
        if f"{kind}_{topic}" in _UNLISTED_DATED:
            continue
        name = _DATED_NAMES.get(f"{kind}_{topic}", _DATED_NAMES[kind])
        found.append((date, (f"docs/{file.name}", f"{name}, {date}", "reference", 0)))
    return [entry for _, entry in sorted(found, reverse=True)]


def build_catalog(root: Path | None = None) -> Catalog:
    """Read the repository and return every document, grouped for the menu."""
    root = root or project_root()

    def collect(entries: tuple[_Entry, ...] | list[_Entry]) -> tuple[Document, ...]:
        documents = (_describe(root, entry) for entry in entries)
        return tuple(doc for doc in documents if doc is not None)

    sections = (
        Section("learn", "Learn the method", collect(_LEARN)),
        Section("notebooks", "Notebooks", collect(_NOTEBOOKS)),
        Section("code", "Code", collect(_CODE)),
        Section(
            "checks",
            "Choices and checks",
            collect([*_CHECKS_BEFORE, *_dated(root), *_CHECKS_AFTER]),
        ),
        Section("resources", "Resources", collect(_RESOURCES)),
    )

    catalog = Catalog(sections=sections, primary=_describe(root, _PRIMARY))
    if catalog.primary is not None:
        catalog.by_path[catalog.primary.path] = catalog.primary
    for section in sections:
        for doc in section.documents:
            catalog.by_path[doc.path] = doc
    return catalog


def read_document(path: str, root: Path | None = None) -> str:
    """The text of one document, refusing any path outside the repository."""
    root = (root or project_root()).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"{path!r} points outside the project directory.")
    if not target.is_file():
        raise FileNotFoundError(f"No file at {path!r}.")
    return target.read_text(encoding="utf-8")


def search(query: str, catalog: Catalog, root: Path | None = None) -> list[dict[str, object]]:
    """Every line a phrase appears on, as its document and the words around it."""
    root = root or project_root()
    needle = query.strip().lower()
    if len(needle) < 2:
        return []

    hits: list[dict[str, object]] = []
    for document in catalog.by_path.values():
        if document.kind in ("notebook", "explore"):
            continue
        try:
            text = read_document(document.path, root)
        except (OSError, ValueError):
            continue
        for line in text.splitlines():
            position = line.lower().find(needle)
            if position < 0:
                continue
            start = max(0, position - 60)
            hits.append(
                {
                    "path": document.path,
                    "title": document.title,
                    "context": line[start : position + len(needle) + 90].strip(),
                }
            )
            if len(hits) >= 200:
                return hits
    return hits
