"""Finds every document in the repository and arranges them for the website.

Read README.md in this folder. It explains everything this file does, and why
the reading order here is not the order of the directory tree.
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
# to print an honest "about n minutes" beside a long document.
_WORDS_PER_MINUTE = 220

# The eight folders under src/thesis_gnn_rul_xai/ that each hold one step of
# the method, in the order the pipeline runs them rather than alphabetically.
_CODE_FOLDERS: tuple[tuple[str, str], ...] = (
    ("data", "From the raw NASA text files to model input"),
    ("graph", "Which sensors are considered connected, and why"),
    ("models", "The network itself: the graph convolution and the full model"),
    ("training", "Fitting the network, and the settings one run needs"),
    ("evaluation", "The four measures a set of predictions is scored with"),
    ("explain", "Which sensor drove this prediction, and when"),
    ("utils", "Paths, seeds and logging, shared by everything else"),
    ("docsite", "This website itself — the only folder the pipeline never calls"),
)

# The seven files under data/, each of which now keeps its own page beside its
# own code. They are listed straight after the data folder overview, so the
# menu reads as the folder and then its parts, in the order the pipeline runs
# them. A file that has not been written yet is skipped rather than listed.
_DATA_MODULE_PAGES: tuple[tuple[str, str], ...] = (
    ("loading", "Reading the three NASA text files, and working out every row's label"),
    ("sensors", "The 21 sensors, the 14 the paper keeps, and why their order carries meaning"),
    ("labeling", "Why the label stops counting above a ceiling, measured rather than argued"),
    ("smoothing", "The filter that takes the jitter out of each engine's signals"),
    ("normalization", "Operating conditions, rescaling, and the rule that keeps test data out"),
    ("windowing", "Cutting each engine's history into fixed-length training examples"),
    ("pipeline", "All nine steps in order, and what comes out at the end"),
)

# The taught pages, in the order they are meant to be read. Each is a file in
# docs/learn/; the blurb is what the navigation shows beneath its title.
_LEARN_PAGES: tuple[tuple[str, str], ...] = (
    ("00_start_here.md", "What this project is, and how to move through it"),
    ("01_the_problem.md", "Jet engines, remaining useful life and the C-MAPSS benchmark"),
    # prose-ok: message passing — a navigation label, explained on the page itself
    ("02_graphs.md", "Graphs, message passing, and what a graph network computes"),
    ("03_explainability.md", "Shapley values, SHAP, and what an attribution means"),
    ("04_the_pipeline.md", "The eight preprocessing steps, one screen each"),
    ("05_the_model.md", "The network layer by layer, and what its shape forbids"),
    ("06_reading_results.md", "The four measures, seeds, and when a difference is real"),
    ("07_evidence.md", "Telling the paper, a choice and a measurement apart"),
)

# The notebooks, in reading order: the two that run the chain, then the folder
# of explanation methods, its README first. A README here is listed as a page,
# a notebook as a notebook; a file that does not exist yet is skipped.
_NOTEBOOKS: tuple[tuple[str, str, str], ...] = (
    ("notebooks/01_preprocessing_fd001.ipynb", "01 — Preprocessing FD001",
     "Steps 1 to 8: the raw text files become model-ready arrays"),
    ("notebooks/02_model_fd001.ipynb", "02 — Building, training and scoring",
     "Steps 9 to 11, then the paper's sensor graph set beside two others"),
    ("notebooks/xai/README.md", "Explaining a trained network",
     "What every explanation notebook shares, and why they load a stored network"),
    ("notebooks/xai/01_shap_fd001.ipynb", "xai 01 — Exact Shapley values",
     "The paper's method, for one prediction and at every cycle of one engine's life"),
    ("notebooks/xai/02_occlusion_fd001.ipynb", "xai 02 — Occlusion",
     "A cheaper method that is not in the paper, set beside the Shapley values"),
)

_HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_MARKUP = re.compile(r"[*_`>#\[\]()!|]")


@dataclass(frozen=True)
class Document:
    """One page of the website: where its text lives and how it is listed."""

    path: str
    title: str
    blurb: str = ""
    kind: str = "reference"
    words: int = 0

    @property
    def minutes(self) -> int:
        """About how long this document takes to read, rounded up to a minute."""
        return max(1, round(self.words / _WORDS_PER_MINUTE))


@dataclass(frozen=True)
class Section:
    """One group in the navigation sidebar, with the pages it contains."""

    key: str
    title: str
    blurb: str
    documents: tuple[Document, ...] = ()


@dataclass
class Catalog:
    """Every document the website can show, grouped into sections."""

    sections: tuple[Section, ...] = ()
    by_path: dict[str, Document] = field(default_factory=dict)

    def get(self, path: str) -> Document | None:
        """The document stored at this repository-relative path, if it exists."""
        return self.by_path.get(path)

    def to_dict(self) -> dict[str, object]:
        """The whole catalog as plain data, for the browser to draw a menu."""
        return {
            "sections": [
                {
                    "key": section.key,
                    "title": section.title,
                    "blurb": section.blurb,
                    "documents": [
                        {**asdict(doc), "minutes": doc.minutes}
                        for doc in section.documents
                    ],
                }
                for section in self.sections
            ]
        }


def _count_words(text: str) -> int:
    """How many words of prose a Markdown file holds, ignoring its code."""
    prose = _MARKUP.sub(" ", _CODE_FENCE.sub(" ", text))
    return len(prose.split())


def _title_of(text: str, fallback: str) -> str:
    """The first level-one heading of a Markdown file, or a given fallback."""
    match = _HEADING.search(text)
    return _MARKUP.sub("", match.group(1)).strip() if match else fallback


def _describe(root: Path, relative: str, blurb: str, kind: str) -> Document | None:
    """Build the catalog entry for one file, or nothing if it is not there."""
    path = root / relative
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return Document(
        path=relative,
        title=_title_of(text, relative.rsplit("/", 1)[-1]),
        blurb=blurb,
        kind=kind,
        words=_count_words(text),
    )


def _reviews(root: Path) -> list[Document]:
    """Every dated audit in docs/, newest first.

    Two naming conventions live here and both are wanted. A REVIEW_ file is an
    outside review *of this repository*; an AUDIT_ file is a check *of the
    paper* against the code, which runs in the other direction.
    """
    blurbs = {
        "REVIEW": "An outside review of this repository, and what became of "
        "each item it raised",
        "AUDIT": "Every equation, table and figure of the paper checked "
        "against the code, agreements included",
    }
    paths = sorted(
        [*(root / "docs").glob("REVIEW_*.md"), *(root / "docs").glob("AUDIT_*.md")],
        key=lambda p: p.name.split("_", 1)[-1],
        reverse=True,
    )
    found: list[Document] = []
    for path in paths:
        doc = _describe(
            root,
            f"docs/{path.name}",
            blurbs.get(path.name.split("_", 1)[0], "A dated audit"),
            "reference",
        )
        if doc is not None:
            found.append(doc)
    return found


def build_catalog(root: Path | None = None) -> Catalog:
    """Read the repository and return every document, grouped for the menu."""
    root = root or project_root()

    def collect(entries: list[Document | None]) -> tuple[Document, ...]:
        return tuple(doc for doc in entries if doc is not None)

    learn = collect(
        [
            _describe(root, f"docs/learn/{name}", blurb, "learn")
            for name, blurb in _LEARN_PAGES
        ]
    )

    notebook_entries: list[Document | None] = [
        _describe(
            root,
            "notebooks/README.md",
            "What every notebook shares: how it is laid out, and where its code lives",
            "reference",
        ),
    ]
    for path, title, blurb in _NOTEBOOKS:
        if path.endswith(".md"):
            notebook_entries.append(_describe(root, path, blurb, "reference"))
        elif (root / path).is_file():
            notebook_entries.append(
                Document(path=path, title=title, blurb=blurb, kind="notebook")
            )
    notebooks = collect(notebook_entries)

    overview = collect(
        [
            _describe(
                root,
                "README.md",
                "The project in full: what it is, how to run it, what it found",
                "reference",
            ),
            _describe(
                root,
                "src/thesis_gnn_rul_xai/README.md",
                "The package as a whole, and every setting it takes",
                "reference",
            ),
            _describe(
                root,
                "CLAUDE.md",
                "The working rules this repository holds itself to",
                "reference",
            ),
            _describe(
                root,
                "data/raw/README.md",
                "Where the NASA files come from and how they are checked",
                "reference",
            ),
        ]
    )

    code_entries: list[Document | None] = []
    for folder, blurb in _CODE_FOLDERS:
        code_entries.append(
            _describe(
                root, f"src/thesis_gnn_rul_xai/{folder}/README.md", blurb, "reference"
            )
        )
        if folder == "data":
            code_entries += [
                _describe(
                    root,
                    f"src/thesis_gnn_rul_xai/data/{module}/README.md",
                    module_blurb,
                    "reference",
                )
                for module, module_blurb in _DATA_MODULE_PAGES
            ]
    code = collect(code_entries)

    evidence = collect(
        [
            _describe(
                root,
                "docs/ASSUMPTIONS.md",
                "Every choice the paper left open, and every finding about it",
                "reference",
            ),
            _describe(
                root,
                "docs/REFERENCES.md",
                "Full citations, with what each work is used for",
                "reference",
            ),
            _describe(
                root,
                "docs/PENDING.md",
                "Known to be wrong, unmeasured or unfinished, and how to finish it",
                "reference",
            ),
            *_reviews(root),
        ]
    )

    sections = (
        Section(
            "learn",
            "Learn",
            "Start here. Eight pages that build the whole method from nothing.",
            learn,
        ),
        Section(
            "notebooks",
            "Notebooks",
            "The same chain, run on the real data, with every number shown.",
            notebooks,
        ),
        Section(
            "explore",
            "Explore",
            "Figures and tables the site computes from this repository, now.",
            (
                Document(
                    path="explore/playground",
                    title="Playground",
                    blurb="Choose the data, preprocessing, temporal stage and explanation; "
                    "train, remove parts and compare",
                    kind="explore",
                ),
                Document(
                    path="explore/graph",
                    title="The sensor graph",
                    blurb="All 14 sensors, their 41 edges and what each layer reaches",
                    kind="explore",
                ),
                Document(
                    path="explore/architecture",
                    title="The network",
                    blurb="Every layer and its parameter count, built and measured live",
                    kind="explore",
                ),
                Document(
                    path="explore/results",
                    title="Stored runs",
                    blurb="Every result in reports/, with the commit that produced it",
                    kind="explore",
                ),
                Document(
                    path="explore/assumptions",
                    title="Assumptions and findings",
                    blurb="Every numbered choice and finding, as cards you can filter",
                    kind="explore",
                ),
            ),
        ),
        Section("overview", "The project", "The repository described in its own words.", overview),
        Section("code", "The code, folder by folder", "One page per step of the method.", code),
        Section("evidence", "Evidence", "What was chosen, what was found, and what was audited.", evidence),
    )

    catalog = Catalog(sections=sections)
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
    """Every place a phrase appears, as a document, a line and its context."""
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
        for number, line in enumerate(text.splitlines(), start=1):
            position = line.lower().find(needle)
            if position < 0:
                continue
            start = max(0, position - 60)
            hits.append(
                {
                    "path": document.path,
                    "title": document.title,
                    "line": number,
                    "context": line[start : position + len(needle) + 90].strip(),
                }
            )
            if len(hits) >= 200:
                return hits
    return hits
