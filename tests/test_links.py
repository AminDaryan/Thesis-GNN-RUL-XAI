"""Every relative link in the documents reaches a file, and every #anchor a heading the website gives that file."""

from __future__ import annotations

import json
import os
import posixpath
import re
from pathlib import Path
from urllib.parse import unquote

import pytest

from thesis_gnn_rul_xai.utils.paths import log_entry_files, project_root

mistune = pytest.importorskip("mistune")
markup = pytest.importorskip("thesis_gnn_rul_xai.docsite.markup")

ROOT = project_root()
# The plug-ins the website's own renderer uses, so a link is read the way the site reads it.
PLUGINS = ["table", "strikethrough", "task_lists", "footnotes", "def_list"]
# Copies of the cited papers are not committed; the manifest names the ones a link may expect.
REFERENCES = ROOT / "docs" / "references"
LISTED_PAPERS = {
    paper["file"]
    for paper in json.loads((REFERENCES / "manifest.json").read_text(encoding="utf-8"))["papers"]
}


def _documents() -> list[Path]:
    """The Markdown files whose links are checked."""
    found = [ROOT / "README.md", ROOT / "CLAUDE.md", ROOT / "AGENTS.md"]
    found += sorted((ROOT / "docs").rglob("*.md"))
    found += sorted((ROOT / "src").rglob("README.md"))
    found += sorted((ROOT / "notebooks").rglob("README.md"))
    found += sorted((ROOT / "data").rglob("README.md"))
    return [p for p in found if p.is_file() and ".ipynb_checkpoints" not in p.parts]


def _notebooks() -> list[Path]:
    """The notebooks, whose Markdown cells are checked."""
    return [p for p in sorted((ROOT / "notebooks").rglob("*.ipynb")) if ".ipynb_checkpoints" not in p.parts]


def _links(text: str) -> list[str]:
    """The address of every link and image in one piece of Markdown, outside code."""
    found: list[str] = []

    def walk(tokens: list[dict]) -> None:
        for token in tokens:
            if token.get("type") in ("link", "image"):
                found.append(token["attrs"]["url"])
            if token.get("type") in ("inline_html", "block_html"):
                found.extend(_HTML_ADDRESS.findall(token.get("raw", "")))
            if isinstance(token.get("children"), list):
                walk(token["children"])

    walk(mistune.create_markdown(renderer=None, plugins=PLUGINS)(text))
    return found


# The address in a link or picture written as HTML, which Markdown passes through untouched.
_HTML_ADDRESS = re.compile(r'(?:href|src)="([^"]+)"')
# A website address that opens one entry of the log, such as #/entry/F-01.
_ENTRY_ROUTE = re.compile(r"#/entry/([AF]-\d{2})$")

_listings: dict[Path, set[str]] = {}


def _exists_exactly(relative: str) -> bool:
    """Whether a path exists with every name spelled in the same case, as Linux and GitHub require."""
    folder = ROOT
    for part in relative.split("/"):
        if folder not in _listings:
            _listings[folder] = set(os.listdir(folder)) if folder.is_dir() else set()
        if part not in _listings[folder]:
            return False
        folder = folder / part
    return True


_anchors: dict[str, set[str]] = {}


def _anchors_of(relative: str) -> set[str]:
    """Every heading anchor the website gives one Markdown file, with its counter for repeats."""
    if relative not in _anchors:
        rendered = markup.render_markdown((ROOT / relative).read_text(encoding="utf-8"), source_path=relative)
        _anchors[relative] = {heading.anchor for heading in rendered.headings}
    return _anchors[relative]


def _problem(source: str, url: str) -> str | None:
    """What is wrong with one link from one document, or nothing."""
    if url.startswith(("http://", "https://", "mailto:")):
        return None
    if url.startswith("#/"):
        entry = _ENTRY_ROUTE.match(url)
        if entry and entry.group(1) not in {p.stem for p in log_entry_files(ROOT)}:
            return f"{url}: the log has no entry {entry.group(1)}"
        page = url[len("#/doc/"):].partition("#")[0] if url.startswith("#/doc/") else ""
        if page and not _exists_exactly(page):
            return f"{url}: no file at {page}"
        return None
    target, _, fragment = url.partition("#")
    resolved = source
    if target:
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), unquote(target)))
        if resolved.startswith(".."):
            return f"{url} leads outside the project"
        local_only = resolved.startswith("docs/references/") and Path(resolved).name in LISTED_PAPERS
        if not _exists_exactly(resolved) and not local_only:
            return f"{url}: no file at {resolved}"
    checkable = fragment and resolved.endswith(".md") and (ROOT / resolved).is_file()
    if checkable and unquote(fragment) not in _anchors_of(resolved):
        return f"{url}: {resolved} has no heading with the anchor #{fragment}"
    return None


def test_every_relative_link_in_the_documents_resolves():
    broken = []
    for path in _documents():
        source = path.relative_to(ROOT).as_posix()
        for url in _links(path.read_text(encoding="utf-8")):
            if problem := _problem(source, url):
                broken.append(f"{source}: {problem}")
    assert broken == []


def test_every_relative_link_in_a_notebook_text_cell_resolves():
    broken = []
    for path in _notebooks():
        source = path.relative_to(ROOT).as_posix()
        for number, cell in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"]):
            if cell["cell_type"] != "markdown":
                continue
            text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for url in _links(text):
                if problem := _problem(source, url):
                    broken.append(f"{source}, cell {number}: {problem}")
    assert broken == []


def test_the_checker_reports_a_missing_file_and_a_missing_anchor():
    assert _problem("README.md", "docs/assumptions/A-07.md") is None
    assert _problem("docs/learn/10_dynamic_shap.md", "../assumptions/A-20.md") is None
    assert "no file" in _problem("README.md", "docs/ASSUMPTIONS" + ".md")
    assert "no heading" in _problem("README.md", "#no-such-section")
    # A repeated heading's anchor carries a counter, as the website numbers it.
    assert _problem("docs/assumptions/README.md", "A-06.md#original-entry") is None
    assert "no heading" in _problem("docs/assumptions/README.md", "A-06.md#original-entry-1")
    # A name in the wrong case exists on Windows and nowhere else.
    assert "no file" in _problem("docs/assumptions/README.md", "../findings/f-01.md")
    # Website addresses: an entry that exists, one that does not, a page that moved.
    assert _problem("docs/learn/01_the_problem.md", "#/entry/F-01") is None
    assert "no entry" in _problem("docs/learn/01_the_problem.md", "#/entry/A-99")
    assert "no file" in _problem("README.md", "#/doc/docs/ASSUMPTIONS" + ".md")
    # A link written as HTML is read too.
    assert _links('See <a href="../audits/NOPE.md">this</a>.') == ["../audits/NOPE.md"]
