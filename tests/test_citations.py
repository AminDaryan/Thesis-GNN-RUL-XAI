"""Every taught page names its published sources the way a paper does.

A claim taken from a published work carries an author-date citation in the
text, such as (Saxena et al., 2008, p. 5), and the page ends with a list of
the works it cites. These tests hold the two halves together: every work
cited in the text is in the page's list, every work in the list is cited, and
every entry is the one docs/REFERENCES.md gives, with a link to a copy that
the manifest knows about. Whether a citation supports its sentence is checked
by reading the source; no test can do that.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted((ROOT / "docs" / "learn").glob("*.md")) + sorted(
    (ROOT / "docs" / "resources").glob("*.md")
)
HEADING = "## References"

# NASA's readme is cited by its group author, shortened after its first use.
GROUP_ABBREVIATIONS = {"NASA Prognostics Center of Excellence": "NASA PCoE"}

_FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.MULTILINE | re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_YEAR = re.compile(r"\((\d{4}|n\.d\.)\)\.")


def _flat(text: str) -> str:
    """Markdown emphasis removed and whitespace collapsed, for comparing entries."""
    return " ".join(text.replace("*", "").split())


def _first_author(entry: str) -> str:
    """The surname of the first author, or the whole name of a group author."""
    head = entry[: _YEAR.search(entry).start()].strip()
    return head.split(",")[0].rstrip(". ").strip()


def _reading_list() -> list[tuple[str, str, str]]:
    """(first author, year, entry) for every work in docs/REFERENCES.md."""
    text = (ROOT / "docs" / "REFERENCES.md").read_text(encoding="utf-8")
    works = []
    for paragraph in text.split("\n\n"):
        if paragraph.startswith("**") and _YEAR.search(paragraph):
            entry = _flat(paragraph)
            works.append((_first_author(entry), _YEAR.search(entry).group(1), entry))
    return works


def _held_files() -> set[str]:
    manifest = json.loads(
        (ROOT / "docs" / "references" / "manifest.json").read_text(encoding="utf-8")
    )
    entries = next(value for value in manifest.values() if isinstance(value, list))
    return {entry["file"] for entry in entries}


def _split(page: Path) -> tuple[str, str]:
    """The page's prose with code removed, and the text of its reference list."""
    text = page.read_text(encoding="utf-8")
    body, _, listing = text.partition("\n" + HEADING + "\n")
    body = _INLINE_CODE.sub(" ", _FENCE.sub(" ", body))
    return " ".join(body.split()), listing


def _listed(listing: str) -> list[str]:
    """Each bullet of the reference list, joined onto one line."""
    entries: list[str] = []
    for line in listing.splitlines():
        if line.startswith("- "):
            entries.append(line[2:])
        elif line.startswith("  ") and entries:
            entries[-1] += " " + line.strip()
        elif line.startswith("#"):
            break
    return [_flat(entry) for entry in entries]


def _cites(body: str, author: str, year: str) -> bool:
    """Whether the prose cites this author and year in either APA form."""
    names = [author] + (
        [GROUP_ABBREVIATIONS[author]] if author in GROUP_ABBREVIATIONS else []
    )
    year = re.escape(year)
    for name in names:
        pattern = (
            re.escape(name)
            + r"(?:\s+et\s+al\.|\s+and\s+colleagues|,?\s+(?:&|and)\s+[A-Z][^()]{0,60}?"
            + r"|\s+\[[^\]]+\])?"
            + r"(?:'s)?,?\s+\(?"
            + year
            + r"[a-z]?(?![0-9])"
        )
        if re.search(pattern, body):
            return True
    return False


WORKS = _reading_list()


def test_the_reading_list_is_parsed():
    authors = {author for author, _, _ in WORKS}
    assert {"Saxena", "Wen", "Kipf", "NASA Prognostics Center of Excellence"} <= authors


@pytest.mark.parametrize("page", PAGES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_the_page_ends_with_its_references(page):
    text = _FENCE.sub(" ", page.read_text(encoding="utf-8"))
    level_two = re.findall(r"^## .+$", text, re.MULTILINE)
    assert level_two and level_two[-1].strip() == HEADING, (page.name, level_two[-1:])


@pytest.mark.parametrize("page", PAGES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_listed_work_is_in_the_reading_list_and_cited(page):
    body, listing = _split(page)
    entries = _listed(listing)
    assert entries, f"{page.name} has an empty reference list"
    for entry in entries:
        matches = [(a, y) for a, y, full in WORKS if full in entry]
        assert matches, f"{page.name}: not as docs/REFERENCES.md gives it: {entry[:90]}"
        author, year = matches[0]
        assert _cites(body, author, year), (
            f"{page.name}: listed but never cited: {author} ({year})"
        )


@pytest.mark.parametrize("page", PAGES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_cited_work_is_listed(page):
    body, listing = _split(page)
    listed = {
        (a, y) for a, y, full in WORKS for entry in _listed(listing) if full in entry
    }
    for author, year, _ in WORKS:
        if (author, year) not in listed and _cites(body, author, year):
            pytest.fail(
                f"{page.name}: cites {author} ({year}) but its reference list does not"
            )


@pytest.mark.parametrize("page", PAGES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_copy_link_names_a_held_file(page):
    _, listing = _split(page)
    held = _held_files() | {"nasa_2008_cmapss_readme.txt"}
    for name in re.findall(r"\[Copy\]\(\.\./references/([^)]+)\)", listing):
        assert name in held, (
            f"{page.name}: links a copy the manifest does not list: {name}"
        )
