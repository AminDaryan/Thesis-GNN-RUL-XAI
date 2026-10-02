"""The cited papers in docs/references/, their manifest and the reading list agree."""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from thesis_gnn_rul_xai.utils.paths import project_root

ROOT = project_root()
FOLDER = ROOT / "docs" / "references"
MANIFEST = json.loads((FOLDER / "manifest.json").read_text(encoding="utf-8"))["papers"]
LISTED = {paper["file"] for paper in MANIFEST}
# The two files of the folder that are committed; every other file is a copy of a paper.
OWN_FILES = {"README.md", "manifest.json"}
# A Markdown link to a file in the folder, from anywhere: "references/x", "../references/x",
# "../../../../docs/references/x".
LINK = re.compile(r"\]\((?:[^)\s]*/)?references/([^)#\s/]+)(?:#[^)]*)?\)")


def _held() -> list[dict[str, object]]:
    """The manifest entries whose file is on this computer."""
    return [paper for paper in MANIFEST if (FOLDER / str(paper["file"])).is_file()]


def test_every_entry_says_what_it_is_and_how_to_get_it():
    for paper in MANIFEST:
        for key in ("id", "file", "citation", "version", "licence", "obtain", "sha256"):
            assert paper.get(key), (paper.get("file"), key)
        assert re.fullmatch(r"[0-9a-f]{64}", str(paper["sha256"])), paper["file"]
        # Free means the fetch script can restore this exact file from its source.
        assert bool(paper["free"]) == bool(paper["source"]), paper["file"]


def test_no_two_entries_share_a_name_or_a_file():
    assert len({paper["id"] for paper in MANIFEST}) == len(MANIFEST)
    assert len(LISTED) == len(MANIFEST)


def test_every_file_follows_the_naming_rule():
    for name in LISTED:
        assert re.fullmatch(r"[a-z]+_\d{4}_[a-z0-9_]+\.(pdf|txt)", name), name


def test_every_copy_on_this_computer_is_listed():
    present = {path.name for path in FOLDER.iterdir() if path.is_file()} - OWN_FILES
    if not present:
        pytest.skip("No copies here; python scripts/fetch_references.py downloads them.")
    assert present <= LISTED, sorted(present - LISTED)


def test_every_copy_present_is_the_version_the_citations_were_checked_against():
    held = _held()
    if not held:
        pytest.skip("No copies here; python scripts/fetch_references.py downloads them.")
    for paper in held:
        data = (FOLDER / str(paper["file"])).read_bytes()
        assert hashlib.sha256(data).hexdigest() == paper["sha256"], paper["file"]
        assert len(data) == paper["bytes"], paper["file"]
        if str(paper["file"]).endswith(".pdf"):
            assert data.startswith(b"%PDF"), paper["file"]


def test_the_reading_list_links_every_copy_and_nothing_else():
    linked = set(LINK.findall((ROOT / "docs" / "REFERENCES.md").read_text(encoding="utf-8")))
    assert linked == LISTED, (sorted(LISTED - linked), sorted(linked - LISTED))


def test_every_link_from_a_page_to_a_paper_names_a_listed_copy():
    pages = sorted((ROOT / "docs").rglob("*.md"))
    pages += sorted((ROOT / "src").rglob("README.md"))
    for page in pages:
        for name in LINK.findall(page.read_text(encoding="utf-8")):
            assert name in LISTED, (page.relative_to(ROOT).as_posix(), name)


def test_the_fetch_script_reports_every_listed_copy_present(capsys):
    if len(_held()) < len(MANIFEST):
        pytest.skip("Some copies are missing; the script would try to download them.")
    import importlib.util

    spec = importlib.util.spec_from_file_location("fetch_references", ROOT / "scripts" / "fetch_references.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == len(MANIFEST)
    assert all(line.startswith("present") for line in out)
