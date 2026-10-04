"""The assumptions and findings are one file per entry, indexed once each, and nothing points at the old single file."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from thesis_gnn_rul_xai.utils.paths import log_entry_files, project_root

ROOT = project_root()
DOCS = ROOT / "docs"
FOLDERS = {"assumptions": "A", "findings": "F"}

# The five dated records keep the paths of the day they were written; docs/audits/README.md says so.
DATED = {
    "docs/audits/AUDIT_2026-08-31_paper_against_code.md",
    "docs/audits/AUDIT_2026-09-08_response.md",
    "docs/audits/AUDIT_2026-09-08_scientific_validity.md",
    "docs/audits/REVIEW_2026-08-27.md",
    "docs/audits/REVIEW_2026-08-29.md",
}
# Each folder README names the old single file once, in the sentence saying what became of it.
NAMES_THE_OLD_FILE_ONCE = {
    "docs/assumptions/README.md",
    "docs/findings/README.md",
    "docs/audits/README.md",
}
# The site's script names every old address once, to send a bookmark to where the page went.
REDIRECTS_OLD_ADDRESSES = "src/thesis_gnn_rul_xai/docsite/assets/app.js"
# The old single log, and an audit or review at its place before the move to docs/audits/.
# An audit's or review's file name counts wherever it is written, except under its new folder.
OLD_PATH = re.compile(r"ASSUMPTIONS\.md|(?<!audits/)\b(?:AUDIT|REVIEW)_20\d\d")
# Text files with no suffix that git may hold, and must be read too.
CHECKED_NAMES = {".gitignore", ".gitattributes"}
CHECKED_SUFFIXES = {".md", ".py", ".js", ".ipynb", ".yaml", ".yml", ".txt", ".toml", ".html", ".css", ".cfg",
                    ".json"}


def _entries(folder: str) -> list[Path]:
    """Every file in one of the two folders except its README."""
    return sorted(p for p in (DOCS / folder).iterdir() if p.is_file() and p.name != "README.md")


@pytest.mark.parametrize("folder", sorted(FOLDERS))
def test_every_entry_is_named_after_its_code_and_opens_with_it(folder: str):
    letter = FOLDERS[folder]
    numbers = []
    for path in _entries(folder):
        named = re.fullmatch(rf"{letter}-(\d{{2}})\.md", path.name)
        assert named, f"docs/{folder}/{path.name} is not named {letter}-NN.md"
        first = path.read_text(encoding="utf-8").split("\n", 1)[0]
        assert first.startswith(f"# {letter}-{named.group(1)} — "), (path.name, first)
        numbers.append(int(named.group(1)))
    assert numbers, f"docs/{folder}/ holds no entries"
    # Numbered consecutively from 01, so a gap means a file was lost, not that a code was skipped.
    assert sorted(numbers) == list(range(1, max(numbers) + 1))


def test_the_shared_list_of_entries_is_exactly_the_two_folders_in_number_order():
    expected = [p for folder in ("assumptions", "findings") for p in
                sorted(_entries(folder), key=lambda p: int(p.stem[2:]))]
    assert log_entry_files(ROOT) == expected


@pytest.mark.parametrize("folder", sorted(FOLDERS))
def test_each_folder_readme_links_every_entry_once_and_nothing_missing(folder: str):
    letter = FOLDERS[folder]
    readme = (DOCS / folder / "README.md").read_text(encoding="utf-8")
    first_cells = re.findall(r"^\| \[([^\]]+)\]\(([^)\s]+)\) \|", readme, re.MULTILINE)
    assert first_cells, f"docs/{folder}/README.md has no table of entries"
    for label, target in first_cells:
        # The cell shows the code it links to, struck through for a withdrawn entry.
        assert label.strip("~") == Path(target).stem, (label, target)
        assert (DOCS / folder / target).resolve().is_file(), f"{folder}/README.md links {target}, which is missing"
    own = [target for _, target in first_cells if "/" not in target]
    assert len(own) == len(set(own)), "an entry is listed twice"
    assert sorted(own) == sorted(p.name for p in _entries(folder))
    assert all(name.startswith(f"{letter}-") for name in own)


def _repository_text_files() -> list[str]:
    """Every tracked or new, not ignored, text file, as a path from the project folder."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is needed to list the repository's files.")
    listed = subprocess.run(
        [git, "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if listed.returncode != 0:
        pytest.skip("This folder is not a git checkout, so its files cannot be listed.")
    return sorted(
        name for name in listed.stdout.splitlines()
        if (Path(name).suffix in CHECKED_SUFFIXES or Path(name).name in CHECKED_NAMES)
        and not name.startswith("reports/")
        and (ROOT / name).is_file()
    )


def test_no_file_points_at_the_old_single_log_or_the_audits_old_place():
    stale = []
    for name in _repository_text_files():
        if name in DATED:
            continue
        found = OLD_PATH.findall((ROOT / name).read_text(encoding="utf-8", errors="replace"))
        if name == "docs/audits/README.md":
            # It names the five records beside it, in their own folder, by their file names.
            found = [match for match in found if match == "ASSUMPTIONS" + ".md"]
        if name in NAMES_THE_OLD_FILE_ONCE:
            assert found == ["ASSUMPTIONS" + ".md"], (name, found)
        elif name == REDIRECTS_OLD_ADDRESSES:
            assert sorted(found) == sorted(["ASSUMPTIONS" + ".md"] * 2 + ["AUDIT" + "_2026"] * 3
                                           + ["REVIEW" + "_2026"] * 2), found
        elif found:
            stale.append((name, found))
    assert stale == []
