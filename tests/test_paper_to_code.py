"""Keeps docs/PAPER_TO_CODE.md pointing at code, tests and log entries that exist.

docs/PAPER_TO_CODE.md explains what the table is for and what these tests can
and cannot establish.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from thesis_gnn_rul_xai.utils.paths import log_entry_files

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "docs" / "PAPER_TO_CODE.md"
NOTHING = chr(0x2014)   # an em dash: the cell names nothing
RANGE = chr(0x2013)     # an en dash, which the table uses between two numbers

# Every numbered part of the paper, read from its text layer on 2026-09-15.
# Headings whose only text introduces their subsections -- 2.3, 3, 3.2 and 4 --
# have no row of their own.
EXPECTED = (
    {("Eq.", str(n)) for n in range(1, 25)}
    | {("Table", str(n)) for n in range(1, 13)}
    | {("Fig.", str(n)) for n in range(1, 11)}
    | {("Section", s) for s in (
        "1", "2", "2.1", "2.2", "2.3.1", "2.3.2", "2.4", "3.1", "3.2.1", "3.2.2",
        "3.2.3", "3.2.3.1", "3.3", "3.4", "4.1", "4.2", "4.3", "4.4", "5",
    )}
)


def read_rows(text: str) -> list[dict[str, str]]:
    """The rows of the table in the given markdown, as dictionaries keyed by column."""
    lines = [line for line in text.splitlines() if line.startswith("| ")]
    header = [cell.strip() for cell in lines[0].strip("|").split(" | ")]
    rows = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        rows.append(dict(zip(header, cells, strict=True)))
    return rows


def paper_items(cell: str) -> set[tuple[str, str]]:
    """The numbered parts of the paper a Paper cell names, with ranges written out."""
    match = re.fullmatch(r"(Section|Eqs?\.|Tables?|Fig\.) ([\d.]+)(?:" + RANGE + r"(\d+))?", cell)
    if not match:
        raise ValueError(f"cannot read the Paper cell {cell!r}")
    kind = {"Eqs.": "Eq.", "Tables": "Table"}.get(match.group(1), match.group(1))
    if match.group(3) is None:
        return {(kind, match.group(2))}
    return {(kind, str(n)) for n in range(int(match.group(2)), int(match.group(3)) + 1)}


def _defined_names(body: list[ast.stmt]) -> dict[str, ast.stmt]:
    """Every function, class and assigned name at one level of a module or class."""
    names: dict[str, ast.stmt] = {}
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names[node.target.id] = node
    return names


def code_problem(reference: str, root: Path = ROOT) -> str | None:
    """Why a `module:name` reference does not resolve in src/, or None if it does."""
    module, _, chain = reference.partition(":")
    base = root / "src" / Path(*module.split("."))
    source = base.with_suffix(".py") if base.with_suffix(".py").is_file() else base / "__init__.py"
    if not chain or not source.is_file():
        return f"{reference}: no module {module}"
    body = ast.parse(source.read_text(encoding="utf-8")).body
    for part in chain.split("."):
        found = _defined_names(body).get(part)
        if found is None:
            return f"{reference}: {module} defines no {part}"
        body = found.body if isinstance(found, ast.ClassDef) else []
    return None


def check_problem(reference: str, root: Path = ROOT) -> str | None:
    """Why a `path::function` reference names nothing, or None if the function exists."""
    path, _, name = reference.partition("::")
    file = root / path
    if not name or not file.is_file():
        return f"{reference}: no file {path}"
    functions = {
        node.name for node in ast.parse(file.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }
    return None if name in functions else f"{reference}: {path} defines no {name}"


def log_codes() -> set[str]:
    """Every assumption and finding whose file in docs/assumptions/ or docs/findings/ opens with its heading."""
    headings = (path.read_text(encoding="utf-8") for path in log_entry_files(ROOT))
    return {match.group(1) for text in headings if (match := re.match(r"# ([AF]-\d{2}) ", text))}


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    return read_rows(TABLE.read_text(encoding="utf-8"))


def test_every_numbered_part_of_the_paper_has_exactly_one_row(rows):
    named = [item for row in rows for item in paper_items(row["Paper"])]
    assert sorted(EXPECTED - set(named)) == []
    assert sorted(set(named) - EXPECTED) == []
    assert sorted({item for item in named if named.count(item) > 1}) == []


def test_every_code_reference_resolves(rows):
    problems = [
        f"{row['Paper']}: {problem}"
        for row in rows
        for reference in re.findall(r"`([\w.]+:[\w.]+)`", row["Code"])
        if (problem := code_problem(reference))
    ]
    assert problems == []


def test_every_check_reference_names_a_test_or_check_that_exists(rows):
    problems = [
        f"{row['Paper']}: {problem}"
        for row in rows
        for reference in re.findall(r"`([\w/.]+\.py::\w+)`", row["Checked by"])
        if (problem := check_problem(reference))
    ]
    assert problems == []


def test_a_row_that_names_code_names_a_check(rows):
    unchecked = [row["Paper"] for row in rows
                 if row["Code"] != NOTHING and row["Checked by"] == NOTHING]
    assert unchecked == []


def test_every_entry_a_row_cites_is_in_the_log(rows):
    known = log_codes()
    missing = sorted({
        f"{row['Paper']}: {code}"
        for row in rows
        for code in re.findall(r"\b([AF]-\d{2})\b", row["Standing"])
        if code not in known
    })
    assert missing == []


def test_a_renamed_function_or_deleted_test_is_reported_by_name():
    """The failures the table exists to catch, shown on references that name nothing."""
    assert code_problem("thesis_gnn_rul_xai.models.layers:GraphConvolution.forward") is None
    assert code_problem("thesis_gnn_rul_xai.models.layers:GraphConvolution.backwards") == (
        "thesis_gnn_rul_xai.models.layers:GraphConvolution.backwards: "
        "thesis_gnn_rul_xai.models.layers defines no backwards"
    )
    assert check_problem("tests/test_metrics.py::test_rmse_is_in_cycles") is None
    assert check_problem("tests/test_metrics.py::test_rmse_is_in_furlongs") == (
        "tests/test_metrics.py::test_rmse_is_in_furlongs: "
        "tests/test_metrics.py defines no test_rmse_is_in_furlongs"
    )
    assert paper_items(f"Tables 7{RANGE}10") == {("Table", "7"), ("Table", "8"),
                                           ("Table", "9"), ("Table", "10")}
