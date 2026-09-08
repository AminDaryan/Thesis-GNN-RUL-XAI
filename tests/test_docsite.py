"""Tests for the documentation website: it must never invent, leak or drift."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thesis_gnn_rul_xai.docsite import catalog, figures, markup, results
from thesis_gnn_rul_xai.utils.paths import project_root


@pytest.fixture(scope="module")
def root() -> Path:
    return project_root()


@pytest.fixture(scope="module")
def site_catalog(root: Path) -> catalog.Catalog:
    return catalog.build_catalog(root)


@pytest.fixture()
def stored_runs(root: Path) -> list[results.Run]:
    """Every training run in reports/, or skip the test if there are none.

    `reports/*` is deliberately not committed -- a run directory is written per
    training run and they accumulate -- so a fresh clone has none, and neither
    does the CI runner. Tests about stored runs therefore skip there for the
    same reason the data-dependent tests do: the artefact is absent, not wrong.
    """
    runs = results.load_runs(root)
    if not runs:
        pytest.skip(
            f"No training runs are stored in {root / 'reports'}. They are not "
            "committed (see .gitignore), so a fresh clone has none. Produce "
            "one with `python scripts/train.py --seeds 5`."
        )
    return runs


# -- the markup renderer ---------------------------------------------------


def test_headings_get_the_anchor_github_would_give_them():
    rendered = markup.render_markdown("## 8. Results so far\n")
    assert rendered.headings[0].anchor == "8-results-so-far"
    assert 'id="8-results-so-far"' in rendered.html


def test_two_headings_with_the_same_words_get_different_anchors():
    rendered = markup.render_markdown("## Files\ntext\n## Files\nmore\n")
    assert [h.anchor for h in rendered.headings] == ["files", "files-1"]


def test_mathematics_survives_the_markdown_parser_untouched():
    # An underscore inside a formula is a subscript, not the start of italics.
    rendered = markup.render_markdown(r"Then $\mathrm{RUL}_{\max} = 125$ holds.")
    assert r"\mathrm{RUL}_{\max} = 125" in rendered.html
    assert "<em>" not in rendered.html
    assert rendered.has_math


def test_a_dollar_sign_inside_code_is_not_treated_as_mathematics():
    rendered = markup.render_markdown("Run `echo $HOME` and `echo $PATH` now.")
    assert not rendered.has_math
    assert "math-inline" not in rendered.html


def test_display_mathematics_is_marked_up_as_display():
    rendered = markup.render_markdown("$$a = b$$")
    assert "math-display" in rendered.html


def test_links_between_documents_are_rewritten_to_site_addresses():
    rendered = markup.render_markdown(
        "See [the log](docs/ASSUMPTIONS.md#a-01) and [up](../README.md).",
        source_path="src/thesis_gnn_rul_xai/data/README.md",
    )
    assert "#/doc/src/thesis_gnn_rul_xai/data/docs/ASSUMPTIONS.md#a-01" in rendered.html
    assert "#/doc/src/thesis_gnn_rul_xai/README.md" in rendered.html


def test_an_external_link_is_left_alone():
    rendered = markup.render_markdown("[NASA](https://www.nasa.gov/)")
    assert 'href="https://www.nasa.gov/"' in rendered.html


def test_an_identifier_in_prose_becomes_a_link_to_its_entry():
    rendered = markup.render_markdown("This follows A-07, unlike F-11.")
    assert 'href="#/entry/A-07"' in rendered.html
    assert 'href="#/entry/F-11"' in rendered.html


def test_an_identifier_inside_code_is_left_alone():
    rendered = markup.render_markdown("The comment reads `ASSUMPTION A-07` there.")
    assert 'href="#/entry/A-07"' not in rendered.html


def test_every_document_in_the_repository_renders():
    for path in sorted(project_root().glob("docs/**/*.md")):
        relative = path.relative_to(project_root()).as_posix()
        rendered = markup.render_markdown(
            path.read_text(encoding="utf-8"), source_path=relative
        )
        assert rendered.html, f"{relative} rendered to nothing"


# -- the catalog -----------------------------------------------------------


def test_the_catalog_finds_the_taught_pages(site_catalog: catalog.Catalog):
    learn = next(s for s in site_catalog.sections if s.key == "learn")
    assert len(learn.documents) == 8
    assert all(doc.words > 200 for doc in learn.documents)


def test_the_catalog_finds_one_readme_for_every_source_folder(
    site_catalog: catalog.Catalog, root: Path
):
    folders = {
        path.parent.name
        for path in (root / "src" / "thesis_gnn_rul_xai").glob("*/README.md")
    }
    listed = {
        doc.path.split("/")[-2]
        for doc in site_catalog.by_path.values()
        if doc.path.startswith("src/") and doc.path.count("/") == 3
    }
    assert folders == listed


def test_the_catalog_lists_every_per_file_page_under_data(
    site_catalog: catalog.Catalog, root: Path
):
    """Each file under data/ keeps its own page beside its code, and the
    website's menu is written by hand, so a new page can be added to the folder
    and forgotten here. This compares the two."""
    on_disk = {
        path.parent.name
        for path in (root / "src" / "thesis_gnn_rul_xai" / "data").glob("*/README.md")
    }
    listed = {
        doc.path.split("/")[-2]
        for doc in site_catalog.by_path.values()
        if doc.path.startswith("src/thesis_gnn_rul_xai/data/")
        and doc.path.count("/") == 4
    }
    assert on_disk == listed


def test_reading_a_file_outside_the_project_is_refused(root: Path):
    with pytest.raises(ValueError, match="outside the project"):
        catalog.read_document("../../../etc/passwd", root)


def test_search_finds_a_phrase_that_is_certainly_there(
    site_catalog: catalog.Catalog, root: Path
):
    hits = catalog.search("piecewise", site_catalog, root)
    assert hits
    assert all({"path", "title", "line", "context"} <= set(hit) for hit in hits)


def test_search_ignores_a_query_too_short_to_mean_anything(
    site_catalog: catalog.Catalog, root: Path
):
    assert catalog.search("a", site_catalog, root) == []


# -- the assumptions log and the stored runs -------------------------------


def test_every_assumption_and_finding_is_picked_up(root: Path):
    entries = results.load_entries(root)
    identifiers = [entry.identifier for entry in entries]

    assert len(identifiers) == len(set(identifiers)), "an entry was listed twice"
    assumptions = sorted(i for i in identifiers if i.startswith("A-"))
    findings = sorted(i for i in identifiers if i.startswith("F-"))

    # The log numbers its entries consecutively from 01, so a gap means the
    # parser missed one rather than that the log skipped a number.
    assert assumptions == [f"A-{n:02d}" for n in range(1, len(assumptions) + 1)]
    assert findings == [f"F-{n:02d}" for n in range(1, len(findings) + 1)]


def test_each_entry_anchor_exists_in_the_rendered_log(root: Path):
    rendered = markup.render_markdown(
        catalog.read_document("docs/ASSUMPTIONS.md", root),
        source_path="docs/ASSUMPTIONS.md",
    )
    anchors = {heading.anchor for heading in rendered.headings}
    for entry in results.load_entries(root):
        assert entry.anchor in anchors, f"{entry.identifier} has no anchor to link to"


def test_a_withdrawn_entry_is_not_reported_as_active(root: Path):
    entries = {entry.identifier: entry for entry in results.load_entries(root)}
    assert entries["A-08"].status == "withdrawn"


def test_an_entry_superseded_in_its_body_is_picked_up(root: Path):
    # A-06 and A-18 carry no status word in the title; they open with a bold
    # "Superseded ... by A-22." line instead.
    entries = {entry.identifier: entry for entry in results.load_entries(root)}
    assert entries["A-06"].status == "superseded"
    assert entries["A-18"].status == "superseded"


def test_a_status_word_in_ordinary_prose_does_not_retire_an_entry(root: Path):
    # A-25's title says the order "reverses" the paper's Fig. 1, and an earlier
    # version of the parser read that as the entry having been reversed.
    entries = {entry.identifier: entry for entry in results.load_entries(root)}
    assert entries["A-25"].status == "active"
    assert "reverses" in entries["A-25"].title


def test_only_the_entries_that_say_so_are_retired(root: Path):
    retired = {
        entry.identifier
        for entry in results.load_entries(root)
        if entry.status != "active"
    }
    assert retired == {"A-06", "A-08", "A-18"}


def test_every_stored_run_names_the_commit_that_produced_it(stored_runs):
    for run in stored_runs:
        assert run.commit and run.commit != "unknown", run.directory
        assert run.dirty == run.commit.endswith("-dirty")


def test_a_single_seed_run_keeps_its_missing_spread_as_missing(stored_runs):
    for run in stored_runs:
        if run.seeds == 1:
            assert run.std.get("rmse") is None
            return
    pytest.skip("no single-seed run is stored")


def test_the_directory_name_of_every_run_carries_its_commit(stored_runs):
    for run in stored_runs:
        assert run.commit in run.directory, run.directory


def test_a_verification_report_is_not_listed_as_a_training_run(stored_runs):
    # scripts/verify_claims.py writes into reports/ too, with no scores in it.
    # Listing one as a run would show a row of dashes that reads like a
    # training run that failed.
    directories = {run.directory for run in stored_runs}
    assert not any(name.startswith("verification_") for name in directories)


def test_every_listed_run_has_the_four_measures(stored_runs):
    for run in stored_runs:
        assert set(run.mean) >= {"rmse", "score", "r2", "asue"}, run.directory
        assert run.mean["rmse"] is not None, run.directory


def test_the_latest_verification_is_found_and_says_what_it_checked(root: Path):
    checked = results.latest_verification(root)
    if checked is None:
        pytest.skip("scripts/verify_claims.py has not been run here")
    assert checked.checked > 0
    assert checked.directory.startswith("verification_")
    assert checked.commit in checked.directory


# -- the figures -----------------------------------------------------------


def test_every_figure_is_listed_exactly_once():
    names = figures.figure_names()
    assert len(names) == len(set(names))


def test_the_figures_that_need_nothing_all_draw():
    for figure in figures.FIGURES:
        if figure.needs_data or figure.needs_torch:
            continue
        drawn = figures.render_figure(figure.name)
        assert drawn.startswith("<svg"), figure.name
        assert "figure-notice" not in drawn, figure.name


def test_an_unknown_figure_says_so_instead_of_failing():
    drawn = figures.render_figure("no-such-figure")
    assert "figure-notice" in drawn
    assert "no such figure" in drawn.lower()


def test_a_figure_needing_the_nasa_files_says_so_when_they_are_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(figures, "raw_data_dir", lambda: tmp_path)
    drawn = figures.render_figure("rul-curve")
    assert "figure-notice--data" in drawn
    assert "nasa" in drawn.lower()
    # It must point at the download rather than draw something invented.
    assert "<svg" not in drawn


def test_the_sensor_graph_figure_reports_what_the_code_measures():
    from thesis_gnn_rul_xai.graph.adjacency import (
        adjacency_report,
        build_physics_adjacency,
    )

    report = adjacency_report(*build_physics_adjacency())
    drawn = figures.render_figure("sensor-graph")
    assert f"{report['n_nodes']} sensors" in drawn
    assert f"{report['n_edges']} edges" in drawn
    assert f"diameter {report['diameter']}" in drawn


def test_the_notebook_map_counts_what_the_pipeline_really_produces(raw_dir: Path):
    """The map of notebook 01 must agree with the pipeline it describes.

    The figure exists so a reader can see the whole chain at once. That is
    only worth having if its numbers are the pipeline's own, so they are
    compared against a real run here rather than against anything written
    down.
    """
    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing

    prepared = run_preprocessing(PreprocessingConfig(), data_dir=raw_dir)
    drawn = figures.render_figure("notebook-map")
    assert drawn.startswith("<svg")
    assert "figure-notice" not in drawn

    def spaced(value: int) -> str:
        return f"{value:,}".replace(",", " ")

    windows = len(prepared.train) + len(prepared.validation)
    # The three sets the notebook ends with, drawn as the shapes it saves.
    assert f"({spaced(len(prepared.train))}, 14, 1)" in drawn
    assert f"({spaced(len(prepared.validation))}, 14, 1)" in drawn
    assert f"({spaced(len(prepared.test))}, 14, 1)" in drawn
    # The total before the split, and the graph the same run built.
    assert spaced(windows) in drawn
    assert f"{prepared.diagnostics['graph']['n_edges']} edges" in drawn


def test_the_notebook_map_needs_the_nasa_files_rather_than_inventing_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(figures, "raw_data_dir", lambda: tmp_path)
    drawn = figures.render_figure("notebook-map")
    assert "figure-notice--data" in drawn
    assert "<svg" not in drawn


def test_the_map_survives_into_the_page_the_site_builds_from_the_notebook(
    root: Path, raw_dir: Path
):
    """The map has to reach the reader, not just sit in the notebook file.

    It went missing once: the cell displayed an SVG object, and nbconvert's
    HTML exporter drops an image/svg+xml output while keeping a text/html one,
    so the notebook held the picture and the website showed nothing where it
    should have been. Nothing failed; the figure was simply absent.
    """
    pytest.importorskip("nbconvert")
    from thesis_gnn_rul_xai.docsite import notebooks

    body, _ = notebooks.render_notebook(
        root / "notebooks" / "01_preprocessing_fd001.ipynb"
    )
    assert "What notebook 01 does" in body, (
        "the pipeline map is not in the page the site builds from notebook 01"
    )


def test_the_message_passing_figure_stops_where_the_graph_stops():
    from thesis_gnn_rul_xai.graph.adjacency import (
        build_physics_adjacency,
        pairs_beyond_reach,
    )

    matrix, symbols = build_physics_adjacency()
    beyond = pairs_beyond_reach(matrix, symbols, 3)
    assert beyond, "the graph no longer has a pair beyond three layers"
    drawn = figures.render_figure("message-passing")
    assert "never reached" in drawn


def test_the_architecture_figure_prints_the_real_parameter_count():
    pytest.importorskip("torch")
    from thesis_gnn_rul_xai.graph.adjacency import (
        build_physics_adjacency,
        normalize_adjacency,
    )
    from thesis_gnn_rul_xai.models.estgnn import ESTGNN

    matrix, _ = build_physics_adjacency()
    model = ESTGNN(normalize_adjacency(matrix, method="row"), in_features=1)
    drawn = figures.render_figure("architecture")
    assert f"{model.count_parameters():,}".replace(",", " ") in drawn


# -- the taught pages ------------------------------------------------------


def test_every_figure_a_taught_page_asks_for_exists(root: Path):
    import re

    known = set(figures.figure_names())
    for page in sorted((root / "docs" / "learn").glob("*.md")):
        for name in re.findall(r"\{\{\s*figure:([a-z0-9-]+)\s*\}\}", page.read_text(encoding="utf-8")):
            assert name in known, f"{page.name} asks for unknown figure {name!r}"


def test_the_taught_pages_do_not_use_an_identifier_the_log_does_not_have(root: Path):
    import re

    known = {entry.identifier for entry in results.load_entries(root)}
    for page in sorted((root / "docs" / "learn").glob("*.md")):
        text = page.read_text(encoding="utf-8")
        for identifier in set(re.findall(r"\b([AF]-\d{2})\b", text)):
            assert identifier in known, f"{page.name} cites unknown entry {identifier}"


# -- the server ------------------------------------------------------------


def test_the_site_reports_what_this_machine_can_do(root: Path):
    from thesis_gnn_rul_xai.docsite.server import site_state

    state = site_state(root)
    assert set(state) >= {"commit", "torch", "jupyter", "hasFD001", "figures"}
    assert isinstance(state["hasFD001"], bool)
    assert json.dumps(state)  # the browser has to be able to read it
